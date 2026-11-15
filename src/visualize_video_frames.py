"""
dataset: https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg
YOLO26 SAHI: https://docs.ultralytics.com/fr/guides/sahi-tiled-inference
Presentation notebook: https://github.com/roboflow/sports/blob/main/examples/soccer/notebooks/train_ball_detector.ipynb?ref=blog.roboflow.com
dataset: https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/browse?queryText=like-image%3A4lTzCS6HkIveoNXIjtBy&pageSize=100&startingIndex=0&browseQuery=true
https://universe.roboflow.com/shootaai/soccer_detection
https://huggingface.co/datasets/martinjolif/football-ball-detection/viewer/default/train?p=2
https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc


https://github.com/JohnComonitski/FootballTrackingDataGeneration?source=post_page-----1b44924968e9-----------------------------------------
"""
from detections.ball import load_ball_detections
from detections.tracker import track_ball, track_ball_native_bytetrack

"""Overlay the ATD tracking (players + ball) on frames of the tactical video.

Tracking positions are normalised pitch coordinates (0-1); the per-frame homography maps them to
image pixels. Alignment: video_frame = frame_id - 1.

    python starter/visualize_video_frames.py                         # a few frames after kickoff
    python starter/visualize_video_frames.py --frames 11190 11500    # specific video frames
    python starter/visualize_video_frames.py --show                  # also open a window per frame
    python starter/visualize_video_frames.py --clip 11185 11435 --play   # annotated video clip
    python starter/visualize_video_frames.py --clip 11185 11435 --opacity 0.5 --panel-width 0.35
    python starter/visualize_video_frames.py --clip 11185 11435 --ball-birdseye   # also plot the
                                                                          # tracked ball in the
                                                                          # bird's-eye panel

Images (and clips, as MP4) are written to starter/frames_out/ by default.
"""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from kloppy import metrica  # noqa: E402

import games  # noqa: E402
import load_homography  # noqa: E402
import pitch_zones  # noqa: E402
from ball_possession.possession_model import PossessionModel, is_valid  # noqa: E402

# BGR
HOME_COLOUR, AWAY_COLOUR, BALL_COLOUR = (40, 39, 214), (180, 119, 31), (0, 255, 255)

# The SORT-style ball track (from detections alone, no ATD ground truth) is drawn in magenta.
TRACK_COLOUR = (255, 0, 255)
# A merged ball-tracking CSV's (x, y) point (see load_tracked_ball_points()) is drawn in green,
# to stay visually distinct from both the ATD yellow point and the live-tracked magenta one.
TRACKED_POINT_COLOUR = (0, 255, 0)
# Metres, same as utils.plot_frame.
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
# START = 11190

START = 11190
DEFAULT_FRAMES = [i + START for i in range(100)]
DEFAULT_DETECTIONS = os.path.join(HERE, "../assets/detections/ball_detection_first_half_yolo26_tactical_sliced.csv")
DEFAULT_TRACKED_BALL_CSV = os.path.join(HERE, "../assets/trackings/ball/tracked_ball.csv")




def draw_ball_track(image, point):
    """Draw the SORT-style tracked ball `point` (from track_ball) on `image`."""
    if point is None:
        return
    centre = (int(round(point["centre"][0])), int(round(point["centre"][1])))
    if point["bbox"] is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in point["bbox"])
        cv2.rectangle(image, (x1, y1), (x2, y2), TRACK_COLOUR, 2)
        label = "%.2f" % point["score"]
    else:
        label = "coasted"
    cv2.circle(image, centre, 4, TRACK_COLOUR, -1, cv2.LINE_AA)
    cv2.putText(image, label, (centre[0] + 6, centre[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TRACK_COLOUR, 1, cv2.LINE_AA)


def load_tracked_ball_points(path):
    """{video_frame: {"point": (x, y), "source": str or None}} from a merged ball-tracking CSV.

    Built for the output of scripts/ball_tracking/merge_sort_atd_ball.py (frame,x,y,source,...),
    but only frame/x/y are required -- any CSV with those three columns works. Frames whose x or
    y is blank (nothing was tracked for them and the CSV was not forward-filled) are skipped.
    """
    points = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in ("frame", "x", "y") if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("{}: missing columns {}".format(path, missing))
        for row in reader:
            if row["x"] in (None, "") or row["y"] in (None, ""):
                continue
            points[int(float(row["frame"]))] = {
                "point": (float(row["x"]), float(row["y"])),
                "source": row.get("source") or None,
            }
    return points


def draw_tracked_ball_point(image, info):
    """Draw a merged ball-tracking CSV's (x, y) point -- see load_tracked_ball_points().

    Deliberately plain (a dot plus its source label): the point already is the tracker's final
    answer for this frame (SORT, ATD or forward-filled), not a detection to second-guess, so
    there is no bbox/score to draw alongside it the way draw_ball_track() does.
    """
    if info is None:
        return
    centre = (int(round(info["point"][0])), int(round(info["point"][1])))
    cv2.circle(image, centre, 5, TRACKED_POINT_COLOUR, -1, cv2.LINE_AA)
    cv2.circle(image, centre, 6, (0, 0, 0), 1, cv2.LINE_AA)
    if info["source"]:
        cv2.putText(image, info["source"], (centre[0] + 8, centre[1] + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, TRACKED_POINT_COLOUR, 1, cv2.LINE_AA)


def load_player_tracks_csv(path):
    """{video_frame: (n, 3) array of (bottom_mid_x, bottom_mid_y, track_id)} from a tracked-players CSV.

    Built for the output of scripts/players_detection/track_with_bytetrack.py
    (frame,track_id,x1,y1,x2,y2,score), but any CSV with those frame/track_id/box columns works.

    The point used is each box's bottom-middle ((x1+x2)/2, y2), i.e. the player's feet, not the
    box centre -- this is where an ATD player's projected pitch position actually sits (same
    convention as scripts/players_detection/visualize_detections.py's
    assign_team_ids_from_atd()), so matching against it (see match_atd_players_one_to_one()) is
    not thrown off by a vertical offset the way matching against the box centre would be.
    """
    import pandas as pd
    df = pd.read_csv(path)
    required = ["frame", "track_id", "x1", "y1", "x2", "y2"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError("{}: missing columns {}".format(path, missing))

    cx = (df["x1"].values + df["x2"].values) / 2.0
    cy = df["y2"].values
    track_id = df["track_id"].values

    boxes = {}
    for video_frame, group in df.groupby("frame"):
        idx = group.index.values
        boxes[int(video_frame)] = np.stack([cx[idx], cy[idx], track_id[idx]], axis=1)

    # print(boxes)
    return boxes


def match_atd_players_to_boxes(tracking_frame, matrix, boxes, threshold):
    """{player: track_id} for ATD players close enough (<= threshold px) to a tracked box's feet.

    `boxes` is this video frame's (n, 3) array of (bottom_mid_x, bottom_mid_y, track_id), see
    load_player_tracks_csv(). Players with no valid ATD position, or with no box within
    `threshold` px of their projected position, are simply left out of the returned mapping.
    """
    matches = {}
    if boxes is None or len(boxes) == 0:
        return matches
    for player, position in tracking_frame.players_coordinates.items():
        if not is_valid(position):
            continue
        px, py = load_homography.project(matrix, position.x, position.y)
        dist = np.sqrt((boxes[:, 0] - float(px)) ** 2 + (boxes[:, 1] - float(py)) ** 2)
        nearest = int(dist.argmin())
        if dist[nearest] <= threshold:
            matches[player] = int(boxes[nearest, 2])
    return matches


def match_atd_players_one_to_one(tracking_frame, matrix, boxes, threshold, video_frame,
                                 home_team, away_team, pitch_dimensions):
    """One-to-one {player: track_id} ATD<->box matches, with the goalkeeper's team corrected.

    Unlike match_atd_players_to_boxes() (every ATD player independently claims its single
    nearest box, so two players can legally claim the same one), this solves the full assignment
    with the Hungarian algorithm (scipy.optimize.linear_sum_assignment) over the (ATD players) x
    (tracked boxes) distance matrix, so each box is matched to at most one ATD player and vice
    versa. Pairs farther apart than `threshold` px are dropped, same as any ATD player or box the
    assignment leaves unmatched -- an ATD point with no one-to-one box is simply left out of the
    returned matches, exactly like the "player not in atd_matches" skip already did in annotate().

    `video_frame`, `home_team`/`away_team` (dataset.metadata.teams) and `pitch_dimensions`
    (dataset.metadata.pitch_dimensions) are then used to fix up the goalkeeper: among the matched
    players, whichever one is alone inside a penalty box (pitch_zones.penalty_box_bounds()) --
    or, if several share a box, whichever is closest to that box's goal line -- is assigned
    whichever team defends that side this half (games.half_of_video_frame() /
    pitch_zones.HOME_SIDE_BY_HALF), not whatever team its own ATD position says. A keeper's ATD
    team is not always reliable, which is the whole reason for this override. Frames outside both
    halves get no override.

    Returns (matches, team_override): matches is {player: track_id} for the surviving one-to-one
    pairs; team_override is {player: team} for any matched player identified as a goalkeeper this
    frame -- look up a player's effective team with team_override.get(player, player.team).
    """
    matches = {}
    team_override = {}

    if boxes is None or len(boxes) == 0:
        return matches, team_override

    players, positions = [], []
    for player, position in tracking_frame.players_coordinates.items():
        if is_valid(position):
            players.append(player)
            positions.append(position)

    if not players:
        return matches, team_override

    px, py = load_homography.project(
        matrix,
        np.array([position.x for position in positions]),
        np.array([position.y for position in positions]),
    )
    atd_pixels = np.stack([px, py], axis=1)

    # (n_players, n_boxes) distance matrix.
    dist = np.sqrt(
        (atd_pixels[:, 0, None] - boxes[None, :, 0]) ** 2
        + (atd_pixels[:, 1, None] - boxes[None, :, 1]) ** 2
    )

    player_rows, box_cols = linear_sum_assignment(dist)
    matched_players = []
    for player_i, box_i in zip(player_rows, box_cols):
        if dist[player_i, box_i] <= threshold:
            player = players[player_i]
            matches[player] = int(boxes[box_i, 2])
            matched_players.append(player)

    if not matched_players:
        return matches, team_override

    half = games.half_of_video_frame(games.GAME, video_frame)
    home_side = pitch_zones.HOME_SIDE_BY_HALF.get(half)
    if home_side is None:
        return matches, team_override

    away_side = "right" if home_side == "left" else "left"
    side_to_team = {home_side: home_team, away_side: away_team}
    matched_positions = {player: tracking_frame.players_coordinates[player]
                         for player in matched_players}

    for side, team in side_to_team.items():
        x_min, x_max, y_min, y_max = pitch_zones.penalty_box_bounds(side, pitch_dimensions)
        goal_x = 0.0 if side == "left" else 1.0
        inside = [player for player in matched_players
                 if x_min <= matched_positions[player].x <= x_max
                 and y_min <= matched_positions[player].y <= y_max]
        if not inside:
            continue
        keeper = min(inside, key=lambda player: abs(matched_positions[player].x - goal_x))
        team_override[keeper] = team

    return matches, team_override


def detect_goalkeepers_from_boxes(player_boxes, matrices, valid, team_sides, pitch_dimensions,
                                  video_frames=None, min_votes=5):
    """{track_id: team} for tracked boxes that are repeatedly the lone box in a penalty area.

    On every frame (restricted to `video_frames`, if given) with a valid homography, each tracked
    box's bottom-middle point (see load_player_tracks_csv()) is unprojected back to normalised
    pitch coordinates
    and tested against both penalty boxes (pitch_zones.penalty_box_bounds()). Whenever exactly
    one tracked box is found inside a given side's penalty area, that is taken to be the
    goalkeeper standing alone in it, and its track_id casts one vote for the team that
    `team_sides` says defends that side. The final team assigned to a track_id is whichever side
    it voted for most often, so a defender or attacker passing through an empty box on a single
    frame does not relabel the goalkeeper.

    `team_sides` is either a fixed {team: "left"/"right"} dict (e.g. from
    pitch_zones.determine_team_sides()) applied to every frame, or a callable video_frame ->
    {team: "left"/"right"} (or None) for a per-frame mapping -- see
    pitch_zones.team_sides_for_video_frame(), which is ground truth rather than a guess and also
    correctly flips the sides at halftime. A frame for which the callable returns None (outside
    both halves) is skipped.

    `min_votes` drops any track_id whose winning side got fewer than that many votes -- with a
    track_id a frame, a short-lived/noisy track (e.g. a tracking-ID switch) can easily pick up a
    stray vote or two just by passing through an empty box, so this filters those out.
    """
    votes = defaultdict(Counter)
    frames = video_frames if video_frames is not None else player_boxes.keys()
    team_sides_fn = team_sides if callable(team_sides) else (lambda _vf: team_sides)

    for video_frame in frames:
        boxes = player_boxes.get(video_frame)
        if boxes is None or len(boxes) == 0:
            continue
        if video_frame >= len(valid) or not valid[video_frame]:
            continue

        frame_sides = team_sides_fn(video_frame)
        if not frame_sides:
            continue
        side_to_team = {side: team for team, side in frame_sides.items()}

        norm_x, norm_y = load_homography.unproject(matrices[video_frame], boxes[:, 0], boxes[:, 1])

        for side, team in side_to_team.items():
            x_min, x_max, y_min, y_max = pitch_zones.penalty_box_bounds(side, pitch_dimensions)
            inside = (norm_x >= x_min) & (norm_x <= x_max) & (norm_y >= y_min) & (norm_y <= y_max)
            print(video_frame, inside)
            if inside.sum() == 1:
                track_id = int(boxes[inside.argmax(), 2])
                votes[track_id][team] += 1

    return {
        track_id: team_votes.most_common(1)[0][0]
        for track_id, team_votes in votes.items()
        if team_votes.most_common(1)[0][1] >= min_votes
    }


class BirdsEye:
    """Top-down pitch panel drawn like utils.plot_frame (metres, origin bottom-left, y up)."""

    MARGIN = 3.0
    GRASS, LINE, BALL = (79, 143, 63), (255, 255, 255), (255, 255, 255)

    def __init__(self, panel_width, length=PITCH_LENGTH, width=PITCH_WIDTH):
        self.length, self.width = length, width
        self.px_per_m = panel_width / (length + 2 * self.MARGIN)
        self.size = (panel_width, int(round((width + 2 * self.MARGIN) * self.px_per_m)))
        self.background = self._draw_pitch()

    def to_px(self, x_m, y_m):
        return (int(round((x_m + self.MARGIN) * self.px_per_m)),
                int(round((self.width - y_m + self.MARGIN) * self.px_per_m)))

    def _draw_pitch(self):
        panel_w, panel_h = self.size
        image = np.full((panel_h, panel_w, 3), self.GRASS, dtype=np.uint8)
        L, W = self.length, self.width
        cv2.rectangle(image, self.to_px(0, W), self.to_px(L, 0), self.GRASS, -1)
        cv2.rectangle(image, self.to_px(0, W), self.to_px(L, 0), self.LINE, 2)
        cv2.line(image, self.to_px(L / 2, 0), self.to_px(L / 2, W), self.LINE, 2)
        cv2.circle(image, self.to_px(L / 2, W / 2), int(round(9.15 * self.px_per_m)),
                   self.LINE, 2, cv2.LINE_AA)
        for side in (0, 1):
            # Penalty area 16.5 m deep and 40.3 m wide, six-yard box 5.5 m by 18.3 m.
            for depth, box_width in ((16.5, 40.3), (5.5, 18.3)):
                x = 0 if side == 0 else L - depth
                y = (W - box_width) / 2
                cv2.rectangle(image, self.to_px(x, y + box_width), self.to_px(x + depth, y),
                              self.LINE, 2)
        return image

    @staticmethod
    def _flip_y(y):
        """ATD y is 0-1 from one touchline; this panel's y-up pitch needs it mirrored."""
        return 1 - y if y > 0.5 else 0.5 + (0.5 - y)

    def render(self, tracking_frame, home_team, track_point_px=None, matrix=None,
              atd_matches=None, team_override=None):
        """Draw players (ATD) and the ball panel marker.

        Normally the ball marker is the ATD ground truth, same as everything else. Pass
        `track_point_px` (a video-pixel (x, y), e.g. from track_ball()'s "centre") together with
        the frame's homography `matrix` to draw the *tracked* (detection-based) ball here instead
        -- unprojected from pixels back to pitch coordinates -- so it can be compared against the
        players without also cluttering the raw video frame with a second ball marker.

        `atd_matches`/`team_override`, if given, are the same (atd_matches, team_override) pair
        returned by annotate() -- see match_atd_players_one_to_one() -- so this panel only draws
        the ATD players that actually have a one-to-one match to a tracked (detection) box on
        this frame, exactly like the main video overlay, instead of every ATD player regardless
        of detections. A matched player overridden as the goalkeeper gets its corrected team
        colour and a "GK" tag, same as on the raw frame.
        """
        image = self.background.copy()
        if tracking_frame is None:
            return image
        team_override = team_override or {}
        radius = max(4, int(round(0.9 * self.px_per_m)))
        for player, position in tracking_frame.players_coordinates.items():
            if not is_valid(position):
                continue
            if atd_matches is not None and player not in atd_matches:
                continue

            y = self._flip_y(position.y)
            centre = self.to_px(position.x * self.length, y * self.width)
            colour = HOME_COLOUR if team_override.get(player, player.team) == home_team else AWAY_COLOUR
            cv2.circle(image, centre, radius, colour, -1, cv2.LINE_AA)
            if player in team_override:
                cv2.putText(image, "GK", (centre[0] - 10, centre[1] - radius - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
            cv2.circle(image, centre, radius, (0, 0, 0), 1, cv2.LINE_AA)

        ball_radius = max(3, radius - 2)
        if track_point_px is not None and matrix is not None:
            norm_x, norm_y = load_homography.unproject(matrix, track_point_px[0],
                                                        track_point_px[1])
            if np.isnan(norm_x) or np.isnan(norm_y):
                cv2.putText(image, "ball track off-pitch", (10, image.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            else:
                y = self._flip_y(float(norm_y))
                centre = self.to_px(float(norm_x) * self.length, y * self.width)
                cv2.circle(image, centre, ball_radius, TRACK_COLOUR, -1, cv2.LINE_AA)
                cv2.circle(image, centre, ball_radius, (0, 0, 0), 2, cv2.LINE_AA)
            return image

        ball = tracking_frame.ball_coordinates
        if is_valid(ball):
            y = self._flip_y(ball.y)
            centre = self.to_px(ball.x * self.length, y * self.width)
            cv2.circle(image, centre, ball_radius, self.BALL, -1, cv2.LINE_AA)
            cv2.circle(image, centre, ball_radius, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, "ball not tracked", (10, image.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        return image


def read_video_frame(capture, video_frame):
    capture.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
    ok, image = capture.read()
    if not ok:
        raise IOError("could not read video frame {}".format(video_frame))
    return image


def build_ball_references(frames, by_frame_id, matrices, valid):
    """Project the ATD ground-truth (yellow) ball position into pixels for each video frame.

    Returns {video_frame: (px, py)}, only for frames with a valid homography and a tracked ball.
    """
    references = {}
    for video_frame in frames:
        if not valid[video_frame]:
            continue
        tracking_frame = by_frame_id.get(video_frame + 1)
        if tracking_frame is None:
            continue
        ball = tracking_frame.ball_coordinates
        if not is_valid(ball):
            continue
        px, py = load_homography.project(matrices[video_frame], ball.x, ball.y)
        references[video_frame] = (float(px), float(py))
    return references


def draw_pitch_outline(image, matrix):
    corners = np.array(load_homography.pitch_corners(matrix), dtype=np.int32)
    cv2.polylines(image, [corners], isClosed=True, color=(255, 255, 255), thickness=1)
    halfway = [load_homography.project(matrix, 0.5, y) for y in (0.0, 1.0)]
    cv2.line(image, tuple(int(v) for v in halfway[0]), tuple(int(v) for v in halfway[1]),
             (255, 255, 255), 1)



def annotate(image, tracking_frame, matrix, home_team, video_frame, ball_track=None,
            highlight_player=None, tracked_ball_points=None, player_boxes=None,
            player_match_threshold=25.0, away_team=None, pitch_dimensions=None):
    """Annotate `image` with the ATD players/ball for `video_frame`.

    player_boxes, if given, is {video_frame: (n, 3) array of (bottom_mid_x, bottom_mid_y, track_id)} --
    see load_player_tracks_csv(). When set, an ATD player is only drawn if it has a one-to-one
    match to one of this frame's tracked boxes (match_atd_players_one_to_one(), if `away_team`
    and `pitch_dimensions` are both given, else the looser nearest-box match_atd_players_to_boxes()
    as a fallback); every other ATD player, including all of them when this frame has no tracked
    boxes at all, is skipped.

    With match_atd_players_one_to_one(), a matched player identified as the goalkeeper alone in a
    penalty box is drawn with its team-corrected colour (see that function) and an extra "GK" tag
    on its label, rather than whatever team its own ATD position says.

    Returns (image, atd_matches, team_override) -- the latter two, as computed above (None/{} when
    `player_boxes` is not given), can be handed to BirdsEye.render() so the bird's-eye panel only
    plots the same detection-matched players as this overlay, instead of every ATD player.
    """

    atd_matches = None
    team_override = {}
    if player_boxes is not None:
        if away_team is not None and pitch_dimensions is not None:
            atd_matches, team_override = match_atd_players_one_to_one(
                tracking_frame, matrix, player_boxes.get(video_frame), player_match_threshold,
                video_frame, home_team, away_team, pitch_dimensions)
        else:
            atd_matches = match_atd_players_to_boxes(
                tracking_frame, matrix, player_boxes.get(video_frame), player_match_threshold)

    for player, position in tracking_frame.players_coordinates.items():
        if not is_valid(position):
            continue
        if atd_matches is not None and player not in atd_matches:
            continue
        px, py = load_homography.project(matrix, position.x, position.y)
        centre = (int(round(float(px))), int(round(float(py))))
        colour = HOME_COLOUR if team_override.get(player, player.team) == home_team else AWAY_COLOUR
        if player == highlight_player:
            cv2.circle(image, centre, 15, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(image, centre, 9, colour, 2)
        label = str(player.jersey_no)
        if atd_matches is not None:
            label = "{} (#{})".format(label, atd_matches[player])
        if player in team_override:
            label = "{} GK".format(label)
        cv2.putText(image, label, (centre[0] - 8, centre[1] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA)

    ball = tracking_frame.ball_coordinates
    has_ball = is_valid(ball)
    if has_ball:
        px, py = load_homography.project(matrix, ball.x, ball.y)
        ball_centre = (int(round(float(px))), int(round(float(py))))
        cv2.circle(image, ball_centre, 6, BALL_COLOUR, -1)
        cv2.circle(image, ball_centre, 7, (0, 0, 0), 1)

    if ball_track is not None:
        draw_ball_track(image, ball_track.get(video_frame))

    if tracked_ball_points is not None:
        draw_tracked_ball_point(image, tracked_ball_points.get(video_frame))

    label = "video frame {} | ATD frame_id {}{}".format(
        video_frame, tracking_frame.frame_id, "" if has_ball else " | ball not tracked")
    x_origin_panel = 1300
    cv2.rectangle(image, (x_origin_panel, 0), (len(label) * 11 + 20, 34), (0, 0, 0), -1)
    cv2.putText(image, label, (10 + x_origin_panel // 2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                cv2.LINE_AA)
    return image, atd_matches, team_override


def draw_touch_alert(image):
    text = "Touche probable"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness, margin = 0.8, 2, 12
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, image.shape[1] - text_width - 2 * margin)
    y = image.shape[0]
    cv2.rectangle(image, (x, y - text_height - baseline - 2 * margin),
                  (image.shape[1], y), (0, 0, 0), -1)
    cv2.putText(image, text, (x + margin, y - margin - baseline), font, scale,
                (0, 255, 255), thickness, cv2.LINE_AA)


GT_PATH = os.path.join(os.path.dirname(HERE), "mtah_challenge", "evaluation", "gt",
                       "BAR-ATH_possession_gt_public.csv")
# label -> (text, BGR colour)
GT_LABELS = {"H": ("Barcelone", HOME_COLOUR), "A": ("Athletic", AWAY_COLOUR),
             "D": ("Dispute", (0, 165, 255)), "X": ("Ballon mort", (128, 128, 128))}


def load_gt(path):
    """Return {video_frame: label} from the possession ground truth CSV."""
    import csv
    try:
        with open(path, newline="") as f:
            return {int(row["frame"]): row["label"] for row in csv.DictReader(f)}

    except:
        print('Can not import gt file')
        return None


def draw_possession(image, label, changed, title="GT", y0=10):
    """Top-right possession box; thick yellow frame and 'CHANGEMENT' while `changed`."""
    text, colour = GT_LABELS.get(label, (label, (128, 128, 128)))
    text = "{}: {}".format(title, text)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness, margin = 0.9, 2, 12
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x1 = image.shape[1] - 10
    x0, y1 = x1 - tw - 2 * margin, y0 + th + baseline + 2 * margin
    cv2.rectangle(image, (x0, y0), (x1, y1), colour, -1)
    cv2.putText(image, text, (x0 + margin, y1 - margin - baseline), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    if changed:
        cv2.rectangle(image, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (0, 255, 255), 4)
        cv2.putText(image, "CHANGEMENT", (x0 - 4, y1 + 28), font, 0.7, (0, 255, 255), 2,
                    cv2.LINE_AA)


def overlay_bottom(image, panel, opacity):
    """Alpha-blend `panel` onto the bottom-centre of `image`, in place."""
    panel_h, panel_w = panel.shape[:2]
    image_h, image_w = image.shape[:2]
    x0, y0 = (image_w - panel_w) // 2, image_h - panel_h
    region = image[y0:y0 + panel_h, x0:x0 + panel_w]
    cv2.addWeighted(panel, opacity, region, 1.0 - opacity, 0, dst=region)
    return image


def render_clip(capture, start, end, by_frame_id, matrices, valid, home_team, out_path, play,
                scale, opacity=0.6, panel_fraction=0.4, gt=None, ball_track=None,
                ball_in_birdseye=False, tracked_ball_points=None, player_boxes=None,
                player_match_threshold=25.0, away_team=None, pitch_dimensions=None):


    """Annotate video frames start..end (inclusive), write them to an MP4 and optionally play.

    The bird's-eye pitch is blended onto the bottom of the video at `opacity` (0-1), at
    `panel_fraction` of the video width.

    ball_in_birdseye=True additionally draws the *tracked* (detection-based) ball in the bird's-
    eye panel, unprojected from pixels to pitch coordinates via the frame's homography -- the
    ATD ground-truth yellow point and the tracked magenta point on the raw video frame are kept.

    tracked_ball_points, if given, overlays a merged ball-tracking CSV's (x, y) point (see
    load_tracked_ball_points()) as a green dot, independent of ball_track/ball_in_birdseye.
    """
    fps = capture.get(cv2.CAP_PROP_FPS) or games.GAMES[games.GAME]["fps"]
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    birds_eye = BirdsEye(int(width * panel_fraction))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    delay = max(1, int(1000 / fps))
    window = "ATD overlay (space: pause, q: quit)"
    model = PossessionModel(home_team)
    touch_alert_until = -1
    highlighted_player = None
    gt_label = gt.get(start - 1) if gt else None
    change_until = -1
    pred_label, pred_change_until = "X", -1

    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    try:
        for video_frame in range(start, end + 1):
            ok, image = capture.read()
            if not ok:
                print("video ended at frame %d" % video_frame)
                break
            tracking_frame = by_frame_id.get(video_frame + 1)
            track_point = ball_track.get(video_frame) if ball_track is not None else None

            output = model.update(video_frame, tracking_frame)
            if output.touch_player is not None:
                highlighted_player = output.touch_player
                print("Frame {}: Touche probable".format(video_frame), flush=True)
                touch_alert_until = video_frame + fps
            if output.label != pred_label:
                if pred_label != "X":
                    pred_change_until = video_frame + fps
                pred_label = output.label

            if tracking_frame is not None and valid[video_frame]:
                _, atd_matches, team_override = annotate(
                    image, tracking_frame, matrices[video_frame], home_team, video_frame,
                    ball_track=ball_track,
                    highlight_player=highlighted_player if video_frame < touch_alert_until
                    else None, tracked_ball_points=tracked_ball_points,
                    player_boxes=player_boxes, player_match_threshold=player_match_threshold,
                    away_team=away_team, pitch_dimensions=pitch_dimensions)
            else:
                atd_matches, team_override = None, {}
                # if ball_track is not None:
                #     draw_ball_track(image, track_point)
                if tracked_ball_points is not None:
                    draw_tracked_ball_point(image, tracked_ball_points.get(video_frame))

            if scale != 1.0:
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

            if ball_in_birdseye and track_point is not None and valid[video_frame]:
                panel = birds_eye.render(tracking_frame, home_team,
                                         track_point_px=track_point["centre"],
                                         matrix=matrices[video_frame],
                                         atd_matches=atd_matches, team_override=team_override)
            else:
                panel = birds_eye.render(tracking_frame, home_team,
                                         atd_matches=atd_matches, team_override=team_override)
            overlay_bottom(image, panel, opacity)

            if video_frame < touch_alert_until:
                draw_touch_alert(image)
            draw_possession(image, pred_label, video_frame < pred_change_until,
                            "Prediction", y0=110)
            if gt:
                new_label = gt.get(video_frame, gt_label)
                if new_label != gt_label and gt_label is not None:
                    change_until = video_frame + fps
                    print("Frame {}: {} -> {}".format(video_frame, gt_label, new_label),
                          flush=True)
                gt_label = new_label
                if gt_label is not None:
                    draw_possession(image, gt_label, video_frame < change_until)

            writer.write(image)

            if play:
                cv2.imshow(window, image)
                key = cv2.waitKey(delay) & 0xFF
                if key == ord(" "):
                    key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    break
    finally:
        writer.release()
    print("wrote %s" % out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, nargs="+", default=DEFAULT_FRAMES,
                        help="video frames (0-based) to render")
    parser.add_argument("--limit", type=int, default=12000,
                        help="ATD frames to load (raised automatically to cover --frames)")
    parser.add_argument("--out", default=os.path.join(HERE, "../more-than-a-hack-2027/starter/frames_out"))
    parser.add_argument("--show", action="store_true", help="display each frame in a window")
    parser.add_argument("--clip", type=int, nargs=2, metavar=("START", "END"),
                        help="render video frames START..END as an annotated MP4 instead")
    parser.add_argument("--play", action="store_true", help="with --clip: play it while rendering")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="with --clip: output size relative to the source (default 0.5)")
    parser.add_argument("--opacity", type=float, default=0.6,
                        help="with --clip: bird's-eye panel opacity, 0-1 (default 0.6)")
    parser.add_argument("--panel-width", type=float, default=0.25,
                        help="with --clip: bird's-eye panel width as a fraction of the video "
                             "width (default 0.25)")
    parser.add_argument("--ball-birdseye", action="store_true",
                        help="also draw the tracked (detection-based) ball in the bird's-eye "
                             "panel, unprojected via the homography, alongside the existing "
                             "yellow (ATD) and magenta (tracked) points on the raw video frame")
    parser.add_argument("--detections", default=DEFAULT_DETECTIONS,
                        help="CSV of ball detections (frame,x1,y1,x2,y2,score,class_id) used "
                             "for SORT-style ball tracking, default assets/ball_detection.csv")
    parser.add_argument("--no-detections", action="store_true",
                        help="don't track/overlay the ball from detections")
    parser.add_argument("--min-detection-score", type=float, default=0.3,
                        help="ignore detections scoring below this threshold (default 0.3)")
    parser.add_argument("--gating-distance", type=float, default=80.0,
                        help="max pixel distance between a predicted and detected ball "
                             "position to associate them (default 80)")
    parser.add_argument("--max-age", type=int, default=10,
                        help="frames to coast the ball track without a matching detection "
                             "before dropping it (default 10)")
    parser.add_argument("--tracker", choices=("sort", "bytetrack"), default="sort",
                        help="ball-tracking algorithm: SORT-style (default) or the real "
                             "BYTETracker (vendored from the ByteTrack submodule)")
    parser.add_argument("--tracked-ball-csv", default=DEFAULT_TRACKED_BALL_CSV,
                        help="merged ball-tracking CSV (frame,x,y[,source,...], e.g. the output "
                             "of scripts/ball_tracking/merge_sort_atd_ball.py) to overlay as a "
                             "green point, independent of --tracker/--detections (default "
                             "assets/detections/tracked_ball.csv, skipped if not found)")
    parser.add_argument("--no-tracked-ball-csv", action="store_true",
                        help="don't overlay --tracked-ball-csv even if it exists")
    parser.add_argument("--player-tracks-csv", default=None,
                        help="tracked-players CSV (frame,track_id,x1,y1,x2,y2[,score], e.g. the "
                             "output of scripts/players_detection/track_with_bytetrack.py); when "
                             "given, an ATD player is only drawn if its projected position is "
                             "within --player-match-threshold px of a tracked box on that frame "
                             "-- unmatched players are left out of the visualization entirely")
    parser.add_argument("--player-match-threshold", type=float, default=60.0,
                        help="max pixel distance between an ATD player position and its nearest "
                             "tracked box's feet point, used with --player-tracks-csv (default 25)")
    parser.add_argument("--detect-goalkeepers", action="store_true",
                        help="with --player-tracks-csv: detect goalkeepers as the tracked box "
                             "that is repeatedly the only one inside a penalty area, and print "
                             "the team (left/right side's defender) assigned to each such "
                             "track_id -- see detect_goalkeepers_from_boxes()")
    parser.add_argument("--goalkeeper-min-votes", type=int, default=5,
                        help="min frames a track_id must be alone in a penalty area to be "
                             "reported as a goalkeeper, used with --detect-goalkeepers "
                             "(default 5)")
    parser.add_argument("--video-type", type=str, default="tactical")

    parser.add_argument("--gt", default=GT_PATH,
                        help="with --clip: possession GT CSV (frame,label); '' to disable")

    args = parser.parse_args()
    if not 0.0 <= args.opacity <= 1.0:
        parser.error("--opacity must be between 0 and 1")
    if not 0.0 < args.panel_width <= 1.0:
        parser.error("--panel-width must be in (0, 1]")
    if args.ball_birdseye and args.no_detections:
        parser.error("--ball-birdseye needs ball detections; drop --no-detections")

    paths = games.asset_paths(source=args.video_type)
    games.require(["video", "atd_meta", "atd_raw", "homography"], source=args.video_type)

    last_frame = args.clip[1] if args.clip else max(args.frames)
    limit = max(args.limit, last_frame + 2)
    print("loading %d ATD frames..." % limit)
    dataset = metrica.load_tracking_epts(meta_data=paths["atd_meta"],
                                         raw_data=paths["atd_raw"], limit=limit)
    by_frame_id = {frame.frame_id: frame for frame in dataset.frames}
    home_team = dataset.metadata.teams[0]
    away_team = dataset.metadata.teams[1]
    pitch_dimensions = dataset.metadata.pitch_dimensions

    print("loading homography...")
    matrices, _, valid = load_homography.load(source=args.video_type)

    frame_range = (range(args.clip[0], args.clip[1] + 1) if args.clip
                   else range(min(args.frames), max(args.frames) + 1))

    detections = None
    if not args.no_detections and os.path.exists(args.detections):
        print("loading ball detections from %s..." % args.detections)
        detections = load_ball_detections(args.detections)
    elif not args.no_detections:
        print("detections file not found, skipping: %s" % args.detections)

    ball_track = None
    if detections:
        references = build_ball_references(frame_range, by_frame_id, matrices, valid)
        print("tracking ball over %d frames (%d with an ATD reference)..."
              % (len(frame_range), len(references)))
        if args.tracker == "bytetrack":
            probe = cv2.VideoCapture(paths["video"])
            frame_size = (int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                         int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)))
            probe.release()
            ball_track = track_ball_native_bytetrack(detections, frame_range, frame_size,
                                                     references,
                                                     track_thresh=args.min_detection_score,
                                                     gating_distance=args.gating_distance,
                                                     track_buffer=args.max_age)
        else:
            ball_track = track_ball(detections, frame_range, references,
                                    args.min_detection_score, args.gating_distance, args.max_age)

    tracked_ball_points = None
    if not args.no_tracked_ball_csv and os.path.exists(args.tracked_ball_csv):
        print("loading tracked ball points from %s..." % args.tracked_ball_csv)
        tracked_ball_points = load_tracked_ball_points(args.tracked_ball_csv)
    elif not args.no_tracked_ball_csv:
        print("tracked ball CSV not found, skipping: %s" % args.tracked_ball_csv)

    player_boxes = None
    if args.player_tracks_csv:
        if os.path.exists(args.player_tracks_csv):
            print("loading tracked player boxes from %s..." % args.player_tracks_csv)
            player_boxes = load_player_tracks_csv(args.player_tracks_csv)
        else:
            print("player tracks CSV not found, skipping: %s" % args.player_tracks_csv)

    if args.detect_goalkeepers:
        if player_boxes is None:
            parser.error("--detect-goalkeepers needs --player-tracks-csv")
        team_sides_fn = lambda video_frame: pitch_zones.team_sides_for_video_frame(
            video_frame, home_team, away_team, game=games.GAME)
        goalkeepers = detect_goalkeepers_from_boxes(
            player_boxes, matrices, valid, team_sides_fn, pitch_dimensions,
            video_frames=frame_range, min_votes=args.goalkeeper_min_votes)

        sample_sides = team_sides_fn(frame_range.start) or {}
        print("detected goalkeepers (track_id -> team):")
        for track_id, team in sorted(goalkeepers.items()):
            print("  track_id %d -> %s (%s box)"
                 % (track_id, team.name, sample_sides.get(team, "?")))

    os.makedirs(args.out, exist_ok=True)
    capture = cv2.VideoCapture(paths["video"])
    if not capture.isOpened():
        raise IOError("could not open {}".format(paths["video"]))

    if args.clip:
        start, end = args.clip
        out_path = os.path.join(args.out, "clip_%06d_%06d.mp4" % (start, end))
        try:
            render_clip(capture=capture,
                        start=start,
                        end=end,
                        by_frame_id=by_frame_id,
                        matrices=matrices,
                        valid=valid,
                        home_team=home_team,
                        out_path=out_path,
                        play=args.play,
                        scale=args.scale,
                        opacity=args.opacity,
                        panel_fraction=args.panel_width,
                        gt=load_gt(args.gt) if args.gt else None,
                        ball_track=ball_track,
                        ball_in_birdseye=args.ball_birdseye,
                        tracked_ball_points=tracked_ball_points,
                        player_boxes=player_boxes,
                        player_match_threshold=args.player_match_threshold,
                        away_team=away_team,
                        pitch_dimensions=pitch_dimensions)


        finally:
            capture.release()
            cv2.destroyAllWindows()
        return

    try:
        for video_frame in args.frames:
            tracking_frame = by_frame_id.get(video_frame + 1)
            if tracking_frame is None:
                print("video frame %d: no ATD record, skipped" % video_frame)
                continue
            if not valid[video_frame]:
                print("video frame %d: no valid homography, skipped" % video_frame)
                continue

            image = read_video_frame(capture, video_frame)
            _, atd_matches, team_override = annotate(
                image, tracking_frame, matrices[video_frame], home_team, video_frame,
                ball_track, tracked_ball_points=tracked_ball_points,
                player_boxes=player_boxes, player_match_threshold=args.player_match_threshold,
                away_team=away_team, pitch_dimensions=pitch_dimensions)

            if args.ball_birdseye:
                track_point = ball_track.get(video_frame) if ball_track is not None else None
                birds_eye = BirdsEye(int(image.shape[1] * args.panel_width))
                if track_point is not None:
                    panel = birds_eye.render(tracking_frame, home_team,
                                             track_point_px=track_point["centre"],
                                             matrix=matrices[video_frame],
                                             atd_matches=atd_matches, team_override=team_override)
                else:
                    panel = birds_eye.render(tracking_frame, home_team,
                                             atd_matches=atd_matches, team_override=team_override)
                overlay_bottom(image, panel, args.opacity)

            out_path = os.path.join(args.out, "frame_%06d.jpg" % video_frame)
            cv2.imwrite(out_path, image)
            print("wrote %s" % out_path)

            if args.show:
                cv2.imshow("ATD overlay (any key: next, q: quit)", image)
                if cv2.waitKey(0) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
