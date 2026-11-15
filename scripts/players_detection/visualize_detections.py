
import os
import sys
import xml.etree.ElementTree as ET
from collections import namedtuple

import cv2
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

from games import require  # noqa: E402
from starter.load_atd import video_frame_of  # noqa: E402
from starter.load_homography import load as load_homography, project, unproject  # noqa: E402
import pitch_zones  # noqa: E402
from utils import GAME_PERIODS  # noqa: E402

# Real pitch size in metres (see src/visualize_video_frames.py's PITCH_LENGTH/PITCH_WIDTH) and the
# standard penalty area dimensions (16.5 m deep, 40.3 m wide), normalised 0-1 the way
# pitch_zones.penalty_box_bounds() expects.
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
_PitchDimensions = namedtuple("_PitchDimensions", ["penalty_area_length", "penalty_area_width"])
PENALTY_AREA_DIMENSIONS = _PitchDimensions(
    penalty_area_length=16.5 / PITCH_LENGTH,
    penalty_area_width=40.3 / PITCH_WIDTH,
)

# team_id 0 is HOME, 1 is AWAY (see _atd_player_teams()/TEAM_COLOURS below).
HOME_TEAM_ID, AWAY_TEAM_ID = 0, 1


def _half_of_video_frame(frame_id):
    """1 or 2 for a frame inside a known half (utils.py's GAME_PERIODS), else None.

    None covers halftime, before kickoff and after the final whistle -- there is no side-to-team
    mapping to apply there.
    """
    if GAME_PERIODS.START_FIRST_HALF.value <= frame_id <= GAME_PERIODS.END_FIRST_HALF.value:
        return 1
    if GAME_PERIODS.START_SECOND_HALF.value <= frame_id <= GAME_PERIODS.END_SECOND_HALF.value:
        return 2
    return None


PLAYER_CLASS_ID = 2

ATD_FIELD_SEP = ":"

# BGR, same convention as src/visualize_video_frames.py's HOME_COLOUR/AWAY_COLOUR.
TEAM_COLOURS = {
    0: (40, 39, 214),   # red-ish
    1: (180, 119, 31),  # blue-ish
}


def _team_colour(team_id):
    """A fixed colour for known team ids, else a deterministic one for anything else."""
    if team_id in TEAM_COLOURS:
        return TEAM_COLOURS[team_id]
    return (
        (team_id * 53 + 40) % 255,
        (team_id * 89 + 120) % 255,
        (team_id * 131 + 180) % 255,
    )


def _atd_player_teams():
    """{player_slot_index (1-based, matches raw-file column order): team_id (0 or 1)}.

    team_id 0 is the first team declared in the XML (Barcelona, the home side -- same convention
    as TEAM_COLOURS / src/visualize_video_frames.py's HOME_COLOUR), team_id 1 is the second.
    """
    meta_path = require(["atd_meta"])[0]
    root = ET.parse(meta_path).getroot()

    team_order = []
    player_team = {}
    for player in root.iter("Player"):
        player_id = int(player.get("id"))
        team_ref = player.get("teamId")
        if team_ref not in team_order:
            team_order.append(team_ref)
        player_team[player_id] = team_order.index(team_ref)

    return player_team


def load_atd_player_centres(video_frames):
    """{video_frame: (n, 3) array of (x, y, team_id)} in normalised (0-1) ATD coordinates.

    Only *player* points are returned (the ball, which has no team, is left out) -- team_id is
    always 0 or 1, the player's team.
    """
    player_team = _atd_player_teams()
    wanted = set(video_frames)
    raw_path = require(["atd_raw"])[0]

    centres = {}
    with open(raw_path) as handle:
        for line in handle:
            sep = line.index(ATD_FIELD_SEP)
            frame_id = int(line[:sep])
            video_frame = video_frame_of(frame_id)

            if video_frame not in wanted:
                continue

            rest = line[sep + 1:].rstrip("\n")
            players_blob, _, _ball_blob = rest.partition(ATD_FIELD_SEP)

            points = []
            for slot, entry in enumerate(players_blob.split(";"), start=1):
                x_str, y_str, _ = entry.split(",")
                if x_str != "NaN" and y_str != "NaN" and slot in player_team:
                    points.append((float(x_str), float(y_str), player_team[slot]))

            centres[video_frame] = np.array(points, dtype=np.float64) if points else \
                np.empty((0, 3), dtype=np.float64)

    return centres


def assign_team_ids_from_atd(df, threshold):
    """Add a "team_id" column to `df` (0, 1 or NaN) without dropping any row.

    For every frame, ATD player centres (projected to pixels via the per-frame homography) and
    detection box bottom-middle points (a player's feet, which is where the ATD centre sits) are
    matched one-to-one using the Hungarian algorithm (minimising total assignment distance);
    pairs whose distance exceeds `threshold` px are discarded. Boxes with no close-enough ATD
    match -- including every row on a frame with no valid homography -- keep team_id NaN.
    """
    frames = sorted(df["frame"].unique())
    atd_centres = load_atd_player_centres(frames)
    matrices, _, valid = load_homography()

    team_ids = np.full(len(df), np.nan)

    centres_x = (df["x1"].values + df["x2"].values) / 2.0
    centres_y = df["y2"].values

    for frame_id, group in df.groupby("frame"):
        idx = group.index.values

        if frame_id >= len(valid) or not valid[frame_id]:
            continue

        atd = atd_centres.get(frame_id)
        if atd is None or len(atd) == 0:
            continue

        px, py = project(matrices[frame_id], atd[:, 0], atd[:, 1])
        atd_points = np.stack([px, py], axis=1)
        atd_teams = atd[:, 2]

        cx = centres_x[idx]
        cy = centres_y[idx]

        # (n_atd_centres, n_boxes) distance matrix.
        dist = np.sqrt(
            (atd_points[:, 0, None] - cx[None, :]) ** 2
            + (atd_points[:, 1, None] - cy[None, :]) ** 2
        )

        # Hungarian algorithm: optimal one-to-one assignment minimising total distance (each ATD
        # centre matched to at most one box and vice versa).
        atd_rows, box_cols = linear_sum_assignment(dist)
        for atd_i, box_i in zip(atd_rows, box_cols):
            if dist[atd_i, box_i] <= threshold:
                team_ids[idx[box_i]] = atd_teams[atd_i]

    df = df.copy()
    df["team_id"] = team_ids
    return df


def flag_penalty_box_detections(df):
    """Add "in_penalty_box" (bool) and "team_id" (0, 1 or NaN) columns to `df`, dropping no row.

    For every frame with a valid homography, each detection's bottom-middle point (its feet) is
    unprojected back to normalised pitch coordinates and tested against both penalty boxes
    (pitch_zones.penalty_box_bounds(), using the standard FIFA penalty area size -- there is no
    per-team "own box" distinction here, just "is this box standing inside either one").

    When several detections fall inside the *same* side's box on a frame, only the one closest to
    that side's goal line (x=0 for "left", x=1 for "right", in normalised pitch coordinates) is
    flagged -- the goalkeeper is the player nearest their own goal, not every player who happens
    to be defending/attacking inside the box. The two sides are handled independently, so a frame
    can have up to one flagged detection per box.

    The flagged goalkeeper's "team_id" is set from which half the frame falls in (utils.py's
    GAME_PERIODS) and which side it is defending -- home defends "left" in the first half and
    "right" in the second (pitch_zones.HOME_SIDE_BY_HALF), away is the mirror image -- overriding
    whatever team_id that detection already had (e.g. from assign_team_ids_from_atd()), since the
    goalkeeper's ATD point is not always reliable. If `df` has no "team_id" column yet, one is
    added (NaN everywhere else). Frames outside both halves (halftime, before kickoff, after the
    final whistle) get no team override, since there is no known side-to-team mapping there.

    Rows on a frame with no valid homography are left with "in_penalty_box" False, same as any
    row with no inside match.
    """
    matrices, _, valid = load_homography()

    df = df.copy()
    if "team_id" not in df.columns:
        df["team_id"] = np.nan
    df["in_penalty_box"] = False

    sides = [("left", 0.0), ("right", 1.0)]
    bounds_by_side = {
        side: pitch_zones.penalty_box_bounds(side, PENALTY_AREA_DIMENSIONS)
        for side, _ in sides
    }

    for frame_id, group in df.groupby("frame"):
        if frame_id >= len(valid) or not valid[frame_id]:
            continue

        cx = (group["x1"].values + group["x2"].values) / 2.0
        cy = group["y2"].values
        norm_x, norm_y = unproject(matrices[frame_id], cx, cy)

        half = _half_of_video_frame(frame_id)
        home_side = pitch_zones.HOME_SIDE_BY_HALF.get(half) if half is not None else None

        for side, goal_x in sides:
            x_min, x_max, y_min, y_max = bounds_by_side[side]
            inside = (
                (norm_x >= x_min) & (norm_x <= x_max)
                & (norm_y >= y_min) & (norm_y <= y_max)
            )
            if not inside.any():
                continue

            # Several players in the box: only the one nearest the goal line is the goalkeeper.
            inside_positions = np.flatnonzero(inside)
            nearest = inside_positions[np.argmin(np.abs(norm_x[inside_positions] - goal_x))]
            label = group.index[nearest]

            df.loc[label, "in_penalty_box"] = True
            if home_side is not None:
                df.loc[label, "team_id"] = HOME_TEAM_ID if side == home_side else AWAY_TEAM_ID

    return df


def draw_detections(frame, detections, class_names=None):
    """Draw bounding boxes and labels on a video frame.

    Boxes with a (non-NaN) "team_id" column are coloured by team -- one colour per team, so the
    two sides are visually distinguishable -- regardless of class_id. Rows without a usable
    team_id fall back to a deterministic per-class colour (class_id 0 if the CSV has no class_id
    column at all, e.g. assets/detections/players/players_tracked_bytetrack.csv).

    When the detections carry a "track_id" column (e.g. a CSV produced by track_with_norfair.py
    or track_with_bytetrack.py), the label shows the track id instead of the class name, since
    identity across frames is then the more useful thing to read off the overlay.

    Boxes with an "in_penalty_box" column set to True (see flag_penalty_box_detections()) get an
    extra "GK" tag drawn above the regular label, on the assumption that a player standing alone
    in a penalty area is that team's goalkeeper.
    """

    has_team_id = "team_id" in detections.columns
    has_track_id = "track_id" in detections.columns
    has_class_id = "class_id" in detections.columns
    has_penalty_box = "in_penalty_box" in detections.columns

    for _, det in detections.iterrows():
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])

        score = float(det["score"])
        class_id = int(det["class_id"]) if has_class_id else 0

        team_id = None
        if has_team_id and pd.notna(det["team_id"]):
            team_id = int(det["team_id"])

        if team_id is not None:
            color = _team_colour(team_id)
        else:
            # Generate a deterministic color for each class
            color = (
                (class_id * 37 + 80) % 255,
                (class_id * 67 + 150) % 255,
                (class_id * 97 + 200) % 255
            )

        # Track id, if available, otherwise fall back to the class name.
        if has_track_id and pd.notna(det["track_id"]):
            identifier = f"ID {int(det['track_id'])}"
        else:
            identifier = (
                class_names.get(class_id, str(class_id))
                if class_names else str(class_id)
            )

        label = (f"{identifier} T{team_id}: {score:.2f}" if team_id is not None
                 else f"{identifier}: {score:.2f}")

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness=2
        )

        # Label background
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        label_y = max(y1, text_h + baseline)

        cv2.rectangle(
            frame,
            (x1, label_y - text_h - baseline),
            (x1 + text_w, label_y),
            color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1, label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # "GK" tag, stacked above the regular label, for a box flagged as alone in a penalty box.
        if has_penalty_box and bool(det["in_penalty_box"]):
            gk_label = "GK"
            (gk_w, gk_h), gk_baseline = cv2.getTextSize(
                gk_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2
            )

            gk_y = label_y - text_h - baseline

            cv2.rectangle(
                frame,
                (x1, gk_y - gk_h - gk_baseline),
                (x1 + gk_w, gk_y),
                (0, 215, 255),
                -1
            )

            cv2.putText(
                frame,
                gk_label,
                (x1, gk_y - gk_baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )

    return frame


def draw_atd_centers(frame, points, radius=6):
    """Draw ATD player centres (already projected to pixel coords) as small circles.

    `points` is an (n, 3) array of (x, y, team_id) in pixel coordinates; each circle is coloured
    by team, with a filled dot in its centre so it stays visible even if detections overlap it.
    """
    for x, y, team_id in points:
        color = _team_colour(int(team_id))
        center = (int(round(x)), int(round(y)))
        cv2.circle(frame, center, radius, color, 2)
        cv2.circle(frame, center, 2, color, -1)

    return frame


def _project_atd_centers_by_frame(frames):
    """{video_frame: (n, 3) array of (px, py, team_id)} for frames with a valid homography."""
    atd_centres = load_atd_player_centres(frames)
    matrices, _, valid = load_homography()

    projected = {}
    for frame_id in frames:
        if frame_id >= len(valid) or not valid[frame_id]:
            continue

        atd = atd_centres.get(frame_id)
        if atd is None or len(atd) == 0:
            continue

        px, py = project(matrices[frame_id], atd[:, 0], atd[:, 1])
        projected[frame_id] = np.stack([px, py, atd[:, 2]], axis=1)

    return projected


def visualize_detections(video_path, csv_path, output_path,
                         confidence_threshold=0.0,
                         start_frame=0,
                         end_frame=None,
                         class_names=None,
                         assign_atd_teams=False,
                         atd_threshold=25.0,
                         show_atd_centers=False,
                         flag_penalty_box=False):

    # Load detections
    df = pd.read_csv(csv_path)

    required_columns = [
        "frame", "x1", "y1", "x2", "y2", "score"
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    # Filter detections by confidence
    df = df[df["score"] >= confidence_threshold]

    if assign_atd_teams:
        n_total = len(df)
        df = assign_team_ids_from_atd(df, atd_threshold)
        n_matched = int(df["team_id"].notna().sum())
        print(f"ATD team assignment: matched {n_matched}/{n_total} detections within "
              f"{atd_threshold}px of an ATD player position")

    if flag_penalty_box:
        df = flag_penalty_box_detections(df)
        n_gk = int(df["in_penalty_box"].sum())
        print(f"Penalty box flagging: {n_gk}/{len(df)} detections tagged \"GK\"")

    # Group detections by frame for faster lookup
    detections_by_frame = {
        int(frame_id): group
        for frame_id, group in df.groupby("frame")
    }

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError("Invalid start frame")

    if end_frame is None:
        end_frame = total_frames - 1

    if end_frame < start_frame or end_frame >= total_frames:
        raise ValueError("Invalid end frame")

    atd_centers_by_frame = (
        _project_atd_centers_by_frame(range(start_frame, end_frame + 1))
        if show_atd_centers else {}
    )

    # Jump to the requested starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_id = start_frame

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")

    # frame_id = 0

    print(f"Video: {width}x{height} @ {fps:.2f} FPS")
    print(f"Total frames: {total_frames}")
    print(f"Detections: {len(df)}")

    window_name = "Video - Detection Visualization"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    paused = False

    while frame_id <= end_frame:

        if not paused:
            ret, frame = cap.read()

            if not ret:
                break

            detections = detections_by_frame.get(frame_id)

            if detections is not None:
                frame = draw_detections(
                    frame,
                    detections,
                    class_names
                )

            if show_atd_centers and frame_id in atd_centers_by_frame:
                frame = draw_atd_centers(frame, atd_centers_by_frame[frame_id])

            # Display frame number
            cv2.putText(
                frame,
                f"Frame: {frame_id}/{end_frame}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            # Save annotated frame
            writer.write(frame)

            frame_id += 1

            # Keep current frame for display
            display_frame = frame.copy()

        # Display video
        cv2.imshow(window_name, display_frame)

        # Keyboard controls
        key = cv2.waitKey(0 if paused else max(1, int(1000 / fps))) & 0xFF

        if key == ord('q'):
            print("Visualization stopped by user.")
            break

        elif key == ord('p'):
            paused = not paused

        elif key == ord('n') and paused:
            paused = False
            ret, next_frame = cap.read()

            if ret and frame_id <= end_frame:
                detections = detections_by_frame.get(frame_id)

                if detections is not None:
                    next_frame = draw_detections(
                        next_frame,
                        detections,
                        None
                    )

                if show_atd_centers and frame_id in atd_centers_by_frame:
                    next_frame = draw_atd_centers(next_frame, atd_centers_by_frame[frame_id])

                cv2.putText(
                    next_frame,
                    f"Frame: {frame_id}/{end_frame}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                writer.write(next_frame)
                frame_id += 1
                display_frame = next_frame.copy()

            paused = True

    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Visualize CSV detections on a video"
    )

    parser.add_argument("--video", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--conf", type=float, default=0.0)
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame to process (inclusive)"
    )

    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Last frame to process (inclusive)"
    )

    parser.add_argument(
        "--assign-atd-teams",
        action="store_true",
        help="For every ATD (src/data/tactical/BAR-ATH_tactical_atd.xml) player position, find "
             "the closest detection box on that frame and, if it is close enough, assign that "
             "box the ATD player's team_id (no detections are dropped). Matched boxes are then "
             "coloured by team in the visualization"
    )

    parser.add_argument(
        "--atd-threshold",
        type=float,
        default=25.0,
        help="Max distance in pixels between an ATD position and its nearest box centre "
             "(used with --assign-atd-teams)"
    )

    parser.add_argument(
        "--show-atd-centers",
        action="store_true",
        help="Overlay each ATD player centre (projected to pixels via the per-frame homography) "
             "as a small circle coloured by team, independently of --assign-atd-teams"
    )

    parser.add_argument(
        "--flag-penalty-box",
        action="store_true",
        help="For every detection, unproject its bottom-middle point (its feet) to pitch "
             "coordinates and check it against both standard-size penalty boxes; boxes inside "
             "either one are tagged \"GK\" above their label"
    )

    args = parser.parse_args()

    visualize_detections(
        video_path=args.video,
        csv_path=args.csv,
        output_path=args.output,
        confidence_threshold=args.conf,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        assign_atd_teams=args.assign_atd_teams,
        atd_threshold=args.atd_threshold,
        show_atd_centers=args.show_atd_centers,
        flag_penalty_box=args.flag_penalty_box
    )
