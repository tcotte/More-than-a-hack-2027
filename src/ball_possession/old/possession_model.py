"""Frame-by-frame possession estimation from ATD tracking.

The model has two deliberately separate signals:

* ``TouchDetector`` reports a probable touch for visualization, based on changes in the ball's
  displacement and direction. The first observation after a tracking gap initializes its history.
* ``PossessionModel`` estimates the owner from the nearest player within
  ``MAX_TOUCH_DISTANCE``. A team change requires that an opposing player remain nearest for
  ``CONFIRM_FRAMES`` consecutive frames. The output dates a confirmed change back to the first
  frame of that run, allowing callers to backfill labels while preserving the existing API.

If the ball is missing for ``DEAD_BALL_FRAMES`` consecutive frames, the model emits ``D`` by
default and dates that transition to the start of the missing run. Optionally it emits ``X``
instead. A missing ball is not a referee/game-state signal, even over long occlusions.

The coordinates are normalized to the pitch dimensions below (0-1). This is a lightweight
heuristic, not a trained model: it does not distinguish controlled possession from a nearby loose
ball, and its thresholds should be evaluated on suitable labeled data before calibration.

Frame indices must increase. When frames are skipped, touch and persistence histories are reset:
unobserved frames cannot establish a consecutive missing-ball or possession-change run.
Optionally, a possession candidate may survive a limited number of observed frames without a
usable nearest player; only frames with actual evidence count toward confirmation.

The optional set-piece heuristic combines a Smart Tagging interval with a ball that remains
within one metre for two seconds at 25 fps. It is an uncalibrated proxy: tags include both dead
and live play, and a stationary ball does not prove a stoppage. It requires the caller to supply
the set-piece signal for each frame and is disabled by default.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0

# Speed gain between frames (m/frame, 0.1 = 2.5 m/s at 25 fps).
ACCEL_THRESHOLD = 0.1
MIN_BALL_MOVEMENT = 0.1
DIRECTION_CHANGE_DEGREES = 20.0
MAX_TOUCH_DISTANCE = 3.0
CONFIRM_FRAMES = 50
DEAD_BALL_FRAMES = 25
SET_PIECE_STILL_FRAMES = 50
SET_PIECE_STILL_RADIUS = 1.0


def is_valid(point):
    return (point is not None and point.x is not None and point.y is not None
            and np.isfinite(point.x) and np.isfinite(point.y))


def closest_player_to_ball(tracking_frame, ball_position, max_distance):
    """Return the nearest valid player and distance, or (None, infinity) if too far away."""
    closest_player, closest_distance = None, float("inf")
    for player, position in tracking_frame.players_coordinates.items():
        if not is_valid(position):
            continue
        player_position = np.array([position.x * PITCH_LENGTH, position.y * PITCH_WIDTH])
        distance = np.linalg.norm(ball_position - player_position)
        if distance < closest_distance:
            closest_player, closest_distance = player, distance
    if closest_distance > max_distance:
        return None, float("inf")
    return closest_player, closest_distance


class TouchDetector:
    """Detect likely touches without treating tracking gaps as a touch."""

    def __init__(self, accel_threshold=ACCEL_THRESHOLD, max_distance=MAX_TOUCH_DISTANCE):
        self.accel_threshold = accel_threshold
        self.max_distance = max_distance
        self.reset()

    def reset(self):
        self.previous_ball = None
        self.previous_velocity = None
        self.turn_active = False
        self.accel_active = False

    def update(self, tracking_frame):
        """Return (True, closest_player) on a touch, else None."""
        ball = tracking_frame.ball_coordinates
        if not is_valid(ball):
            self.reset()
            return None

        ball_position = np.array([ball.x * PITCH_LENGTH, ball.y * PITCH_WIDTH])
        touch_detected = False
        if self.previous_ball is not None:
            velocity = ball_position - self.previous_ball
            movement = np.linalg.norm(velocity)
            if self.previous_velocity is not None:
                accelerating = movement - np.linalg.norm(self.previous_velocity) > \
                    self.accel_threshold
                touch_detected = accelerating and not self.accel_active
                self.accel_active = accelerating
            if movement > MIN_BALL_MOVEMENT and self.previous_velocity is not None:
                previous_movement = np.linalg.norm(self.previous_velocity)
                if previous_movement > MIN_BALL_MOVEMENT:
                    cosine = np.dot(velocity, self.previous_velocity) / (
                        movement * previous_movement)
                    direction_changed = cosine < np.cos(np.deg2rad(
                        DIRECTION_CHANGE_DEGREES))
                    touch_detected = touch_detected or (direction_changed
                                                        and not self.turn_active)
                    self.turn_active = direction_changed
                else:
                    self.turn_active = False
            self.previous_velocity = velocity
        else:
            self.previous_velocity = None
        self.previous_ball = ball_position
        if not touch_detected:
            return None

        closest_player, _ = closest_player_to_ball(
            tracking_frame, ball_position, self.max_distance)
        if closest_player is None:
            return None
        return True, closest_player


@dataclass
class ModelOutput:
    label: str                       # "H", "A", "D" or "X" (no answer yet)
    touch_player: Optional[object]   # player of the touch detected on this frame
    change_frame: Optional[int]      # start frame of a newly confirmed state


class PossessionModel:
    def __init__(self, home_team, confirm_frames=CONFIRM_FRAMES,
                 dead_ball_frames=DEAD_BALL_FRAMES, use_set_piece_heuristic=False,
                 missing_ball_label="D", max_confirmation_gap=0):
        if missing_ball_label not in ("D", "X"):
            raise ValueError("missing_ball_label must be D or X")
        if max_confirmation_gap < 0:
            raise ValueError("max_confirmation_gap must be non-negative")
        self.home_team = home_team
        self.confirm_frames = confirm_frames
        self.dead_ball_frames = dead_ball_frames
        self.detector = TouchDetector()
        self.label = "X"
        self.pending = None          # (candidate label, start frame, last frame, count)
        self.missing_ball_frames = 0
        self.last_frame = None
        self.use_set_piece_heuristic = use_set_piece_heuristic
        self.missing_ball_label = missing_ball_label
        self.max_confirmation_gap = max_confirmation_gap
        self.still_ball_anchor = None
        self.still_ball_start = None

    def _team_label(self, player):
        return "H" if player.team == self.home_team else "A"

    def update(self, frame_index, tracking_frame, set_piece=False, broadcast_dead=False):
        if self.last_frame is not None:
            if frame_index <= self.last_frame:
                raise ValueError("frame indices must be strictly increasing")
            if frame_index != self.last_frame + 1:
                self.detector.reset()
                self.pending = None
                self.missing_ball_frames = 0
                self.still_ball_anchor = None
                self.still_ball_start = None
        self.last_frame = frame_index
        touch_player, change_frame = None, None
        if broadcast_dead:
            self.detector.reset()
            self.pending = None
            self.missing_ball_frames = 0
            self.still_ball_anchor = None
            self.still_ball_start = None
            if self.label != "D":
                self.label = "D"
                change_frame = frame_index
            return ModelOutput(self.label, None, change_frame)
        if tracking_frame is None:
            self.detector.reset()
        else:
            touch = self.detector.update(tracking_frame)
            if touch is not None:
                touch_player = touch[1]
            ball = tracking_frame.ball_coordinates

        if tracking_frame is None or not is_valid(ball):
            if (self.pending is not None
                    and frame_index - self.pending[2] > self.max_confirmation_gap):
                self.pending = None
            self.still_ball_anchor = None
            self.still_ball_start = None
            self.missing_ball_frames += 1
            if (self.missing_ball_frames >= self.dead_ball_frames
                    and self.label != self.missing_ball_label):
                self.pending = None
                self.label = self.missing_ball_label
                change_frame = frame_index - self.dead_ball_frames + 1
            return ModelOutput(self.label, touch_player, change_frame)

        self.missing_ball_frames = 0
        ball_position = np.array([ball.x * PITCH_LENGTH, ball.y * PITCH_WIDTH])
        if self.use_set_piece_heuristic and set_piece:
            if (self.still_ball_anchor is None
                    or np.linalg.norm(ball_position - self.still_ball_anchor)
                    > SET_PIECE_STILL_RADIUS):
                self.still_ball_anchor = ball_position
                self.still_ball_start = frame_index
            elif frame_index - self.still_ball_start + 1 >= SET_PIECE_STILL_FRAMES:
                self.pending = None
                if self.label != "D":
                    self.label = "D"
                    change_frame = self.still_ball_start
                return ModelOutput(self.label, touch_player, change_frame)
        else:
            self.still_ball_anchor = None
            self.still_ball_start = None
        nearest_player, _ = closest_player_to_ball(
            tracking_frame, ball_position, MAX_TOUCH_DISTANCE)

        if nearest_player is None:
            if (self.pending is not None
                    and frame_index - self.pending[2] > self.max_confirmation_gap):
                self.pending = None
        else:
            nearest_team = self._team_label(nearest_player)
            if self.label in ("X", "D"):
                self.label = nearest_team
                self.pending = None
                change_frame = frame_index
            elif nearest_team == self.label:
                self.pending = None
            else:
                if (self.pending is not None
                        and self.pending[0] == nearest_team
                        and frame_index - self.pending[2] <= self.max_confirmation_gap + 1):
                    candidate, start_frame, _, count = self.pending
                    count += 1
                    self.pending = (candidate, start_frame, frame_index, count)
                else:
                    self.pending = (nearest_team, frame_index, frame_index, 1)

                if self.pending[3] >= self.confirm_frames:
                    self.label = nearest_team
                    change_frame = self.pending[1]
                    self.pending = None

        return ModelOutput(self.label, touch_player, change_frame)


def smart_attacking_hints(events, start, end, home_name, away_name):
    names = {home_name.strip().casefold(): "H", away_name.strip().casefold(): "A"}
    attacking = {"build up", "progression", "creating chances", "attacking transition",
                     "long ball", "ball in final third", "ball in the box", "players in the box"}
    candidates = {}
    for event in events:
            if (event["code"] or "").strip().casefold() not in attacking:
                continue
            label = names.get((event["tags"].get("Team") or "").strip().casefold())
            if label is not None:
                for frame in range(max(start, event["start_frame"]), min(end + 1, event["end_frame"])):
                    candidates.setdefault(frame, set()).add(label)
    return {frame: next(iter(labels)) for frame, labels in candidates.items() if len(labels) == 1}


def smart_set_piece_frames(events, start, end):
    set_pieces = [event for event in events
                      if (event["code"] or "").strip().lower() == "set pieces"]
    if not set_pieces:
            raise ValueError("Smart Tagging contains no set-piece events")
    frames = set()
    for event in set_pieces:
            frames.update(range(max(start, event["start_frame"]), min(end + 1, event["end_frame"])))
    return frames


def sustained_missing_frames(frames, minimum):
    if minimum < 1:
            raise ValueError("minimum missing frames must be positive")
    result, run = set(), []
    for frame in sorted(frames):
            if run and frame != run[-1] + 1:
                if len(run) >= minimum:
                    result.update(run)
                run = []
            run.append(frame)
    if len(run) >= minimum:
            result.update(run)
    return result


def predict_labels(n_frames, start, end, by_frame_id, home_team, ball_track=None,
                       matrices=None, valid=None, set_piece_frames=None, missing_ball_label="D",
                       broadcast_dead_frames=None, fallback_atd_ball=False, player_overrides=None,
                       away_team=None, smart_team_hints=None, max_confirmation_gap=0,
                       integrate_broadcast=False):
    from ball_possession.old.detection_input import possession_frame

    model = PossessionModel(home_team, use_set_piece_heuristic=set_piece_frames is not None,
                                missing_ball_label=missing_ball_label,
                                max_confirmation_gap=max_confirmation_gap)
    labels = ["X"] * n_frames
    for video_frame in range(start, end + 1):
            frame = by_frame_id.get(video_frame + 1)
            matrix = matrices[video_frame] if ball_track is not None and valid[video_frame] else None
            frame = possession_frame(
                frame, ball_track, video_frame, matrix, fallback_atd_ball,
                player_overrides.get(video_frame) if player_overrides is not None else None,
                home_team, away_team)
            output = model.update(video_frame, frame,
                                  set_piece=set_piece_frames is not None
                                  and video_frame in set_piece_frames,
                                  broadcast_dead=integrate_broadcast
                                  and broadcast_dead_frames is not None
                                  and video_frame in broadcast_dead_frames)
            if output.change_frame is not None and output.change_frame < video_frame:
                labels[output.change_frame:video_frame] = [output.label] * (
                    video_frame - output.change_frame)
            labels[video_frame] = output.label
    if smart_team_hints is not None:
            for video_frame, label in smart_team_hints.items():
                if start <= video_frame <= end and labels[video_frame] == "X":
                    labels[video_frame] = label
    if broadcast_dead_frames is not None and not integrate_broadcast:
            for video_frame in broadcast_dead_frames:
                if start <= video_frame <= end:
                    labels[video_frame] = "D"
    return labels
