"""Frame-by-frame possession estimation from ATD tracking.

The model has two deliberately separate signals:

* ``TouchDetector`` reports a probable touch for visualization. It looks for a meaningful
  acceleration or direction change in the tracked ball, then associates it with the nearest
  player. The first ball observation after startup or a tracking gap only initializes its
  history; displacement across that gap is not evidence of a touch.
* ``PossessionModel`` estimates the owner from the nearest player within
  ``MAX_TOUCH_DISTANCE``. A team change requires that an opposing player remain nearest for
  ``CONFIRM_FRAMES`` consecutive frames. The output dates a confirmed change back to the first
  frame of that run, allowing callers to backfill labels while preserving the existing API.

If the ball is missing for ``DEAD_BALL_FRAMES`` consecutive frames, the model emits ``D`` and
dates that transition to the start of the missing run. A single missing frame does not mean a
dead ball: tracking can lose the ball during live play. Sustained absence is still only a proxy
for a stoppage, not a referee/game-state signal, and can mislabel long occlusions.

The coordinates are normalized to the pitch dimensions below. This is a lightweight heuristic,
not a trained model: it does not distinguish controlled possession from a nearby loose ball, and
its distance, persistence, and missing-ball thresholds should be evaluated on labeled data before
being treated as calibrated.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0

# Speed gain between two frames (m/frame, 0.1 = 2.5 m/s at 25 fps) that signals a kick.
ACCEL_THRESHOLD = 0.1
# Ignore smaller per-frame displacements and shallow direction changes to reduce tracking jitter.
MIN_BALL_MOVEMENT = 0.1
DIRECTION_CHANGE_DEGREES = 20.0
MAX_TOUCH_DISTANCE = 3.0
CONFIRM_FRAMES = 50
DEAD_BALL_FRAMES = 25


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
    """Detect likely touches without treating initialization or tracking gaps as a touch."""

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
                 dead_ball_frames=DEAD_BALL_FRAMES):
        self.home_team = home_team
        self.confirm_frames = confirm_frames
        self.dead_ball_frames = dead_ball_frames
        self.detector = TouchDetector()
        self.label = "X"
        self.pending = None          # (candidate label, start frame, last frame, count)
        self.missing_ball_frames = 0

    def _team_label(self, player):
        return "H" if player.team == self.home_team else "A"

    def update(self, frame_index, tracking_frame):
        touch_player, change_frame = None, None
        if tracking_frame is None:
            self.detector.reset()
        else:
            touch = self.detector.update(tracking_frame)
            if touch is not None:
                touch_player = touch[1]
            ball = tracking_frame.ball_coordinates

        if tracking_frame is None or not is_valid(ball):
            self.pending = None
            self.missing_ball_frames += 1
            if (self.missing_ball_frames >= self.dead_ball_frames
                    and self.label != "D"):
                self.label = "D"
                change_frame = frame_index - self.dead_ball_frames + 1
            return ModelOutput(self.label, touch_player, change_frame)

        self.missing_ball_frames = 0
        ball_position = np.array([ball.x * PITCH_LENGTH, ball.y * PITCH_WIDTH])
        nearest_player, _ = closest_player_to_ball(
            tracking_frame, ball_position, MAX_TOUCH_DISTANCE)

        if nearest_player is None:
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
                        and self.pending[2] == frame_index - 1):
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
