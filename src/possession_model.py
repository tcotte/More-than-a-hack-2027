"""Possession model: the single place where the prediction lives.

Everything else (visualize_video_frames.py, predict_possession.py) only calls `PossessionModel`.

    model = PossessionModel(home_team)
    for frame_index, tracking_frame in ...:        # consecutive video frames
        out = model.update(frame_index, tracking_frame)   # tracking_frame may be None
        out.label, out.touch_player, out.change_frame

Pipeline:
  1. TouchDetector: a touch is a ball direction change or a sudden ball acceleration. The touching
     player is the closest one, if within MAX_TOUCH_DISTANCE metres.
  2. Possession: the first touch gives the ball to that team. Afterwards a team change is only
     accepted if the other team keeps the ball CONFIRM_FRAMES without the current team touching
     it again (this removes the flicker of isolated noisy touches). The change is dated back to
     the first touch of the new team (`change_frame`).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0

# Speed gain between two frames (m/frame, 0.1 = 2.5 m/s at 25 fps) that signals a kick.
ACCEL_THRESHOLD = 0.1
MAX_TOUCH_DISTANCE = 3.0
CONFIRM_FRAMES = 50


def is_valid(point):
    return (point is not None and point.x is not None and point.y is not None
            and not np.isnan(point.x) and not np.isnan(point.y))


class TouchDetector:
    """Detect a probable touch when the ball changes direction or suddenly accelerates."""

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
            if movement > 0.05 and self.previous_velocity is not None:
                previous_movement = np.linalg.norm(self.previous_velocity)
                if previous_movement > 0.05:
                    cosine = np.dot(velocity, self.previous_velocity) / (
                        movement * previous_movement)
                    direction_changed = cosine < np.cos(np.deg2rad(5))
                    touch_detected = touch_detected or (direction_changed
                                                        and not self.turn_active)
                    self.turn_active = direction_changed
                else:
                    self.turn_active = False
            elif movement > 0.05:
                self.turn_active = False
            self.previous_velocity = velocity
        else:
            self.previous_velocity = None

        self.previous_ball = ball_position
        if not touch_detected:
            return None

        closest_player, closest_distance = None, float("inf")
        for player, position in tracking_frame.players_coordinates.items():
            if not is_valid(position):
                continue
            player_position = np.array([position.x * PITCH_LENGTH, position.y * PITCH_WIDTH])
            distance = np.linalg.norm(ball_position - player_position)
            if distance < closest_distance:
                closest_player, closest_distance = player, distance
        if closest_distance > self.max_distance:
            return None
        return True, closest_player


@dataclass
class ModelOutput:
    label: str                       # "H", "A" or "X" (no possession known yet)
    touch_player: Optional[object]   # player of the touch detected on this frame
    change_frame: Optional[int]      # set on the frame where a team change is confirmed


class PossessionModel:
    def __init__(self, home_team, confirm_frames=CONFIRM_FRAMES):
        self.home_team = home_team
        self.confirm_frames = confirm_frames
        self.detector = TouchDetector()
        self.label = "X"
        self.pending = None          # (label, frame of its first touch)

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
                team = self._team_label(touch_player)
                if self.label == "X":
                    self.label, change_frame = team, frame_index
                elif team == self.label:
                    self.pending = None
                elif self.pending is None:
                    self.pending = (team, frame_index)

        if self.pending is not None and frame_index - self.pending[1] >= self.confirm_frames:
            self.label, change_frame = self.pending
            self.pending = None
        return ModelOutput(self.label, touch_player, change_frame)
