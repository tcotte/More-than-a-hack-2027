"""Predict possession from ball movement and nearby players."""

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
SPEED_CHANGE_THRESHOLD = 0.1  # metres per frame
DIRECTION_CHANGE_DEGREES = 20.0
MIN_BALL_MOVEMENT = 0.1  # metres per frame
MAX_PLAYER_DISTANCE = 3.0  # metres
POSSESSION_CONFIRM_FRAMES = 5
SMART_TAG_CONFLICT_DELAY = 5
POSSESSION_COOLDOWN_FRAMES = 50  # two seconds at 25 fps
STATIONARY_DIAMETER = 1.0  # metres
STATIONARY_FRAMES = 50  # two seconds of observations at 25 fps
STATIONARY_MAX_MISSING_FRAMES = 3
OUTSIDE_MARGIN = 1.0  # metres
OUTSIDE_FRAMES = 3
PLAYER_SPEED_THRESHOLD = 8.0  # km/h
SLOW_PLAYERS_FRAMES = 25  # one second at 25 fps
MIN_TRACKED_PLAYERS = 18


def is_valid(point):
    return (point is not None and point.x is not None and point.y is not None
            and np.isfinite(point.x) and np.isfinite(point.y))


class TouchDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous_position = None
        self.previous_vector = None

    def update(self, tracking_frame):
        ball = tracking_frame.ball_coordinates if tracking_frame is not None else None
        if not is_valid(ball):
            self.reset()
            return None

        position = (ball.x, ball.y)
        if self.previous_position is None:
            self.previous_position = position
            return None

        vector = ((position[0] - self.previous_position[0]) * PITCH_LENGTH,
                  (position[1] - self.previous_position[1]) * PITCH_WIDTH)
        touched = False
        if self.previous_vector is not None:
            speed = math.hypot(*vector)
            previous_speed = math.hypot(*self.previous_vector)
            speed_changed = abs(speed - previous_speed) >= SPEED_CHANGE_THRESHOLD
            direction_changed = False
            if speed >= MIN_BALL_MOVEMENT and previous_speed >= MIN_BALL_MOVEMENT:
                cosine = sum(a * b for a, b in zip(vector, self.previous_vector)) / (
                    speed * previous_speed)
                direction_changed = cosine <= math.cos(math.radians(DIRECTION_CHANGE_DEGREES))
            touched = speed_changed or direction_changed
        self.previous_position = position
        self.previous_vector = vector
        if not touched:
            return None

        closest_player = None
        closest_distance = MAX_PLAYER_DISTANCE
        for player, player_position in tracking_frame.players_coordinates.items():
            if not is_valid(player_position):
                continue
            distance = math.hypot(
                (player_position.x - ball.x) * PITCH_LENGTH,
                (player_position.y - ball.y) * PITCH_WIDTH,
            )
            if distance <= closest_distance:
                closest_player = player
                closest_distance = distance
        return closest_player


class PlayerMotionDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous_positions = None
        self.slow_frames = 0

    def update(self, tracking_frame):
        positions = {
            player: (point.x, point.y)
            for player, point in tracking_frame.players_coordinates.items()
            if is_valid(point)
        } if tracking_frame is not None else {}
        previous = self.previous_positions
        self.previous_positions = positions
        if (len(positions) < MIN_TRACKED_PLAYERS or previous is None
                or not positions.keys() <= previous.keys()):
            self.slow_frames = 0
            return None

        fast_players = sum(
            math.hypot((position[0] - previous[player][0]) * PITCH_LENGTH,
                       (position[1] - previous[player][1]) * PITCH_WIDTH)
            * 25 * 3.6 >= PLAYER_SPEED_THRESHOLD
            for player, position in positions.items()
        )
        slow = fast_players <= 1
        self.slow_frames = self.slow_frames + 1 if slow else 0
        return slow


@dataclass
class ModelOutput:
    label: str
    touch_player: Optional[object]
    change_frame: Optional[int]


class PossessionModel:
    """Confirm team changes and detect dead balls."""

    def __init__(self, home_team, missing_ball_label="D"):
        self.home_team = home_team
        self.label = "A"
        self.last_possession = "A"
        self.detector = TouchDetector()
        self.player_motion = PlayerMotionDetector()
        self.last_frame = None
        self.stationary_positions = deque()
        self.stationary_missing_frames = 0
        self.outside_frames = 0
        self.dead_ball_position = None
        self.dead_ball_cause = None
        self.pending_team = None
        self.pending_frames = 0
        self.pending_required = POSSESSION_CONFIRM_FRAMES
        self.last_team_change = None

    def clear_pending(self):
        self.pending_team = None
        self.pending_frames = 0
        self.pending_required = POSSESSION_CONFIRM_FRAMES

    def reset_observations(self):
        self.detector.reset()
        self.stationary_positions.clear()
        self.stationary_missing_frames = 0
        self.outside_frames = 0
        self.clear_pending()

    def update(self, frame_index, tracking_frame, smart_team=None):
        if smart_team not in (None, "H", "A"):
            raise ValueError("smart_team must be H, A or None")
        if self.last_frame is not None:
            if frame_index <= self.last_frame:
                raise ValueError("frame indices must be strictly increasing")
            if frame_index != self.last_frame + 1:
                self.reset_observations()
                self.player_motion.reset()
        self.last_frame = frame_index
        players_slow = self.player_motion.update(tracking_frame)
        ball = tracking_frame.ball_coordinates if tracking_frame is not None else None
        if self.label == "D" and self.dead_ball_cause == "players":
            if players_slow is False:
                self.label = self.last_possession
                self.dead_ball_cause = None
                self.reset_observations()
                self.detector.update(tracking_frame)
                return ModelOutput(self.label, None, frame_index)
            return ModelOutput(self.label, None, None)

        if self.label != "D" and self.player_motion.slow_frames >= SLOW_PLAYERS_FRAMES:
            self.label = "D"
            self.dead_ball_cause = "players"
            self.reset_observations()
            return ModelOutput(self.label, None, frame_index)

        if not is_valid(ball):
            self.detector.reset()
            self.clear_pending()
            self.outside_frames = 0
            self.stationary_missing_frames += 1
            if self.stationary_missing_frames > STATIONARY_MAX_MISSING_FRAMES:
                self.stationary_positions.clear()
            return ModelOutput(self.label, None, None)

        self.stationary_missing_frames = 0
        position = (ball.x * PITCH_LENGTH, ball.y * PITCH_WIDTH)
        inside = (0 <= position[0] <= PITCH_LENGTH and 0 <= position[1] <= PITCH_WIDTH)
        if self.label == "D":
            if inside and math.dist(position, self.dead_ball_position) > STATIONARY_DIAMETER:
                if self.player_motion.slow_frames >= SLOW_PLAYERS_FRAMES:
                    self.dead_ball_cause = "players"
                    self.dead_ball_position = None
                    return ModelOutput(self.label, None, None)
                self.label = self.last_possession
                self.dead_ball_position = None
                self.dead_ball_cause = None
                self.reset_observations()
                self.detector.update(tracking_frame)
                return ModelOutput(self.label, None, frame_index)
            return ModelOutput(self.label, None, None)

        outside = (position[0] < -OUTSIDE_MARGIN
                   or position[0] > PITCH_LENGTH + OUTSIDE_MARGIN
                   or position[1] < -OUTSIDE_MARGIN
                   or position[1] > PITCH_WIDTH + OUTSIDE_MARGIN)
        self.outside_frames = self.outside_frames + 1 if outside else 0
        while any(math.dist(position, previous) > STATIONARY_DIAMETER
                  for previous in self.stationary_positions):
            self.stationary_positions.popleft()
        self.stationary_positions.append(position)
        if len(self.stationary_positions) >= STATIONARY_FRAMES or self.outside_frames >= OUTSIDE_FRAMES:
            self.label = "D"
            self.dead_ball_position = position
            self.dead_ball_cause = "ball"
            self.reset_observations()
            return ModelOutput(self.label, None, frame_index)

        player = self.detector.update(tracking_frame)
        change_frame = None
        if player is not None:
            label = "H" if player.team == self.home_team else "A"
            if label == self.label:
                self.clear_pending()
            elif self.pending_team != label:
                self.pending_team = label
                self.pending_frames = 0
                self.pending_required = POSSESSION_CONFIRM_FRAMES

        if self.pending_team is not None:
            nearby = any(
                is_valid(player_position)
                and ("H" if candidate.team == self.home_team else "A") == self.pending_team
                and math.hypot((player_position.x - ball.x) * PITCH_LENGTH,
                               (player_position.y - ball.y) * PITCH_WIDTH) <= MAX_PLAYER_DISTANCE
                for candidate, player_position in tracking_frame.players_coordinates.items()
            )
            if not nearby:
                self.clear_pending()
            else:
                self.pending_frames += 1
                if (self.pending_frames == POSSESSION_CONFIRM_FRAMES
                        and smart_team is not None and smart_team != self.pending_team):
                    self.pending_required += SMART_TAG_CONFLICT_DELAY
                cooldown_complete = (self.last_team_change is None
                                     or frame_index - self.last_team_change
                                     >= POSSESSION_COOLDOWN_FRAMES)
                if self.pending_frames >= self.pending_required and cooldown_complete:
                    self.label = self.pending_team
                    self.last_possession = self.label
                    self.last_team_change = frame_index
                    change_frame = frame_index
                    self.clear_pending()
        return ModelOutput(self.label, player, change_frame)
