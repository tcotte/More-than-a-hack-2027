"""Use the same detection-based ball coordinates for visualization and evaluation."""

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
from kloppy.domain import Player, Point


from ball_possession.possession_model import is_valid


@dataclass
class PossessionFrame:
    players_coordinates: Mapping[Player, Point]
    ball_coordinates: Optional[Point]


def build_ball_references(frames, by_frame_id, matrices, valid):
    """Project available ATD ball positions into the corresponding video frames."""
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


def frame_with_detected_ball(tracking_frame, track_point, matrix):
    """Keep ATD players and use the detection track, or a missing ball if unavailable."""
    if tracking_frame is None:
        return None
    ball = None
    if track_point is not None and matrix is not None:
        x, y = load_homography.unproject(matrix, *track_point["centre"])
        if np.isfinite(x) and np.isfinite(y):
            ball = Point(float(x), float(y))
    return PossessionFrame(tracking_frame.players_coordinates, ball)
