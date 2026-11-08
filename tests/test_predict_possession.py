import os
import sys
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from kloppy.domain import Point

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "mtah_challenge"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ball_possession.predict_possession import predict_labels


@dataclass(frozen=True)
class Player:
    team: str


class PredictionTests(unittest.TestCase):
    def test_selected_track_drives_labels_and_preserves_outside_frames(self):
        frame = SimpleNamespace(
            players_coordinates={Player("home"): Point(0.2, 0.5),
                                 Player("away"): Point(0.8, 0.5)},
            ball_coordinates=Point(0.2, 0.5),
        )
        frames = {2: frame, 3: frame, 4: frame}
        matrices = [np.eye(3)] * 5
        valid = [True] * 5
        track = {1: {"centre": (0.8, 0.5)}}

        self.assertEqual(predict_labels(5, 1, 3, frames, "home"),
                         ["X", "H", "H", "H", "X"])
        self.assertEqual(predict_labels(5, 1, 3, frames, "home", track, matrices, valid),
                         ["X", "A", "A", "A", "X"])
        self.assertEqual(frame.ball_coordinates.x, 0.2)

    def test_confirmed_changes_are_backfilled_inside_window(self):
        frame = SimpleNamespace(
            players_coordinates={Player("home"): Point(0.2, 0.5),
                                 Player("away"): Point(0.8, 0.5)},
            ball_coordinates=Point(0.2, 0.5),
        )
        frames = {i + 1: frame for i in range(1, 53)}
        track = {1: {"centre": (0.2, 0.5)}}
        track.update({i: {"centre": (0.8, 0.5)} for i in range(2, 52)})
        labels = predict_labels(54, 1, 52, frames, "home", track,
                                [np.eye(3)] * 54, [True] * 54)
        self.assertEqual(labels, ["X", "H"] + ["A"] * 51 + ["X"])


if __name__ == "__main__":
    unittest.main()
