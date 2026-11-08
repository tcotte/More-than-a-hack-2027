import os
import sys
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from kloppy.domain import Point

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "mtah_challenge"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ball_possession.possession_model import PossessionModel
from ball_possession.predict_possession import predict_labels


@dataclass(frozen=True)
class Player:
    team: str


class MissingBallPolicyTests(unittest.TestCase):
    def setUp(self):
        self.frame = SimpleNamespace(
            players_coordinates={Player("home"): Point(0.5, 0.5)},
            ball_coordinates=Point(0.5, 0.5),
        )

    def test_loss_can_abstain_and_backfill_then_resume(self):
        frames = {1: self.frame, 27: self.frame}
        labels = predict_labels(27, 0, 26, frames, "home", missing_ball_label="X")
        self.assertEqual(labels, ["H"] + ["X"] * 25 + ["H"])
        self.assertEqual(predict_labels(27, 0, 26, frames, "home"),
                         ["H"] + ["D"] * 25 + ["H"])

    def test_set_piece_dead_ball_still_detected_when_abstaining(self):
        model = PossessionModel("home", use_set_piece_heuristic=True, missing_ball_label="X")
        for i in range(49):
            self.assertEqual(model.update(i, self.frame, set_piece=True).label, "H")
        result = model.update(49, self.frame, set_piece=True)
        self.assertEqual((result.label, result.change_frame), ("D", 0))
        for i in range(50, 75):
            result = model.update(i, None, set_piece=True)
        self.assertEqual((result.label, result.change_frame), ("X", 50))

    def test_invalid_missing_label_is_rejected(self):
        with self.assertRaises(ValueError):
            PossessionModel("home", missing_ball_label="H")


if __name__ == "__main__":
    unittest.main()
