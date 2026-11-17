import os
import sys
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from kloppy.domain import Point

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "mtah_challenge"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ball_possession.possession_model import PossessionModel, predict_labels


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

    def test_candidate_survives_short_gap_but_missing_frames_do_not_count(self):
        opponent_frame = SimpleNamespace(
            players_coordinates={Player("away"): Point(0.5, 0.5)},
            ball_coordinates=Point(0.5, 0.5),
        )
        model = PossessionModel("home", confirm_frames=3, max_confirmation_gap=1)
        self.assertEqual(model.update(0, self.frame).label, "H")
        self.assertEqual(model.update(1, opponent_frame).label, "H")
        self.assertEqual(model.update(2, opponent_frame).label, "H")
        self.assertEqual(model.update(3, None).label, "H")
        result = model.update(4, opponent_frame)
        self.assertEqual((result.label, result.change_frame), ("A", 1))

    def test_candidate_expires_after_long_gap_or_skipped_frames(self):
        opponent_frame = SimpleNamespace(
            players_coordinates={Player("away"): Point(0.5, 0.5)},
            ball_coordinates=Point(0.5, 0.5),
        )
        for skipped in (False, True):
            with self.subTest(skipped=skipped):
                model = PossessionModel("home", confirm_frames=2, max_confirmation_gap=1)
                model.update(0, self.frame)
                model.update(1, opponent_frame)
                if not skipped:
                    model.update(2, None)
                    model.update(3, None)
                self.assertEqual(model.update(4, opponent_frame).label, "H")
                result = model.update(5, opponent_frame)
                self.assertEqual((result.label, result.change_frame), ("A", 4))

    def test_loose_ball_gap_is_tolerated_but_current_team_cancels_candidate(self):
        opponent_frame = SimpleNamespace(
            players_coordinates={Player("away"): Point(0.5, 0.5)},
            ball_coordinates=Point(0.5, 0.5),
        )
        loose_frame = SimpleNamespace(players_coordinates={},
                                      ball_coordinates=Point(0.5, 0.5))
        model = PossessionModel("home", confirm_frames=2, max_confirmation_gap=1)
        model.update(0, self.frame)
        model.update(1, opponent_frame)
        model.update(2, loose_frame)
        self.assertEqual(model.update(3, self.frame).label, "H")
        self.assertEqual(model.update(4, opponent_frame).label, "H")
        result = model.update(5, opponent_frame)
        self.assertEqual((result.label, result.change_frame), ("A", 4))

    def test_confirmed_dead_ball_resets_candidate_and_allows_restart(self):
        opponent_frame = SimpleNamespace(
            players_coordinates={Player("away"): Point(0.5, 0.5)},
            ball_coordinates=Point(0.5, 0.5),
        )
        model = PossessionModel("home", confirm_frames=3, max_confirmation_gap=2)
        model.update(0, self.frame)
        model.update(1, opponent_frame)
        self.assertEqual(model.update(2, opponent_frame, broadcast_dead=True).label, "D")
        self.assertEqual(model.update(3, opponent_frame, broadcast_dead=True).label, "D")
        result = model.update(4, opponent_frame)
        self.assertEqual((result.label, result.change_frame), ("A", 4))


if __name__ == "__main__":
    unittest.main()
