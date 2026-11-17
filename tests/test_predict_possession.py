import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from kloppy.domain import Point

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "mtah_challenge"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ball_possession.detection_input import (
    broadcast_missing_ball_frames, load_possession_ball_track, player_team_overrides,
)
from ball_possession.possession_model import (
    TouchDetector, predict_labels, smart_attacking_hints, sustained_missing_frames,
)


@dataclass(frozen=True)
class Player:
    team: str


@dataclass(frozen=True)
class TrackedPlayer:
    team: str
    jersey_no: int


class PredictionTests(unittest.TestCase):
    def test_merged_ball_track_ignores_ffill_and_backfills_missing_run(self):
        frame = SimpleNamespace(
            players_coordinates={Player("home"): Point(0.2, 0.5),
                                 Player("away"): Point(0.8, 0.5)},
            ball_coordinates=Point(0.2, 0.5),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ball.csv")
            with open(path, "w") as handle:
                handle.write("frame,x,y,source\n0,0.8,0.5,yolo_sort\n")
                for i in range(1, 26):
                    handle.write("{},0.8,0.5,ffill\n".format(i))
                handle.write("26,0.2,0.5,atd\n27,,,missing\n")
            track = load_possession_ball_track(path, 0, 27)
        self.assertEqual(track, {0: {"centre": (0.8, 0.5)},
                                 26: {"centre": (0.2, 0.5)}})
        frames = {i + 1: frame for i in range(28)}
        labels = predict_labels(28, 0, 27, frames, "home", track,
                                [np.eye(3)] * 28, [True] * 28)
        self.assertEqual(labels, ["A"] + ["D"] * 25 + ["H", "H"])

    def test_merged_ball_track_rejects_bad_schema_source_and_duplicate_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ball.csv")
            for contents, error in (
                    ("frame,x,y\n0,1,2\n", "missing columns"),
                    ("frame,x,y,source\n0,1,2,unknown\n", "unknown ball source"),
                    ("frame,x,y,source\n0,1,2,atd\n0,1,2,ffill\n", "duplicate frame"),
                    ("frame,x,y,source\n0,nan,2,atd\n", "invalid ball position")):
                with self.subTest(error=error):
                    with open(path, "w") as handle:
                        handle.write(contents)
                    with self.assertRaisesRegex(ValueError, error):
                        load_possession_ball_track(path, 0, 0)

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

    def test_broadcast_missing_overrides_backfilled_labels_only_inside_window(self):
        frame = SimpleNamespace(
            players_coordinates={Player("home"): Point(0.2, 0.5),
                                 Player("away"): Point(0.8, 0.5)},
            ball_coordinates=Point(0.2, 0.5),
        )
        frames = {i + 1: frame for i in range(1, 53)}
        track = {1: {"centre": (0.2, 0.5)}}
        track.update({i: {"centre": (0.8, 0.5)} for i in range(2, 52)})
        labels = predict_labels(54, 1, 52, frames, "home", track,
                                [np.eye(3)] * 54, [True] * 54,
                                broadcast_dead_frames={0, 5, 53})
        self.assertEqual(labels[0], "X")
        self.assertEqual(labels[4:7], ["A", "D", "A"])
        self.assertEqual(labels[53], "X")

    def test_integrated_broadcast_stoppage_changes_possession_state(self):
        home = SimpleNamespace(players_coordinates={Player("home"): Point(0.2, 0.5)},
                               ball_coordinates=Point(0.2, 0.5))
        away = SimpleNamespace(players_coordinates={Player("away"): Point(0.8, 0.5)},
                               ball_coordinates=Point(0.8, 0.5))
        labels = predict_labels(3, 0, 2, {1: home, 2: home, 3: away}, "home",
                                broadcast_dead_frames={1}, integrate_broadcast=True)
        self.assertEqual(labels, ["H", "D", "A"])

    def test_broadcast_mapping_uses_one_based_atd_and_handles_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            mapping = os.path.join(directory, "mapping.csv")
            atd = os.path.join(directory, "atd.txt")
            with open(mapping, "w", newline="") as handle:
                handle.write("frame_tactical,frame_broadcast\n10,0\n11,1\n12,2\n")
            with open(atd, "w") as handle:
                handle.write("1:players:NaN,NaN,NaN\n2:players:0.1,0.2,3\n"
                             "3:players:0.1,NaN,NaN\n")
            self.assertEqual(broadcast_missing_ball_frames(mapping, atd, 10, 11),
                             ({10}, 2))
            self.assertEqual(broadcast_missing_ball_frames(mapping, atd, 12, 13),
                             ({12}, 1))
            with open(mapping, "a") as handle:
                handle.write("13,8\n")
            with self.assertRaisesRegex(ValueError, "missing 1 referenced"):
                broadcast_missing_ball_frames(mapping, atd, 10, 13)

    def test_conflicting_correspondences_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            mapping = os.path.join(directory, "mapping.csv")
            with open(mapping, "w") as handle:
                handle.write("frame_tactical,frame_broadcast\n10,0\n10,1\n")
            with self.assertRaisesRegex(ValueError, "conflicting broadcast"):
                broadcast_missing_ball_frames(mapping, "unused", 10, 10)

    def test_sustained_broadcast_gaps_preserve_start_and_do_not_bridge_unknown_frames(self):
        self.assertEqual(sustained_missing_frames({1, 2, 4, 5, 6, 8}, 3), {4, 5, 6})
        self.assertEqual(sustained_missing_frames({1, 3}, 1), {1, 3})
        with self.assertRaises(ValueError):
            sustained_missing_frames({1}, 0)

    def test_atd_fallback_when_ball_detection_is_missing(self):
        frame = SimpleNamespace(players_coordinates={Player("home"): Point(0.2, 0.5)},
                                ball_coordinates=Point(0.2, 0.5))
        frames = {2: frame, 3: frame}
        self.assertEqual(predict_labels(4, 1, 2, frames, "home", {}, [np.eye(3)] * 4,
                                        [True] * 4), ["X"] * 4)
        self.assertEqual(predict_labels(4, 1, 2, frames, "home", {}, [np.eye(3)] * 4,
                                        [True] * 4, fallback_atd_ball=True),
                         ["X", "H", "H", "X"])

    def test_player_matches_override_team_without_discarding_unmatched_players(self):
        player = TrackedPlayer("home", 2)
        frame = SimpleNamespace(players_coordinates={player: Point(0.2, 0.5),
                                                     TrackedPlayer("away", 3): Point(0.8, 0.5)},
                                ball_coordinates=Point(0.2, 0.5))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "players.csv")
            with open(path, "w") as handle:
                handle.write("video_frame,player_key,jersey_no,side\n"
                             "1,home_2,2,away\n2,away_3,3,away\n")
            overrides = player_team_overrides(path, 1, 2)
        self.assertEqual(overrides, {1: {("home", 2): "away"}})
        self.assertEqual(predict_labels(4, 1, 2, {2: frame, 3: frame}, "home",
                                        player_overrides=overrides, away_team="away"),
                         ["X", "A", "A", "X"])
        self.assertEqual(player.team, "home")

    def test_smart_hints_use_unambiguous_attacking_phases_only(self):
        def event(code, team, start, end):
            return {"code": code, "tags": {"Team": team}, "start_frame": start, "end_frame": end}
        hints = smart_attacking_hints([
            event("BUILD UP", "Barcelona", 1, 4),
            event("PROGRESSION", "Athletic", 3, 5),
            event("DEFENDING IN MIDDLE THIRD", "Athletic", 1, 2),
            event("SET PIECES", "Barcelona", 5, 6),
            event("BUILD UP", "N/A", 5, 6),
        ], 1, 5, "Barcelona", "Athletic")
        self.assertEqual(hints, {1: "H", 2: "H", 4: "A"})
        self.assertEqual(predict_labels(7, 1, 5, {}, "home", missing_ball_label="X",
                                        smart_team_hints=hints),
                         ["X", "H", "H", "X", "A", "X", "X"])
        frame = SimpleNamespace(players_coordinates={Player("away"): Point(0.2, 0.5)},
                                ball_coordinates=Point(0.2, 0.5))
        self.assertEqual(predict_labels(3, 1, 1, {2: frame}, "home",
                                        smart_team_hints={1: "H"}), ["X", "A", "X"])

    def test_touch_detection_works_with_points_without_speed_and_resets_after_gap(self):
        player = Player("home")
        detector = TouchDetector()
        def frame(x):
            return SimpleNamespace(players_coordinates={player: Point(x, 0.5)},
                                   ball_coordinates=Point(x, 0.5))
        self.assertIsNone(detector.update(frame(0.2)))
        self.assertIsNone(detector.update(frame(0.2)))
        self.assertEqual(detector.update(frame(0.202)), (True, player))
        self.assertIsNone(detector.update(SimpleNamespace(ball_coordinates=None)))
        self.assertIsNone(detector.update(frame(0.3)))

    def test_touch_detection_sees_direction_changes_without_speed_change(self):
        player = Player("home")
        detector = TouchDetector()
        def frame(x, y):
            return SimpleNamespace(players_coordinates={player: Point(x, y)},
                                   ball_coordinates=Point(x, y))
        self.assertIsNone(detector.update(frame(0.2, 0.5)))
        self.assertIsNone(detector.update(frame(0.202, 0.5)))
        self.assertEqual(detector.update(frame(0.202, 0.5 + 0.21 / 68)), (True, player))


if __name__ == "__main__":
    unittest.main()
