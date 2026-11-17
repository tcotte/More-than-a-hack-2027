import csv
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from kloppy.domain import Point

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "mtah_challenge"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ball_possession import detection_input, predict_possession
from ball_possession.detection_input import smart_attacking_hints
from ball_possession.possession_model import PossessionModel, TouchDetector
import visualize_video_frames as visualizer


@dataclass(frozen=True)
class Player:
    team: object


@dataclass(frozen=True)
class MovingPlayer:
    team: str
    number: int


@dataclass(frozen=True)
class Team:
    name: str


class TouchPossessionTests(unittest.TestCase):
    def test_touch_requires_significant_speed_change_or_direction_change(self):
        home = Player("home")

        def frame(x, y):
            ball = Point(x / 105, y / 68)
            return SimpleNamespace(ball_coordinates=ball, players_coordinates={home: ball})

        detector = TouchDetector()
        self.assertIsNone(detector.update(frame(10, 34)))
        self.assertIsNone(detector.update(frame(11, 34)))
        self.assertIsNone(detector.update(frame(12.09, 34)))
        self.assertEqual(detector.update(frame(13.29, 34)), home)

        detector = TouchDetector()
        self.assertIsNone(detector.update(frame(10, 34)))
        self.assertIsNone(detector.update(frame(11, 34)))
        self.assertEqual(detector.update(frame(11, 35)), home)

    def test_possession_changes_only_on_touch_and_persists_through_missing_ball(self):
        home, away = Player("home"), Player("away")

        def frame(x, player):
            position = Point(x / 105, 0.5)
            other = away if player == home else home
            return SimpleNamespace(ball_coordinates=position,
                                   players_coordinates={player: position,
                                                        other: Point((x + 5) / 105, 0.5)})

        model = PossessionModel("home")
        self.assertEqual(model.update(0, frame(0, home)).label, "A")
        self.assertEqual(model.update(1, frame(1, home)).label, "A")
        for i in range(2, 6):
            self.assertEqual(model.update(i, frame(i + 0.3, home)).label, "A")
        self.assertEqual(model.update(6, frame(6.3, home)).label, "H")
        for i in range(7, 56):
            self.assertEqual(model.update(i, frame(i + 0.6, away)).label, "H")
        result = model.update(56, frame(56.6, away))
        self.assertEqual((result.label, result.touch_player, result.change_frame),
                         ("A", None, 56))
        self.assertEqual(model.update(57, None).label, "A")
        self.assertEqual(model.update(58, frame(58, home)).label, "A")

    def test_dead_ball_is_not_blocked_by_team_change_cooldown(self):
        home, away = Player("home"), Player("away")
        ball = Point(0.5, 0.5)
        close_home = SimpleNamespace(ball_coordinates=ball,
                                     players_coordinates={home: ball, away: Point(0.6, 0.5)})
        close_away = SimpleNamespace(ball_coordinates=ball,
                                     players_coordinates={home: Point(0.6, 0.5), away: ball})
        outside = SimpleNamespace(ball_coordinates=Point(-1.1 / 105, 0.5),
                                  players_coordinates={})
        model = PossessionModel("home")
        with patch.object(model.detector, "update", return_value=None) as detect:
            for i in range(5):
                detect.return_value = home if i == 0 else None
                result = model.update(i, close_home)
            self.assertEqual((result.label, result.change_frame), ("H", 4))

            detect.return_value = None
            self.assertEqual(model.update(5, outside).label, "H")
            self.assertEqual(model.update(6, outside).label, "H")
            self.assertEqual(model.update(7, outside).label, "D")
            self.assertEqual(model.update(8, close_away).label, "H")
            for i in range(9, 54):
                detect.return_value = away if i == 9 else None
                self.assertEqual(model.update(i, close_away).label, "H")
            detect.return_value = None
            result = model.update(54, close_away)
        self.assertEqual((result.label, result.change_frame), ("A", 54))

    def test_smart_tag_conflict_delays_confirmation_but_does_not_block_it(self):
        home, away = Player("home"), Player("away")
        position = Point(0.5, 0.5)
        frame = SimpleNamespace(ball_coordinates=position,
                                players_coordinates={home: position, away: Point(0.6, 0.5)})
        model = PossessionModel("home")
        with patch.object(model.detector, "update", side_effect=[home] + [None] * 9):
            for i in range(9):
                hint = "A" if i == 4 else None
                self.assertEqual(model.update(i, frame, hint).label, "A")
            result = model.update(9, frame)
        self.assertEqual((result.label, result.change_frame), ("H", 9))

    def test_pending_change_is_cancelled_when_opponent_leaves_or_ball_is_missing(self):
        home, away = Player("home"), Player("away")
        ball = Point(0.5, 0.5)
        close = SimpleNamespace(ball_coordinates=ball,
                                players_coordinates={home: ball, away: Point(0.6, 0.5)})
        far = SimpleNamespace(ball_coordinates=ball,
                              players_coordinates={home: Point(0.6, 0.5), away: Point(0.7, 0.5)})
        for interruption in (far, None):
            model = PossessionModel("home")
            with patch.object(model.detector, "update", side_effect=[home] + [None] * 9):
                for i in range(4):
                    self.assertEqual(model.update(i, close).label, "A")
                self.assertEqual(model.update(4, interruption).label, "A")
                for i in range(5, 10):
                    self.assertEqual(model.update(i, close).label, "A")

    def test_change_does_not_require_a_more_distant_incumbent(self):
        home, away = Player("home"), Player("away")
        ball = Point(50 / 105, 0.5)
        challenger = Point(51 / 105, 0.5)
        for incumbent in (None, Point(50.5 / 105, 0.5), challenger, Point(float("nan"), 0.5)):
            players = {home: challenger}
            if incumbent is not None:
                players[away] = incumbent
            frame = SimpleNamespace(ball_coordinates=ball, players_coordinates=players)
            model = PossessionModel("home")
            with self.subTest(incumbent=incumbent), \
                    patch.object(model.detector, "update", side_effect=[home] + [None] * 4):
                for i in range(4):
                    self.assertEqual(model.update(i, frame).label, "A")
                self.assertEqual(model.update(4, frame).label, "H")

    def test_current_team_becoming_closer_does_not_cancel_pending_change(self):
        home, away = Player("home"), Player("away")
        ball = Point(50 / 105, 0.5)
        challenger = Point(51 / 105, 0.5)
        close = SimpleNamespace(ball_coordinates=ball,
                                players_coordinates={home: challenger, away: Point(54 / 105, 0.5)})
        interrupted = SimpleNamespace(ball_coordinates=ball,
                                      players_coordinates={home: challenger,
                                                           away: Point(50.5 / 105, 0.5)})
        model = PossessionModel("home")
        with patch.object(model.detector, "update", side_effect=[home] + [None] * 4):
            for i in range(4):
                self.assertEqual(model.update(i, close).label, "A")
            self.assertEqual(model.update(4, interrupted).label, "H")

    def test_change_requires_challenger_within_three_metres(self):
        home, away = Player("home"), Player("away")
        ball = Point(50 / 105, 0.5)
        for distance, expected in ((2.99, "H"), (3.01, "A")):
            frame = SimpleNamespace(
                ball_coordinates=ball,
                players_coordinates={home: Point((50 + distance) / 105, 0.5),
                                     away: Point(55 / 105, 0.5)})
            model = PossessionModel("home")
            with self.subTest(distance=distance), \
                    patch.object(model.detector, "update", side_effect=[home] + [None] * 4):
                for i in range(5):
                    result = model.update(i, frame)
            self.assertEqual(result.label, expected)

    def test_smart_attacking_hints_ignore_unknown_conflicting_and_non_attacking_tags(self):
        events = [
            {"code": "PROGRESSION", "start_frame": 1, "end_frame": 5,
             "tags": {"Team": "Barcelona"}},
            {"code": "BUILD UP", "start_frame": 3, "end_frame": 6,
             "tags": {"Team": "Atletic Club"}},
            {"code": "SET PIECES", "start_frame": 0, "end_frame": 7,
             "tags": {"Team": "Barcelona"}},
            {"code": "PROGRESSION", "start_frame": 0, "end_frame": 7,
             "tags": {"Team": "N/A"}},
        ]
        self.assertEqual(smart_attacking_hints(events, 0, 6, "Barcelona", "Atletic Club"),
                         {1: "H", 2: "H", 5: "A"})

    def test_smart_tagging_rejects_mismatched_time_base(self):
        with patch.object(detection_input.load_smart_tagging, "load", return_value=[]), \
                patch.object(detection_input.load_smart_tagging, "check_time_base",
                             return_value={"agrees": False}):
            with self.assertRaisesRegex(ValueError, "timestamps"):
                detection_input.load_smart_attacking_hints(0, 10, "Barcelona", "Atletic Club")

    def test_stationary_ball_becomes_dead_and_resumes_last_possession_after_moving(self):
        home, away = Player("home"), Player("away")

        def frame(x, player):
            position = Point(x / 105, 0.5)
            other = away if player == home else home
            return SimpleNamespace(ball_coordinates=position,
                                   players_coordinates={player: position,
                                                        other: Point((x + 5) / 105, 0.5)})

        model = PossessionModel("home")
        for i, x in enumerate((20, 21, 22.3)):
            model.update(i, frame(x, home))
        for i in range(3, 6):
            self.assertEqual(model.update(i, frame(22.3, home)).label, "A")
        for i in range(6, 51):
            self.assertEqual(model.update(i, frame(22.3, home)).label, "H")
        result = model.update(51, frame(22.3, away))
        self.assertEqual((result.label, result.change_frame, result.touch_player), ("D", 51, None))
        self.assertEqual(model.update(52, None).label, "D")
        self.assertEqual(model.update(53, frame(22.5, away)).label, "D")
        result = model.update(54, frame(23.4, away))
        self.assertEqual((result.label, result.change_frame), ("H", 54))
        model.update(55, frame(24.4, away))
        for i in range(56, 60):
            self.assertEqual(model.update(i, frame(i - 30.3, away)).label, "H")
        self.assertEqual(model.update(60, frame(29.7, away)).label, "A")

    def test_stationary_window_enforces_diameter_and_resets_on_long_missing_or_gap(self):
        def frame(x):
            return SimpleNamespace(ball_coordinates=Point(x / 105, 0.5), players_coordinates={})

        model = PossessionModel("home")
        for i in range(25):
            model.update(i, frame(50))
        for i in range(25, 49):
            model.update(i, frame(49.1))
        self.assertEqual(model.update(49, frame(50.9)).label, "A")
        for i in range(50, 98):
            self.assertEqual(model.update(i, frame(50.9)).label, "A")
        self.assertEqual(model.update(98, frame(50.9)).label, "D")

        model = PossessionModel("home")
        for i in range(49):
            model.update(i, frame(50))
        for i in range(49, 53):
            self.assertEqual(model.update(i, None).label, "A")
        for i in range(53, 102):
            self.assertEqual(model.update(i, frame(50)).label, "A")
        self.assertEqual(model.update(103, frame(50)).label, "A")
        for i in range(104, 152):
            self.assertEqual(model.update(i, frame(50)).label, "A")
        self.assertEqual(model.update(152, frame(50)).label, "D")

    def test_stationary_detection_tolerates_three_missing_frames_but_does_not_count_them(self):
        frame = SimpleNamespace(ball_coordinates=Point(0.5, 0.5), players_coordinates={})
        model = PossessionModel("home")
        for i in range(49):
            self.assertEqual(model.update(i, frame).label, "A")
        for i in range(49, 52):
            self.assertEqual(model.update(i, None).label, "A")
        self.assertEqual(model.update(52, frame).label, "D")

        model = PossessionModel("home")
        for i in range(49):
            model.update(i, frame)
        model.update(49, None)
        moved = SimpleNamespace(ball_coordinates=Point(0.52, 0.5), players_coordinates={})
        self.assertEqual(model.update(50, moved).label, "A")

    def test_outside_pitch_requires_three_consecutive_frames_beyond_margin(self):
        def frame(x, y=34):
            return SimpleNamespace(ball_coordinates=Point(x / 105, y / 68),
                                   players_coordinates={})

        model = PossessionModel("home")
        self.assertEqual(model.update(0, frame(-1.1)).label, "A")
        self.assertEqual(model.update(1, frame(-1.1)).label, "A")
        self.assertEqual(model.update(2, frame(-1.0)).label, "A")
        self.assertEqual(model.update(3, frame(-1.1)).label, "A")
        self.assertEqual(model.update(4, None).label, "A")
        self.assertEqual(model.update(5, frame(-1.1)).label, "A")
        self.assertEqual(model.update(6, frame(-1.1)).label, "A")
        self.assertEqual(model.update(7, frame(-1.1)).label, "D")
        self.assertEqual(model.update(8, frame(-0.2)).label, "D")
        self.assertEqual(model.update(9, frame(0.1)).label, "A")

    def test_slow_players_detect_dead_ball_even_without_ball_and_resume_when_two_run(self):
        players = [MovingPlayer("home", i) for i in range(18)]
        offsets = [0.0] * len(players)

        def frame(fast_players=(), available=18):
            for i in fast_players:
                offsets[i] += 9 / 3.6 / 25
            return SimpleNamespace(
                ball_coordinates=None,
                players_coordinates={
                    player: Point((10 + i * 2 + offsets[i]) / 105, 0.5)
                    for i, player in enumerate(players[:available])
                })

        model = PossessionModel("home")
        for i in range(25):
            self.assertEqual(model.update(i, frame((0,))).label, "A")
        result = model.update(25, frame((0,)))
        self.assertEqual((result.label, result.change_frame), ("D", 25))
        self.assertEqual(model.update(26, frame((0,), available=17)).label, "D")
        self.assertEqual(model.update(27, frame((0, 1))).label, "D")
        result = model.update(28, frame((0, 1)))
        self.assertEqual((result.label, result.change_frame), ("A", 28))

    def test_slow_players_require_enough_tracking_and_consecutive_frames(self):
        players = [MovingPlayer("home", i) for i in range(18)]

        def frame(count=18):
            return SimpleNamespace(
                ball_coordinates=None,
                players_coordinates={p: Point((10 + i * 2) / 105, 0.5)
                                     for i, p in enumerate(players[:count])})

        model = PossessionModel("home")
        for i in range(30):
            self.assertEqual(model.update(i, frame(17)).label, "A")
        for i in range(30, 55):
            self.assertEqual(model.update(i, frame()).label, "A")
        self.assertEqual(model.update(56, frame()).label, "A")
        for i in range(57, 81):
            self.assertEqual(model.update(i, frame()).label, "A")
        self.assertEqual(model.update(81, frame()).label, "D")

    def test_two_fast_players_reset_slow_player_confirmation(self):
        players = [MovingPlayer("home", i) for i in range(18)]
        offsets = [0.0] * len(players)

        def frame(fast=()):
            for i in fast:
                offsets[i] += 9 / 3.6 / 25
            return SimpleNamespace(
                ball_coordinates=None,
                players_coordinates={p: Point((10 + i * 2 + offsets[i]) / 105, 0.5)
                                     for i, p in enumerate(players)})

        model = PossessionModel("home")
        for i in range(25):
            self.assertEqual(model.update(i, frame()).label, "A")
        self.assertEqual(model.update(25, frame((0, 1))).label, "A")
        for i in range(26, 50):
            self.assertEqual(model.update(i, frame()).label, "A")
        self.assertEqual(model.update(50, frame()).label, "D")

    def test_export_uses_tracked_positions_and_ignores_ffill(self):
        home, away = Player("home"), Player("away")
        frames = [SimpleNamespace(frame_id=i + 1, ball_coordinates=Point(0.9, 0.5),
                                  players_coordinates={home: Point(0.23, 0.5),
                                                       away: Point(0.8, 0.5)})
                  for i in range(15)]
        dataset = SimpleNamespace(frames=frames, metadata=SimpleNamespace(teams=["home"]))
        with tempfile.TemporaryDirectory() as directory:
            track = os.path.join(directory, "track.csv")
            with open(track, "w", newline="") as handle:
                handle.write("frame,x,y,source\n1,0.2,0.5,yolo_sort\n2,0.21,0.5,yolo_sort\n")
                for i in range(3, 13):
                    handle.write("{},0.23,0.5,yolo_sort\n".format(i))
                handle.write("13,0.9,0.5,ffill\n")
            argv = ["predict_possession.py", "--start", "1", "--end", "13",
                    "--tracked-ball-csv", track, "--out", directory]
            with patch.object(sys, "argv", argv), \
                    patch.object(predict_possession.games, "GAMES", {"BAR-ATH": {"n_video_frames": 15}}), \
                    patch.object(predict_possession.games, "require"), \
                    patch.object(predict_possession.metrica, "load_tracking_epts", return_value=dataset), \
                    patch.object(predict_possession.load_homography, "load",
                                 return_value=([np.eye(3)] * 15, None, [True] * 15)):
                predict_possession.main()
            with open(os.path.join(directory, "touch_possession.csv"), newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual([row["frame"] for row in rows], [str(i) for i in range(15)])
        self.assertEqual([row["label"] for row in rows], ["X"] + ["A"] * 6 + ["H"] * 7 + ["X"])

    def test_export_dead_ball_ignores_forward_filled_observation(self):
        frames = [SimpleNamespace(frame_id=i + 1, ball_coordinates=Point(0.5, 0.5),
                                  players_coordinates={}) for i in range(72)]
        dataset = SimpleNamespace(frames=frames, metadata=SimpleNamespace(teams=["home"]))
        with tempfile.TemporaryDirectory() as directory:
            track = os.path.join(directory, "track.csv")
            with open(track, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["frame", "x", "y", "source"])
                for i in range(72):
                    writer.writerow([i, 0.52 if i == 71 else 0.5, 0.5,
                                     "ffill" if i == 20 else "yolo_sort"])
            argv = ["predict_possession.py", "--end", "71",
                    "--tracked-ball-csv", track, "--out", directory]
            with patch.object(sys, "argv", argv), \
                    patch.object(predict_possession.games, "GAMES", {"BAR-ATH": {"n_video_frames": 72}}), \
                    patch.object(predict_possession.games, "require"), \
                    patch.object(predict_possession.metrica, "load_tracking_epts", return_value=dataset), \
                    patch.object(predict_possession.load_homography, "load",
                                 return_value=([np.eye(3)] * 72, None, [True] * 72)):
                predict_possession.main()
            with open(os.path.join(directory, "touch_possession.csv"), newline="") as handle:
                labels = [row["label"] for row in csv.DictReader(handle)]
        self.assertEqual(labels, ["A"] * 50 + ["D"] * 21 + ["A"])

    def test_export_applies_smart_tag_confirmation(self):
        home, away = Team("home"), Team("away")
        player, opponent = Player(home), Player(away)
        frames = [SimpleNamespace(frame_id=i + 1, ball_coordinates=Point(0.9, 0.5),
                                  players_coordinates={player: Point(0.23, 0.5),
                                                       opponent: Point(0.8, 0.5)})
                  for i in range(17)]
        dataset = SimpleNamespace(frames=frames, metadata=SimpleNamespace(teams=[home, away]))
        events = [{"code": "PROGRESSION", "start_frame": 0, "end_frame": 17,
                   "tags": {"Team": "away"}}]
        with tempfile.TemporaryDirectory() as directory:
            track = os.path.join(directory, "track.csv")
            with open(track, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["frame", "x", "y", "source"])
                for i in range(17):
                    writer.writerow([i, [0.2, 0.21, 0.23][min(i, 2)], 0.5, "yolo_sort"])
            argv = ["predict_possession.py", "--end", "16", "--smart-tagging",
                    "--tracked-ball-csv", track, "--out", directory]
            with patch.object(sys, "argv", argv), \
                    patch.object(predict_possession.games, "GAMES", {"BAR-ATH": {"n_video_frames": 17}}), \
                    patch.object(predict_possession.games, "require"), \
                    patch.object(predict_possession.metrica, "load_tracking_epts", return_value=dataset), \
                    patch.object(detection_input.load_smart_tagging, "load", return_value=events), \
                    patch.object(detection_input.load_smart_tagging, "check_time_base",
                                 return_value={"agrees": True}), \
                    patch.object(predict_possession.load_homography, "load",
                                 return_value=([np.eye(3)] * 17, None, [True] * 17)):
                predict_possession.main()
            with open(os.path.join(directory, "touch_possession.csv"), newline="") as handle:
                labels = [row["label"] for row in csv.DictReader(handle)]
        self.assertEqual(labels, ["A"] * 11 + ["H"] * 6)

    def test_visualizer_displays_initial_away_then_model_changes(self):
        home, away = Player("home"), Player("away")
        positions = [0, 1, 2.3] + [i + 0.3 for i in range(3, 12)] + [i + 0.6 for i in range(12, 62)]
        players = [home] * 12 + [away] * 50
        frames = {
            i + 1: SimpleNamespace(
                ball_coordinates=Point(0.9, 0.5),
                players_coordinates={player: Point(x / 105, 0.5),
                                     (away if player == home else home): Point((x + 5) / 105, 0.5)})
            for i, (x, player) in enumerate(zip(positions, players))
        }
        track = {i: {"centre": (x / 105, 0.5)} for i, x in enumerate(positions)}

        class Capture:
            def get(self, prop):
                return 25 if prop == visualizer.cv2.CAP_PROP_FPS else 100

            def set(self, prop, value):
                pass

            def read(self):
                return True, np.zeros((100, 100, 3), dtype=np.uint8)

        with patch.object(visualizer.cv2, "VideoWriter"), \
                patch.object(visualizer, "annotate", return_value=(None, None, {})), \
                patch.object(visualizer, "draw_possession") as draw:
            visualizer.render_clip(
                Capture(), 0, 61, frames, [np.eye(3)] * 62, [True] * 62, "home",
                "unused.mp4", False, 1.0, possession_ball_source="tracked",
                possession_ball_track=track)
        self.assertEqual([call.args[1] for call in draw.call_args_list], ["A"] * 6 + ["H"] * 50 + ["A"] * 6)

    def test_visualizer_displays_dead_ball_prediction(self):
        frame = SimpleNamespace(ball_coordinates=Point(0.5, 0.5), players_coordinates={})

        class Capture:
            def get(self, prop):
                return 25 if prop == visualizer.cv2.CAP_PROP_FPS else 100

            def set(self, prop, value):
                pass

            def read(self):
                return True, np.zeros((100, 100, 3), dtype=np.uint8)

        with patch.object(visualizer.cv2, "VideoWriter"), \
                patch.object(visualizer, "annotate", return_value=(None, None, {})), \
                patch.object(visualizer, "draw_possession") as draw:
            visualizer.render_clip(
                Capture(), 0, 49, {i + 1: frame for i in range(50)},
                [np.eye(3)] * 50, [True] * 50, "home", "unused.mp4", False, 1.0)
        self.assertEqual([call.args[1] for call in draw.call_args_list], ["A"] * 49 + ["D"])

    def test_visualizer_displays_player_motion_dead_ball_without_ball_observations(self):
        players = [MovingPlayer("home", i) for i in range(18)]
        frame = SimpleNamespace(
            ball_coordinates=None,
            players_coordinates={p: Point((10 + i * 2) / 105, 0.5)
                                 for i, p in enumerate(players)})

        class Capture:
            def get(self, prop):
                return 25 if prop == visualizer.cv2.CAP_PROP_FPS else 100

            def set(self, prop, value):
                pass

            def read(self):
                return True, np.zeros((100, 100, 3), dtype=np.uint8)

        with patch.object(visualizer.cv2, "VideoWriter"), \
                patch.object(visualizer, "annotate", return_value=(None, None, {})), \
                patch.object(visualizer, "draw_possession") as draw:
            visualizer.render_clip(
                Capture(), 0, 25, {i + 1: frame for i in range(26)},
                [np.eye(3)] * 26, [True] * 26, "home", "unused.mp4", False, 1.0,
                possession_ball_source="tracked", possession_ball_track={})
        self.assertEqual([call.args[1] for call in draw.call_args_list], ["A"] * 25 + ["D"])

    def test_visualizer_applies_smart_tag_confirmation(self):
        home, away = Player("home"), Player("away")
        positions = [0, 1, 2.3] + [2.3] * 14
        frames = {
            i + 1: SimpleNamespace(ball_coordinates=Point(0.9, 0.5),
                                   players_coordinates={home: Point(x / 105, 0.5),
                                                        away: Point((x + 5) / 105, 0.5)})
            for i, x in enumerate(positions)
        }
        track = {i: {"centre": (x / 105, 0.5)} for i, x in enumerate(positions)}

        class Capture:
            def get(self, prop):
                return 25 if prop == visualizer.cv2.CAP_PROP_FPS else 100

            def set(self, prop, value):
                pass

            def read(self):
                return True, np.zeros((100, 100, 3), dtype=np.uint8)

        with patch.object(visualizer.cv2, "VideoWriter"), \
                patch.object(visualizer, "annotate", return_value=(None, None, {})), \
                patch.object(visualizer, "draw_possession") as draw:
            visualizer.render_clip(
                Capture(), 0, 16, frames, [np.eye(3)] * 17, [True] * 17, "home",
                "unused.mp4", False, 1.0, possession_ball_source="tracked",
                possession_ball_track=track, smart_hints={6: "A"})
        self.assertEqual([call.args[1] for call in draw.call_args_list], ["A"] * 11 + ["H"] * 6)

if __name__ == "__main__":
    unittest.main()
