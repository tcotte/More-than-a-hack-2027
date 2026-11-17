import os
import sys
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from kloppy.domain import Point

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import visualize_video_frames as visualizer
from ball_possession.possession_model import PossessionModel


@dataclass(frozen=True)
class Player:
    team: str
    jersey_no: int = 1


class PossessionDetectionTests(unittest.TestCase):
    def setUp(self):
        self.home = Player("home")
        self.away = Player("away")
        self.frame = SimpleNamespace(
            players_coordinates={self.home: Point(0.2, 0.5), self.away: Point(0.8, 0.5)},
            ball_coordinates=Point(0.2, 0.5),
            frame_id=1,
        )
        self.matrix = np.array([[1000, 0, 0], [0, 500, 0], [0, 0, 1]])
        self.track = {"centre": (800, 250), "bbox": None, "score": None}

    def test_detected_ball_drives_possession_instead_of_atd(self):
        frame = visualizer.frame_with_detected_ball(self.frame, self.track, self.matrix)
        self.assertAlmostEqual(frame.ball_coordinates.x, 0.8)
        self.assertAlmostEqual(frame.ball_coordinates.y, 0.5)
        self.assertEqual(PossessionModel("home").update(0, self.frame).label, "H")
        self.assertEqual(PossessionModel("home").update(0, frame).label, "A")
        self.assertEqual(self.frame.ball_coordinates.x, 0.2)

    def test_missing_detection_or_homography_does_not_fall_back_to_atd(self):
        for track, matrix in ((None, self.matrix), (self.track, None)):
            with self.subTest(track=track, matrix=matrix):
                frame = visualizer.frame_with_detected_ball(self.frame, track, matrix)
                self.assertIsNone(frame.ball_coordinates)
                model = PossessionModel("home", dead_ball_frames=2)
                self.assertEqual(model.update(0, frame).label, "X")
                self.assertEqual(model.update(1, frame).label, "D")
        self.assertIsNone(visualizer.frame_with_detected_ball(None, self.track, self.matrix))

    def test_render_uses_selected_ball_source(self):
        class Capture:
            def get(self, prop):
                return 25 if prop == visualizer.cv2.CAP_PROP_FPS else 100

            def set(self, prop, value):
                pass

            def read(self):
                return True, np.zeros((100, 100, 3), dtype=np.uint8)

        for source, expected in (("atd", "H"), ("detections", "A"), ("tracked", "A")):
            with self.subTest(source=source):
                with patch.object(visualizer.cv2, "VideoWriter"), \
                        patch.object(visualizer, "draw_possession") as draw:
                    visualizer.render_clip(
                        Capture(), 0, 0, {1: self.frame}, [self.matrix], [True],
                        "home", "unused.mp4", False, 1.0,
                        ball_track={0: self.track}, possession_ball_source=source,
                        possession_ball_track={0: self.track},
                    )
                self.assertEqual(draw.call_args.args[1], expected)

    def test_annotate_returns_matches_and_team_overrides(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        matches = {self.home: 12}
        overrides = {self.home: "away"}
        with patch.object(visualizer, "match_atd_players_one_to_one",
                          return_value=(matches, overrides)):
            result = visualizer.annotate(
                image, self.frame, self.matrix, "home", 0,
                player_boxes={0: np.zeros((1, 3))}, away_team="away",
                pitch_dimensions=object(),
            )
        self.assertIs(result[0], image)
        self.assertEqual(result[1:], (matches, overrides))


if __name__ == "__main__":
    unittest.main()
