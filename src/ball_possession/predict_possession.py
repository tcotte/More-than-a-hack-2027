"""Export possession predictions based on the tracked-ball CSV.

A nearby opposing team gains possession after a probable touch and five consecutive frames
near the ball. An opposing Smart Tagging hint adds five confirmation frames. Team changes are
at least two seconds apart; dead-ball transitions are not delayed. Possession starts with Away;
outside the requested interval the label is X.
The ball is dead after two seconds within a one-metre diameter or three frames more than one
metre outside the pitch. Up to three missing frames are tolerated during stationary detection,
but forward-filled positions are not counted as observations. Independently, the ball is dead
after one second with at least 18 tracked players and at most one moving at 8 km/h or more;
if this signal caused the dead ball, play resumes when at least two players reach that speed.
"""

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
sys.path.insert(0, os.path.join(ROOT, "mtah_challenge"))
sys.path.insert(0, SRC)

from kloppy import metrica  # noqa: E402

import games  # noqa: E402
from ball_possession.detection_input import (  # noqa: E402
    load_possession_ball_track, load_smart_attacking_hints, possession_frame,
)
from ball_possession.possession_model import PossessionModel  # noqa: E402
from starter import load_homography  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0, help="first video frame to predict")
    parser.add_argument("--end", type=int, help="last video frame to predict (inclusive)")
    parser.add_argument("--tracked-ball-csv", default=os.path.join(
        ROOT, "assets", "trackings", "ball", "tracked_ball.csv"))
    parser.add_argument("--smart-tagging", action="store_true",
                        help="use tactical Smart Tagging as an additional team hint")
    parser.add_argument("--name", default="touch_possession", help="file name, without .csv")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    args = parser.parse_args()

    n_frames = games.GAMES[games.GAME]["n_video_frames"]
    end = args.end if args.end is not None else n_frames - 1
    if not 0 <= args.start <= end < n_frames:
        parser.error("--start and --end must define an interval within the tactical video")
    if not os.path.isfile(args.tracked_ball_csv):
        parser.error("tracked ball file not found: %s" % args.tracked_ball_csv)

    games.require(["atd_meta", "atd_raw", "homography"], source="tactical")
    paths = games.asset_paths(source="tactical")
    print("loading ATD frames 0..%d..." % end)
    dataset = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"],
                                         limit=end + 2)
    by_frame_id = {frame.frame_id: frame for frame in dataset.frames}
    print("loading tracked ball positions from %s..." % args.tracked_ball_csv)
    ball_track = load_possession_ball_track(args.tracked_ball_csv, args.start, end)
    print("using %d tracked positions; ffill and missing rows are ignored" % len(ball_track))
    if not ball_track:
        parser.error("no usable tracked ball positions in the selected interval")
    matrices, _, valid = load_homography.load(source="tactical")
    smart_hints = {}
    if args.smart_tagging:
        smart_hints = load_smart_attacking_hints(args.start, end,
                                                dataset.metadata.teams[0].name,
                                                dataset.metadata.teams[1].name)
        print("using Smart Tagging team hints on %d frames" % len(smart_hints))

    model = PossessionModel(dataset.metadata.teams[0])
    labels = ["X"] * n_frames
    touches = 0
    for frame_index in range(args.start, end + 1):
        matrix = matrices[frame_index] if valid[frame_index] else None
        frame = possession_frame(by_frame_id.get(frame_index + 1), ball_track, frame_index, matrix)
        result = model.update(frame_index, frame, smart_hints.get(frame_index))
        labels[frame_index] = result.label
        if result.touch_player is not None:
            touches += 1
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, args.name + ".csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "label"])
        writer.writerows(enumerate(labels))
    print("detected %d probable touches" % touches)
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
