"""Write a possession submission (frame,label) from the TouchDetector prediction.

    python src/predict_possession.py                        # whole match
    python src/predict_possession.py --end 13990            # only the first frames
    python mtah_challenge/evaluation/run_evaluation.py

On each probable touch the team of the closest player gets the ball. The label is held until the
next touch; X before the first touch. Video frame = ATD frame_id - 1.
"""

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from kloppy import metrica  # noqa: E402

import games  # noqa: E402
from possession_model import PossessionModel  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(HERE), "mtah_challenge", "evaluation", "results")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="touch_detector", help="file name, without .csv")
    parser.add_argument("--end", type=int, help="last video frame to predict (default: all)")
    parser.add_argument("--out", default=RESULTS_DIR)
    args = parser.parse_args()

    n_frames = games.GAMES[games.GAME]["n_video_frames"]
    last = min(args.end, n_frames - 1) if args.end is not None else n_frames - 1

    paths = games.asset_paths(source="tactical")
    print("loading ATD frames 0..%d..." % last)
    dataset = metrica.load_tracking_epts(meta_data=paths["atd_meta"],
                                         raw_data=paths["atd_raw"], limit=last + 2)
    by_frame_id = {frame.frame_id: frame for frame in dataset.frames}
    model = PossessionModel(dataset.metadata.teams[0])
    labels = ["X"] * n_frames
    for video_frame in range(last + 1):
        output = model.update(video_frame, by_frame_id.get(video_frame + 1))
        if output.change_frame is not None and output.change_frame < video_frame:
            # a confirmed change is dated back to the first touch of the new team
            labels[output.change_frame:video_frame] = [output.label] * (
                video_frame - output.change_frame)
        labels[video_frame] = output.label

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, args.name + ".csv")
    pd.DataFrame({"frame": range(n_frames), "label": labels}).to_csv(path, index=False)
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
