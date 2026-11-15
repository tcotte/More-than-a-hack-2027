"""Write a frame-by-frame possession submission from ATD or ball detections.

    python src/ball_possession/predict_possession.py
    python src/ball_possession/predict_possession.py --start 11185 --end 18684 --detections CSV
    python mtah_challenge/evaluation/run_evaluation.py

Frames outside the requested interval are X. Video frame = ATD frame_id - 1.
"""

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
sys.path.insert(0, os.path.join(ROOT, "mtah_challenge"))
sys.path.insert(0, SRC)

from kloppy import metrica  # noqa: E402

import games  # noqa: E402
# import load_homography, load_smart_tagging  # noqa: E402
from ball_possession.detection_input import build_ball_references, frame_with_detected_ball  # noqa: E402
from ball_possession.possession_model import PossessionModel  # noqa: E402
from detections.ball import load_ball_detections  # noqa: E402
from detections.tracker import track_ball  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")


def predict_labels(n_frames, start, end, by_frame_id, home_team, ball_track=None,
                   matrices=None, valid=None, set_piece_frames=None, missing_ball_label="D"):
    model = PossessionModel(home_team, use_set_piece_heuristic=set_piece_frames is not None,
                            missing_ball_label=missing_ball_label)
    labels = ["X"] * n_frames
    for video_frame in range(start, end + 1):
        frame = by_frame_id.get(video_frame + 1)
        if ball_track is not None:
            matrix = matrices[video_frame] if valid[video_frame] else None
            frame = frame_with_detected_ball(frame, ball_track.get(video_frame), matrix)
        output = model.update(video_frame, frame,
                              set_piece=set_piece_frames is not None
                              and video_frame in set_piece_frames)
        if output.change_frame is not None and output.change_frame < video_frame:
            labels[output.change_frame:video_frame] = [output.label] * (
                video_frame - output.change_frame)
        labels[video_frame] = output.label
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="touch_detector", help="file name, without .csv")
    parser.add_argument("--start", type=int, default=0, help="first video frame to predict")
    parser.add_argument("--end", type=int, help="last video frame to predict (default: all)")
    parser.add_argument("--detections", help="ball detections CSV; omit to use the ATD ball")
    parser.add_argument("--min-detection-score", type=float, default=0.3)
    parser.add_argument("--gating-distance", type=float, default=80.0)
    parser.add_argument("--max-age", type=int, default=10)
    parser.add_argument("--smart-set-pieces", action="store_true",
                        help="experimental dead-ball heuristic using automatic set-piece tags")
    parser.add_argument("--missing-ball-label", choices=("D", "X"), default="D",
                        help="label after sustained ball tracking loss (default: D); "
                             "X abstains instead of assuming a stoppage")
    parser.add_argument("--out", default=RESULTS_DIR)
    args = parser.parse_args()

    n_frames = games.GAMES[games.GAME]["n_video_frames"]
    last = args.end if args.end is not None else n_frames - 1
    if not 0 <= args.start <= last < n_frames:
        parser.error("--start and --end must define an interval within the tactical video")
    if args.detections and not os.path.isfile(args.detections):
        parser.error("detections file not found: %s" % args.detections)

    paths = games.asset_paths(source="tactical")
    required = ["atd_meta", "atd_raw"]
    if args.detections:
        required.append("homography")
    if args.smart_set_pieces:
        required.append("smart_tagging")
    games.require(required, source="tactical")
    print("loading ATD frames 0..%d..." % last)
    dataset = metrica.load_tracking_epts(meta_data=paths["atd_meta"],
                                         raw_data=paths["atd_raw"], limit=last + 2)
    by_frame_id = {frame.frame_id: frame for frame in dataset.frames}
    ball_track, matrices, valid = None, None, None
    if args.detections:
        print("loading ball detections from %s..." % args.detections)
        detections = load_ball_detections(args.detections)
        if not detections:
            parser.error("detections file contains no detections: %s" % args.detections)
        print("loading homography...")
        matrices, _, valid = load_homography.load(source="tactical")
        frames = range(args.start, last + 1)
        references = build_ball_references(frames, by_frame_id, matrices, valid)
        ball_track = track_ball(detections, frames, references, args.min_detection_score,
                                args.gating_distance, args.max_age)
        print("tracking ball over %d frames (%d ATD references, %d tracked positions)..."
              % (len(frames), len(references), len(ball_track)))
        if not ball_track:
            parser.error("no ball track found for the selected interval")
    set_piece_frames = None
    if args.smart_set_pieces:
        events = load_smart_tagging.load()
        set_pieces = [event for event in events
                      if (event["code"] or "").strip().lower() == "set pieces"]
        if not set_pieces:
            parser.error("Smart Tagging contains no set-piece events")
        set_piece_frames = set()
        for event in set_pieces:
            set_piece_frames.update(range(max(args.start, event["start_frame"]),
                                          min(last + 1, event["end_frame"])))
        print("experimental set-piece heuristic: %d tagged frames in selected interval"
              % len(set_piece_frames))
    labels = predict_labels(n_frames, args.start, last, by_frame_id, dataset.metadata.teams[0],
                            ball_track, matrices, valid, set_piece_frames,
                            args.missing_ball_label)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, args.name + ".csv")
    pd.DataFrame({"frame": range(n_frames), "label": labels}).to_csv(path, index=False)
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
