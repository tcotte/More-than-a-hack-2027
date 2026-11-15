"""Track player/referee/goalkeeper detections across frames with ultralytics' ByteTrack.

Same CSV input as track_with_norfair.py, but:

  - ball detections (class_id == 0) are dropped before tracking -- this script only tracks
    people on the pitch.
  - detections scoring below --min-score (default 0.5) are dropped.
  - classes 1 (referee), 2 (player) and 3 (goalkeeper) are *not* distinguished: they are treated
    as one undifferentiated class before tracking, and tracked with a single
    ultralytics.trackers.byte_tracker.BYTETracker (the implementation behind
    `model.track(tracker="bytetrack.yaml")`) instead of Norfair's IoU tracker.
  - the output CSV has no class_id column at all -- only the track_id identifies a row, since
    the classes have been merged and are no longer meaningful.

    python scripts/players_detection/track_with_bytetrack.py
    python scripts/players_detection/track_with_bytetrack.py --frames 11190 20000 --match-thresh 0.9
"""

import argparse
import csv
import os
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DETECTIONS = os.path.join(REPO_ROOT, "assets", "detections", "players",
                                  "players_tactical_first_half.csv")
DEFAULT_OUT = os.path.join(REPO_ROOT, "assets", "detections", "players",
                           "players_tracked_bytetrack.csv")

BALL_CLASS_ID = 0


def load_detections(path):
    """{frame: [{"bbox": (x1, y1, x2, y2), "score": float, "class_id": int}, ...]}."""
    detections = defaultdict(list)
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            detections[int(float(row["frame"]))].append({
                "bbox": tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2")),
                "score": float(row["score"]),
                "class_id": int(float(row["class_id"])),
            })
    return detections


def track(detections, frames, min_score=0.5, track_high_thresh=0.25, track_low_thresh=0.1,
         new_track_thresh=0.25, track_buffer=30, match_thresh=0.8, fuse_score=True):
    """[(frame, track_id, bbox, score)], one row per detection that BYTETracker actually
    matched to a track on its own frame (no rows for frames a track merely coasts through --
    every row here comes from a real detection).

    tracker.update() is called every frame in `frames`, even those with no surviving detection,
    so track_buffer counts real elapsed frames rather than only frames that had a detection.
    """
    args = SimpleNamespace(track_high_thresh=track_high_thresh, track_low_thresh=track_low_thresh,
                          new_track_thresh=new_track_thresh, track_buffer=track_buffer,
                          match_thresh=match_thresh, fuse_score=fuse_score)
    tracker = BYTETracker(args)

    rows = []
    for frame in sorted(frames):
        dets = [det for det in detections.get(frame, [])
               if det["class_id"] != BALL_CLASS_ID and det["score"] >= min_score]

        if dets:
            # The class column is a dummy 0 -- classes 1/2/3 are merged and BYTETracker itself
            # never looks at it, but Boxes still needs a column there.
            data = np.array([list(det["bbox"]) + [det["score"], 0] for det in dets],
                            dtype=np.float64)
        else:
            data = np.zeros((0, 6), dtype=np.float64)

        online = tracker.update(Boxes(data, orig_shape=(1, 1)))

        for *_, track_id, _score, _cls, idx in online:
            det = dets[int(idx)]
            rows.append([frame, int(track_id), det["bbox"], det["score"]])

    return rows


def write_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "score"])
        for frame, track_id, bbox, score in rows:
            x1, y1, x2, y2 = bbox
            writer.writerow([frame, track_id, x1, y1, x2, y2, score])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detections", default=DEFAULT_DETECTIONS,
                        help="per-frame, multi-class detections CSV "
                             "(frame,x1,y1,x2,y2,score,class_id)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="tracked CSV to write")
    parser.add_argument("--frames", type=int, nargs=2, metavar=("START", "END"),
                        help="video frame range to track, inclusive (default: the detections "
                             "CSV's own min..max frame)")
    parser.add_argument("--min-score", type=float, default=0.5,
                        help="drop detections scoring below this before tracking (default 0.5)")
    parser.add_argument("--track-high-thresh", type=float, default=0.25,
                        help="first-stage (high-score) match threshold, same meaning as "
                             "ultralytics' bytetrack.yaml (default 0.25)")
    parser.add_argument("--track-low-thresh", type=float, default=0.1,
                        help="second-stage (low-score) match threshold (default 0.1)")
    parser.add_argument("--new-track-thresh", type=float, default=0.25,
                        help="minimum score to start a new track (default 0.25)")
    parser.add_argument("--track-buffer", type=int, default=30,
                        help="frames a track survives with no matching detection before it is "
                             "dropped (default 30)")
    parser.add_argument("--match-thresh", type=float, default=0.8,
                        help="IoU association threshold (default 0.8)")
    parser.add_argument("--no-fuse-score", action="store_true",
                        help="disable fusing detection score into the association cost")
    args = parser.parse_args()

    print("loading detections from %s..." % args.detections)
    detections = load_detections(args.detections)

    frames = (range(args.frames[0], args.frames[1] + 1) if args.frames
             else range(min(detections), max(detections) + 1))

    print("tracking %d frames with ultralytics BYTETracker (ball dropped, classes 1/2/3 merged, "
         "score >= %.2f)..." % (len(frames), args.min_score))
    rows = track(detections, frames, args.min_score,
                args.track_high_thresh, args.track_low_thresh, args.new_track_thresh,
                args.track_buffer, args.match_thresh, not args.no_fuse_score)

    write_csv(rows, args.out)

    n_tracks = len({row[1] for row in rows})
    print("wrote %d rows (%d tracks) to %s" % (len(rows), n_tracks, args.out))


if __name__ == "__main__":
    main()
