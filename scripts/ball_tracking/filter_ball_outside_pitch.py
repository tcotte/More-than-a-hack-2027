"""Drop ball points that fall outside the pitch, using the per-frame tactical homography.

Filters two kinds of CSV in place (a .bak of the original is written alongside):

  - a raw detections CSV (frame,x1,y1,x2,y2,score,class_id), e.g.
    assets/detections/ball_detection_first_half_yolo26_tactical_sliced.csv -- rows whose bbox
    centre unprojects off the pitch are dropped entirely.

  - a merged ball-tracking CSV (frame,x,y,source,matched,score,bbox_x1,bbox_y1,bbox_x2,bbox_y2),
    e.g. assets/detections/tracked_ball.csv -- rows whose (x, y) unprojects off the pitch are
    blanked (x/y/source/matched/score/bbox set empty, source becomes "missing") rather than
    removed, so every video frame keeps its row and downstream readers that forward-fill or
    expect one row per frame keep working; see visualize_video_frames.load_tracked_ball_points().

    python scripts/ball_tracking/filter_ball_outside_pitch.py --kind detections PATH
    python scripts/ball_tracking/filter_ball_outside_pitch.py --kind merged PATH
"""

import argparse
import csv
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from detections.pitch_filter import load_pitch_checker  # noqa: E402

DETECTIONS_FIELDS = ["frame", "x1", "y1", "x2", "y2", "score", "class_id"]
MERGED_FIELDS = ["frame", "x", "y", "source", "matched", "score",
                 "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]


def filter_detections(path, is_on_pitch):
    """[row, ...] (dicts, DETECTIONS_FIELDS) with off-pitch rows dropped; (kept, dropped)
    counts alongside.
    """
    kept, dropped = [], 0
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame = int(float(row["frame"]))
            cx = (float(row["x1"]) + float(row["x2"])) / 2
            cy = (float(row["y1"]) + float(row["y2"])) / 2
            if is_on_pitch(frame, cx, cy):
                kept.append(row)
            else:
                dropped += 1
    return kept, dropped


def filter_merged(path, is_on_pitch):
    """[row, ...] (dicts, MERGED_FIELDS) with off-pitch rows blanked out (kept, not removed);
    (kept_unchanged, blanked) counts alongside.
    """
    rows, blanked = [], 0
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame = int(float(row["frame"]))
            if row["x"] not in (None, "") and row["y"] not in (None, ""):
                if not is_on_pitch(frame, float(row["x"]), float(row["y"])):
                    for key in ("x", "y", "matched", "score",
                               "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"):
                        row[key] = ""
                    row["source"] = "missing"
                    blanked += 1
            rows.append(row)
    return rows, blanked


def write_csv(rows, fields, path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="CSV to filter, overwritten in place (a .bak is kept)")
    parser.add_argument("--kind", required=True, choices=("detections", "merged"),
                        help="\"detections\" (frame,x1,y1,x2,y2,...) or \"merged\" "
                             "(frame,x,y,source,...)")
    parser.add_argument("--video-type", default="tactical", choices=("tactical", "broadcast"),
                        help="which recording's homography to check against (default tactical)")
    parser.add_argument("--margin", type=float, default=0.05,
                        help="fraction of the pitch the cutoff is loosened by on every side, so "
                             "near-boundary plays (throw-ins, corners) are not dropped along "
                             "with genuine false positives (default 0.05)")
    parser.add_argument("--out", help="write here instead of overwriting --path in place")
    args = parser.parse_args()

    is_on_pitch = load_pitch_checker(source=args.video_type, margin=args.margin)

    if args.kind == "detections":
        rows, dropped = filter_detections(args.path, is_on_pitch)
        print("dropped %d/%d off-pitch detections" % (dropped, dropped + len(rows)))
        fields = DETECTIONS_FIELDS
    else:
        rows, blanked = filter_merged(args.path, is_on_pitch)
        print("blanked %d/%d off-pitch tracked points" % (blanked, len(rows)))
        fields = MERGED_FIELDS

    out_path = args.out or args.path
    if out_path == args.path:
        backup_path = args.path + ".bak"
        if not os.path.exists(backup_path):
            shutil.copy2(args.path, backup_path)
        print("original kept at %s" % backup_path)

    write_csv(rows, fields, out_path)
    print("wrote %d rows to %s" % (len(rows), out_path))


if __name__ == "__main__":
    main()
