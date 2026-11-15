"""Replace teleporting ball positions in a merged ball-tracking CSV with the last trusted point.

Reads a merged CSV (frame,x,y,source,matched,score,bbox_x1,bbox_y1,bbox_x2,bbox_y2 -- the output
of merge_sort_atd_ball.py / assets/detections/tracked_ball.csv) and, for any frame whose (x, y)
implies a frame-to-frame speed above --max-speed pixels/frame compared to the last accepted
point (see detections/teleport_filter.filter_teleports()), overwrites x/y with that last accepted
position. The row's other columns (source, matched, score, bbox) are left as they were, except a
"teleport_corrected" column is added/overwritten recording True/False, so corrected rows stay
identifiable without losing their original source label.

    python scripts/ball_tracking/filter_ball_teleports.py assets/detections/tracked_ball.csv
"""

import argparse
import csv
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from detections.teleport_filter import DEFAULT_MAX_SPEED, filter_teleports  # noqa: E402

BASE_FIELDS = ["frame", "x", "y", "source", "matched", "score",
              "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
EXTRA_FIELD = "teleport_corrected"


def filter_csv(path, max_speed=DEFAULT_MAX_SPEED):
    """[row, ...] (dicts, BASE_FIELDS + EXTRA_FIELD) with teleporting points overwritten with
    the last accepted position; (rows, corrected_count) returned alongside.
    """
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    points = [(int(float(row["frame"])),
              float(row["x"]) if row["x"] not in (None, "") else None,
              float(row["y"]) if row["y"] not in (None, "") else None)
             for row in rows]
    corrected = filter_teleports(points, max_speed)

    count = 0
    for row, (_, x, y, is_corrected) in zip(rows, corrected):
        row[EXTRA_FIELD] = str(is_corrected)
        if is_corrected:
            row["x"], row["y"] = x, y
            # The original bbox/score belonged to the rejected (teleporting) detection, not to
            # the replacement position, so clear them rather than leave a mismatched box.
            for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"):
                row[key] = ""
            count += 1
    return rows, count


def write_csv(rows, path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASE_FIELDS + [EXTRA_FIELD])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="merged ball-tracking CSV to filter, overwritten in place "
                                     "(a .bak is kept)")
    parser.add_argument("--max-speed", type=float, default=DEFAULT_MAX_SPEED,
                        help="max plausible ball speed in pixels/frame; a jump from the last "
                             "accepted point faster than this (scaled by the frame gap) is "
                             "treated as a teleport (default %.0f)" % DEFAULT_MAX_SPEED)
    parser.add_argument("--out", help="write here instead of overwriting --path in place")
    args = parser.parse_args()

    rows, corrected = filter_csv(args.path, args.max_speed)
    print("corrected %d/%d teleporting ball points (max-speed=%.0f px/frame)"
         % (corrected, len(rows), args.max_speed))

    out_path = args.out or args.path
    if out_path == args.path:
        backup_path = args.path + ".bak"
        if not os.path.exists(backup_path):
            shutil.copy2(args.path, backup_path)
        print("original kept at %s" % backup_path)

    write_csv(rows, out_path)
    print("wrote %d rows to %s" % (len(rows), out_path))


if __name__ == "__main__":
    main()
