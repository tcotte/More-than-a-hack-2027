"""Merge the YOLO-detection SORT ball track with the ATD ball position as a fallback.

We trust assets/detections/ball_detection_first_half_yolo26_tactical_sliced.csv -- run through a
SORT-style single-target tracker (detections/tracker.track_ball) -- more than Metrica's Automatic
Tracking Data (src/data/tactical/BAR-ATH_tactical_atd.xml/.txt) for where the ball is. So for
every video frame in range:

  - if the SORT tracker produced a point for that frame (a real YOLO detection, or a coasted
    Kalman prediction within --max-age frames of one) AND that point unprojects, via the
    per-frame homography, inside the pitch (see detections/pitch_filter.py), that point is
    used, in the same video-pixel coordinates the detections are in. source = "yolo_sort". A
    SORT point that lands off the pitch (stands, hoardings, a camera operator) is discarded and
    the frame falls through to the ATD fallback below, same as a frame SORT never tracked.
  - otherwise (no YOLO detection recently enough for the tracker to still be coasting), the ATD
    ground-truth ball position for that frame is projected from normalised pitch coordinates into
    pixels via the per-frame homography -- the same projection
    visualize_video_frames.build_ball_references() uses. source = "atd".
  - if neither is available (the ATD also has no ball that frame, or no valid homography),
    x/y are left blank. source = "missing".

The SORT tracker is run with no ATD reference at all, so a gap in it is a genuine gap in the
detections, not a frame where the reference below would have been folded into the match anyway
(see track_ball()'s `references` parameter, which this deliberately does not pass).

    python scripts/ball_tracking/merge_sort_atd_ball.py
    python scripts/ball_tracking/merge_sort_atd_ball.py --frames 11190 15000 --max-age 15
    python scripts/ball_tracking/merge_sort_atd_ball.py --video-type none  # SORT only, no ATD
"""

import argparse
import csv
import os
import sys
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from kloppy import metrica  # noqa: E402

import games  # noqa: E402
import load_homography  # noqa: E402
from ball_possession.possession_model import is_valid  # noqa: E402
from detections.ball import load_ball_detections  # noqa: E402
from detections.pitch_filter import load_pitch_checker  # noqa: E402
from detections.teleport_filter import DEFAULT_MAX_SPEED, filter_teleports  # noqa: E402
from detections.tracker import track_ball  # noqa: E402
DEFAULT_DETECTIONS = os.path.join(REPO_ROOT, "assets", "detections",
                                  "ball_detection_first_half_yolo26_tactical_sliced.csv")
DEFAULT_OUT = os.path.join(REPO_ROOT, "assets", "detections", "ball_merged_sort_atd.csv")


def load_atd_ball_pixels(frames, video_type="tactical"):
    """{video_frame: (px, py)} ATD ground-truth ball position in pixels, for the given frames.

    Only present where the ATD has a tracked ball AND the homography for that frame is valid;
    missing elsewhere, same contract as visualize_video_frames.build_ball_references().
    """
    frames = list(frames)
    if not frames:
        return {}

    paths = games.asset_paths(source=video_type)
    games.require(["atd_meta", "atd_raw", "homography"], source=video_type)

    limit = max(frames) + 2
    print("loading %d ATD frames for the fallback..." % limit)
    dataset = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"],
                                         limit=limit)
    by_frame_id = {frame.frame_id: frame for frame in dataset.frames}

    matrices, _, valid = load_homography.load(source=video_type)

    positions = {}
    for video_frame in frames:
        if video_frame >= len(valid) or not valid[video_frame]:
            continue
        tracking_frame = by_frame_id.get(video_frame + 1)
        if tracking_frame is None:
            continue
        ball = tracking_frame.ball_coordinates
        if not is_valid(ball):
            continue
        px, py = load_homography.project(matrices[video_frame], ball.x, ball.y)
        positions[video_frame] = (float(px), float(py))
    return positions


def merge(detections_path, video_type="tactical", min_score=0.3, gating_distance=80.0,
         max_age=10, frames=None, ffill_missing=False, max_speed=DEFAULT_MAX_SPEED):
    """[(frame, x, y, source, matched, score, bbox, teleport_corrected)], one row per frame in
    `frames` (default: the detections CSV's own min..max frame), SORT-tracked YOLO ball where
    available, ATD ball as fallback, else missing (or, with `ffill_missing`, the last known
    position). `bbox` is (x1, y1, x2, y2) or None; it is only ever set for a real YOLO
    detection, never for a coasted prediction, an ATD fallback or a forward-filled frame.
    `teleport_corrected` is True where the point implied a frame-to-frame speed above
    `max_speed` pixels/frame and was replaced by the last accepted position instead (see
    detections/teleport_filter.py); `source` still names where the (now overwritten) point
    would otherwise have come from.

    `video_type=None` means no homography/ATD is used at all: the off-pitch check (which needs
    the homography) and the ATD fallback (which needs both the ATD files and the homography) are
    both skipped, so a frame SORT has no point for is simply "missing" (or forward-filled), never
    "atd". Use this when the ATD/homography files are not available, or the fallback is not
    wanted.
    """
    print("loading ball detections from %s..." % detections_path)
    detections = load_ball_detections(detections_path)

    if frames is None:
        frames = range(min(detections), max(detections) + 1)

    print("SORT-tracking the ball over %d frames (no ATD reference -- detections only)..."
         % len(frames))
    # No `references`: the SORT tracker must stand on the YOLO detections alone, so a gap in it
    # is a real gap, for the ATD fallback below to fill.
    sort_track = track_ball(detections, frames, references=None, min_score=min_score,
                            gating_distance=gating_distance, max_age=max_age)

    if video_type is not None:
        # Drop SORT points the homography says are off the pitch (stands, hoardings, a camera
        # operator) -- they are not plausible ball positions, however the tracker scored them.
        # The frame falls through to the ATD fallback below, same as a frame the tracker never
        # had.
        is_on_pitch = load_pitch_checker(source=video_type)
        off_pitch = [frame for frame, point in sort_track.items()
                    if not is_on_pitch(frame, *point["centre"])]
        for frame in off_pitch:
            del sort_track[frame]
        print("%d/%d SORT ball points were off the pitch per the homography; dropped"
             % (len(off_pitch), len(off_pitch) + len(sort_track)))
    else:
        print("video_type=None: skipping the homography off-pitch check")

    missing = [frame for frame in frames if frame not in sort_track]
    if video_type is not None:
        print("%d/%d frames have no SORT ball; filling those from ATD..."
             % (len(missing), len(frames)))
        atd_positions = load_atd_ball_pixels(missing, video_type)
    else:
        print("%d/%d frames have no SORT ball; video_type=None, so no ATD fallback -- left "
             "missing" % (len(missing), len(frames)))
        atd_positions = {}

    rows = []
    last_known = None  # (x, y) of the most recent yolo_sort/atd/ffill row, for ffill_missing.
    for frame in frames:
        point = sort_track.get(frame)
        if point is not None:
            cx, cy = point["centre"]
            rows.append((frame, cx, cy, "yolo_sort", point["matched"], point["score"],
                        point["bbox"]))
            last_known = (cx, cy)
            continue
        atd_point = atd_positions.get(frame)
        if atd_point is not None:
            rows.append((frame, atd_point[0], atd_point[1], "atd", None, None, None))
            last_known = atd_point
        elif ffill_missing and last_known is not None:
            rows.append((frame, last_known[0], last_known[1], "ffill", None, None, None))
        else:
            rows.append((frame, None, None, "missing", None, None, None))

    # Catch ball positions that "teleport" between frames -- a jump too fast for a real ball to
    # have made, regardless of which of the sources above produced it -- and replace them with
    # the last accepted position instead of trusting the outlier. See detections/teleport_filter.
    corrected = filter_teleports([(row[0], row[1], row[2]) for row in rows], max_speed)
    fixed_rows = []
    n_corrected = 0
    for row, (_, x, y, is_corrected) in zip(rows, corrected):
        if is_corrected:
            n_corrected += 1
            # The original bbox belonged to the rejected (teleporting) detection, not to the
            # replacement position, so drop it rather than leave a mismatched box.
            fixed_rows.append((row[0], x, y, row[3], row[4], row[5], None, True))
        else:
            fixed_rows.append(row + (False,))
    print("%d/%d ball points were teleports (> %.0f px/frame); replaced with the last accepted "
         "position" % (n_corrected, len(rows), max_speed))
    return fixed_rows


def write_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "x", "y", "source", "matched", "score",
                         "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "teleport_corrected"])
        for frame, x, y, source, matched, score, bbox, teleport_corrected in rows:
            x1, y1, x2, y2 = bbox if bbox is not None else (None, None, None, None)
            writer.writerow([frame, x, y, source, matched, score, x1, y1, x2, y2,
                            teleport_corrected])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detections", default=DEFAULT_DETECTIONS,
                        help="CSV of YOLO ball detections (frame,x1,y1,x2,y2,score,class_id)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="merged CSV to write")
    parser.add_argument("--video-type", default="tactical",
                        choices=("tactical", "broadcast", "none"),
                        help="which recording's ATD/homography to fall back to (default "
                             "tactical -- matches the default --detections file); \"none\" "
                             "disables the homography off-pitch check and the ATD fallback "
                             "entirely, leaving unmatched frames missing (see merge()'s "
                             "video_type=None)")
    parser.add_argument("--min-detection-score", type=float, default=0.7,
                        help="ignore YOLO detections scoring below this (default 0.7)")
    parser.add_argument("--gating-distance", type=float, default=80.0,
                        help="max pixel distance between a predicted and detected ball position "
                             "to associate them in the SORT tracker (default 80)")
    parser.add_argument("--max-age", type=int, default=10,
                        help="frames the SORT tracker coasts without a matching detection before "
                             "dropping the track and falling back to ATD (default 10)")
    parser.add_argument("--frames", type=int, nargs=2, metavar=("START", "END"),
                        help="video frame range to merge, inclusive (default: the detections "
                             "CSV's own min..max frame)")
    parser.add_argument("--ffill-missing", action="store_true",
                        help="for a frame with neither a SORT ball nor an ATD ball, copy the "
                             "last known ball position forward instead of leaving it blank "
                             "(source=\"ffill\"); does nothing for frames before the first known "
                             "position")
    parser.add_argument("--max-speed", type=float, default=DEFAULT_MAX_SPEED,
                        help="max plausible ball speed in pixels/frame; a point implying a "
                             "faster jump from the last accepted position (scaled by the frame "
                             "gap) is treated as a teleport and replaced by that last accepted "
                             "position (default %.0f)" % DEFAULT_MAX_SPEED)
    args = parser.parse_args()

    video_type = None if args.video_type == "none" else args.video_type
    frames = range(args.frames[0], args.frames[1] + 1) if args.frames else None
    rows = merge(args.detections, video_type, args.min_detection_score,
                args.gating_distance, args.max_age, frames, args.ffill_missing, args.max_speed)

    write_csv(rows, args.out)

    counts = Counter(row[3] for row in rows)
    print("wrote %d rows to %s" % (len(rows), args.out))
    print("  yolo_sort: %d   atd: %d   ffill: %d   missing: %d"
         % (counts.get("yolo_sort", 0), counts.get("atd", 0), counts.get("ffill", 0),
            counts.get("missing", 0)))


if __name__ == "__main__":
    main()
