"""Track every detection (ball, players, ...) across frames with Norfair, and propagate team_id.

assets/detections/players/players_tactical_first_half.csv has one row per detected box per video
frame (class_id in {0, 1, 2, 3}: whichever ball/referee/player/goalkeeper split the detector
uses) but no identity across frames and no team. assets/detections/players/team_distinction.csv
has team_id (0/1) for player boxes (class_id == 2), but only on a handful of sampled frames -- a
separate, more expensive classifier that was not run on every frame.

This script closes both gaps with a single Norfair IoU tracker:

  - class_id is passed as each Detection's `label`, so Norfair never matches a box across
    classes -- a ball detection can never continue a player's track, or vice versa, regardless
    of how close the boxes are.
  - that gives every box a persistent track_id. Wherever team_distinction.csv has a team_id for
    a box that a track passes through, it is recorded as a "vote" for that track. After tracking
    the whole clip, every track's team_id is the majority of the votes it collected, and every
    row belonging to that track (team_id known or not) is written out with it -- so team_id is
    propagated from the few sampled frames to every frame the player was tracked on.

    python scripts/players_detection/track_with_norfair.py
    python scripts/players_detection/track_with_norfair.py --frames 11190 20000 --distance-threshold 0.6
"""

import argparse
import csv
import os
from collections import Counter, defaultdict

import numpy as np
from norfair import Detection, Tracker

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DETECTIONS = os.path.join(REPO_ROOT, "assets", "detections", "players",
                                  "players_tactical_first_half.csv")
DEFAULT_TEAM_IDS = os.path.join(REPO_ROOT, "assets", "detections", "players",
                                "team_distinction.csv")
DEFAULT_OUT = os.path.join(REPO_ROOT, "assets", "detections", "players",
                           "players_tracked_with_teams.csv")

PLAYER_CLASS_ID = 2
BBOX_ROUND = 3  # decimals the (frame, x1, y1, x2, y2) join key is rounded to.


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


def load_team_ids(path):
    """{(frame, x1, y1, x2, y2) rounded: team_id} from a sparse team_id CSV, for an exact join
    against load_detections()'s boxes -- the two files come from the same detector run, so the
    coordinates match to float precision wherever both have a row.
    """
    team_ids = {}
    if not path or not os.path.exists(path):
        return team_ids
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if "team_id" not in row or row["team_id"] in (None, ""):
                continue
            key = (int(float(row["frame"])),) + tuple(
                round(float(row[name]), BBOX_ROUND) for name in ("x1", "y1", "x2", "y2"))
            team_ids[key] = int(float(row["team_id"]))
    return team_ids


def track(detections, team_ids, frames, distance_threshold=0.7, hit_counter_max=15,
         initialization_delay=None, min_score=0.0):
    """[(frame, track_id, class_id, bbox, score, team_id or None)], one row per detection that
    Norfair actually matched to a track on its own frame (no rows for frames a track merely
    coasts through -- every row here comes from a real detection). team_id is the majority of
    `team_ids`' votes collected by that track over the whole run, so it is filled in even on
    frames `team_ids` itself has no entry for.
    """
    # IoU distance is 1 - IoU, in [0, 1]; distance_threshold=0.7 means two boxes are matched as
    # long as they overlap by more than 0.3 IoU.
    tracker = Tracker(distance_function="iou", distance_threshold=distance_threshold,
                      hit_counter_max=hit_counter_max, initialization_delay=initialization_delay)

    rows = []
    team_votes = defaultdict(Counter)
    for frame in sorted(frames):
        norfair_detections = []
        for det in detections.get(frame, []):
            if det["score"] < min_score:
                continue
            x1, y1, x2, y2 = det["bbox"]
            key = (frame,) + tuple(round(value, BBOX_ROUND) for value in det["bbox"])
            data = {
                "frame": frame,
                "class_id": det["class_id"],
                "bbox": det["bbox"],
                "score": det["score"],
                "team_id": team_ids.get(key),
            }
            norfair_detections.append(Detection(
                points=np.array([[x1, y1], [x2, y2]]),
                scores=np.array([det["score"], det["score"]]),
                data=data,
                label=det["class_id"],
            ))

        for tracked_object in tracker.update(detections=norfair_detections):
            data = tracked_object.last_detection.data
            if data is None or data["frame"] != frame:
                continue  # coasting on an older detection -- not a real match this frame.
            rows.append([frame, tracked_object.id, data["class_id"], data["bbox"], data["score"]])
            if data["team_id"] is not None:
                team_votes[tracked_object.id][data["team_id"]] += 1

    majority_team = {track_id: votes.most_common(1)[0][0] for track_id, votes in team_votes.items()}
    return [row + [majority_team.get(row[1]) if row[2] == PLAYER_CLASS_ID else None]
           for row in rows]


def write_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "track_id", "class_id", "x1", "y1", "x2", "y2", "score",
                         "team_id"])
        for frame, track_id, class_id, bbox, score, team_id in rows:
            x1, y1, x2, y2 = bbox
            writer.writerow([frame, track_id, class_id, x1, y1, x2, y2, score, team_id])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detections", default=DEFAULT_DETECTIONS,
                        help="per-frame, multi-class detections CSV "
                             "(frame,x1,y1,x2,y2,score,class_id)")
    parser.add_argument("--team-ids", default=DEFAULT_TEAM_IDS,
                        help="sparse CSV with a team_id column for some player (class_id==2) "
                             "boxes; pass '' to skip team propagation entirely")
    parser.add_argument("--out", default=DEFAULT_OUT, help="tracked CSV to write")
    parser.add_argument("--frames", type=int, nargs=2, metavar=("START", "END"),
                        help="video frame range to track, inclusive (default: the detections "
                             "CSV's own min..max frame)")
    parser.add_argument("--distance-threshold", type=float, default=0.7,
                        help="max IoU distance (1 - IoU) to match a detection to a track "
                             "(default 0.7, i.e. boxes must overlap by more than 0.3 IoU)")
    parser.add_argument("--hit-counter-max", type=int, default=15,
                        help="frames a track survives with no matching detection before it is "
                             "dropped (default 15)")
    parser.add_argument("--initialization-delay", type=int, default=None,
                        help="consecutive matches required before a new track is reported "
                             "(default: Norfair's own default, hit_counter_max // 2)")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="ignore detections scoring below this (default 0.0)")
    args = parser.parse_args()

    print("loading detections from %s..." % args.detections)
    detections = load_detections(args.detections)
    print("loading team ids from %s..." % (args.team_ids or "(none)"))
    team_ids = load_team_ids(args.team_ids)

    frames = (range(args.frames[0], args.frames[1] + 1) if args.frames
             else range(min(detections), max(detections) + 1))

    print("tracking %d frames with Norfair (IoU, per-class labels)..." % len(frames))
    rows = track(detections, team_ids, frames, args.distance_threshold, args.hit_counter_max,
                args.initialization_delay, args.min_score)

    write_csv(rows, args.out)

    n_tracks = len({row[1] for row in rows})
    n_player_rows = sum(1 for row in rows if row[2] == PLAYER_CLASS_ID)
    n_team_known = sum(1 for row in rows if row[2] == PLAYER_CLASS_ID and row[5] is not None)
    print("wrote %d rows (%d tracks) to %s" % (len(rows), n_tracks, args.out))
    print("  player rows: %d   team_id known (via propagation): %d (%.1f%%)"
         % (n_player_rows, n_team_known, 100.0 * n_team_known / n_player_rows if n_player_rows else 0.0))


if __name__ == "__main__":
    main()
