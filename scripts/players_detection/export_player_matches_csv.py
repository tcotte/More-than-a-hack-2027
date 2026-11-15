"""Match ATD tracking players to detection boxes, frame by frame, and dump the result to a CSV.

This is the same matching/goalkeeper logic src/visualize_video_frames.py's annotate() draws on
top of the video (see match_atd_players_one_to_one(), match_atd_players_to_boxes() and the
penalty-box goalkeeper override), but instead of rendering it on images it writes one row per
matched player per frame to a CSV -- so the result can be inspected, filtered or re-plotted later
without re-running the matching (see visualize_player_matches_csv.py for that).

Output columns:
    video_frame, atd_frame_id, side, jersey_no, player_key, track_id, is_goalkeeper,
    atd_x, atd_y, px, py, box_x1, box_y1, box_x2, box_y2, box_score

`side` is "home"/"away" (after the goalkeeper team override, if any). `player_key` is
"{side}_{jersey_no}" at *export* time, i.e. before the override -- a stable id across frames.
`atd_x`/`atd_y` are the player's normalised (0-1) pitch position; `px`/`py` are that position
projected to image pixels via the frame's homography. `box_*` are the matched detection box, if
the tracks CSV had x1/y1/x2/y2/score columns (same frame, same track_id); blank otherwise.

    python scripts/players_detection/export_player_matches_csv.py \
        --player-tracks-csv assets/trackings/players/tracked_players.csv \
        --frames 11185 11435 --out assets/trackings/players/player_matches.csv
"""

import argparse
import csv
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kloppy import metrica  # noqa: E402

import games  # noqa: E402
import load_homography  # noqa: E402
from ball_possession.possession_model import is_valid  # noqa: E402
from visualize_video_frames import (  # noqa: E402
    load_player_tracks_csv,
    match_atd_players_one_to_one,
    match_atd_players_to_boxes,
)

OUTPUT_FIELDS = [
    "video_frame", "atd_frame_id", "side", "jersey_no", "player_key", "track_id",
    "is_goalkeeper", "atd_x", "atd_y", "px", "py",
    "box_x1", "box_y1", "box_x2", "box_y2", "box_score",
]


def load_raw_boxes(path):
    """{(video_frame, track_id): (x1, y1, x2, y2, score or None)} for the box lookup columns.

    Separate from load_player_tracks_csv(), which only keeps the (bottom_mid_x, bottom_mid_y,
    track_id) triplet needed for matching -- this keeps the full box so it can be written out
    alongside each match.
    """
    df = pd.read_csv(path)
    has_score = "score" in df.columns
    boxes = {}
    for row in df.itertuples(index=False):
        key = (int(row.frame), int(row.track_id))
        score = getattr(row, "score") if has_score else None
        boxes[key] = (row.x1, row.y1, row.x2, row.y2, score)
    return boxes


def export(video_frames, by_frame_id, matrices, valid, home_team, away_team, pitch_dimensions,
          player_boxes, raw_boxes, threshold, one_to_one):
    """Yield one dict per matched player per video frame, in OUTPUT_FIELDS order."""
    for video_frame in video_frames:
        tracking_frame = by_frame_id.get(video_frame + 1)
        if tracking_frame is None or not valid[video_frame]:
            continue

        boxes = player_boxes.get(video_frame)
        if one_to_one:
            matches, team_override = match_atd_players_one_to_one(
                tracking_frame, matrices[video_frame], boxes, threshold, video_frame,
                home_team, away_team, pitch_dimensions)
        else:
            matches = match_atd_players_to_boxes(
                tracking_frame, matrices[video_frame], boxes, threshold)
            team_override = {}

        for player, position in tracking_frame.players_coordinates.items():
            if not is_valid(position) or player not in matches:
                continue

            track_id = matches[player]
            effective_team = team_override.get(player, player.team)
            side = "home" if effective_team == home_team else "away"
            px, py = load_homography.project(matrices[video_frame], position.x, position.y)
            box = raw_boxes.get((video_frame, track_id), (None, None, None, None, None))

            yield {
                "video_frame": video_frame,
                "atd_frame_id": tracking_frame.frame_id,
                "side": side,
                "jersey_no": player.jersey_no,
                "player_key": "{}_{}".format(
                    "home" if player.team == home_team else "away", player.jersey_no),
                "track_id": track_id,
                "is_goalkeeper": player in team_override,
                "atd_x": position.x,
                "atd_y": position.y,
                "px": float(px),
                "py": float(py),
                "box_x1": box[0], "box_y1": box[1], "box_x2": box[2], "box_y2": box[3],
                "box_score": box[4],
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--player-tracks-csv", required=True,
                        help="tracked-players CSV (frame,track_id,x1,y1,x2,y2[,score]), e.g. "
                             "the output of scripts/players_detection/track_with_bytetrack.py")
    parser.add_argument("--frames", type=int, nargs=2, metavar=("START", "END"), default=None,
                        help="video frames (0-based, inclusive) to export; defaults to the "
                             "full range covered by --player-tracks-csv")
    parser.add_argument("--out", default=os.path.join(
        HERE, "../../assets/trackings/players/player_matches.csv"),
                        help="CSV to write")
    parser.add_argument("--limit", type=int, default=None,
                        help="ATD frames to load (defaults to covering --frames)")
    parser.add_argument("--player-match-threshold", type=float, default=60.0,
                        help="max pixel distance between an ATD player position and its "
                             "matched box (default 60)")
    parser.add_argument("--nearest-only", action="store_true",
                        help="use the looser nearest-box match_atd_players_to_boxes() instead "
                             "of the default one-to-one Hungarian assignment (no goalkeeper "
                             "team override in that case)")
    parser.add_argument("--video-type", default="tactical")
    args = parser.parse_args()

    print("loading tracked player boxes from %s..." % args.player_tracks_csv)
    player_boxes = load_player_tracks_csv(args.player_tracks_csv)
    raw_boxes = load_raw_boxes(args.player_tracks_csv)

    if args.frames is not None:
        start, end = args.frames
    else:
        tracked_frames = list(player_boxes.keys())
        start, end = min(tracked_frames), max(tracked_frames)
    video_frames = range(start, end + 1)

    limit = max(args.limit or 0, end + 2)
    print("loading %d ATD frames..." % limit)
    paths = games.asset_paths(source=args.video_type)
    games.require(["atd_meta", "atd_raw", "homography"], source=args.video_type)
    dataset = metrica.load_tracking_epts(meta_data=paths["atd_meta"],
                                         raw_data=paths["atd_raw"], limit=limit)
    by_frame_id = {frame.frame_id: frame for frame in dataset.frames}
    home_team = dataset.metadata.teams[0]
    away_team = dataset.metadata.teams[1]
    pitch_dimensions = dataset.metadata.pitch_dimensions

    print("loading homography...")
    matrices, _, valid = load_homography.load(source=args.video_type)

    rows = list(export(video_frames, by_frame_id, matrices, valid, home_team, away_team,
                       pitch_dimensions, player_boxes, raw_boxes,
                       args.player_match_threshold, one_to_one=not args.nearest_only))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("wrote %d matched player rows (%d video frames) to %s"
         % (len(rows), len(video_frames), args.out))


if __name__ == "__main__":
    main()
