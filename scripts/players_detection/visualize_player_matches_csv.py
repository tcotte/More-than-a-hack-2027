"""Visualize export_player_matches_csv.py's output: draw the matched players back onto video frames.

Deliberately standalone from the ATD/homography machinery -- it only needs the CSV (already has
the projected pixel position and, if available, the detection box for every matched player) and
the raw video to grab frame images from, so it can be handed to anyone who just wants to eyeball
the matches without re-running any matching.

    python scripts/players_detection/visualize_player_matches_csv.py \
        --csv assets/trackings/players/player_matches.csv \
        --frames 11185 11250 --show
    python scripts/players_detection/visualize_player_matches_csv.py \
        --csv assets/trackings/players/player_matches.csv --clip 11185 11435 --play
    python scripts/players_detection/visualize_player_matches_csv.py \
        --csv assets/trackings/players/player_matches.csv --frames 11185 11250 --circles-only

Images (and clips, as MP4) are written to scripts/players_detection/frames_out/ by default.
"""

import argparse
import os
import sys
from collections import defaultdict

import cv2
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

import games  # noqa: E402

# BGR, same convention as src/visualize_video_frames.py's HOME_COLOUR/AWAY_COLOUR.
HOME_COLOUR, AWAY_COLOUR = (40, 39, 214), (180, 119, 31)


def load_matches(path):
    """{video_frame: [row, ...]} from export_player_matches_csv.py's CSV, rows as dicts."""
    df = pd.read_csv(path)
    by_frame = defaultdict(list)
    for row in df.to_dict(orient="records"):
        by_frame[int(row["video_frame"])].append(row)
    return by_frame


def draw_matches(image, rows, circles_only):
    """Draw one CSV row per matched player: a team-coloured point/box, jersey no, track id, GK tag.

    `centre` is always the row's (px, py) -- the ATD position projected to image pixels, i.e. the
    player's actual pitch location, not the detection box's centre. When `circles_only` is False
    (the default) the matched detection box is also drawn, as a rectangle, alongside that circle;
    pass --circles-only to skip the box and draw only the ATD-centred circle.
    """
    for row in rows:
        colour = HOME_COLOUR if row["side"] == "home" else AWAY_COLOUR
        centre = (int(round(row["px"])), int(round(row["py"])))

        if not circles_only and pd.notna(row.get("box_x1")):
            x1, y1, x2, y2 = (int(round(row[key])) for key in ("box_x1", "box_y1", "box_x2", "box_y2"))
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)

        cv2.circle(image, centre, 9, colour, 2)

        label = "{} (#{})".format(int(row["jersey_no"]), int(row["track_id"]))
        if row.get("is_goalkeeper") in (True, "True", 1):
            label += " GK"
        cv2.putText(image, label, (centre[0] - 8, centre[1] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA)
    return image


def read_video_frame(capture, video_frame):
    capture.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
    ok, image = capture.read()
    if not ok:
        raise IOError("could not read video frame %d" % video_frame)
    return image


def render_clip(capture, by_frame, start, end, out_path, play, scale, circles_only):
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for video_frame in range(start, end + 1):
            image = read_video_frame(capture, video_frame)
            draw_matches(image, by_frame.get(video_frame, []), circles_only)
            if scale != 1.0:
                image = cv2.resize(image, (width, height))
            writer.write(image)
            if play:
                cv2.imshow("player matches (q: quit)", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        writer.release()
    print("wrote %s" % out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True,
                        help="output of export_player_matches_csv.py")
    parser.add_argument("--frames", type=int, nargs=2, metavar=("START", "END"), default=None,
                        help="video frames (0-based, inclusive) to render as JPGs; defaults to "
                             "the full range present in --csv")
    parser.add_argument("--clip", type=int, nargs=2, metavar=("START", "END"),
                        help="render video frames START..END as an annotated MP4 instead")
    parser.add_argument("--play", action="store_true", help="with --clip: play it while rendering")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="with --clip: output size relative to the source (default 0.5)")
    parser.add_argument("--circles-only", action="store_true",
                        help="draw only a circle at each player's projected ATD centre, "
                             "skipping the matched detection box (drawn by default alongside it)")
    parser.add_argument("--out", default=os.path.join(HERE, "frames_out"))
    parser.add_argument("--show", action="store_true", help="display each frame in a window")
    parser.add_argument("--video-type", default="tactical")
    args = parser.parse_args()

    print("loading matches from %s..." % args.csv)
    by_frame = load_matches(args.csv)

    paths = games.asset_paths(source=args.video_type)
    games.require(["video"], source=args.video_type)
    capture = cv2.VideoCapture(paths["video"])
    if not capture.isOpened():
        raise IOError("could not open {}".format(paths["video"]))

    os.makedirs(args.out, exist_ok=True)

    try:
        if args.clip:
            start, end = args.clip
            out_path = os.path.join(args.out, "clip_%06d_%06d.mp4" % (start, end))
            render_clip(capture, by_frame, start, end, out_path, args.play, args.scale,
                       args.circles_only)
            return

        if args.frames is not None:
            start, end = args.frames
        else:
            start, end = min(by_frame), max(by_frame)

        for video_frame in range(start, end + 1):
            image = read_video_frame(capture, video_frame)
            draw_matches(image, by_frame.get(video_frame, []), args.circles_only)
            out_path = os.path.join(args.out, "frame_%06d.jpg" % video_frame)
            cv2.imwrite(out_path, image)
            print("wrote %s" % out_path)

            if args.show:
                cv2.imshow("player matches (any key: next, q: quit)", image)
                if cv2.waitKey(0) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
