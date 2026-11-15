"""Play the broadcast and tactical videos side by side, synced via frame_correspondence.csv.

merge_frame_timestamps.py matches broadcast frames to tactical frames that show the same
reconstructed game-clock moment (logical_clock_seconds). This script walks that correspondence
table and, for each matched pair, seeks both videos to the right frame and shows them next to
each other in one window -- a visual sanity check that the sync is actually correct.

    python src/multi_views/view_frame_correspondence.py --broadcast-video src/data/broadcast/BAR-ATH_broadcast.mp4 --tactical-video src/data/tactical/BAR-ATH_tactical.mp4 --start-time 00:30

ATD player/ball tracking is overlaid on both panels by default (one call to
kloppy.metrica.load_tracking_epts per source, plus its own per-source homography -- see
visualize_video_frames.py, which this borrows the "video_frame = frame_id - 1" alignment and the
projection/drawing pattern from). Pass --no-tracking to skip it (faster startup, raw video only).

A top-down bird's-eye pitch panel (visualize_video_frames.BirdsEye) is drawn below each video
panel, one per source, so off-ball shape and ball position are easy to read even when the camera
angle hides them. Pass --no-birdseye to drop just the panels (tracking overlay is kept).

A SORT-style ball track from YOLO detections (assets/detections/..._tactical_sliced.csv and
..._broadcast_sliced.csv by default) is drawn on each panel and its bird's-eye panel in magenta,
same as visualize_video_frames.py's --ball-birdseye: the detection closest to the ATD ball (when
a valid homography exists) updates a per-source Kalman filter, so the track still coasts through
frames where the ATD ball is missing or the homography is invalid. Pass --no-ball-track to skip
it for both sources.

Smart Tagging (load_smart_tagging.py) -- Metrica's automatic phase-of-play/set-piece annotation
of the tactical video -- is shown as a stack of banners on the tactical panel for every tag whose
interval covers the current tactical frame. Pass --no-smart-tagging to skip it.

Controls: space to pause/resume, Left/Right to step one matched pair back/forward while paused,
q or Esc to quit.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

from kloppy import metrica  # noqa: E402

import games  # noqa: E402
from starter import load_homography, load_smart_tagging  # noqa: E402
from ball_possession.possession_model import is_valid  # noqa: E402
from visualize_video_frames import BirdsEye, build_ball_references, draw_ball_track  # noqa: E402
from detections.ball import load_ball_detections  # noqa: E402
from detections.tracker import track_ball, track_ball_native_bytetrack  # noqa: E402


DEFAULT_CORRESPONDENCE_CSV = os.path.join(os.path.dirname(SRC_DIR),
"assets", "frame_timestamps", "frame_correspondence.csv")

DEFAULT_BALL_DETECTIONS = {
    "tactical": os.path.join(os.path.dirname(SRC_DIR), "assets", "detections",
                             "ball_detection_first_half_yolo26_tactical_sliced.csv"),
    "broadcast": os.path.join(os.path.dirname(SRC_DIR), "assets", "detections",
                              "ball_detection_first_half_yolo26_broadcast_sliced.csv"),
}

# Each video frame is resized to this height (keeping its own aspect ratio) before being placed
# side by side, so the combined window has a sane, predictable size regardless of source res.
DISPLAY_HEIGHT = 540

# Width, in pixels, of each source's bird's-eye panel, drawn below its video panel.
BIRDSEYE_WIDTH = 480

# BGR, same palette as visualize_video_frames.py.
HOME_COLOUR, AWAY_COLOUR, BALL_COLOUR = (40, 39, 214), (180, 119, 31), (0, 255, 255)


def parse_time(value):
    """Seconds (float), "MM:SS", or "HH:MM:SS" -> seconds (float). Mirrors read_timestamp.py."""
    if ":" not in value:
        return float(value)
    parts = [float(part) for part in value.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    hours, minutes, seconds = parts
    return hours * 3600.0 + minutes * 60.0 + seconds


def resolve_start_index(correspondence, start_index, start_time):
    """--start-time (if given) overrides --start-index: the row whose logical clock is closest.

    `start_time` accepts the same formats as read_timestamp.py's --start-time (plain seconds,
    "MM:SS", "HH:MM:SS"), matching the correspondence table's own `logical_clock_seconds` column.
    """
    if start_time is None:
        return start_index
    target_seconds = parse_time(start_time)
    closest_position = (correspondence["logical_clock_seconds"] - target_seconds).abs().argmin()
    return int(closest_position)


class TrackingSource:
    """ATD tracking + per-frame homography for one video source (broadcast or tactical).

    Mirrors visualize_video_frames.py: frame_id is 1-based and video_frame = frame_id - 1, so the
    tracking frame for video frame N is looked up as by_frame_id[N + 1].
    """

    def __init__(self, source, limit):
        paths = games.asset_paths(source=source)
        games.require(["atd_meta", "atd_raw", "homography"], source=source)
        print("loading %d ATD frames for %s..." % (limit, source))
        dataset = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"],
                                              limit=limit)
        self.by_frame_id = {frame.frame_id: frame for frame in dataset.frames}
        self.home_team = dataset.metadata.teams[0]
        self.matrices, _, self.valid = load_homography.load(source=source)
        self.birds_eye = BirdsEye(BIRDSEYE_WIDTH)

    def lookup(self, video_frame):
        """(tracking_frame, matrix) for `video_frame`, or (None, None) if either is unavailable."""
        tracking_frame = self.by_frame_id.get(video_frame + 1)
        if tracking_frame is None:
            return None, None
        if video_frame >= len(self.valid) or not self.valid[video_frame]:
            return tracking_frame, None
        return tracking_frame, self.matrices[video_frame]


def annotate_tracking(image, tracking_frame, matrix, home_team):
    """Draw ATD players (coloured by team) and the ball onto `image` in place, pitch -> pixel via
    `matrix`. Same palette/approach as visualize_video_frames.py's annotate(), minus the label."""
    if tracking_frame is None or matrix is None:
        return image

    for player, position in tracking_frame.players_coordinates.items():
        if not is_valid(position):
            continue
        px, py = load_homography.project(matrix, position.x, position.y)
        centre = (int(round(float(px))), int(round(float(py))))
        colour = HOME_COLOUR if player.team == home_team else AWAY_COLOUR
        cv2.circle(image, centre, 9, colour, 2)
        cv2.putText(image, str(player.jersey_no), (centre[0] - 8, centre[1] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA)

    ball = tracking_frame.ball_coordinates
    if is_valid(ball):
        px, py = load_homography.project(matrix, ball.x, ball.y)
        ball_centre = (int(round(float(px))), int(round(float(py))))
        cv2.circle(image, ball_centre, 6, BALL_COLOUR, -1)
        cv2.circle(image, ball_centre, 7, (0, 0, 0), 1)
    return image


def format_smart_tagging_label(event):
    """One-line label for a Smart Tagging event: "CODE (Tag: value, Tag: value)"."""
    tags = ", ".join("{}: {}".format(group, value) for group, value in event["tags"].items()
                      if value and value != "N/A")
    return "{}{}".format(event["code"], " ({})".format(tags) if tags else "")


def active_smart_tagging(events, video_frame):
    """Smart Tagging events whose [start_frame, end_frame] interval covers `video_frame`."""
    return [event for event in events
            if event["start_frame"] <= video_frame <= event["end_frame"]]


def draw_smart_tagging(image, events, y0=50):
    """Stack one black-background/yellow-text banner per active Smart Tagging event, top-left."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness, margin = 0.6, 2, 8
    y = y0
    for event in events:
        text = format_smart_tagging_label(event)
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(image, (10, y),
                      (10 + text_width + 2 * margin, y + text_height + baseline + 2 * margin),
                      (0, 0, 0), -1)
        cv2.putText(image, text, (10 + margin, y + text_height + margin), font, scale,
                    (0, 255, 255), thickness, cv2.LINE_AA)
        y += text_height + baseline + 2 * margin + 4
    return image


def resize_to_height(frame, height):
    """Resize a BGR frame to `height`, keeping its aspect ratio."""
    scale = height / frame.shape[0]
    width = int(round(frame.shape[1] * scale))
    return cv2.resize(frame, (width, height))


def read_frame_at(cap, frame_number, last_read_frame):
    """Seek `cap` to `frame_number` and read it, skipping the seek if already positioned there.

    cap.set(CAP_PROP_POS_FRAMES, ...) decodes from the nearest previous keyframe every time it is
    called, which is slow for every single frame of a long video; when the caller is stepping
    forward one frame at a time (the common case here) a plain read() is both correct and much
    faster.
    """
    if last_read_frame is None or frame_number != last_read_frame + 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    return frame if ret else None


def resize_to_width(frame, width):
    """Resize a BGR frame to `width`, keeping its aspect ratio."""
    scale = width / frame.shape[1]
    height = int(round(frame.shape[0] * scale))
    return cv2.resize(frame, (width, height))


def build_side_by_side(broadcast_frame, tactical_frame, logical_clock_seconds,
                        frame_broadcast, frame_tactical, birdseye_images=None):
    """Stack the two (possibly missing) frames horizontally with a small info banner on each.

    `birdseye_images`, if given, is {"broadcast": image_or_None, "tactical": image_or_None}: a
    top-down pitch panel drawn below the matching video panel, resized to the same width so the
    two columns stay aligned.
    """
    height, width = DISPLAY_HEIGHT, int(DISPLAY_HEIGHT * 16 / 9)
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(blank, "no frame", (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 255), 2, cv2.LINE_AA)

    columns = []
    for frame, label, frame_number in (
            (broadcast_frame, "broadcast", frame_broadcast),
            (tactical_frame, "tactical", frame_tactical)):
        panel = resize_to_height(frame, DISPLAY_HEIGHT) if frame is not None else blank.copy()
        cv2.putText(panel, "{}  frame {}".format(label, frame_number), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        birdseye = birdseye_images.get(label) if birdseye_images else None
        if birdseye is not None:
            birdseye_panel = resize_to_width(birdseye, panel.shape[1])
            column = cv2.vconcat([panel, birdseye_panel])
        else:
            column = panel
        columns.append(column)

    # The two columns can differ in total height (birdseye aspect ratio follows panel width,
    # which differs if the two videos have different source aspect ratios); pad the shorter one.
    max_height = max(column.shape[0] for column in columns)
    columns = [cv2.copyMakeBorder(column, 0, max_height - column.shape[0], 0, 0,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))
               if column.shape[0] < max_height else column
               for column in columns]

    combined = cv2.hconcat(columns)
    cv2.putText(combined, "logical clock: {:.2f}s".format(logical_clock_seconds),
                (10, DISPLAY_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                cv2.LINE_AA)
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--broadcast-video", required=True)
    parser.add_argument("--tactical-video", required=True)
    parser.add_argument("--correspondence-csv", default=DEFAULT_CORRESPONDENCE_CSV)
    parser.add_argument("--start-index", type=int, default=0,
                        help="row of the correspondence table to start at (default: 0)")
    parser.add_argument("--start-time", default=None,
                        help="start at the matched pair closest to this logical-clock time, "
                             "as seconds or MM:SS / HH:MM:SS (e.g. 00:30); overrides "
                             "--start-index")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="playback speed, in matched pairs per second (default: 25)")
    parser.add_argument("--no-tracking", action="store_true",
                        help="skip loading ATD tracking; show raw video only (faster startup)")
    parser.add_argument("--no-birdseye", action="store_true",
                        help="skip the bird's-eye pitch panels (tracking overlay is kept)")
    parser.add_argument("--tracking-limit", type=int, default=None,
                        help="ATD frames to load per source (default: enough to cover every "
                             "matched frame in the correspondence table)")
    parser.add_argument("--no-ball-track", action="store_true",
                        help="skip the SORT-style ball-detection track (both sources)")
    parser.add_argument("--tactical-ball-detections", default=DEFAULT_BALL_DETECTIONS["tactical"],
                        help="CSV of tactical ball detections (frame,x1,y1,x2,y2,score,class_id)")
    parser.add_argument("--broadcast-ball-detections", default=DEFAULT_BALL_DETECTIONS["broadcast"],
                        help="CSV of broadcast ball detections (frame,x1,y1,x2,y2,score,class_id)")
    parser.add_argument("--min-detection-score", type=float, default=0.3,
                        help="ignore detections scoring below this threshold (default 0.3)")
    parser.add_argument("--gating-distance", type=float, default=80.0,
                        help="max pixel distance between a predicted/reference and detected "
                             "ball position to associate them (default 80)")
    parser.add_argument("--max-age", type=int, default=10,
                        help="frames to coast the ball track without a matching detection "
                             "before dropping it (default 10)")
    parser.add_argument("--tracker", choices=("sort", "bytetrack"), default="sort",
                        help="ball-tracking algorithm: SORT-style (default) or BYTE-style "
                             "(also tries low-score detections to recover an existing track)")
    parser.add_argument("--no-smart-tagging", action="store_true",
                        help="skip the Smart Tagging banners on the tactical panel")
    args = parser.parse_args()

    correspondence = pd.read_csv(args.correspondence_csv)

    tracking = {}
    if not args.no_tracking:
        default_limit = {
            "broadcast": int(correspondence["frame_broadcast"].max()) + 2,
            "tactical": int(correspondence["frame_tactical"].max()) + 2,
        }
        for source in ("broadcast", "tactical"):
            limit = args.tracking_limit or default_limit[source]
            tracking[source] = TrackingSource(source, limit)

    ball_tracks = {}
    if not args.no_ball_track:
        ball_detection_paths = {"tactical": args.tactical_ball_detections,
                                "broadcast": args.broadcast_ball_detections}
        frame_columns = {"tactical": "frame_tactical", "broadcast": "frame_broadcast"}
        for source in ("tactical", "broadcast"):
            detections_path = ball_detection_paths[source]
            if source not in tracking:
                print("--no-tracking disables the ATD references needed for the %s ball "
                      "track; skipping" % source)
                continue
            if not os.path.exists(detections_path):
                print("%s ball detections file not found, skipping: %s"
                      % (source, detections_path))
                continue
            print("loading %s ball detections from %s..." % (source, detections_path))
            detections = load_ball_detections(detections_path)
            tracking_source = tracking[source]
            frame_column = frame_columns[source]
            frame_range = range(int(correspondence[frame_column].min()),
                                int(correspondence[frame_column].max()) + 1)
            references = build_ball_references(frame_range, tracking_source.by_frame_id,
                                               tracking_source.matrices, tracking_source.valid)
            print("tracking ball over %d %s frames (%d with an ATD reference)..."
                  % (len(frame_range), source, len(references)))

            if args.tracker == "bytetrack":
                video_path = games.asset_paths(source=source)["video"]
                probe = cv2.VideoCapture(video_path)
                frame_size = (int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                             int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)))
                probe.release()
                ball_tracks[source] = track_ball_native_bytetrack(
                    detections, frame_range, frame_size, references,
                    track_thresh=args.min_detection_score, gating_distance=args.gating_distance,
                    track_buffer=args.max_age)
            else:
                ball_tracks[source] = track_ball(detections, frame_range, references,
                                                 args.min_detection_score, args.gating_distance,
                                                 args.max_age)


    smart_tagging_events = None
    if not args.no_smart_tagging:
        try:
            smart_tagging_events = load_smart_tagging.load()
            print("loaded %d Smart Tagging events" % len(smart_tagging_events))
        except IOError as error:
            print("Smart Tagging data not found, skipping: %s" % error)

    broadcast_cap = cv2.VideoCapture(args.broadcast_video)
    tactical_cap = cv2.VideoCapture(args.tactical_video)
    if not broadcast_cap.isOpened():
        raise ValueError("Cannot open broadcast video: {}".format(args.broadcast_video))
    if not tactical_cap.isOpened():
        raise ValueError("Cannot open tactical video: {}".format(args.tactical_video))

    window = "broadcast | tactical"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    index = resolve_start_index(correspondence, args.start_index, args.start_time)
    paused = False
    last_broadcast_frame = last_tactical_frame = None
    frame_delay_ms = max(1, int(round(1000.0 / args.fps)))

    try:
        while 0 <= index < len(correspondence):
            row = correspondence.iloc[index]
            frame_broadcast = int(row["frame_broadcast"])
            frame_tactical = int(row["frame_tactical"])

            broadcast_frame = read_frame_at(broadcast_cap, frame_broadcast, last_broadcast_frame)
            tactical_frame = read_frame_at(tactical_cap, frame_tactical, last_tactical_frame)
            last_broadcast_frame, last_tactical_frame = frame_broadcast, frame_tactical

            for source, frame_image, video_frame in (
                    ("broadcast", broadcast_frame, frame_broadcast),
                    ("tactical", tactical_frame, frame_tactical)):
                if frame_image is None or source not in tracking:
                    continue
                tracking_source = tracking[source]
                tracking_frame, matrix = tracking_source.lookup(video_frame)
                annotate_tracking(frame_image, tracking_frame, matrix, tracking_source.home_team)

            track_points = {}
            for source, video_frame, frame_image in (
                    ("broadcast", frame_broadcast, broadcast_frame),
                    ("tactical", frame_tactical, tactical_frame)):
                track_point = ball_tracks.get(source, {}).get(video_frame)
                if track_point is not None:
                    track_points[source] = track_point
                    if frame_image is not None:
                        draw_ball_track(frame_image, track_point)

            if smart_tagging_events is not None and tactical_frame is not None:
                active_tags = active_smart_tagging(smart_tagging_events, frame_tactical)
                if active_tags:
                    draw_smart_tagging(tactical_frame, active_tags)

            birdseye_images = None
            if tracking and not args.no_birdseye:
                birdseye_images = {}
                for source, video_frame in (("broadcast", frame_broadcast),
                                            ("tactical", frame_tactical)):
                    if source not in tracking:
                        continue
                    tracking_source = tracking[source]
                    tracking_frame, matrix = tracking_source.lookup(video_frame)
                    track_point = track_points.get(source)
                    if track_point is not None:
                        birdseye_images[source] = tracking_source.birds_eye.render(
                            tracking_frame, tracking_source.home_team,
                            track_point_px=track_point["centre"], matrix=matrix)
                    else:
                        birdseye_images[source] = tracking_source.birds_eye.render(
                            tracking_frame, tracking_source.home_team)

            combined = build_side_by_side(
                broadcast_frame, tactical_frame, row["logical_clock_seconds"],
                frame_broadcast, frame_tactical, birdseye_images)
            cv2.imshow(window, combined)

            key = cv2.waitKey(0 if paused else frame_delay_ms) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                paused = not paused
            elif key in (81, 2424832, ord("a")) and paused:  # Left arrow / 'a'
                index -= 1
                continue
            elif key in (83, 2555904, ord("d")) and paused:  # Right arrow / 'd'
                index += 1
                continue
            elif not paused:
                index += 1
    finally:
        broadcast_cap.release()
        tactical_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
