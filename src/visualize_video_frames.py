"""Overlay the ATD tracking (players + ball) on frames of the tactical video.

Tracking positions are normalised pitch coordinates (0-1); the per-frame homography maps them to
image pixels. Alignment: video_frame = frame_id - 1.

    python starter/visualize_video_frames.py                         # a few frames after kickoff
    python starter/visualize_video_frames.py --frames 11190 11500    # specific video frames
    python starter/visualize_video_frames.py --show                  # also open a window per frame
    python starter/visualize_video_frames.py --clip 11185 11435 --play   # annotated video clip
    python starter/visualize_video_frames.py --clip 11185 11435 --opacity 0.5 --panel-width 0.35

Images (and clips, as MP4) are written to starter/frames_out/ by default.
"""

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from kloppy import metrica  # noqa: E402

import games  # noqa: E402
import load_homography  # noqa: E402
from possession_model import PITCH_LENGTH, PITCH_WIDTH, PossessionModel, is_valid  # noqa: E402

# BGR
HOME_COLOUR, AWAY_COLOUR, BALL_COLOUR = (40, 39, 214), (180, 119, 31), (0, 255, 255)
START = 11190
DEFAULT_FRAMES = [i + START for i in range(100)]


class BirdsEye:
    """Top-down pitch panel drawn like utils.plot_frame (metres, origin bottom-left, y up)."""

    MARGIN = 3.0
    GRASS, LINE, BALL = (79, 143, 63), (255, 255, 255), (255, 255, 255)

    def __init__(self, panel_width, length=PITCH_LENGTH, width=PITCH_WIDTH):
        self.length, self.width = length, width
        self.px_per_m = panel_width / (length + 2 * self.MARGIN)
        self.size = (panel_width, int(round((width + 2 * self.MARGIN) * self.px_per_m)))
        self.background = self._draw_pitch()

    def to_px(self, x_m, y_m):
        return (int(round((x_m + self.MARGIN) * self.px_per_m)),
                int(round((self.width - y_m + self.MARGIN) * self.px_per_m)))

    def _draw_pitch(self):
        panel_w, panel_h = self.size
        image = np.full((panel_h, panel_w, 3), self.GRASS, dtype=np.uint8)
        L, W = self.length, self.width
        cv2.rectangle(image, self.to_px(0, W), self.to_px(L, 0), self.GRASS, -1)
        cv2.rectangle(image, self.to_px(0, W), self.to_px(L, 0), self.LINE, 2)
        cv2.line(image, self.to_px(L / 2, 0), self.to_px(L / 2, W), self.LINE, 2)
        cv2.circle(image, self.to_px(L / 2, W / 2), int(round(9.15 * self.px_per_m)),
                   self.LINE, 2, cv2.LINE_AA)
        for side in (0, 1):
            # Penalty area 16.5 m deep and 40.3 m wide, six-yard box 5.5 m by 18.3 m.
            for depth, box_width in ((16.5, 40.3), (5.5, 18.3)):
                x = 0 if side == 0 else L - depth
                y = (W - box_width) / 2
                cv2.rectangle(image, self.to_px(x, y + box_width), self.to_px(x + depth, y),
                              self.LINE, 2)
        return image

    def render(self, tracking_frame, home_team):
        image = self.background.copy()
        if tracking_frame is None:
            return image
        radius = max(4, int(round(0.9 * self.px_per_m)))
        for player, position in tracking_frame.players_coordinates.items():
            if not is_valid(position):
                continue

            if position.y > 0.5:
                y = 1 - position.y
            else:
                y = 0.5 + (0.5 - position.y)

            centre = self.to_px(position.x * self.length, y * self.width)
            colour = HOME_COLOUR if player.team == home_team else AWAY_COLOUR
            cv2.circle(image, centre, radius, colour, -1, cv2.LINE_AA)
            cv2.circle(image, centre, radius, (0, 0, 0), 1, cv2.LINE_AA)
        ball = tracking_frame.ball_coordinates
        if is_valid(ball):

            if ball.y > 0.5:
                y = 1 - ball.y
            else:
                y = 0.5 + (0.5 - ball.y)

            centre = self.to_px(ball.x * self.length, y * self.width)
            ball_radius = max(3, radius - 2)
            cv2.circle(image, centre, ball_radius, self.BALL, -1, cv2.LINE_AA)
            cv2.circle(image, centre, ball_radius, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, "ball not tracked", (10, image.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        return image


def read_video_frame(capture, video_frame):
    capture.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
    ok, image = capture.read()
    if not ok:
        raise IOError("could not read video frame {}".format(video_frame))
    return image


def draw_pitch_outline(image, matrix):
    corners = np.array(load_homography.pitch_corners(matrix), dtype=np.int32)
    cv2.polylines(image, [corners], isClosed=True, color=(255, 255, 255), thickness=1)
    halfway = [load_homography.project(matrix, 0.5, y) for y in (0.0, 1.0)]
    cv2.line(image, tuple(int(v) for v in halfway[0]), tuple(int(v) for v in halfway[1]),
             (255, 255, 255), 1)


def annotate(image, tracking_frame, matrix, home_team, video_frame, highlight_player=None):
    draw_pitch_outline(image, matrix)

    for player, position in tracking_frame.players_coordinates.items():
        if not is_valid(position):
            continue
        px, py = load_homography.project(matrix, position.x, position.y)
        centre = (int(round(float(px))), int(round(float(py))))
        colour = HOME_COLOUR if player.team == home_team else AWAY_COLOUR
        if player == highlight_player:
            cv2.circle(image, centre, 15, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(image, centre, 9, colour, 2)
        cv2.putText(image, str(player.jersey_no), (centre[0] - 8, centre[1] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA)

    ball = tracking_frame.ball_coordinates
    has_ball = is_valid(ball)
    if has_ball:
        px, py = load_homography.project(matrix, ball.x, ball.y)
        centre = (int(round(float(px))), int(round(float(py))))
        cv2.circle(image, centre, 6, BALL_COLOUR, -1)
        cv2.circle(image, centre, 7, (0, 0, 0), 1)

    label = "video frame {} | ATD frame_id {}{}".format(
        video_frame, tracking_frame.frame_id, "" if has_ball else " | ball not tracked")
    cv2.rectangle(image, (0, 0), (len(label) * 11 + 20, 34), (0, 0, 0), -1)
    cv2.putText(image, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                cv2.LINE_AA)
    return image


def draw_touch_alert(image):
    text = "Touche probable"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness, margin = 0.8, 2, 12
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, image.shape[1] - text_width - 2 * margin)
    y = image.shape[0]
    cv2.rectangle(image, (x, y - text_height - baseline - 2 * margin),
                  (image.shape[1], y), (0, 0, 0), -1)
    cv2.putText(image, text, (x + margin, y - margin - baseline), font, scale,
                (0, 255, 255), thickness, cv2.LINE_AA)


GT_PATH = os.path.join(os.path.dirname(HERE), "mtah_challenge", "evaluation", "gt",
                       "BAR-ATH_possession_gt_public.csv")
# label -> (text, BGR colour)
GT_LABELS = {"H": ("Barcelone", HOME_COLOUR), "A": ("Athletic", AWAY_COLOUR),
             "D": ("Dispute", (0, 165, 255)), "X": ("Ballon mort", (128, 128, 128))}


def load_gt(path):
    """Return {video_frame: label} from the possession ground truth CSV."""
    import csv
    with open(path, newline="") as f:
        return {int(row["frame"]): row["label"] for row in csv.DictReader(f)}


def draw_possession(image, label, changed, title="GT", y0=10):
    """Top-right possession box; thick yellow frame and 'CHANGEMENT' while `changed`."""
    text, colour = GT_LABELS.get(label, (label, (128, 128, 128)))
    text = "{}: {}".format(title, text)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness, margin = 0.9, 2, 12
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x1 = image.shape[1] - 10
    x0, y1 = x1 - tw - 2 * margin, y0 + th + baseline + 2 * margin
    cv2.rectangle(image, (x0, y0), (x1, y1), colour, -1)
    cv2.putText(image, text, (x0 + margin, y1 - margin - baseline), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    if changed:
        cv2.rectangle(image, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (0, 255, 255), 4)
        cv2.putText(image, "CHANGEMENT", (x0 - 4, y1 + 28), font, 0.7, (0, 255, 255), 2,
                    cv2.LINE_AA)


def overlay_bottom(image, panel, opacity):
    """Alpha-blend `panel` onto the bottom-centre of `image`, in place."""
    panel_h, panel_w = panel.shape[:2]
    image_h, image_w = image.shape[:2]
    x0, y0 = (image_w - panel_w) // 2, image_h - panel_h
    region = image[y0:y0 + panel_h, x0:x0 + panel_w]
    cv2.addWeighted(panel, opacity, region, 1.0 - opacity, 0, dst=region)
    return image


def render_clip(capture, start, end, by_frame_id, matrices, valid, home_team, out_path, play,
                scale, opacity=0.6, panel_fraction=0.4, gt=None):
    """Annotate video frames start..end (inclusive), write them to an MP4 and optionally play.

    The bird's-eye pitch is blended onto the bottom of the video at `opacity` (0-1), at
    `panel_fraction` of the video width.
    """
    fps = capture.get(cv2.CAP_PROP_FPS) or games.GAMES[games.GAME]["fps"]
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    birds_eye = BirdsEye(int(width * panel_fraction))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    delay = max(1, int(1000 / fps))
    window = "ATD overlay (space: pause, q: quit)"
    model = PossessionModel(home_team)
    touch_alert_until = -1
    highlighted_player = None
    gt_label = gt.get(start - 1) if gt else None
    change_until = -1
    pred_label, pred_change_until = "X", -1

    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    try:
        for video_frame in range(start, end + 1):
            ok, image = capture.read()
            if not ok:
                print("video ended at frame %d" % video_frame)
                break
            tracking_frame = by_frame_id.get(video_frame + 1)
            output = model.update(video_frame, tracking_frame)
            if output.touch_player is not None:
                highlighted_player = output.touch_player
                print("Frame {}: Touche probable".format(video_frame), flush=True)
                touch_alert_until = video_frame + fps
            if output.label != pred_label:
                if pred_label != "X":
                    pred_change_until = video_frame + fps
                pred_label = output.label
            if tracking_frame is not None and valid[video_frame]:
                annotate(image, tracking_frame, matrices[video_frame], home_team, video_frame,
                         highlighted_player if video_frame < touch_alert_until else None)
            if scale != 1.0:
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            overlay_bottom(image, birds_eye.render(tracking_frame, home_team), opacity)
            if video_frame < touch_alert_until:
                draw_touch_alert(image)
            draw_possession(image, pred_label, video_frame < pred_change_until,
                            "Prediction", y0=110)
            if gt:
                new_label = gt.get(video_frame, gt_label)
                if new_label != gt_label and gt_label is not None:
                    change_until = video_frame + fps
                    print("Frame {}: {} -> {}".format(video_frame, gt_label, new_label),
                          flush=True)
                gt_label = new_label
                if gt_label is not None:
                    draw_possession(image, gt_label, video_frame < change_until)
            writer.write(image)

            if play:
                cv2.imshow(window, image)
                key = cv2.waitKey(delay) & 0xFF
                if key == ord(" "):
                    key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    break
    finally:
        writer.release()
    print("wrote %s" % out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, nargs="+", default=DEFAULT_FRAMES,
                        help="video frames (0-based) to render")
    parser.add_argument("--limit", type=int, default=12000,
                        help="ATD frames to load (raised automatically to cover --frames)")
    parser.add_argument("--out", default=os.path.join(HERE, "../more-than-a-hack-2027/starter/frames_out"))
    parser.add_argument("--show", action="store_true", help="display each frame in a window")
    parser.add_argument("--clip", type=int, nargs=2, metavar=("START", "END"),
                        help="render video frames START..END as an annotated MP4 instead")
    parser.add_argument("--play", action="store_true", help="with --clip: play it while rendering")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="with --clip: output size relative to the source (default 0.5)")
    parser.add_argument("--opacity", type=float, default=0.6,
                        help="with --clip: bird's-eye panel opacity, 0-1 (default 0.6)")
    parser.add_argument("--panel-width", type=float, default=0.25,
                        help="with --clip: bird's-eye panel width as a fraction of the video "
                             "width (default 0.4)")
    parser.add_argument("--gt", default=GT_PATH,
                        help="with --clip: possession GT CSV (frame,label); '' to disable")
    args = parser.parse_args()
    if not 0.0 <= args.opacity <= 1.0:
        parser.error("--opacity must be between 0 and 1")
    if not 0.0 < args.panel_width <= 1.0:
        parser.error("--panel-width must be in (0, 1]")

    paths = games.asset_paths(source="tactical")
    games.require(["video", "atd_meta", "atd_raw", "homography"], source="tactical")

    last_frame = args.clip[1] if args.clip else max(args.frames)
    limit = max(args.limit, last_frame + 2)
    print("loading %d ATD frames..." % limit)
    dataset = metrica.load_tracking_epts(meta_data=paths["atd_meta"],
                                         raw_data=paths["atd_raw"], limit=limit)
    by_frame_id = {frame.frame_id: frame for frame in dataset.frames}
    home_team = dataset.metadata.teams[0]

    print("loading homography...")
    matrices, _, valid = load_homography.load(source="tactical")

    os.makedirs(args.out, exist_ok=True)
    capture = cv2.VideoCapture(paths["video"])
    if not capture.isOpened():
        raise IOError("could not open {}".format(paths["video"]))

    if args.clip:
        start, end = args.clip
        out_path = os.path.join(args.out, "clip_%06d_%06d.mp4" % (start, end))
        try:
            render_clip(capture, start, end, by_frame_id, matrices, valid, home_team, out_path,
                        args.play, args.scale, args.opacity, args.panel_width,
                        load_gt(args.gt) if args.gt else None)
        finally:
            capture.release()
            cv2.destroyAllWindows()
        return

    try:
        for video_frame in args.frames:
            tracking_frame = by_frame_id.get(video_frame + 1)
            if tracking_frame is None:
                print("video frame %d: no ATD record, skipped" % video_frame)
                continue
            if not valid[video_frame]:
                print("video frame %d: no valid homography, skipped" % video_frame)
                continue

            image = read_video_frame(capture, video_frame)
            annotate(image, tracking_frame, matrices[video_frame], home_team, video_frame)

            out_path = os.path.join(args.out, "frame_%06d.jpg" % video_frame)
            cv2.imwrite(out_path, image)
            print("wrote %s" % out_path)

            if args.show:
                cv2.imshow("ATD overlay (any key: next, q: quit)", image)
                if cv2.waitKey(0) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
