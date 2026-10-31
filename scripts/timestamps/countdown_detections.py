"""Read the on-screen match clock from a video, starting at a chosen time.

The broadcast/tactical overlay prints the running clock ("MM:SS") in the top-left corner of the
frame. This script seeks a video to a start time given on the command line, then OCRs that fixed
rectangle on every frame (or every Nth frame) with Tesseract, printing what it reads and -- unlike
the other starter scripts -- optionally writing a CSV of (frame, video_time, clock_text).

    pip install pytesseract opencv-python
    # plus the Tesseract OCR *engine* itself (pytesseract only talks to it):
    #   Windows: https://github.com/UB-Mannheim/tesseract/wiki
    #   macOS:   brew install tesseract
    #   Linux:   apt install tesseract-ocr

    python src/read_timestamp.py path\\to\\video.mp4 --start-time 00:02 --end-time 00:10 --show

--start-time / --end-time accept plain seconds ("125") or "MM:SS" / "HH:MM:SS". --show opens a
window and plays the video back in real time (paced to its own fps) next to the printed readings.
"""

import argparse
import csv
import os
import shutil
import sys
import time

import cv2

class RECTANGLES:
    """Predefined clock rectangles for different videos, measured on a 1920x1080 frame.

    The rectangle is (x0, y0, width, height) of the top-left corner. Override with --rect if a
    different video/crop needs a different box.
    """
    TACTICAL = (135, 8, 97, 37)
    BROADCAST = (99, 120, 80, 48)

class VIDEO_TYPES:
    """Predefined video types for different videos, measured on a 1920x1080 frame.

    The rectangle is (x0, y0, width, height) of the top-left corner. Override with --rect if a
    different video/crop needs a different box.
    """
    TACTICAL = "tactical"
    BROADCAST = "broadcast"

# Params
# Pixel rectangle of the clock overlay, measured on a 1920x1080 frame: top-left corner (x0, y0),
# then width/height. Override with --rect if a different video/crop needs a different box.
RECT_X, RECT_Y, RECT_W, RECT_H = 135, 8, 97, 37

# Tesseract is only ever asked for digits and a colon, which is what a clock can contain.
TESS_CONFIG = r"--psm 7 -c tessedit_char_whitelist=0123456789:"

# Common install locations to try when "tesseract" is not already on PATH (mainly a Windows
# problem -- the installer does not always add it).
WINDOWS_TESSERACT_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
)


def configure_tesseract(explicit_cmd=None):
    """Point pytesseract at a tesseract binary, or raise a readable error.

    Import is deferred to here (rather than module scope) so that importing this module for its
    helpers, e.g. parse_time(), never requires pytesseract to be installed.
    """
    import pytesseract

    if explicit_cmd:
        pytesseract.pytesseract.tesseract_cmd = explicit_cmd
        return pytesseract

    if shutil.which("tesseract"):
        return pytesseract

    for candidate in WINDOWS_TESSERACT_CANDIDATES:
        if os.path.isfile(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return pytesseract

    raise RuntimeError(
        "Could not find the Tesseract OCR binary. Install it (see the module docstring) or "
        "pass --tesseract-cmd pointing at tesseract.exe.")


def parse_time(value):
    """Seconds (float), "MM:SS", or "HH:MM:SS" -> seconds (float)."""
    if ":" not in value:
        return float(value)
    parts = [float(part) for part in value.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    hours, minutes, seconds = parts
    return hours * 3600.0 + minutes * 60.0 + seconds


def preprocess(crop):
    """Clock crop -> a binary image Tesseract reads reliably.

    The overlay is bright white digits on a dark background, so a plain threshold after
    grayscaling isolates them; upscaling gives Tesseract more pixels per character to work with.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return cv2.resize(binary, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)


def read_clock(pytesseract_module, frame, rect):
    """OCR the clock rectangle of one frame. Returns the cleaned text, e.g. "01:11"."""
    x, y, w, h = rect
    crop = frame[y:y + h, x:x + w]
    processed = preprocess(crop)
    text = pytesseract_module.image_to_string(processed, config=TESS_CONFIG)
    return "".join(char for char in text if char in "0123456789:").strip()


def iter_clock_readings(video_path, start_time=0.0, end_time=None, step=1, rect=None,
                         tesseract_cmd=None, start_frame=None, end_frame=None,
                         return_frame=False):
    """Yield (frame_number, video_time_seconds, clock_text) from start_time to end_time.

    step > 1 skips frames, which is the usual way to make this fast over a long clip: the clock
    only changes once a second, so sampling every fps/2 frames loses nothing a human would notice.

    start_frame/end_frame, when given, take precedence over start_time/end_time -- useful when
    you already know the exact frame number (e.g. from the ATD) rather than a wall-clock time.

    return_frame=True appends the raw BGR frame as a 4th tuple element, for callers that also
    want to display the video (see --show) rather than just the OCR text.
    """
    pytesseract_module = configure_tesseract(tesseract_cmd)
    rect = rect or (RECT_X, RECT_Y, RECT_W, RECT_H)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video: {}".format(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = start_frame if start_frame is not None else int(round(start_time * fps))
    if end_frame is None:
        end_frame = int(round(end_time * fps)) if end_time is not None else None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_number = start_frame

    try:
        while end_frame is None or frame_number < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            clock = read_clock(pytesseract_module, frame, rect)
            if return_frame:
                yield frame_number, frame_number / fps, clock, frame
            else:
                yield frame_number, frame_number / fps, clock

            frame_number += step
            if step > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    finally:
        cap.release()


def show_frame(frame, rect, clock, window="read_timestamp"):
    """Draw the clock rectangle and the OCR reading on the frame and display it.

    Returns False if the user pressed 'q' or Esc, so the caller's playback loop can stop early.
    """
    x, y, w, h = rect
    annotated = frame.copy()
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(annotated, clock or "?", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow(window, annotated)
    return cv2.waitKey(1) & 0xFF not in (ord("q"), 27)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", help="path to the input video")
    parser.add_argument("--start-time", default="0", type=parse_time,
                        help="where to start reading, as seconds or MM:SS / HH:MM:SS (default: 0)")
    parser.add_argument("--end-time", default=None, type=parse_time,
                        help="where to stop reading, same formats (default: end of video)")
    parser.add_argument("--start-frame", type=int, default=None,
                        help="start at this exact frame number instead of --start-time")
    parser.add_argument("--end-frame", type=int, default=None,
                        help="stop at this exact frame number instead of --end-time")
    parser.add_argument("--step", type=int, default=1,
                        help="read every Nth frame instead of every frame (default: 1)")
    # parser.add_argument("--rect", nargs=4, type=int, metavar=("X0", "Y0", "WIDTH", "HEIGHT"),
    #                     default=[RECT_X, RECT_Y, RECT_W, RECT_H],
    #                     help="clock rectangle in pixels (default: {} {} {} {})".format(
    #                         RECT_X, RECT_Y, RECT_W, RECT_H))
    parser.add_argument("--tesseract-cmd", default=None,
                        help="path to tesseract.exe/tesseract if it is not on PATH")
    parser.add_argument("--output", default=None,
                        help="optional CSV path to also write (frame, video_time, clock)")
    parser.add_argument("--show", action="store_true",
                        help="play the video in a window alongside the OCR readings, in real "
                             "time (paced to the video's own fps rather than OCR speed)")
    args = parser.parse_args()

    video_type = VIDEO_TYPES.TACTICAL if args.video.lower().endswith('tactical.mp4') else VIDEO_TYPES.BROADCAST

    if not os.path.isfile(args.video):
        sys.exit("No such video: {}".format(args.video))

    writer = None
    output_file = None
    if args.output:
        output_file = open(args.output, "w", newline="")
        writer = csv.writer(output_file)
        writer.writerow(["video_type","frame", "video_time", "clock"])

    # rect = tuple(args.rect)
    fps = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS) or 25.0
    frame_budget = args.step / fps  # seconds of video time each loop iteration advances

    rectangle_coords = RECTANGLES.TACTICAL if video_type == VIDEO_TYPES.TACTICAL else RECTANGLES.BROADCAST

    try:
        for reading in iter_clock_readings(
                args.video, args.start_time, args.end_time, args.step,
                rectangle_coords, args.tesseract_cmd, args.start_frame, args.end_frame,
                return_frame=args.show):
            if args.show:
                frame_number, video_time, clock, frame = reading
            else:
                frame_number, video_time, clock = reading

            loop_start = time.monotonic()
            print("frame {:>7}  t={:7.2f}s  clock='{}'".format(frame_number, video_time, clock))
            if writer:
                writer.writerow([video_type, frame_number, "{:.2f}".format(video_time), clock])

            if args.show:
                if not show_frame(frame, rectangle_coords, clock):
                    print("\nStopped by user.")
                    break
                # Sleep off whatever time is left in the frame budget after OCR + printing, so
                # playback tracks the video's own clock instead of running as fast as OCR allows.
                remaining = frame_budget - (time.monotonic() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        if args.show:
            cv2.destroyAllWindows()
        if output_file:
            output_file.close()
            print("\nWrote readings to {}".format(args.output))
