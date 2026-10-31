"""Clean the OCR'd clock column of the broadcast/tactical timestamp CSVs and match frames.

read_timestamp.py OCRs the on-screen "MM:SS" match clock once per frame and writes it, as-is, to
assets/frame_timestamps/{broadcast,tactical}.csv. Tesseract is not perfect: a sizeable minority of
rows hold garbage ("::", ":24:", "27313", ...) instead of a clean timecode. This module

    1. parses/validates the clock column of each CSV, discarding anything that is not a
       plausible "MM:SS" reading (clean_clock_column / load_timestamps),
    2. reconstructs a frame-accurate `logical_clock` for each CSV (add_logical_clock) -- this
       anchors on the first second whose clock was cleanly read on a full run of frames (one
       game-clock second spans ~25 frames at 25fps, so a single stray reading does not count),
       then counts frames from there, which keeps advancing correctly through a broadcast replay
       (where the raw on-screen clock briefly shows an earlier/later moment instead of the live
       one) and has centisecond resolution instead of the raw clock's whole seconds,
    3. matches broadcast frames to tactical frames with the closest `logical_clock_seconds`
       (build_frame_correspondence), producing a frame-to-frame correspondence table between the
       two video sources.

Run as a script to read both CSVs under assets/frame_timestamps and write the correspondence next
to them:

    python src/merge_frame_timestamps.py
"""

import argparse
import os
import re

import pandas as pd

# Params
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "assets", "frame_timestamps")
BROADCAST_CSV = os.path.join(DEFAULT_DIR, "broadcast_logical_clock.csv")
TACTICAL_CSV = os.path.join(DEFAULT_DIR, "tactical_logical_clock.csv")
CORRESPONDENCE_CSV = os.path.join(DEFAULT_DIR, "frame_correspondence.csv")

# A clean reading is either "MM:SS" (colon kept by Tesseract) or "MMSS" (colon dropped/merged).
# Minutes are capped generously (extra time can push a half past 45/90) and seconds must be a
# real 0-59 value -- that alone rejects the bulk of the OCR noise ("::", ":3", "27313", ...).
_COLON_RE = re.compile(r"^(\d{1,3}):(\d{2})$")
_NO_COLON_RE = re.compile(r"^(\d{2})(\d{2})$")
MAX_MINUTES = 130

# The strict "DD:DD" shape (colon kept, exactly two digits each side) is the cleanest possible
# OCR reading -- used by add_logical_clock to anchor the countdown, as opposed to the looser
# shapes _COLON_RE/_NO_COLON_RE also accept (3-digit minutes, colon dropped).
_STRICT_MM_SS_RE = re.compile(r"^\d{2}:\d{2}$")

FPS = 25  # the clock overlay changes once a second; a trustworthy reading spans this many frames


def parse_clock(raw):
    """OCR clock text -> total seconds (int), or None if it is not a plausible "MM:SS" reading."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    match = _COLON_RE.match(text) or _NO_COLON_RE.match(text)
    if not match:
        return None

    minutes, seconds = int(match.group(1)), int(match.group(2))
    if seconds > 59 or minutes > MAX_MINUTES:
        return None
    return minutes * 60 + seconds


def clean_clock_column(df):
    """Add a `clock_seconds` column (float, NaN where unparsable) parsed from `clock`."""
    df = df.copy()
    df["clock_seconds"] = df["clock"].map(parse_clock).astype("Int64")
    return df


def flag_live_frames(df, tolerance=1.5):
    """Flag which rows show a clock that tracks real time, vs. a replay (or bad OCR).

    In a live frame, `clock_seconds - video_time` is (up to a 1-second rounding wobble, since
    the clock only has whole-second resolution) a constant: the match clock and the video's own
    timeline advance in lockstep. The broadcast feed breaks that whenever it cuts to a replay --
    the overlay clock then reads an earlier (or, once the replay ends, a later) moment, so that
    offset jumps away from its usual value for the duration of the cutaway. The tactical feed is
    a single continuous camera and essentially never does this (see module docstring), but the
    check is harmless to run on it too, where it mostly just rejects stray OCR noise.

    Adds `is_live` (True for rows whose offset sits within `tolerance` seconds of the dominant/
    most common offset in this file) and `is_replay` (clock parsed fine but the offset is off --
    i.e. a replay or a bad OCR reading, as opposed to no reading at all).
    """
    df = df.copy()
    has_clock = df["clock_seconds"].notna()
    offset = df["clock_seconds"].astype(float) - df["video_time"]

    if has_clock.any():
        # The dominant offset across the whole file is the live one: a replay is, by nature,
        # a small fraction of the frames, so it never outweighs the live majority.
        dominant_offset = offset[has_clock].round().mode().iloc[0]
        is_live = has_clock & ((offset - dominant_offset).abs() <= tolerance)
    else:
        is_live = pd.Series(False, index=df.index)

    df["is_live"] = is_live
    df["is_replay"] = has_clock & ~is_live
    return df


def seconds_to_timecode(seconds):
    """Seconds (float) -> "MM:SS:CS" (minutes:seconds:centiseconds), zero-padded."""
    total_centiseconds = int(round(seconds * 100))
    minutes, remainder = divmod(total_centiseconds, 100 * 60)
    secs, centis = divmod(remainder, 100)
    return "{:02d}:{:02d}:{:02d}".format(minutes, secs, centis)


def timecode_to_seconds(text):
    """Inverse of seconds_to_timecode: "MM:SS:CS" -> seconds (float). None for missing/NA."""
    if pd.isna(text):
        return None
    minutes, secs, centis = (int(part) for part in str(text).split(":"))
    return minutes * 60.0 + secs + centis / 100.0


def _find_sustained_reading(df, is_strict, min_run_length):
    """First (frame, clock_seconds) of a run of >= min_run_length consecutive frames that all
    share the same strict "DD:DD" reading -- i.e. a game-clock second that was actually read
    throughout, not just caught on one lucky/noisy frame. Returns (None, None) if there is none.

    A run requires every frame in it to be both strictly read and numbered one after another
    (frame deltas of exactly 1); any gap, OCR failure, or value change starts a new run.
    """
    strict_value = df["clock_seconds"].where(is_strict)
    same_as_previous = (strict_value == strict_value.shift()) & (df["frame"].diff() == 1)
    run_id = (~same_as_previous.fillna(False)).cumsum()

    run_sizes = strict_value.notna().groupby(run_id).transform("sum")
    candidates = df.index[strict_value.notna() & (run_sizes >= min_run_length)]
    if len(candidates) == 0:
        return None, None

    first_index = run_id.loc[candidates].idxmin()
    # idxmin above returns the run_id's index label with the smallest run_id value, which is the
    # first row of the first qualifying run (run ids increase monotonically with frame order).
    run_start = candidates[run_id.loc[candidates] == run_id.loc[first_index]].min()
    return df.loc[run_start, "frame"], float(strict_value.loc[run_start])


def add_logical_clock(df, step_seconds=0.04, min_run_length=FPS):
    """Add `logical_clock`: a synthetic, OCR-noise-free timecode derived from frame count.

    The OCR'd `clock` column only has whole-second resolution and is noisy; this instead starts
    counting from the first game-clock second that was read cleanly ("DD:DD", colon kept, two
    digits each side) for at least `min_run_length` consecutive frames in a row (default 25, one
    full second at 25fps) -- a single strict-looking reading surrounded by noise is not trusted
    as the anchor. From there it adds `step_seconds` (one frame at 25fps = 0.04s) per frame,
    through the last frame with a clean "DD:DD" reading. Frames outside that span -- before the
    clock is first sustainedly read or after it stops being read -- get NA. Requires `df` to
    already have `clock_seconds` (see clean_clock_column/load_timestamps) and be sorted by a
    contiguous `frame` column.
    """
    df = df.copy()
    is_strict = df["clock"].map(lambda text: bool(_STRICT_MM_SS_RE.match(text.strip()))
                                 if isinstance(text, str) else False)
    valid = is_strict & df["clock_seconds"].notna()
    df["logical_clock"] = pd.NA
    if not valid.any():
        df["logical_clock_seconds"] = pd.NA
        return df

    first_frame, start_seconds = _find_sustained_reading(df, is_strict, min_run_length)
    if first_frame is None:
        df["logical_clock_seconds"] = pd.NA
        return df

    last_frame = df.loc[valid, "frame"].iloc[-1]

    in_range = (df["frame"] >= first_frame) & (df["frame"] <= last_frame)
    elapsed = start_seconds + (df.loc[in_range, "frame"] - first_frame) * step_seconds
    df.loc[in_range, "logical_clock"] = elapsed.map(seconds_to_timecode)
    df["logical_clock_seconds"] = df["logical_clock"].map(timecode_to_seconds)
    return df


def load_timestamps(csv_path):
    """Read one frame_timestamps CSV, clean its `clock` column, and flag replay/noise frames.

    Returns the dataframe with video_type/frame/video_time/clock plus `clock_seconds`, `is_live`,
    `is_replay`, `logical_clock` and `logical_clock_seconds` (see clean_clock_column /
    flag_live_frames / add_logical_clock), sorted by frame. Rows with an unreadable or
    untrustworthy clock are kept (not dropped), so callers can still see/use every frame if they
    want to.
    """
    df = pd.read_csv(csv_path)
    df = clean_clock_column(df)
    df = flag_live_frames(df)
    df = add_logical_clock(df)
    return df.sort_values("frame").reset_index(drop=True)


def build_frame_correspondence(broadcast_df, tactical_df, tolerance_seconds=0.02):
    """Match each broadcast frame to the closest tactical frame showing the same logical clock.

    Unlike the raw OCR `clock` (whole-second resolution, and wrong during a broadcast replay --
    see module docstring), `logical_clock_seconds` reconstructs real elapsed match time from the
    frame count, so it keeps advancing correctly through replays and lets frames be matched to
    within one centisecond instead of one second. This does a nearest-match join (pd.merge_asof)
    on that column: for every broadcast frame, the tactical frame whose logical clock is closest,
    as long as it is within `tolerance_seconds` (default 0.02s, i.e. within half a tactical frame
    at 25fps) -- broadcast frames with no tactical frame that close are dropped.
    """
    broadcast_valid = broadcast_df.dropna(subset=["logical_clock_seconds"]) \
        .sort_values("logical_clock_seconds")
    tactical_valid = tactical_df.dropna(subset=["logical_clock_seconds"]) \
        .sort_values("logical_clock_seconds")

    merged = pd.merge_asof(
        broadcast_valid, tactical_valid, on="logical_clock_seconds", direction="nearest",
        tolerance=tolerance_seconds, suffixes=("_broadcast", "_tactical"),
    )
    merged = merged.dropna(subset=["frame_tactical"]).reset_index(drop=True)
    return merged[[
        "logical_clock_seconds", "frame_broadcast", "video_time_broadcast",
        "frame_tactical", "video_time_tactical",
    ]]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--broadcast-csv", default=BROADCAST_CSV)
    parser.add_argument("--tactical-csv", default=TACTICAL_CSV)
    parser.add_argument("--output", default=CORRESPONDENCE_CSV)
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="max seconds between matched logical clocks (default: 0.02)")
    args = parser.parse_args()

    broadcast_df = load_timestamps(args.broadcast_csv)
    tactical_df = load_timestamps(args.tactical_csv)

    print("broadcast: {} frames, {} with a readable clock, {} with a logical clock".format(
        len(broadcast_df), broadcast_df["clock_seconds"].notna().sum(),
        broadcast_df["logical_clock_seconds"].notna().sum()))
    print("tactical:  {} frames, {} with a readable clock, {} with a logical clock".format(
        len(tactical_df), tactical_df["clock_seconds"].notna().sum(),
        tactical_df["logical_clock_seconds"].notna().sum()))

    correspondence = build_frame_correspondence(
        broadcast_df, tactical_df, tolerance_seconds=args.tolerance)
    correspondence.to_csv(args.output, index=False)
    print("Wrote {} matched frame pairs to {}".format(len(correspondence), args.output))


if __name__ == "__main__":
    main()
