"""Write a copy of a frame_timestamps CSV with an added `logical_clock` column.

See merge_frame_timestamps.add_logical_clock for what the column means. The match clock resets
to 00:00 at the start of the second half, so a single anchor/extrapolation over the whole file
would keep counting up through halftime instead of restarting. To handle that, the frames are
split into a first half and a second half (by row count) and `add_logical_clock` is run
independently on each half, each finding its own sustained-run anchor and extrapolating only
within its half. Usage:

    python src/add_logical_clock.py assets/frame_timestamps/tactical.csv
"""

import argparse
import os

import pandas as pd

from merge_frame_timestamps import add_logical_clock, load_timestamps


def add_logical_clock_per_half(df):
    """Run add_logical_clock separately on the first and second half of `df` (by row count).

    Splitting avoids a single logical-clock anchor/extrapolation spanning both halves of the
    match, which would keep counting up through halftime instead of restarting at 00:00 for the
    second half.
    """
    midpoint = len(df) // 2
    first_half = add_logical_clock(df.iloc[:midpoint].reset_index(drop=True))
    second_half = add_logical_clock(df.iloc[midpoint:].reset_index(drop=True))
    return pd.concat([first_half, second_half], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv_path", help="frame_timestamps CSV to read (e.g. tactical.csv)")
    parser.add_argument("--output", default=None,
                        help="output CSV path (default: <input>_logical_clock.csv)")
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        root, ext = os.path.splitext(args.csv_path)
        output_path = root + "_logical_clock" + ext

    df = load_timestamps(args.csv_path)
    df = add_logical_clock_per_half(df)
    df[["video_type", "frame", "video_time", "clock", "logical_clock"]].to_csv(
        output_path, index=False)
    print("Wrote {} rows to {}".format(len(df), output_path))


if __name__ == "__main__":
    main()
