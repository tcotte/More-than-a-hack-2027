"""Write a copy of a frame_timestamps CSV with an added `logical_clock` column.

See merge_frame_timestamps.add_logical_clock for what the column means. Usage:

    python src/add_logical_clock.py assets/frame_timestamps/tactical.csv
"""

import argparse
import os

from merge_frame_timestamps import add_logical_clock, load_timestamps


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
    df = add_logical_clock(df)
    df[["video_type", "frame", "video_time", "clock", "logical_clock"]].to_csv(
        output_path, index=False)
    print("Wrote {} rows to {}".format(len(df), output_path))


if __name__ == "__main__":
    main()
