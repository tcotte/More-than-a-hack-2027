"""Reject ball points that the per-frame homography says are off the pitch.

A detection or a tracked point is only a plausible ball position if unprojecting its pixel
coordinates through that frame's homography (see load_homography.unproject()) lands roughly
inside the pitch rectangle -- normalised (0, 0)-(1, 1). Anything that lands well outside it (the
stands, an advertising board, a camera operator) is not the ball, however confident the detector
was, and should not feed a tracker or sit in a merged output.

`margin` loosens the cutoff by that fraction of the pitch on every side, so a ball a couple of
metres out of play -- a throw-in, a corner -- is not discarded along with genuine false
positives; only points clearly off the pitch are dropped.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(HERE)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import load_homography  # noqa: E402

DEFAULT_MARGIN = 0.05


def on_pitch(matrix, px, py, margin=DEFAULT_MARGIN):
    """True if pixel (px, py), unprojected through `matrix`, falls inside the pitch rectangle
    expanded by `margin` on every side. False if `matrix` is None/NaN (no homography to check
    against) or the point unprojects outside the (expanded) rectangle.
    """
    if matrix is None:
        return False
    x, y = load_homography.unproject(matrix, px, py)
    if math.isnan(x) or math.isnan(y):
        return False
    return (-margin <= x <= 1 + margin) and (-margin <= y <= 1 + margin)


def load_pitch_checker(source="tactical", margin=DEFAULT_MARGIN, data_dir=None):
    """A ready-to-use `is_on_pitch(frame, px, py)` closure over that source's homography, so
    callers do not have to load and index the matrices/valid arrays themselves.

    A frame outside the homography's range, or with no valid matrix for it, is treated as
    off-pitch: there is nothing to check the point against, so it is not trusted.
    """
    matrices, _, valid = load_homography.load(data_dir=data_dir, source=source)

    def is_on_pitch(frame, px, py):
        if frame < 0 or frame >= len(valid) or not valid[frame]:
            return False
        return on_pitch(matrices[frame], px, py, margin)

    return is_on_pitch
