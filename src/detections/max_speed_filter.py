"""Replace ball positions that "teleport" between frames with the last trusted position.

A real ball cannot cover an unbounded pixel distance from one tracked frame to the next; a jump
far beyond what it could plausibly travel in that many frames is a tracking error -- SORT (or its
Kalman filter) locking onto a different blob, a stray high-score false detection -- not a real
ball. This compares each tracked point against the last point it *accepted* and, when the jump is
implausible, discards the new point and reuses that last accepted position instead of trusting an
outlier or leaving a gap.

Speed, not raw distance, is what is compared, because two tracked points are not always on
consecutive frames: a gap of several frames with no point at all (no SORT, no ATD fallback)
stretches the elapsed time, over which a real ball can cover further. `max_speed` is
pixels/frame; the threshold for one step is `max_speed * frame_gap`.

Rejections compound against the *original* last accepted point, not the held one: if a point is
rejected, the frame/position used for the next comparison does not move, so the allowed distance
keeps growing with every further frame until either a point lands close enough to be accepted
again, or the gap alone makes a jump plausible.
"""

import math

DEFAULT_MAX_SPEED = 200.0  # px/frame. ~4x the 99th percentile of real frame-to-frame speed seen
                          # on BAR-ATH (45 px/frame); genuine teleports measured on that data are
                          # 600-1200 px/frame, so there is a wide margin on both sides.


def filter_teleports(points, max_speed=DEFAULT_MAX_SPEED):
    """points: [(frame, x, y), ...] sorted by frame, ascending, x/y None for frames with no
    tracked point (gaps are passed through unchanged, never compared against).

    Returns [(frame, x, y, corrected), ...] in the same order and length; `corrected` is True
    where the original (x, y) was replaced by the last accepted position because the implied
    speed from it exceeded `max_speed` pixels/frame.
    """
    out = []
    last_frame, last_xy = None, None
    for frame, x, y in points:
        if x is None or y is None:
            out.append((frame, x, y, False))
            continue
        if last_xy is None:
            out.append((frame, x, y, False))
            last_frame, last_xy = frame, (x, y)
            continue

        gap = max(frame - last_frame, 1)
        distance = math.hypot(x - last_xy[0], y - last_xy[1])
        if distance > max_speed * gap:
            out.append((frame, last_xy[0], last_xy[1], True))
            # Deliberately do not update last_frame/last_xy: the rejected point carries no
            # information about where the ball really is, so the next comparison is still
            # against the last point we trusted, over a correspondingly larger gap.
        else:
            out.append((frame, x, y, False))
            last_frame, last_xy = frame, (x, y)
    return out
