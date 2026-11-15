from types import SimpleNamespace

import numpy as np

from detections.ball import _closest_detection
from detections.bytetrack.byte_tracker import BYTETracker


class KalmanPointTracker:
    """Constant-velocity Kalman filter over a single 2D point (SORT's motion model)."""

    def __init__(self, cx, cy):
        self.x = np.array([cx, cy, 0.0, 0.0])
        self.P = np.eye(4) * 10.0
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.Q = np.eye(4) * 1.0
        self.R = np.eye(2) * 5.0
        self.time_since_update = 0
        self.hits = 1

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        return self.x[:2]

    def update(self, cx, cy):
        z = np.array([cx, cy])
        residual = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ residual
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1


def track_ball(detections, frames, references=None, min_score=0.3, gating_distance=80.0,
               max_age=10):
    """SORT-style single-target tracking of the ball, preferring the ATD ground-truth position.

    For each frame in the sorted `frames` range: if `references` gives a ground-truth (yellow)
    ball pixel position for that frame, the detection closest to it (gated by `gating_distance`)
    is used to update the Kalman filter. Otherwise, falls back to the detection closest to the
    filter's own motion prediction, so frames with no ATD ball are still coasted/tracked. Frames
    with no match for up to `max_age` frames are coasted on the prediction; beyond that the track
    is dropped until a new detection re-initialises it.

    Returns {video_frame: {"centre": (cx, cy), "bbox": bbox or None, "matched": bool,
    "score": float or None}}.
    """
    references = references or {}
    tracker = None
    track = {}
    for frame in sorted(frames):
        dets = [det for det in detections.get(frame, []) if det["score"] >= min_score]
        reference = references.get(frame)

        predicted = tracker.predict() if tracker is not None else None
        matched_det = None
        if dets:
            if reference is not None:
                matched_det = _closest_detection(dets, reference[0], reference[1],
                                                 gating_distance)
            elif tracker is not None:
                matched_det = _closest_detection(dets, predicted[0], predicted[1],
                                                 gating_distance)
            else:
                matched_det = max(dets, key=lambda det: det["score"])

        if matched_det is not None:
            x1, y1, x2, y2 = matched_det["bbox"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if tracker is None:
                tracker = KalmanPointTracker(cx, cy)
            else:
                tracker.update(cx, cy)
            track[frame] = {"centre": (cx, cy), "bbox": matched_det["bbox"], "matched": True,
                             "score": matched_det["score"]}
        elif tracker is not None and tracker.time_since_update <= max_age:
            track[frame] = {"centre": tuple(predicted), "bbox": None, "matched": False,
                             "score": None}
        else:
            tracker = None

    return track


def _choose_ball_track(stracks, reference, gating_distance=80.0):
    """Among the BYTETracker outputs on one frame, pick the one STrack that is the ball.

    There can be more than one if a false-positive detection persists long enough to spawn its
    own track id; BYTETracker itself has no notion of which of its tracks matters to us. With an
    ATD reference (ground-truth pixel position) available and close enough, prefer it, same
    gating convention as the rest of this module. Otherwise fall back to the highest-score,
    oldest (lowest track_id) candidate, on the assumption that a spurious track is newer and
    less confident than the real, continuously-detected ball.
    """
    if not stracks:
        return None
    if reference is not None:
        ref_x, ref_y = reference

        def distance(strack):
            x1, y1, x2, y2 = strack.tlbr
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            return ((cx - ref_x) ** 2 + (cy - ref_y) ** 2) ** 0.5

        best = min(stracks, key=distance)
        if distance(best) <= gating_distance:
            return best
    return min(stracks, key=lambda strack: (-strack.score, strack.track_id))


def track_ball_native_bytetrack(detections, frames, frame_size, references=None, track_thresh=0.5,
                                match_thresh=0.8, track_buffer=30, frame_rate=25, min_score=0.0,
                                gating_distance=80.0):
    """Ball tracking with the real BYTETracker (vendored from the ByteTrack submodule at
    detections/bytetrack/), run as a single-target tracker frame by frame.

    `detections` should already be ball-only. `frame_size` is (height, width) of the source
    video; it is passed as both BYTETracker's img_info and img_size so its internal letterbox
    rescaling (`bboxes /= scale`) is a no-op, since our boxes are already in full video-pixel
    coordinates rather than a resized model input. `track_thresh`/`match_thresh`/`track_buffer`
    are BYTETracker's own high-score, IoU-match and lost-track-buffer thresholds (its low-score
    cutoff is a fixed 0.1, hard-coded in byte_tracker.py); `min_score` is an extra floor applied
    before detections even reach BYTETracker, for filtering out garbage near-zero scores. Unlike
    track_ball()/track_ball_bytetrack(), association uses BYTETracker's own IoU + Hungarian
    (lap) matching rather than nearest-centre gating.

    `references`, when given, only break ties between multiple BYTETracker ids on the same
    frame (see _choose_ball_track) -- they do not feed BYTETracker's own association.

    Returns the same shape as the other two trackers: {video_frame: {"centre", "bbox",
    "matched", "score"}}, plus "track_id" naming which BYTETracker id produced the point.
    """
    references = references or {}
    args = SimpleNamespace(track_thresh=track_thresh, track_buffer=track_buffer,
                           match_thresh=match_thresh, mot20=False)
    tracker = BYTETracker(args, frame_rate=frame_rate)
    track = {}
    for frame in sorted(frames):
        dets = [det for det in detections.get(frame, []) if det["score"] >= min_score]
        if dets:
            output_results = np.array([list(det["bbox"]) + [det["score"]] for det in dets],
                                      dtype=np.float64)
        else:
            output_results = np.zeros((0, 5), dtype=np.float64)

        online_targets = tracker.update(output_results, frame_size, frame_size)
        reference = references.get(frame)

        chosen = _choose_ball_track(online_targets, reference, gating_distance)
        if chosen is not None:
            x1, y1, x2, y2 = chosen.tlbr
            track[frame] = {"centre": ((x1 + x2) / 2, (y1 + y2) / 2), "bbox": (x1, y1, x2, y2),
                            "matched": True, "score": float(chosen.score),
                            "track_id": chosen.track_id}
            continue

        # Nothing confirmed this frame: coast on the best lost tracklet's last prediction, so
        # the overlay still follows the ball through a short gap instead of disappearing.
        coasted = _choose_ball_track(tracker.lost_stracks, reference, gating_distance)
        if coasted is not None and tracker.frame_id - coasted.end_frame <= tracker.max_time_lost:
            x1, y1, x2, y2 = coasted.tlbr
            track[frame] = {"centre": ((x1 + x2) / 2, (y1 + y2) / 2), "bbox": None,
                            "matched": False, "score": None, "track_id": coasted.track_id}

    return track