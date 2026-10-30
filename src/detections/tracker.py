import numpy as np

from detections.ball import _closest_detection


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