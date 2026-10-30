import csv
from collections import defaultdict


def load_ball_detections(path):
    """Load per-video-frame ball detections (bboxes + score) from a CSV like ball_detection.csv.

    Returns {video_frame: [{"bbox": (x1, y1, x2, y2), "score": float}, ...]}, sorted by
    descending score within each frame.
    """
    detections = defaultdict(list)
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(float(row["frame"]))
            detections[frame].append({
                "bbox": tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2")),
                "score": float(row["score"]),
            })
    for frame in detections:
        detections[frame].sort(key=lambda det: det["score"], reverse=True)
    return detections


def _closest_detection(dets, ref_x, ref_y, gating_distance):
    def distance(det):
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return ((cx - ref_x) ** 2 + (cy - ref_y) ** 2) ** 0.5

    best = min(dets, key=distance)
    return best if distance(best) <= gating_distance else None

