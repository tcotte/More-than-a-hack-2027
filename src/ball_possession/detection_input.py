"""Use the same detection-based ball coordinates for visualization and evaluation."""

import csv
import math
from dataclasses import dataclass, replace
from typing import Mapping, Optional

import numpy as np
from kloppy.domain import Player, Point
from starter import load_homography, load_smart_tagging

from ball_possession.possession_model import is_valid


@dataclass
class PossessionFrame:
    players_coordinates: Mapping[Player, Point]
    ball_coordinates: Optional[Point]


def build_ball_references(frames, by_frame_id, matrices, valid):
    """Project available ATD ball positions into the corresponding video frames."""
    references = {}
    for video_frame in frames:
        if not valid[video_frame]:
            continue
        tracking_frame = by_frame_id.get(video_frame + 1)
        if tracking_frame is None:
            continue
        ball = tracking_frame.ball_coordinates
        if not is_valid(ball):
            continue
        px, py = load_homography.project(matrices[video_frame], ball.x, ball.y)
        references[video_frame] = (float(px), float(py))
    return references


def frame_with_detected_ball(tracking_frame, track_point, matrix):
    """Keep ATD players and use the detection track, or a missing ball if unavailable."""
    if tracking_frame is None:
        return None
    ball = None
    if track_point is not None and matrix is not None:
        x, y = load_homography.unproject(matrix, *track_point["centre"])
        if np.isfinite(x) and np.isfinite(y):
            ball = Point(float(x), float(y))
    return PossessionFrame(tracking_frame.players_coordinates, ball)


def frame_with_team_overrides(tracking_frame, overrides, home_team, away_team):
    """Apply matched-player team corrections without discarding unmatched ATD players."""
    if tracking_frame is None or not overrides:
        return tracking_frame
    players = {}
    for player, position in tracking_frame.players_coordinates.items():
        original_side = "home" if player.team == home_team else "away"
        side = overrides.get((original_side, player.jersey_no))
        if side is not None and side != original_side:
            player = replace(player, team=home_team if side == "home" else away_team)
        players[player] = position
    return PossessionFrame(players, tracking_frame.ball_coordinates)


def possession_frame(tracking_frame, ball_track, video_frame, matrix, fallback_atd_ball=False,
                     overrides=None, home_team=None, away_team=None):
    """Prepare a frame consistently for batch prediction and visualization."""
    if ball_track is not None:
        detected = frame_with_detected_ball(tracking_frame, ball_track.get(video_frame), matrix)
        if not (fallback_atd_ball and detected is not None
                and not is_valid(detected.ball_coordinates)):
            tracking_frame = detected
    return frame_with_team_overrides(tracking_frame, overrides, home_team, away_team)


def load_possession_ball_track(path, start, end):
    """Load usable merged pixel positions; forward-filled positions are not observations."""
    track = {}
    seen = set()
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"frame", "x", "y", "source"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("{}: missing columns {}".format(path, ", ".join(sorted(missing))))
        for row in reader:
            frame = int(row["frame"])
            if not start <= frame <= end:
                continue
            if frame in seen:
                raise ValueError("{}: duplicate frame {}".format(path, frame))
            seen.add(frame)
            source = row["source"]
            if source not in ("yolo_sort", "atd", "ffill", "missing"):
                raise ValueError("{}: unknown ball source {!r} on frame {}".format(
                    path, source, frame))
            if source in ("ffill", "missing"):
                continue
            x, y = float(row["x"]), float(row["y"])
            if not (np.isfinite(x) and np.isfinite(y)):
                raise ValueError("{}: invalid ball position on frame {}".format(path, frame))
            track[frame] = {"centre": (x, y)}
    return track


def smart_attacking_hints(events, start, end, home_name, away_name):
    """Return unambiguous attacking-team hints for the inclusive frame range."""
    names = {home_name.strip().casefold(): "H", away_name.strip().casefold(): "A"}
    attacking = {"build up", "progression", "creating chances", "attacking transition",
                 "long ball", "ball in final third", "ball in the box"}
    candidates = {}
    for event in events:
        if (event["code"] or "").strip().casefold() not in attacking:
            continue
        label = names.get((event["tags"].get("Team") or "").strip().casefold())
        if label is not None:
            for frame in range(max(start, event["start_frame"]), min(end + 1, event["end_frame"])):
                candidates.setdefault(frame, set()).add(label)
    return {frame: next(iter(labels)) for frame, labels in candidates.items() if len(labels) == 1}


def load_smart_attacking_hints(start, end, home_name, away_name):
    events = load_smart_tagging.load()
    if not load_smart_tagging.check_time_base(events)["agrees"]:
        raise ValueError("Smart Tagging timestamps do not match the tactical video")
    return smart_attacking_hints(events, start, end, home_name, away_name)


def player_team_overrides(path, start, end):
    """Load only matched-player team changes; ATD slots remain the player identifiers."""
    overrides = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["video_frame"])
            if not start <= frame <= end:
                continue
            original_side, separator, jersey = row["player_key"].partition("_")
            side = row["side"]
            if not separator or original_side not in ("home", "away") or side not in ("home", "away"):
                raise ValueError("invalid player side or key on video frame %d" % frame)
            key = (original_side, int(jersey))
            if int(row["jersey_no"]) != key[1]:
                raise ValueError("player key and jersey disagree on video frame %d" % frame)
            if side != original_side:
                previous = overrides.setdefault(frame, {}).setdefault(key, side)
                if previous != side:
                    raise ValueError("conflicting player matches on video frame %d" % frame)
    return overrides


def broadcast_missing_ball_frames(correspondence_path, atd_path, start, end):
    """Map missing broadcast balls to tactical frames; absent correspondences stay unknown."""
    broadcast_to_tactical = {}
    tactical_to_broadcast = {}
    with open(correspondence_path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            tactical = int(row["frame_tactical"])
            broadcast = int(row["frame_broadcast"])
            if tactical < 0 or broadcast < 0:
                raise ValueError("correspondence frame numbers must be non-negative")
            if not start <= tactical <= end:
                continue
            previous = tactical_to_broadcast.get(tactical)
            if previous is not None and previous != broadcast:
                raise ValueError("conflicting broadcast frames for tactical frame %d" % tactical)
            tactical_to_broadcast[tactical] = broadcast
            broadcast_to_tactical.setdefault(broadcast + 1, set()).add(tactical)

    missing = set()
    remaining = set(broadcast_to_tactical)
    if not remaining:
        return missing, 0
    with open(atd_path, encoding="utf-8") as handle:
        for line in handle:
            frame_id, _, ball = line.strip().split(":")
            frame_id = int(frame_id)
            if frame_id not in remaining:
                continue
            x, y, _ = ball.split(",")
            if not (math.isfinite(float(x)) and math.isfinite(float(y))):
                missing.update(broadcast_to_tactical[frame_id])
            remaining.remove(frame_id)
            if not remaining:
                break
    if remaining:
        raise ValueError("broadcast ATD is missing %d referenced frame(s)" % len(remaining))
    return missing, len(tactical_to_broadcast)
