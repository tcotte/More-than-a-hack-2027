"""Detect whether an ATD player position falls inside a penalty box.

Penalty box geometry (depth/width) comes straight from kloppy's EPTS metadata
(dataset.metadata.pitch_dimensions), normalised 0-1 the same way ATD positions are, so this
needs no pitch-size assumptions of its own.

determine_team_sides() tells you which goal (x=0, "left", or x=1, "right") a team defends. The
ATD data carries no explicit per-team attacking-direction attribute, but a settled team spends a
half mostly on its own side of the pitch -- its goalkeeper especially so -- which makes the mean
x position over any non-trivial sample of frames a reliable tell.

team_sides_for_video_frame() is the ground-truth alternative: teams swap ends at halftime, and
for BAR-ATH the sides are known (confirmed against the tracking) rather than guessed --
video frames 11185..90651 (1st half) have the home team defending the "left" box, and video
frames 110808..190197 (2nd half) have it defending "right". See more-than-a-hack-2027/DATA.md
and games.half_video_blocks() for the frame boundaries themselves.
"""

from collections import defaultdict

import numpy as np

import games

# Which side the home team defends, by half. The away team always defends the other one.
HOME_SIDE_BY_HALF = {1: "left", 2: "right"}


def team_sides_for_video_frame(video_frame, home_team, away_team, game=games.GAME):
    """{team: "left"/"right"} for the half `video_frame` falls in, using the known kickoff sides.

    Unlike determine_team_sides() (a statistical guess from where each team's players spend
    their time), this is ground truth for which half a frame belongs to (see
    games.half_of_video_frame()) -- the two halves of a match have fixed, known sides, and they
    swap at halftime. Returns None for a video frame outside both halves (before kickoff,
    half-time, after the final whistle), where there is nothing to report.
    """
    half = games.half_of_video_frame(game, video_frame)
    if half is None:
        return None
    home_side = HOME_SIDE_BY_HALF[half]
    away_side = "right" if home_side == "left" else "left"
    return {home_team: home_side, away_team: away_side}


def determine_team_sides(dataset, frame_range=None):
    """{team: "left" or "right"} -- the goal (x=0 / x=1) each team defends.

    frame_range, if given, restricts the sample to dataset.frames[frame_range] (a slice);
    otherwise every loaded frame is used. Needs at least two teams with tracked positions.

    This is a statistical guess (see module docstring); prefer team_sides_for_video_frame() when
    you know the game's half boundaries, since it is exact and does not depend on how settled the
    teams' shapes happen to be in the sampled frames.
    """
    frames = dataset.frames[frame_range] if frame_range is not None else dataset.frames

    totals = defaultdict(float)
    counts = defaultdict(int)
    for frame in frames:
        for player, position in frame.players_coordinates.items():
            if position is None or position.x is None or np.isnan(position.x):
                continue
            totals[player.team] += position.x
            counts[player.team] += 1

    means = {team: totals[team] / counts[team] for team in totals if counts[team]}
    if len(means) < 2:
        raise ValueError("not enough tracked positions to tell the teams' sides apart")

    ordered = sorted(means, key=means.get)
    return {ordered[0]: "left", ordered[-1]: "right"}


def penalty_box_bounds(side, pitch_dimensions):
    """(x_min, x_max, y_min, y_max), normalised 0-1, of the "left" (x=0) or "right" (x=1) box."""
    depth = pitch_dimensions.penalty_area_length
    width = pitch_dimensions.penalty_area_width
    y_min, y_max = 0.5 - width / 2, 0.5 + width / 2
    if side == "left":
        return 0.0, depth, y_min, y_max
    if side == "right":
        return 1.0 - depth, 1.0, y_min, y_max
    raise ValueError('side must be "left" or "right", got {!r}'.format(side))


def in_penalty_box(position, side, pitch_dimensions):
    """Whether `position` (normalised ATD coordinates) is inside the "left"/"right" box."""
    if position is None or position.x is None or np.isnan(position.x):
        return False
    x_min, x_max, y_min, y_max = penalty_box_bounds(side, pitch_dimensions)
    return x_min <= position.x <= x_max and y_min <= position.y <= y_max


def penalty_box_zone(player, position, team_sides, pitch_dimensions):
    """"own", "opponent" or None -- which box `position` is in, relative to `player`'s team.

    team_sides is the {team: "left"/"right"} mapping from determine_team_sides().
    """
    own_side = team_sides.get(player.team)
    if own_side is None:
        return None
    opponent_side = "right" if own_side == "left" else "left"
    if in_penalty_box(position, own_side, pitch_dimensions):
        return "own"
    if in_penalty_box(position, opponent_side, pitch_dimensions):
        return "opponent"
    return None


if __name__ == "__main__":
    import argparse
    import os
    import sys

    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE)

    import games  # noqa: E402
    from kloppy import metrica  # noqa: E402

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=30000, help="ATD frames to load")
    parser.add_argument("--video-type", default="tactical")
    args = parser.parse_args()

    paths = games.asset_paths(source=args.video_type)
    games.require(["atd_meta", "atd_raw"], source=args.video_type)
    print("loading %d ATD frames..." % args.limit)
    dataset = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"],
                                         limit=args.limit)

    team_sides = determine_team_sides(dataset)
    for team, side in team_sides.items():
        print("{}: defends the {} box".format(team.name, side))

    pitch_dimensions = dataset.metadata.pitch_dimensions
    counts = defaultdict(int)
    for frame in dataset.frames:
        for player, position in frame.players_coordinates.items():
            zone = penalty_box_zone(player, position, team_sides, pitch_dimensions)
            if zone is not None:
                counts[(player.team.name, player.jersey_no, zone)] += 1

    print("players seen in a penalty box (frames, zone):")
    for (team_name, jersey_no, zone), n in sorted(counts.items(), key=lambda kv: -kv[1])[:20]:
        print("  {} #{}: {} frames in {} box".format(team_name, jersey_no, n, zone))
