# KekeLaPlancha - Technical document

## Objective and approach

Our system assigns a label to each frame of the tactical video: `H` for Barcelona possession,
`A` for Athletic Club possession, `D` for a dead ball and `X` for an abstention. The submitted
file covers all 199,319 frames of the match.

Possession is estimated by a rule-based model, without training a possession classifier. It
combines player positions, a precomputed ball trajectory and attacking-phase hints. The aim is
to avoid changing possession immediately whenever a player is merely close to the ball. The
rules and thresholds are implemented in
[`possession_model.py`](src/ball_possession/possession_model.py).

## Data and processing

- **Challenge-provided data**: automatic tactical player and ball tracking, tactical
  homographies and tactical Smart Tagging. Player tracking provides positions and teams; the
  homography converts ball coordinates between the image and the pitch. Processing assumes
  25 frames per second and a 105 × 68 m pitch.
- **Ball trajectory**: the intermediate file `assets/trackings/ball/tracked_ball.csv` contains
  positions labelled `yolo_sort`, `atd`, or `ffill` when a position has been carried forward.
  The possession export uses `yolo_sort` and `atd` positions, and ignores `ffill` and `missing`.
  `yolo_sort` positions may include tracker predictions and therefore do not necessarily
  correspond to direct detections.
- **Smart Tagging**: `--smart-tagging` uses unambiguous attacking phases as team hints. Unknown
  or conflicting tags are neutral. Tags are not copied directly into the output.
- **Not used by this export**: broadcast video, camera correspondence and player-match
  corrections. The corresponding tools in the repository are not called by the submission
  command.

The repository describes a trajectory generation process using ball detection, SORT-style
tracking and ATD fallback, with homography and jump filtering. The associated code is in
[`merge_sort_atd_ball.py`](scripts/ball_tracking/merge_sort_atd_ball.py). **The exact command,
parameters and correspondence with the intermediate file used for this submission have not been
verified**; they need to be confirmed with its producer.

## Detector training data

The possession model itself is not trained. The repository contains a ball detection notebook,
[`train_ball_detector.ipynb`](notebooks/train_ball_detector.ipynb), that uses
[football-ball-detection-rejhg, version 2](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg).
Its logs report 1,966 training images, including 8 backgrounds, and 121 validation images for
one YOLO26s run. It also contains tiled inference with SAHI. According to the team, this dataset
does not contain data from the provided match.

The identity of the annotators and the annotation method for this dataset are not documented in
the repository. The exact relationship between the model from this notebook, the match detections
and the trajectory used for this submission remains to be confirmed; the figures above describe
the notebook and do not by themselves establish that relationship.

## Possession rules

The model starts with `A` in possession. A probable touch is attributed to the nearest valid
player within 3 m of the ball when the ball's speed changes by at least 0.1 m/frame or its
direction changes by at least 20°. For a direction change, both displacements must be at least
0.1 m/frame.

A touch attributed to the opposing team starts a pending change: a player from that team must
remain within 3 m of the ball for five consecutive frames. The model does not require that
player to be closer than every player from the current team. If an unambiguous Smart Tagging
hint indicates the other team at confirmation time, five additional frames are required. Team
changes are separated by at least 50 frames. The output changes when the change is confirmed;
earlier frames are not rewritten.

When the ball is invalid or missing, the current possession is retained and pending confirmation
is cancelled. The last known team therefore retains possession during loose balls or passes until
an opposing change is confirmed or a dead ball is detected.

## Dead balls and restarts

The model predicts `D` in any of these cases:

1. 50 observed ball positions fit within a diameter of 1 m; up to three consecutive missing
   observations are tolerated but do not count as observed positions.
2. The ball is observed more than 1 m outside the pitch for three consecutive frames.
3. For 25 frames, at least 18 players are tracked and at most one reaches 8 km/h.

For the first two rules, the model resumes the last possession when the ball returns inside the
pitch and moves more than 1 m from the stoppage position, unless the slow-player signal maintains
the stoppage. For the third, it resumes when at least two players reach 8 km/h. A dead-ball
transition is not delayed by the minimum interval between team changes.

## Limitations and evaluation

Proximity is not proof of control, and tracking or homography errors can create or miss a touch.
The dead-ball rules can confuse slow play with a stoppage; the minimum interval between changes
can delay a genuine recovery. The model starts with `A` and does not automatically reset its
state at half-time.

On the complete generated file, local evaluation on the five public minutes gives a Spell F1 of
**0.3724** and a Macro F1 of **0.7584**. These are not full-match scores: the full-match ground
truth is private. The public window is too short to establish performance over the whole match.

One known error concerns the period before the second-half kickoff: the FAQ states that frames
110,808 through 113,741 are dead balls, but the file predicts only 1,485 of the 2,934 as `D`.
The format validator does not detect this kind of error.

Versions were compared using the public scores, and the present version was subsequently chosen.
The rules reserve those five minutes for pipeline checks, not for choosing or tuning a version.
**The compliance of this process must be clarified with the organisers before submission.**

## Reproduction and validation

Requirements: Python 3.12 or later, dependencies from [`pyproject.toml`](pyproject.toml), the
challenge's tactical data and the intermediate ball track at the path above. From the repository
root in PowerShell:

```powershell
$env:MTAH_DATA_DIR = (Resolve-Path .\src\data).Path
.\.venv\Scripts\python.exe src\ball_possession\predict_possession.py --smart-tagging --name KekeLaPlancha --out evaluation\results
.\.venv\Scripts\python.exe mtah_challenge\evaluation\validate_submission.py evaluation\results\KekeLaPlancha.csv
```

This command reproduced the submitted file exactly when the same intermediate file and local
data were available. SHA-256 of the ball track used:
`2c6d87c17d3edd22d0f9be0a80b9c117b1403878e23465620eb9a5dacd7edbfb`.
SHA-256 of the resulting possession CSV:
`a0d61caaca3a92331da9d162ee2114c1e1561dfcc2e6c1d1213b3bcf0b015270`.

This check does not yet demonstrate reproduction from the provided data alone: the exact
generation of the intermediate track and organiser access to all components still need to be
confirmed. The rules prohibit publishing match data or derived data that could reconstruct it;
the intermediate track is currently tracked by Git and must not be published without checking
the applicable rights.
