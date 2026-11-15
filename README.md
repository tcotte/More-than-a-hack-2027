## Commands

The visualizer uses `games.py` and the homography utilities from `mtah_challenge`. Match data
defaults to `mtah_challenge\data`; if your data is in `src\data`, set the following in PowerShell
from the repository root:

```powershell
$env:MTAH_DATA_DIR = (Resolve-Path .\src\data).Path
.\.venv\Scripts\python.exe src\visualize_video_frames.py --clip 23000 30000 --play --video-type tactical --detections "assets\detections\ball_detection_first_half_yolo26_tactical_sliced.csv" --ball-birdseye --possession-ball-source detections
```

The output is written to `frames_out` by default. `--possession-ball-source detections` uses
the tracked CSV ball position for possession and touch detection, with ATD player positions.
The tracker uses the ATD ball to select between candidate detections when available and can
extrapolate for up to `--max-age` frames. When the track or homography is missing, the model
treats the ball as missing instead of falling back to ATD. Without this option, possession
continues to use the ATD ball.

### Evaluate possession with tactical detections

From Git Bash at the repository root, with the match data in `src/data`:

```bash
export MTAH_DATA_DIR="$(pwd -W)/src/data"
./.venv/Scripts/python.exe src/ball_possession/predict_possession.py --start 11185 --end 18684 --detections "assets/detections/ball_detection_first_half_yolo26_tactical_sliced.csv" --name possession_yolo_tactical_public
./.venv/Scripts/python.exe mtah_challenge/evaluation/run_evaluation.py --results evaluation/results --out evaluation
```

The public ground truth covers frames 11185–18684; the detections begin at 11190.
The submission has 199319 rows, with `X` outside the selected interval, and is suitable only
for this public-window test, not a full-match submission. The ball track uses the same ATD-guided
association and default tracker parameters as the visualizer. The score is a pipeline check, not
a reliable full-match ranking; see `mtah_challenge/EVALUATION.md`.

Multi-view command (visualize both videos with ball track and Smart Tagging banners): python.exe multi_view.py --broadcast-video "assets\video\broadcast.mp4" --tactical-video "assets\video\tactical.mp4" --correspondence-csv "assets\frame_correspondence.csv" --detections "assets\ball_detection_first_half_yolo26_sliced.csv" --play --ball-birdseye


# More Than a Hack 2027

This repository contains the code and data for the More Than a Hack 2027 evaluation. The evaluation is based on a single 
match between Barcelona and Athletic Bilbao, and the goal is to track the ball and players in both tactical and broadcast 
videos.

## Execution before submission

### Ball detection
1. Train YOLO ball detection model to detect the ball (→ notebook `notebooks/train_yolo_ball_detection.ipynb`). 
The model is trained on a public dataset from Roboflow (*football-ball-detection-rejhg*).
2. Use the sliced YOLO ball detection model to detect the ball in the tactical video and save the results in a CSV file.
Image are sliced in 2x2 tiles perform the detection. The model used is a YOLO26s.
3. Use the same YOLO ball detection model to detect the ball in the broadcast video and save the results in a CSV file.
4. Use scripts/ball_tracking/merge_sort_atd_ball.py building one merged ball-position CSV per video frame by combining two sources, preferring detections:
- Primary: YOLO + SORT tracker — ball detections are run through a SORT-style tracker (track_ball, no ATD reference), giving a point per frame for real detections or short Kalman-coasted gaps (--max-age). Points that unproject off the pitch (stands, crowd) are discarded.
- Fallback: ATD ground truth — for frames SORT couldn't cover, the Metrica ATD ball position is projected into pixels via the per-frame homography.
- If neither source has a point, the row is "missing" (or forward-filled with --ffill-missing).
- A teleport filter then replaces any frame-to-frame jump faster than --max-speed px/frame with the last accepted position, regardless of source.
A teleport filter then replaces any frame-to-frame jump faster than --max-speed px/frame with the last accepted position, regardless of source.
Output CSV columns: frame, x, y, source (yolo_sort/atd/ffill/missing), matched, score, bbox, teleport_corrected.
```
python scripts/ball_tracking/merge_sort_atd_ball.py 
--detections "~\PycharmProjects\Hackathon\assets\detections\ball\full_match_tactical_sliced.csv"
--out "~\PycharmProjects\Hackathon\assets\trackings\ball\tracked_ball.csv"
--min-detection-score 0.8
--ffill-missing
--max-age
3
--video-type
tactical
```
→ Using the *ffill* parameter, the script will always forward-fill missing ball positions. Therefore, even if the ball
is not detected in a frame, the last known position will be used. This is important for maintaining continuity in the 
ball tracking data, but it may introduce inaccuracies if the ball is not visible for extended periods. 

### Player detection
1. Train YOLO player detection model to detect players (→ notebook `notebooks/train_yolo_player_detection.ipynb`). 
The model is trained on a public dataset from Roboflow (*football-players-detection-3zvbc*).
2. Use the trained YOLO player detection model to detect players (execution on notebook 
`notebooks/train_yolo_player_detection.ipynb`)  in the tactical video and save the results in a CSV 
file (`assets/detections/players/full_match_players_tactical.csv`).
3. Use Bytetrack tracker (script `scripts/players_detection/track_with_bytetrack.py` with default parameters) on the 
resulted csv file to track players in the tactical video and save the results in another CSV file 
(`assets/trackings/players/full_match_players_tactical.csv`).
4. Compare the resulted CSV file with the ATD ground truth CSV file to remove the false positive detections and assign
the correct team for each goalkeeper:
- Matches them to YOLO-detected boxes (nearest-box or Hungarian 1-to-1 assignment); unmatched players are skipped.
- Detects the goalkeeper by checking which matched player falls inside a team's penalty box 
(pitch_zones.penalty_box_bounds), picking the one closest to the goal line if multiple; overrides that player's team 
accordingly (ATD team labels aren't always reliable for keepers).

# TODO define script for that 
→ `scripts/players_detection/export_player_matches_csv.py`
and verification `scripts/players_detection/visualize_player_matches_csv.py`

### Frames recognition
1. Define rectangles where the countdown is located in both videos (broadcast and tactical).
2. Use Tesseract OCR to detect the countdown in each frame of both videos. The results will be stored in two CSV files, 
one for each video.
3. Use the script `scripts/timestamps/add_logical_clock.py` to create logic_data files for both CSVs to match the 
number of frames per second. These logic_data files will be used to match the frames of both videos.
4. Use the script `scripts/timestamps/merge_frame_timestamps.py` to match the frames of both videos using the logic_data 
CSVs. This will create a new CSV file with the frame numbers of both videos and the related (
file *assets/frame_timestamps/frame_correspondence.csv*)

## Used datasets
- Player detection: Roboflow public dataset → *football-players-detection-3zvbc*
- Ball detection: Roboflow public dataset → *football-ball-detection-rejhg*

## Model architectures
- Player detection: YOLOv26s
- Ball detection: YOLOv26s + SAHI
- Frame OCR: Tesseract OCR