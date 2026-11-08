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


## Ideas 

- Try to use the ball detection on the broadcast video to see if it adds accuracy.
- Try to exploit the smart tagging provided in *data* folder.
- Try to exploit at the maximum the multi-view visualization to get more precise information about the ball position and 
players positions.
- Use LLMs to analyze the multi-view visualization and extract information about the ball position and players positions.
- Use VLM (maybe Soccer CLIP) to get more information about some events in the multi-view visualization 
(e.g. offside, fouls, etc...).
- Try BoT SORT tracking algorithm.


## SoccerNet

It seems that the SoccerNet dataset is a good source of data to train a model to detect the ball and players in soccer 
videos:
```
!git clone {SoccerNet repository url}
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="./SoccerNet")
mySoccerNetDownloader.downloadDataTask(task="tracking", split=["training", "test"])
```
[source] https://github.com/francescopiocirillo/soccer-players-tracking/blob/main/scripts/0%20-%20dataset_download.py


## SOTA

[1] Graph CNNs https://arxiv.org/pdf/2211.02125
[2] Soccer CLIP https://ieeexplore.ieee.org/document/10916659
[3] Football possession estimation https://www.mdpi.com/1424-8220/26/4/1252
[4] Agentic Multi-View Reasoning in Sports Video Understanding https://arxiv.org/html/2607.11844
[5] Research on Multi-player Tracking in Soccer Videos by Combining Tracking-Prediction-Detection 
https://www.researchgate.net/publication/394906183_Research_on_Multi-player_Tracking_in_Soccer_Videos_by_Combining_Tracking-Prediction-Detection
