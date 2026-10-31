Steps to reproduce the multi view computation:
1. Use the notebook *notebooks/ORC_timestamp.ipynb* on *broadcast* and *tactical* videos to detect the countdown for 
both videos (this notebook uses *scripts/timestamps/countdown_detections.py* script). These countdowns will be stored 
as csv files which reference all the frame numbers and the related countdown detected by OCR.
2. Create logic_data files for both .csv to match the number of frames per seconds 
(using *scripts/timestamps/add_logical_clock.py* script). These logic_data files will be used to match the frames 
of both videos.
3. Use this logic_data csv to match the frames of both videos (using *scripts/timestamps/merge_frame_timestamps.py* script). 
This will create a new csv file with the frame numbers of both videos and the related countdown.