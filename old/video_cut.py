import cv2

if __name__ == '__main__':
    input_video = r"C:\Users\Tristan_COTTE\PycharmProjects\Hackathon\more-than-a-hack-2027\data\tactical\BAR-ATH_tactical.mp4"
    output_video = "extracted_video.mp4"

    # Choose the start and end times in seconds
    start_time = 13000   # Start at 10 seconds
    end_time = 13200     # End at 30 seconds

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        raise ValueError("Cannot open input video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



    # Convert seconds to frame positions
    start_frame = start_time
    end_frame = end_time

    # MP4 video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_video,
        fourcc,
        fps,
        (width, height)
    )

    # Move to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_number = start_frame

    while frame_number < end_frame:
        ret, frame = cap.read()

        if not ret:
            break

        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()

    print(f"Video extracted successfully: {output_video}")
    print(f"Duration: {end_time - start_time} seconds")