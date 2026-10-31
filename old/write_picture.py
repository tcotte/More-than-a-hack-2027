import cv2
import os

if __name__ == '__main__':
    video_path = r"../src/data/broadcast/BAR-ATH_broadcast.mp4"
    output_folder = "pictures"

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_number = 12355
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        filename = os.path.join(
            output_folder,
            f"frame_{frame_number:05d}.jpg"
        )

        cv2.imwrite(filename, frame)
        frame_number += 1
        break

    cap.release()

    print(f"Saved {frame_number} pictures.")