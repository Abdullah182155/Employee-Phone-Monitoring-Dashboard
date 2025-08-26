import cv2
import os
import glob

def extract_three_frames_per_second(video_paths, output_dir):
    """
    Extract exactly 3 frames per second from multiple videos
    and save them all in one folder.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_count_total = 0  # unique counter for filenames

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"⚠️ Could not open {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f"🎥 {video_path} | FPS: {fps:.2f} | Duration: {duration:.2f}s")

        # 🎯 3 frames per second → take frames at equal intervals in each second
        frames_per_second = 3
        interval = fps / frames_per_second  # step between frames

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # pick frames based on interval
            if int(frame_num % interval) == 0:
                frame_name = f"frame_{frame_count_total:06d}.jpg"
                cv2.imwrite(os.path.join(output_dir, frame_name), frame)
                frame_count_total += 1

            frame_num += 1

        cap.release()

    print(f"✅ Done! Extracted {frame_count_total} frames into '{output_dir}'.")


if __name__ == "__main__":
    # Example: take all videos in "videos/" folder
    video_folder = "test"
    output_folder = "all_frames"

    video_files = glob.glob(os.path.join(video_folder, "*.mp4"))

    extract_three_frames_per_second(video_files, output_folder)
