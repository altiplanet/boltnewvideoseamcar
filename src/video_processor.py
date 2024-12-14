import cv2
import numpy as np
from frame_processor import FrameProcessor

class VideoProcessor:
    def __init__(self, input_path, output_path, target_width, target_height, num_workers=4):
        print("[INFO] Initializing VideoProcessor...")
        print(f"[INFO] Input path: {input_path}, Output path: {output_path}")
        print(f"[INFO] Target dimensions: width={target_width}, height={target_height}")
        print(f"[INFO] Number of workers: {num_workers}")

        self.input_path = input_path
        self.output_path = output_path
        self.target_width = target_width
        self.target_height = target_height
        self.frame_processor = FrameProcessor(num_workers)

    def read_video(self):
        """Read video frames into memory."""
        print("[INFO] Reading video frames...")
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open video: {self.input_path}")
            return [], 0

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[DEBUG] End of video stream reached.")
                break
            frames.append(frame)

        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] Total frames read: {len(frames)}, FPS: {fps}")
        cap.release()
        return frames, fps

    def write_video(self, frames, fps):
        """Write processed frames to output video."""
        if not frames:
            print("[ERROR] No frames to write!")
            return

        print("[INFO] Writing processed frames to video...")
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        for i, frame in enumerate(frames):
            out.write(frame)
            if i % 10 == 0:
                print(f"[DEBUG] Written frame {i}/{len(frames)}")

        out.release()
        print("[INFO] Video writing completed.")

    def process(self):
        """Process the entire video."""
        print("[INFO] Starting video processing...")

        # Read video
        frames, fps = self.read_video()
        if not frames:
            print("[ERROR] No frames read from video. Exiting.")
            return

        # Process width reduction
        if self.target_width < frames[0].shape[1]:
            print(f"[INFO] Reducing width to {self.target_width}...")
            frames = self.frame_processor.process_video(frames, self.target_width)

        # Process height reduction
        if self.target_height < frames[0].shape[0]:
            print(f"[INFO] Reducing height to {self.target_height}...")
            frames = [cv2.transpose(f) for f in frames]
            frames = self.frame_processor.process_video(frames, self.target_height)
            frames = [cv2.transpose(f) for f in frames]

        # Write video
        self.write_video(frames, fps)
        print("[INFO] Video processing completed successfully.")
