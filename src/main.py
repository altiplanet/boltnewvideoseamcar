import argparse
from video_processor import VideoProcessor
import time

def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated video seam carving")
    parser.add_argument("input_path", help="Path to input video")
    parser.add_argument("output_path", help="Path to output video")
    parser.add_argument("target_width", type=int, help="Target width")
    parser.add_argument("target_height", type=int, help="Target height")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    
    args = parser.parse_args()
    
    print("[INFO] Arguments parsed successfully.")
    print(f"[DEBUG] Input Path: {args.input_path}")
    print(f"[DEBUG] Output Path: {args.output_path}")
    print(f"[DEBUG] Target Width: {args.target_width}")
    print(f"[DEBUG] Target Height: {args.target_height}")
    print(f"[DEBUG] Number of Workers: {args.workers}")

    print("[INFO] Initializing VideoProcessor.")
    start_time = time.time()

    processor = VideoProcessor(
        args.input_path,
        args.output_path,
        args.target_width,
        args.target_height,
        args.workers
    )

    print("[INFO] Starting video processing.")
    processor.process()

    elapsed_time = time.time() - start_time
    print(f"[INFO] Video processing completed in {elapsed_time:.4f} seconds.")

if __name__ == "__main__":
    print("[INFO] Starting main program.")
    main()
    print("[INFO] Program finished successfully.")
