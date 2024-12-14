import argparse
from video_processor import VideoProcessor

def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated video seam carving")
    parser.add_argument("input_path", help="Path to input video")
    parser.add_argument("output_path", help="Path to output video")
    parser.add_argument("target_width", type=int, help="Target width")
    parser.add_argument("target_height", type=int, help="Target height")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    
    args = parser.parse_args()
    
    processor = VideoProcessor(
        args.input_path,
        args.output_path,
        args.target_width,
        args.target_height,
        args.workers
    )
    
    processor.process()

if __name__ == "__main__":
    main()