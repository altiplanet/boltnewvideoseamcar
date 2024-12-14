import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from energy_calculator import EnergyCalculator
from seam_finder import SeamFinder
import time

class FrameProcessor:
    def __init__(self, num_workers=4):
        self.energy_calculator = EnergyCalculator()
        self.seam_finder = SeamFinder()
        self.num_workers = num_workers
        print(f"[INFO] Initialized FrameProcessor with {self.num_workers} workers.")

    def remove_seam(self, frame, seam):
        """Remove a seam from the frame."""
        print("[INFO] Removing a seam from the frame.")
        height, width = frame.shape[:2]
        new_frame = np.zeros((height, width-1, frame.shape[2]), dtype=frame.dtype)
        
        for y in range(height):
            new_frame[y, :seam[y]] = frame[y, :seam[y]]
            new_frame[y, seam[y]:] = frame[y, seam[y]+1:]
        
        print(f"[DEBUG] Seam removed. New frame size: {new_frame.shape}")
        return new_frame

    def process_frame_group(self, frames):
        """Process a group of consecutive frames."""
        print(f"[INFO] Processing a group of {len(frames)} frames.")
        start_time = time.time()
        
        # Calculate energy maps
        energy_maps = []
        print("[INFO] Calculating temporal energy.")
        temporal_energy = self.energy_calculator.calculate_temporal_energy(frames)
        
        for idx, frame in enumerate(frames):
            print(f"[INFO] Calculating spatial energy for frame {idx+1}.")
            spatial_energy = self.energy_calculator.calculate_frame_energy(frame)
            total_energy = spatial_energy + temporal_energy
            energy_maps.append(total_energy)

        print("[INFO] Finding optimal seam.")
        seam = self.seam_finder.find_seam(np.mean(energy_maps, axis=0))
        
        # Remove seam from all frames
        processed_frames = []
        for idx, frame in enumerate(frames):
            print(f"[INFO] Removing seam from frame {idx+1}.")
            processed_frames.append(self.remove_seam(frame, seam))

        elapsed_time = time.time() - start_time
        print(f"[INFO] Frame group processed in {elapsed_time:.4f} seconds.")
        return processed_frames

    def process_video(self, frames, target_width):
        """Process entire video in parallel."""
        print("[INFO] Starting video processing.")
        num_seams = frames[0].shape[1] - target_width
        current_frames = frames

        for i in range(num_seams):
            print(f"[INFO] Removing seam {i+1}/{num_seams}.")

            # Process frames in groups
            frame_groups = [current_frames[j:j+5] for j in range(0, len(current_frames), 5)]

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                print("[INFO] Processing frame groups in parallel.")
                processed_groups = list(executor.map(self.process_frame_group, frame_groups))

            # Flatten processed groups
            current_frames = [frame for group in processed_groups for frame in group]

        print("[INFO] Video processing completed.")
        return current_frames
