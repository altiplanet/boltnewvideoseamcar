import numpy as np
import cupy as cp
import cv2
import time

class EnergyCalculator:
    def __init__(self):
        self.device = cp.cuda.Device()
        print(f"[INFO] Initialized GPU device: {self.device}")

    def calculate_frame_energy(self, frame):
        """Calculate energy map for a single frame using GPU acceleration."""
        start_time = time.time()
        print("[INFO] Starting energy calculation for a frame.")

        # Transfer frame to GPU
        print("[INFO] Transferring frame to GPU.")
        frame_gpu = cp.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        print("[DEBUG] Frame shape on GPU:", frame_gpu.shape)

        # Calculate gradients on GPU
        print("[INFO] Calculating Sobel gradients on GPU.")
        sobel_x = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        grad_x = cp.abs(cp.correlate2d(frame_gpu, sobel_x, mode='same'))
        grad_y = cp.abs(cp.correlate2d(frame_gpu, sobel_y, mode='same'))

        # Calculate energy map
        print("[INFO] Calculating energy map.")
        energy = cp.sqrt(grad_x**2 + grad_y**2)

        # Transfer back to CPU
        print("[INFO] Transferring energy map back to CPU.")
        energy_cpu = cp.asnumpy(energy)
        print(f"[DEBUG] Energy map shape on CPU: {energy_cpu.shape}")

        elapsed_time = time.time() - start_time
        print(f"[INFO] Energy calculation completed in {elapsed_time:.4f} seconds.")
        return energy_cpu

    def calculate_temporal_energy(self, frames):
        """Calculate temporal energy between consecutive frames."""
        start_time = time.time()
        print("[INFO] Starting temporal energy calculation.")

        temporal_energy = np.zeros_like(frames[0], dtype=np.float32)
        print(f"[DEBUG] Temporal energy initial shape: {temporal_energy.shape}")

        for i in range(len(frames) - 1):
            print(f"[INFO] Processing frame pair {i} and {i+1}.")

            # Transfer frames to GPU
            frame1_gpu = cp.asarray(frames[i])
            frame2_gpu = cp.asarray(frames[i + 1])
            print("[DEBUG] Frames transferred to GPU.")

            # Calculate absolute difference
            diff = cp.abs(frame2_gpu - frame1_gpu)
            print("[DEBUG] Calculated frame difference on GPU.")

            # Accumulate temporal energy
            temporal_energy += cp.asnumpy(diff)

        temporal_energy /= (len(frames) - 1)
        print("[INFO] Temporal energy calculation completed.")

        elapsed_time = time.time() - start_time
        print(f"[INFO] Total temporal energy calculation time: {elapsed_time:.4f} seconds.")
        return temporal_energy
