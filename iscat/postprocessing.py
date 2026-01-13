import numpy as np
import cv2
from tqdm import tqdm

def apply_background_subtraction(signal_frames, reference_frames):
    """
    Performs background subtraction using the formula (Signal - Reference) / Reference
    and normalizes the result to an 8-bit range for video encoding.

    Args:
        signal_frames (list): A list of 16-bit signal frames.
        reference_frames (list): A list of 16-bit reference frames.

    Returns:
        list: A list of 8-bit, normalized frames ready for video encoding.
    """
    if not signal_frames or not reference_frames: return []
    
    subtracted_frames = []
    print("Applying background subtraction...")
    for signal_frame, ref_frame in tqdm(zip(signal_frames, reference_frames), total=len(signal_frames)):
        # Add a small epsilon to the denominator to prevent division by zero.
        subtracted = (signal_frame.astype(float) - ref_frame.astype(float)) / (ref_frame.astype(float) + 1e-9)
        subtracted_frames.append(subtracted)
        
    # Normalize the contrast range across the whole video for consistent brightness.
    # This robustly clips outliers by finding the 0.5 and 99.5 percentile values.
    min_val, max_val = np.percentile(subtracted_frames, [0.5, 99.5])
    
    final_frames_8bit = []
    if max_val > min_val:
        for frame in subtracted_frames:
            # Scale the frame data to the 0-255 range.
            norm_frame = 255 * (frame - min_val) / (max_val - min_val)
            final_frames_8bit.append(np.clip(norm_frame, 0, 255).astype(np.uint8))
    else: # Handle the edge case of a video with no contrast.
        final_frames_8bit = [np.full(signal_frames[0].shape, 128, dtype=np.uint8) for _ in signal_frames]

    return final_frames_8bit

def save_video(filename, frames, fps, size):
    """
    Encodes and saves a list of frames to an .mp4 video file.

    Args:
        filename (str): The output path for the video file.
        frames (list): A list of 8-bit numpy array frames.
        fps (int): The desired frames per second for the video.
        size (tuple): The (width, height) of the video.
    """
    print(f"Saving final video to {filename}...")
    # Define the codec and create VideoWriter object.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, fps, size, isColor=False)
    
    for frame in frames:
        video_writer.write(frame)
        
    video_writer.release()
    print("Simulation finished successfully!")