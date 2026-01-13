import os
import warnings
import numpy as np
import cv2

# Import simulation components from other files
from config import PARAMS
from trajectory import simulate_trajectories
from optics import compute_ipsf_stack
from rendering import generate_video_and_masks
from postprocessing import apply_background_subtraction, save_video

# Suppress RankWarning from numpy's polyfit, which can occur in Mie scattering calculations
warnings.filterwarnings("ignore", category=np.RankWarning)

def main():
    """
    Main function to run the entire iSCAT simulation and video generation pipeline.
    """
    # --- Setup Output Directories ---
    # Ensure the output directories for the video and masks exist.
    if PARAMS["mask_generation_enabled"]:
        base_mask_dir = PARAMS["mask_output_directory"]
        print(f"Checking for mask output directories at {base_mask_dir}...")
        os.makedirs(base_mask_dir, exist_ok=True)
        for i in range(PARAMS["num_particles"]):
            particle_mask_dir = os.path.join(base_mask_dir, f"particle_{i+1}")
            os.makedirs(particle_mask_dir, exist_ok=True)
    
    output_dir = os.path.dirname(PARAMS["output_filename"])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
            
    # --- Step 1: Simulate particle movement ---
    # This generates the 3D coordinates for each particle over time.
    trajectories_nm = simulate_trajectories(PARAMS)
    
    # --- Step 2: Compute unique iPSF stacks ---
    # To save computation time, only compute the iPSF once for each unique type of particle.
    unique_particles = {}
    print("Pre-computing unique particle iPSF stacks...")
    for i in range(PARAMS["num_particles"]):
        # A unique particle is defined by its diameter and refractive index.
        key = (PARAMS["particle_diameters_nm"][i], 
               PARAMS["particle_refractive_indices"][i].real, 
               PARAMS["particle_refractive_indices"][i].imag)
        if key not in unique_particles:
            unique_particles[key] = compute_ipsf_stack(
                PARAMS,
                PARAMS["particle_diameters_nm"][i],
                PARAMS["particle_refractive_indices"][i]
            )
    
    # Assign the correct pre-computed iPSF interpolator to each particle.
    ipsf_interpolators = [
        unique_particles[(PARAMS["particle_diameters_nm"][i], 
                          PARAMS["particle_refractive_indices"][i].real, 
                          PARAMS["particle_refractive_indices"][i].imag)]
        for i in range(PARAMS["num_particles"])
    ]

    # --- Step 3: Generate raw video frames and masks ---
    # This is the main rendering loop that generates the raw 16-bit data.
    raw_signal_frames, raw_reference_frames = generate_video_and_masks(
        PARAMS, 
        trajectories_nm, 
        ipsf_interpolators
    )
    
    # --- Step 4: Process frames for final video ---
    # This performs background subtraction and normalization to 8-bit.
    final_frames = apply_background_subtraction(raw_signal_frames, raw_reference_frames)
    
    if not final_frames:
        print("Video generation failed or produced no frames. Exiting.")
        return

    # --- Step 5: Save the final video ---
    # Encodes the processed frames into an .mp4 file.
    img_size = (PARAMS["image_size_pixels"], PARAMS["image_size_pixels"])
    save_video(PARAMS["output_filename"], final_frames, PARAMS["fps"], img_size)

if __name__ == '__main__':
    main()