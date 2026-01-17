import os
import warnings

import numpy as np
import cv2

# Import simulation components from other files
from config import PARAMS
from materials import resolve_particle_refractive_indices
from trajectory import simulate_trajectories
from optics import compute_ipsf_stack
from rendering import generate_video_and_masks
from postprocessing import apply_background_subtraction, save_video

# Suppress RankWarning from numpy's polyfit, which can occur in Mie scattering calculations
warnings.filterwarnings("ignore", category=np.RankWarning)


def run_simulation(params: dict) -> None:
    """
    Run the complete iSCAT simulation and video generation pipeline for a given
    parameter dictionary.

    This function is the core programmatic entry point for the simulation. It
    performs:

        1. Output directory preparation (video and masks).
        2. Per-particle refractive index resolution from materials/overrides.
        3. 3D Brownian motion trajectory simulation.
        4. Pre-computation of unique iPSF Z-stacks for each particle type.
        5. Frame-by-frame rendering of signal/reference frames and masks.
        6. Background subtraction, normalization, and final .mp4 encoding.

    When called as:

        run_simulation(PARAMS)

    it reproduces the exact behavior of the original `main()` function, so
    existing script usage is preserved. Future wrappers can call this function
    with modified copies of PARAMS to generate many videos without relying on
    global state.

    Args:
        params (dict): Simulation parameter dictionary. Typically a dictionary
            following the structure of config.PARAMS, possibly with overrides
            applied (e.g., for presets or randomized generation).
    """
    # --- Setup Output Directories ---
    # Ensure the output directories for the video and masks exist.
    if params["mask_generation_enabled"]:
        base_mask_dir = params["mask_output_directory"]
        print(f"Checking for mask output directories at {base_mask_dir}...")
        os.makedirs(base_mask_dir, exist_ok=True)
        for i in range(params["num_particles"]):
            particle_mask_dir = os.path.join(base_mask_dir, f"particle_{i+1}")
            os.makedirs(particle_mask_dir, exist_ok=True)

    output_dir = os.path.dirname(params["output_filename"])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --- Resolve per-particle refractive indices from materials/overrides ---
    # This combines:
    #   - params["particle_materials"] (if provided) -> material-based lookup, and
    #   - params["particle_refractive_indices"] (if provided) -> explicit overrides.
    #
    # The result is a single complex refractive index per particle, stored back
    # into params["particle_refractive_indices"] as a numpy array. This ensures
    # subsequent optics code sees a consistent, resolved value regardless of how
    # the user specified the particle properties.
    particle_refractive_indices = resolve_particle_refractive_indices(params)
    diameters_nm = params["particle_diameters_nm"]

    # --- Step 1: Simulate particle movement ---
    # This generates the 3D coordinates for each particle over time.
    trajectories_nm = simulate_trajectories(params)

    # --- Step 2: Compute unique iPSF stacks ---
    # To save computation time, only compute the iPSF once for each unique type
    # of particle. A unique particle type is defined by its diameter and complex
    # refractive index (n + i k) within the medium.
    unique_particles = {}
    print("Pre-computing unique particle iPSF stacks...")
    num_particles = params["num_particles"]

    for i in range(num_particles):
        n_complex = particle_refractive_indices[i]
        key = (
            diameters_nm[i],
            float(n_complex.real),
            float(n_complex.imag),
        )
        if key not in unique_particles:
            unique_particles[key] = compute_ipsf_stack(
                params,
                diameters_nm[i],
                n_complex,
            )

    # Assign the correct pre-computed iPSF interpolator to each particle.
    ipsf_interpolators = [
        unique_particles[
            (
                diameters_nm[i],
                float(particle_refractive_indices[i].real),
                float(particle_refractive_indices[i].imag),
            )
        ]
        for i in range(num_particles)
    ]

    # --- Step 3: Generate raw video frames and masks ---
    # This is the main rendering loop that generates the raw 16-bit data.
    raw_signal_frames, raw_reference_frames = generate_video_and_masks(
        params,
        trajectories_nm,
        ipsf_interpolators,
    )

    # --- Step 4: Process frames for final video ---
    # This performs background subtraction and normalization to 8-bit.
    final_frames = apply_background_subtraction(
        raw_signal_frames,
        raw_reference_frames,
        params,
    )

    if not final_frames:
        print("Video generation failed or produced no frames. Exiting.")
        return

    # --- Step 5: Save the final video ---
    # Encodes the processed frames into an .mp4 file.
    img_size = (params["image_size_pixels"], params["image_size_pixels"])
    save_video(params["output_filename"], final_frames, params["fps"], img_size)


def main():
    """
    Script entry point: run the simulation using the global config.PARAMS.

    This preserves the behavior of the original implementation so that running
    this file as a script still performs a single simulation configured by
    config.PARAMS.
    """
    run_simulation(PARAMS)


if __name__ == '__main__':
    main()