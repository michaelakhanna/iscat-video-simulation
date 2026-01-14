import numpy as np
import cv2
from tqdm import tqdm

from mask_generation import generate_and_save_mask_for_particle
from trackability import TrackabilityModel


def add_noise(frame, params):
    """
    Applies simulated shot (Poisson) and read (Gaussian) noise to an image frame.

    Args:
        frame (numpy.ndarray): The ideal, noise-free image frame.
        params (dict): The main simulation parameter dictionary.

    Returns:
        numpy.ndarray: The noisy image frame.
    """
    noisy_frame = frame.copy()
    if params["shot_noise_enabled"]:
        # Scale the noise component to allow for artistic control.
        ideal_frame = noisy_frame
        full_noisy_frame = np.random.poisson(ideal_frame).astype(float)
        noise_component = full_noisy_frame - ideal_frame
        noisy_frame = ideal_frame + noise_component * params["shot_noise_scaling_factor"]

    if params["gaussian_noise_enabled"]:
        noisy_frame += np.random.normal(scale=params["read_noise_std"], size=frame.shape)

    return noisy_frame


def generate_video_and_masks(params, trajectories, ipsf_interpolators):
    """
    Generates all video frames and segmentation masks by placing particles according
    to their trajectories and applying the appropriate iPSF. Includes motion blur.

    Args:
        params (dict): The main simulation parameter dictionary.
        trajectories (numpy.ndarray): The 3D array of particle trajectories with
            shape (num_particles, num_frames, 3) in nanometers.
        ipsf_interpolators (list): A list of iPSF interpolator objects, one for
            each particle.

    Returns:
        tuple[list, list]: A tuple containing two lists: one for the raw signal
            frames and one for the raw reference frames, both as 16-bit integer
            arrays.
    """
    num_frames = int(params["fps"] * params["duration_seconds"])
    dt = 1.0 / params["fps"]
    num_particles = params["num_particles"]

    img_size = params["image_size_pixels"]
    pixel_size_nm = params["pixel_size_nm"]
    os_factor = params["psf_oversampling_factor"]
    final_size = (img_size, img_size)
    os_size = img_size * os_factor

    # --- Bit depth and camera count handling ---
    # The raw simulated frames are stored as uint16 but their meaningful dynamic
    # range is controlled by PARAMS["bit_depth"]. This allows us to simulate
    # 12-bit, 14-bit, or 16-bit cameras while keeping the storage format simple.
    bit_depth = params["bit_depth"]
    if not isinstance(bit_depth, int) or bit_depth <= 0:
        raise ValueError("PARAMS['bit_depth'] must be a positive integer.")

    max_supported_bit_depth = 16  # Limited by uint16 storage in this implementation.
    if bit_depth > max_supported_bit_depth:
        raise ValueError(
            f"PARAMS['bit_depth']={bit_depth} exceeds the maximum supported bit depth "
            f"of {max_supported_bit_depth} for uint16 storage."
        )

    max_camera_count = (1 << bit_depth) - 1

    E_ref = params["reference_field_amplitude"]
    background = params["background_intensity"]

    num_subsamples = params["motion_blur_subsamples"] if params["motion_blur_enabled"] else 1
    sub_dt = dt / num_subsamples

    all_signal_frames = []
    all_reference_frames = []

    # Initialize the human trackability confidence model if masks are enabled.
    if params["mask_generation_enabled"]:
        trackability_model = TrackabilityModel(params, num_particles)
        trackability_threshold = params.get("trackability_confidence_threshold", 0.8)
        if not (0.0 <= trackability_threshold <= 1.0):
            raise ValueError(
                "PARAMS['trackability_confidence_threshold'] must be between 0 and 1."
            )
    else:
        trackability_model = None
        trackability_threshold = None

    print("Generating video frames and masks...")
    for f in tqdm(range(num_frames)):
        # Accumulators for the motion-blurred electric field of each particle.
        blurred_particle_fields = [
            np.zeros((os_size, os_size), dtype=np.complex128)
            for _ in range(num_particles)
        ]

        # --- Subsample rendering for motion blur ---
        for s in range(num_subsamples):
            current_time = f * dt + s * sub_dt
            frame_idx_floor = int(current_time / dt)
            frame_idx_ceil = min(frame_idx_floor + 1, num_frames - 1)
            interp_factor = (current_time / dt) - frame_idx_floor

            # Linearly interpolate particle positions between trajectory points.
            current_pos_nm = (
                (1.0 - interp_factor) * trajectories[:, frame_idx_floor, :]
                + interp_factor * trajectories[:, frame_idx_ceil, :]
            )

            for i in range(num_particles):
                px, py, pz = current_pos_nm[i]

                # Get the pre-computed scattered field (iPSF) for the particle's z-position.
                E_sca_2D = ipsf_interpolators[i]([pz])[0]

                # Upscale to the oversampled resolution for higher accuracy placement.
                resized_real = cv2.resize(
                    np.real(E_sca_2D),
                    (os_size, os_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                resized_imag = cv2.resize(
                    np.imag(E_sca_2D),
                    (os_size, os_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                E_sca_2D_rescaled = resized_real + 1j * resized_imag

                # --- Position the PSF on the oversampled canvas by circularly shifting it ---
                # The PSF returned by the interpolator is centered in the array. We translate
                # this pattern so that its center coincides with the particle's (x, y) position
                # in the oversampled image grid. This avoids creating static rectangular
                # support regions from zero-padding/cropping.
                center_x_px = int(round(px / pixel_size_nm * os_factor))
                center_y_px = int(round(py / pixel_size_nm * os_factor))

                # Compute integer shifts relative to the optical center of the field of view.
                shift_x = center_x_px - os_size // 2
                shift_y = center_y_px - os_size // 2

                # Circularly shift the PSF to the particle position.
                E_sca_particle_inst = np.roll(
                    E_sca_2D_rescaled,
                    shift=(shift_y, shift_x),
                    axis=(0, 1),
                )

                # Apply signal multiplier and accumulate for motion blur.
                blurred_particle_fields[i] += (
                    E_sca_particle_inst * params["particle_signal_multipliers"][i]
                )

        # Average the fields from all subsamples to create the final motion-blurred field.
        for i in range(num_particles):
            blurred_particle_fields[i] /= num_subsamples

        # --- Mask Generation for this Frame ---
        if params["mask_generation_enabled"]:
            for i in range(num_particles):
                # Skip particles that have already been declared "lost".
                if trackability_model.is_particle_lost(i):
                    continue

                E_sca_particle_blurred = blurred_particle_fields[i]

                # Contrast is the change in intensity caused by the particle's scattered field.
                contrast_os = np.abs(E_ref + E_sca_particle_blurred) ** 2 - np.abs(E_ref) ** 2
                contrast_final = cv2.resize(
                    contrast_os, final_size, interpolation=cv2.INTER_AREA
                )

                # Compute the human trackability confidence for this particle in this frame.
                position_nm = trajectories[i, f, :]  # [x, y, z] at the frame time
                confidence = trackability_model.update_and_compute_confidence(
                    particle_index=i,
                    frame_index=f,
                    position_nm=position_nm,
                    contrast_image=contrast_final,
                )

                if confidence >= trackability_threshold:
                    # Delegate mask creation and saving to the mask_generation subsystem.
                    generate_and_save_mask_for_particle(
                        contrast_image=contrast_final,
                        params=params,
                        particle_index=i,
                        frame_index=f,
                    )
                else:
                    # Once a particle is considered lost, its mask is no longer generated
                    # for the remainder of the video.
                    trackability_model.lost[i] = True

        # --- Final Video Frame Generation ---
        E_sca_total = np.sum(blurred_particle_fields, axis=0)

        # Interfere the total scattered field with the reference field to get intensity.
        intensity_os = np.abs(E_ref + E_sca_total) ** 2
        intensity = cv2.resize(intensity_os, final_size, interpolation=cv2.INTER_AREA)

        # Scale intensity to camera counts.
        if np.max(intensity) > 0:
            intensity_scaled = background + (intensity - E_ref**2) * background
        else:
            intensity_scaled = background * np.ones_like(intensity)

        signal_frame_noisy = add_noise(intensity_scaled, params)
        all_signal_frames.append(
            np.clip(signal_frame_noisy, 0, max_camera_count).astype(np.uint16)
        )

        # Generate a corresponding noisy reference frame for background subtraction.
        reference_frame_ideal = np.full(final_size, background, dtype=float)
        reference_frame_noisy = add_noise(reference_frame_ideal, params)
        all_reference_frames.append(
            np.clip(reference_frame_noisy, 0, max_camera_count).astype(np.uint16)
        )

        # If all particles are lost according to the trackability model, we can
        # terminate video generation early. The existing frames (up to and
        # including this one) remain valid and are passed on to post-processing.
        if params["mask_generation_enabled"] and trackability_model.are_all_particles_lost():
            print(
                f"All particles lost according to the trackability model at frame {f}. "
                "Terminating video generation early."
            )
            break

    print("Frame and mask generation complete.")
    return all_signal_frames, all_reference_frames