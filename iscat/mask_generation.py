import os
import cv2
import numpy as np


def generate_binary_mask(contrast_image: np.ndarray, mask_threshold: float) -> np.ndarray:
    """
    Generate a binary mask from a contrast image by thresholding the
    normalized magnitude of the contrast.

    This function reproduces the existing mask-generation logic used in
    rendering.generate_video_and_masks. It normalizes the absolute value
    of the contrast by its maximum and then thresholds it.

    Args:
        contrast_image (np.ndarray): 2D array representing the per-particle
            contrast image at the final (non-oversampled) resolution.
        mask_threshold (float): Threshold in [0, 1] applied to the normalized
            contrast magnitude.

    Returns:
        np.ndarray: A uint8 binary mask with values 0 or 255.
    """
    # Compute the maximum magnitude of the contrast signal.
    max_val = np.max(np.abs(contrast_image))

    if max_val > 1e-9:
        # Normalize contrast so the strongest signal has magnitude 1.
        normalized_contrast = np.abs(contrast_image) / max_val

        # Create a binary mask: pixels above the threshold are foreground.
        mask = (normalized_contrast > mask_threshold).astype(np.uint8) * 255
    else:
        # If the particle has no visible signal, return an all-zero mask.
        mask = np.zeros_like(contrast_image, dtype=np.uint8)

    return mask


def save_mask(
    mask: np.ndarray,
    base_mask_directory: str,
    particle_index: int,
    frame_index: int,
) -> None:
    """
    Save a single-particle mask image to disk using the established directory
    and filename conventions.

    Directory structure:
        base_mask_directory/
            particle_1/
                frame_0000.png
                frame_0001.png
                ...
            particle_2/
                ...

    Args:
        mask (np.ndarray): The binary mask image to save (uint8, 0 or 255).
        base_mask_directory (str): Root directory for all particle masks.
        particle_index (int): Zero-based particle index.
        frame_index (int): Zero-based frame index.
    """
    particle_dir = os.path.join(base_mask_directory, f"particle_{particle_index + 1}")
    # Ensure the particle-specific directory exists. This is redundant with the
    # setup code in main.py but harmless and makes this function robust if used
    # in other contexts.
    os.makedirs(particle_dir, exist_ok=True)

    filename = os.path.join(particle_dir, f"frame_{frame_index:04d}.png")
    cv2.imwrite(filename, mask)


def generate_and_save_mask_for_particle(
    contrast_image: np.ndarray,
    params: dict,
    particle_index: int,
    frame_index: int,
) -> None:
    """
    High-level helper that generates and saves a mask for a single particle
    in a single frame, using the existing PARAMS configuration.

    This function directly mirrors the previous inline behavior in
    rendering.generate_video_and_masks and is the preferred entry point
    from the rendering pipeline.

    Args:
        contrast_image (np.ndarray): 2D contrast image for the particle at
            the final image resolution.
        params (dict): Global simulation parameter dictionary (PARAMS).
            Must contain "mask_threshold" and "mask_output_directory".
        particle_index (int): Zero-based index of the particle.
        frame_index (int): Zero-based index of the frame.
    """
    mask_threshold = params["mask_threshold"]
    base_mask_dir = params["mask_output_directory"]

    mask = generate_binary_mask(contrast_image, mask_threshold)
    save_mask(mask, base_mask_dir, particle_index, frame_index)