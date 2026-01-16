import numpy as np


def generate_reference_and_background_maps(
    params: dict,
    fov_shape_os: tuple,
    final_fov_shape: tuple,
):
    """
    Generate stationary reference field and background intensity maps for the
    simulated field of view.

    This function centralizes the optical background / substrate model. In the
    current implementation, the reference field and background are spatially
    uniform, matching the behavior of the original code. Future chip-pattern
    presets (gold holes, nanopillars, etc.) will modify these maps while
    leaving the rest of the rendering pipeline unchanged.

    Args:
        params (dict):
            Global simulation parameter dictionary (PARAMS). Must contain:
                - "reference_field_amplitude"
                - "background_intensity"
        fov_shape_os (tuple[int, int]):
            (height, width) of the oversampled field-of-view grid
            (without PSF padding).
        final_fov_shape (tuple[int, int]):
            (height, width) of the final video frames.

    Returns:
        tuple:
            E_ref_os (np.ndarray):
                Complex 2D array of shape fov_shape_os containing the reference
                field E_ref(x, y) on the oversampled field of view.
            E_ref_final (np.ndarray):
                Complex 2D array of shape final_fov_shape containing the
                reference field at the final image resolution.
            background_final (np.ndarray):
                Float 2D array of shape final_fov_shape containing the
                background intensity in camera counts for each pixel.
    """
    E_ref_amplitude = float(params["reference_field_amplitude"])
    background_intensity = float(params["background_intensity"])

    # Oversampled field-of-view (no padding).
    E_ref_os = np.full(fov_shape_os, E_ref_amplitude, dtype=np.complex128)

    # Final image resolution maps. For a spatially uniform background the maps
    # are simply constant arrays at the desired shape.
    E_ref_final = np.full(final_fov_shape, E_ref_amplitude, dtype=np.complex128)
    background_final = np.full(final_fov_shape, background_intensity, dtype=float)

    return E_ref_os, E_ref_final, background_final