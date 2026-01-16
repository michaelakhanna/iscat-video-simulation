import numpy as np


def _generate_gold_hole_pattern(
    shape: tuple,
    pixel_size_nm: float,
    hole_diameter_um: float,
    hole_edge_to_edge_spacing_um: float,
    hole_intensity_factor: float,
    gold_intensity_factor: float,
) -> np.ndarray:
    """
    Generate a dimensionless intensity pattern map for a gold film with circular
    holes arranged on a square grid.

    The pattern is defined in physical space as follows:
        - Circular holes of diameter `hole_diameter_um`.
        - Square grid with center-to-center pitch:
              pitch_um = hole_diameter_um + hole_edge_to_edge_spacing_um
        - The grid is aligned with the x and y axes, with one hole centered at
          the origin; the field of view is centered on (0, 0).

    Each pixel is labeled as either "hole" or "gold" based on its distance to
    the nearest hole center. Pixels inside a hole have intensity
    `hole_intensity_factor`, and pixels in gold have `gold_intensity_factor`.
    The resulting map is then normalized to have unit mean so that the global
    brightness remains controlled by the background_intensity parameter.

    Args:
        shape (tuple[int, int]): (height, width) of the desired pattern map.
        pixel_size_nm (float): Physical pixel size in nanometers for this grid.
        hole_diameter_um (float): Hole diameter in micrometers.
        hole_edge_to_edge_spacing_um (float): Gold spacing between hole edges in micrometers.
        hole_intensity_factor (float): Relative background intensity inside holes.
        gold_intensity_factor (float): Relative background intensity in gold regions.

    Returns:
        np.ndarray: 2D array of shape `shape`, dtype float, dimensionless
            multiplicative factors with mean ~1.0.
    """
    height, width = int(shape[0]), int(shape[1])

    if height <= 0 or width <= 0:
        raise ValueError("Pattern shape must have positive height and width.")

    pixel_size_nm = float(pixel_size_nm)
    if pixel_size_nm <= 0.0:
        raise ValueError("pixel_size_nm must be positive for pattern generation.")

    hole_diameter_um = float(hole_diameter_um)
    hole_edge_to_edge_spacing_um = float(hole_edge_to_edge_spacing_um)
    if hole_diameter_um <= 0.0:
        raise ValueError("hole_diameter_um must be positive.")
    if hole_edge_to_edge_spacing_um < 0.0:
        raise ValueError("hole_edge_to_edge_spacing_um must be non-negative.")

    pitch_um = hole_diameter_um + hole_edge_to_edge_spacing_um
    radius_um = hole_diameter_um / 2.0

    if pitch_um <= 0.0:
        raise ValueError("Computed pitch (hole_diameter_um + hole_edge_to_edge_spacing_um) must be positive.")

    hole_intensity_factor = float(hole_intensity_factor)
    gold_intensity_factor = float(gold_intensity_factor)
    if hole_intensity_factor <= 0.0 or gold_intensity_factor <= 0.0:
        raise ValueError("hole_intensity_factor and gold_intensity_factor must be positive.")

    # Convert pixel size to micrometers for coordinate generation.
    pixel_size_um = pixel_size_nm * 1e-3

    # Define coordinates so that the field of view is centered at (0, 0) in
    # physical units and each pixel coordinate corresponds to the center of
    # that pixel. This ensures that the pattern is symmetric with respect to
    # the image center and that the grid is aligned with the axes.
    x_indices = np.arange(width, dtype=float)
    y_indices = np.arange(height, dtype=float)

    x_um = (x_indices - width / 2.0 + 0.5) * pixel_size_um
    y_um = (y_indices - height / 2.0 + 0.5) * pixel_size_um

    X_um, Y_um = np.meshgrid(x_um, y_um)

    # Compute coordinates relative to the nearest hole center using a periodic
    # wrapping with period equal to the pitch. The expression
    #
    #   ((coord + pitch/2) % pitch) - pitch/2
    #
    # maps any coordinate onto the interval [-pitch/2, pitch/2), i.e., into a
    # single unit cell centered at the origin. The radial distance inside this
    # unit cell can then be compared to the hole radius to decide whether the
    # point lies inside a hole or within the gold film.
    half_pitch = pitch_um / 2.0

    dx_um = (X_um + half_pitch) % pitch_um - half_pitch
    dy_um = (Y_um + half_pitch) % pitch_um - half_pitch

    r_um = np.sqrt(dx_um * dx_um + dy_um * dy_um)

    # Initialize pattern with gold intensity everywhere, then overwrite the
    # hole regions.
    pattern = np.full((height, width), gold_intensity_factor, dtype=float)
    hole_mask = r_um <= radius_um
    pattern[hole_mask] = hole_intensity_factor

    # Normalize pattern to unit mean so that the global brightness is not
    # changed; only relative spatial variations are introduced.
    mean_val = float(pattern.mean())
    if mean_val > 0.0:
        pattern /= mean_val

    return pattern


def generate_reference_and_background_maps(
    params: dict,
    fov_shape_os: tuple,
    final_fov_shape: tuple,
):
    """
    Generate stationary reference field and background intensity maps for the
    simulated field of view.

    This function centralizes the optical background / substrate model. It
    supports both a uniform background (no chip pattern) and parameterized chip
    pattern presets such as a gold film with circular holes. The returned maps
    are:

        - E_ref_os: complex reference field on the oversampled field of view.
        - E_ref_final: complex reference field at the final image resolution.
        - background_final: scalar background intensity map (camera counts)
          at the final image resolution.

    The chip pattern is represented as a dimensionless spatial multiplier on
    both the reference field amplitude and the background intensity. This
    satisfies the design requirement that the pattern be part of the physical
    image formation (via E_ref) and the noise model (via background_intensity),
    while keeping the rest of the rendering pipeline unchanged.

    Behavior:
        - If chip_pattern_enabled is False, or chip_substrate_preset is
          "empty_background", or chip_pattern_model is "none", the maps are
          spatially uniform as in the original implementation.
        - Otherwise, the pattern model and substrate preset determine the
          spatial maps. Currently, "gold_holes_v1" with presets
          "default_gold_holes" and "lab_default_gold_holes" are supported.

    Args:
        params (dict):
            Global simulation parameter dictionary (PARAMS). Must contain:
                - "reference_field_amplitude"
                - "background_intensity"
                - "pixel_size_nm"
                - "psf_oversampling_factor"
            and for chip patterns:
                - "chip_pattern_enabled"
                - "chip_pattern_model"
                - "chip_substrate_preset"
                - "chip_pattern_dimensions" (dict), when using gold_holes_v1
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

    chip_enabled = bool(params.get("chip_pattern_enabled", False))
    pattern_model = str(params.get("chip_pattern_model", "gold_holes_v1"))
    substrate_preset = str(params.get("chip_substrate_preset", "empty_background"))

    # Default: no spatial pattern, uniform maps (matches original behavior).
    use_uniform_background = (
        (not chip_enabled)
        or (substrate_preset == "empty_background")
        or (pattern_model == "none")
    )

    if use_uniform_background:
        # Oversampled field-of-view (no padding).
        E_ref_os = np.full(fov_shape_os, E_ref_amplitude, dtype=np.complex128)

        # Final image resolution maps: constant arrays at the desired shape.
        E_ref_final = np.full(final_fov_shape, E_ref_amplitude, dtype=np.complex128)
        background_final = np.full(final_fov_shape, background_intensity, dtype=float)

        return E_ref_os, E_ref_final, background_final

    # At this point, a chip pattern is requested. We validate and construct the
    # appropriate pattern maps.
    if pattern_model != "gold_holes_v1":
        raise ValueError(
            f"Unsupported chip_pattern_model '{pattern_model}'. "
            "Currently supported: 'none', 'gold_holes_v1'."
        )

    # Only gold-hole presets are implemented at this stage. This can be extended
    # to additional substrates (e.g., nanopillars) later without changing the
    # interface of this function.
    if substrate_preset not in ("default_gold_holes", "lab_default_gold_holes"):
        raise ValueError(
            f"Unsupported chip_substrate_preset '{substrate_preset}' for "
            "chip_pattern_model 'gold_holes_v1'. Supported presets are "
            "'empty_background', 'default_gold_holes', and 'lab_default_gold_holes'."
        )

    dims = params.get("chip_pattern_dimensions", {})
    if not isinstance(dims, dict):
        raise TypeError(
            "PARAMS['chip_pattern_dimensions'] must be a dictionary when "
            "chip_pattern_model is 'gold_holes_v1'."
        )

    # Geometry parameters. For the lab default preset we apply canonical values
    # (15 µm holes, 2 µm spacing, 20 nm metal thickness) when the user has not
    # overridden them. For the generic default_gold_holes preset, the values in
    # chip_pattern_dimensions are used as-is with reasonable fallbacks.
    if substrate_preset == "lab_default_gold_holes":
        hole_diameter_um = float(dims.get("hole_diameter_um", 15.0))
        hole_edge_to_edge_spacing_um = float(dims.get("hole_edge_to_edge_spacing_um", 2.0))
        hole_depth_nm = float(dims.get("hole_depth_nm", 20.0))  # currently unused in optics
    else:  # "default_gold_holes"
        hole_diameter_um = float(dims.get("hole_diameter_um", 15.0))
        hole_edge_to_edge_spacing_um = float(dims.get("hole_edge_to_edge_spacing_um", 2.0))
        hole_depth_nm = float(dims.get("hole_depth_nm", 20.0))  # currently unused in optics

    # Relative intensity levels for holes and gold regions. These control the
    # spatial modulation of both the reference field and the background
    # intensity. The pattern is normalized to unit mean afterwards to keep the
    # global brightness consistent with background_intensity.
    hole_intensity_factor = float(dims.get("hole_intensity_factor", 0.7))
    gold_intensity_factor = float(dims.get("gold_intensity_factor", 1.0))

    # Validate geometry and intensity factors.
    if hole_diameter_um <= 0.0:
        raise ValueError("chip_pattern_dimensions['hole_diameter_um'] must be positive.")
    if hole_edge_to_edge_spacing_um < 0.0:
        raise ValueError(
            "chip_pattern_dimensions['hole_edge_to_edge_spacing_um'] must be non-negative."
        )
    if hole_intensity_factor <= 0.0 or gold_intensity_factor <= 0.0:
        raise ValueError(
            "chip_pattern_dimensions['hole_intensity_factor'] and "
            "'gold_intensity_factor' must be positive."
        )

    pixel_size_nm = float(params["pixel_size_nm"])
    if pixel_size_nm <= 0.0:
        raise ValueError("PARAMS['pixel_size_nm'] must be positive.")

    os_factor = float(params.get("psf_oversampling_factor", 1.0))
    if os_factor <= 0.0:
        raise ValueError("PARAMS['psf_oversampling_factor'] must be positive.")

    # Generate pattern maps at both resolutions:
    #   - pattern_final: at the final image resolution, using the nominal pixel size.
    #   - pattern_os: at the oversampled resolution, with a smaller effective pixel size.
    pattern_final = _generate_gold_hole_pattern(
        shape=final_fov_shape,
        pixel_size_nm=pixel_size_nm,
        hole_diameter_um=hole_diameter_um,
        hole_edge_to_edge_spacing_um=hole_edge_to_edge_spacing_um,
        hole_intensity_factor=hole_intensity_factor,
        gold_intensity_factor=gold_intensity_factor,
    )

    pattern_os = _generate_gold_hole_pattern(
        shape=fov_shape_os,
        pixel_size_nm=pixel_size_nm / os_factor,
        hole_diameter_um=hole_diameter_um,
        hole_edge_to_edge_spacing_um=hole_edge_to_edge_spacing_um,
        hole_intensity_factor=hole_intensity_factor,
        gold_intensity_factor=gold_intensity_factor,
    )

    # Construct reference field maps by modulating the amplitude with the square
    # root of the pattern maps. This ensures that the local reference
    # *intensity* scales proportionally to the pattern, which is physically
    # reasonable in a reflection-based iSCAT configuration.
    E_ref_os = (E_ref_amplitude * np.sqrt(pattern_os)).astype(np.complex128)
    E_ref_final = (E_ref_amplitude * np.sqrt(pattern_final)).astype(np.complex128)

    # Construct the background intensity map at the final resolution by
    # multiplying the base background intensity with the pattern. This causes
    # the detector noise (shot and read noise) to carry the same spatial
    # structure, so that the chip pattern survives background subtraction in a
    # realistic way.
    background_final = (background_intensity * pattern_final).astype(float)

    return E_ref_os, E_ref_final, background_final