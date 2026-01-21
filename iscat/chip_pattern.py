import math
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
        hole_edge_to_edge_spacing_um (float): Gold spacing between hole edges
            in micrometers.
        hole_intensity_factor (float): Relative background intensity inside
            holes.
        gold_intensity_factor (float): Relative background intensity in gold
            regions.

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
        raise ValueError(
            "Computed pitch (hole_diameter_um + hole_edge_to_edge_spacing_um) "
            "must be positive."
        )

    hole_intensity_factor = float(hole_intensity_factor)
    gold_intensity_factor = float(gold_intensity_factor)
    if hole_intensity_factor <= 0.0 or gold_intensity_factor <= 0.0:
        raise ValueError(
            "hole_intensity_factor and gold_intensity_factor must be positive."
        )

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


def _generate_nanopillar_pattern(
    shape: tuple,
    pixel_size_nm: float,
    pillar_diameter_um: float,
    pillar_edge_to_edge_spacing_um: float,
    pillar_intensity_factor: float,
    background_intensity_factor: float,
) -> np.ndarray:
    """
    Generate a dimensionless intensity pattern map for a nanopillar array.

    Geometry:
        - Circular pillars of diameter `pillar_diameter_um`.
        - Square grid with center-to-center pitch:
              pitch_um = pillar_diameter_um + pillar_edge_to_edge_spacing_um

    The pattern returned by this helper labels pixels inside a pillar with
    `pillar_intensity_factor` and pixels in the background with
    `background_intensity_factor`, followed by a normalization to unit mean.

    Implementation detail:
        This uses the same circular-lattice generator as the gold-hole pattern,
        but interprets the "hole" region as the pillar region and the "gold"
        region as the background.

    Args:
        shape (tuple[int, int]): (height, width) of the desired pattern map.
        pixel_size_nm (float): Physical pixel size in nanometers for this grid.
        pillar_diameter_um (float): Pillar diameter in micrometers.
        pillar_edge_to_edge_spacing_um (float): Spacing between pillar edges
            in micrometers.
        pillar_intensity_factor (float): Relative background intensity on top
            of pillars.
        background_intensity_factor (float): Relative background intensity in
            regions outside pillars.

    Returns:
        np.ndarray: 2D array of shape `shape`, dtype float, dimensionless
            multiplicative factors with mean ~1.0.
    """
    return _generate_gold_hole_pattern(
        shape=shape,
        pixel_size_nm=pixel_size_nm,
        hole_diameter_um=pillar_diameter_um,
        hole_edge_to_edge_spacing_um=pillar_edge_to_edge_spacing_um,
        hole_intensity_factor=pillar_intensity_factor,
        gold_intensity_factor=background_intensity_factor,
    )


def _resolve_gold_hole_parameters(params: dict) -> dict:
    """
    Resolve geometry and optical-intensity parameters for the gold film with
    circular holes from the global PARAMS dictionary.

    This helper is the single source of truth for:
        - hole_diameter_um
        - hole_edge_to_edge_spacing_um
        - hole_depth_nm
        - hole_intensity_factor
        - gold_intensity_factor
        - pitch_um
        - radius_um

    Both "default_gold_holes" and "lab_default_gold_holes" currently share the
    same default values when fields are omitted, matching the existing code
    behavior. Differences between these presets can be introduced here later
    without touching the calling sites.
    """
    dims = params.get("chip_pattern_dimensions", {})
    if not isinstance(dims, dict):
        raise TypeError(
            "PARAMS['chip_pattern_dimensions'] must be a dictionary when "
            "using chip_pattern_model 'gold_holes_v1'."
        )

    # Substrate preset is kept for potential future specialization of defaults.
    substrate_preset_raw = params.get("chip_substrate_preset", "empty_background")
    substrate_preset = str(substrate_preset_raw).strip().lower()

    # Geometry defaults (identical to previous implementation).
    hole_diameter_um = float(dims.get("hole_diameter_um", 15.0))
    hole_edge_to_edge_spacing_um = float(dims.get("hole_edge_to_edge_spacing_um", 2.0))
    hole_depth_nm = float(dims.get("hole_depth_nm", 20.0))  # bookkeeping only

    if hole_diameter_um <= 0.0:
        raise ValueError(
            "chip_pattern_dimensions['hole_diameter_um'] must be positive."
        )
    if hole_edge_to_edge_spacing_um < 0.0:
        raise ValueError(
            "chip_pattern_dimensions['hole_edge_to_edge_spacing_um'] must be "
            "non-negative."
        )

    pitch_um = hole_diameter_um + hole_edge_to_edge_spacing_um
    if pitch_um <= 0.0:
        raise ValueError(
            "Computed pitch (hole_diameter_um + hole_edge_to_edge_spacing_um) "
            "must be positive."
        )

    radius_um = hole_diameter_um / 2.0

    # Optical intensity parameters (identical defaults to previous code).
    hole_intensity_factor = float(dims.get("hole_intensity_factor", 0.7))
    gold_intensity_factor = float(dims.get("gold_intensity_factor", 1.0))

    if hole_intensity_factor <= 0.0 or gold_intensity_factor <= 0.0:
        raise ValueError(
            "chip_pattern_dimensions['hole_intensity_factor'] and "
            "'gold_intensity_factor' must be positive."
        )

    return {
        "hole_diameter_um": hole_diameter_um,
        "hole_edge_to_edge_spacing_um": hole_edge_to_edge_spacing_um,
        "hole_depth_nm": hole_depth_nm,
        "hole_intensity_factor": hole_intensity_factor,
        "gold_intensity_factor": gold_intensity_factor,
        "pitch_um": pitch_um,
        "radius_um": radius_um,
        "substrate_preset": substrate_preset,
    }


def _resolve_nanopillar_parameters(params: dict) -> dict:
    """
    Resolve geometry and optical-intensity parameters for a circular nanopillar
    array from the global PARAMS dictionary.

    This helper is the single source of truth for:
        - pillar_diameter_um
        - pillar_edge_to_edge_spacing_um
        - pillar_height_nm
        - pillar_intensity_factor
        - background_intensity_factor
        - pitch_um
        - radius_um
    """
    dims = params.get("chip_pattern_dimensions", {})
    if not isinstance(dims, dict):
        raise TypeError(
            "PARAMS['chip_pattern_dimensions'] must be a dictionary when "
            "using chip_pattern_model 'nanopillars_v1'."
        )

    substrate_preset_raw = params.get("chip_substrate_preset", "empty_background")
    substrate_preset = str(substrate_preset_raw).strip().lower()

    pillar_diameter_um = float(dims.get("pillar_diameter_um", 1.0))
    pillar_edge_to_edge_spacing_um = float(
        dims.get("pillar_edge_to_edge_spacing_um", 2.0)
    )
    pillar_height_nm = float(dims.get("pillar_height_nm", 20.0))  # bookkeeping only

    if pillar_diameter_um <= 0.0:
        raise ValueError(
            "chip_pattern_dimensions['pillar_diameter_um'] must be positive."
        )
    if pillar_edge_to_edge_spacing_um < 0.0:
        raise ValueError(
            "chip_pattern_dimensions['pillar_edge_to_edge_spacing_um'] must be "
            "non-negative."
        )

    pitch_um = pillar_diameter_um + pillar_edge_to_edge_spacing_um
    if pitch_um <= 0.0:
        raise ValueError(
            "Computed pitch (pillar_diameter_um + pillar_edge_to_edge_spacing_um) "
            "must be positive."
        )

    radius_um = pillar_diameter_um / 2.0

    pillar_intensity_factor = float(dims.get("pillar_intensity_factor", 1.3))
    background_intensity_factor = float(dims.get("background_intensity_factor", 1.0))

    if pillar_intensity_factor <= 0.0 or background_intensity_factor <= 0.0:
        raise ValueError(
            "chip_pattern_dimensions['pillar_intensity_factor'] and "
            "'background_intensity_factor' must be positive."
        )

    return {
        "pillar_diameter_um": pillar_diameter_um,
        "pillar_edge_to_edge_spacing_um": pillar_edge_to_edge_spacing_um,
        "pillar_height_nm": pillar_height_nm,
        "pillar_intensity_factor": pillar_intensity_factor,
        "background_intensity_factor": background_intensity_factor,
        "pitch_um": pitch_um,
        "radius_um": radius_um,
        "substrate_preset": substrate_preset,
    }


def _map_position_nm_to_gold_hole_unit_cell(
    params: dict,
    x_nm: float,
    y_nm: float,
    pitch_um: float,
) -> tuple:
    """
    Map a lateral position (x_nm, y_nm) from the simulation's FOV coordinates
    into the canonical unit cell of a circular lattice (e.g., the gold-hole
    lattice or a nanopillar lattice).

    The mapping is consistent with _generate_gold_hole_pattern and the
    Brownian-exclusion logic:

        - The field of view is a square of side length
              img_size_nm = image_size_pixels * pixel_size_nm
          centered at (0, 0) in physical coordinates.

        - The input positions (x_nm, y_nm) are interpreted as distances from
          the FOV corner (0, 0). We convert them to centered coordinates,
          then to micrometers, and finally wrap them into the interval
          [-pitch_um/2, pitch_um/2) in each dimension.

    Returns:
        dx_um, dy_um, r_um, x_um, y_um, img_size_nm
        where:
            (dx_um, dy_um) are coordinates inside the canonical unit cell,
            r_um = sqrt(dx_um**2 + dy_um**2),
            (x_um, y_um) are the centered coordinates in micrometers, and
            img_size_nm is the full FOV extent in nanometers.
    """
    img_size_pixels = int(params["image_size_pixels"])
    pixel_size_nm = float(params["pixel_size_nm"])
    if img_size_pixels <= 0 or pixel_size_nm <= 0.0:
        raise ValueError(
            "PARAMS['image_size_pixels'] and PARAMS['pixel_size_nm'] must be "
            "positive when substrate exclusion is active."
        )

    img_size_nm = img_size_pixels * pixel_size_nm

    # Centered coordinates in nm.
    x_nm_centered = float(x_nm) - img_size_nm / 2.0
    y_nm_centered = float(y_nm) - img_size_nm / 2.0

    # Convert to micrometers.
    x_um = x_nm_centered * 1e-3
    y_um = y_nm_centered * 1e-3

    # Map into the unit cell using periodic wrapping.
    half_pitch = pitch_um / 2.0
    dx_um = (x_um + half_pitch) % pitch_um - half_pitch
    dy_um = (y_um + half_pitch) % pitch_um - half_pitch
    r_um = math.hypot(dx_um, dy_um)

    return dx_um, dy_um, r_um, x_um, y_um, img_size_nm


def is_position_in_chip_solid(params: dict, x_nm: float, y_nm: float) -> bool:
    """
    Determine whether a lateral position (x_nm, y_nm) lies inside a solid region
    of the configured chip/substrate pattern.

    This function is used by the Brownian motion simulator to enforce the
    design requirement that particles cannot occupy space where gold or other
    solid structures are present (CDD Sections 3.2 and 3.6). It operates in
    continuous physical coordinates so it can be called per Brownian step
    without constructing any additional images.

    Current behavior:

        - If `chip_pattern_enabled` is False, or `chip_substrate_preset` is
          "empty_background", or `chip_pattern_model` is "none", the function
          always returns False (no solid regions are modeled).

        - For `chip_pattern_model == "gold_holes_v1"` with `chip_substrate_preset`
          in {"default_gold_holes", "lab_default_gold_holes"}, the substrate is
          modeled as a gold film with circular holes on a square grid. The gold
          film is treated as occupying all lateral area outside the holes. The
          function returns True when the given (x_nm, y_nm) projects into the
          gold film (solid) and False when it lands inside a hole (fluid).

        - For `chip_pattern_model == "nanopillars_v1"` with
          `chip_substrate_preset == "nanopillars"`, the substrate is modeled as
          circular gold pillars on a non-reflective background. The pillars are
          treated as solid; the background is fluid. The function returns True
          when the position lies inside a pillar and False otherwise.

    Future chip pattern models and substrate presets can extend this function
    in a backward-compatible way by adding additional geometry branches.

    Args:
        params (dict): Global simulation parameter dictionary (PARAMS).
        x_nm (float): Lateral x-position of the particle center in nanometers,
            measured from the field-of-view corner (same convention as
            simulate_trajectories and rendering).
        y_nm (float): Lateral y-position of the particle center in nanometers,
            measured from the field-of-view corner.

    Returns:
        bool: True if the position lies inside a solid region of the chip/
        substrate (e.g., gold film or pillars), False otherwise.
    """
    chip_enabled = bool(params.get("chip_pattern_enabled", False))

    pattern_model_raw = params.get("chip_pattern_model", "none")
    pattern_model = str(pattern_model_raw).strip().lower()

    substrate_preset_raw = params.get("chip_substrate_preset", "empty_background")
    substrate_preset = str(substrate_preset_raw).strip().lower()

    # No solid structure when the background is empty or the pattern model is
    # explicitly disabled.
    if not chip_enabled or substrate_preset == "empty_background" or pattern_model == "none":
        return False

    # Gold film with circular holes: solid region is outside the holes.
    if pattern_model == "gold_holes_v1":
        if substrate_preset not in ("default_gold_holes", "lab_default_gold_holes"):
            # Unknown substrate preset for this pattern model; treat as no solid regions.
            return False

        geom = _resolve_gold_hole_parameters(params)
        radius_um = geom["radius_um"]
        pitch_um = geom["pitch_um"]

        _, _, r_um, _, _, _ = _map_position_nm_to_gold_hole_unit_cell(
            params, x_nm, y_nm, pitch_um
        )

        inside_hole = (r_um <= radius_um)
        # In this geometry, the gold film occupies all area outside the holes.
        return not inside_hole

    # Nanopillar array: solid region is inside the pillars.
    if pattern_model == "nanopillars_v1":
        if substrate_preset != "nanopillars":
            return False

        geom = _resolve_nanopillar_parameters(params)
        radius_um = geom["radius_um"]
        pitch_um = geom["pitch_um"]

        _, _, r_um, _, _, _ = _map_position_nm_to_gold_hole_unit_cell(
            params, x_nm, y_nm, pitch_um
        )

        inside_pillar = (r_um <= radius_um)
        return inside_pillar

    # For unimplemented pattern models we conservatively treat the domain as
    # fluid everywhere so that enabling such a model does not silently
    # introduce inconsistent dynamics.
    return False


def project_position_to_fluid_region(params: dict, x_nm: float, y_nm: float) -> tuple:
    """
    Given a lateral position (x_nm, y_nm), project it into the nearest fluid
    region of the chip (i.e., outside solid regions) if it currently lies in a
    solid region.

    This function is used to correct Brownian steps that would otherwise place
    the particle center inside the solid chip/substrate. It preserves the
    underlying random step statistics (no resampling) by deterministically
    mapping such positions back to the nearest point inside the allowed fluid
    region, just inside or outside the solid/fluid boundary as appropriate.

    Behavior is defined only for:
        chip_pattern_enabled = True, and one of:
            - chip_pattern_model  = "gold_holes_v1",
              chip_substrate_preset in {"default_gold_holes", "lab_default_gold_holes"}:
                solid = gold film outside holes; projection moves a point from
                gold into the nearest point just inside a hole.

            - chip_pattern_model  = "nanopillars_v1",
              chip_substrate_preset == "nanopillars":
                solid = circular pillars; projection moves a point from inside
                a pillar to the nearest point just outside the pillar.

    For all other configurations, the input position is returned unchanged.

    Args:
        params (dict): Global simulation parameter dictionary (PARAMS).
        x_nm (float): Lateral x-position of the particle center in nanometers.
        y_nm (float): Lateral y-position of the particle center in nanometers.

    Returns:
        tuple[float, float]: Corrected (x_nm, y_nm) in nanometers, guaranteed
        (under the supported configurations) to lie in a fluid region.
    """
    # If there is no substrate exclusion or the position is already fluid,
    # return it unchanged.
    if not is_position_in_chip_solid(params, x_nm, y_nm):
        return float(x_nm), float(y_nm)

    chip_enabled = bool(params.get("chip_pattern_enabled", False))
    pattern_model_raw = params.get("chip_pattern_model", "none")
    pattern_model = str(pattern_model_raw).strip().lower()
    substrate_preset_raw = params.get("chip_substrate_preset", "empty_background")
    substrate_preset = str(substrate_preset_raw).strip().lower()

    if not chip_enabled:
        # Unsupported configuration for projection logic; leave position as-is.
        return float(x_nm), float(y_nm)

    # --- Gold film with circular holes: project from gold into a hole ---
    if (
        pattern_model == "gold_holes_v1"
        and substrate_preset in ("default_gold_holes", "lab_default_gold_holes")
    ):
        geom = _resolve_gold_hole_parameters(params)
        radius_um = geom["radius_um"]
        pitch_um = geom["pitch_um"]

        dx_um, dy_um, r_um, x_um, y_um, img_size_nm = _map_position_nm_to_gold_hole_unit_cell(
            params, x_nm, y_nm, pitch_um
        )

        # If for some reason we are already in the hole (should not happen here),
        # leave unchanged.
        if r_um <= radius_um or r_um == 0.0:
            return float(x_nm), float(y_nm)

        # Project radially from the current position in the unit cell to just
        # inside the hole boundary.
        # Use a small inward offset (1 nm) to avoid numerical ambiguity exactly
        # on the boundary.
        r_target_um = max(radius_um - 1e-3, 0.0)  # 1e-3 µm = 1 nm
        scale = r_target_um / r_um if r_um > 0.0 else 0.0

        new_dx_um = dx_um * scale
        new_dy_um = dy_um * scale

        # Compute how much we moved within the unit cell.
        delta_dx_um = new_dx_um - dx_um
        delta_dy_um = new_dy_um - dy_um

        # Apply the same deltas in the global (centered) coordinates. This keeps
        # the particle in the same lattice cell while pushing it into the hole.
        new_x_um = x_um + delta_dx_um
        new_y_um = y_um + delta_dy_um

        new_x_nm_centered = new_x_um * 1e3
        new_y_nm_centered = new_y_um * 1e3

        new_x_nm = new_x_nm_centered + img_size_nm / 2.0
        new_y_nm = new_y_nm_centered + img_size_nm / 2.0

        return float(new_x_nm), float(new_y_nm)

    # --- Nanopillars: project from pillar interior into background fluid ---
    if pattern_model == "nanopillars_v1" and substrate_preset == "nanopillars":
        geom = _resolve_nanopillar_parameters(params)
        radius_um = geom["radius_um"]
        pitch_um = geom["pitch_um"]

        dx_um, dy_um, r_um, x_um, y_um, img_size_nm = _map_position_nm_to_gold_hole_unit_cell(
            params, x_nm, y_nm, pitch_um
        )

        # In this geometry, solid region is inside the pillar (r <= radius).
        # We project radially outward to just outside the pillar boundary.
        if r_um == 0.0:
            # Extremely unlikely: exactly at pillar center. Choose a fixed
            # direction along +x for the outward step.
            new_dx_um = radius_um + 1e-3  # 1 nm outside the pillar radius
            new_dy_um = 0.0
        else:
            r_target_um = radius_um + 1e-3  # 1 nm outside the pillar radius
            scale = r_target_um / r_um
            new_dx_um = dx_um * scale
            new_dy_um = dy_um * scale

        delta_dx_um = new_dx_um - dx_um
        delta_dy_um = new_dy_um - dy_um

        new_x_um = x_um + delta_dx_um
        new_y_um = y_um + delta_dy_um

        new_x_nm_centered = new_x_um * 1e3
        new_y_nm_centered = new_y_um * 1e3

        new_x_nm = new_x_nm_centered + img_size_nm / 2.0
        new_y_nm = new_y_nm_centered + img_size_nm / 2.0

        return float(new_x_nm), float(new_y_nm)

    # Unsupported configuration for projection logic; leave position as-is.
    return float(x_nm), float(y_nm)


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
    pattern presets such as a gold film with circular holes or a nanopillar
    array. The returned maps are:

        - E_ref_os: complex reference field on the oversampled field of view.
        - E_ref_final: complex reference field at the final image resolution.
        - background_final: scalar background intensity map (camera counts)
          at the final image resolution.

    The chip pattern is represented as a dimensionless spatial multiplier on
    both the reference field amplitude and the background intensity. This
    satisfies the design requirement that the pattern be part of the physical
    image formation (via E_ref) and the noise model (via background_intensity),
    while keeping the rest of the rendering pipeline unchanged.

    The same geometric parameters that define the pattern here are also used
    by is_position_in_chip_solid / project_position_to_fluid_region to enforce
    substrate exclusion in the Brownian motion simulation.

    Behavior:
        - If chip_pattern_enabled is False, or chip_substrate_preset is
          "empty_background", or chip_pattern_model is "none", the maps are
          spatially uniform as in the original implementation.
        - Otherwise, the pattern model and substrate preset determine the
          spatial maps. Currently, "gold_holes_v1" with presets
          "default_gold_holes" and "lab_default_gold_holes", and
          "nanopillars_v1" with preset "nanopillars" are supported.

    Note:
        This function is intentionally time-independent. Any temporal evolution
        of the chip pattern contrast is applied at render time via
        `compute_contrast_scale_for_frame` and the reconstructed dimensionless
        pattern maps.
    """
    E_ref_amplitude = float(params["reference_field_amplitude"])
    background_intensity = float(params["background_intensity"])

    chip_enabled = bool(params.get("chip_pattern_enabled", False))

    pattern_model_raw = params.get("chip_pattern_model", "gold_holes_v1")
    substrate_preset_raw = params.get("chip_substrate_preset", "empty_background")

    # Normalize to lowercase/stripped so behavior is consistent with the
    # trajectory-side functions (is_position_in_chip_solid, etc.).
    pattern_model = str(pattern_model_raw).strip().lower()
    substrate_preset = str(substrate_preset_raw).strip().lower()

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

    pixel_size_nm = float(params["pixel_size_nm"])
    if pixel_size_nm <= 0.0:
        raise ValueError("PARAMS['pixel_size_nm'] must be positive.")

    os_factor = float(params.get("psf_oversampling_factor", 1.0))
    if os_factor <= 0.0:
        raise ValueError("PARAMS['psf_oversampling_factor'] must be positive.")

    # At this point, a chip pattern is requested. We validate and construct the
    # appropriate pattern maps based on the selected model and substrate preset.
    if pattern_model == "gold_holes_v1":
        if substrate_preset not in ("default_gold_holes", "lab_default_gold_holes"):
            raise ValueError(
                f"Unsupported chip_substrate_preset '{substrate_preset_raw}' for "
                "chip_pattern_model 'gold_holes_v1'. Supported presets are "
                "'empty_background', 'default_gold_holes', and 'lab_default_gold_holes'."
            )

        geom = _resolve_gold_hole_parameters(params)
        hole_diameter_um = geom["hole_diameter_um"]
        hole_edge_to_edge_spacing_um = geom["hole_edge_to_edge_spacing_um"]
        hole_depth_nm = geom["hole_depth_nm"]  # currently unused in optics
        hole_intensity_factor = geom["hole_intensity_factor"]
        gold_intensity_factor = geom["gold_intensity_factor"]

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

    elif pattern_model == "nanopillars_v1":
        if substrate_preset != "nanopillars":
            raise ValueError(
                f"Unsupported chip_substrate_preset '{substrate_preset_raw}' for "
                "chip_pattern_model 'nanopillars_v1'. Supported presets are "
                "'empty_background' and 'nanopillars'."
            )

        geom = _resolve_nanopillar_parameters(params)
        pillar_diameter_um = geom["pillar_diameter_um"]
        pillar_edge_to_edge_spacing_um = geom["pillar_edge_to_edge_spacing_um"]
        pillar_height_nm = geom["pillar_height_nm"]  # currently unused in optics
        pillar_intensity_factor = geom["pillar_intensity_factor"]
        background_intensity_factor = geom["background_intensity_factor"]

        pattern_final = _generate_nanopillar_pattern(
            shape=final_fov_shape,
            pixel_size_nm=pixel_size_nm,
            pillar_diameter_um=pillar_diameter_um,
            pillar_edge_to_edge_spacing_um=pillar_edge_to_edge_spacing_um,
            pillar_intensity_factor=pillar_intensity_factor,
            background_intensity_factor=background_intensity_factor,
        )

        pattern_os = _generate_nanopillar_pattern(
            shape=fov_shape_os,
            pixel_size_nm=pixel_size_nm / os_factor,
            pillar_diameter_um=pillar_diameter_um,
            pillar_edge_to_edge_spacing_um=pillar_edge_to_edge_spacing_um,
            pillar_intensity_factor=pillar_intensity_factor,
            background_intensity_factor=background_intensity_factor,
        )

    else:
        raise ValueError(
            f"Unsupported chip_pattern_model '{pattern_model_raw}'. "
            "Currently supported models are 'none', 'gold_holes_v1', and 'nanopillars_v1'."
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


def compute_contrast_scale_for_frame(
    params: dict,
    frame_index: int,
    num_frames: int,
) -> float:
    """
    Compute the scalar contrast scale factor for the chip pattern in a given
    frame, based on the selected chip_pattern_contrast_model.

    The scale factor alpha_f returned by this function is used to modulate the
    deviation of the dimensionless pattern from unity:

        p_frame(x, y) = 1 + alpha_f * (p_base(x, y) - 1),

    where p_base(x, y) is the base pattern with mean 1.0. With this
    construction, the global mean of p_frame remains 1.0 for any alpha_f in
    [0, 1], so the overall brightness is controlled purely by
    background_intensity.

    Supported models:
        - "static":
            alpha_f = 1.0 for all frames (no temporal contrast change).

        - "time_dependent_v1":
            alpha_f decays linearly from 1.0 at the first frame to
            1.0 - A at the last frame, where A is given by
            PARAMS["chip_pattern_contrast_amplitude"] in [0, 1]. This models a
            deterministic, monotonic reduction in chip-pattern contrast over
            the duration of the video (e.g., due to slow external drifts).

    Args:
        params (dict): Global simulation parameter dictionary (PARAMS).
        frame_index (int): Zero-based index of the current frame.
        num_frames (int): Total number of frames in the video.

    Returns:
        float: Contrast scale factor alpha_f.

    Raises:
        ValueError: If an unsupported contrast model is selected or if the
            frame index is out of range.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive when computing contrast scale.")
    if frame_index < 0 or frame_index >= num_frames:
        raise ValueError(
            f"frame_index={frame_index} is out of range for num_frames={num_frames}."
        )

    model_raw = params.get("chip_pattern_contrast_model", "static")
    model = str(model_raw).strip().lower()

    if model == "static":
        # No temporal evolution: always use the base pattern.
        return 1.0

    if model == "time_dependent_v1":
        # Maximum fractional reduction in contrast A, clamped to [0, 1].
        amplitude = float(params.get("chip_pattern_contrast_amplitude", 0.5))
        if amplitude <= 0.0:
            return 1.0
        if amplitude > 1.0:
            amplitude = 1.0

        # Normalized time coordinate in [0, 1]. For a single-frame video, we
        # treat the contrast as unchanged.
        if num_frames == 1:
            t_frac = 0.0
        else:
            t_frac = frame_index / float(num_frames - 1)

        # Linear decay from alpha = 1.0 at the first frame to
        # alpha = 1.0 - amplitude at the last frame.
        alpha = 1.0 - amplitude * t_frac
        return float(alpha)

    raise ValueError(
        f"Unsupported chip_pattern_contrast_model '{model_raw}'. "
        "Supported models are 'static' and 'time_dependent_v1'."
    )