# File: single_frame_viewer.py
import argparse
import copy
import json
import os
from typing import Any

import cv2
import numpy as np

from config import PARAMS
from main import generate_single_frame_views
from param_utils import build_params_from_controls
from postprocessing import compute_single_frame_contrast


def _resolve_output_dir(output_dir_arg: str | None) -> str:
    """
    Resolve and create the output directory for the single-frame viewer.

    If output_dir_arg is provided, it is used directly. Otherwise a default
    directory on the user's Desktop is chosen.
    """
    if output_dir_arg:
        out_dir = os.path.abspath(os.path.expanduser(output_dir_arg))
    else:
        out_dir = os.path.join(
            os.path.expanduser("~"),
            "Desktop",
            "iscat_single_frame",
        )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _compute_center_position_nm(params_viewer: dict) -> tuple[float, float]:
    """
    Compute the (x, y) center position in nanometers for the given parameter
    dictionary based on image_size_pixels and pixel_size_nm.
    """
    image_size_pixels = float(params_viewer["image_size_pixels"])
    pixel_size_nm = float(params_viewer["pixel_size_nm"])
    L_nm = image_size_pixels * pixel_size_nm
    x_center_nm = 0.5 * L_nm
    y_center_nm = 0.5 * L_nm
    return x_center_nm, y_center_nm


def _pick_initial_xy_outside_solid(params: dict) -> tuple[float, float]:
    """
    Heuristic helper: choose an initial (x, y) that is visually near the
    center but much less likely to fall into a solid chip/substrate region.

    Rationale:
        For regular gold-hole grids, the geometric center of the FOV may land
        in gold or in a hole depending on global phase and offsets. The
        trajectory subsystem enforces an exclusion rule that disallows
        starting inside solid regions. Rather than trying to reconstruct that
        exact geometry here, we bias the initial position slightly away from
        the exact center by a modest fraction of the FOV so that we are
        typically in fluid above the substrate.

    Strategy:
        - Start from the geometric center.
        - Apply a small, deterministic offset in one direction proportional to
          the FOV size (e.g., ~5% of the linear size) to avoid the very center
          of a periodic grid.
        - This keeps the particle near the visual center while avoiding a
        configuration that is known to be inside a solid region for the
        current chip pattern.
    """
    x_center_nm, y_center_nm = _compute_center_position_nm(params)
    image_size_pixels = float(params["image_size_pixels"])
    pixel_size_nm = float(params["pixel_size_nm"])

    # Offset of ~5% of the FOV size in nanometers.
    offset_fraction = 0.05
    offset_nm = offset_fraction * image_size_pixels * pixel_size_nm

    x_init = x_center_nm + offset_nm
    y_init = y_center_nm + offset_nm

    return x_init, y_init


def _tailor_params_for_single_centered_particle(params_base: dict, output_dir: str) -> dict:
    """
    Create a tailored parameter dictionary for a single, centered, static
    particle observed in exactly one frame, leaving the rest of the pipeline
    unchanged.

    This function:
        - Starts from config.PARAMS but first applies the central parameter
          schema via build_params_from_controls to obtain a consistent base.
        - Forces exactly one frame by adjusting duration_seconds.
        - Ensures a single particle with appropriate arrays of length 1.
        - Places the particle near the center in x and y.
        - Places the particle at a fixed z below focus with unconstrained
          z-motion.
        - Disables masks, trackability, and motion blur.
        - Redirects all outputs into the provided output_dir.
        - Uses a background subtraction method that is meaningful for a
          single noisy frame.
    """
    # Start from the schema-controlled base derived from the original PARAMS.
    # For now we do not apply any explicit overrides from the CLI; an empty
    # control_values dict means "use config.PARAMS + schema defaults".
    schema_base = build_params_from_controls(control_values={})

    # Work on a copy so further tailoring does not affect other callers.
    params = copy.deepcopy(schema_base)

    # --- Video configuration: force exactly one frame ---
    fps = float(params.get("fps", 24.0))
    if fps <= 0.0:
        fps = 24.0
        params["fps"] = fps

    # Choose duration such that fps * duration_seconds = 1 frame.
    duration_seconds = 1.0 / fps
    params["duration_seconds"] = duration_seconds

    # --- Single particle setup ---
    params["num_particles"] = 1

    # Reuse the first diameter/material from the schema-controlled base.
    base_diameters = params.get("particle_diameters_nm", [100.0])
    default_diameter = float(base_diameters[0]) if base_diameters else 100.0
    params["particle_diameters_nm"] = [default_diameter]

    base_materials = params.get("particle_materials", ["Gold"])
    default_material = base_materials[0] if base_materials else "Gold"
    params["particle_materials"] = [default_material]

    # Rely on material-based lookup for refractive indices; set to [None].
    params["particle_refractive_indices"] = [None]

    # Simple spherical particle for this viewer.
    params["particle_shape_models"] = ["spherical"]

    # Ensure a single signal multiplier.
    base_signal_multipliers = params.get("particle_signal_multipliers", [1.0])
    default_multiplier = float(base_signal_multipliers[0]) if base_signal_multipliers else 1.0
    params["particle_signal_multipliers"] = [default_multiplier]

    # --- Chip/substrate configuration for viewer ---
    params["chip_pattern_enabled"] = False

    # --- Initial position: near center ---
    x_init_nm, y_init_nm = _compute_center_position_nm(params)

    # Use the configured z_stack_range_nm to choose a z that is safely within
    # the PSF stack but below focus. For the viewer, we adopt the convention
    # that negative z is "below focus" and use one quarter of the range.
    z_stack_range_nm = float(params.get("z_stack_range_nm", 30500.0))
    z_initial_nm = -0.25 * z_stack_range_nm

    # Use unconstrained z-motion for this viewer so we are free to place the
    # particle at a negative z without reflective constraints.
    params["z_motion_constraint_model"] = "unconstrained"

    # Explicit initial position for the single particle.
    params["particle_initial_positions_nm"] = [[x_init_nm, y_init_nm, z_initial_nm]]

    # --- Disable masks, trackability, and motion blur for simplicity ---
    params["mask_generation_enabled"] = False
    params["trackability_enabled"] = False
    params["motion_blur_enabled"] = False
    params["motion_blur_subsamples"] = 1

    # --- Viewer-specific postprocessing choice ---
    params["background_subtraction_method"] = "reference_frame"

    # NOTE: We intentionally leave shot_noise_enabled and
    # gaussian_noise_enabled as they are in the schema-controlled base so
    # that the single frame retains realistic noise characteristics.

    # --- Output paths within the viewer directory ---
    video_path = os.path.join(output_dir, "single_frame.mp4")
    params["output_filename"] = video_path

    # Masks are disabled, but the main pipeline expects this directory field
    # to exist; set it to a subdirectory of the viewer output dir.
    params["mask_output_directory"] = os.path.join(output_dir, "masks")

    # Ensure output directory exists.
    os.makedirs(os.path.dirname(params["output_filename"]), exist_ok=True)
    os.makedirs(params["mask_output_directory"], exist_ok=True)

    return params


def _complex_to_json(obj: complex) -> dict[str, float]:
    """
    Convert a complex number into a JSON-friendly dict representation.
    """
    return {"real": float(obj.real), "imag": float(obj.imag)}


def _numpy_to_native(obj: Any) -> Any:
    """
    Convert NumPy scalar/array types to plain Python types and lists so that
    the structure becomes JSON-serializable.

    - np.ndarray -> list
    - np.generic scalars -> Python scalars
    - complex numbers -> {"real": float, "imag": float}
    """
    if isinstance(obj, np.ndarray):
        # Convert array to nested lists, handling complex entries explicitly.
        if np.iscomplexobj(obj):
            return [
                _complex_to_json(v) if isinstance(v, complex) else _complex_to_json(complex(v))
                for v in obj.flatten()
            ]
        return obj.tolist()

    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, complex):
        return _complex_to_json(obj)

    return obj


def _make_params_json_serializable(params: dict) -> dict:
    """
    Recursively convert a parameter dictionary containing NumPy arrays,
    NumPy scalars, and complex numbers into a JSON-serializable structure.

    Complex numbers are represented as {"real": ..., "imag": ...}.
    """
    def convert(value: Any) -> Any:
        # Handle dicts
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}

        # Handle sequences (lists/tuples)
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]

        # First normalize NumPy / complex types to native.
        native = _numpy_to_native(value)

        # After normalization, complex numbers will have been converted already.
        if isinstance(native, dict):
            # Could be a pre-existing dict that should itself be converted.
            return {str(k): convert(v) for k, v in native.items()}

        return native

    return convert(params)


def _dump_params_to_json(params: dict, output_path: str) -> None:
    """
    Serialize the given parameter dictionary (after JSON-ification) to the
    specified JSON file path.
    """
    serializable_params = _make_params_json_serializable(params)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_params, f, indent=2, sort_keys=True)


def _save_frame_as_png(frame: np.ndarray, path: str) -> None:
    """
    Save a single 2D numeric frame as an 8-bit PNG, applying per-frame
    min/max scaling when necessary.

    If frame is already uint8, it is written as-is. For floating-point or
    higher-bit-depth integer arrays, the data are linearly mapped to [0, 255]
    based on their own min/max to ensure a visible dynamic range.
    """
    arr = np.asarray(frame)
    if arr.dtype == np.uint8:
        img = arr
    else:
        arr_float = arr.astype(float)
        vmin = np.min(arr_float)
        vmax = np.max(arr_float)
        if vmax > vmin:
            norm = (arr_float - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(arr_float, dtype=float)
        img = np.clip(norm * 255.0, 0, 255).astype(np.uint8)

    if not cv2.imwrite(path, img):
        print(f"Failed to write image to {path}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the single-frame viewer script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate a single iSCAT frame with a centered particle using the "
            "existing simulation pipeline, and save multiple 2D view images "
            "along with the resolved parameter dictionary."
        )
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory in which to store the single-frame video and PNG views "
            "plus parameter JSON. Defaults to ~/Desktop/iscat_single_frame."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional random seed for NumPy to make the viewer run "
            "reproducible."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point for the single-frame viewer.

    This script:
        - Builds a viewer-specific parameter dictionary starting from the
          global config.PARAMS but routed through the central parameter schema
          (param_schema.PARAM_SCHEMA) and helper (param_utils.build_params_from_controls).
        - Calls the viewer-core generate_single_frame_views(...) to compute
          in-memory views of the single frame (raw signal, reference, contrast,
          final 8-bit), without invoking the full run_simulation pipeline.
        - Saves multiple views derived from the frame:

              * single_frame_raw_signal.png    (raw signal frame, pre-subtraction)
              * single_frame_reference.png     (raw reference frame)
              * single_frame_contrast.png      (contrast view, single-frame)
              * single_frame_final.png         (final 8-bit frame)

          plus:
              * params_used.json               (fully resolved PARAMS)
    """
    args = parse_args()
    output_dir = _resolve_output_dir(args.output_dir)

    if args.seed is not None:
        np.random.seed(args.seed)

    # Tailor params for a single near-centered particle and a single frame,
    # starting from the schema-controlled PARAMS base.
    params_viewer = _tailor_params_for_single_centered_particle(PARAMS, output_dir)

    # Generate all in-memory views using the viewer-core function.
    views = generate_single_frame_views(params_viewer)

    raw_signal = views.get("raw_signal_frame", None)
    raw_reference = views.get("raw_reference_frame", None)
    contrast = views.get("contrast_frame", None)
    final_8bit = views.get("final_frame_8bit", None)
    params_resolved = views.get("params_resolved", params_viewer)

    # --- Save raw signal and reference views ---
    if raw_signal is not None:
        raw_signal_path = os.path.join(output_dir, "single_frame_raw_signal.png")
        _save_frame_as_png(raw_signal, raw_signal_path)
        print(f"Raw signal image: {raw_signal_path}")
    else:
        print("Raw signal frame was not produced; skipping raw signal export.")

    if raw_reference is not None:
        raw_reference_path = os.path.join(output_dir, "single_frame_reference.png")
        _save_frame_as_png(raw_reference, raw_reference_path)
        print(f"Raw reference image: {raw_reference_path}")
    else:
        print("Raw reference frame was not produced; skipping raw reference export.")

    # --- Save contrast view ---
    if contrast is None and raw_signal is not None and raw_reference is not None:
        try:
            contrast = compute_single_frame_contrast(raw_signal, raw_reference, params_resolved)
        except ValueError as exc:
            print(f"Could not compute single-frame contrast: {exc}")

    if contrast is not None:
        contrast_path = os.path.join(output_dir, "single_frame_contrast.png")
        _save_frame_as_png(contrast, contrast_path)
        print(f"Contrast image: {contrast_path}")
    else:
        print("Skipping contrast export due to missing contrast frame.")

    # --- Save the final postprocessed 8-bit frame ---
    if final_8bit is not None:
        final_image_path = os.path.join(output_dir, "single_frame_final.png")
        _save_frame_as_png(final_8bit, final_image_path)
        print(f"Final processed image: {final_image_path}")
    else:
        print("No final 8-bit frame was produced; skipping final PNG export.")

    # Dump the fully resolved parameter dictionary after the run.
    params_json_path = os.path.join(output_dir, "params_used.json")
    _dump_params_to_json(params_resolved, params_json_path)
    print(f"Resolved parameters: {params_json_path}")

    print("Single-frame viewer run complete.")


if __name__ == "__main__":
    main()