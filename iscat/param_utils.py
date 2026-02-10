from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from config import PARAMS as BASE_PARAMS
from param_schema import PARAM_SCHEMA, ParamSpec


def _coerce_bool(value: Any) -> bool:
    """
    Convert a variety of truthy/falsey representations into a proper bool.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        return v in ("1", "true", "yes", "on", "y", "t")
    return bool(value)


def _validate_and_normalize_value(spec: ParamSpec, raw_value: Any) -> Any:
    """
    Validate and normalize a raw control value according to the parameter
    specification.

    This enforces:
      - type coercion (float/int/bool/enum)
      - min/max bounds for numeric types (if provided)
      - choices restriction for enums, with fallback to default
    """
    ptype = spec["type"]

    if ptype == "float":
        value = float(raw_value)
        if "min" in spec and value < spec["min"]:
            value = float(spec["min"])
        if "max" in spec and value > spec["max"]:
            value = float(spec["max"])
        return value

    if ptype == "int":
        value = int(raw_value)
        if "min" in spec and value < spec["min"]:
            value = int(spec["min"])
        if "max" in spec and value > spec["max"]:
            value = int(spec["max"])
        return value

    if ptype == "bool":
        return _coerce_bool(raw_value)

    if ptype == "enum":
        choices = spec.get("choices", [])
        if raw_value in choices:
            return raw_value
        # Try a case-insensitive match for strings if possible
        if isinstance(raw_value, str):
            lowered = raw_value.strip().lower()
            for c in choices:
                if isinstance(c, str) and c.strip().lower() == lowered:
                    return c
        # Fallback to default or first choice
        if "default" in spec:
            default = spec["default"]
            if default in choices:
                return default
        return choices[0] if choices else raw_value

    # Unknown type; return raw value unchanged
    return raw_value


def get_default_control_values() -> Dict[str, Any]:
    """
    Return a dict of schema_key -> default control value.

    The default for each control is taken from, in order of precedence:
      1. BASE_PARAMS at the underlying PARAMS key, if present.
      2. The schema's 'default' field.

    For list-valued PARAMS fields that correspond to scalar controls
    (e.g., 'particle_diameters_nm'), the first element of the list is used
    as the default scalar control value.
    """
    defaults: Dict[str, Any] = {}
    for schema_key, spec in PARAM_SCHEMA.items():
        base_key = spec["key"]

        if base_key in BASE_PARAMS:
            raw = BASE_PARAMS[base_key]
        else:
            raw = spec.get("default")

        # Unwrap single-element lists for scalar-like controls.
        if isinstance(raw, (list, tuple)) and raw:
            if base_key in (
                "particle_diameters_nm",
                "particle_materials",
                "particle_signal_multipliers",
            ):
                raw = raw[0]

        defaults[schema_key] = raw
    return defaults


def build_params_from_controls(control_values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a PARAMS-like dict from BASE_PARAMS and a set of control values.

    Parameters
    ----------
    control_values:
        Dict mapping schema keys (e.g., "particle_diameter_nm") to values
        provided by the user/UI.

    Behavior
    --------
    - Start from a deepcopy of BASE_PARAMS so the original config is untouched.
    - For each entry in PARAM_SCHEMA:
        * Determine a value to use:
            - If control_values contains the schema key, use that.
            - Else, if BASE_PARAMS already has the underlying PARAMS key,
              use BASE_PARAMS[key].
            - Else, fall back to the schema's "default".
        * Validate and normalize the value according to the spec["type"].
        * Write the value into the resulting params dict under spec["key"].
          For a small set of list-valued keys (e.g., "particle_diameters_nm"),
          the value is wrapped in a single-element list to match the
          single-particle viewer use case.

    Returns
    -------
    dict:
        A full PARAMS-like dictionary ready to be passed into
        generate_single_frame_views or the main simulation pipeline.
    """
    params = deepcopy(BASE_PARAMS)

    for schema_key, spec in PARAM_SCHEMA.items():
        base_key = spec["key"]

        # 1. Determine raw value: override -> base PARAMS -> schema default
        if schema_key in control_values:
            raw_value = control_values[schema_key]
        elif base_key in params:
            raw_value = params[base_key]
        else:
            raw_value = spec.get("default")

        # If base PARAMS entry is list-like for a scalar control, unwrap it.
        # This allows using the same schema key for both scalar and list
        # representations of single-particle fields.
        if isinstance(raw_value, (list, tuple)) and raw_value:
            # Only unwrap if the PARAMS field is expected to be list-valued
            # and we are controlling a single-element case via the schema.
            if base_key in (
                "particle_diameters_nm",
                "particle_materials",
                "particle_signal_multipliers",
            ):
                raw_value = raw_value[0]

        # 2. Validate & normalize according to spec
        value = _validate_and_normalize_value(spec, raw_value)

        # 3. Apply to params dict at the right key.
        param_key = base_key

        # Handle list-valued parameters in the single-particle viewer context:
        # we store the (validated) scalar value as a single-element list.
        if param_key in (
            "particle_diameters_nm",
            "particle_materials",
            "particle_signal_multipliers",
        ):
            params[param_key] = [value]
        else:
            params[param_key] = value

    return params