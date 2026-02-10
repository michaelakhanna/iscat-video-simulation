from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

ParamType = Literal["float", "int", "bool", "enum"]


class ParamSpec(TypedDict, total=False):
    """
    Container for user-facing parameter metadata.

    Keys:
      - key:         the key in the PARAMS dict (string, no nesting for now)
      - type:        "float" | "int" | "bool" | "enum"
      - default:     default value (used when PARAMS/base does not contain key)
      - min:         numeric lower bound (for float/int)
      - max:         numeric upper bound (for float/int)
      - choices:     list of allowed values (for enum)
      - ui_label:    human-readable name
      - group:       logical UI group ("Particle", "Optics", "Imaging", "Noise", ...)
      - description: short human-readable description for prompts/tooltips
    """
    key: str
    type: ParamType
    default: Any
    min: float
    max: float
    choices: List[Any]
    ui_label: str
    group: str
    description: str


PARAM_SCHEMA: Dict[str, ParamSpec] = {
    # Particle-related controls
    "particle_diameter_nm": ParamSpec(
        key="particle_diameters_nm",
        type="float",
        default=100.0,
        min=5.0,
        max=500.0,
        ui_label="Particle diameter (nm)",
        group="Particle",
        description="Optical diameter of the particle in nanometers.",
    ),
    "particle_material": ParamSpec(
        key="particle_materials",
        type="enum",
        default="Gold",
        choices=["Gold", "Silver", "Polystyrene"],
        ui_label="Particle material",
        group="Particle",
        description="Material label used for refractive index lookup.",
    ),
    # Optics
    "wavelength_nm": ParamSpec(
        key="wavelength_nm",
        type="float",
        default=635.0,
        min=400.0,
        max=800.0,
        ui_label="Wavelength (nm)",
        group="Optics",
        description="Illumination wavelength in vacuum (nm).",
    ),
    "numerical_aperture": ParamSpec(
        key="numerical_aperture",
        type="float",
        default=1.4,
        min=0.8,
        max=1.49,
        ui_label="Numerical aperture",
        group="Optics",
        description="Objective numerical aperture.",
    ),
    # Imaging / background
    "background_intensity": ParamSpec(
        key="background_intensity",
        type="float",
        default=100.0,
        min=0.0,
        max=500.0,
        ui_label="Background intensity",
        group="Imaging",
        description="Average background intensity level (camera counts).",
    ),
    # Noise toggles
    "shot_noise_enabled": ParamSpec(
        key="shot_noise_enabled",
        type="bool",
        default=True,
        ui_label="Shot noise",
        group="Noise",
        description="Include photon shot (Poisson) noise in the simulation.",
    ),
    "gaussian_noise_enabled": ParamSpec(
        key="gaussian_noise_enabled",
        type="bool",
        default=True,
        ui_label="Gaussian/read noise",
        group="Noise",
        description="Include Gaussian readout noise in the simulation.",
    ),
}