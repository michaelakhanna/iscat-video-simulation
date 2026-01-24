from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from optics import IPSFZInterpolator


@dataclass(frozen=True)
class SubParticle:
    """
    Describes a spherical sub-particle within a (potentially non-spherical)
    composite particle shape.

    Geometry:
        - offset_nm: 3D vector giving the body-fixed position of this
          sub-particle relative to the composite's reference point, in nm.

    Optics:
        - diameter_nm: physical diameter of the sub-particle, in nm.
        - refractive_index: complex refractive index (n + i k).
        - ipsf_interpolator: spherical iPSF interpolator for this
          sub-particle type.
        - signal_multiplier: local amplitude scaling applied on top of the
          parent ParticleInstance.signal_multiplier.
    """
    offset_nm: np.ndarray
    diameter_nm: float
    refractive_index: complex
    ipsf_interpolator: IPSFZInterpolator
    signal_multiplier: float = 1.0


@dataclass(frozen=True)
class ParticleType:
    """
    Describes an optical "particle type" in the simulation.

    In the current (spherical) implementation a particle type corresponds to
    a single spherical particle characterized by:
        - diameter_nm: physical diameter in nanometers.
        - refractive_index: complex refractive index (n + i k).
        - ipsf_interpolator: spherical iPSF Z-interpolator defined on a
          type-specific z-grid.

    To prepare for non-spherical particles modeled as rigid clusters of
    spherical sub-particles (CDD Sections 2.3, 3.2, 3.3), this class also
    carries:

        - is_composite:
            False -> the particle is treated as a single sphere; the renderer
                     uses ipsf_interpolator directly (current behavior).
            True  -> the particle is a rigid composite; the renderer ignores
                     ipsf_interpolator and instead loops over sub_particles.

        - sub_particles:
            Tuple of SubParticle objects describing the rigid internal
            geometry in a body-fixed frame. For spherical particles this
            tuple is empty (is_composite=False). For future non-spherical
            particles it will list all sub-spheres.

    Current code only constructs spherical ParticleType instances with
    is_composite=False and sub_particles=().
    """
    diameter_nm: float
    refractive_index: complex
    ipsf_interpolator: IPSFZInterpolator

    is_composite: bool = False
    sub_particles: Tuple[SubParticle, ...] = ()

    @property
    def type_key(self) -> Tuple[float, float, float]:
        """
        Return a tuple that uniquely identifies this particle type within the
        current simulation:

            (diameter_nm, n.real, n.imag)

        This matches the key used in main.run_simulation when grouping
        particles by type.
        """
        n = self.refractive_index
        return (self.diameter_nm, float(n.real), float(n.imag))


@dataclass
class ParticleInstance:
    """
    Represents a single particle instance in the simulation.

    Each instance:
        - References exactly one ParticleType (optical behavior and iPSF).
        - Stores its full 3D trajectory in nanometers over all frames.
        - Stores its per-particle signal multiplier (scalar amplitude factor).
        - Optionally stores a per-frame orientation for non-spherical
          composite particles.

    Orientation representation:
        - orientation_matrices is either:
            * None (for spherical particles, current code),
            * or a numpy array of shape (num_frames, 3, 3) where each 3x3
              matrix is a rotation mapping body-fixed coordinates into the
              lab/world frame at that frame index.

        - In this structural refactor, orientation_matrices is always None,
          and the renderer ignores orientation for all particles. This keeps
          behavior identical to the previous spherical-only implementation
          while preparing the interface for rotational Brownian motion.
    """
    index: int
    particle_type: ParticleType
    trajectory_nm: np.ndarray
    signal_multiplier: float
    orientation_matrices: Optional[np.ndarray] = None


def build_particle_types_and_instances(
    params: dict,
    trajectories_nm: np.ndarray,
    particle_refractive_indices: np.ndarray,
    ipsf_interpolators_by_type: Dict[Tuple[float, float, float], IPSFZInterpolator],
    orientations: Optional[np.ndarray] = None,
) -> Tuple[Dict[Tuple[float, float, float], ParticleType], List[ParticleInstance]]:
    """
    Construct ParticleType and ParticleInstance objects for the current
    simulation run.

    This helper centralizes the mapping from per-particle scalar parameters
    (diameter, refractive index, signal multiplier) and trajectories into
    structured objects, using the existing per-type iPSF interpolators that
    were computed in main.run_simulation.

    It does not change any behavior of the simulation; the returned objects
    are an additional representation that downstream components (e.g.,
    rendering) now consume. The numerical results of the simulation remain
    governed by the same trajectories and iPSF stacks as before.

    Orientation handling:
        - If `orientations` is None, all ParticleInstance objects are created
          with orientation_matrices=None (spherical behavior).
        - If `orientations` is provided, it must have shape
          (num_particles, num_frames, 3, 3). The i-th ParticleInstance then
          receives orientations[i] as its orientation_matrices. This provides
          a consistent per-frame orientation timebase that can be used by
          future non-spherical composite renderers and motion-blur logic.

    Args:
        params (dict):
            Global parameter dictionary (PARAMS) for this simulation.
            Must contain:
                - "num_particles"
                - "particle_diameters_nm"
                - "particle_signal_multipliers"
        trajectories_nm (np.ndarray):
            Particle trajectories with shape (num_particles, num_frames, 3),
            as returned by trajectory.simulate_trajectories.
        particle_refractive_indices (np.ndarray):
            Complex refractive indices for each particle, shape (num_particles,).
        ipsf_interpolators_by_type (dict):
            Mapping from type_key = (diameter_nm, n_real, n_imag) to the
            IPSFZInterpolator computed for that type in main.run_simulation.
        orientations (Optional[np.ndarray]):
            Optional orientation array with shape
            (num_particles, num_frames, 3, 3). When provided, each particle's
            orientation_matrices field is populated from this array. When
            None, orientation_matrices is left as None for all particles.

    Returns:
        tuple:
            - A dictionary mapping type_key -> ParticleType.
            - A list of ParticleInstance objects of length num_particles.

    Raises:
        ValueError: If the lengths or shapes of the inputs are inconsistent
            with PARAMS["num_particles"].
    """
    num_particles = int(params["num_particles"])

    diameters_nm = params["particle_diameters_nm"]
    if len(diameters_nm) != num_particles:
        raise ValueError(
            "Length of PARAMS['particle_diameters_nm'] "
            f"({len(diameters_nm)}) must match PARAMS['num_particles'] ({num_particles})."
        )

    signal_multipliers = params.get("particle_signal_multipliers", None)
    if signal_multipliers is None or len(signal_multipliers) != num_particles:
        raise ValueError(
            "PARAMS['particle_signal_multipliers'] must be provided and have "
            f"length equal to PARAMS['num_particles'] ({num_particles})."
        )

    if trajectories_nm.shape[0] != num_particles or trajectories_nm.shape[2] != 3:
        raise ValueError(
            "trajectories_nm must have shape (num_particles, num_frames, 3). "
            f"Got {trajectories_nm.shape} for num_particles={num_particles}."
        )

    if particle_refractive_indices.shape[0] != num_particles:
        raise ValueError(
            "particle_refractive_indices must have length num_particles. "
            f"Got {particle_refractive_indices.shape[0]} for num_particles={num_particles}."
        )

    num_frames = trajectories_nm.shape[1]

    if orientations is not None:
        orientations = np.asarray(orientations, dtype=float)
        if orientations.shape != (num_particles, num_frames, 3, 3):
            raise ValueError(
                "orientations must have shape (num_particles, num_frames, 3, 3) "
                f"when provided. Got {orientations.shape}."
            )

    # Build ParticleType objects from the provided per-type interpolators.
    # At this stage all types are spherical: is_composite=False and
    # sub_particles=().
    particle_types: Dict[Tuple[float, float, float], ParticleType] = {}
    for type_key, interpolator in ipsf_interpolators_by_type.items():
        diam_nm, n_real, n_imag = type_key
        n_complex = complex(n_real, n_imag)
        particle_types[type_key] = ParticleType(
            diameter_nm=float(diam_nm),
            refractive_index=n_complex,
            ipsf_interpolator=interpolator,
            is_composite=False,
            sub_particles=(),
        )

    # Build ParticleInstance objects, one per particle, referencing the
    # appropriate ParticleType and its trajectory. orientation_matrices is
    # populated from the provided orientations array when available; otherwise
    # it is left as None (spherical case).
    instances: List[ParticleInstance] = []
    for i in range(num_particles):
        n_complex = particle_refractive_indices[i]
        type_key = (
            float(diameters_nm[i]),
            float(n_complex.real),
            float(n_complex.imag),
        )

        try:
            ptype = particle_types[type_key]
        except KeyError as exc:
            raise KeyError(
                "No ParticleType found for particle index {i} with type_key "
                f"{type_key}. This indicates a mismatch between the keys used "
                "to build ipsf_interpolators_by_type and the per-particle "
                "diameter/refractive_index arrays."
            ) from exc

        if orientations is not None:
            orientation_matrices = orientations[i].copy()
        else:
            orientation_matrices = None

        instance = ParticleInstance(
            index=i,
            particle_type=ptype,
            trajectory_nm=trajectories_nm[i],
            signal_multiplier=float(signal_multipliers[i]),
            orientation_matrices=orientation_matrices,
        )
        instances.append(instance)

    return particle_types, instances