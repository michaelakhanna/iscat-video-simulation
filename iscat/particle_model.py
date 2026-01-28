from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

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

    Spherical case (current default):
        - diameter_nm: physical diameter in nanometers.
        - refractive_index: complex refractive index (n + i k).
        - ipsf_interpolator: spherical iPSF Z-interpolator defined on a
          type-specific z-grid.

    Composite case (future non-spherical shapes):
        - is_composite:
            False -> the particle is treated as a single sphere; the renderer
                     uses ipsf_interpolator directly (current behavior).
            True  -> the particle is a rigid composite; the renderer ignores
                     ipsf_interpolator and instead loops over sub_particles.

        - sub_particles:
            Tuple of SubParticle objects describing the rigid internal
            geometry in a body-fixed frame. For spherical particles this
            tuple is empty (is_composite=False). For non-spherical particles
            it lists all spherical sub-components that will be positioned
            by orientation and translation.
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
            * None (for spherical particles, current default),
            * or a numpy array of shape (num_frames, 3, 3) where each 3x3
              matrix is a rotation mapping body-fixed coordinates into the
              lab/world frame at that frame index.

        - In the current configuration, rotations are only used when:
            * params["rotational_diffusion_enabled"] is True, and
            * the particle is composite (ptype.is_composite=True).
          Spherical particles ignore orientation; their PSF is radially
          symmetric.
    """
    index: int
    particle_type: ParticleType
    trajectory_nm: np.ndarray
    signal_multiplier: float
    orientation_matrices: Optional[np.ndarray] = None


def _resolve_shape_models(params: dict, num_particles: int) -> List[str]:
    """
    Resolve the per-particle shape model strings, defaulting to 'spherical'
    when no shape information is provided.

    This keeps the behavior of existing configurations unchanged: if
    PARAMS does not define 'particle_shape_models', all particles are
    treated as simple spheres.
    """
    raw = params.get("particle_shape_models", None)
    if raw is None:
        return ["spherical"] * num_particles

    if not isinstance(raw, (list, tuple)):
        raise TypeError(
            "PARAMS['particle_shape_models'] must be a list or tuple of "
            f"length num_particles when provided (got type {type(raw)})."
        )

    if len(raw) != num_particles:
        raise ValueError(
            "Length of PARAMS['particle_shape_models'] "
            f"({len(raw)}) must match PARAMS['num_particles'] ({num_particles})."
        )

    models: List[str] = []
    for idx, entry in enumerate(raw):
        if entry is None:
            models.append("spherical")
        else:
            models.append(str(entry).strip())
    return models


def _build_composite_subparticles_for_instance(
    params: dict,
    particle_index: int,
    base_diameter_nm: float,
    base_refractive_index: complex,
    ipsf_interpolators_by_type: Dict[Tuple[float, float, float], IPSFZInterpolator],
) -> Tuple[SubParticle, ...]:
    """
    Build the list of SubParticle objects for a single composite particle
    instance, using the composite_shape_library definition and the already
    constructed iPSF interpolators.

    Structural behavior:
        - All spherical optical types (diameter, refractive_index) used by
          sub-particles *must* have been included in the set of optical types
          collected in main.run_simulation and therefore must have entries in
          ipsf_interpolators_by_type.
        - main.run_simulation is responsible for scanning
          PARAMS['composite_shape_library'] and adding any sub-particle
          optical types to its type_keys_required set before computing PSF
          stacks. This ensures that when we arrive here, the necessary
          interpolators are already available and no additional PSF
          computation is needed.

    This separation makes the PSF precomputation logic and the composite
    geometry logic consistent and prevents hidden coupling between composite
    definitions and base particle arrays.

    Args:
        params (dict): Global parameter dictionary.
        particle_index (int): Index of the logical particle for which sub-
            particles are being constructed.
        base_diameter_nm (float): Optical diameter of the parent particle.
        base_refractive_index (complex): Optical refractive index of the
            parent particle.
        ipsf_interpolators_by_type (dict): Mapping from type_key
            (diameter_nm, n_real, n_imag) to IPSFZInterpolator, as computed
            in main.run_simulation. Must contain entries for all optical
            types referenced by the composite shapes.

    Returns:
        Tuple[SubParticle, ...]: Immutable sequence of SubParticle objects.

    Raises:
        ValueError/TypeError: For malformed composite_shape_library entries.
        KeyError: If a sub-particle optical type was not included in the
            precomputed interpolator set (indicating a bug or inconsistent
            configuration), with a clear message.
    """
    library: Dict[str, Any] = params.get("composite_shape_library", {})
    if not isinstance(library, dict):
        raise TypeError(
            "PARAMS['composite_shape_library'] must be a dictionary when provided."
        )

    shape_models = _resolve_shape_models(params, int(params["num_particles"]))
    shape_model = shape_models[particle_index]
    if shape_model.lower() == "spherical":
        return ()

    if shape_model not in library:
        raise ValueError(
            f"Particle {particle_index}: shape model '{shape_model}' is not "
            "defined in PARAMS['composite_shape_library']."
        )

    shape_def = library[shape_model]
    sub_defs = shape_def.get("sub_particles", None)
    if not isinstance(sub_defs, list) or len(sub_defs) == 0:
        raise ValueError(
            f"Composite shape '{shape_model}' must define a non-empty "
            "'sub_particles' list in composite_shape_library."
        )

    sub_particles: List[SubParticle] = []
    for sub_idx, sub_def in enumerate(sub_defs):
        if not isinstance(sub_def, dict):
            raise TypeError(
                f"Composite shape '{shape_model}' sub_particles[{sub_idx}] "
                "must be a dictionary."
            )

        if "offset_nm" not in sub_def:
            raise ValueError(
                f"Composite shape '{shape_model}' sub_particles[{sub_idx}] "
                "must define 'offset_nm'."
            )

        offset_nm_arr = np.asarray(sub_def["offset_nm"], dtype=float)
        if offset_nm_arr.shape != (3,):
            raise ValueError(
                f"Composite shape '{shape_model}' sub_particles[{sub_idx}]."
                "offset_nm must be a length-3 array-like [dx, dy, dz] in nm."
            )

        diam_sub = sub_def.get("diameter_nm", None)
        if diam_sub is None:
            diam_sub_val = float(base_diameter_nm)
        else:
            diam_sub_val = float(diam_sub)

        n_sub = sub_def.get("refractive_index", None)
        if n_sub is None:
            n_sub_complex = complex(base_refractive_index)
        else:
            n_sub_complex = complex(n_sub)

        type_key_sub = (
            float(diam_sub_val),
            float(n_sub_complex.real),
            float(n_sub_complex.imag),
        )

        if type_key_sub not in ipsf_interpolators_by_type:
            # This indicates that the PSF precomputation step in main did not
            # include this optical type in its type_keys_required set. That is
            # either a configuration bug (e.g., composite shapes changed
            # without re-running main) or a code bug. We raise a clear error
            # message rather than silently falling back to an arbitrary type.
            raise KeyError(
                "Composite sub-particle for particle index "
                f"{particle_index} (shape '{shape_model}', sub index {sub_idx}) "
                f"requires optical type_key={type_key_sub}, but no iPSF "
                "interpolator was precomputed for this type. The set of required "
                "optical types must be collected from composite_shape_library "
                "before PSF computation in main.run_simulation."
            )

        ipsf_interp = ipsf_interpolators_by_type[type_key_sub]

        signal_multiplier_local = float(sub_def.get("signal_multiplier", 1.0))
        if signal_multiplier_local < 0.0:
            raise ValueError(
                f"Composite shape '{shape_model}' sub_particles[{sub_idx}]."
                "signal_multiplier must be non-negative."
            )

        sub_particles.append(
            SubParticle(
                offset_nm=offset_nm_arr,
                diameter_nm=diam_sub_val,
                refractive_index=n_sub_complex,
                ipsf_interpolator=ipsf_interp,
                signal_multiplier=signal_multiplier_local,
            )
        )

    return tuple(sub_particles)


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

    Behavior preservation:
        - When PARAMS does not define 'particle_shape_models' or defines all
          entries as "spherical", every particle is represented as a simple
          sphere. ParticleType.is_composite=False and sub_particles=().
          This matches the previous behavior exactly.

        - Composite shapes can be enabled by:
            * Defining PARAMS["composite_shape_library"][shape_name] with
              sub-particle offsets and optical overrides, and
            * Setting PARAMS["particle_shape_models"][i] = shape_name for
              the desired particles.

          The PSF precomputation step in main.run_simulation is responsible
          for including any additional optical types used by sub-particles in
          its iPSF stack computation. Here we simply enforce the existence of
          those types and construct SubParticle objects.

    Orientation handling:
        - If `orientations` is None, all ParticleInstance objects are created
          with orientation_matrices=None (spherical / orientation-ignored).
        - If `orientations` is provided, it must have shape
          (num_particles, num_frames, 3, 3). The i-th ParticleInstance then
          receives orientations[i] as its orientation_matrices. Composite
          particles use these matrices to rotate sub-particle offsets during
          rendering; spherical particles ignore them.

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
            with PARAMS["num_particles"] or the trajectory/orientation shapes.
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

    # Resolve per-particle shape models once.
    shape_models = _resolve_shape_models(params, num_particles)

    # Build the base spherical ParticleType objects from the provided per-type
    # interpolators. These represent the optical types (diameter, n) that were
    # used to compute the PSF stacks. Composite ParticleTypes will be created
    # per logical particle below and share these iPSF building blocks via
    # SubParticle definitions.
    spherical_types: Dict[Tuple[float, float, float], ParticleType] = {}
    for type_key, interpolator in ipsf_interpolators_by_type.items():
        diam_nm, n_real, n_imag = type_key
        n_complex = complex(n_real, n_imag)
        spherical_types[type_key] = ParticleType(
            diameter_nm=float(diam_nm),
            refractive_index=n_complex,
            ipsf_interpolator=interpolator,
            is_composite=False,
            sub_particles=(),
        )

    # We maintain a dictionary of all ParticleType objects (both spherical and
    # composite) so that if multiple particles share the same composite shape
    # and optical base type, they can share the same ParticleType instance.
    particle_types: Dict[Tuple[float, float, float, str], ParticleType] = {}

    def _get_or_create_particle_type_for_particle(
        p_index: int,
        base_type_key: Tuple[float, float, float],
    ) -> ParticleType:
        """
        For a given particle index and its base spherical type_key, either
        return the existing ParticleType (spherical or composite) or create
        a new one if this combination of base type and shape model has not
        been seen before.

        The composite key is (diameter, n.real, n.imag, shape_model).
        """
        shape_model = shape_models[p_index]
        # Normalize 'spherical' explicitly.
        if shape_model.lower() == "spherical":
            # For spherical particles we always reuse the spherical ParticleType.
            diam_nm, n_real, n_imag = base_type_key
            spherical_ptype = spherical_types[base_type_key]
            composite_key = (float(diam_nm), float(n_real), float(n_imag), "spherical")
            # Ensure mapping is consistent but do not duplicate objects.
            particle_types.setdefault(composite_key, spherical_ptype)
            return spherical_ptype

        # Composite case: use both optical type and shape model in the key.
        diam_nm, n_real, n_imag = base_type_key
        composite_key = (float(diam_nm), float(n_real), float(n_imag), shape_model)

        if composite_key in particle_types:
            return particle_types[composite_key]

        # Create a new composite ParticleType: populate sub_particles by
        # resolving composite_shape_library for this particle.
        base_ptype = spherical_types[base_type_key]
        sub_particles = _build_composite_subparticles_for_instance(
            params=params,
            particle_index=p_index,
            base_diameter_nm=base_ptype.diameter_nm,
            base_refractive_index=base_ptype.refractive_index,
            ipsf_interpolators_by_type=ipsf_interpolators_by_type,
        )

        composite_ptype = ParticleType(
            diameter_nm=base_ptype.diameter_nm,
            refractive_index=base_ptype.refractive_index,
            ipsf_interpolator=base_ptype.ipsf_interpolator,
            is_composite=True,
            sub_particles=sub_particles,
        )
        particle_types[composite_key] = composite_ptype
        return composite_ptype

    # Build ParticleInstance objects, one per particle, referencing the
    # appropriate ParticleType (spherical or composite) and its trajectory.
    instances: List[ParticleInstance] = []
    for i in range(num_particles):
        n_complex = particle_refractive_indices[i]
        base_type_key = (
            float(diameters_nm[i]),
            float(n_complex.real),
            float(n_complex.imag),
        )

        if base_type_key not in spherical_types:
            raise KeyError(
                "No base spherical ParticleType found for particle index "
                f"{i} with type_key {base_type_key}. This indicates a mismatch "
                "between the keys used to build ipsf_interpolators_by_type and "
                "the per-particle diameter/refractive_index arrays."
            )

        ptype = _get_or_create_particle_type_for_particle(i, base_type_key)

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

    # Expose only the underlying spherical types keyed by optical type for
    # external tooling that might depend on the old mapping. The rendering
    # pipeline only needs the list of instances.
    return spherical_types, instances