import numpy as np

from trajectory import stokes_einstein_diffusion_coefficient


class TrackabilityModel:
    """
    Implements the Human Trackability Confidence Model described in the
    Code Design Document (Section 3.4.1).

    This class maintains per-particle state across frames and computes, for
    each particle in each frame, a confidence in [0, 1] that a human could
    reliably track that particle at its simulated location.

    The confidence is modeled as a product of three terms:

        confidence = P_pos * P_signal * P_excl

    where:
        - P_pos encodes the positional probability based on Brownian diffusion.
        - P_signal encodes how strong the particle's contrast signal is relative
          to the expected detector noise.
        - P_excl encodes the exclusivity of the candidate location. In this
          implementation we assume no ambiguous competing candidates within the
          search area, so P_excl = 1.

    The model uses the true simulated coordinates as the candidate location.
    This is appropriate for synthetic data generation, where the goal is to
    gate mask generation according to human trackability rather than to
    re-estimate the particle position from scratch.
    """

    def __init__(self, params: dict, num_particles: int):
        """
        Initialize the trackability model.

        Args:
            params (dict): Global simulation parameter dictionary (PARAMS).
            num_particles (int): Number of particles being simulated.
        """
        self.params = params
        self.num_particles = int(num_particles)
        self.dt = 1.0 / float(params["fps"])

        # Precompute diffusion coefficients and characteristic radial
        # displacement scales for each particle.
        temp_K = float(params["temperature_K"])
        viscosity = float(params["viscosity_Pa_s"])
        diameters_nm = params["particle_diameters_nm"]

        if len(diameters_nm) != self.num_particles:
            raise ValueError(
                "Length of PARAMS['particle_diameters_nm'] must match "
                "PARAMS['num_particles']."
            )

        self.diffusion_coefficients_m2_s = np.zeros(self.num_particles, dtype=float)
        self.r_sigma_nm = np.zeros(self.num_particles, dtype=float)

        for i in range(self.num_particles):
            D_m2_s = stokes_einstein_diffusion_coefficient(
                diameters_nm[i], temp_K, viscosity
            )
            self.diffusion_coefficients_m2_s[i] = D_m2_s
            # For 2D Brownian motion, <r^2> = 4 * D * dt. Use the square root
            # as a characteristic radial scale in nanometers.
            self.r_sigma_nm[i] = np.sqrt(4.0 * D_m2_s * self.dt) * 1e9

        # Estimate the detector noise level in "contrast units" so that the
        # signal plausibility term can be expressed in terms of SNR.
        self._noise_contrast_std = self._estimate_contrast_noise_std(params)

        # Per-particle state: last known 3D position (in nm) and "lost" flags.
        self.last_positions_nm = [None] * self.num_particles
        self.lost = np.zeros(self.num_particles, dtype=bool)

    @staticmethod
    def _estimate_contrast_noise_std(params: dict) -> float:
        """
        Estimate the standard deviation of the detector noise in the units used
        for the contrast images passed to the trackability model.

        The contrast images used by the masks are proportional to
        (I - I_background) / I_background, so we convert the noise from camera
        counts into these normalized units.

        Args:
            params (dict): Global simulation parameter dictionary.

        Returns:
            float: Estimated noise standard deviation in contrast units.
        """
        background = float(params.get("background_intensity", 0.0))

        shot_enabled = bool(params.get("shot_noise_enabled", False))
        shot_scale = float(params.get("shot_noise_scaling_factor", 0.0)) if shot_enabled else 0.0

        gaussian_enabled = bool(params.get("gaussian_noise_enabled", False))
        read_noise_std_counts = (
            float(params.get("read_noise_std", 0.0)) if gaussian_enabled else 0.0
        )

        # Poisson (shot) noise standard deviation: sqrt(I), with I ~ background.
        if shot_enabled and background > 0.0:
            shot_noise_std_counts = np.sqrt(background) * shot_scale
        else:
            shot_noise_std_counts = 0.0

        # Combine independent noise sources in quadrature.
        noise_std_counts = np.sqrt(
            shot_noise_std_counts**2 + read_noise_std_counts**2
        )

        if background > 0.0:
            noise_contrast_std = noise_std_counts / background
        else:
            # Degenerate case: if background is zero, fall back to counts units.
            noise_contrast_std = noise_std_counts if noise_std_counts > 0.0 else 1.0

        return float(noise_contrast_std)

    def reset(self) -> None:
        """
        Reset internal state so that the model can be reused for a new video.
        """
        self.last_positions_nm = [None] * self.num_particles
        self.lost[:] = False

    def is_particle_lost(self, particle_index: int) -> bool:
        """
        Check whether a given particle has already been declared "lost".

        Args:
            particle_index (int): Zero-based particle index.

        Returns:
            bool: True if the particle is lost, False otherwise.
        """
        return bool(self.lost[particle_index])

    def are_all_particles_lost(self) -> bool:
        """
        Returns True if all particles have been declared lost.
        """
        return bool(np.all(self.lost))

    def update_and_compute_confidence(
        self,
        particle_index: int,
        frame_index: int,
        position_nm: np.ndarray,
        contrast_image: np.ndarray,
    ) -> float:
        """
        Update the trackability state for a single particle and compute the
        trackability confidence for the current frame.

        Args:
            particle_index (int): Zero-based particle index.
            frame_index (int): Zero-based frame index (currently unused but
                included for future extensions that use longer history).
            position_nm (np.ndarray): The simulated [x, y, z] coordinates of the
                particle in nanometers at this frame.
            contrast_image (np.ndarray): The per-particle contrast image at the
                final image resolution for this frame. This is typically the
                noise-free interferometric contrast generated from the iPSF.

        Returns:
            float: Trackability confidence in [0, 1]. If the particle has
                already been marked lost, this returns 0.
        """
        if self.lost[particle_index]:
            return 0.0

        # Ensure we are working with a 1D array [x, y, z].
        position_nm = np.asarray(position_nm, dtype=float)
        if position_nm.shape != (3,):
            raise ValueError(
                "position_nm must be a 1D array of shape (3,) representing [x, y, z] in nm."
            )

        # --- Positional probability term (P_pos) ---
        last_pos = self.last_positions_nm[particle_index]
        if last_pos is None:
            # No previous information: treat the first frame as fully plausible.
            P_pos = 1.0
        else:
            delta_xy_nm = position_nm[:2] - last_pos[:2]
            r_nm = float(np.linalg.norm(delta_xy_nm))
            sigma_r_nm = self.r_sigma_nm[particle_index]

            if sigma_r_nm <= 0.0:
                # Degenerate case: diffusion coefficient numerically zero.
                P_pos = 1.0 if r_nm == 0.0 else 0.0
            else:
                # Use a Gaussian radial model with a conservative scaling so that
                # typical Brownian steps yield P_pos close to 1. Very large jumps
                # (many sigma) will reduce P_pos towards 0.
                scaled_r = r_nm / (3.0 * sigma_r_nm)
                P_pos = float(np.exp(-0.5 * scaled_r**2))
                P_pos = max(0.0, min(1.0, P_pos))

        # --- Signal plausibility term (P_signal) ---
        # Use the peak absolute contrast within the per-particle contrast image
        # together with the estimated detector noise level to define an SNR.
        contrast_abs = np.abs(contrast_image)
        max_contrast = float(contrast_abs.max()) if contrast_abs.size else 0.0
        noise_std = self._noise_contrast_std

        if noise_std <= 0.0:
            # If the noise level is zero (or numerically negligible), any non-zero
            # contrast is perfectly plausible.
            P_signal = 1.0 if max_contrast > 0.0 else 0.0
        else:
            snr = max_contrast / noise_std
            # Map SNR to [0, 1] using a smooth, saturating function. With this
            # choice, SNR ~ 1 gives moderate confidence, and SNR > 3 is very close
            # to 1. This matches the intuition that particles clearly above the
            # noise floor are highly trackable.
            P_signal = float(1.0 - np.exp(-0.5 * snr**2))
            P_signal = max(0.0, min(1.0, P_signal))

        # --- Exclusivity term (P_excl) ---
        # In this implementation, we assume there is a single clear candidate
        # within the search region for each particle, so we set P_excl = 1.
        P_excl = 1.0

        confidence = P_pos * P_signal * P_excl

        # Update the last known position for this particle so that subsequent
        # frames can use it as context.
        self.last_positions_nm[particle_index] = position_nm.copy()

        return confidence