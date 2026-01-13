import numpy as np
from config import BOLTZMANN_CONSTANT

def stokes_einstein_diffusion_coefficient(diameter_nm, temp_K, viscosity_Pa_s):
    """
    Calculates the diffusion coefficient for a spherical particle in a fluid
    using the Stokes-Einstein equation.

    Args:
        diameter_nm (float): The diameter of the particle in nanometers.
        temp_K (float): The absolute temperature of the fluid in Kelvin.
        viscosity_Pa_s (float): The dynamic viscosity of the fluid in Pascal-seconds.

    Returns:
        float: The diffusion coefficient in square meters per second (m^2/s).
    """
    radius_m = diameter_nm * 1e-9 / 2
    return (BOLTZMANN_CONSTANT * temp_K) / (6 * np.pi * viscosity_Pa_s * radius_m)

def simulate_trajectories(params):
    """
    Simulates 3D Brownian motion trajectories for a set of particles.

    Args:
        params (dict): A dictionary of simulation parameters.

    Returns:
        numpy.ndarray: A 3D array of shape (num_particles, num_frames, 3)
                       containing the [x, y, z] coordinates of each particle
                       for each frame, in nanometers.
    """
    num_frames = int(params["fps"] * params["duration_seconds"])
    dt = 1 / params["fps"]
    num_particles = params["num_particles"]
    
    # Initialize particle positions randomly if not explicitly provided.
    if "particle_initial_positions_nm" in params:
        initial_positions = np.array(params["particle_initial_positions_nm"], dtype=float)
    else:
        img_size_nm = params["image_size_pixels"] * params["pixel_size_nm"]
        initial_positions = np.random.rand(num_particles, 3) * [img_size_nm, img_size_nm, params["z_stack_range_nm"]]
        initial_positions[:, 2] -= params["z_stack_range_nm"] / 2

    trajectories = np.zeros((num_particles, num_frames, 3))
    trajectories[:, 0, :] = initial_positions

    # Calculate trajectory for each particle independently.
    for i in range(num_particles):
        D_m2_s = stokes_einstein_diffusion_coefficient(
            params["particle_diameters_nm"][i],
            params["temperature_K"],
            params["viscosity_Pa_s"]
        )
        sigma_m = np.sqrt(2 * D_m2_s * dt)
        sigma_nm = sigma_m * 1e9 # Convert standard deviation to nanometers
        
        # Generate random steps from a normal distribution.
        steps = np.random.normal(scale=sigma_nm, size=(num_frames - 1, 3))
        trajectories[i, 1:, :] = initial_positions[i] + np.cumsum(steps, axis=0)

    print("Generated Brownian motion trajectories.")
    return trajectories