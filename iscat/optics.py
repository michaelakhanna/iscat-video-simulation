import numpy as np
from scipy.special import jn, yn
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

def mie_an_bn(m, x):
    """
    Calculates Mie scattering coefficients a_n and b_n.

    Args:
        m (complex): The complex refractive index ratio (particle/medium).
        x (float): The size parameter (2*pi*r/lambda).
    """
    nmax = int(np.ceil(x + 4 * x**(1/3) + 2))
    n = np.arange(1, nmax + 1)
    
    # Riccati-Bessel functions
    psi_n_x = np.sqrt(0.5 * np.pi * x) * jn(n + 0.5, x)
    psi_n_mx = np.sqrt(0.5 * np.pi * m * x) * jn(n + 0.5, m * x)
    chi_n_x = -np.sqrt(0.5 * np.pi * x) * yn(n + 0.5, x)
    
    # Derivatives of Riccati-Bessel functions
    psi_prime_n_x = np.sqrt(0.5 * np.pi / x) * jn(n + 0.5, x) / 2 + np.sqrt(0.5 * np.pi * x) * (jn(n-1 + 0.5, x) - (n+1)/x * jn(n + 0.5, x))
    psi_prime_n_mx = np.sqrt(0.5 * np.pi / (m*x)) * jn(n + 0.5, m*x) / 2 + np.sqrt(0.5 * np.pi * m*x) * (jn(n-1 + 0.5, m*x) - (n+1)/(m*x) * jn(n + 0.5, m*x))

    xi_n_x = psi_n_x + 1j * chi_n_x
    xi_prime_n_x = psi_prime_n_x - 1j * np.sqrt(0.5*np.pi/x)*yn(n+0.5,x)/2 - 1j*np.sqrt(0.5*np.pi*x)*(yn(n-1+0.5,x) - (n+1)/x*yn(n+0.5,x))
    
    # Mie coefficients calculation
    a_n = (m**2 * psi_n_mx * psi_prime_n_x - psi_n_x * psi_prime_n_mx) / \
          (m**2 * psi_n_mx * xi_prime_n_x - xi_n_x * psi_prime_n_mx)
    b_n = (psi_n_mx * psi_prime_n_x - psi_n_x * psi_prime_n_mx) / \
          (psi_n_mx * xi_prime_n_x - xi_n_x * psi_prime_n_mx)
          
    return a_n, b_n

def mie_S1_S2(m, x, mu):
    """
    Calculates Mie scattering amplitude functions S1 and S2.
    
    Args:
        m (complex): complex refractive index ratio
        x (float): size parameter
        mu (float): cos(theta) where theta is the scattering angle
    """
    nmax = int(np.ceil(x + 4 * x**(1/3) + 2))
    a_n, b_n = mie_an_bn(m, x)
    
    S1 = 0j
    S2 = 0j
    pi_n = np.zeros(nmax + 2)
    tau_n = np.zeros(nmax + 2)
    pi_n[1] = 1.0

    # Summation over n for S1 and S2
    for n in range(1, nmax + 1):
        if n > 1:
            pi_n[n] = ((2 * n - 1) / (n - 1)) * mu * pi_n[n - 1] - (n / (n - 1)) * pi_n[n - 2]
        
        tau_n[n] = n * mu * pi_n[n] - (n + 1) * pi_n[n - 1]
        
        factor = (2 * n + 1) / (n * (n + 1))
        S1 += factor * (a_n[n-1] * pi_n[n] + b_n[n-1] * tau_n[n])
        S2 += factor * (a_n[n-1] * tau_n[n] + b_n[n-1] * pi_n[n])
        
    return S1, S2

def compute_ipsf_stack(params, particle_diameter_nm, particle_refractive_index):
    """
    Computes a complex 3D vectorial interferometric Point Spread Function (iPSF)
    stack using the Debye-Born integral, calculated via FFT for efficiency.

    Args:
        params (dict): The main simulation parameter dictionary.
        particle_diameter_nm (float): The diameter of the particle for this iPSF.
        particle_refractive_index (complex): The complex refractive index of the particle.

    Returns:
        scipy.interpolate.RegularGridInterpolator: An interpolator object that can
                                                   return the complex 2D iPSF for any
                                                   given z-position within the stack's range.
    """
    # --- Setup k-space coordinates and optical parameters ---
    os_factor = params["psf_oversampling_factor"]
    pupil_samples = params["pupil_samples"]
    psf_size_nm = params["image_size_pixels"] * params["pixel_size_nm"]
    n_medium = params["refractive_index_medium"]
    wavelength_medium_nm = params["wavelength_nm"] / n_medium
    k_medium = 2 * np.pi / wavelength_medium_nm

    dk = (2 * np.pi / psf_size_nm) * os_factor
    kx = np.arange(-pupil_samples // 2, pupil_samples // 2) * dk
    ky = np.arange(-pupil_samples // 2, pupil_samples // 2) * dk
    Kx, Ky = np.meshgrid(kx, ky)
    K_sq = Kx**2 + Ky**2
    
    # --- Define the pupil aperture and coordinates ---
    sin_theta = np.sqrt(K_sq) / k_medium
    max_sin_theta = params["numerical_aperture"] / n_medium
    aperture_mask = (sin_theta <= max_sin_theta).astype(float)
    
    cos_theta = np.zeros_like(sin_theta)
    valid_mask = sin_theta <= 1
    cos_theta[valid_mask] = np.sqrt(1 - sin_theta[valid_mask]**2)
    
    # --- Calculate Mie scattering amplitudes across the pupil ---
    m = particle_refractive_index / n_medium
    radius_nm = particle_diameter_nm / 2
    x = 2 * np.pi * radius_nm / wavelength_medium_nm
    mu = np.zeros_like(cos_theta)
    mu[valid_mask] = cos_theta[valid_mask]
    S1_vec, S2_vec = np.vectorize(mie_S1_S2)(m, x, mu)
    
    # --- Define aberration and apodization functions ---
    z_values = np.arange(-params["z_stack_range_nm"]/2, params["z_stack_range_nm"]/2 + 1, params["z_stack_step_nm"])
    rho = sin_theta / max_sin_theta
    zernike_spherical = np.sqrt(5) * (6 * rho**4 - 6 * rho**2 + 1)
    spherical_phase = params["spherical_aberration_strength"] * zernike_spherical * 2 * np.pi
    apodization = np.exp(-params["apodization_factor"] * (rho**2))
    
    print(f"Computing iPSF stack for {particle_diameter_nm} nm particle...")
    ipsf_stack_complex = np.zeros((len(z_values), pupil_samples, pupil_samples), dtype=np.complex128)

    # --- Compute the iPSF for each Z-slice ---
    for i, z in enumerate(tqdm(z_values)):
        defocus_phase = k_medium * z * cos_theta
        aberration_phase = defocus_phase + spherical_phase
        
        # This simulates the complex, random aberrations of a real lens system.
        aberration_phase += (np.random.rand(pupil_samples, pupil_samples) - 0.5) * params["random_aberration_strength"] * 2 * np.pi

        pupil_function = (-1j * wavelength_medium_nm) * aperture_mask * apodization * S2_vec * np.exp(1j * aberration_phase)
        
        # The Amplitude Spread Function (ASF) is the Fourier transform of the pupil function.
        asf = fftshift(ifft2(ifftshift(pupil_function)))
        ipsf_stack_complex[i, :, :] = asf
        
    # Create an interpolator for fast lookups later.
    interpolator = RegularGridInterpolator((z_values,), ipsf_stack_complex, method='linear', bounds_error=False, fill_value=0)
    
    print("iPSF stack computation complete.")
    return interpolator