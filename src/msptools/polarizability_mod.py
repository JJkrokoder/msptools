from .backend import get_backend
from typing import Callable
from scipy.special import spherical_jn as sph_jn
from scipy.special import spherical_yn as sph_yn
from scipy.constants import pi
from numpy.typing import ArrayLike
from .permittivity import permittivity_ridx
from scipy.constants import c, e, h

def select_computation_method(material: str, wavelength: float) -> Callable[[float, str], float]:
    """Select the polarizability computation method based on the material and excitation wavelength."""  

def hankel_plus(n: int, x: float, derivative: bool = False) -> complex:
    """Compute the spherical Hankel function of the first kind."""
    return sph_jn(n, x, derivative) * 1j - sph_yn(n, x, derivative)

def Clausius_Mossotti(radius: float, medium_permittivity: float, particle_permittivity: float) -> float:
    """
    Calculate the polarizability of a spherical particle using the Clausius-Mossotti relation.
    
    Parameters
    ----------
    radius : 
        The radius of the spherical particle.
    medium_permittivity :
        The permittivity of the surrounding medium.
    particle_permittivity :
        The permittivity of the particle material.
        
    
    Returns
    -------
    float
        The polarizability of the spherical particle.
    """

    polarizability = 4 * pi * (radius**3) * (particle_permittivity - medium_permittivity) / (particle_permittivity + 2 * medium_permittivity)

    return polarizability

def Core_Shell_Clausius_Mossotti(radius_core: float | ArrayLike,
                                 radius_shell: float | ArrayLike,
                                 medium_permittivity: float | ArrayLike,
                                 particle_permittivity_core: complex | ArrayLike,
                                 particle_permittivity_shell: complex | ArrayLike) -> complex | ArrayLike:
    """
    Calculate the polarizability of a core-shell particle using the Clausius-Mossotti relation.
    
    Parameters
    ----------
    radius_core : 
        The radius of the core of the particle.
    radius_shell :
        The radius of the shell of the particle (including the core).
    medium_permittivity :
        The permittivity of the surrounding medium.
    particle_permittivity_core :
        The permittivity of the core material.
    particle_permittivity_shell :
        The permittivity of the shell material.
        
    Returns
    -------
    complex | ArrayLike
        The polarizability of the core-shell particle.
        
    Notes
    -----
    Ensure proper broadcasting of the input parameters if they are arrays.
    """
    
    
    b3 = radius_core**3
    a3 = radius_shell**3
    e1 = particle_permittivity_core
    e2 = particle_permittivity_shell
    e_m = medium_permittivity  
    
    prefactor = 4 * pi * b3 * e_m
    
    numerator = (e2 - e_m) * (e1 + 2 * e2) * b3 + (e1 - e2) * (2 * e2 + e_m) * a3
    denominator = (e2 + 2 * e_m) * (e1 + 2 * e2) * b3 + 2 * (e1 - e2) * (e2 - e_m) * a3  

    return prefactor * numerator / denominator


def Mie_size_dipole_approximation(radius: float, medium_permittivity: float, particle_permittivity: float, wave_number: float) -> complex:
    """
    Calculate the polarizability of a spherical particle using Mie size dipole approximation.
    The bessel and hankel functions are expanded to second order in size parameter.

    Parameters
    ----------
    radius : 
        The radius of the spherical particle.
    medium_permittivity :
        The permittivity of the surrounding medium.
    particle_permittivity :
        The permittivity of the particle material.
    wave_number :
        The wave number of the incident light (in vacuum). 

    Returns
    -------
    complex
        The polarizability of the spherical particle using Mie size expansion. 

    Notes
    -----
    The Mie size expansion is used for particles that are small or comparable to the wavelength of light.
    The formula is derived from the Mie theory.
    The expansion is given by:
    alpha = alpha_0 [1 - (k^2 r^2 / 10) * (ε + ε_m)]/[1 - i(k_m^3 alpha_0 / 6π) - (k^2 r^2 / 10)(ε + 10ε_m)*(ε - ε_m)/(ε + 2ε_m)]
    where alpha_0 is the Clausius-Mossotti polarizability.
    - Wave number and radius should be in consistent units.
    """

    k_m = wave_number * (medium_permittivity)**0.5
    k = wave_number
    e_m = medium_permittivity
    e_p = particle_permittivity
    rho = k * radius

    alpha_0 = Clausius_Mossotti(radius, e_m, e_p)

    epsilon_ratio = (e_p - e_m) / (e_p + 2 * e_m)

    A_term = (rho**2 / 10) * (e_p + e_m)
    B_term = (rho**2 / 10) * (e_p + 10 * e_m) * epsilon_ratio
    C_term = 1j * k_m**3 * alpha_0 / (6 * pi)

    polarizability_mie = alpha_0 * (1 - A_term) / (1 - C_term - B_term)

    return polarizability_mie

def Mie_electric_dipole_polarizability(radius: float, medium_permittivity: float, particle_permittivity: float, wave_number: float) -> complex:
    """
    Calculate the electric dipole polarizability of a spherical particle using Mie theory.
    
    Parameters
    ----------
    radius : 
        The radius of the spherical particle.
    medium_permittivity :
        The permittivity of the surrounding medium.
    particle_permittivity :
        The permittivity of the particle material.
    wave_number :
        The wave number of the incident light (in vacuum).

    Returns
    -------
    complex
        The electric dipole polarizability of the spherical particle using Mie theory.
    Notes
    -----
    The electric dipole polarizability is derived from the first Mie coefficient (a1).
    The formula is given by:
    alpha_e = 6*pi/k_m^3*tE1
    where k_m is the wave number in the medium and tE1 is the first Mie coefficient for the electric dipole.
    - Wave number and radius should be in consistent units.
    """
    k_m = wave_number * medium_permittivity**0.5
    k_p = wave_number * particle_permittivity**0.5
    x_p = k_p * radius
    x_m = k_m * radius
    eps_m = medium_permittivity
    eps_p = particle_permittivity

    t11 = eps_p * sph_jn(1,x_p) * (sph_jn(1,x_m) + x_m * sph_jn(1,x_m,derivative=True))
    t12 = eps_m * sph_jn(1,x_m) * (sph_jn(1,x_p) + x_p * sph_jn(1,x_p,derivative=True))
    t21 = eps_m * hankel_plus(1,x_m) * (sph_jn(1,x_p) + x_p * sph_jn(1,x_p,derivative=True))
    t22 = eps_p * sph_jn(1,x_p) * (hankel_plus(1,x_m) + x_m * hankel_plus(1,x_m,derivative=True))

    tE1 = (t11 - t12) / (t21 - t22)

    alpha_e = 6 * pi / (k_m**3) * tE1
    return alpha_e

def Aden_Kerker_core_shell_polarizability(radius_core: float | ArrayLike, 
                      radius_shell: float | ArrayLike, 
                      medium_permittivity: complex | ArrayLike,
                      particle_permittivity_core: complex | ArrayLike,
                      particle_permittivity_shell: complex | ArrayLike,
                      wave_number: complex | ArrayLike) -> complex | ArrayLike:
    """
    Calculate the electric dipole polarizability of a core-shell particle using the Aden-Kerker formulation of Mie theory.
    
    Parameters    
    ----------
    radius_core :
        The radius of the core particle.
    radius_shell :
        The radius of the shell particle.
    medium_permittivity :
        The permittivity of the surrounding medium.
    particle_permittivity_core :
        The permittivity of the core material.
    particle_permittivity_shell :
        The permittivity of the shell material.
    wave_number :
        The wave number of the incident light (in vacuum).

    Returns
    -------
    complex | ArrayLike
        The electric dipole polarizability of the core-shell particle using the Aden-Kerker formulation of Mie theory.
    """
    
    k_m = wave_number * medium_permittivity ** 0.5
    k_2 = wave_number * particle_permittivity_shell ** 0.5
    k_1 = wave_number * particle_permittivity_core ** 0.5
    
    eps1 = particle_permittivity_core
    eps2 = particle_permittivity_shell
    epsm = medium_permittivity

    xm = k_m * radius_shell
    x2 = k_2 * radius_shell
    x1 = k_2 * radius_core
    xc = k_1 * radius_core

    # --- Core-shell coupling term A1 ---
    num_A = (
        eps2 * sph_jn(1, xc) * (sph_jn(1, x1) + x1 * sph_jn(1, x1, True))
        - eps1 * sph_jn(1, x1) * (sph_jn(1, xc) + xc * sph_jn(1, xc, True))
    )

    den_A = (
        eps2 * sph_jn(1, xc) * (sph_yn(1, x1) + x1 * sph_yn(1, x1, True))
        - eps1 * sph_yn(1, x1) * (sph_jn(1, xc) + xc * sph_jn(1, xc, True))
    )

    A1 = num_A / den_A

    # --- Build outer response ---
    j2 = sph_jn(1, x2)
    j2_p = sph_jn(1, x2) + x2 * sph_jn(1, x2, True)

    y2 = sph_yn(1, x2)
    y2_p = sph_yn(1, x2) + x2 * sph_yn(1, x2, True)

    jm = sph_jn(1, xm)
    jm_p = sph_jn(1, xm) + xm * sph_jn(1, xm, True)

    hm = hankel_plus(1, xm)
    hm_p = hankel_plus(1, xm) + xm * hankel_plus(1, xm, True)

    core_shell_term = j2_p - A1 * y2_p

    num = eps2 * j2 * jm_p - epsm * jm * core_shell_term
    den = eps2 * j2 * hm_p - epsm * hm * core_shell_term

    tE1 = num / den
    return 6 * pi / (k_m**3) * tE1

def polarizability_to_matrix(polarizability: ArrayLike | float | int | complex, num_particles: int, dimensions: int, xp) -> ArrayLike:
    """
    Convert the polarizability from various input formats to a matrix form suitable for MSP calculations.

    Parameters
    ----------
    polarizability :
        The polarizability of the particles. Can be a scalar, a 1D array of length N, or a 3D array of shape (N, d, d).
    num_particles :
        The number of particles in the system.
    dimensions :
        The dimensionality of the system (e.g., 3 for 3D).
        
    Returns
    -------
    ArrayLike
        The polarizability in matrix form, with shape (N, d, d).
    
    """
        
    if xp.isscalar(polarizability):
        pol_identity = polarizability * xp.eye(dimensions)[None, :, :]
        polarizability = xp.repeat(pol_identity, num_particles, axis=0)
    elif polarizability.ndim == 1:
        polarizability = polarizability[:, None, None] * xp.eye(dimensions)[None, :, :]
    elif polarizability.ndim == 3 and polarizability.shape[1] == dimensions and polarizability.shape[2] == dimensions:
        pass
    else:
        raise ValueError("Invalid polarizability shape. Expected scalar, 1D array of length N, or 3D array of shape (N, d, d). Got {}".format(polarizability.shape))

    return polarizability

def compute_sphere_polarizability_DA(radius_nm: float | ArrayLike,
                                     medium_permittivity: float,
                                     particle_material: str,
                                     wavelength_nm: float | ArrayLike,
                                     method: str = 'Mie') -> complex|ArrayLike:
    """
    Compute the polarizability of a spherical particle using the Mie electric dipole formula.
    
    Parameters
    ----------
    radius_nm :
        The radius of the spherical particle.
    medium_permittivity :
        The permittivity of the surrounding medium.
    particle_material :
        The material of the particle.
    wavelength_nm :
        The wavelength of the incident light in nanometers.
    
    Returns
    -------
    complex|ArrayLike
        The polarizability of the spherical particle using the Mie electric dipole formula.
    """
    wave_number = 2 * pi / wavelength_nm
    frequency_eV =  h * c / (wavelength_nm * 1e-9) / e
    particle_permittivity = permittivity_ridx(frequency_eV, particle_material)
    if method == 'Mie':
        polarizability = Mie_electric_dipole_polarizability(radius_nm, medium_permittivity, particle_permittivity, wave_number)
    elif method == 'Mie_SA':
        polarizability = Mie_size_dipole_approximation(radius_nm, medium_permittivity, particle_permittivity, wave_number)
    return polarizability

def compute_core_shell_polarizability_DA(radius_core_nm: float | ArrayLike,
                                         radius_shell_nm: float | ArrayLike,
                                         medium_permittivity: float,
                                         material_core: str,
                                         material_shell: str,
                                         wavelength_nm: float | ArrayLike,
                                         method: str = 'Aden-Kerker') -> complex|ArrayLike:
    """
    Compute the polarizability of a core-shell particle using the Mie electric dipole formula.
    
    Parameters
    ----------
    radius_core_nm :
        The radius of the core of the particle in nanometers.
    radius_shell_nm :
        The radius of the shell of the particle in nanometers (including the core).
    medium_permittivity :
        The permittivity of the surrounding medium.
    material_core :
        The material of the core of the particle.
    material_shell :
        The material of the shell of the particle.
    wavelength_nm :
        The wavelength of the incident light in nanometers.
    method :
        The method to compute the polarizability. Options are 'Aden-Kerker' for the full Mie solution for core-shell particles, or 'Clausius-Mossotti' for the quasistatic approximation. 
        
    Returns
    -------
    complex|ArrayLike
        The polarizability of the core-shell particle using the Mie electric dipole formula.
    """
    
    wave_number = 2 * pi / wavelength_nm
    frequency_eV =  h * c / (wavelength_nm * 1e-9) / e
    particle_permittivity_core = permittivity_ridx(frequency_eV, material_core)
    particle_permittivity_shell = permittivity_ridx(frequency_eV, material_shell)
    if method == 'Aden-Kerker':
        polarizability = Aden_Kerker_core_shell_polarizability(radius_core_nm, radius_shell_nm, medium_permittivity, particle_permittivity_core, particle_permittivity_shell, wave_number)
    elif method == 'Clausius-Mossotti':
        polarizability = Core_Shell_Clausius_Mossotti(radius_core_nm, radius_shell_nm, medium_permittivity, particle_permittivity_core, particle_permittivity_shell)
    return polarizability