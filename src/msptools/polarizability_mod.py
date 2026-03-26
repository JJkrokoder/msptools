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

def compute_sphere_polarizability_DA(radius_nm: float, medium_permittivity: float, particle_material: str, wavelength_nm: float | ArrayLike) -> complex|ArrayLike:
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
    polarizability = Mie_electric_dipole_polarizability(radius_nm, medium_permittivity, particle_permittivity, wave_number)
    return polarizability