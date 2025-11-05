import numpy as np
from typing import Callable
from scipy.special import spherical_jn as sph_jn
from scipy.special import spherical_yn as sph_yn

def select_computation_method(material: str, wavelength: float) -> Callable[[float, str], float]:
    """Select the polarizability computation method based on the material and excitation wavelength."""  

def hankel_first_kind(n: int, x: float, derivative: bool = False) -> complex:
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

    polarizability = 4 * np.pi * (radius**3) * (particle_permittivity - medium_permittivity) / (particle_permittivity + 2 * medium_permittivity)

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

    k_m = wave_number * np.sqrt(medium_permittivity)
    k = wave_number
    e_m = medium_permittivity
    e_p = particle_permittivity
    rho = k * radius

    alpha_0 = Clausius_Mossotti(radius, e_m, e_p)

    epsilon_ratio = (e_p - e_m) / (e_p + 2 * e_m)

    A_term = (rho**2 / 10) * (e_p + e_m)
    B_term = (rho**2 / 10) * (e_p + 10 * e_m) * epsilon_ratio
    C_term = 1j * k_m**3 * alpha_0 / (6 * np.pi)

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
    alpha_e = 2*pi/k_m^2*tEl=1
    where k_m is the wave number in the medium and tEl=1 is the first Mie coefficient for the electric dipole.
    - Wave number and radius should be in consistent units.
    """

    k_m = wave_number * np.sqrt(medium_permittivity)
    x = wave_number * radius * particle_permittivity ** 0.5
    x_m = k_m * radius
    eps_m = medium_permittivity
    eps_p = particle_permittivity

    t11 = eps_p * sph_jn(1,x) * (sph_jn(1,x_m) + x_m * sph_jn(1,x_m,derivative=True))
    t12 = eps_m * sph_jn(1,x_m) * (sph_jn(1,x) + x * sph_jn(1,x,derivative=True))
    t21 = eps_m * hankel_first_kind(1,x_m) * (sph_jn(1,x) + x * sph_jn(1,x,derivative=True))
    t22 = eps_p * sph_jn(1,x) * (hankel_first_kind(1,x_m) + x_m * hankel_first_kind(1,x_m,derivative=True))


    t1 = (t11 - t12) / (t21 - t22)

    alpha_e = 2 * np.pi / (k_m**3) * t1
    return alpha_e