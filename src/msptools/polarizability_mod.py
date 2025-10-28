import numpy as np
from typing import Callable

def select_computation_method(material: str, wavelength: float) -> Callable[[float, str], float]:
    """Select the polarizability computation method based on the material and excitation wavelength."""    

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

def CM_with_Correction(radius: float, medium_permittivity: float, particle_permittivity: float, wave_number: float) -> complex:
    """
    Calculate the polarizability of a spherical particle using the Clausius-Mossotti relation with second order corrections.
    
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
        The polarizability of the spherical particle with radiative corrections.

    Notes
    -----
    The radiative correction is applied as per Draine and Goodman (1993):
    alpha_corrected = alpha_CM * [1 - k^2 r^2 / 10 * [(ε + 10ε_m)*(ε - ε_m)/(ε + 2ε_m) - ε - ε_m] + i(k_m^3 alpha_CM / 6π)]
    where alpha_CM is the Clausius-Mossotti polarizability.
    - Wave number and radius should be in consistent units.
    """

    rho = wave_number * radius
    k_m = wave_number * np.sqrt(medium_permittivity)
    eps = particle_permittivity
    eps_m = medium_permittivity

    alpha_CM = Clausius_Mossotti(radius, medium_permittivity, particle_permittivity)
    radiative_correction = 1 + rho**2 / 10 * ((eps + 10 * eps_m) * (eps - eps_m) / (eps + 2 * eps_m) - eps - eps_m) + 1j * k_m**3 * alpha_CM / (6 * np.pi)
    polarizability_corrected = alpha_CM * radiative_correction

    return polarizability_corrected


def Mie_size_expansion(radius: float, medium_permittivity: float, particle_permittivity: float, wave_number: float) -> complex:
    """
    Calculate the polarizability of a spherical particle using Mie size expansion.
    
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

    alpha_0 = Clausius_Mossotti(radius, medium_permittivity, particle_permittivity)

    size_parameter = wave_number * radius
    epsilon_ratio = (particle_permittivity - medium_permittivity) / (particle_permittivity + 2 * medium_permittivity)

    numerator = 1 - (size_parameter**2 / 10) * (particle_permittivity + medium_permittivity)
    denominator = (1 - 1j * (k_m**3) * alpha_0 / (6 * np.pi) - 
                   (size_parameter**2 / 10) * (particle_permittivity + 10 * medium_permittivity) * epsilon_ratio)

    polarizability_mie = alpha_0 * (numerator / denominator)

    return polarizability_mie


