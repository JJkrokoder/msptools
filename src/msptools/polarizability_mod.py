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

def CM_with_Radiative_Correction(radius: float, medium_permittivity: float, particle_permittivity: float, wave_number: float) -> complex:
    """
    Calculate the polarizability of a spherical particle using the Clausius-Mossotti relation with radiative corrections.
    
    Parameters
    ----------
    radius : 
        The radius of the spherical particle.
    medium_permittivity :
        The permittivity of the surrounding medium.
    particle_permittivity :
        The permittivity of the particle material.
    wave_number :
        The wave number of the incident light.
        
    
    Returns
    -------
    complex
        The polarizability of the spherical particle with radiative corrections.

    Notes
    -----
    The radiative correction is applied as per Draine and Goodman (1993):
    alpha_corrected = alpha_CM / [1 - i(k^3)alpha_CM/(6π)]
    - Wave number and radius should be in consistent units.
    """

    alpha_CM = Clausius_Mossotti(radius, medium_permittivity, particle_permittivity)
    radiative_correction = 1 - 1j * (wave_number**3) * alpha_CM / (6 * np.pi)
    polarizability_corrected = alpha_CM / radiative_correction

    return polarizability_corrected


