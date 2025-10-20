import numpy as np
from typing import Callable

def select_computation_method(material: str, wavelength: float) -> Callable[[float, str], float]:
    """Select the polarizability computation method based on the material and excitation wavelength."""
    

def dielectric_Drude(frequency: float, plasma_frequency: float, collision_frequency: float) -> complex:
    """
    Calculate the dielectric constant using the Drude model.
    
    Parameters
    ----------
    frequency : float
        The frequency of the incident light.
    plasma_frequency : float
        The plasma frequency of the material.
    collision_frequency : float
        The collision frequency of the electrons in the material.
    
    Returns
    -------
    complex
        The dielectric constant of the material.
    """
    epsilon_inf = 1.0  # High-frequency dielectric constant
    epsilon = epsilon_inf - (plasma_frequency**2) / (frequency**2 + 1j * frequency * collision_frequency)
    return epsilon

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


