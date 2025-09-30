import numpy as np


def calculate_sphere_polarizability(radius: float, material: str) -> float:
    """
    Calculate the polarizability of a spherical particle using the Clausius-Mossotti relation.
    
    Parameters
    ----------
    radius : float
        The radius of the spherical particle.
    material : str
        The material of the particle. Currently supports 'gold' and 'silver'.
    
    Returns
    -------
    float
        The polarizability of the spherical particle.
    """
    
    # Placeholder dielectric constants for materials at a specific wavelength (in nm)
    dielectric_constants = {
        'gold': -11.0 + 1.2j,  # Example value at ~520 nm
        'silver': -15.0 + 0.5j  # Example value at ~400 nm
    }
    
    if material not in dielectric_constants:
        raise ValueError(f"Material '{material}' not supported. Choose from {list(dielectric_constants.keys())}.")
    
    epsilon_particle = dielectric_constants[material]
    epsilon_medium = 1.0  # Assuming vacuum or air
    
    polarizability = 4 * np.pi * (radius**3) * (epsilon_particle - epsilon_medium) / (epsilon_particle + 2 * epsilon_medium)
    
    return polarizability


