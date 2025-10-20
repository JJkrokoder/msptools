

def permittivity_Drude(frequency: float, plasma_frequency: float, collision_frequency: float, epsilon_inf: float) -> complex:
    """
    Calculate the permittivity using the Drude model.
    
    Parameters
    ----------
    frequency : float
        The frequency of the incident light.
    plasma_frequency : float
        The plasma frequency of the material.
    collision_frequency : float
        The collision frequency of the electrons in the material.
    epsilon_inf : float
        The high-frequency dielectric constant of the material.
    
    Returns
    -------
    complex
        The dielectric constant of the material.
    """
    epsilon = epsilon_inf - (plasma_frequency**2) / (frequency**2 + 1j * frequency * collision_frequency)
    return epsilon
