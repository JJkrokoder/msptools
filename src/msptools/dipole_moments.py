import numpy as np
from typing import List

def calculate_dipole_moments_linear(polarizabilities: np.ndarray,
                                    electric_field : np.ndarray) -> np.ndarray:
    """
    Calculate the dipole moments using a linear relationship with the electric field.
    
    Parameters
    ----------
    polarizabilities :
        Polarizabilities of the particles, can be a single value or an array for each particle.
    electric_field :
        The electric field at the positions of the particles.
    
    Returns
    -------
    np.ndarray
        The calculated dipole moments for each particle.
    """
    
    dipole_moments = np.copy(electric_field)
    for i in range(electric_field.shape[0]):
        if isinstance(polarizabilities[i], (complex, float, int)):
            dipole_moments[i] = polarizabilities[i] * electric_field[i]
        else:
            dipole_moments[i] = np.dot(polarizabilities[i], electric_field[i])
    return dipole_moments

def polarizability_to_matrix(polarizability, num_particles : int, dimensions : int) -> np.ndarray:
    """
    Convert polarizability to a matrix form suitable for calculations.
    
    Parameters
    ----------
    polarizability : complex, float, int, list, or np.ndarray
        The polarizability value(s).
    num_particles :
        The number of particles in the system.
    dimensions :
        The number of dimensions of the system.
    
    Returns
    -------
    np.ndarray
        A matrix representation of the polarizability.
    """
    
    if isinstance(polarizability, (complex, float, int)):
        return np.eye(dimensions*num_particles) * polarizability

    elif isinstance(polarizability, (list, np.ndarray)):
       return np.diag([polarizability[i] for i in range(num_particles) for _ in range(dimensions)])