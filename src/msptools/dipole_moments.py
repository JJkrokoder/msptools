import numpy as np
from typing import List

def calculate_dipole_moments_linear(polarizability: np.ndarray | complex | List[complex],
                                    electric_field : np.ndarray) -> np.ndarray:
    
    dipole_moments = np.zeros_like(electric_field, dtype=complex)
 
    if isinstance(polarizability, (complex, float, int)):
        polarizability += 0j  # Ensure polarizability is treated as a complex number
        dipole_moments = polarizability * electric_field
    elif isinstance(polarizability, (list, np.ndarray)):
        number_of_polarizabilities = len(polarizability) if isinstance(polarizability, list) else polarizability.shape[0]
        if number_of_polarizabilities != electric_field.shape[0]:
            raise ValueError("Polarizability and electric field must have the same number of elements.")
        for i in range(number_of_polarizabilities):
            dipole_moments[i,:] = polarizability[i] * electric_field[i,:]
    else:
        raise TypeError("Polarizability must be a complex number, float, int, list, or numpy array.")


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