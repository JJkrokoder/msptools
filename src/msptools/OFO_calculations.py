from typing import Iterable
import numpy as np

def calculate_forces_eppgrad(medium_permittivity: float, dipole_moments: np.ndarray, field_gradient: np.ndarray) -> np.ndarray:
    """
    Calculate the force on a set of dipoles in an electric field gradient.

    Parameters
    ----------
    medium_permittivity :
        The permittivity of the medium in which the dipoles are located.
    dipole_moments :
        An array representing the dipole moments of the particles. Shape should be (N, d), 
        where N is the number of dipoles and d is the dimensionality.
    field_gradient :
        An array representing the electric field gradient at the location of the dipoles. 
        Shape should be (N, d, d), where N is the number of dipoles and d is the dimensionality.

    Returns
    -------
    Forces :
        An array representing the force on each dipole.
    """

    forces = (medium_permittivity / 2) * np.real(np.einsum('ij,ikj->ik', dipole_moments, np.conj(field_gradient)))

    return forces

