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
    
    dimensions = dipole_moments.shape[1]
    num_particles = dipole_moments.shape[0]

    if field_gradient.shape != (num_particles, dimensions, dimensions):
        raise ValueError("Field gradient shape must match (N, d, d) where N is number of dipoles and d is dimensionality.")
    
    DipFieldProd = np.zeros_like(dipole_moments, dtype=complex)

    for i in range(num_particles):
        particle_dipole = dipole_moments[i]
        particle_gradient = field_gradient[i]
        DipFieldProd[i] = particle_gradient @ particle_dipole.conj()
    
    forces = (medium_permittivity / 2) * np.real(DipFieldProd)

    return forces


