from .backend import get_backend
from numpy.typing import ArrayLike


def calculate_dipole_moments_linear(polarizability: ArrayLike,
                                    electric_field : ArrayLike) -> ArrayLike:
    """
    Calculate the dipole moments of particles in an electric field using a linear polarizability model.
    
    Parameters
    ----------
    polarizability :
        The polarizability of the particles. This is in general an (N, d, d) array, where N is the number of particles and d is the dimensionality of the system. It can also be a scalar (complex, float, or int) which will be applied to all particles.
    electric_field :
        The electric field at the location of the particles. This should be an array of shape (N, d), where N is the number of particles and d is the dimensionality of the system.
    
    Returns
    -------
    ArrayLike
        An array of shape (N, d) representing the dipole moments of the particles.
    """
    
    xp = get_backend(electric_field)
    dipole_moments = xp.einsum('ikl,il->ik', polarizability, electric_field)


    return dipole_moments
