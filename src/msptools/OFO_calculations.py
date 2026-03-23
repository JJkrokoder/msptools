from typing import Iterable
from .backend import get_backend
from numpy.typing import ArrayLike

def calculate_forces_eppgrad(medium_permittivity: float, dipole_moments: ArrayLike, field_gradient: ArrayLike) -> ArrayLike:
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
    
    Notes
    -----
    The force is calculated using the formula:
        F = (ε/2) * Re{ p · ∇E* }
    where ε is the medium permittivity, p is the dipole moment, and ∇E* is the complex conjugate of the electric field gradient.
    """
    xp = get_backend(dipole_moments)

    forces = (medium_permittivity / 2) * xp.real(xp.einsum('im,inm->in', dipole_moments, xp.conj(field_gradient)))

    return forces

