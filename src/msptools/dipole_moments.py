from .backend import get_backend
from numpy.typing import ArrayLike


def calculate_dipole_moments_linear(polarizability: ArrayLike,
                                    electric_field: ArrayLike) -> ArrayLike:
    """
    Calculate the dipole moments of particles in an electric field using a linear polarizability model.
    """
    xp = get_backend(electric_field)

    if xp.isscalar(polarizability) or polarizability.ndim < 2:
        # scalar or per-particle scalar case: no tensor contraction needed
        return polarizability[..., None] * electric_field if not xp.isscalar(polarizability) else polarizability * electric_field

    # dipole_moments[i,k] = sum_l polarizability[i,k,l] * electric_field[i,l]
    dipole_moments = (polarizability * electric_field[:, None, :]).sum(axis=-1)

    return dipole_moments
