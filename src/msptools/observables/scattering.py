from msptools.backend import get_backend
from msptools.GreenTensor_Electric import G_0_function, G_1_function
from numpy.typing import ArrayLike

def compute_scattering_field(
    positions: ArrayLike,
    particle_positions: ArrayLike,
    particle_dipoles: ArrayLike,
    k_magnitude: float,
) -> ArrayLike:
    """
    Compute the scattering field at specified positions due to particles with given dipoles.

    Parameters
    ----------
    positions : ArrayLike
        The positions at which to compute the scattering field.
    particle_positions : ArrayLike
        The positions of the particles.
    particle_dipoles : ArrayLike
        The dipole moments of the particles.
    k_magnitude : float
        The wave number in the medium.

    Returns
    -------
    ArrayLike
        The computed scattering field at the specified positions.
    """
    
    xp = get_backend(positions)
    
    # Initialize the scattering field array
    scattering_field = xp.zeros_like(positions, dtype=xp.complex128)
    
    # obtain relative vectors from each particle to each position
    relative_vectors = positions[:, None, :] - particle_positions[None, :, :]
    rel_distances = xp.linalg.norm(relative_vectors, axis=-1)
    G_0 = G_0_function(rel_distances, k_magnitude)
    G_1 = G_1_function(rel_distances, k_magnitude)
    rp = xp.einsum('ijk,jk->ij', relative_vectors, particle_dipoles)
    rpr_vec = rp [..., None] * relative_vectors
    scattering_field = k_magnitude**2 * (xp.einsum('ij,jk->ik', G_0, particle_dipoles) + xp.einsum('ij,ijk->ik', G_1, rpr_vec))
    
    return scattering_field
    
    
    

