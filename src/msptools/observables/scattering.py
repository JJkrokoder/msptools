from msptools.backend import get_backend
from msptools.GreenTensor_Electric import G_0_function, G_1_function, pairwise_green_tensor
from numpy.typing import ArrayLike
import numpy as np
from scipy.constants import pi

def sample_fibonacci_sphere(samples: int, radius: float, xp: object) -> ArrayLike:
    """
    Generate points uniformly distributed on the surface of a sphere using the Fibonacci lattice method.

    Parameters
    ----------
    samples : int
        The number of points to generate on the sphere.
    radius : float
        The radius of the sphere.

    Returns
    -------
    ArrayLike
        An array of shape (samples, 3) containing the Cartesian coordinates of the points on the sphere.
    """
    
    
    points = []
    phi = xp.pi * (3. - xp.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius_at_y = xp.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = xp.cos(theta) * radius_at_y
        z = xp.sin(theta) * radius_at_y

        points.append((x * radius, y * radius, z * radius))

    return xp.array(points)

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
    
    n_dipoles = particle_dipoles.shape[0]
    
    scattering_field = xp.zeros_like(positions, dtype=xp.complex128)
    
    if n_dipoles > 1e3:
        for i in range(positions.shape[0]):
            print(f"Progress: {(i+1)/(positions.shape[0])*100:.2f}%", end="\r")
            relative_vectors = positions[i] - particle_positions
            rel_distances = xp.linalg.norm(relative_vectors, axis=-1)
            G_0 = G_0_function(rel_distances, k_magnitude)
            G_1 = G_1_function(rel_distances, k_magnitude)
            rp = xp.einsum('ik,ik->i', relative_vectors, particle_dipoles)
            rpr_vec = rp[..., None] * relative_vectors
            scattering_field[i:i+1] = k_magnitude**2 * (xp.einsum('j,jk->k', G_0, particle_dipoles) + xp.einsum('j,jk->k', G_1, rpr_vec))
    else:
        relative_vectors = positions[:, None, :] - particle_positions[None, :, :]
        rel_distances = xp.linalg.norm(relative_vectors, axis=-1)
        G_0 = G_0_function(rel_distances, k_magnitude)
        G_1 = G_1_function(rel_distances, k_magnitude)
        rp = xp.einsum('ijk,jk->ij', relative_vectors, particle_dipoles)
        rpr_vec = rp [..., None] * relative_vectors
        scattering_field = k_magnitude**2 * (xp.einsum('ij,jk->ik', G_0, particle_dipoles) + xp.einsum('ij,ijk->ik', G_1, rpr_vec))
        
    return scattering_field

def obtain_effective_dipole(particle_positions: ArrayLike, particle_dipoles: ArrayLike, k_magnitude: float, n_Sample: int = 500) -> ArrayLike:
    """
    Compute the effective dipole moments of particles due to their interactions.

    Parameters
    ----------
    particle_positions : ArrayLike
        The positions of the particles.
    particle_dipoles : ArrayLike
        The dipole moments of the particles.
    k_magnitude : float
        The wave number in the medium.

    Returns
    -------
    ArrayLike
        The effective dipole moments of the particles.
    """
    
    xp = get_backend(particle_positions)
    
    wl = 2 * pi / k_magnitude
    sys_extension = xp.max(xp.linalg.norm(particle_positions, axis=-1))
    if sys_extension > 0.1 * wl:
        print("Warning: The system extension is comparable to the wavelength. The effective dipole approximation may not be accurate.")
    
    max_length = max(sys_extension, wl)
    
    sphere_radius = 1e3 * max_length
    eval_positions = sample_fibonacci_sphere(n_Sample, sphere_radius, xp)
    Esca = compute_scattering_field(eval_positions, particle_positions, particle_dipoles, k_magnitude)
    superG = k_magnitude**2 * pairwise_green_tensor(eval_positions, k_magnitude).reshape(-1, 3)
    superE = Esca.reshape(-1)
    
    P_fit, residual, *_ = xp.linalg.lstsq(superG, superE, rcond=None)
    
    return P_fit.flatten(), residual
    
    
    
    
    

