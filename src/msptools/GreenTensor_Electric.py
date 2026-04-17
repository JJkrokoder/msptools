from .backend import get_backend
from numpy.typing import ArrayLike
import numpy as np
from scipy.constants import pi
from cmath import exp

def G_0_function(r: float | ArrayLike, wave_number: float) -> complex | ArrayLike:
    """
    Computes the G_0 function for a given distance r and wave number.

    Parameters
    ----------
    r :
        The distance between two points.
    wave_number :
        The wave number.

    Returns
    -------
    complex | ArrayLike
        The value of the G_0 function.
    """
    if isinstance(r, (float, int)):
        if r > 0:
            kr = wave_number * r
            return exp(1j * wave_number * r) / (4 * pi * r) * (1 + 1j/kr - 1/kr**2)
        else:
            return 0.0j
    else:
        xp = get_backend(r)
        
        kr = wave_number * r 
        inv_kr = xp.where(kr > 0, 1/kr, 0.0)
        inv_r = inv_kr * wave_number
        inv_kr2 = inv_kr * inv_kr
        exp_ikr = xp.exp(1j * kr)
        
        return exp_ikr * (inv_r / (4 * pi)) * (1 + 1j*inv_kr - inv_kr2)

def G_1_function(r: float | ArrayLike, wave_number: float) -> complex | ArrayLike:
    """
    Computes the G_1 function for a given distance r and wave number.

    Parameters
    ----------
    r :
        The distance between two points.
    wave_number :
        The wave number.

    Returns
    -------
    complex | ArrayLike
        The value of the G_1 function.
    """
    if isinstance(r, (float, int)):
        if r > 0:
            kr = wave_number * r
            return -exp(1j * wave_number * r) / (4 * pi * r**3) * (1 + 3j/kr - 3/kr**2)
        else:
            return 0.0j
    else:
        xp = get_backend(r)
        
        kr = wave_number * r 
        inv_kr = xp.where(kr > 0, 1/kr, 0.0)
        inv_r = inv_kr * wave_number
        inv_r3 = inv_r * inv_r * inv_r
        inv_kr2 = inv_kr * inv_kr
        exp_ikr = xp.exp(1j * kr)
        
        result = -exp_ikr * (inv_r3 / (4 * pi)) * (1 + 3j*inv_kr - 3*inv_kr2)
        return result

def G_0_derivative_function(r: float | ArrayLike, wave_number: float) -> complex:
    """
    Computes the derivative of the G_0 function with respect to r.

    Parameters
    ----------
    r : float
        The distance between two points.
    wave_number : float
        The wave number.

    Returns
    -------
    complex
        The value of the derivative of the G_0 function.
    """
    if isinstance(r, (float, int)):
        xp = np
    else:
        xp = get_backend(r)
    return wave_number * xp.exp(1j * wave_number * r) / (4 * xp.pi * r) * \
           (1j - 2/(wave_number * r) - 3j/(wave_number * r)**2 + 3/(wave_number * r)**3)

def G_1_derivative_function(r: float | ArrayLike, wave_number: float) -> complex:
    """
    Computes the derivative of the G_1 function with respect to r.

    Parameters
    ----------
    r : float
        The distance between two points.
    wave_number : float
        The wave number.

    Returns
    -------
    complex
        The value of the derivative of the G_1 function.
    """
    if isinstance(r, (float, int)):
        xp = np
    else:
        xp = get_backend(r)
    return -wave_number * xp.exp(1j * wave_number * r) / (4 * xp.pi * r**3) * \
           (1j - 6/(wave_number * r) - 15j/(wave_number * r)**2 + 15/(wave_number * r)**3)

def v_cross_derivative(r_vec: ArrayLike, coordinate: int) -> np.ndarray:
    """
    Computes the derivative of a vector cross dyadic product with respect to a specific coordinate.

    Parameters
    ----------
    r_vec : 
        The vector for which the derivative is computed.
    coordinate : 
        The coordinate with respect to which the derivative is taken (0, 1, or 2).

    Returns
    -------
    np.ndarray
        The derivative of the cross product with respect to the specified coordinate.
    """
    xp = get_backend(r_vec)

    dimensions = r_vec.shape[0]
    if coordinate < 0 or coordinate >= dimensions:
        raise ValueError("Coordinate must be in the range [0, {}]".format(dimensions - 1))

    der_R_cross = xp.zeros((dimensions, dimensions))

    for i in range(dimensions):
        if i == coordinate:
            der_R_cross[i, i] = 2 * r_vec[i]
        else:
            der_R_cross[i, coordinate] = r_vec[i]
            der_R_cross[coordinate, i] = r_vec[i]

    return der_R_cross

def construct_green_tensor(positions : np.ndarray, wave_number: float) -> np.ndarray:
    """
    Constructs the Green's tensor for a given set of positions and wave number.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (num_particles, dimension) containing the positions of the particles.
    wave_number : float
        The wave number.

    Returns
    -------
    np.ndarray
        Green's tensor of shape (num_particles, num_particles, dimension, dimension).
    """

    rel_vec_matrix = positions[:, None, :] - positions[None, :, :]
    green_tensor = construct_green_tensor_from_rel_vecs(rel_vec_matrix, wave_number)
    return green_tensor

def scat_green_field_from_rel_vecs_dipoles(rel_vecs: ArrayLike, 
                                         p: ArrayLike, 
                                         k: float) -> ArrayLike:
    """
    Applies the Green's function to a set of dipole moments given the relative position vectors and wave number.
    
    Parameters   
    ----------
    rel_vecs :
        Relative position vectors between particles, of shape (num_particles, num_particles, dimension).
    p :
        Dipole moments of the particles, of shape (num_particles, dimension).
    k :
        Wave number.
        
    Returns
    -------
    ArrayLike
        The resulting field after applying the Green's function, of shape (num_particles, dimension).
    """
    xp = get_backend(rel_vecs)

    r = xp.linalg.norm(rel_vecs, axis=-1)

    # scalars
    G0 = G_0_function(r, k)
    G1 = G_1_function(r, k)

    # direct contraction WITHOUT forming tensor
    rp = xp.sum(rel_vecs * p, axis=-1)          # (B, N)

    term1 = G0[...,None] * p                   # isotropic part
    term2 = G1[...,None] * rel_vecs * rp[...,None] 
    
    scattering_field = xp.sum(k**2*(term1 + term2), axis=1)  # sum over source particles

    return scattering_field

def construct_green_tensor_from_rel_vecs(rel_vecs : np.ndarray, wave_number: float) -> np.ndarray:
    """
    Constructs the Green's tensor for a given set of relative position vectors and wave number.

    Parameters
    ----------
    rel_vecs : np.ndarray
        Array of shape (num_pairs, dimension) containing the relative position vectors between pairs of particles.
    wave_number : float
        The wave number.

    Returns
    -------
    np.ndarray
        Green's tensor of shape (num_pairs, dimension, dimension).
    """
    xp = get_backend(rel_vecs)
    dimensions = rel_vecs.shape[-1]
    distances = xp.linalg.norm(rel_vecs, axis=-1)
    G_0_values = G_0_function(distances, wave_number)
    G_1_values = G_1_function(distances, wave_number)
    R_cross_values = rel_vecs[:, :, :, None] * rel_vecs[:, :, None, :]
    green_tensor = G_0_values[:, :, None, None] * xp.eye(dimensions) + G_1_values[:, :, None, None] * R_cross_values
    
    return green_tensor

def pairwise_green_tensor(relative_positions : ArrayLike, wave_number: float) -> ArrayLike:
    """
    Constructs the pairwise Green's tensor for a given set of relative positions and wave number.

    Parameters
    ----------
    relative_positions : ArrayLike
        Array of shape (num_pairs, dimension) or (dimension,) containing the relative positions between pairs of particles.
    wave_number : float
        The wave number.

    Returns
    -------
    ArrayLike
        Pairwise Green's tensor of shape (num_pairs, dimension, dimension).
    """
    xp = get_backend(relative_positions)
    
    dimensions = relative_positions.shape[-1]
    
    distances = xp.linalg.norm(relative_positions, axis=-1)
    Identity = xp.eye(dimensions)
    G_0_values = G_0_function(distances, wave_number)
    G_1_values = G_1_function(distances, wave_number)
    if relative_positions.ndim == 1:
        R_cross_values = relative_positions[:, None] * relative_positions[None, :]
        green_tensor = G_0_values * Identity + G_1_values * R_cross_values
    else:
        R_cross_values = relative_positions[:, :, None] * relative_positions[:, None, :]
        green_tensor = G_0_values[:, None, None] * Identity + G_1_values[:, None, None] * R_cross_values
    return green_tensor


def pair_green_tensor_derivative(pos_i: np.ndarray, pos_j: np.ndarray, coordinate : int,  wave_number: float):
    """
    Constructs the derivative of the pair Green's tensor with respect to a specific coordinate.

    Parameters
    ----------
    pos_i : np.ndarray
        Position of the first particle.
    pos_j : np.ndarray
        Position of the second particle.
    coordinate : int
        The coordinate with respect to which the derivative is taken (0, 1, or 2).
    wave_number : float
        The wave number.

    Returns
    -------
    np.ndarray
        Derivative of the pair Green's tensor with respect to the specified coordinate.
    """
    xp = get_backend(pos_i)
    dimensions = pos_i.shape[0]
    R_vec = pos_i - pos_j
    r = xp.linalg.norm(R_vec)

    g_1 = G_1_function(r, wave_number)
    der_g_0 = G_0_derivative_function(r, wave_number) * R_vec[coordinate] / r
    der_g_1 = G_1_derivative_function(r, wave_number) * R_vec[coordinate] / r
    R_cross = R_vec[:, None] @ R_vec[None, :]
    der_R_cross = v_cross_derivative(R_vec, coordinate)

    derivative_tensor = der_g_0 * xp.eye(dimensions) + der_g_1 * R_cross + g_1 * der_R_cross
    
    return derivative_tensor 

def construct_green_tensor_gradient(positions : np.ndarray, wave_number: float) -> np.ndarray:
    """
    Constructs the derivative of the Green's tensor for a given set of positions and wave number.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (num_particles, dimension) containing the positions of the particles.
    wave_number : float
        The wave number.

    Returns
    -------
    np.ndarray
        Derivative of Green's tensor of shape (num_particles, num_particles, dimension, dimension, dimension).
    """
    xp = get_backend(positions)
    
    num_particles, dimensions = positions.shape
    green_tensor_derivative = xp.zeros((num_particles, num_particles, dimensions, dimensions, dimensions), dtype=xp.complex128)

    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            for coord in range(dimensions):
                green_tensor_derivative[i, j, coord, :, :] = pair_green_tensor_derivative(positions[i], positions[j], coord, wave_number)
                green_tensor_derivative[j, i, coord, :, :] = -green_tensor_derivative[i, j, coord, :, :]
    return green_tensor_derivative
 

def scattering_term(rel_vecs : ArrayLike, wave_number : float, dipole_moments : ArrayLike) -> ArrayLike:
    """
    Compute the scattering term for the MSP without explicitly constructing the Green's tensor.

    Parameters
    ----------
    rel_vecs :
        Relative position vectors between particles.
    wave_number :
        Wave number of the incident wave.
    dipole_moments :
        Dipole moments of the particles.
        
    Returns
    -------
    xp.ndarray
        The scattering term for the MSP.
    """
    xp = get_backend(rel_vecs)
    k2 = wave_number**2
    
    num_particles, dimensions = rel_vecs.shape[0], rel_vecs.shape[-1]
    
    # mask self-interactions
    mask = ~xp.eye(num_particles, dtype=bool)
    
    scattering_field = xp.zeros((num_particles, dimensions), dtype=xp.complex128)
    
    for j in range(num_particles):
        rel_vecs_j = rel_vecs[j][mask[j]]
        dipoles_l = dipole_moments[mask[j]]
        
        G_blocks = pairwise_green_tensor(rel_vecs_j, wave_number)
        
        scattering_field[j] = xp.einsum('lnm,lm->n', G_blocks, dipoles_l*k2)
    return scattering_field

def scattering_term_batched(
    rel_vecs: ArrayLike,
    wave_number: float,
    dipole_moments: ArrayLike,
    batch_size: int = 128
):
    xp = get_backend(rel_vecs)
    N, d = rel_vecs.shape[0], rel_vecs.shape[-1]
    k2 = wave_number**2

    scattering_field = xp.zeros((N, d), dtype=xp.complex128)

    for i0 in range(0, N, batch_size):
        i1 = min(i0 + batch_size, N)

        # (B, 1, d) - (1, N, d) → (B, N, d)
        rel_vec_block = rel_vecs[i0:i1,:,:]

        # Compute Green tensor: (B, N, d, d)
        G = construct_green_tensor_from_rel_vecs(rel_vec_block, wave_number)

        # Mask self-interaction ONLY if batch overlaps diagonal
        if i0 <= i1:
            idx = xp.arange(i0, i1)
            G[xp.arange(i1 - i0), idx, :, :] = 0

        # Contract: (B,N,d,d) × (N,d) → (B,d)
        scattering_field[i0:i1] = k2 * xp.einsum(
            'ijmn,jn->im',
            G,
            dipole_moments
        )

    return scattering_field

        