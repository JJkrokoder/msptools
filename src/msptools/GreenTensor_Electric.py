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
        mask = r > 0
        result = xp.zeros_like(r, dtype=complex)
        kr = wave_number * r[mask]
        result[mask] = xp.exp(1j * kr) / (4 * pi * r[mask]) * (1 + 1j/kr - 1/kr**2)
        return result

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
        mask = r > 0
        result = xp.zeros_like(r, dtype=complex)
        kr = wave_number * r[mask]
        result[mask] = -xp.exp(1j * kr) / (4 * xp.pi * r[mask]**3) * (1 + 3j/kr - 3/kr**2)
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
    xp = get_backend(positions)
    dimensions = positions.shape[1]
    rel_vec_matrix = positions[:, None, :] - positions[None, :, :]
    distances = xp.linalg.norm(rel_vec_matrix, axis=-1)
    G_0_matrix = G_0_function(distances, wave_number)
    G_1_matrix = G_1_function(distances, wave_number)
    R_cross_matrix = rel_vec_matrix[:, :, :, None] * rel_vec_matrix[:, :, None, :]
    green_tensor = G_0_matrix[:, :, None, None] * xp.eye(dimensions) + G_1_matrix[:, :, None, None] * R_cross_matrix
    
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
    G_0_values = G_0_function(distances, wave_number)
    G_1_values = G_1_function(distances, wave_number)
    if relative_positions.ndim == 1:
        R_cross_values = relative_positions[:, None] * relative_positions[None, :]
        green_tensor = G_0_values * xp.eye(dimensions) + G_1_values * R_cross_values
    else:
        R_cross_values = relative_positions[:, :, None] * relative_positions[:, None, :]
        green_tensor = G_0_values[:, None, None] * xp.eye(dimensions) + G_1_values[:, None, None] * R_cross_values
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
 

def scattering_term(positions : ArrayLike, wave_number : float, dipole_moments : ArrayLike) -> ArrayLike:
    """
    Compute the scattering term for the MSP without explicitly constructing the Green's tensor.

    Parameters
    ----------
    positions :
        Positions of the particles.
    wave_number :
        Wave number of the incident wave.
    dipole_moments :
        Dipole moments of the particles.
        
    Returns
    -------
    xp.ndarray
        The scattering term for the MSP.
    """
    xp = get_backend(positions)
    num_particles, dimensions = positions.shape
    
    # Pairwise differences
    rel_vec_matrix = positions[:, None, :] - positions[None, :, :]
    
    # mask self-interactions
    mask = ~xp.eye(num_particles, dtype=bool)
    
    scattering_field = xp.zeros_like(positions, dtype=xp.complex128)
    
    for j in range(num_particles):
        rel_vecs_j = rel_vec_matrix[j][mask[j]]
        dipoles_l = dipole_moments[mask[j]]
        
        G_blocks = pairwise_green_tensor(rel_vecs_j, wave_number)
        
        scattering_field[j] = xp.einsum('lnm,lm->n', G_blocks, dipoles_l*wave_number**2)
    return scattering_field

        