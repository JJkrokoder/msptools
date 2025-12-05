import numpy as np


def plane_wave_function(direction: np.ndarray,
                        amplitude_vec: np.ndarray,
                        positions: np.ndarray,
                        k_magnitude: float) -> np.ndarray:
    """
    Calculate the electric field of a plane wave at given positions.

    Parameters
    ----------
    direction :
        The propagation direction of the plane wave as a 3-element list or array.
        It is assumed to be normalized.
    amplitude :
        The amplitude vector of the plane wave.
    positions :
        The positions at which to evaluate the field. It should be an array of shape (N, 3) where N is the number of positions.
    k_magnitude :
        The magnitude of the wave vector.

    Returns
    -------
    np.ndarray
        The electric field at specified positions.

    Notes
    -----
    The electric field of a plane wave is given by:
    E(r) = A * exp(i * k · r)
    where A is the amplitude, k is the wave vector, and r is the position vector
    - positions and k_magnitude should be in consistent units.
    """
    
    k_vector = direction * k_magnitude

    phase_factors = np.exp(1j * positions @ k_vector)
    electric_field = phase_factors[:, np.newaxis] * amplitude_vec.T
    return electric_field

def plane_wave_gradient(direction: np.ndarray,
                        amplitude_vec: np.ndarray,
                        positions: np.ndarray,
                        k_magnitude: float) -> np.ndarray:
    """
    Calculate the gradient of the electric field of a plane wave at given positions.

    Parameters
    ----------
    direction :
        The propagation direction of the plane wave as a 3-element list or array.
        It is assumed to be normalized.
    amplitude :
        The amplitude vector of the plane wave.
    positions :
        The positions at which to evaluate the field gradient.
    k_magnitude :
        The magnitude of the wave vector.

    Returns
    -------
    np.ndarray
        The gradient of the electric field at specified positions.
    """
    k_vector = direction * k_magnitude
    phase_factors = np.exp(1j * positions @ k_vector)
    gradient = 1j * np.einsum('ij,k -> ijk',np.outer(phase_factors, k_vector), amplitude_vec)
    return gradient