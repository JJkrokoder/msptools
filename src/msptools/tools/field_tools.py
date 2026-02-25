import logging

logging.basicConfig(level=logging.INFO)
try:
    import cupy as np

    logging.log(logging.INFO, "Using CUDA backend")
except:
    logging.log(logging.INFO, "Using Fallback numpy backend")
    import numpy as np


def plane_wave_function(
    direction: np.ndarray,
    amplitude_vec: np.ndarray,
    positions: np.ndarray,
    k_magnitude: float,
) -> np.ndarray:
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
    electric_field = np.outer(phase_factors, amplitude_vec.T)
    return electric_field


def plane_wave_gradient(
    direction: np.ndarray,
    amplitude_vec: np.ndarray,
    positions: np.ndarray,
    k_magnitude: float,
) -> np.ndarray:
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
    gradient = 1j * np.einsum(
        "ij,k -> ijk", np.outer(phase_factors, k_vector), amplitude_vec
    )
    return gradient


def gaussian_paraxial_function(
    direction: np.ndarray,
    amplitude_vec: np.ndarray,
    positions: np.ndarray,
    k_magnitude: float,
    beam_waist: float,
) -> np.ndarray:
    """
    Calculate the electric field of a Gaussian paraxial beam at given positions.

    Parameters
    ----------
    direction :
        The propagation direction of the beam as a 3-element list or array.
        It is assumed to be normalized.
    amplitude :
        The amplitude vector of the beam.
    positions :
        The positions at which to evaluate the field. It should be an array of shape (N, 3) where N is the number of positions.
    k_magnitude :
        The magnitude of the wave vector.
    beam_waist :
        The beam waist (radius at which the field amplitude drops to 1/e of its maximum value). It should be in the same units as positions and k_magnitude.

    Returns
    -------
    np.ndarray
        The electric field of the Gaussian paraxial beam at specified positions.

    Notes
    -----
    The electric field of a Gaussian paraxial beam can be approximated as:
    E(r) = A * (w0 / w(z)) * exp(-r_perp^2 / w(z)^2) * exp(i * (k * z + k * r_perp^2 / (2 * R(z)) - psi(z)))
    where:
    - A is the amplitude,
    - w0 is the beam waist,
    - w(z) is the beam radius at position z,
    - r_perp is the radial distance from the beam axis,
    - R(z) is the radius of curvature of the beam's wavefronts at position z,
    - psi(z) is the Gouy phase at position z.
    - positions and k_magnitude should be in consistent units.
    """


def standing_wave_function(
    direction: np.ndarray,
    amplitude_vec: np.ndarray,
    positions: np.ndarray,
    k_magnitude: float,
) -> np.ndarray:
    """
    Calculate the electric field of a standing wave at given positions.

    Parameters
    ----------
    amplitude :
        The amplitude vector of the standing wave.
    positions :
        The positions at which to evaluate the field.
    k_magnitude :
        The magnitude of the wave vector.

    Returns
    -------
    np.ndarray
        The electric field at specified positions.
    """

    phase_factors = np.cos(positions @ (direction * k_magnitude))
    electric_field = np.outer(phase_factors, amplitude_vec.T)
    return electric_field


def standing_wave_gradient(
    direction: np.ndarray,
    amplitude_vec: np.ndarray,
    positions: np.ndarray,
    k_magnitude: float,
) -> np.ndarray:
    """
    Calculate the gradient of the electric field of a standing wave at given positions.

    Parameters
    ----------
    amplitude :
        The amplitude vector of the standing wave.
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
    phase_factors = -np.sin(positions @ k_vector)
    gradient = np.einsum(
        "ij,k -> ijk", np.outer(phase_factors, k_vector), amplitude_vec
    )
    return gradient
