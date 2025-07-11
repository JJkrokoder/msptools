import numpy as np

def G_0_function(r: float, wave_number: float) -> complex:
    """
    Computes the G_0 function for a given distance r and wave number.

    Parameters
    ----------
    r : float
        The distance between two points.
    wave_number : float
        The wave number.

    Returns
    -------
    complex
        The value of the G_0 function.
    """

    return np.exp(1j * wave_number * r) / (4 * np.pi * r) * (1 + 1j/(wave_number * r) - 1/(wave_number * r)**2)

def G_1_function(r: float, wave_number: float) -> complex:
    """
    Computes the G_1 function for a given distance r and wave number.

    Parameters
    ----------
    r : float
        The distance between two points.
    wave_number : float
        The wave number.

    Returns
    -------
    complex
        The value of the G_1 function.
    """

    return -np.exp(1j * wave_number * r) / (4 * np.pi * r) * (1 + 3j/(wave_number * r) - 3/(wave_number * r)**2) / r**2

def G_0_derivative_function(r: float, wave_number: float) -> complex:
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
    
    return wave_number * np.exp(1j * wave_number * r) / (4 * np.pi * r) * \
           (1j - 2/(wave_number * r) - 3j/(wave_number * r)**2 + 3/(wave_number * r)**3)

def G_1_derivative_function(r: float, wave_number: float) -> complex:
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
    
    return -wave_number * np.exp(1j * wave_number * r) / (4 * np.pi * r**3) * \
           (1j - 6/(wave_number * r) - 15j/(wave_number * r)**2 + 15/(wave_number * r)**3)

def v_cross_derivative(r_vec: np.ndarray, coordinate: int) -> np.ndarray:
    """
    Computes the derivative of a vector cross dyadic product with respect to a specific coordinate.

    Parameters
    ----------
    r_vec : np.ndarray
        The vector for which the derivative is computed.
    coordinate : int
        The coordinate with respect to which the derivative is taken (0, 1, or 2).

    Returns
    -------
    np.ndarray
        The derivative of the cross product with respect to the specified coordinate.
    """

    dimensions = r_vec.shape[0]
    if coordinate < 0 or coordinate >= dimensions:
        raise ValueError("Coordinate must be in the range [0, {}]".format(dimensions - 1))

    der_R_cross = np.zeros((dimensions, dimensions))

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
    
    num_particles, dimensions = positions.shape
    green_tensor = np.zeros((num_particles, num_particles, dimensions, dimensions), dtype=np.complex128)

    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            green_tensor[i, j, :, :] = pair_green_tensor(positions[i], positions[j], wave_number)
            green_tensor[j, i, :, :] = green_tensor[i, j, :, :]
    return green_tensor

def pair_green_tensor(pos_i: np.ndarray, pos_j: np.ndarray, wave_number: float) -> np.ndarray:
    """
    Constructs the pair Green's tensor for two particles at positions pos_i and pos_j.

    Parameters
    ----------
    pos_i : np.ndarray
        Position of the first particle.
    pos_j : np.ndarray
        Position of the second particle.
    wave_number : float
        The wave number.

    Returns
    -------
    np.ndarray
        Pair Green's tensor of shape (dimension, dimension).
    """
    dimensions = pos_i.shape[0]
    R_vec = pos_i - pos_j
    r = np.linalg.norm(R_vec)

    g_0 = G_0_function(r, wave_number)
    g_1 = G_1_function(r, wave_number)

    R_cross = R_vec[:, None] @ R_vec[None, :]

    return g_0 * np.eye(dimensions) + g_1 * R_cross

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
    
    dimensions = pos_i.shape[0]
    R_vec = pos_i - pos_j
    r = np.linalg.norm(R_vec)

    g_1 = G_1_function(r, wave_number)
    der_g_0 = G_0_derivative_function(r, wave_number) * R_vec[coordinate] / r
    der_g_1 = G_1_derivative_function(r, wave_number) * R_vec[coordinate] / r**3
    R_cross = R_vec[:, None] @ R_vec[None, :]
    der_R_cross = v_cross_derivative(R_vec, coordinate)

    derivative_tensor = der_g_0 * np.eye(dimensions) + der_g_1 * R_cross + g_1 * der_R_cross
    
    return derivative_tensor 
