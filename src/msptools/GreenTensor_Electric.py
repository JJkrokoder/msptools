import numpy as np

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
    
    R_vec = pos_i - pos_j
    r = np.linalg.norm(R_vec)

    g_0 = np.exp(1j * wave_number * r) / (4 * np.pi * r) * (1 + 1j/(wave_number * r) - 1/(wave_number * r)**2)
    g_1 = -np.exp(1j * wave_number * r) / (4 * np.pi * r) * (1 + 3j/(wave_number * r) - 3/(wave_number * r)**2)

    R_cross = np.array([[R_vec[0]**2, R_vec[0]*R_vec[1], R_vec[0]*R_vec[2]],
                        [R_vec[0]*R_vec[1], R_vec[1]**2, R_vec[1]*R_vec[2]],
                        [R_vec[0]*R_vec[2], R_vec[1]*R_vec[2], R_vec[2]**2]])
    
    return g_0 * np.eye(3) + g_1 * R_cross


