import numpy as np
from msptools.dipole_moments import calculate_dipole_moments_linear, polarizability_to_matrix

def solve_MSP_from_arrays(polarizability,
                          external_field : np.ndarray,
                          wave_number : float,
                          green_tensor : np.ndarray,
                          method : str = 'Iterative',
                          **kwargs) -> np.ndarray:

    """
    Solve the Multiple Scattering Problem (MSP) using the provided arrays.

    Parameters
    ----------
    polarizability :
        Polarizability of the particles, can be a complex number, float, int, list, or numpy array.
    external_field :
        External field on particles positions.
    wave_number :
        Wave number of the incident wave.
    green_tensor :
        Green's tensor for the system.
    method :
        Method to solve the MSP, either 'Iterative' or 'Inverse'. The default is 'Iterative'.

    Returns
    -------
    np.ndarray
        The solution to the MSP.

    """
    
    if green_tensor.ndim != 4 or green_tensor.shape[0] != green_tensor.shape[1] or green_tensor.shape[2] != green_tensor.shape[3]:
        raise ValueError("Invalid green_tensor shape. Expected shape (N, N, d, d), got {}".format(green_tensor.shape))
    if green_tensor.shape[0] != external_field.shape[0]:
        raise ValueError("The first dimension of green_tensor must match the number of particles in external_field. Expected {}, got {}".format(external_field.shape[0], green_tensor.shape[0]))
    if green_tensor.shape[2] != external_field.shape[1]:
        raise ValueError("The third dimension of green_tensor must match the system dimensionality. Expected {}, got {}".format(external_field.shape[1], green_tensor.shape[2]))

    if method == 'Iterative':
        if 'tolerance' in kwargs:
            tolerance = kwargs['tolerance']
            return array_MSP_iterative(polarizability, external_field, wave_number, green_tensor, tolerance=tolerance)
        else:
            return array_MSP_iterative(polarizability, external_field, wave_number, green_tensor)
    elif method == 'Inverse':
        return array_MSP_inverse(polarizability, external_field, wave_number, green_tensor)
    else:
        raise ValueError("Unknown method: {}".format(method))

def array_MSP_iterative(polarizability : np.ndarray,
                          external_field : np.ndarray,
                          wave_number : float,
                          green_tensor : np.ndarray,
                          num_iterations : int = 500,
                          tolerance : float = 1e-6) -> np.ndarray:
    
    """
    Solve the MSP using an iterative method.

    Parameters
    ----------
    polarizability :
        Polarizability of the particles.
    external_field :
        External field on particles positions.
    wave_number :
        Wave number of the incident wave.
    green_tensor :
        Green's tensor for the system.
    num_iterations : optional
        Maximum number of iterations for the iterative method. Default is 500.
    tolerance : optional
        Convergence tolerance for the iterative method. Default is 1e-6.

    Returns
    -------
    np.ndarray
        The solution to the MSP.
    """

    num_particles = external_field.shape[0]
    dimensions = external_field.shape[1]

    green_tensor_matrix = green_tensor.transpose(0,2,1,3).reshape(num_particles * dimensions, num_particles * dimensions)
    external_field_array = external_field.reshape(num_particles * dimensions, 1)
    old_field = external_field_array.copy()

    for iteration in range(num_iterations):
        
        dipole_moments = calculate_dipole_moments_linear(polarizability, old_field.reshape(num_particles, dimensions))
        dipole_moments = dipole_moments.reshape(num_particles*dimensions, 1)
        scattered_field = wave_number**2 * green_tensor_matrix @ dipole_moments
        new_field = external_field_array + scattered_field

        if np.any(np.abs(new_field)/np.abs(external_field_array) > 1e6 ):
            raise ValueError("The new field is significantly larger than the external field, indicating potential divergence in the iterative method.")

        if np.allclose(new_field, old_field, rtol=tolerance):
            break
        old_field = new_field.copy()

    if iteration == num_iterations - 1:
        print(f"Warning: MSP iterative solution did not converge within {num_iterations} iterations.")

    return new_field.reshape(num_particles, dimensions)

def array_MSP_inverse(polarizability : np.ndarray,
                        external_field : np.ndarray,
                        wave_number : float,
                        green_tensor : np.ndarray) -> np.ndarray:
        """
        Solve the MSP using the inverse method.
    
        Parameters
        ----------
        polarizability :
            Polarizability of the particles.
        external_field :
            External field on particles positions.
        wave_number :
            Wave number of the incident wave.
        green_tensor :
            Green's tensor for the system.
    
        Returns
        -------
        np.ndarray
            The solution to the MSP.
        """

        num_particles = external_field.shape[0]
        dimensions = external_field.shape[1]

        green_tensor_matrix = green_tensor.transpose(0,2,1,3).reshape(num_particles * dimensions, num_particles * dimensions)
        external_field_array = external_field.reshape(num_particles * dimensions, 1)
        polarizability_matrix = polarizability_to_matrix(polarizability, num_particles, dimensions)

        MSP_matrix = np.eye(num_particles * dimensions) - wave_number**2 * green_tensor_matrix @ polarizability_matrix
        MSP_matrix_inv = np.linalg.inv(MSP_matrix)
        total_field = MSP_matrix_inv @ external_field_array
        return total_field.reshape(num_particles, dimensions)
