from .backend import get_backend
from typing import Iterable
from numpy.typing import ArrayLike

from msptools.dipole_moments import calculate_dipole_moments_linear

def solve_MSP_from_arrays(polarizability: ArrayLike,
                          external_field : ArrayLike,
                          wave_number : float,
                          green_tensor : ArrayLike,
                          method : str = 'Inverse',
                          **kwargs) -> ArrayLike:

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
    ArrayLike
        The solution to the MSP.

    """
    
    print(f"Solving MSP with method '{method}'...")
    
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

def array_MSP_iterative(polarizability : ArrayLike,
                          external_field : ArrayLike,
                          wave_number : float,
                          green_tensor : ArrayLike,
                          num_iterations : int = 500,
                          tolerance : float = 1e-6) -> ArrayLike:
    
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
    xp.ndarray
        The solution to the MSP.
    """

    xp = get_backend(external_field)
    old_field = external_field.copy()

    for iteration in range(num_iterations):
        
        dipole_moments = calculate_dipole_moments_linear(polarizability, old_field)
        scattered_field = wave_number**2 * xp.einsum('ijmn,jn->im', green_tensor, dipole_moments)
        new_field = external_field + scattered_field

        if xp.allclose(new_field, old_field, rtol=tolerance):
            break
        old_field = new_field.copy()

    if iteration == num_iterations - 1:
        print(f"Warning: MSP iterative solution did not converge within {num_iterations} iterations.")

    return new_field

def array_MSP_inverse(polarizability : ArrayLike,
                        external_field : ArrayLike,
                        wave_number : float,
                        green_tensor : ArrayLike) -> ArrayLike:
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
        xp.ndarray
            The solution to the MSP.
        """
        xp = get_backend(external_field)
        num_particles = external_field.shape[0]
        dimensions = external_field.shape[1]
        print(f"Green tensor shape: {green_tensor.shape}, External field shape: {external_field.shape}, Polarizability shape: {polarizability.shape}")
        
        scattering_matrix = wave_number**2 * xp.einsum('ijmk,jkl->ijml', green_tensor, polarizability)

        MSP_matrix = xp.eye(num_particles * dimensions) - scattering_matrix.transpose(0,2,1,3).reshape(num_particles * dimensions, num_particles * dimensions)
        
        print(f"Using backend: {xp.__name__} for MSP inverse solution.")
        total_field = xp.linalg.solve(MSP_matrix, external_field.flatten())
        total_field = total_field.reshape(num_particles, dimensions)
        return total_field

def MSP_gradient_from_arrays(dipole_moments: ArrayLike,
                             external_gradient : ArrayLike,
                             wave_number : float,
                             green_tensor_derivative : ArrayLike) -> ArrayLike:
    """
    Compute the gradient of the MSP solution with respect to particle positions.

    Parameters
    ----------
    polarizability :
        Polarizability of the particles.
    MS_field :
        Multiple scattering field on particles positions.
    external_gradient :
        Gradient of the external field on particles positions.
    wave_number :
        Wave number of the incident wave.
    green_tensor_derivative :
        Derivative of the Green's tensor with respect to particle positions.

    Returns
    -------
    xp.ndarray
        The gradient of the MSP solution with respect to particle positions.
    
    Notes
    -----
    The gradient is returned as an array of shape (N, d, d) where N is the number of particles and d is the dimensionality.
    """

    xp = get_backend(external_gradient)
    scattered_gradient = wave_number**2 * xp.einsum('ijcmn,jn->icm', green_tensor_derivative, dipole_moments)
    
    MSP_gradient = external_gradient + scattered_gradient

    return MSP_gradient