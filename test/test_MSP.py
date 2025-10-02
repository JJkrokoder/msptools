import pytest
import numpy as np
from msptools.MSP import solve_MSP_from_arrays, array_MSP_iterative, array_MSP_inverse
from msptools.dipole_moments import calculate_dipole_moments_linear, polarizability_to_matrix

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True)

def create_identity_green_tensor(num_particles, dimension):
    """
    Create an identity Green's tensor for testing purposes.
    
    Parameters
    ----------
    num_particles : int
        Number of particles.
    dimension : int
        Dimensionality of the system.
    
    Returns
    -------
    np.ndarray
        Identity Green's tensor of shape (num_particles, num_particles, dimension, dimension).
    """
    green_tensor = np.zeros((num_particles, num_particles, dimension, dimension))

    for i in range(num_particles):
        for j in range(num_particles):
            if i == j:
                green_tensor[i, j, :, :] = np.eye(dimension)

    return green_tensor

class Test_MSP_solver_from_arrays:
    num_particles = 5
    dimension = 3
    polarizability = 2.0 + 1.0j
    external_field = np.random.rand(num_particles, dimension)
    wave_number = 2.0

    def test_error_handling(self):
        with pytest.raises(ValueError):
            solve_MSP_from_arrays(polarizability=self.polarizability, external_field=self.external_field, wave_number=self.wave_number, green_tensor=np.eye(3), method='Unknown')
    
    def test_incorrect_green_tensor_shape(self):
        with pytest.raises(ValueError):
            solve_MSP_from_arrays(polarizability=self.polarizability, external_field=self.external_field, wave_number=self.wave_number, green_tensor=np.random.rand(3, 3, 2, 2), method='Iterative')
        
    @pytest.mark.parametrize("method", ['Iterative', 'Inverse'])
    def test_zero_polarizability(self, method):
        zero_polarizability = 0.0
        green_tensor = np.random.rand(self.num_particles, self.num_particles, self.dimension, self.dimension) + 1j * np.random.rand(self.num_particles, self.num_particles, self.dimension, self.dimension)
        total_field = solve_MSP_from_arrays(zero_polarizability, self.external_field, self.wave_number, green_tensor, method=method)
        assert np.allclose(total_field, self.external_field), "Total field should equal external field when polarizability is zero."
    
    def test_scalar_scattering_matrix(self):
        identity_green_tensor = create_identity_green_tensor(self.num_particles, self.dimension)
        g_factor = 0.001
        green_tensor = identity_green_tensor * g_factor
        total_field = solve_MSP_from_arrays(self.polarizability, self.external_field, self.wave_number, green_tensor, method='Iterative')
        factor = 1/(1 - self.wave_number**2 * self.polarizability * g_factor)
        assert np.allclose(total_field, factor * self.external_field, rtol=1e-6), "Total field did not match expected value."

class Test_MSP_iterative:
    num_particles = 2
    dimension = 3
    polarizability = 1.0 + 0.5j
    external_field = np.random.rand(num_particles, dimension)
    wave_number = 1.0
    green_tensor = np.random.rand(num_particles, num_particles, dimension, dimension)\
        + 1j * np.random.rand(num_particles, num_particles, dimension, dimension)
    tolerance = 1e-6

    @pytest.mark.parametrize("GT_scale", [5, 2, 1, 0.5])
    def test_non_convergence_error(self, GT_scale):
        with pytest.raises(ValueError, match="The new field is significantly larger than the external field, indicating potential divergence in the iterative method."):
            array_MSP_iterative(self.polarizability, self.external_field, self.wave_number, self.green_tensor * GT_scale, tolerance=self.tolerance)
        
    def test_zero_scattering(self):
        zero_green_tensor = np.zeros((self.num_particles, self.num_particles, self.dimension, self.dimension))
        total_field = array_MSP_iterative(self.polarizability, self.external_field, self.wave_number, zero_green_tensor, tolerance=self.tolerance)
        assert np.allclose(total_field, self.external_field), "Total field should equal external field when green tensor is zero."
    
    def test_convergence(self):
        small_green_tensor = 0.1 * self.green_tensor
        total_field = array_MSP_iterative(self.polarizability, self.external_field, self.wave_number, small_green_tensor, tolerance=self.tolerance)
        polarizability_matrix = polarizability_to_matrix(self.polarizability, self.num_particles, self.dimension)
        scattering_matrix = self.wave_number**2 *\
            small_green_tensor.transpose(0,2,1,3).reshape(self.num_particles * self.dimension, self.num_particles * self.dimension)\
                @ polarizability_matrix
        scattering_field = np.zeros_like(self.external_field.flatten(), dtype=np.complex128)
        total_scattering_matrix = scattering_matrix.copy()

        for i in range(500):
            scattering_field += total_scattering_matrix @ self.external_field.flatten()
            total_scattering_matrix = total_scattering_matrix @ scattering_matrix

        expected_field = self.external_field + scattering_field.reshape(self.num_particles, self.dimension)

        assert np.allclose(total_field, expected_field, rtol=self.tolerance), "Total field did not converge to expected value."
    
    def test_convergence_with_tolerance(self):
        small_green_tensor = 1e-4 * self.green_tensor
        total_field = array_MSP_iterative(self.polarizability, self.external_field, self.wave_number, small_green_tensor, tolerance=self.tolerance)
        dipole_moments = calculate_dipole_moments_linear(self.polarizability, total_field)
        new_iteration_field = self.external_field.flatten() + self.wave_number**2 * small_green_tensor.transpose(0,2,1,3).reshape(self.num_particles * self.dimension, self.num_particles * self.dimension) @ dipole_moments.flatten()
        new_iteration_field = new_iteration_field.reshape(self.num_particles, self.dimension)

        assert np.allclose(total_field, new_iteration_field, rtol=self.tolerance), "Total field did not converge to expected value with specified tolerance."

class Test_MSP_inverse:
    num_particles = 3
    dimension = 3
    polarizability = 1.0 + 0.5j
    external_field = np.random.rand(num_particles, dimension)
    wave_number = 1.0
    green_tensor = (np.random.rand(num_particles, num_particles, dimension, dimension)\
        + 1j * np.random.rand(num_particles, num_particles, dimension, dimension)) * 1e-3
    
    def test_invertibility(self):
        total_field = array_MSP_inverse(self.polarizability, self.external_field, self.wave_number, self.green_tensor)
        assert total_field is not None, "Total field should not be None."

    def test_autoconsistency(self):
        total_field = array_MSP_inverse(self.polarizability, self.external_field, self.wave_number, self.green_tensor)
        polarizability_matrix = polarizability_to_matrix(self.polarizability, self.num_particles, self.dimension)
        green_tensor_matrix = self.green_tensor.transpose(0,2,1,3).reshape(self.num_particles * self.dimension, self.num_particles * self.dimension)
        MSP_matrix = np.eye(self.num_particles * self.dimension) - self.wave_number**2 * green_tensor_matrix @ polarizability_matrix
        MSP_matrix_inv = np.linalg.inv(MSP_matrix)
        expected_field = MSP_matrix_inv @ self.external_field.flatten()
        
        assert np.allclose(total_field.flatten(), expected_field), "Total field from inverse method did not match expected value."

    def test_zero_green_tensor(self):
        zero_green_tensor = np.zeros((self.num_particles, self.num_particles, self.dimension, self.dimension))
        total_field = array_MSP_inverse(self.polarizability, self.external_field, self.wave_number, zero_green_tensor)
        assert np.allclose(total_field, self.external_field), "Total field should equal external field when green tensor is zero."
    
    def test_consistency_with_iterative(self):
        iterative_field = array_MSP_iterative(self.polarizability, self.external_field, self.wave_number, self.green_tensor)
        inverse_field = array_MSP_inverse(self.polarizability, self.external_field, self.wave_number, self.green_tensor)

        assert np.allclose(iterative_field, inverse_field, rtol=1e-6), "Fields from iterative and inverse methods did not match."



