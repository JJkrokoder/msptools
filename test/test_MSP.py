import pytest
try:
    import cupy as np
except ImportError:
    import numpy as np
from msptools.MSP import *
from msptools.dipole_moments import calculate_dipole_moments_linear
from msptools.polarizability_mod import polarizability_to_matrix
from msptools.GreenTensor_Electric import construct_green_tensor
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

    idx = np.arange(num_particles)
    green_tensor[idx, idx, :, :] = np.eye(dimension)

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
        zero_polarizability = np.zeros((self.num_particles, self.dimension, self.dimension))
        green_tensor = np.random.rand(self.num_particles, self.num_particles, self.dimension, self.dimension) + 1j * np.random.rand(self.num_particles, self.num_particles, self.dimension, self.dimension)
        total_field = solve_MSP_from_arrays(zero_polarizability, self.external_field, self.wave_number, green_tensor, method=method)
        assert np.allclose(total_field, self.external_field), "Total field should equal external field when polarizability is zero."
    
    def test_scalar_scattering_matrix(self):
        identity_green_tensor = create_identity_green_tensor(self.num_particles, self.dimension)
        g_factor = 0.001
        green_tensor = identity_green_tensor * g_factor
        polarizability_array = np.repeat(self.polarizability * np.eye(self.dimension)[None,:,:], self.num_particles, axis=0)
        total_field = solve_MSP_from_arrays(polarizability_array, self.external_field, self.wave_number, green_tensor, method='Iterative')
        factor = 1/(1 - self.wave_number**2 * g_factor * self.polarizability)
        assert np.allclose(total_field, factor * self.external_field, rtol=1e-6), "Total field did not match expected value."

class Test_MSP_iterative:
    num_particles = 2
    dimension = 3
    polarizability = np.repeat((1.0 + 0.5j)*np.eye(dimension)[None,:,:], num_particles, axis=0)
    external_field = np.random.rand(num_particles, dimension)
    wave_number = 1.0
    green_tensor = np.random.rand(num_particles, num_particles, dimension, dimension)\
        + 1j * np.random.rand(num_particles, num_particles, dimension, dimension)
    tolerance = 1e-6

       
    def test_zero_scattering(self):
        zero_green_tensor = np.zeros((self.num_particles, self.num_particles, self.dimension, self.dimension))
        total_field = array_MSP_iterative(self.polarizability, self.external_field, self.wave_number, zero_green_tensor, tolerance=self.tolerance)
        assert np.allclose(total_field, self.external_field), "Total field should equal external field when green tensor is zero."
    
    def test_convergence(self):
        small_green_tensor = 0.1 * self.green_tensor
        total_field = array_MSP_iterative(self.polarizability, self.external_field, self.wave_number, small_green_tensor, tolerance=self.tolerance)
        scattering_matrix = self.wave_number**2 *np.einsum('ijmk,jkl->ijml', small_green_tensor, self.polarizability)
        scattering_field = np.zeros_like(self.external_field)
        total_scattering_matrix = scattering_matrix.copy()

        for i in range(500):
            scattering_field = scattering_field + np.einsum('ijml,jl->im', total_scattering_matrix, self.external_field)
            total_scattering_matrix = np.einsum('ijmn,jkno->ikmo', total_scattering_matrix, scattering_matrix)

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
    polarizability = np.repeat((1.0 + 0.5j)*np.eye(dimension)[None,:,:], num_particles, axis=0)
    external_field = np.random.rand(num_particles, dimension)
    wave_number = 1.0
    green_tensor = (np.random.rand(num_particles, num_particles, dimension, dimension)\
        + 1j * np.random.rand(num_particles, num_particles, dimension, dimension)) * 1e-3
    
    def test_invertibility(self):
        total_field = array_MSP_inverse(self.polarizability, self.external_field, self.wave_number, self.green_tensor)
        assert total_field is not None, "Total field should not be None."

    def test_autoconsistency(self):
        total_field = array_MSP_inverse(self.polarizability, self.external_field, self.wave_number, self.green_tensor)
        MSP_matrix = np.eye(self.num_particles * self.dimension) - self.wave_number**2 * np.einsum('ijmk,jkl->ijml', self.green_tensor, self.polarizability).transpose(0,2,1,3).reshape(self.num_particles * self.dimension, self.num_particles * self.dimension)
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

class Test_MSP_examples:

    dimension = 3
    @pytest.mark.parametrize("x", [5.0, 10.0, 20.0])
    def test_2_particles_at_z0(self, x):

        polarizability_scalar = 1.0 + 0.5j
        polarizability = np.repeat(polarizability_scalar*np.eye(self.dimension)[None,:,:], 2, axis=0)
        external_field = np.array([[1, 0, 0],
                                   [1, 0, 0]])
        num_particles = external_field.shape[0]
        dimension = external_field.shape[1]
        wave_number = 1.0

        positions = np.array([[0, 0, 0],
                              [x, 0, 0]])
        
        green_tensor = construct_green_tensor(positions, wave_number)

        total_field = solve_MSP_from_arrays(polarizability, external_field, wave_number, green_tensor)

        expected_field_particle_1 = external_field[0] / (1 - wave_number**2 * polarizability_scalar * green_tensor[0,1, 0,0])
        expected_field_particle_2 = external_field[1] / (1 - wave_number**2 * polarizability_scalar * green_tensor[1,0, 0,0])

        assert np.allclose(total_field[0], expected_field_particle_1), "Total field at particle 1 did not match expected value."
        assert np.allclose(total_field[1], expected_field_particle_2), "Total field at particle 2 did not match expected value."

class Test_MSP_gradient_from_arrays:
    
    dimension = 3
    wave_number = 1.0
    
    def test_zero_green_tensor_derivative(self):
        num_particles = 2
        external_gradient = np.random.rand(num_particles, self.dimension, self.dimension)
        dipole_moments = np.random.rand(num_particles, self.dimension) + 1j * np.random.rand(num_particles, self.dimension)
        zero_green_tensor_derivative = np.zeros((num_particles, num_particles, self.dimension, self.dimension, self.dimension))
        
        gradient = MSP_gradient_from_arrays(dipole_moments, external_gradient, self.wave_number, zero_green_tensor_derivative)
        
        assert np.allclose(gradient, external_gradient), "Gradient should equal external gradient when green tensor derivative is zero."
    
    