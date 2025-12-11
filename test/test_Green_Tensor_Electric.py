import numpy as np
import pytest
from msptools.GreenTensor_Electric import *

class Test_ConstructGreenTensor:

    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    wave_number = 1.0
    num_particles = positions.shape[0]
    dimensions = positions.shape[1]
    green_tensor = construct_green_tensor(positions, wave_number)

    def test_shapeandsymmetry(self): 
        assert self.green_tensor.shape == (self.num_particles, self.num_particles, self.dimensions, self.dimensions), "Green tensor shape mismatch."
        assert np.allclose(self.green_tensor, self.green_tensor.transpose(1, 0, 2, 3)), "Green tensor is not symmetric."
        assert np.allclose(self.green_tensor, self.green_tensor.transpose(0, 1, 3, 2)), "Green tensor is not symmetric in coordinates."

    def test_zero_diagonal_blocks(self):
        for i in range(self.num_particles):
            assert np.allclose(self.green_tensor[i, i], np.zeros((self.dimensions, self.dimensions))), f"Diagonal block {i} is not a zero matrix."

class Test_G_funtions:
    
    wave_number = 2.0

    def test_G_0_farfield(self):
        distance = 1e5 / self.wave_number
        g0 = G_0_function(distance, self.wave_number)
        expected_g0 = np.exp(1j * self.wave_number * distance) / (4 * np.pi * distance)
        assert np.allclose(g0, expected_g0, rtol=1e-5), "G_0 function does not match far-field approximation."
    
    def test_G_1_farfield(self):
        distance = 1e5 / self.wave_number
        g1 = G_1_function(distance, self.wave_number)
        expected_g1 = -np.exp(1j * self.wave_number * distance) / (4 * np.pi * distance**3)
        assert np.allclose(g1, expected_g1, rtol=1e-5), "G_1 function does not match far-field approximation."

    def test_G_0_nearfield(self):
        distance = 1e-5 / self.wave_number
        g0 = G_0_function(distance, self.wave_number)
        expected_g0 = -1 / (4 * np.pi * self.wave_number**2 * distance**3)
        assert np.allclose(g0, expected_g0, rtol=1e-5), "G_0 function does not match near-field approximation."
    
    def test_G_1_nearfield(self):
        distance = 1e-5 / self.wave_number
        g1 = G_1_function(distance, self.wave_number)
        expected_g1 = 3 / (4 * np.pi * self.wave_number**2 * distance**5)
        assert np.allclose(g1, expected_g1, rtol=1e-5), "G_1 function does not match near-field approximation."
    
    def test_G_0_derivative_farfield(self):
        distance = 1e5 / self.wave_number
        der_g0 = G_0_derivative_function(distance, self.wave_number)
        expected_der_g0 = 1j * self.wave_number * np.exp(1j * self.wave_number * distance) / (4 * np.pi * distance)
        assert np.allclose(der_g0, expected_der_g0, rtol=1e-5), "Derivative of G_0 function does not match far-field approximation."
    
    def test_G_0_derivative_nearfield(self):
        distance = 1e-5 / self.wave_number
        der_g0 = G_0_derivative_function(distance, self.wave_number)
        expected_der_g0 = 3 / (4 * np.pi * self.wave_number**2 * distance**4)
        assert np.allclose(der_g0, expected_der_g0, rtol=1e-5), "Derivative of G_0 function does not match near-field approximation."
    
    def test_G_1_derivative_farfield(self):
        distance = 1e5 / self.wave_number
        der_g1 = G_1_derivative_function(distance, self.wave_number)
        expected_der_g1 = -self.wave_number * np.exp(1j * self.wave_number * distance) / (4 * np.pi * distance**3) * (1j)
        assert np.allclose(der_g1, expected_der_g1, rtol=1e-5), "Derivative of G_1 function does not match far-field approximation."
    
    def test_G_1_derivative_nearfield(self):
        distance = 1e-5 / self.wave_number
        der_g1 = G_1_derivative_function(distance, self.wave_number)
        expected_der_g1 = -15 / (4 * np.pi * self.wave_number**2 * distance**6)
        assert np.allclose(der_g1, expected_der_g1, rtol=1e-5), "Derivative of G_1 function does not match near-field approximation."
    

class Test_PairGreenTensor:

    wave_number = 2.0
    
    def test_symmetry(self):
        pos_i = np.array([0, 0, 0])
        pos_j = np.array([1.5, 0, 0])

        g_ij = pair_green_tensor(pos_i, pos_j, self.wave_number)

        assert np.allclose(g_ij, g_ij.T), "Pair Green's tensor is not symmetric."
        assert np.allclose(g_ij, pair_green_tensor(pos_j, pos_i, self.wave_number)), "Pair Green's tensor is not equal when positions are swapped."
    
    @pytest.mark.parametrize("translation_vector", [
        np.array([1, 0, 0]),
        np.array([1, 1.5, 1])
    ])
    def test_translational_invariance(self, translation_vector):
        pos_i = np.array([0, 0, 0])
        pos_j = np.array([1.5, 0, 0])

        g_ij = pair_green_tensor(pos_i, pos_j, self.wave_number)
        translated_g_ij = pair_green_tensor(pos_i + translation_vector, pos_j + translation_vector, self.wave_number)

        assert np.allclose(g_ij, translated_g_ij), "Pair Green's tensor is not translationally invariant."
    
    @pytest.mark.parametrize("angle", [0, np.pi/4, np.pi/2])
    def test_rotational_invariance(self, angle):
        pos_i = np.array([1, 1, 1])
        pos_j = np.array([1.5, 0, 0])

        g_ij = pair_green_tensor(pos_i, pos_j, self.wave_number)

        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0],
                             [0, 0, 1]])


        rotated_g_ij = pair_green_tensor(rotation @ pos_i, rotation @ pos_j, self.wave_number)

        assert np.allclose(rotation @ g_ij @ rotation.T, rotated_g_ij), "Pair Green's tensor is not rotationally invariant."


    @pytest.mark.parametrize("norm_distance", [5, 1e2, 1e5])
    def test_farfield_consistency(self, norm_distance):

        distance = norm_distance / self.wave_number
        direction_vector = np.array([1, 1, 0.5])
        direction_vector /= np.linalg.norm(direction_vector)

        pos_i = np.array([0, 0, 0])
        pos_j = distance * direction_vector
        R_vec = pos_i - pos_j

        g_ij = pair_green_tensor(pos_i, pos_j, self.wave_number)
        
        g0_ff = np.exp(1j * norm_distance) / (4 * np.pi * distance)
        g1_ff = -np.exp(1j * norm_distance) / (4 * np.pi * distance) / distance**2
        g_ff = g0_ff * np.eye(3) + g1_ff * (R_vec[:, None] @ R_vec[None, :]) 

        assert np.allclose(g_ij, g_ff, atol=2/(4*np.pi * distance * norm_distance)), "Pair Green's tensor does not match far-field approximation."
    
    @pytest.mark.parametrize("unit_vector", [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1])])
    def test_nearfield_consistency(self, unit_vector):
        distance = 1e-5 / self.wave_number
        pos_i = np.array([0, 0, 0])
        pos_j = distance * unit_vector

        g_ij = pair_green_tensor(pos_i, pos_j, self.wave_number)

        g0_nf = -1 / (4 * np.pi * self.wave_number**2 * distance**3)
        g1_nf = 3 / (4 * np.pi * self.wave_number**2 * distance**5)
        R_vec = pos_i - pos_j
        g_nf = g0_nf * np.eye(3) + g1_nf * (R_vec[:, None] @ R_vec[None, :])

        assert np.allclose(g_ij, g_nf, rtol=1e-5), "Pair Green's tensor does not match near-field approximation."


class Test_Pair_GreenTensor_Derivative:

    wave_number = 2.0

    def test_antisymmetry(self):
        pos_i = np.array([0, 0, 0])
        pos_j = np.array([1.5, 0, 0])

        for coord in range(3):
            der_g_ij = pair_green_tensor_derivative(pos_i, pos_j, coord, self.wave_number)
            der_g_ji = pair_green_tensor_derivative(pos_j, pos_i, coord, self.wave_number)

            assert np.allclose(der_g_ij, -der_g_ji), f"Derivative of pair Green's tensor is not antisymmetric for coordinate {coord}."
    
    @pytest.mark.parametrize("coordinate", [0, 1, 2])
    def test_numerical_derivative(self, coordinate):
        pos_i = np.array([0.5, 0.5, 0.5])
        pos_j = np.array([1.5, 0, 0])
        h = 1e-8

        der_g_analytical = pair_green_tensor_derivative(pos_i, pos_j, coordinate, self.wave_number)

        pos_i_plus = pos_i.copy()
        pos_i_plus[coordinate] += h
        g_plus = pair_green_tensor(pos_i_plus, pos_j, self.wave_number)

        pos_i_minus = pos_i.copy()
        pos_i_minus[coordinate] -= h
        g_minus = pair_green_tensor(pos_i_minus, pos_j, self.wave_number)

        der_g_numerical = (g_plus - g_minus) / (2 * h)

        print(f"Analytical derivative (coord {coordinate}):\n", der_g_analytical)
        print(f"Numerical derivative (coord {coordinate}):\n", der_g_numerical)

        assert np.allclose(der_g_analytical, der_g_numerical, atol=1e-5), f"Analytical and numerical derivatives do not match for coordinate {coordinate}."


class Test_ConstructGreenTensorGradient:
    
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    wave_number = 1.0
    num_particles = positions.shape[0]
    dimensions = positions.shape[1]
    green_tensor_gradient = construct_green_tensor_gradient(positions, wave_number)

    def test_shape(self):
        expected_shape = (self.num_particles, self.num_particles, self.dimensions, self.dimensions, self.dimensions)
        assert self.green_tensor_gradient.shape == expected_shape, "Green tensor gradient shape mismatch."

    def test_antisymmetry(self):
        anti_transpose = -self.green_tensor_gradient.transpose(1, 0, 2, 3, 4)
        assert np.allclose(self.green_tensor_gradient, anti_transpose), "Green tensor gradient is not antisymmetric with respect to particle indices."

