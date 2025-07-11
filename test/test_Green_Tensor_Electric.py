import numpy as np
import pytest
from msptools.GreenTensor_Electric import construct_green_tensor, pair_green_tensor

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



