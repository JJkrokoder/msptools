import numpy as np
import pytest
from msptools.GreenTensor_Electric import construct_green_tensor, pair_green_tensor

class Test_ConstructGreenTensor:

    def test_shapeandsymmetry(self):
        num_particles = 3
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        wave_number = 1.0
        dimensions = 3
        
        green_tensor = construct_green_tensor(positions, wave_number)
        
        assert green_tensor.shape == (num_particles, num_particles, dimensions, dimensions), "Green tensor shape mismatch."
        assert np.allclose(green_tensor, green_tensor.transpose(1, 0, 2, 3)), "Green tensor is not symmetric."


class Test_PairGreenTensor:

    def test_symmetry(self):
        pos_i = np.array([0, 0, 0])
        pos_j = np.array([1.5, 0, 0])
        wave_number = 2.0
        
        g_ij = pair_green_tensor(pos_i, pos_j, wave_number)
        
        assert np.allclose(g_ij, g_ij.T), "Pair Green's tensor is not symmetric."
        assert np.allclose(g_ij, pair_green_tensor(pos_j, pos_i, wave_number)), "Pair Green's tensor is not equal when positions are swapped."
    

        



