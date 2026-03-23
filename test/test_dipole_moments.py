import pytest
import numpy as np
from msptools.dipole_moments import calculate_dipole_moments_linear


class TestDipoleMomentsLin:

    dimensions = 3  # Assuming 3D space for the dipole moments
    
    @pytest.mark.parametrize("electric_field", [
        np.array([[1, 0, 0], [0, 1+2j, -3 + 5j]]),
    ])
    @pytest.mark.parametrize("polarizability", [
        np.repeat(np.eye(3)[None,:,:], 2, axis=0),
        np.repeat((1 + 0j)*np.eye(3)[None,:,:], 2, axis=0)
    ])
    def test_identity_dipole_moment(self, electric_field, polarizability):
        dipole_moments = calculate_dipole_moments_linear(polarizability, electric_field)
        assert np.allclose(dipole_moments, electric_field), "Dipole moments should equal electric field for unit polarizability."

    def test_different_polarizabilities(self):
        electric_field = np.array([[1, 0, 0], [0, 1+2j, -3 + 5j], [0, 0, 1]])
        polarizabilities = np.array([(1 + 0j)*np.eye(3), (2 + 0j)*np.eye(3), (6j)*np.eye(3)])
        dipole_moments = calculate_dipole_moments_linear(polarizabilities, electric_field)
        
        assert np.allclose(dipole_moments, np.einsum('ikl,il->ik', polarizabilities, electric_field)), "Dipole moments should be the product of polarizability and electric field for each particle."
