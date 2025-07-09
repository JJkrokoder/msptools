import pytest
import numpy as np
from msptools.dipole_moments import calculate_dipole_moments_linear, polarizability_to_matrix


class TestDipoleMomentsLin:

    dimensions = 3  # Assuming 3D space for the dipole moments
    
    @pytest.mark.parametrize("electric_field", [
        np.array([[1, 0, 0], [0, 1+2j, -3 + 5j]]),
    ])
    @pytest.mark.parametrize("polarizability", [
        1, 1 + 0j])  
    def test_unit_dipole_moment(self, electric_field, polarizability):
        dipole_moments = calculate_dipole_moments_linear(polarizability, electric_field)
        assert np.allclose(dipole_moments, electric_field), "Dipole moments should equal electric field for unit polarizability."

    def test_non_supported_polarizability(self):
        with pytest.raises(TypeError):
            calculate_dipole_moments_linear("invalid_type", np.array([[1, 0, 0], [0, 1+2j, -3 + 5j]]))
    
    def test_different_polarizabilities(self):
        electric_field = np.array([[1, 0, 0], [0, 1+2j, -3 + 5j], [0, 0, 1]])
        polarizabilities = [1 + 0j, 2 + 0j, 6j]
        dipole_moments = calculate_dipole_moments_linear(polarizabilities, electric_field)
        
        for i in range(electric_field.shape[0]):
            assert np.allclose(dipole_moments[i, :], polarizabilities[i] * electric_field[i, :]), \
                f"Dipole moment for particle {i} should match polarizability times electric field"

    

class TestPolarizabilityToMatrix:
    
    dimensions = 3 
    
    @pytest.mark.parametrize(["polarizability", "num_particles", "expected_shape"], [
        (1 + 0j, 1, (3, 3)),
        ([1 + 0j, 2 + 0j, 6j], 3, (9, 9)),
        (1 + 2j, 3, (9, 9))])

    
    def test_dimensions(self, polarizability, num_particles, expected_shape):
        result = polarizability_to_matrix(polarizability, num_particles, self.dimensions)
        assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}."
    
    @pytest.mark.parametrize("num_particles", [1, 2, 3, 5])
    def test_different_scalar_polarizabilities(self, num_particles):
        polarizability = np.random.rand(num_particles) + 1j * np.random.rand(num_particles)
        result = polarizability_to_matrix(polarizability, num_particles, self.dimensions)
        assert np.allclose(np.diag(result[:self.dimensions, :self.dimensions]), polarizability[0].repeat(self.dimensions)), "Diagonal elements should match the polarizability values."