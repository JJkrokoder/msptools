import pytest
import numpy as np
from msptools.dipole_moments import calculate_dipole_moments_linear


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
    
    


