import numpy as np
from msptools.OFO_calculations import *
import pytest

np.random.seed(42)

def create_symmetric_matrix(dimensions):
    """
    Create a random symmetric matrix of shape (dimensions, dimensions).
    """
    matrix = np.random.rand(dimensions, dimensions)
    return (matrix + matrix.T) / 2  # Ensure symmetric property


class Test_calculateforces:
    
    medium_permittivity = 2.5

    def test_force_propto_dipole_moment(self):
        """
        Test that the force is proportional to the dipole moment.
        The system consider contains only one particle with a real dipole moment
        and a symmetric field gradient matrix.
        """
        medium_permittivity = self.medium_permittivity
        field_gradient = create_symmetric_matrix(3)
        _, eigenvectors = np.linalg.eigh(field_gradient)
        dipole_moments = np.array([eigenvectors[:, 0]])
        field_gradient = field_gradient.reshape(1, 3, 3)

        forces = calculate_forces_eppgrad(medium_permittivity, dipole_moments, field_gradient)

        cross_product = np.cross(dipole_moments[0], forces[0])
        assert np.allclose(cross_product, 0), "Force is not proportional to the dipole moment"
    
    @pytest.mark.parametrize("num_particles", [1, 2, 3, 5])
    def test_zero_force4imag_gradient(self, num_particles):
        """
        Test that the force is zero when the field gradient is purely imaginary for real dipole moments.
        """
        medium_permittivity = self.medium_permittivity
        dipole_moments = np.random.rand(num_particles, 3)
        field_gradient = np.random.rand(num_particles, 3, 3) * 1j

        forces = calculate_forces_eppgrad(medium_permittivity, dipole_moments, field_gradient)

        assert np.allclose(forces, 0), "Force should be zero for purely imaginary field gradient when dipole moments are real"
    
    
    def test_zero_force_dipolegradient_perpendicular(self):
        """
        Test that the force is zero when the dipole moment and field gradient are perpendicular.
        """
        medium_permittivity = self.medium_permittivity
        num_particles = 3
        dipole_moments = np.array([[1, 0, 0] * num_particles]).reshape(num_particles, 3)
        field_gradient = np.array([[[0, 1, 0], [0, 0, 1], [0, 0, 0]]] * num_particles).reshape(num_particles, 3, 3)

        forces = calculate_forces_eppgrad(medium_permittivity, dipole_moments, field_gradient)

        assert np.allclose(forces, 0), "Force should be zero when dipole moment and field gradient are perpendicular"