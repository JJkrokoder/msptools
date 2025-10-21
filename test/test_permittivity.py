import pytest
import numpy as np
from msptools.permittivity import permittivity_Drude

@pytest.mark.parametrize("frequency", [1e2, 1e3, 1e4, 1e5])
def test_high_frequency_Drude_permittivity(frequency):
    epsilon_inf = 1
    plasma_frequency = 1
    collision_frequency = 0.01
    epsilon_inf = permittivity_Drude(frequency=frequency, plasma_frequency=plasma_frequency, collision_frequency=collision_frequency, epsilon_inf=epsilon_inf)
    assert isinstance(epsilon_inf, complex)
    assert np.isclose(abs(epsilon_inf - 1), plasma_frequency ** 2 / np.sqrt(frequency**4 + (frequency * collision_frequency)**2), rtol=1e-12)

class TestDrudePermittivity_parts():
    frequency = 1
    plasma_frequency = 1
    collision_frequency = 0.1
    epsilon_inf = 5
    epsilon = permittivity_Drude(frequency=frequency, plasma_frequency=plasma_frequency, collision_frequency=collision_frequency, epsilon_inf=epsilon_inf)

    def test_real_part_Drude_permittivity(self):
        expected_real = self.epsilon_inf - self.plasma_frequency**2 / (self.frequency**2 + self.collision_frequency**2)
        assert np.isclose(self.epsilon.real, expected_real, rtol=1e-12), f"Expected real part {expected_real:4f}, got {self.epsilon.real:4f}"

    def test_imaginary_part_Drude_permittivity(self):
        expected_imag = (self.plasma_frequency**2 * self.collision_frequency) / (self.frequency**3 + self.frequency * (self.collision_frequency)**2)
        assert np.isclose(self.epsilon.imag, expected_imag, rtol=1e-12), f"Expected imaginary part {expected_imag:4f}, got {self.epsilon.imag:4f}"
        