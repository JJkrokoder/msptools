import pytest
import numpy as np
from msptools.permittivity import permittivity_Drude, permittivity_ridx

@pytest.mark.parametrize("frequency", [1e2, 1e3, 1e4, 1e5])
def test_high_frequency_Drude_permittivity(frequency):
    epsilon_inf = 1
    plasma_frequency = 1
    collision_frequency = 0.01
    epsilon_inf = permittivity_Drude(frequency=frequency, plasma_frequency=plasma_frequency, collision_frequency=collision_frequency, epsilon_inf=epsilon_inf)
    assert isinstance(epsilon_inf, complex)
    assert np.isclose(abs(epsilon_inf - 1), plasma_frequency ** 2 / np.sqrt(frequency**4 + (frequency * collision_frequency)**2), rtol=1e-12)

class Test_DrudePermittivity_parts():
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

class Test_RidxPermittivity():
    def test_gold_permittivity_at_500nm(self):
        frequency_ev = 2.48  # Corresponds to 500 nm
        material = "Au"
        epsilon = permittivity_ridx(frequency=frequency_ev, material=material)
        expected_epsilon = -2.5676 + 3.6391j  # Known value for gold at 500 nm
        assert np.isclose(epsilon.real, expected_epsilon.real, rtol=2e-1), f"Expected real part {expected_epsilon.real}, got {epsilon.real}"
        assert np.isclose(epsilon.imag, expected_epsilon.imag, rtol=2e-1), f"Expected imaginary part {expected_epsilon.imag}, got {epsilon.imag}"