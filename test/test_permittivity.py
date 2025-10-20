import pytest
import numpy as np

@pytest.mark.parametrize("frequency", [1e2, 1e3, 1e4, 1e5])
def test_high_frequency_Drude_permittivity(frequency):
    from msptools.permittivity import permittivity_Drude
    epsilon_inf = 1
    plasma_frequency = 1
    collision_frequency = 0.01
    epsilon_inf = permittivity_Drude(frequency=frequency, plasma_frequency=plasma_frequency, collision_frequency=collision_frequency, epsilon_inf=epsilon_inf)
    assert isinstance(epsilon_inf, complex)
    assert np.isclose(abs(epsilon_inf - 1), plasma_frequency ** 2 / np.sqrt(frequency**4 + (frequency * collision_frequency)**2), rtol=1e-2)
