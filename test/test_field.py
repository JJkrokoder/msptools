import numpy as np
import msptools as msp


class Test_General_Field():

    def test_initialize_with_frequency_ev(self):
        field = msp.Field(frequency=2.0, frequency_unit="eV")
        assert np.isclose(field.frequency, 2.0), "Field frequency should be initialized to 2.0 eV"

def test_plane_wave_func_in_zero():
    direction = np.array([0, 0, 1])
    amplitude = np.array([1.0, 0.0, 0.0])
    wave_number_nm = 2 * np.pi / 500  # Corresponds to 500 nm wavelength
    positions = np.array([[0.0, 0.0, 0.0]])

    expected_field = amplitude 
    computed_field = msp.plane_wave_function(direction, amplitude, positions, wave_number_nm)

    assert np.allclose(computed_field, expected_field), f"Expected {expected_field}, got {computed_field}"