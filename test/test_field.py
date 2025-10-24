import numpy as np
import msptools as msp

class Test_General_Field():

    def test_initialize_with_frequency_ev(self):
        field = msp.Field(frequency=2.0, frequency_unit="eV")
        assert np.isclose(field.get_frequency(), 2.0), "Field frequency should be initialized to 2.0 eV"
    
    def test_initialize_with_wavelength_cm(self):
        field = msp.Field(wavelength=500.0, wavelength_unit="cm")
        wavelength_nm = 500.0 * 1e7  # Convert cm to nm
        frequency_eV = 1239.84193 / wavelength_nm
        assert np.isclose(field.get_frequency(), frequency_eV), f"Field frequency should be initialized to ~{frequency_eV} eV"
        assert np.isclose(field.get_wavelength(), wavelength_nm), f"Field wavelength should be initialized to {wavelength_nm} nm"

def test_plane_wave_func_in_zero():
    direction = np.array([0, 0, 1])
    amplitude = np.array([1.0, 0.0, 0.0])
    wave_number_nm = 2 * np.pi / 500  # Corresponds to 500 nm wavelength
    positions = np.array([[0.0, 0.0, 0.0]])

    expected_field = amplitude 
    computed_field = msp.plane_wave_function(direction, amplitude, positions, wave_number_nm)

    assert np.allclose(computed_field, expected_field), f"Expected {expected_field}, got {computed_field}"

def test_plane_wave_func_periodicity_x():
    direction = np.array([1, 0, 0])
    amplitude = np.array([1.0, 0.0, 0.0])
    wave_number_nm = 2 * np.pi / 500  # Corresponds to 500 nm wavelength
    positions = np.array([[0.0, 0.0, 0.0],
                          [500.0, 0.0, 0.0],
                          [1000.0, 0.0, 0.0]])

    expected_field = np.array([[1.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0]])
    computed_field = msp.plane_wave_function(direction, amplitude, positions, wave_number_nm)

    assert np.allclose(computed_field, expected_field), f"Expected {expected_field}, got {computed_field}"

def test_plane_wave_func_phase_shift_y():
    direction = np.array([0, 1, 0])
    amplitude = np.array([1.0, 0.0, 0.0])
    wave_number_nm = 2 * np.pi / 500  # Corresponds to 500 nm wavelength
    positions = np.array([[0.0, 0.0, 0.0],
                          [0.0, 125.0, 0.0],
                          [0.0, 250.0, 0.0],
                          [0.0, 375.0, 0.0],
                          [0.0, 500.0, 0.0]])

    expected_field = np.array([[1.0, 0.0, 0.0],
                               [1.0j, 0.0, 0.0],
                               [-1.0, 0.0, 0.0],
                               [-1.0j, 0.0, 0.0],
                               [1.0, 0.0, 0.0]])
    computed_field = msp.plane_wave_function(direction, amplitude, positions, wave_number_nm)

    assert np.allclose(computed_field, expected_field, atol=1e-4), f"Expected {expected_field}, got {computed_field}"

def test_plane_wave_func_periodicity_xy():
    direction = np.array([1, 1, 0])/np.sqrt(2)
    amplitude = np.array([1.0, 0.0, 0.0])
    wave_number_nm = 2 * np.pi / 500  # Corresponds to 500 nm wavelength
    positions = np.array([[0.0, 0.0, 0.0],
                          [250.0, 250.0, 0.0],
                          [500.0, 500.0, 0.0],
                          [750.0, 750.0, 0.0],
                          [1000.0, 1000.0, 0.0]])/ np.sqrt(2)

    expected_field = np.array([[1.0, 0.0, 0.0],
                               [-1.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0],
                               [-1.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0]])
    
    computed_field = msp.plane_wave_function(direction, amplitude, positions, wave_number_nm)

    assert np.allclose(computed_field, expected_field, atol=1e-4), f"Expected {expected_field}, got {computed_field}"