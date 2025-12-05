import numpy as np
import msptools as msp

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


class Test_Plane_Wave_Gradient():

    def test_plane_wave_gradient_shape(self):
        direction = np.array([0, 0, 1])
        amplitude = np.array([1.0, 0.0, 0.0])
        wave_number_nm = 2 * np.pi / 500  # Corresponds to 500 nm wavelength
        positions = np.array([[0.0, 0.0, 0.0],
                              [0.0, 0.0, 125.0],
                              [0.0, 0.0, 250.0]])
        
        computed_gradient = msp.plane_wave_gradient(direction, amplitude, positions, wave_number_nm)

        assert computed_gradient.shape == (3, 3, 3), f"Expected gradient shape (3, 3, 3), got {computed_gradient.shape}"
    
    def test_phase_periodicity_for_Ez(self):
        direction = np.array([1, 1, 0])/np.sqrt(2)
        amplitude = np.array([0.0, 0.0, 1.0])
        wave_number_nm = 2 * np.pi / 500  # Corresponds to 500 nm wavelength
        positions = np.array([[0.0, 0.0, 0.0],
                              [250.0, 250.0, 0.0],
                              [500.0, 500.0, 0.0],
                              [750.0, 750.0, 0.0],
                              [1000.0, 1000.0, 0.0]])/ np.sqrt(2)

        expected_grad_Ez = np.array([[1j * wave_number_nm/np.sqrt(2), 1j * wave_number_nm/np.sqrt(2), 0.0],
                                    [-1j * wave_number_nm/np.sqrt(2), -1j * wave_number_nm/np.sqrt(2), 0.0],
                                    [1j * wave_number_nm/np.sqrt(2), 1j * wave_number_nm/np.sqrt(2), 0.0],
                                    [-1j * wave_number_nm/np.sqrt(2), -1j * wave_number_nm/np.sqrt(2), 0.0],
                                    [1j * wave_number_nm/np.sqrt(2), 1j * wave_number_nm/np.sqrt(2), 0.0]])

        computed_gradient = msp.plane_wave_gradient(direction, amplitude, positions, wave_number_nm)
        computed_grad_Ez = computed_gradient[:, :, 2]

        assert np.allclose(computed_grad_Ez, expected_grad_Ez, atol=1e-4), f"Expected {expected_grad_Ez}, got {computed_grad_Ez}"
    
    def test_phase_shift_by_system_translation(self):
        direction = np.array([0, 0, 1])
        amplitude = np.array([1.0, 0.0, 0.0])
        wave_number_nm = 2 * np.pi / 500  # Corresponds to 500 nm wavelength
        positions1 = np.array([[0.0, 0.0, 0.0],
                              [0.0, 0.0, 125.0],
                              [0.0, 0.0, 250.0]])
        
        positions2 = positions1 + np.array([50.0, 50.0, 50.0])

        grad1 = msp.plane_wave_gradient(direction, amplitude, positions1, wave_number_nm)
        grad2 = msp.plane_wave_gradient(direction, amplitude, positions2, wave_number_nm)

        phase_shift = np.exp(1j * wave_number_nm * 50.0)

        assert np.allclose(grad2, grad1 * phase_shift, atol=1e-4), f"Expected phase-shifted gradients."