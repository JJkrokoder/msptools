import numpy as np
import msptools as msp

class Test_Plane_Wave_Field():

    def test_initialize_plane_wave_field(self):
        direction = np.array([0, 1, 1])
        amplitude = 1.0
        polarization = np.array([1.2, 0.0, 0.0])
        wavelength = 500.0  # nm

        field = msp.PlaneWaveField(direction=direction,
                                   amplitude=amplitude,
                                   polarization=polarization,
                                   wavelength_nm=wavelength)
        
        assert np.isclose(field.wavelength_nm, wavelength), f"Field wavelength should be initialized to {wavelength} nm"
        assert np.allclose(field.direction, np.array(direction)/np.linalg.norm(direction)), "Field direction should be normalized"
        expected_amplitude_vec = amplitude * np.array(polarization) / np.linalg.norm(polarization)
        assert np.allclose(field.amplitude * field.polarization, expected_amplitude_vec), "Field amplitude vector should match expected value"

    def test_plane_wave_field_external_function(self):
        direction = np.array([0, 0, 1])
        amplitude = 1.0
        polarization = np.array([1.0, 0.0, 0.0])
        wavelength = 500.0  # nm

        field = msp.PlaneWaveField(direction=direction,
                                   amplitude=amplitude,
                                   polarization=polarization,
                                   wavelength_nm=wavelength)
        
        positions = np.array([[0.0, 0.0, 0.0],
                              [0.0, 0.0, 125.0],
                              [0.0, 0.0, 250.0]])
        
        expected_field = np.array([[1.0, 0.0, 0.0],
                                   [1.0j, 0.0, 0.0],
                                   [-1.0, 0.0, 0.0]])
        
        computed_field = field.get_external_field_in_positions(positions, medium_permittivity=1.0)

        assert np.allclose(computed_field, expected_field, atol=1e-4), f"Expected {expected_field}, got {computed_field}"
    
    def test_plane_wave_field_external_gradient_function_units_and_formula(self):
        direction = np.array([0, 0, 1])
        amplitude = 1.0
        polarization = np.array([1.0, 0.0, 0.0])
        wavelength = 500.0  # nm

        field = msp.PlaneWaveField(direction=direction,
                                   amplitude=amplitude,
                                   polarization=polarization,
                                   wavelength_nm=wavelength)
        
        positions_nm = np.array([[0.0, 0.0, 0.0],
                              [0.0, 0.0, 125.0],
                              [0.0, 0.0, 250.0]])
        
        k_magnitude = 2 * np.pi / wavelength  # in nm^-1
        expected_gradient = 1j * k_magnitude * np.einsum('ij,k -> ijk',
                                                            np.outer(np.exp(1j*positions_nm[:, 2] * k_magnitude), np.array(direction)),
                                                            np.array(polarization))

        computed_gradient = field.get_external_gradient_in_positions(positions_nm, medium_permittivity=1.0)

        assert np.allclose(computed_gradient, expected_gradient, atol=1e-4), f"Expected {expected_gradient}, got {computed_gradient}"

class Test_Standing_Wave_Field():

    def test_initialize_standing_wave_field(self):
        direction = np.array([0, 1, 1])
        amplitude = 1.0
        polarization = np.array([1.2, 0.0, 0.0])
        wavelength = 500.0  # nm

        field = msp.StandingWaveField(direction=direction,
                                      amplitude=amplitude,
                                      polarization=polarization,
                                      wavelength_nm=wavelength)
        
        assert np.isclose(field.wavelength_nm, wavelength), f"Field wavelength should be initialized to {wavelength} nm"
        assert np.allclose(field.direction, np.array(direction)/np.linalg.norm(direction)), "Field direction should be normalized"
        expected_amplitude_vec = amplitude * np.array(polarization) / np.linalg.norm(polarization)
        assert np.allclose(field.amplitude * field.polarization, expected_amplitude_vec), "Field amplitude vector should match expected value"

