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


class Test_Plane_Wave_Field():

    def test_initialize_plane_wave_field(self):
        direction = [0, 1, 1]
        amplitude = 1.0
        polarization = [1.0, 0.0, 0.0]
        frequency = 2.0  # eV

        field = msp.PlaneWaveField(direction=direction,
                                   amplitude=amplitude,
                                   polarization=polarization,
                                   frequency=frequency,
                                   frequency_unit="eV")
        
        assert np.isclose(field.get_frequency(), frequency), "Field frequency should be initialized to 2.0 eV"
        assert np.allclose(field.get_direction(), np.array(direction)/np.linalg.norm(direction)), "Field direction should be normalized"
        expected_amplitude_vec = amplitude * np.array(polarization) / np.linalg.norm(polarization)
        assert np.allclose(field.get_amplitude() * field.get_polarization(), expected_amplitude_vec), "Field amplitude vector should match expected value"
    
    def test_plane_wave_field_external_function(self):
        direction = [0, 0, 1]
        amplitude = 1.0
        polarization = [1.0, 0.0, 0.0]
        wavelength = 500.0  # nm

        field = msp.PlaneWaveField(direction=direction,
                                   amplitude=amplitude,
                                   polarization=polarization,
                                   wavelength=wavelength,
                                   wavelength_unit="nm")
        
        positions = np.array([[0.0, 0.0, 0.0],
                              [0.0, 0.0, 125.0],
                              [0.0, 0.0, 250.0]])
        
        expected_field = np.array([[1.0, 0.0, 0.0],
                                   [1.0j, 0.0, 0.0],
                                   [-1.0, 0.0, 0.0]])
        
        computed_field = field.external_field_function(positions)

        assert np.allclose(computed_field, expected_field, atol=1e-4), f"Expected {expected_field}, got {computed_field}"
    
    def test_plane_wave_field_external_gradient_function_units_and_formula(self):
        direction = [0, 0, 1]
        amplitude = 1.0
        polarization = [1.0, 0.0, 0.0]
        wavelength = 500.0  # nm

        field = msp.PlaneWaveField(direction=direction,
                                   amplitude=amplitude,
                                   polarization=polarization,
                                   wavelength=wavelength,
                                   wavelength_unit="nm")
        
        positions_nm = np.array([[0.0, 0.0, 0.0],
                              [0.0, 0.0, 125.0],
                              [0.0, 0.0, 250.0]])
        
        k_magnitude = 2 * np.pi / wavelength  # in nm^-1
        expected_gradient = 1j * k_magnitude * np.einsum('ij,k -> ijk',
                                                            np.outer(np.exp(1j*positions_nm[:, 2] * k_magnitude), direction),
                                                            np.array(polarization))

        computed_gradient = field.external_gradient_function(positions_nm)

        assert np.allclose(computed_gradient, expected_gradient, atol=1e-4), f"Expected {expected_gradient}, got {computed_gradient}"


