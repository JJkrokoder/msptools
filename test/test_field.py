import numpy as np
import msptools as msp

class Test_Plane_Wave_Field():

    medium_permittivity = 1.7
    
    def test_initialize_plane_wave_field(self):
        direction = np.array([0, 1, 1])
        amplitude = 1.0
        polarization = np.array([1.2, 0.0, 0.0])
        wavelength = 500.0  # nm

        field = msp.PlaneWaveField(direction=direction,
                                   amplitude=amplitude,
                                   polarization=polarization,
                                   vacuum_wavelength_nm=wavelength,
                                    medium_permittivity=self.medium_permittivity)
        
        medium_wl = wavelength / np.sqrt(self.medium_permittivity)
        
        assert np.isclose(field.monochromatic_data.vacuum_wavelength_nm, wavelength), f"Field wavelength should be initialized to {wavelength} nm"
        assert np.isclose(field.monochromatic_data.medium_wavelength_nm, medium_wl), f"Field medium wavelength should be initialized to {medium_wl} nm"
        assert np.allclose(field.direction, np.array(direction)/np.linalg.norm(direction)), "Field direction should be normalized"
        expected_amplitude_vec = amplitude * np.array(polarization) / np.linalg.norm(polarization)
        assert np.allclose(field.amplitude * field.polarization, expected_amplitude_vec), "Field amplitude vector should match expected value"

    def test_plane_wave_field_external_function(self):
        direction = np.array([0, 0, 1])
        amplitude = 1.0
        polarization = np.array([1.0, 0.0, 0.0])
        wavelength = 500.0  # nm
        medium_permittivity = 1.0

        field = msp.PlaneWaveField(direction=direction,
                                   amplitude=amplitude,
                                   polarization=polarization,
                                   vacuum_wavelength_nm=wavelength,
                                   medium_permittivity=medium_permittivity)
        
        positions = np.array([[0.0, 0.0, 0.0],
                              [0.0, 0.0, 125.0],
                              [0.0, 0.0, 250.0]])
        
        expected_field = np.array([[1.0, 0.0, 0.0],
                                   [1.0j, 0.0, 0.0],
                                   [-1.0, 0.0, 0.0]])
        
        computed_field = field.evaluate(positions)

        assert np.allclose(computed_field, expected_field, atol=1e-4), f"Expected {expected_field}, got {computed_field}"
    
    def test_plane_wave_field_external_gradient_function_units_and_formula(self):
        direction = np.array([0, 0, 1])
        amplitude = 1.0
        polarization = np.array([1.0, 0.0, 0.0])
        wavelength = 500.0  # nm
        medium_wl = wavelength / np.sqrt(self.medium_permittivity)

        field = msp.PlaneWaveField(direction=direction,
                                   amplitude=amplitude,
                                   polarization=polarization,
                                   vacuum_wavelength_nm=wavelength,
                                   medium_permittivity=self.medium_permittivity)
        
        positions_nm = np.array([[0.0, 0.0, 0.0],
                              [0.0, 0.0, 125.0],
                              [0.0, 0.0, 250.0]])
        
        k_magnitude = 2 * np.pi / medium_wl  # in nm^-1
        expected_gradient = 1j * k_magnitude * np.einsum('ij,k -> ijk',
                                                            np.outer(np.exp(1j*positions_nm[:, 2] * k_magnitude), np.array(direction)),
                                                            np.array(polarization))

        computed_gradient = field.evaluate_gradient(positions_nm)

        assert np.allclose(computed_gradient, expected_gradient, atol=1e-4), f"Expected {expected_gradient}, got {computed_gradient}"
        
    def test_plane_wave_field_complex_fieldgradient(self):
        direction = np.array([0, 0, 1])
        amplitude = 1.0
        polarization = np.array([1.0, 0.0, 0.0])
        wavelength = 500.0  # nm
        medium_wl = wavelength / np.sqrt(self.medium_permittivity)

        field = msp.PlaneWaveField(direction=direction,
                                   amplitude=amplitude,
                                   polarization=polarization,
                                   vacuum_wavelength_nm=wavelength,
                                   medium_permittivity=self.medium_permittivity)
        
        positions_nm = np.array([[0.0, 0.0, 0.0],
                              [0.0, 0.0, 50.0],
                              [0.0, 0.0, 100.0]])
        
        computed_FieldGradient = field.eval_complex_field_grad(positions_nm)
        
        k_vec = 2 * np.pi / medium_wl * direction
        n_positions = positions_nm.shape[0]
        intensity = np.abs(amplitude)**2
        expected_FieldGradient = np.tile(-1j * k_vec, (n_positions, 1)) * intensity
        print("Computed Field Gradient:", computed_FieldGradient)
        print("Expected Field Gradient:", expected_FieldGradient)
        
        assert np.allclose(computed_FieldGradient, expected_FieldGradient), f"Expected {expected_FieldGradient}, got {computed_FieldGradient}"
        
class Test_Standing_Wave_Field():

    medium_permittivity = 1.7
    
    def test_initialize_standing_wave_field(self):
        direction = np.array([0, 1, 1])
        amplitude = 1.5
        polarization = np.array([1.2, 0.0, 0.0])
        wavelength = 500.0  # nm
        medium_wl = wavelength / np.sqrt(self.medium_permittivity)
        
        field = msp.StandingWaveField(direction=direction,
                                      amplitude=amplitude,
                                      polarization=polarization,
                                      vacuum_wavelength_nm=wavelength,
                                      medium_permittivity=self.medium_permittivity)
        
        assert np.isclose(field.monochromatic_data.vacuum_wavelength_nm, wavelength), f"Field wavelength should be initialized to {wavelength} nm"
        assert np.allclose(field.direction, np.array(direction)/np.linalg.norm(direction)), "Field direction should be normalized"
        expected_amplitude_vec = amplitude * np.array(polarization) / np.linalg.norm(polarization)
        assert np.allclose(field.amplitude * field.polarization, expected_amplitude_vec), "Field amplitude vector should match expected value"
    
    def test_standing_wave_field_external_function(self):
        direction = np.array([0, 0, 1])
        amplitude = 1.0
        polarization = np.array([1.0, 0.0, 0.0])
        wavelength = 500.0  # nm
        medium_wl = wavelength / np.sqrt(self.medium_permittivity)

        field = msp.StandingWaveField(direction=direction,
                                      amplitude=amplitude,
                                      polarization=polarization,
                                      vacuum_wavelength_nm=wavelength,
                                      medium_permittivity=self.medium_permittivity)
        
        positions = np.array([[0.0, 0.0, 0.0],
                              [0.0, 0.0, 50.0],
                              [0.0, 0.0, 100.0]])
        
        k_vec = 2 * np.pi / medium_wl * direction
        
        phase_factor = np.cos(np.dot(positions, k_vec))
        expected_field = amplitude * np.outer(phase_factor, polarization)
        
        computed_field = field.evaluate(positions)

        assert np.allclose(computed_field, expected_field, atol=1e-4), f"Expected {expected_field}, got {computed_field}"

    def test_standing_wave_field_gradient(self):
        direction = np.array([0, 0, 1])
        amplitude = 1.0
        polarization = np.array([1.0, 0.0, 0.0])
        wavelength = 500.0  # nm
        medium_wl = wavelength / np.sqrt(self.medium_permittivity)

        field = msp.StandingWaveField(direction=direction,
                                      amplitude=amplitude,
                                      polarization=polarization,
                                      vacuum_wavelength_nm=wavelength,
                                      medium_permittivity=self.medium_permittivity)
        
        positions_nm = np.array([[0.0, 0.0, 0.0],
                              [0.0, 0.0, 50.0],
                              [0.0, 0.0, 100.0]])
        
        k_vec = 2 * np.pi / medium_wl * direction
        phase_factor = -np.sin(np.dot(positions_nm, k_vec))
        expected_gradient = amplitude * np.einsum('ij,k -> ikj', np.outer(phase_factor, polarization), k_vec)
        computed_gradient = field.evaluate_gradient(positions_nm)

        assert np.allclose(computed_gradient, expected_gradient, atol=1e-4), f"Expected {expected_gradient}, got {computed_gradient}"
    
    def test_standing_wave_field_complex_fieldgradient(self):
        direction = np.array([0, 0, 1])
        amplitude = 1.0
        polarization = np.array([1.0, 0.0, 0.0])
        wavelength = 500.0  # nm
        medium_wl = wavelength / np.sqrt(self.medium_permittivity)

        field = msp.StandingWaveField(direction=direction,
                                      amplitude=amplitude,
                                      polarization=polarization,
                                      vacuum_wavelength_nm=wavelength,
                                      medium_permittivity=self.medium_permittivity)
        
        positions_nm = np.array([[0.0, 0.0, 0.0],
                              [0.0, 0.0, 50.0],
                              [0.0, 0.0, 100.0]])
        
        computed_FieldGradient = field.eval_complex_field_grad(positions_nm)
        
        k_vec = 2 * np.pi / medium_wl * direction
        intensity = np.abs(amplitude)**2
        phase_factor = np.sin(np.dot(positions_nm, -2*k_vec))
        expected_FieldGradient =0.5*intensity * np.outer(phase_factor, k_vec)
        
        print("Computed Field Gradient:", computed_FieldGradient)
        print("Expected Field Gradient:", expected_FieldGradient)
        
        assert np.allclose(computed_FieldGradient, expected_FieldGradient), f"Expected {expected_FieldGradient}, got {computed_FieldGradient}"
    

class Test_Sum_Field():

    medium_permittivity = 1.7
    wavelength = 500.0  # nm
    medium_wl = wavelength / np.sqrt(medium_permittivity)
    
    field1 = msp.PlaneWaveField(direction=np.array([0, 0, 1]),
                                amplitude=1.0,
                                polarization=np.array([1.0, 0.0, 0.0]),
                                vacuum_wavelength_nm=wavelength,
                                medium_permittivity=medium_permittivity)
    
    field2 = msp.StandingWaveField(direction=np.array([0, 1, 0]),
                                amplitude=0.5,
                                polarization=np.array([0.0, 1.0, 0.0]),
                                vacuum_wavelength_nm=600.0,
                                medium_permittivity=medium_permittivity)
    
    def test_sum_field_initialization(self):
        
        sum_field = self.field1 + self.field2
         
        assert isinstance(sum_field, msp.SumField), "The sum of two fields should be an instance of SumField"
        assert len(sum_field.fields) == 2, "SumField should contain both fields"
        assert sum_field.fields[0] == self.field1, "First field in SumField should be field1"
        assert sum_field.fields[1] == self.field2, "Second field in SumField should be field2"
        
    def test_sum_field_evaluation(self):
        
        sum_field = self.field1 + self.field2
        
        position = np.array([[100.0, 0.0, 0.0]])
        field_1_eval = self.field1.evaluate(position)
        field_2_eval = self.field2.evaluate(position)
        expected_sum = field_1_eval + field_2_eval
        computed_sum = sum_field.evaluate(position)
        assert np.allclose(computed_sum, expected_sum, atol=1e-4), f"Expected {expected_sum}, got {computed_sum}"
    
    def test_associative_sum_field(self):
        
        field3 = msp.PlaneWaveField(direction=np.array([1, 0, 0]),
                                amplitude=0.3,
                                polarization=np.array([0.0, 0.0, 1.0]),
                                vacuum_wavelength_nm=700.0,
                                medium_permittivity=self.medium_permittivity)
        
        sum_field_1 = self.field1 + (self.field2 + field3)
        sum_field_2 = (self.field1 + self.field2) + field3

        field_tuple = (self.field1, self.field2, field3) 
        
        assert sum_field_1.simplify() == sum_field_2.simplify(), "Sum of fields should be associative after simplification"
        assert sum_field_1.simplify().fields == field_tuple, "Simplified sum field should contain all original fields in order"
        assert sum_field_2.simplify().fields == field_tuple, "Simplified sum field should contain all original fields in order"
    
class Test_Scaled_Field():
    
    medium_permittivity = 1.7
    wavelength = 500.0  # nm
    medium_wl = wavelength / np.sqrt(medium_permittivity)
    scalar = 2.5
    
    field = msp.PlaneWaveField(direction=np.array([0, 0, 1]),
                                amplitude=1.0,
                                polarization=np.array([1.0, 0.0, 0.0]),
                                vacuum_wavelength_nm=wavelength,
                                medium_permittivity=medium_permittivity)
    
    def test_scaled_field_simplify(self):
        
        scaled_field = self.scalar * self.field
        
        same_field = self.field * 1
        
        doubly_multiplied_field = self.scalar * (self.field * 1)
        
        assert isinstance(scaled_field, msp.PlaneWaveField), "The product of a scalar and a PW field should be an instance of PlaneWaveField"
        assert same_field == self.field, "Multiplying by 1 should return the original field"
        assert doubly_multiplied_field.amplitude == scaled_field.amplitude, "Multiplying by 1 should not change the scaled field"
        
    def test_scaled_field_evaluation(self):
        
        scaled_field = self.scalar * self.field
        
        position = np.array([[100.0, 0.0, 0.0]])
        original_eval = self.field.evaluate(position)
        expected_scaled_eval = self.scalar * original_eval
        computed_scaled_eval = scaled_field.evaluate(position)
        assert np.allclose(computed_scaled_eval, expected_scaled_eval, atol=1e-4), f"Expected {expected_scaled_eval}, got {computed_scaled_eval}"    

class Test_Traslated_Field():
    
    medium_permittivity = 1.2
    wavelength = 500.0  # nm
    medium_wl = wavelength / np.sqrt(medium_permittivity)
    displacement = np.array([10.0, 20.0, 30.0])
    
    PW_field = msp.PlaneWaveField(direction=np.array([0, 0, 1]),
                                amplitude=1.0,
                                polarization=np.array([1.0, 0.0, 0.0]),
                                vacuum_wavelength_nm=wavelength,
                                medium_permittivity=medium_permittivity)

    def test_translated_field_evaluation(self):
        
        translated_field = self.PW_field.translate(self.displacement)
        
        position = np.array([[100.0, 100.0, 100.0]])
        original_eval = self.PW_field.evaluate(position - self.displacement)
        expected_translated_eval = original_eval
        computed_translated_eval = translated_field.evaluate(position)
        assert np.allclose(computed_translated_eval, expected_translated_eval, atol=1e-4), f"Expected {expected_translated_eval}, got {computed_translated_eval}"

def test_distributive_property_of_sum_and_scalar_multiplication():
    
    scalar = 2.5
    medium_permittivity = 1.7
    wavelength = 500.0  # nm
    medium_wl = wavelength / np.sqrt(medium_permittivity)
    
    field1 = msp.PlaneWaveField(direction=np.array([0, 0, 1]),
                                amplitude=1.0,
                                polarization=np.array([1.0, 0.0, 0.0]),
                                vacuum_wavelength_nm=wavelength,
                                medium_permittivity=medium_permittivity)
    
    field2 = msp.StandingWaveField(direction=np.array([0, 1, 0]),
                                amplitude=0.5,
                                polarization=np.array([0.0, 1.0, 0.0]),
                                vacuum_wavelength_nm=600.0,
                                medium_permittivity=medium_permittivity)
    
    sum_field = field1 + field2
    scaled_sum_field = scalar * sum_field
    expected_scaled_sum = scalar * field1 + scalar * field2
    computed_scaled_sum = scaled_sum_field.evaluate(np.array([[100.0, 100.0, 100.0]]))
    expected_scaled_sum_eval = expected_scaled_sum.evaluate(np.array([[100.0, 100.0, 100.0]]))
    
    assert np.allclose(computed_scaled_sum, expected_scaled_sum_eval, atol=1e-4), f"Expected {expected_scaled_sum_eval}, got {computed_scaled_sum}"

class Test_Curl():
    
    def test_curl_of_plane_wave_field(self):
        direction = np.array([0, 0, 1])
        amplitude = 1.0
        polarization = np.array([1.0, 0.0, 0.0])
        wavelength = 500.0  # nm
        medium_permittivity = 1.0

        field = msp.PlaneWaveField(direction=direction,
                                   amplitude=amplitude,
                                   polarization=polarization,
                                   vacuum_wavelength_nm=wavelength,
                                   medium_permittivity=medium_permittivity)
        
        positions = np.array([[0.0, 0.0, 0.0],
                              [0.0, 0.0, 50.0],
                              [0.0, 0.0, 100.0]])
        
        computed_curl = field.eval_curl(positions)
        
        k_vec = 2 * np.pi / wavelength * direction
        expected_curl = np.cross(1j*k_vec, field.evaluate(positions))
        
        assert np.allclose(computed_curl, expected_curl, atol=1e-4), f"Expected {expected_curl}, got {computed_curl}"
    
