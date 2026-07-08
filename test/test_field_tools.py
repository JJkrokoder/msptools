try:
    import cupy as np
except ImportError:
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

        expected_grad_Ez = np.array([[1j * wave_number_nm/2**0.5, 1j * wave_number_nm/2**0.5, 0.0],
                                    [-1j * wave_number_nm/2**0.5, -1j * wave_number_nm/2**0.5, 0.0],
                                    [1j * wave_number_nm/2**0.5, 1j * wave_number_nm/2**0.5, 0.0],
                                    [-1j * wave_number_nm/2**0.5, -1j * wave_number_nm/2**0.5, 0.0],
                                    [1j * wave_number_nm/2**0.5, 1j * wave_number_nm/2**0.5, 0.0]])

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

class Test_Plane_Wave_Double_Gradient():
    
    def test_plane_wave_double_gradient_shape(self):
        direction = np.array([0, 0, 1])
        amplitude = np.array([1.0, 0.0, 0.0])
        wave_number_nm = 2 * np.pi / 500  # Corresponds to 500 nm wavelength
        positions = np.array([[0.0, 0.0, 0.0],
                              [0.0, 0.0, 250.0]])
        
        computed_double_gradient = msp.plane_wave_double_gradient(direction, amplitude, positions, wave_number_nm)

        assert computed_double_gradient.shape == (2, 3, 3, 3), f"Expected double gradient shape (2, 3, 3, 3), got {computed_double_gradient.shape}"
    
    def test_plane_Wave_double_gradient_analytical(self):
        direction = np.array([0, 0, 1])
        amplitude = np.array([1.0, 0.0, 0.0])
        wave_number_nm = 2 * np.pi / 500  # Corresponds to 500 nm wavelength
        positions = np.array([[0.0, 0.0, 0.0],
                              [0.0, 0.0, 125.0],
                              [0.0, 0.0, 250.0]])
        
        k_vec = wave_number_nm * direction
        computed_double_gradient = msp.plane_wave_double_gradient(direction, amplitude, positions, wave_number_nm)
        
        expected_double_gradient = -np.einsum('i,jk,l->ijkl', np.exp(1j * np.dot(positions, k_vec)), np.outer(k_vec, k_vec), amplitude)

        assert np.allclose(computed_double_gradient, expected_double_gradient), f"Expected {expected_double_gradient}, got {computed_double_gradient}"

class Test_Plane_Wave_Triple_Gradient():
    
    def test_plane_wave_triple_gradient_shape(self):
        direction = np.array([0, 0, 1])
        amplitude = np.array([1.0, 0.0, 0.0])
        wave_number_nm = 2 * np.pi / 500  # Corresponds to 500 nm wavelength
        positions = np.array([[0.0, 0.0, 0.0],
                              [0.0, 0.0, 250.0]])
        
        computed_triple_gradient = msp.plane_wave_triple_gradient(direction, amplitude, positions, wave_number_nm)

        assert computed_triple_gradient.shape == (2, 3, 3, 3, 3), f"Expected triple gradient shape (2, 3, 3, 3, 3), got {computed_triple_gradient.shape}"
    
    def test_plane_wave_triple_gradient_analytical(self):
        direction = np.array([0, 0, 1])
        amplitude = np.array([1.0, 0.0, 0.0])
        wave_number_nm = 2 * np.pi / 500  # Corresponds to 500 nm wavelength
        positions = np.array([[0.0, 0.0, 0.0],
                              [0.0, 0.0, 50.0],
                              [0.0, 0.0, 100.0]])
        
        k_vec = wave_number_nm * direction
        computed_triple_gradient = msp.plane_wave_triple_gradient(direction, amplitude, positions, wave_number_nm)
        
        expected_triple_gradient = -1j * np.einsum('i,jkl,m->ijklm', np.exp(1j * np.dot(positions, k_vec)), 
                                                   np.einsum('j,kl->jkl', k_vec, np.outer(k_vec, k_vec)), 
                                                   amplitude)

        assert np.allclose(computed_triple_gradient, expected_triple_gradient), f"Expected {expected_triple_gradient}, got {computed_triple_gradient}"
     
class Test_Standing_Wave_Function():

    def test_standing_wave_function_analytical(self):
        direction = np.array([0, 0, 1])
        amplitude = np.array([1.0, 0.0, 0.0])
        wave_number_nm = 2 * np.pi / 500  # Corresponds to 500 nm wavelength in vacuum
        positions = np.array([[0.0, 0.0, 50.0],
                              [0.0, 0.0, 100.0],
                              [0.0, 0.0, 150.0]])
        
        k_vec = wave_number_nm * direction
        computed_field = msp.standing_wave_function(direction, amplitude, positions, wave_number_nm)
        
        expected_field = np.einsum('i,j->ji', amplitude, np.cos(np.einsum('ij,j->i', positions, k_vec)))

        assert np.allclose(computed_field, expected_field), f"Expected {expected_field}, got {computed_field}"

class Test_Standing_Wave_Gradient():

    def test_standing_wave_gradient_analytical(self):
        direction = np.array([0, 0, 1])
        amplitude = np.array([1.0, 0.0, 0.0])
        wave_number_nm = 2 * np.pi / 500  # Corresponds to 500 nm wavelength in vacuum
        positions = np.array([[0.0, 0.0, 50.0],
                              [0.0, 0.0, 100.0],
                              [0.0, 0.0, 150.0]])
        
        k_vec = wave_number_nm * direction
        computed_gradient = msp.standing_wave_gradient(direction, amplitude, positions, wave_number_nm)
        
        phase_term = np.sin(np.einsum('ij,j->i', positions, k_vec))
        direction_term = np.einsum('i,j->ij', -k_vec, amplitude)
        expected_gradient = np.einsum('ij,k->kij', direction_term, phase_term)

        assert np.allclose(computed_gradient, expected_gradient), f"Expected {expected_gradient}, got {computed_gradient}"
        
class Test_Standing_Wave_Double_Gradient():

    def test_standing_wave_double_gradient_analytical(self):
        direction = np.array([0, 0, 1])
        amplitude = np.array([1.0, 0.0, 0.0])
        wave_number_nm = 2 * np.pi / 500  # Corresponds to 500 nm wavelength in vacuum
        positions = np.array([[0.0, 0.0, 50.0],
                              [0.0, 0.0, 100.0],
                              [0.0, 0.0, 150.0]])
        
        k_vec = wave_number_nm * direction
        computed_double_gradient = msp.standing_wave_double_gradient(direction, amplitude, positions, wave_number_nm)
        
        phase_term = -np.cos(np.einsum('ij,j->i', positions, k_vec))
        direction_term = np.einsum('i,j,k->ijk', k_vec, k_vec, amplitude)
        expected_double_gradient = np.einsum('ijk,l->lijk', direction_term, phase_term)

        assert np.allclose(computed_double_gradient, expected_double_gradient), f"Expected {expected_double_gradient}, got {computed_double_gradient}"
        
class Test_Standing_Wave_Triple_Gradient():

    def test_standing_wave_triple_gradient_analytical(self):
        direction = np.array([0, 0, 1])
        amplitude = np.array([1.0, 0.0, 0.0])
        wave_number_nm = 2 * np.pi / 500  # Corresponds to 500 nm wavelength in vacuum
        positions = np.array([[0.0, 0.0, 50.0],
                              [0.0, 0.0, 100.0],
                              [0.0, 0.0, 150.0]])
        
        k_vec = wave_number_nm * direction
        computed_triple_gradient = msp.standing_wave_triple_gradient(direction, amplitude, positions, wave_number_nm)
        
        phase_term = np.sin(np.dot(positions, k_vec))
        direction_term = np.einsum('i,j,k,l->ijkl', k_vec, k_vec, k_vec, amplitude)
        expected_triple_gradient = np.einsum('ijkl,m->mijkl', direction_term, phase_term)

        assert np.allclose(computed_triple_gradient, expected_triple_gradient), f"Expected {expected_triple_gradient}, got {computed_triple_gradient}"        
        