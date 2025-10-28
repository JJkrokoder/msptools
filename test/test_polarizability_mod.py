import numpy as np
import msptools as mspt

def test_Clausius_Mossotti():
    radius = 0.08  # um
    medium_permittivity = 1
    ϵ1 = -2.5676
    ϵ2 = 3.6391
    particle_permittivity = ϵ1 + ϵ2 * 1j  # e.g., gold at 500 nm

    alpha = mspt.polarizability_mod.Clausius_Mossotti(radius, medium_permittivity, particle_permittivity)
    expected_alpha = 4 * np.pi * radius**3 * (particle_permittivity - medium_permittivity) / (particle_permittivity + 2 * medium_permittivity)
    
    assert np.isclose(alpha, expected_alpha), f"Expected {expected_alpha}, got {alpha}"

def test_CM_with_Correction_low_k_limit():
    radius = 0.08  # um
    medium_permittivity = 1.33
    particle_permittivity = 2.5 + 0.1j  # Example permittivity
    wave_number_um = 1e-6  # Very small wave number to simulate low-frequency limit

    alpha_corrected = mspt.polarizability_mod.CM_with_Correction(radius, medium_permittivity, particle_permittivity, wave_number_um)
    alpha_CM = mspt.polarizability_mod.Clausius_Mossotti(radius, medium_permittivity, particle_permittivity)
    alpha_approx = alpha_CM * (1 + 1j * (wave_number_um**3) * alpha_CM / (6 * np.pi))
    assert np.isclose(alpha_corrected, alpha_approx), f"Expected {alpha_approx}, got {alpha_corrected} in low k limit"

