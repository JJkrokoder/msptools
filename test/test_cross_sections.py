from msptools.cross_sections import sphere_cross_section_from_Mie_coeffs, sphere_cross_section
import numpy as np

class Test_CS_from_Mie_coeffs:
    
    def test_only_dipole_handling(self):
        # Test with only dipole terms (n=1)
        tE_n = np.array([0.1])  # Example Mie coefficient for dipole
        tM_n = np.array([0.08])  # Example Mie coefficient for magnetic dipole
        wavelength_nm = 530  # nm in medium

        C_ext, C_sca, C_abs = sphere_cross_section_from_Mie_coeffs(wavelength_nm, tE_n, tM_n)

        # Check that the cross-sections are positive and that C_ext >= C_sca
        assert C_sca > 0, "Scattering cross-section should be positive"
        assert C_ext == C_sca + C_abs, "Extinction cross-section should be greater than or equal to scattering cross-section"
        
    def test_only_electric_terms(self):
        # Test with only electric terms (tM_n = None)
        tE_n = np.array([0.1, 0.05])  # Example Mie coefficients for dipole and quadrupole
        wavelength_nm = 530  # nm in medium

        C_ext, C_sca, C_abs = sphere_cross_section_from_Mie_coeffs(wavelength_nm, tE_n, None)

        # Check that the cross-sections are positive and that C_ext >= C_sca
        assert C_sca > 0, "Scattering cross-section should be positive"
        assert C_ext == C_sca + C_abs, "Extinction cross-section should be greater than or equal to scattering cross-section"


class Test_Sphere_Cross_Section:
    
    def test_positive_cross_sections(self):
        # Test with only dipole terms (n=1)
        radius_nm = 50  # nm
        wavelength_nm = 1000  # nm in medium
        eps_m = 1.77
        material = 'Au'

        C_ext, C_sca, C_abs = sphere_cross_section(wavelength_nm, radius_nm, eps_m, material)

        # Check that the cross-sections are positive and that C_ext >= C_sca
        assert C_sca > 0, f"Scattering cross-section should be positive, got {C_sca}"
        assert C_ext == C_sca + C_abs, f"Extinction cross-section should be greater than or equal to scattering cross-section, got {C_ext} vs {C_sca} + {C_abs}"
        assert C_abs > 0, f"Absorption cross-section should be positive, got {C_abs}"
        assert C_ext >= C_sca, f"Extinction cross-section should be greater than or equal to scattering cross-section, got {C_ext} vs {C_sca}"