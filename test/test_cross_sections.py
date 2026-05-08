from msptools.cross_sections import sphere_cross_section_from_Mie_coeffs, sphere_cross_section, core_shell_sphere_cross_section
import numpy as np
from pathlib import Path

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


class Test_ITMO_spectra:
    
    n_water = 1.33
    rtol = 0.05  # 5% relative tolerance for comparison with ITMO data
    atol = 5e-2  # Absolute tolerance for comparison with ITMO data
    
    def test_Ag_r200_water(self):
        # Test with Ag sphere of radius 200 nm in water
        radius_nm = 200  # nm
        eps_m = self.n_water**2
        material = 'Ag'
        
        data_file = Path(__file__).parent / "ITMO_spectra" / "Ag_r200_water.txt"
        
        data = np.genfromtxt(data_file, skip_header=21, names=True, delimiter=', ')
        
        wavelength_nm = data[data.dtype.names[0]]
        
        Q_ext_ITMO = data['Qext']
        Q_sca_ITMO = data['Qsca']
        Q_abs_ITMO = data['Qabs']
        
        Q_ext_E1_ITMO = data['Qext_E_dipole']
        Q_ext_M1_ITMO = data['Qext_H_dipole']
        Q_sca_E1_ITMO = data['Qsca_E_dipole']
        Q_sca_M1_ITMO = data['Qsca_H_dipole']
        Q_abs_E1_ITMO = data['Qabs_E_dipole']
        Q_abs_M1_ITMO = data['Qabs_H_dipole']
        
        Q_ext_E2_ITMO = data['Qext_E_quadrupole']
        Q_ext_M2_ITMO = data['Qext_H_quadrupole']
        Q_sca_E2_ITMO = data['Qsca_E_quadrupole']
        Q_sca_M2_ITMO = data['Qsca_H_quadrupole']
        Q_abs_E2_ITMO = data['Qabs_E_quadrupole']
        Q_abs_M2_ITMO = data['Qabs_H_quadrupole']

        Q_ext, Q_sca, Q_abs = sphere_cross_section(wavelength_nm, radius_nm, eps_m, material)
        Q_E1, Q_M1 = sphere_cross_section(wavelength_nm=wavelength_nm, radius_nm=radius_nm, medium_permittivity=eps_m, particle_material=material, multipole_order=1)
        Q_E2, Q_M2 = sphere_cross_section(wavelength_nm=wavelength_nm, radius_nm=radius_nm, medium_permittivity=eps_m, particle_material=material, multipole_order=2)

        assert np.allclose(Q_ext, Q_ext_ITMO, rtol=self.rtol, atol=self.atol), "Extinction cross-section does not match ITMO data within 5% relative tolerance"
        assert np.allclose(Q_sca, Q_sca_ITMO, rtol=self.rtol, atol=self.atol), "Scattering cross-section does not match ITMO data within 5% relative tolerance"
        assert np.allclose(Q_abs, Q_abs_ITMO, rtol=self.rtol, atol=self.atol), "Absorption cross-section does not match ITMO data within 5% relative tolerance"
        assert np.allclose(Q_E1[0], Q_ext_E1_ITMO, rtol=self.rtol, atol=self.atol), "Electric dipole contribution to extinction cross-section does not match ITMO data within 5% relative tolerance"
        assert np.allclose(Q_M1[0], Q_ext_M1_ITMO, rtol=self.rtol, atol=self.atol), "Magnetic dipole contribution to extinction cross-section does not match ITMO data within 5% relative tolerance"
        assert np.allclose(Q_E1[1], Q_sca_E1_ITMO, rtol=self.rtol, atol=self.atol), "Electric dipole contribution to scattering cross-section does not match ITMO data within 5% relative tolerance"
        assert np.allclose(Q_M1[1], Q_sca_M1_ITMO, rtol=self.rtol, atol=self.atol), "Magnetic dipole contribution to scattering cross-section does not match ITMO data within 5% relative tolerance"
        assert np.allclose(Q_E1[2], Q_abs_E1_ITMO, rtol=self.rtol, atol=self.atol), "Electric dipole contribution to absorption cross-section does not match ITMO data within 5% relative tolerance"
        assert np.allclose(Q_M1[2], Q_abs_M1_ITMO, rtol=self.rtol, atol=self.atol), "Magnetic dipole contribution to absorption cross-section does not match ITMO data within 5% relative tolerance"
        assert np.allclose(Q_E2[0], Q_ext_E2_ITMO, rtol=self.rtol, atol=self.atol), "Electric quadrupole contribution to extinction cross-section does not match ITMO data within 5% relative tolerance"
        assert np.allclose(Q_M2[0], Q_ext_M2_ITMO, rtol=self.rtol, atol=self.atol), "Magnetic quadrupole contribution to extinction cross-section does not match ITMO data within 5% relative tolerance"
        assert np.allclose(Q_E2[1], Q_sca_E2_ITMO, rtol=self.rtol, atol=self.atol), "Electric quadrupole contribution to scattering cross-section does not match ITMO data within 5% relative tolerance"
        assert np.allclose(Q_M2[1], Q_sca_M2_ITMO, rtol=self.rtol, atol=self.atol), "Magnetic quadrupole contribution to scattering cross-section does not match ITMO data within 5% relative tolerance"
        assert np.allclose(Q_E2[2], Q_abs_E2_ITMO, rtol=self.rtol, atol=self.atol), "Electric quadrupole contribution to absorption cross-section does not match ITMO data within 5% relative tolerance"
        assert np.allclose(Q_M2[2], Q_abs_M2_ITMO, rtol=self.rtol, atol=self.atol), "Magnetic quadrupole contribution to absorption cross-section does not match ITMO data within 5% relative tolerance"

    
    def test_Au_r150(self):
        # Test with Au sphere of radius 150 nm in water
        radius_nm = 150  # nm
        eps_m = self.n_water**2
        material = 'Au'
        rtol = 0.5
        atol = 1.8

        data_file = Path(__file__).parent / "ITMO_spectra" / "Au_r150.txt"
        
        data = np.genfromtxt(data_file, skip_header=21, names=True, delimiter=', ')
        wavelength_nm = data[data.dtype.names[0]]
        Q_ext_ITMO = data['Qext']
        Q_sca_ITMO = data['Qsca']
        Q_abs_ITMO = data['Qabs']
        
        Q_ext_E1_ITMO = data['Qext_E_dipole']
        Q_ext_M1_ITMO = data['Qext_H_dipole']
        Q_sca_E1_ITMO = data['Qsca_E_dipole']
        Q_sca_M1_ITMO = data['Qsca_H_dipole']
        Q_abs_E1_ITMO = data['Qabs_E_dipole']
        Q_abs_M1_ITMO = data['Qabs_H_dipole']
        
        Q_ext_E2_ITMO = data['Qext_E_quadrupole']
        Q_ext_M2_ITMO = data['Qext_H_quadrupole']
        Q_sca_E2_ITMO = data['Qsca_E_quadrupole']
        Q_sca_M2_ITMO = data['Qsca_H_quadrupole']
        Q_abs_E2_ITMO = data['Qabs_E_quadrupole']
        Q_abs_M2_ITMO = data['Qabs_H_quadrupole']

        Q_ext, Q_sca, Q_abs = sphere_cross_section(wavelength_nm, radius_nm, eps_m, material)
        Q_E1, Q_M1 = sphere_cross_section(wavelength_nm=wavelength_nm, radius_nm=radius_nm, medium_permittivity=eps_m, particle_material=material, multipole_order=1)
        Q_E2, Q_M2 = sphere_cross_section(wavelength_nm=wavelength_nm, radius_nm=radius_nm, medium_permittivity=eps_m, particle_material=material, multipole_order=2)
        
        assert np.allclose(Q_ext, Q_ext_ITMO, rtol=rtol, atol=atol), f"Extinction cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_sca, Q_sca_ITMO, rtol=rtol, atol=atol), f"Scattering cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_abs, Q_abs_ITMO, rtol=rtol, atol=atol), f"Absorption cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_E1[0], Q_ext_E1_ITMO, rtol=rtol, atol=atol), f"Electric dipole contribution to extinction cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_M1[0], Q_ext_M1_ITMO, rtol=rtol, atol=atol), f"Magnetic dipole contribution to extinction cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_E1[1], Q_sca_E1_ITMO, rtol=rtol, atol=atol), f"Electric dipole contribution to scattering cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_M1[1], Q_sca_M1_ITMO, rtol=rtol, atol=atol), f"Magnetic dipole contribution to scattering cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_E1[2], Q_abs_E1_ITMO, rtol=rtol, atol=atol), f"Electric dipole contribution to absorption cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_M1[2], Q_abs_M1_ITMO, rtol=rtol, atol=atol), f"Magnetic dipole contribution to absorption cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_E2[0], Q_ext_E2_ITMO, rtol=rtol, atol=atol), f"Electric quadrupole contribution to extinction cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_M2[0], Q_ext_M2_ITMO, rtol=rtol, atol=atol), f"Magnetic quadrupole contribution to extinction cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_E2[1], Q_sca_E2_ITMO, rtol=rtol, atol=atol), f"Electric quadrupole contribution to scattering cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_M2[1], Q_sca_M2_ITMO, rtol=rtol, atol=atol), f"Magnetic quadrupole contribution to scattering cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_E2[2], Q_abs_E2_ITMO, rtol=rtol, atol=atol), f"Electric quadrupole contribution to absorption cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_M2[2], Q_abs_M2_ITMO, rtol=rtol, atol=atol), f"Magnetic quadrupole contribution to absorption cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        
    def test_silicon_r150(self):
        # Test with Si sphere of radius 150 nm in water
        radius_nm = 150  # nm
        eps_m = self.n_water**2
        material = 'Si'
        rtol = 0.08
        atol = 2.9

        data_file = Path(__file__).parent / "ITMO_spectra" / "silicon_r150.txt"
        
        data = np.genfromtxt(data_file, skip_header=21, names=True, delimiter=', ')
        wavelength_nm = data[data.dtype.names[0]]
        Q_ext_ITMO = data['Qext']
        Q_sca_ITMO = data['Qsca']
        Q_abs_ITMO = data['Qabs']
        
        Q_ext_E1_ITMO = data['Qext_E_dipole']
        Q_ext_M1_ITMO = data['Qext_H_dipole']
        Q_sca_E1_ITMO = data['Qsca_E_dipole']
        Q_sca_M1_ITMO = data['Qsca_H_dipole']
        Q_abs_E1_ITMO = data['Qabs_E_dipole']
        Q_abs_M1_ITMO = data['Qabs_H_dipole']
        
        Q_ext_E2_ITMO = data['Qext_E_quadrupole']
        Q_ext_M2_ITMO = data['Qext_H_quadrupole']
        Q_sca_E2_ITMO = data['Qsca_E_quadrupole']
        Q_sca_M2_ITMO = data['Qsca_H_quadrupole']
        Q_abs_E2_ITMO = data['Qabs_E_quadrupole']
        Q_abs_M2_ITMO = data['Qabs_H_quadrupole']

        Q_ext, Q_sca, Q_abs = sphere_cross_section(wavelength_nm, radius_nm, eps_m, material)
        Q_E1, Q_M1 = sphere_cross_section(wavelength_nm=wavelength_nm, radius_nm=radius_nm, medium_permittivity=eps_m, particle_material=material, multipole_order=1)
        Q_E2, Q_M2 = sphere_cross_section(wavelength_nm=wavelength_nm, radius_nm=radius_nm, medium_permittivity=eps_m, particle_material=material, multipole_order=2)
        
        assert np.allclose(Q_ext, Q_ext_ITMO, rtol=rtol, atol=atol), f"Extinction cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_sca, Q_sca_ITMO, rtol=rtol, atol=atol), f"Scattering cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_abs, Q_abs_ITMO, rtol=rtol, atol=atol), f"Absorption cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_E1[0], Q_ext_E1_ITMO, rtol=rtol, atol=atol), f"Electric dipole contribution to extinction cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_M1[0], Q_ext_M1_ITMO, rtol=rtol, atol=atol), f"Magnetic dipole contribution to extinction cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_E1[1], Q_sca_E1_ITMO, rtol=rtol, atol=atol), f"Electric dipole contribution to scattering cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_M1[1], Q_sca_M1_ITMO, rtol=rtol, atol=atol), f"Magnetic dipole contribution to scattering cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_E1[2], Q_abs_E1_ITMO, rtol=rtol, atol=atol), f"Electric dipole contribution to absorption cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_M1[2], Q_abs_M1_ITMO, rtol=rtol, atol=atol), f"Magnetic dipole contribution to absorption cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_E2[0], Q_ext_E2_ITMO, rtol=rtol, atol=atol), f"Electric quadrupole contribution to extinction cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_M2[0], Q_ext_M2_ITMO, rtol=rtol, atol=atol), f"Magnetic quadrupole contribution to extinction cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_E2[1], Q_sca_E2_ITMO, rtol=rtol, atol=atol), f"Electric quadrupole contribution to scattering cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_M2[1], Q_sca_M2_ITMO, rtol=rtol, atol=atol), f"Magnetic quadrupole contribution to scattering cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_E2[2], Q_abs_E2_ITMO, rtol=rtol, atol=atol), f"Electric quadrupole contribution to absorption cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_M2[2], Q_abs_M2_ITMO, rtol=rtol, atol=atol), f"Magnetic quadrupole contribution to absorption cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
    
    
    def test_Au_62Si180(self):
        # Test with Au core of radius 62 nm and Si shell of thickness 180 nm in water
        r_core_nm = 62  # nm
        r_shell_nm = 180  # nm
        eps_m = self.n_water**2
        core_material = 'Au'
        shell_material = 'Si'
        rtol = 0.4
        atol = 2.9

        data_file = Path(__file__).parent / "ITMO_spectra" / "Au62Si180.txt"
        
        data = np.genfromtxt(data_file, skip_header=21, names=True, delimiter=', ')
        wavelength_nm = data[data.dtype.names[0]]
        Q_ext_ITMO = data['Qext']
        Q_sca_ITMO = data['Qsca']
        Q_abs_ITMO = data['Qabs']
        
        Q_ext_E1_ITMO = data['Qext_E_dipole']
        Q_ext_M1_ITMO = data['Qext_H_dipole']
        Q_sca_E1_ITMO = data['Qsca_E_dipole']
        Q_sca_M1_ITMO = data['Qsca_H_dipole']
        Q_abs_E1_ITMO = data['Qabs_E_dipole']
        Q_abs_M1_ITMO = data['Qabs_H_dipole']
        
        Q_ext_E2_ITMO = data['Qext_E_quadrupole']
        Q_ext_M2_ITMO = data['Qext_H_quadrupole']
        Q_sca_E2_ITMO = data['Qsca_E_quadrupole']
        Q_sca_M2_ITMO = data['Qsca_H_quadrupole']
        Q_abs_E2_ITMO = data['Qabs_E_quadrupole']
        Q_abs_M2_ITMO = data['Qabs_H_quadrupole']

        Q_ext, Q_sca, Q_abs = core_shell_sphere_cross_section(wavelength_nm, r_core_nm, r_shell_nm, eps_m, core_material, shell_material)
        Q_E1, Q_M1 = core_shell_sphere_cross_section(wavelength_nm=wavelength_nm, radius_core_nm=r_core_nm, radius_shell_nm=r_shell_nm, medium_permittivity=eps_m, core_material=core_material, shell_material=shell_material, multipole_order=1)
        Q_E2, Q_M2 = core_shell_sphere_cross_section(wavelength_nm=wavelength_nm, radius_core_nm=r_core_nm, radius_shell_nm=r_shell_nm, medium_permittivity=eps_m, core_material=core_material, shell_material=shell_material, multipole_order=2)
        
        assert np.allclose(Q_ext, Q_ext_ITMO, rtol=rtol, atol=atol), f"Extinction cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_sca, Q_sca_ITMO, rtol=rtol, atol=atol), f"Scattering cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_abs, Q_abs_ITMO, rtol=rtol, atol=atol), f"Absorption cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_E1[0], Q_ext_E1_ITMO, rtol=rtol, atol=atol), f"Electric dipole contribution to extinction cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_M1[0], Q_ext_M1_ITMO, rtol=rtol, atol=atol), f"Magnetic dipole contribution to extinction cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_E1[1], Q_sca_E1_ITMO, rtol=rtol, atol=atol), f"Electric dipole contribution to scattering cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_M1[1], Q_sca_M1_ITMO, rtol=rtol, atol=atol), f"Magnetic dipole contribution to scattering cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_E1[2], Q_abs_E1_ITMO, rtol=rtol, atol=atol), f"Electric dipole contribution to absorption cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_M1[2], Q_abs_M1_ITMO, rtol=rtol, atol=atol), f"Magnetic dipole contribution to absorption cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_E2[0], Q_ext_E2_ITMO, rtol=rtol, atol=atol), f"Electric quadrupole contribution to extinction cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_M2[0], Q_ext_M2_ITMO, rtol=rtol, atol=atol), f"Magnetic quadrupole contribution to extinction cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_E2[1], Q_sca_E2_ITMO, rtol=rtol, atol=atol), f"Electric quadrupole contribution to scattering cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_M2[1], Q_sca_M2_ITMO, rtol=rtol, atol=atol), f"Magnetic quadrupole contribution to scattering cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_E2[2], Q_abs_E2_ITMO, rtol=rtol, atol=atol), f"Electric quadrupole contribution to absorption cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        assert np.allclose(Q_M2[2], Q_abs_M2_ITMO, rtol=rtol, atol=atol), f"Magnetic quadrupole contribution to absorption cross-section does not match ITMO data within {rtol*100:.1f}% relative tolerance"
        
        