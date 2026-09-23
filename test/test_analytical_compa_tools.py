from posixpath import sep

import numpy as np
from msptools.tools.analytical_comparison import (LOF_pair_spheres, LOF_pair_spheres_individual)
import msptools as msp

def obtain_msp_force(R, radius, mat1, mat2, medium_eps, wl):
    # Create two spheres with given parameters
    sphere1 = msp.SphereType(radius=radius, material=mat1, radius_unit="nm")
    sphere2 = msp.SphereType(radius=radius, material=mat2, radius_unit="nm")
    
    # Configure the plane wave field
    pw_field = msp.PlaneWaveField(vacuum_wavelength_nm=wl,
                                   medium_permittivity=medium_eps,
                                   polarization=np.array([0, 1, 0]),
                                   direction=np.array([0, 0, 1]),
                                   amplitude=1.0)

    sys = msp.System("CPU")

    sys.set_system(
        field=pw_field,
        particle_types=[sphere1, sphere2],
        positions_unit="nm",
        medium_permittivity=medium_eps
    )
    sys.add_particles(
        positions=np.array([[0, 0, 0]]),
        particle_type=sphere1
    )

    if isinstance(R, (float, int)):
        sys.add_particles(positions=np.array([[R, 0, 0]]), particle_type=sphere2)
        solver = msp.ForceCalculator(system=sys)
        f_msp = solver.compute_forces(positions=np.array([[0, 0, 0], [R, 0, 0]]), method="Inverse")
        forces_msp = np.array([f_msp[0, 0], f_msp[1, 0]]) 
    else:
        forces_msp = np.zeros((2, len(R)), dtype=complex)
        sys.add_particles(positions=np.array([[R[0], 0, 0]]), particle_type=sphere2)
        solver = msp.ForceCalculator(system=sys)
        for i, s in enumerate(R):
            f_msp = solver.compute_forces(positions = np.array([[0, 0, 0], [s, 0, 0]]), method="Inverse")
            forces_msp[0, i] = f_msp[0, 0]
            forces_msp[1, i] = f_msp[1, 0]
    return forces_msp

    
class TestLOFPairSpheres:
    """
    Test class for the LOF_pair_spheres function. Reproduces the results from Sukhov's paper (10.1364/OE.23.000247)
    """
    alpha_1 = 1.0 + 0.5j
    alpha_2 = 0.8 + 0.3j
    R = np.array([1.0, 2.0, 3.0])
    k_medium = 2 * np.pi / 0.5
    
    def test_no_sphere_2(self):
        # Test when alpha_2 is zero, the force should be zero
        force = LOF_pair_spheres(self.R, self.alpha_1, 0.0, self.k_medium, 'longitudinal')
        assert np.allclose(force, 0.0), "Force should be zero when alpha_2 is zero."
        
    def test_far_spheres(self):
        # Test when R is very large, the force should approach zero
        large_R = np.array([1e6, 1e7, 1e8])
        force = LOF_pair_spheres(large_R, self.alpha_1, self.alpha_2, self.k_medium, 'longitudinal')
        assert np.allclose(force, 0.0, atol=1e-4), "Force should approach zero for very large R."

    def test_far_single_scattering(self):
        # Test when R is very large with SingleScattering=True, multiple scattering contributions should be negligible
        large_R = np.array([1e1, 1e2, 1e3])
        force_ss = LOF_pair_spheres(large_R, self.alpha_1, self.alpha_2, self.k_medium, 'longitudinal', SingleScattering=True)
        force_ms = LOF_pair_spheres(large_R, self.alpha_1, self.alpha_2, self.k_medium, 'longitudinal', SingleScattering=False)
        assert np.allclose(force_ss, force_ms, rtol=7e-2), "Single scattering should approximate the full force for large R."

    def test_msp_comparison(self):
        # Compare the analytical LOF with the MSPTools computed force
        radius = 50.0  # nm
        mat1 = "Au"
        mat2 = "SiO2"
        medium_eps = 1.0
        wl = 1064.0  # nm
        k_medium = 2 * np.pi / wl * np.sqrt(medium_eps)
        R=self.R*radius*2.0  # Scale R to actual distances based on radius
        
        forces_msp = obtain_msp_force(R, radius, mat1, mat2, medium_eps, wl)
        
        # Compute polarizabilities for the given materials and wavelength
        alpha_1 = msp.compute_sphere_polarizability(radius_nm=radius, medium_permittivity=medium_eps, particle_material=mat1, wavelength_nm=wl)
        alpha_2 = msp.compute_sphere_polarizability(radius_nm=radius, medium_permittivity=medium_eps, particle_material=mat2, wavelength_nm=wl)
        
        force_analytical = LOF_pair_spheres(R, alpha_1, alpha_2, k_medium, 'transverse')
        
        assert np.allclose(np.sum(forces_msp, axis=0), force_analytical, rtol=1e-2), "MSPTools and analytical forces should match within tolerance."

    def test_msp_individual_comparison(self):
        # Compare the individual contributions to the LOF with MSPTools computed forces
        radius = 50.0  # nm
        mat1 = "Au"
        mat2 = "SiO2"
        medium_eps = 1.0
        wl = 1064.0  # nm
        k_medium = 2 * np.pi / wl * np.sqrt(medium_eps)
        R=self.R*radius*2.0  # Scale R to actual distances based on radius
        
        forces_msp = obtain_msp_force(R, radius, mat1, mat2, medium_eps, wl)
        
        # Compute polarizabilities for the given materials and wavelength
        alpha_1 = msp.compute_sphere_polarizability(radius_nm=radius, medium_permittivity=medium_eps, particle_material=mat1, wavelength_nm=wl)
        alpha_2 = msp.compute_sphere_polarizability(radius_nm=radius, medium_permittivity=medium_eps, particle_material=mat2, wavelength_nm=wl)
        
        force_analytical_individual = LOF_pair_spheres_individual(R, alpha_1, alpha_2, k_medium, 'transverse')
        
        assert np.allclose(forces_msp, force_analytical_individual, rtol=1e-2), "MSPTools and analytical individual forces should match within tolerance."


