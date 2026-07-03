import msptools as msp
try:
    import cupy as np
except ImportError:
    import numpy as np

class TestSystem:
    
    medium_permittivity = 1.3
    vacuum_wavelength_nm = 500.0
    medium_wavelength_nm = vacuum_wavelength_nm / medium_permittivity**0.5
    
    def test_initialize_system(self):
        field = msp.PlaneWaveField(direction=np.array([0, 0, 1]),
                                   vacuum_wavelength_nm=self.vacuum_wavelength_nm,
                                   medium_permittivity=self.medium_permittivity,
                                   amplitude= 1.0,
                                   polarization=np.array([1.0, 0.0, 0.0]))
        type1 = msp.SphereType(radius=1.0, material="Au", radius_unit="nm")
        system = msp.System()
        system.set_system(field=field, medium_permittivity=self.medium_permittivity, particle_types=type1, positions_unit="nm")

        assert system.medium_permittivity == self.medium_permittivity, "Medium permittivity should match the input"
        assert len(system.particle_types) == 1, "There should be one particle type in the system"
        assert system.particle_types[0].radius == 1.0, "Particle type radius should be initialized to 1.0"

    def test_add_particles_single_type(self):
        field = msp.PlaneWaveField(direction=np.array([0, 0, 1]),
                                   vacuum_wavelength_nm=self.vacuum_wavelength_nm,
                                   medium_permittivity=self.medium_permittivity,
                                   amplitude= 1.0,
                                   polarization=np.array([1.0, 0.0, 0.0]))
        type1 = msp.SphereType(radius=1.0, material="Au", radius_unit="nm")
        system = msp.System()
        system.set_system(field=field, medium_permittivity=self.medium_permittivity, particle_types=type1, positions_unit="nm")
        
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        system.add_particles(positions, particle_type=type1)
        
        assert len(system.particles.positions) == 2, "There should be two particles in the system"
        
    def test_get_field_in_particles(self):
        field = msp.PlaneWaveField(direction=np.array([0, 0, 1]),
                                   vacuum_wavelength_nm=self.vacuum_wavelength_nm,
                                   medium_permittivity=self.medium_permittivity,
                                   amplitude= 1.0,
                                   polarization=np.array([1.0, 0.0, 0.0]))
        type1 = msp.SphereType(radius=1.0, material="Au", radius_unit="nm")
        system = msp.System()
        system.set_system(field=field, medium_permittivity=self.medium_permittivity, particle_types=type1, positions_unit="nm")
        
        positions = np.array([[0.0, 0.0, 0.0], [250.0, 0.0, 0.0]])
        system.add_particles(positions, particle_type=type1)
        
        field_values = system.get_field_in_particles(positions)
        
        expected_field_1 = field.evaluate(np.array([[0.0, 0.0, 0.0]]))
        expected_field_2 = field.evaluate(np.array([[250.0, 0.0, 0.0]]))
        
        assert field_values.shape == (2, 3), "Field values should have shape (num_particles, 3)"
        assert np.allclose(field_values[0], expected_field_1), "Field at first particle position should match evaluation"
        assert np.allclose(field_values[1], expected_field_2), "Field at second particle position should match evaluation"


class TestExamples:
    
    def test_pressure_radiation(self):
        
        wavelength_nm = 1000.0
        eps_m = 1.77
        medium_wl = wavelength_nm / eps_m**0.5
        k_m = 2 * np.pi / medium_wl
        field = msp.PlaneWaveField(direction=np.array([0, 0, 1]),
                                   vacuum_wavelength_nm=wavelength_nm,
                                   medium_permittivity=eps_m,
                                   amplitude= 1.0,
                                   polarization=np.array([1.0, 0.0, 0.0]))
        type1 = msp.SphereType(radius=100.0, material="Au", radius_unit="nm")
        system = msp.System() 
        system.set_system(field=field, medium_permittivity=eps_m, particle_types=type1, positions_unit="nm")
        
        positions = np.array([[0.0, 0.0, 0.0]])
        system.add_particles(positions, particle_type=type1)
        
        force_calculator = msp.ForceCalculator(system)
        forces = force_calculator.compute_forces(positions)
        polarizability = type1.polarizability
        expected_force = 0.5 * eps_m * k_m * np.imag(polarizability)
        
        assert forces.shape == (1, 3), "Forces should have shape (num_particles, 3)"
        assert np.isclose(forces[0, 2], expected_force), f"Expected force magnitude {expected_force} nm^2, got {forces[0, 2]} nm^2"