import msptools as msp
import numpy as np

class TestSystem:
    
    medium_permittivity = 1.0
    
    def test_initialize_system(self):
        field = msp.PlaneWaveField(direction=[0, 0, 1], frequency=1.0, frequency_unit="eV", amplitude= 1.0, polarization=np.array([1.0, 0.0, 0.0]))
        type1 = msp.SphereType(radius=1.0, material="Au", radius_unit="nm")
        system = msp.System(field=field, medium_permittivity=self.medium_permittivity, particle_types=type1, positions_unit="nm")

        assert system.field.get_frequency() == 1.0, "Field frequency should be initialized to 1.0"
        assert system.medium_permittivity == self.medium_permittivity, "Medium permittivity should match the input"
        assert len(system.particle_types) == 1, "There should be one particle type in the system"
        assert system.particle_types[0].radius == 1.0, "Particle type radius should be initialized to 1.0"

    def test_add_particles_single_type(self):
        field = msp.PlaneWaveField(direction=[0, 0, 1], frequency=1.0, frequency_unit="eV", amplitude= 1.0, polarization=np.array([1.0, 0.0, 0.0]))
        type1 = msp.SphereType(radius=1.0, material="Au", radius_unit="nm")
        system = msp.System(field=field, medium_permittivity=self.medium_permittivity, particle_types=[type1], positions_unit="nm")
        
        positions = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        system.add_particles(positions, particle_type=type1)
        
        assert len(system.particles.positions) == 2, "There should be two particles in the system"
        assert all(isinstance(pos, list) for pos in system.particles.positions), "Positions should be stored as lists"
    
    def test_get_field_in_particles(self):
        field = msp.PlaneWaveField(direction=[0, 0, 1], wavelength=1.0, wavelength_unit="um", amplitude= 1.0, polarization=np.array([1.0, 0.0, 0.0]))
        type1 = msp.SphereType(radius=1.0, material="Au", radius_unit="nm")
        system = msp.System(field=field, medium_permittivity=self.medium_permittivity, particle_types=[type1], positions_unit="nm")
        
        positions = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        system.add_particles(positions, particle_type=type1)
        
        field_values = system.get_field_in_particles()
        
        assert field_values.shape == (2, 3), "Field values should have shape (num_particles, 3)"
        # assert np.allclose(field_values[0], field.external_field_function(np.array([1.369, 0.0, 0.0]))), "Field at first particle position should match evaluation"
        # assert np.allclose(field_values[1], field.external_field_function(np.array([2.0, 0.0, 0.0]))), "Field at second particle position should match evaluation"
    
    