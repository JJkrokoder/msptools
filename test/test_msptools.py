import msptools as msp
import numpy as np

class TestSystem:
    
    medium_permittivity = 1.0
    
    def test_initialize_system(self):
        field = msp.PlaneWaveField(direction=[0, 0, 1], frequency=1.0, frequency_unit="eV", amplitude= 1.0, polarization=np.array([1.0, 0.0, 0.0]))
        type1 = msp.SphereType(radius=1.0, material="Au")
        system = msp.System(field=field, medium_permittivity=self.medium_permittivity, particle_types=type1, positions_unit="nm")

        assert system.field.get_frequency() == 1.0, "Field frequency should be initialized to 1.0"
        assert system.medium_permittivity == self.medium_permittivity, "Medium permittivity should match the input"
        assert len(system.particle_types) == 1, "There should be one particle type in the system"
        assert system.particle_types[0].radius == 1.0, "Particle type radius should be initialized to 1.0"

    def test_add_particles_single_type(self):
        field = msp.PlaneWaveField(direction=[0, 0, 1], frequency=1.0, frequency_unit="eV", amplitude= 1.0, polarization=np.array([1.0, 0.0, 0.0]))
        type1 = msp.SphereType(radius=1.0, material="Au")
        system = msp.System(field=field, medium_permittivity=self.medium_permittivity, particle_types=[type1], positions_unit="nm")
        
        positions = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        system.add_particles(positions, particle_type=type1)
        
        assert len(system.particles.positions) == 2, "There should be two particles in the system"
        assert all(isinstance(pos, list) for pos in system.particles.positions), "Positions should be stored as lists"

    