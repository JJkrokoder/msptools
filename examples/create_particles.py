import msptools as msp
import numpy as np

# Create a type
type1 = msp.SphereType(radius=2.5, material="wood")

# Create a ParticleData object with the custom type
system = msp.System(particle_types=type1, field=msp.Field(frequency=1.0,
                                                          frequency_unit="eV"),
                                                          positions_unit="nm")
print(f"Number of types in the system: {len(system.particle_types)}")  # Should be 1
print(f"Particle radius for type1: {system.particle_types[0].radius}")  # Should show radius of type1

# Create a particle at origin with the current added type
system.add_particles([0.0, 0.0, 0.0])
print(f"\nNumber of particles in the system: {len(system.particles.positions)}")  # Should be 1



