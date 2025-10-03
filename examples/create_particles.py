import msptools as msp
import numpy as np

# Create a type
type1 = msp.SphereType(radius=2.5, material="wood")

# Create a ParticleData object with the custom type
particles = msp.ParticleData(types=type1)
print(f"Number of types in the system: {len(particles.types)}")  # Should be 1
print(f"Particle radius for type1: {particles.types[0].radius}")  # Should show radius of type1

# Create a particle at origin with the current added type
particles.add_particles([0.0, 0.0, 0.0])
print(f"\nNumber of particles in the system: {len(particles.positions)}")  # Should be 1
print(f"Type assignments for particles: {particles.type_assignments}")  # Should show [0]

# Create another type
type2 = msp.SphereType(radius=1.0, material="plastic")
# Add particles in the x,y diagonal with the new type
positions = [[i +1, i+1, 0.0] for i in range(5)]
particles.add_particles(positions, type=type2)
print(f"\nNew Number of particles in the system: {len(particles.positions)}")   # Should be 6
print(f"Type assignments for particles: {particles.type_assignments}")  # Should show [0, 1, 1, 1, 1, 1]


