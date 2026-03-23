import msptools as msp
import numpy as np
import matplotlib.pyplot as plt

# Define system parameters
wavelength_nm = 1064  # Wavelength in nm
polarization = [1,0,0]  # Polarization vector
propagation_direction = [0,0,1]  # Propagation direction vector
medium_permittivity = 1.0  # Relative permittivity of the medium

gold_type = msp.SphereType(material='Au', radius=100, radius_unit='nm')

laser_power = 0.5  # Laser power in Watts
amplitude = np.sqrt(2 * laser_power / (3e8 * 8.854e-12))  # Calculate amplitude from power

ext_field = msp.PlaneWaveField(wavelength=wavelength_nm,
                               wavelength_unit='nm',
                             polarization=polarization,
                             direction=propagation_direction,
                             amplitude=1.0,
                             medium_permittivity=medium_permittivity)

system = msp.System(particle_types=gold_type,
                    field=ext_field,
                    medium_permittivity=medium_permittivity,
                    positions_unit ='nm')

# Define particle positions
num_particles = 1
particle_positions = np.array([[0, 0, 0]])  # Single particle at origin
system.add_particles(positions=particle_positions)

# Compute the particle polarizability
particle_polarizability = system.particle_types[0].polarizability
print(f'Particle polarizability at {wavelength_nm} nm: {particle_polarizability} nm^3')
#cross section 
cross_section_m = np.imag((2 * np.pi * particle_polarizability) / (wavelength_nm*medium_permittivity))*1e-18  # Convert from nm^2 to m^2
print(f'Extinction cross section at {wavelength_nm} nm: {cross_section_m} m^2')

# compute forces
force_calculator = msp.ForceCalculator(system)
forces = force_calculator.compute_forces()
print(f'Z force in water (N): {forces[0,2]}')



