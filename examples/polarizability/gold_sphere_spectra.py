from msptools.polarizability_mod import compute_sphere_polarizability_DA
import numpy as np
import matplotlib.pyplot as plt

particle_material = "Au"
particle_radius_nm = 140.0
wavelength_nm = np.linspace(400, 5000, 3000)
water_permittivity = 1.33**2

alpha = compute_sphere_polarizability_DA(radius_nm=particle_radius_nm,
                                         wavelength_nm=wavelength_nm,
                                         particle_material=particle_material,
                                         medium_permittivity=water_permittivity)

plt.figure(figsize=(10, 6))
plt.plot(wavelength_nm, alpha.real, label="Real part")
plt.plot(wavelength_nm, alpha.imag, label="Imaginary part")
plt.title(f"Polarizability of a {particle_radius_nm} nm {particle_material} sphere in water")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Polarizability (nm^3)")
plt.legend()
plt.grid()
plt.show()