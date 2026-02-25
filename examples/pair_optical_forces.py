import msptools as msptools
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
try:
    import cupy as np

    logging.log(logging.INFO, "Using CUDA backend")
except:
    logging.log(logging.INFO, "Using Fallback numpy backend")
    import numpy as np

radius_nm = 10
wavelength_nm = 532
medium_permittivity = 1
# logarithimic spacing
x = np.logspace(np.log10(2.02 * radius_nm), np.log10(7 * radius_nm), 100)
type1 = msptools.SphereType(material="Au", radius=radius_nm, radius_unit="nm")

ext_field_s = msptools.PlaneWaveField(
    wavelength=wavelength_nm,
    wavelength_unit="nm",
    direction=[0, 0, 1],
    polarization=[0, 1, 0],
    amplitude=1,
)
ext_field_p = msptools.PlaneWaveField(
    wavelength=wavelength_nm,
    wavelength_unit="nm",
    direction=[0, 0, 1],
    polarization=[1, 0, 0],
    amplitude=1,
)

system_s = msptools.System(
    medium_permittivity=medium_permittivity,
    particle_types=type1,
    field=ext_field_s,
    positions_unit="nm",
)
system_p = msptools.System(
    medium_permittivity=medium_permittivity,
    particle_types=type1,
    field=ext_field_p,
    positions_unit="nm",
)

polarizability = system_s.particle_types[0].polarizability
print(f"Polarizability at {system_s.field.get_wavelength()} nm: {polarizability} nm^3")

forces_s_list = []
forces_p_list = []
for system in [system_s, system_p]:
    forces_distance = []
    system.add_particles(positions=[[0, 0, 0], [10, 0, 0]])
    for dist in x:
        system.particles.set_position(1, [dist, 0, 0])
        forces = msptools.ForceCalculator(system)
        force_values = forces.compute_forces()
        forces_distance.append(force_values)
    forces_array = np.array(forces_distance)
    if system.field.polarization[0] == 0:
        forces_s_list.append(forces_array)
    else:
        forces_p_list.append(forces_array)

forces_s = np.array(forces_s_list)[0]
forces_p = np.array(forces_p_list)[0]

forces_p_analytical = -3 * np.abs(polarizability) ** 2 / x**4 / (4 * np.pi)
forces_s_analytical = 3 / 2 * np.abs(polarizability) ** 2 / x**4 / (4 * np.pi)

plt.figure(figsize=(8, 6))
plt.plot(x, forces_s[:, 1, 0], label="s-polarization", color="blue")
plt.plot(x, forces_p[:, 1, 0], label="p-polarization", color="red")
plt.scatter(
    x,
    forces_s_analytical,
    label="s-polarization analytical",
    color="cyan",
    s=10,
    marker="x",
)
plt.scatter(
    x,
    forces_p_analytical,
    label="p-polarization analytical",
    color="orange",
    s=10,
    marker="x",
)
# plt.yscale('log')
plt.xlabel("Distance (nm)")
plt.ylabel("Optical Force (a.u.)")
plt.title(
    f"Optical Forces between two Au NPs ({radius_nm} nm radius) at {wavelength_nm} nm wavelength"
)
plt.legend()
plt.grid(True, which="both", ls="--")


# Far-field calculation (larger separations) and modified analytical expressions
x_far = np.logspace(np.log10(3 * wavelength_nm), np.log10(6 * wavelength_nm), 200)
k_m = 2 * np.pi * np.sqrt(medium_permittivity) / wavelength_nm  # wavenumber in 1/nm

forces_s_far_list = []
forces_p_far_list = []
for system in [system_s, system_p]:
    forces_distance = []
    # ensure both particles present and update positions for each distance
    for dist in x_far:
        system.particles.set_position(1, [dist, 0, 0])
        forces = msptools.ForceCalculator(system)
        force_values = forces.compute_forces()
        forces_distance.append(force_values)
    forces_array = np.array(forces_distance)
    if system.field.polarization[0] == 0:
        forces_s_far_list.append(forces_array)
    else:
        forces_p_far_list.append(forces_array)

forces_s_far = np.array(forces_s_far_list)[0]
forces_p_far = np.array(forces_p_far_list)[0]

# Example far-field analytical models (radiative term with oscillatory cos(k r) / r^2 decay)
forces_p_far_analytical = (
    np.abs(polarizability) ** 2 * np.cos(k_m * x_far) * k_m**2 / x_far**2 / (4 * np.pi)
)
forces_s_far_analytical = (
    -np.abs(polarizability) ** 2
    * np.sin(k_m * x_far)
    * k_m**3
    / (2 * x_far)
    / (4 * np.pi)
)

plt.figure(figsize=(8, 6))
plt.plot(
    x_far, forces_s_far[:, 1, 0], label="s-polarization (numerical, far)", color="blue"
)
plt.plot(
    x_far, forces_p_far[:, 1, 0], label="p-polarization (numerical, far)", color="red"
)
plt.plot(
    x_far,
    forces_s_far_analytical,
    label="s-polarization analytical (far)",
    color="cyan",
    linestyle="--",
)
plt.plot(
    x_far,
    forces_p_far_analytical,
    label="p-polarization analytical (far)",
    color="orange",
    linestyle="--",
)
plt.xlabel("Distance (nm)")
plt.ylabel("Optical Force (a.u.)")
plt.title(
    f"Far-field Optical Forces between two Au NPs ({radius_nm} nm radius) at {wavelength_nm} nm"
)
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
