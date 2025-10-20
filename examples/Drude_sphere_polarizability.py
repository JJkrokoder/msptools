import msptools as msp
import numpy as np
from scipy.constants import c, hbar, e

wavelength = 1000e-9
frequency = c / wavelength * 2 * np.pi * hbar / e
plasma_frequency = 9
plasma_wavelength = c * hbar  / (plasma_frequency * e) / (2 * np.pi)
collision_frequency = 0.05
collision_wavelength = c * hbar / (collision_frequency * e) / (2 * np.pi)
epsilon_inf = 9

drude_epsilon = msp.polarizability_mod.permittivity_Drude(frequency=frequency,
                                                          plasma_frequency=plasma_frequency,
                                                          collision_frequency=collision_frequency,
                                                          epsilon_inf=epsilon_inf)

print()
print("Drude Model Permittivity Calculation for Gold-like Material")
print("-----------------------------------")
print("Parameters:")
print(f"  Wavelength: {wavelength*1e9} nm")
print(f"  Frequency: {frequency:.2f} eV")
print(f"  Plasma Frequency: {plasma_frequency:.2f} eV")
print(f"  Plasma Wavelength: {plasma_wavelength*1e9:.2f} nm")
print(f"  Collision Frequency: {collision_frequency:.2f} eV")
print(f"  Collision Wavelength: {collision_wavelength*1e9:.2f} nm")
print(f"  Epsilon Inf: {epsilon_inf}")
print(f"Drude permittivity at {wavelength*1e9} nm: {drude_epsilon:.4f}")
print()

wavelengths = np.linspace(500, 1100, 80)
frequencies = c / (wavelengths * 1e-9) * 2 * np.pi * hbar / e
permittivities = [msp.polarizability_mod.permittivity_Drude(frequency=freq,
                                                            plasma_frequency=plasma_frequency,
                                                            collision_frequency=collision_frequency,
                                                            epsilon_inf=epsilon_inf) for freq in frequencies]

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(wavelengths, np.real(permittivities), label='Real Part')
plt.plot(wavelengths, np.imag(permittivities), label='Imaginary Part')
plt.title('Drude Model Permittivity for a Gold-like Material', fontsize=14)
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Permittivity', fontsize=12)
plt.legend()
plt.grid()
plt.show()



