import msptools as msp
import numpy as np
from scipy.constants import c, hbar, e
import refractiveindex
import matplotlib.pyplot as plt

wavelength = 1000e-9
frequency = c / wavelength * 2 * np.pi * hbar / e
plasma_frequency = 9
plasma_wavelength = c * hbar  / (plasma_frequency * e) / (2 * np.pi)
collision_frequency = 0.05
collision_wavelength = c * hbar / (collision_frequency * e) / (2 * np.pi)
epsilon_inf = 9

drude_epsilon = msp.permittivity.permittivity_Drude(frequency=frequency,
                                                          plasma_frequency=plasma_frequency,
                                                          collision_frequency=collision_frequency,
                                                          epsilon_inf=epsilon_inf)

drude_epsilon_tabulated = refractiveindex.RefractiveIndexMaterial(shelf='main', book='Au', page='McPeak').get_epsilon(wavelength_nm=wavelength*1e9)

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
print(f"McPeak's permittivity at {wavelength*1e9} nm: {drude_epsilon_tabulated:.4f}")
print()

wavelengths = np.linspace(500, 1100, 80)
frequencies = c / (wavelengths * 1e-9) * 2 * np.pi * hbar / e
permittivities = [msp.permittivity.permittivity_Drude(frequency=freq,
                                                            plasma_frequency=plasma_frequency,
                                                            collision_frequency=collision_frequency,
                                                            epsilon_inf=epsilon_inf) for freq in frequencies]

                                                            
Au = refractiveindex.RefractiveIndexMaterial(shelf='main', book='Au', page='McPeak')

tabulated_permittivities = [Au.get_epsilon(wavelength_nm=wavelength) for wavelength in wavelengths]


plt.figure(figsize=(8, 6))
plt.plot(wavelengths, np.real(permittivities), 'k-', label='Real Part')
plt.plot(wavelengths, np.imag(permittivities), 'r-', label='Imaginary Part')
plt.plot(wavelengths, np.real(tabulated_permittivities), 'k--', label='Real Part (Tabulated)')
plt.plot(wavelengths, np.imag(tabulated_permittivities), 'r--', label='Imaginary Part (Tabulated)')
plt.title('Drude Model Permittivity for a Gold-like Material', fontsize=14)
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Permittivity', fontsize=12)
plt.legend()
plt.grid()
plt.show()



