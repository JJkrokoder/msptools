import msptools as mspt
import numpy as np
import matplotlib.pyplot as plt

wavelengths = np.concatenate((
    np.linspace(300, 450, 15),
    np.linspace(450, 550, 30),
    np.linspace(700, 1100, 30)
))

frequencies = mspt.nm_to_eV(wavelengths)

print(f"\nCalculating gold permittivities from {wavelengths.min()} nm to {wavelengths.max()} nm")
permittivities = np.array([mspt.permittivity_ridx(freq, material='Au') for freq in frequencies])
print(" Done")

frohlich_wavelength = wavelengths[np.argmin(np.abs(permittivities + 2))]
print(f"\nVacuum Frohlich condition (minimum of |ε + 2|) met at wavelength: {frohlich_wavelength:.2f} nm")

print("\nWavelength (nm) | Real Part | Imaginary Part")
for wl, perm in zip(wavelengths, permittivities):
    print(f"{wl:.1f} | {perm.real:.3f} | {perm.imag:.3f}")

plt.figure(figsize=(8, 6))
plt.plot(wavelengths, permittivities.real, label='Real Part', color='black')
plt.plot(wavelengths, permittivities.imag, label='Imaginary Part', color='red')
plt.axvline(frohlich_wavelength, color='blue', linestyle='--', label=f'Frohlich Wavelength: {frohlich_wavelength:.2f} nm')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Permittivity')
plt.title('Gold Permittivity')
plt.legend()
plt.grid()
plt.show()


