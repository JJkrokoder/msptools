import numpy as np
import matplotlib.pyplot as plt
import msptools as msp

wavelenght_nm = 642
k_mag = 2 * np.pi / wavelenght_nm 

wave_vec = np.array([0,0,1])
amplitude_vec = np.array([1,0,0])

z_array = np.linspace(0, 1000, 100)  # Propagation distances from 0 to 1 micrometer
positions = np.zeros((len(z_array), 3))
positions[:, 2] = z_array

field = msp.standing_wave_function(direction=wave_vec,
                                   amplitude_vec=amplitude_vec,
                                   k_magnitude=k_mag,
                                      positions=positions)


# gradient
gradient = msp.standing_wave_gradient(direction=wave_vec,
                                   amplitude_vec=amplitude_vec,
                                   k_magnitude=k_mag,
                                      positions=positions)

fig1 = plt.figure(figsize=(10, 4))
ax1 = fig1.add_subplot(1, 2, 1)
ax1.plot(z_array, field[:, 0])
ax1.set_title('Standing Wave Electric Field (X Component)')
ax1.set_xlabel('Position z (nm)')
ax1.set_ylabel('Electric Field E_x')
ax1.legend()
ax2 = fig1.add_subplot(1, 2, 2)
ax2.plot(z_array, gradient[:, 2, 0])
ax2.set_title('Standing Wave Electric Field Gradient (dE_x/dz)')
ax2.set_xlabel('Position z (nm)')
ax2.set_ylabel('Gradient dE_x/dz')
ax2.legend()
plt.tight_layout()
plt.show()

