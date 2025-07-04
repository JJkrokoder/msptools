import numpy as np

def calculate_dipole_moments_linear(polarizability, electric_field : np.ndarray) -> np.ndarray:
    
    dipole_moments = np.zeros_like(electric_field, dtype=complex)
 
    if isinstance(polarizability, (complex, float, int)):
        polarizability += 0j  # Ensure polarizability is treated as a complex number
        dipole_moments = polarizability * electric_field
    elif isinstance(polarizability, (list, np.ndarray)):
        number_of_polarizabilities = len(polarizability) if isinstance(polarizability, list) else polarizability.shape[0]
        if number_of_polarizabilities != electric_field.shape[0]:
            raise ValueError("Polarizability and electric field must have the same number of elements.")

        if isinstance(polarizability[0], complex):
            for i in range(len(polarizability)):
                dipole_moments[i] = polarizability[i] * electric_field[i]
        elif isinstance(polarizability[0], (np.ndarray)):
            for i in range(len(polarizability)):
                dipole_moments[i] = polarizability[i] @ electric_field[i]
    else:
        raise TypeError("Polarizability must be a complex number, float, int, list, or numpy array.")


    return dipole_moments

