from scipy.constants import c, h, e, hbar

multipliers = {
    "a": 1e-18,
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "m": 1e-3,
    "c": 1e-2,
    "d": 1e-1,
    "": 1,
    "da": 1e1,
    "h": 1e2,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
    "P": 1e15,
    "E": 1e18,
}

def frequency_to_eV(frequency: float, frequency_unit: str) -> float:
    """
    Convert frequency to energy in electron volts / hbar (eV).
    
    Parameters:
        frequency:
            Frequency value to be converted.
        frequency_unit:
            Unit of the frequency value. Supported units are Hertz (Hz) multipliers 
            from atto (a, 1e-18) to exa (E, 1e18) and electron volts multipliers from
            atto (a, 1e-18) to exa (E, 1e18).

    Returns:
        Frequency in electron volts / hbar (eV).
    
    Notes:
        Supported multipliers for Hertz (Hz) and electron volts (eV) are:
        - atto (a, 1e-18)
        - femto (f, 1e-15)
        - pico (p, 1e-12)
        - nano (n, 1e-9)
        - micro (u, 1e-6)
        - milli (m, 1e-3)
        - centi (c, 1e-2)
        - deci (d, 1e-1)
        - (no prefix, 1)
        - deca (da, 1e1)
        - hecto (h, 1e2)
        - kilo (k, 1e3)
        - Mega (M, 1e6)
        - Giga (G, 1e9)
        - Tera (T, 1e12)
        - Peta (P, 1e15)
        - Exa (E, 1e18)
    """

    if frequency_unit.endswith('Hz'):
        factor = multipliers.get(frequency_unit[:-2], 1)
        frequency_in_eV = frequency * factor * hbar / e
    elif frequency_unit.endswith('eV'):
        factor = multipliers.get(frequency_unit[:-2], 1)
        frequency_in_eV = frequency * factor
    else:
        raise ValueError(f"Unsupported frequency unit: {frequency_unit}")


    return frequency_in_eV


def wavelength_to_nm(wavelength: float, wavelength_unit: str) -> float:
    """
    Convert wavelength to nanometers (nm).
    
    Parameters:
        wavelength:
            Wavelength value to be converted.
        wavelength_unit:
            Unit of the wavelength value. Supported units are meters (m) multipliers 
            from atto (a, 1e-18) to exa (E, 1e18).

    Returns:
        Wavelength in nanometers (nm).

    Notes:
        Supported multipliers for meters (m) are:
        - atto (a, 1e-18)
        - femto (f, 1e-15)
        - pico (p, 1e-12)
        - nano (n, 1e-9)
        - micro (u, 1e-6)
        - milli (m, 1e-3)
        - centi (c, 1e-2)
        - deci (d, 1e-1)
        - (no prefix, 1)
        - deca (da, 1e1)
        - hecto (h, 1e2)
        - kilo (k, 1e3)
        - Mega (M, 1e6)
        - Giga (G, 1e9)
        - Tera (T, 1e12)
        - Peta (P, 1e15)
        - Exa (E, 1e18)
    """

    if wavelength_unit.endswith('m'):
        factor = multipliers.get(wavelength_unit[:-1], 1)
        wavelength_in_nm = wavelength * factor * 1e9
    else:
        raise ValueError(f"Unsupported wavelength unit: {wavelength_unit}")

    return wavelength_in_nm

def nm_to_eV(wavelength_nm: float) -> float:
    """
    Convert wavelength in nanometers (nm) to frequency in electron volts / hbar (eV).
    
    Parameters:
        wavelength_nm:
            Wavelength value in nanometers (nm).

    Returns:
        Energy in electron volts / hbar (eV).
    """
    wavelength_m = wavelength_nm * 1e-9
    frequency_eV = h * c / wavelength_m / e
    return frequency_eV
