import msptools as msp
from scipy.constants import h, e, hbar, c
import numpy as np

a_tolerance = 1e-12

def test_frequency_to_eV():
    freq_hz = 1e14  # 100 THz
    energy_ev = msp.frequency_to_eV(freq_hz, "Hz")
    expected_energy_ev = (hbar * freq_hz) / e
    assert np.isclose(energy_ev, expected_energy_ev, atol=a_tolerance), f"Expected {expected_energy_ev}, got {energy_ev}"

    freq_kev = 1.0  # 1 keV/hbar
    energy_ev_kev = msp.frequency_to_eV(freq_kev, "keV")
    expected_energy_ev_kev = freq_kev * 1e3
    assert np.isclose(energy_ev_kev, expected_energy_ev_kev, atol=a_tolerance), f"Expected {expected_energy_ev_kev}, got {energy_ev_kev}"

def test_wavelength_to_nm():
    wavelength_m = 500e-9  # 500 nm
    wavelength_nm = msp.wavelength_to_nm(wavelength_m, "m")
    expected_wavelength_nm = 500.0
    assert np.isclose(wavelength_nm, expected_wavelength_nm, atol=a_tolerance), f"Expected {expected_wavelength_nm}, got {wavelength_nm}"

def test_nm_to_eV():
    wavelength_nm = 500.0  # 500 nm
    energy_ev = msp.nm_to_eV(wavelength_nm)
    expected_energy_ev = (h * c) / (wavelength_nm * 1e-9 * e)
    assert np.isclose(energy_ev, expected_energy_ev, atol=a_tolerance), f"Expected {expected_energy_ev}, got {energy_ev}"

def test_eV_to_wavenumber_um():
    frequency_eV = 2.0  # 2 eV
    wavenumber_um = msp.frequency_to_wavenumber_um(frequency_eV)
    expected_wavenumber_um = (frequency_eV * e) / (hbar * c) / 1e6
    assert np.isclose(wavenumber_um, expected_wavenumber_um, atol=a_tolerance), f"Expected {expected_wavenumber_um}, got {wavenumber_um}"

def test_get_multiplier_nanometers():
    multiplier_km = msp.get_multiplier_nanometers("km")
    expected_multiplier_km = 1e12  # 1 km = 1e12 nm
    assert np.isclose(multiplier_km, expected_multiplier_km, atol=a_tolerance), f"Expected {expected_multiplier_km}, got {multiplier_km}"

    multiplier_cm = msp.get_multiplier_nanometers("cm")
    expected_multiplier_cm = 1e7  # 1 cm = 1e7 nm
    assert np.isclose(multiplier_cm, expected_multiplier_cm, atol=a_tolerance), f"Expected {expected_multiplier_cm}, got {multiplier_cm}"
