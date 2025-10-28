import refractiveindex as ridx
from scipy.constants import c, e, h
from .unit_calcs import *
from .ridx_usage import obtain_ridx_material_info
import numpy as np

def permittivity_Drude(frequency: float, plasma_frequency: float, collision_frequency: float, epsilon_inf: float) -> complex:
    """
    Calculate the permittivity using the Drude model.
    
    Parameters
    ----------
    frequency : float
        The frequency of the incident light.
    plasma_frequency : float
        The plasma frequency of the material.
    collision_frequency : float
        The collision frequency of the electrons in the material.
    epsilon_inf : float
        The high-frequency dielectric constant of the material.
    
    Returns
    -------
    complex
        The dielectric constant of the material.
    """
    epsilon = epsilon_inf - (plasma_frequency**2) / (frequency**2 + 1j * frequency * collision_frequency)
    return epsilon

def permittivity_ridx(frequency: float | np.ndarray, material: str) -> complex:
    """
    Calculate the permittivity using the RefractiveIndex package.
    
    Parameters
    ----------
    frequency :
        The frequency of the incident light in eV.
    material :
        The name of the material as recognized by the RefractiveIndex package.
    
    Returns
    -------
    complex
        The dielectric constant of the material.
    """
    
    shelf, book, page = obtain_ridx_material_info(material)
    Material = ridx.RefractiveIndexMaterial(shelf=shelf, book=book, page=page)
    if isinstance(frequency, np.ndarray):
        wavelength_nm = eV_to_nm(frequency)
        epsilon = np.array([Material.get_epsilon(wavelength_nm=w) for w in wavelength_nm])
    else:
        wavelength_nm = eV_to_nm(frequency)
        epsilon = Material.get_epsilon(wavelength_nm=wavelength_nm)
    return epsilon
