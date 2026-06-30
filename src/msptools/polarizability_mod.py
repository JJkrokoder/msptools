from .backend import get_backend
from typing import Callable
from scipy.constants import pi
from numpy.typing import ArrayLike
from .permittivity import permittivity_ridx
from .tools.unit_calcs import nm_to_eV
from .tools.mie_theory import tE_n_coefficient, hankel_plus, tEn_aden_kerker_coefficient
from scipy.constants import c, e, h

def select_computation_method(material: str, wavelength: float) -> Callable[[float, str], float]:
    """Select the polarizability computation method based on the material and excitation wavelength."""  

def Clausius_Mossotti(radius: float, medium_permittivity: float, particle_permittivity: float) -> float:
    """
    Calculate the polarizability of a spherical particle using the Clausius-Mossotti relation.
    
    Parameters
    ----------
    radius : 
        The radius of the spherical particle.
    medium_permittivity :
        The permittivity of the surrounding medium.
    particle_permittivity :
        The permittivity of the particle material.
        
    
    Returns
    -------
    float
        The polarizability of the spherical particle.
    """

    polarizability = 4 * pi * (radius**3) * (particle_permittivity - medium_permittivity) / (particle_permittivity + 2 * medium_permittivity)

    return polarizability

def Core_Shell_Clausius_Mossotti(radius_core: float | ArrayLike,
                                 radius_shell: float | ArrayLike,
                                 medium_permittivity: float | ArrayLike,
                                 particle_permittivity_core: complex | ArrayLike,
                                 particle_permittivity_shell: complex | ArrayLike) -> complex | ArrayLike:
    """
    Calculate the polarizability of a core-shell particle using the Clausius-Mossotti relation.
    
    Parameters
    ----------
    radius_core : 
        The radius of the core of the particle.
    radius_shell :
        The radius of the shell of the particle (including the core).
    medium_permittivity :
        The permittivity of the surrounding medium.
    particle_permittivity_core :
        The permittivity of the core material.
    particle_permittivity_shell :
        The permittivity of the shell material.
        
    Returns
    -------
    complex | ArrayLike
        The polarizability of the core-shell particle.
        
    Notes
    -----
    Ensure proper broadcasting of the input parameters if they are arrays.
    """
    
    
    a3 = radius_core**3
    b3 = radius_shell**3
    e1 = particle_permittivity_core
    e2 = particle_permittivity_shell
    e_m = medium_permittivity  
    
    prefactor = 4 * pi * b3
    
    numerator = (e2 - e_m) * (e1 + 2 * e2) * b3 + (e1 - e2) * (2 * e2 + e_m) * a3
    denominator = (e2 + 2 * e_m) * (e1 + 2 * e2) * b3 + 2 * (e1 - e2) * (e2 - e_m) * a3  

    return prefactor * numerator / denominator


def Mie_size_dipole_approximation(radius: float, medium_permittivity: float, particle_permittivity: float, wave_number: float) -> complex:
    """
    Calculate the polarizability of a spherical particle using Mie size dipole approximation.
    The bessel and hankel functions are expanded to second order in size parameter.

    Parameters
    ----------
    radius : 
        The radius of the spherical particle.
    medium_permittivity :
        The permittivity of the surrounding medium.
    particle_permittivity :
        The permittivity of the particle material.
    wave_number :
        The wave number of the incident light (in vacuum). 

    Returns
    -------
    complex
        The polarizability of the spherical particle using Mie size expansion. 

    Notes
    -----
    The Mie size expansion is used for particles that are small or comparable to the wavelength of light.
    The formula is derived from the Mie theory.
    The expansion is given by:
    alpha = alpha_0 [1 - (k^2 r^2 / 10) * (ε + ε_m)]/[1 - i(k_m^3 alpha_0 / 6π) - (k^2 r^2 / 10)(ε + 10ε_m)*(ε - ε_m)/(ε + 2ε_m)]
    where alpha_0 is the Clausius-Mossotti polarizability.
    - Wave number and radius should be in consistent units.
    """

    k_m = wave_number * (medium_permittivity)**0.5
    k = wave_number
    e_m = medium_permittivity
    e_p = particle_permittivity
    rho = k * radius

    alpha_0 = Clausius_Mossotti(radius, e_m, e_p)

    epsilon_ratio = (e_p - e_m) / (e_p + 2 * e_m)

    A_term = (rho**2 / 10) * (e_p + e_m)
    B_term = (rho**2 / 10) * (e_p + 10 * e_m) * epsilon_ratio
    C_term = 1j * k_m**3 * alpha_0 / (6 * pi)

    polarizability_mie = alpha_0 * (1 - A_term) / (1 - C_term - B_term)

    return polarizability_mie

def Mie_electric_dipole_polarizability(radius: float, medium_permittivity: float, particle_permittivity: float, wave_number: float) -> complex:
    """
    Calculate the electric dipole polarizability of a spherical particle using Mie theory.
    
    Parameters
    ----------
    radius : 
        The radius of the spherical particle.
    medium_permittivity :
        The permittivity of the surrounding medium.
    particle_permittivity :
        The permittivity of the particle material.
    wave_number :
        The wave number of the incident light (in vacuum).

    Returns
    -------
    complex
        The electric dipole polarizability of the spherical particle using Mie theory.
    Notes
    -----
    The electric dipole polarizability is derived from the first Mie coefficient (a1).
    The formula is given by:
    alpha_e = 6*pi/k_m^3*tE1
    where k_m is the wave number in the medium and tE1 is the first Mie coefficient for the electric dipole.
    - Wave number and radius should be in consistent units.
    """
    k_m = wave_number * medium_permittivity**0.5
    x = k_m * radius
    m = (particle_permittivity**0.5) / (medium_permittivity**0.5)

    tE1 = tE_n_coefficient(n=1, x_m=x, m=m)
    alpha_e = 6 * pi / (k_m**3) * tE1
    return alpha_e

def Aden_Kerker_core_shell_polarizability(radius_core: float | ArrayLike, 
                      radius_shell: float | ArrayLike, 
                      medium_permittivity: complex | ArrayLike,
                      particle_permittivity_core: complex | ArrayLike,
                      particle_permittivity_shell: complex | ArrayLike,
                      wave_number: complex | ArrayLike) -> complex | ArrayLike:
    """
    Calculate the electric dipole polarizability of a core-shell particle using the Aden-Kerker formulation of Mie theory.
    
    Parameters    
    ----------
    radius_core :
        The radius of the core particle.
    radius_shell :
        The radius of the shell particle.
    medium_permittivity :
        The permittivity of the surrounding medium.
    particle_permittivity_core :
        The permittivity of the core material.
    particle_permittivity_shell :
        The permittivity of the shell material.
    wave_number :
        The wave number of the incident light (in vacuum).

    Returns
    -------
    complex | ArrayLike
        The electric dipole polarizability of the core-shell particle using the Aden-Kerker formulation of Mie theory.
    """
    
    k_m = wave_number * medium_permittivity ** 0.5

    x2 = k_m * radius_shell
    x1 = k_m * radius_core
    
    m1 = (particle_permittivity_core**0.5) / (medium_permittivity**0.5)
    m2 = (particle_permittivity_shell**0.5) / (medium_permittivity**0.5)

    tE1 = tEn_aden_kerker_coefficient(n=1, x_core=x1, x_shell=x2, m_1=m1, m_2=m2)
    
    return 6 * pi / (k_m**3) * tE1

def polarizability_to_matrix(polarizability: ArrayLike | float | int | complex, num_particles: int, dimensions: int, xp) -> ArrayLike:
    """
    Convert the polarizability from various input formats to a matrix form suitable for MSP calculations.

    Parameters
    ----------
    polarizability :
        The polarizability of the particles. Can be a scalar, a 1D array of length N, or a 3D array of shape (N, d, d).
    num_particles :
        The number of particles in the system.
    dimensions :
        The dimensionality of the system (e.g., 3 for 3D).
        
    Returns
    -------
    ArrayLike
        The polarizability in matrix form, with shape (N, d, d).
    
    """
        
    if xp.isscalar(polarizability):
        pol_identity = polarizability * xp.eye(dimensions)[None, :, :]
        polarizability = xp.repeat(pol_identity, num_particles, axis=0)
    elif polarizability.ndim == 1:
        polarizability = polarizability[:, None, None] * xp.eye(dimensions)[None, :, :]
    elif polarizability.ndim == 3 and polarizability.shape[1] == dimensions and polarizability.shape[2] == dimensions:
        pass
    else:
        raise ValueError("Invalid polarizability shape. Expected scalar, 1D array of length N, or 3D array of shape (N, d, d). Got {}".format(polarizability.shape))

    return polarizability

def compute_sphere_polarizability_DA(radius_nm: float | ArrayLike,
                                     medium_permittivity: float,
                                     particle_material: str,
                                     wavelength_nm: float | ArrayLike,
                                     method: str = 'Mie') -> complex|ArrayLike:
    """
    Compute the polarizability of a spherical particle using the Mie electric dipole formula.
    
    Parameters
    ----------
    radius_nm :
        The radius of the spherical particle.
    medium_permittivity :
        The permittivity of the surrounding medium.
    particle_material :
        The material of the particle.
    wavelength_nm :
        The wavelength of the incident light in nanometers.
    method :
        The method to compute the polarizability. 
        Options are 'Mie' for the full Mie solution, 'Mie_SA' for the size expansion approximation, or 'Clausius-Mossotti' for the quasistatic approximation.
    
    Returns
    -------
    complex|ArrayLike
        The polarizability of the spherical particle using the Mie electric dipole formula.
    """
    wave_number = 2 * pi / wavelength_nm
    frequency_eV =  nm_to_eV(wavelength_nm)
    particle_permittivity = permittivity_ridx(frequency_eV, particle_material)
    if method == 'Mie':
        polarizability = Mie_electric_dipole_polarizability(radius_nm, medium_permittivity, particle_permittivity, wave_number)
    elif method == 'Mie_SA':
        polarizability = Mie_size_dipole_approximation(radius_nm, medium_permittivity, particle_permittivity, wave_number)
    elif method == 'Clausius-Mossotti':
        polarizability = Clausius_Mossotti(radius_nm, medium_permittivity, particle_permittivity)
    
    return polarizability

def compute_core_shell_polarizability_DA(radius_core_nm: float | ArrayLike,
                                         radius_shell_nm: float | ArrayLike,
                                         medium_permittivity: float,
                                         material_core: str,
                                         material_shell: str,
                                         wavelength_nm: float | ArrayLike,
                                         method: str = 'Aden-Kerker') -> complex|ArrayLike:
    """
    Compute the polarizability of a core-shell particle using the Mie electric dipole formula.
    
    Parameters
    ----------
    radius_core_nm :
        The radius of the core of the particle in nanometers.
    radius_shell_nm :
        The radius of the shell of the particle in nanometers (including the core).
    medium_permittivity :
        The permittivity of the surrounding medium.
    material_core :
        The material of the core of the particle.
    material_shell :
        The material of the shell of the particle.
    wavelength_nm :
        The wavelength of the incident light in nanometers.
    method :
        The method to compute the polarizability. Options are 'Aden-Kerker' for the full Mie solution for core-shell particles, or 'Clausius-Mossotti' for the quasistatic approximation. 
        
    Returns
    -------
    complex|ArrayLike
        The polarizability of the core-shell particle using the Mie electric dipole formula.
    """
    
    wave_number = 2 * pi / wavelength_nm
    frequency_eV =  nm_to_eV(wavelength_nm)
    particle_permittivity_core = permittivity_ridx(frequency_eV, material_core)
    particle_permittivity_shell = permittivity_ridx(frequency_eV, material_shell)
    if method == 'Aden-Kerker':
        polarizability = Aden_Kerker_core_shell_polarizability(radius_core_nm, radius_shell_nm, medium_permittivity, particle_permittivity_core, particle_permittivity_shell, wave_number)
    elif method == 'Clausius-Mossotti':
        polarizability = Core_Shell_Clausius_Mossotti(radius_core_nm, radius_shell_nm, medium_permittivity, particle_permittivity_core, particle_permittivity_shell)
    return polarizability

def Mie_electric_quadrupole_polarizability(radius: float, medium_permittivity: float, particle_permittivity: float, wave_number: float) -> complex:
    """
    Calculate the electric quadrupole polarizability of a spherical particle using Mie theory.
    
    Parameters
    ----------
    radius : 
        The radius of the spherical particle.
    medium_permittivity :
        The permittivity of the surrounding medium.
    particle_permittivity :
        The permittivity of the particle material.
    wave_number :
        The wave number of the incident light (in vacuum).
    
    Returns
    -------
    complex
        The electric quadrupole polarizability of the spherical particle using Mie theory.
    Notes
    -----
    The electric quadrupole polarizability is derived from the second Mie coefficient (a2).
    The formula is given by:
    alpha_q = 40*pi/k_m^5*tE2
    where k_m is the wave number in the medium and tE2 is the second Mie coefficient for the electric quadrupole.
    - Wave number and radius should be in consistent units.
    """
    
    k_m = wave_number * medium_permittivity**0.5
    x = k_m * radius
    m = (particle_permittivity**0.5) / (medium_permittivity**0.5)

    tE2 = tE_n_coefficient(n=2, x_m=x, m=m)
    alpha_q = 40 * pi / (k_m**5) * tE2
    return alpha_q

def Aden_Kerker_core_shell_quadrupole_polarizability(radius_core: float | ArrayLike, 
                      radius_shell: float | ArrayLike, 
                      medium_permittivity: complex | ArrayLike,
                      particle_permittivity_core: complex | ArrayLike,
                      particle_permittivity_shell: complex | ArrayLike,
                      wave_number: complex | ArrayLike) -> complex | ArrayLike:
    """
    Calculate the electric quadrupole polarizability of a core-shell particle using the Aden-Kerker formulation of Mie theory.
    
    Parameters    
    ----------
    radius_core :
        The radius of the core particle.
    radius_shell :
        The radius of the shell particle.
    medium_permittivity :
        The permittivity of the surrounding medium.
    particle_permittivity_core :
        The permittivity of the core material.
    particle_permittivity_shell :
        The permittivity of the shell material.
    wave_number :
        The wave number of the incident light (in vacuum).
    
    Returns
    -------
    complex | ArrayLike
        The electric quadrupole polarizability of the core-shell particle using the Aden-Kerker formulation of Mie theory.
    """
    
    k_m = wave_number * medium_permittivity ** 0.5
    
    x2 = k_m * radius_shell
    x1 = k_m * radius_core
    
    m1 = (particle_permittivity_core**0.5) / (medium_permittivity**0.5)
    m2 = (particle_permittivity_shell**0.5) / (medium_permittivity**0.5)
    
    tE2 = tEn_aden_kerker_coefficient(n=2, x_core=x1, x_shell=x2, m_1=m1, m_2=m2)
    
    return 40 * pi / (k_m**5) * tE2

def compute_sphere_polarizability_QA(radius_nm: float | ArrayLike,
                                     medium_permittivity: float,
                                     particle_material: str,
                                     wavelength_nm: float | ArrayLike,
                                     method: str = 'Mie') -> complex|ArrayLike:
    """
    Compute the quadrupole polarizability of a spherical particle using the Mie electric quadrupole formula.
    
    Parameters
    ----------
    radius_nm :
        The radius of the spherical particle.
    medium_permittivity :
        The permittivity of the surrounding medium.
    particle_material :
        The material of the particle.
    wavelength_nm :
        The wavelength of the incident light in nanometers.
    method :
        The method to compute the polarizability.
        Options are 'Mie' for the full Mie solution.
    
    Returns
    -------
    complex|ArrayLike
        The quadrupole polarizability of the spherical particle using the Mie electric quadrupole formula.
    """
    
    wave_number = 2 * pi / wavelength_nm
    frequency_eV =  nm_to_eV(wavelength_nm)
    particle_permittivity = permittivity_ridx(frequency_eV, particle_material)
    if method == 'Mie':
        polarizability = Mie_electric_quadrupole_polarizability(radius_nm, medium_permittivity, particle_permittivity, wave_number)
    else:
        raise ValueError("Invalid method for quadrupole polarizability. Only 'Mie' is supported.")

    return polarizability

def compute_core_shell_polarizability_QA(radius_core_nm: float | ArrayLike,
                                         radius_shell_nm: float | ArrayLike,
                                         medium_permittivity: float,
                                         material_core: str,
                                         material_shell: str,
                                         wavelength_nm: float | ArrayLike,
                                         method: str = 'Aden-Kerker') -> complex|ArrayLike:
    """
    Compute the quadrupole polarizability of a core-shell particle using the Mie electric quadrupole formula.
    
    Parameters
    ----------
    radius_core_nm :
        The radius of the core of the particle in nanometers.
    radius_shell_nm :
        The radius of the shell of the particle in nanometers (including the core).
    medium_permittivity :
        The permittivity of the surrounding medium.
    material_core :
        The material of the core of the particle.
    material_shell :
        The material of the shell of the particle.
    wavelength_nm :
        The wavelength of the incident light in nanometers.
    method :
        The method to compute the polarizability. Options are 'Aden-Kerker' for the full Mie solution for core-shell particles.
    
    Returns
    -------
    complex|ArrayLike
        The quadrupole polarizability of the core-shell particle using the Mie electric quadrupole formula.
    """
    
    wave_number = 2 * pi / wavelength_nm
    frequency_eV =  nm_to_eV(wavelength_nm)
    particle_permittivity_core = permittivity_ridx(frequency_eV, material_core)
    particle_permittivity_shell = permittivity_ridx(frequency_eV, material_shell)
    if method == 'Aden-Kerker':
        polarizability = Aden_Kerker_core_shell_quadrupole_polarizability(radius_core_nm, radius_shell_nm, medium_permittivity, particle_permittivity_core, particle_permittivity_shell, wave_number)
    else:
        raise ValueError("Invalid method for quadrupole polarizability. Only 'Aden-Kerker' is supported.")
    return polarizability
 
        
        
        