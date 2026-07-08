from .backend import get_backend
from scipy.constants import pi
from numpy.typing import ArrayLike
from .permittivity import permittivity_ridx
from .tools.unit_calcs import nm_to_eV
from .tools.mie_theory import (tE_n_coefficient,
                               tEn_aden_kerker_coefficient,
                               tM_n_coefficient,
                               tMn_aden_kerker_coefficient)

_MULTIPOLE_PREFACTOR = {1: 6 * pi, 2: 40 * pi}

def Mie_multipole_polarizability(radius: float,
                                 medium_permittivity: float,
                                 particle_permittivity: float,
                                 wave_number: float,
                                 order: int = 1,
                                 EM_field: str = 'electric') -> complex:

    """
    Calculate the multipole polarizability of a spherical particle using Mie theory.
    
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
    order :
        The order of the multipole expansion.
    EM_field :
        The type of electromagnetic field ('electric', 'magnetic').

    Returns
    -------
    complex
        The multipole polarizability of the spherical particle.
    """
    
    k_m = wave_number * medium_permittivity**0.5
    n = order
    x = k_m * radius
    m = (particle_permittivity**0.5) / (medium_permittivity**0.5)
    tn_func = tE_n_coefficient if EM_field == 'electric' else tM_n_coefficient
    t_n = tn_func(n, x, m)
    return _MULTIPOLE_PREFACTOR[n] / (k_m**(2*n + 1)) * t_n

def Mie_core_shell_multipole_polarizability(radius_core: float,
                                           radius_shell: float,
                                           medium_permittivity: float,
                                           particle_permittivity_core: float,
                                           particle_permittivity_shell: float,
                                           wave_number: float,
                                           order: int = 1,
                                           EM_field: str = 'electric') -> complex:
    """
    Calculate the multipole polarizability of a core-shell spherical particle using Mie theory.

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
    wave_number :
        The wave number of the incident light (in vacuum).
    order :
        The order of the multipole expansion.
    EM_field :
        The type of electromagnetic field ('electric', 'magnetic').

    Returns
    -------
    complex
        The multipole polarizability of the core-shell particle.
    """
    
    k_m = wave_number * medium_permittivity**0.5
    n = order
    x1, x2 = k_m * radius_core, k_m * radius_shell
    m1 = (particle_permittivity_core**0.5) / (medium_permittivity**0.5)
    m2 = (particle_permittivity_shell**0.5) / (medium_permittivity**0.5)
    tn_func = tEn_aden_kerker_coefficient if EM_field == 'electric' else tMn_aden_kerker_coefficient
    t_n = tn_func(n, x_core=x1, x_shell=x2, m_1=m1, m_2=m2)
    return _MULTIPOLE_PREFACTOR[n] / (k_m**(2*n + 1)) * t_n

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

_SPHERE_METHODS = {
    ("electric", 1): {"Mie": lambda *a: Mie_multipole_polarizability(*a, order=1, EM_field="electric"),
                       "Mie_SA": Mie_size_dipole_approximation,
                       "Clausius-Mossotti": Clausius_Mossotti},
    ("electric", 2): {"Mie": lambda *a: Mie_multipole_polarizability(*a, order=2, EM_field="electric")},
    ("magnetic", 1): {"Mie": lambda *a: Mie_multipole_polarizability(*a, order=1, EM_field="magnetic")},
    ("magnetic", 2): {"Mie": lambda *a: Mie_multipole_polarizability(*a, order=2, EM_field="magnetic")},
}

_CORE_SHELL_METHODS = {
    ("electric", 1): {"Aden-Kerker": lambda *a: Mie_core_shell_multipole_polarizability(*a, order=1, EM_field="electric"),
                       "Clausius-Mossotti": Core_Shell_Clausius_Mossotti},
    ("electric", 2): {"Aden-Kerker": lambda *a: Mie_core_shell_multipole_polarizability(*a, order=2, EM_field="electric")},
    ("magnetic", 1): {"Aden-Kerker": lambda *a: Mie_core_shell_multipole_polarizability(*a, order=1, EM_field="magnetic")},
    ("magnetic", 2): {"Aden-Kerker": lambda *a: Mie_core_shell_multipole_polarizability(*a, order=2, EM_field="magnetic")},
}

def compute_sphere_polarizability(radius_nm: float | ArrayLike,
                                  medium_permittivity: float,
                                  particle_material: str,
                                  wavelength_nm: float | ArrayLike,
                                  order: int = 1,
                                  EM_field: str = 'electric',
                                  method: str = 'Mie') -> complex|ArrayLike:
    """
    Compute the multipole polarizability of a spherical particle.
    
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
    order :
        The order of the multipole expansion.
    EM_field :
        The type of electromagnetic field (electric or magnetic).
    method :
        The method to compute the polarizability. 
        Options are 'Mie' for the full Mie solution, 'Mie_SA' for the size expansion approximation, or 'Clausius-Mossotti' for the quasistatic approximation.
    
    Returns
    -------
    complex|ArrayLike
        The polarizability of the spherical particle using the Mie multipole formula.
    """
    
    wave_number = 2 * pi / wavelength_nm
    frequency_eV =  nm_to_eV(wavelength_nm)
    particle_permittivity = permittivity_ridx(frequency_eV, particle_material)
    try:
        polarizability_func = _SPHERE_METHODS[(EM_field, order)][method]
    except KeyError:
        valid_methods = _SPHERE_METHODS.get((EM_field, order), {})
        raise ValueError(f"Invalid method '{method}' for {EM_field} multipole order {order}. Valid methods are: {list(valid_methods.keys())}")
    
    return polarizability_func(radius_nm, medium_permittivity, particle_permittivity, wave_number)

def compute_core_shell_polarizability(radius_core_nm: float | ArrayLike,
                                      radius_shell_nm: float | ArrayLike,
                                      medium_permittivity: float,
                                      material_core: str,
                                      material_shell: str,
                                      wavelength_nm: float | ArrayLike,
                                      order: int = 1,
                                      EM_field: str = 'electric',
                                      method: str = 'Aden-Kerker') -> complex|ArrayLike:
    """
    Compute the multipole polarizability of a core-shell particle.

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
    order :
        The order of the multipole expansion.
    EM_field :
        The type of electromagnetic field (electric or magnetic).
    method :
        The method to compute the polarizability.
        Options are 'Aden-Kerker' for the Aden-Kerker approximation or 'Clausius-Mossotti' for the quasistatic approximation.

    Returns
    -------
    complex|ArrayLike
        The polarizability of the core-shell particle using the specified method.
    """

    wave_number = 2 * pi / wavelength_nm
    frequency_eV =  nm_to_eV(wavelength_nm)
    core_permittivity = permittivity_ridx(frequency_eV, material_core)
    shell_permittivity = permittivity_ridx(frequency_eV, material_shell)
    
    try:
        polarizability_func = _CORE_SHELL_METHODS[(EM_field, order)][method]
    except KeyError:
        valid_methods = _CORE_SHELL_METHODS.get((EM_field, order), {})
        raise ValueError(f"Invalid method '{method}' for {EM_field} multipole order {order}. Valid methods are: {list(valid_methods.keys())}")

    return polarizability_func(radius_core_nm, radius_shell_nm, medium_permittivity, core_permittivity, shell_permittivity, wave_number)

