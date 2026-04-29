from numpy.typing import ArrayLike
from .backend import get_backend
from scipy.constants import pi
from .tools.mie_theory import tE_n_coefficient, tM_n_coefficient, select_multipole_orders_sphere
from .permittivity import permittivity_ridx
from .tools.unit_calcs import nm_to_eV
import numpy as np


def sphere_cross_section_from_Mie_coeffs(size_parameter : ArrayLike,
                                        mie_coeffs_tE : ArrayLike,
                                        mie_coeffs_tM : ArrayLike | None) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Compute the extinction, scattering, and absorption cross-sections of a sphere from its Mie coefficients.

    Parameters
    ----------
    size_parameter : ArrayLike
        The size parameter of the sphere (2 * pi * radius / medium_wavelength).
    mie_coeffs_tE : ArrayLike
        Set of electric multipolar Mie coefficients (ordered as tE1, tE2, tE3, ...)
    mie_coeffs_tM : ArrayLike
        Set of magnetic multipolar Mie coefficients (ordered as tM1, tM2, tM3, ...)

    Returns
    -------
    tuple[ArrayLike, ArrayLike, ArrayLike]
        A tuple containing the extinction, scattering, and absorption cross-sections (nm²).
    """
    
    xp = get_backend(mie_coeffs_tE)
    
    if mie_coeffs_tM is None:
        mie_coeffs_tM = xp.zeros_like(mie_coeffs_tE)
    
    N_electric = mie_coeffs_tE.shape[0]
    N_magnetic = mie_coeffs_tM.shape[0]
    n_E = xp.arange(1, N_electric + 1).reshape((-1,) + (1,) * (mie_coeffs_tE.ndim - 1))
    n_M = xp.arange(1, N_magnetic + 1).reshape((-1,) + (1,) * (mie_coeffs_tM.ndim - 1))
    
    
    C_ext = 2 / (size_parameter ** 2) * (xp.sum((2 * n_E + 1) * mie_coeffs_tE.imag, axis=0) + xp.sum((2 * n_M + 1) * mie_coeffs_tM.imag, axis=0))
    
    C_scat = 2 / (size_parameter ** 2) * (xp.sum((2 * n_E + 1) * abs(mie_coeffs_tE)**2, axis=0) + xp.sum((2 * n_M + 1) * abs(mie_coeffs_tM)**2, axis=0))
    
    C_abs = C_ext - C_scat
     
    return C_ext, C_scat, C_abs

def sphere_cross_section(wavelength_nm : ArrayLike,
                        radius_nm : ArrayLike,
                        medium_permittivity : ArrayLike,
                        particle_material : str) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Compute the extinction, scattering, and absorption cross-sections of a sphere using Mie theory.

    Parameters
    ----------
    wavelength_nm :
        Wavelength of light in vacuum (nm).
    radius_nm :
        Radius of the sphere (nm).
    medium_permittivity :
        Permittivity of the surrounding medium.
    particle_material :
        Material of the sphere.

    Returns
    -------
    C_ext, C_sca, C_abs :
        The extinction, scattering, and absorption cross-sections (nm²).
    """
    
    if not np.isscalar(wavelength_nm):
        xp = get_backend(wavelength_nm)
    if not np.isscalar(radius_nm):
        xp = get_backend(radius_nm)
    if not np.isscalar(medium_permittivity):
        xp = get_backend(medium_permittivity)
    else:
        xp = np
    
    particle_permittivity = permittivity_ridx(nm_to_eV(wavelength_nm), particle_material)
    
    m = particle_permittivity**0.5 / medium_permittivity**0.5
    
    x = 2 * pi * radius_nm / wavelength_nm * medium_permittivity**0.5
    
    N_electric, N_magnetic = select_multipole_orders_sphere(x, m)
    
    tE_n = xp.array([tE_n_coefficient(n, x, m) for n in range(1, N_electric + 1)])
    tM_n = xp.array([tM_n_coefficient(n, x, m) for n in range(1, N_magnetic + 1)])
    
    return sphere_cross_section_from_Mie_coeffs(x, tE_n, tM_n)
     
     
     
     





