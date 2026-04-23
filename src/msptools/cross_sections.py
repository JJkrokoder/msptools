from numpy.typing import ArrayLike
from .backend import get_backend
from scipy.constants import pi
from .tools.mie_theory import tE_n_coefficient, tM_n_coefficient


def sphere_cross_section_from_Mie_coeffs(medium_wavelength : ArrayLike,
                                        mie_coeffs_tE : ArrayLike,
                                        mie_coeffs_tM : ArrayLike | None) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Compute the extinction, scattering, and absorption cross-sections of a sphere from its Mie coefficients.

    Parameters
    ----------
    medium_wavelength : ArrayLike
        Wavelength of light in the surrounding medium (nm).
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
    n_E = xp.arange(1, N_electric + 1)
    n_M = xp.arange(1, N_magnetic + 1)
    
    C_ext = (medium_wavelength / (2 * pi)) ** 2 * xp.sum((2 * n_E + 1) * mie_coeffs_tE.imag) + xp.sum((2 * n_M + 1) * mie_coeffs_tM.imag)
    
    C_scat = (medium_wavelength / (2 * pi)) ** 2 * xp.sum((2 * n_E + 1) * xp.abs(mie_coeffs_tE) ** 2) + xp.sum((2 * n_M + 1) * xp.abs(mie_coeffs_tM) ** 2)
    
    C_abs = C_ext - C_scat
     
    return C_ext, C_scat, C_abs

def sphere_cross_section(wavelength_nm : ArrayLike,
                        radius_nm : ArrayLike,
                        medium_permittivity : ArrayLike,
                        particle_permittivity : ArrayLike) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Compute the extinction, scattering, and absorption cross-sections of a sphere using Mie theory.

    Parameters
    ----------
    wavelength_nm : ArrayLike
        Wavelength of light in the surrounding medium (nm).
    radius_nm : ArrayLike
        Radius of the sphere (nm).
    medium_permittivity : ArrayLike
        Permittivity of the surrounding medium.
    particle_permittivity : ArrayLike
        Permittivity of the sphere.

    Returns
    -------
    tuple[ArrayLike, ArrayLike, ArrayLike]
        A tuple containing the extinction, scattering, and absorption cross-sections (nm²).
    """
    
    tE_n, tM_n = mie_coefficients(wavelength_nm, radius_nm, medium_permittivity, particle_permittivity)
    
    return sphere_cross_section_from_Mie_coeffs(wavelength_nm, tE_n, tM_n)
     
     
     
     





