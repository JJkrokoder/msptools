from scipy.special import spherical_jn as sph_jn
from scipy.special import spherical_yn as sph_yn
from scipy.constants import pi
from numpy.typing import ArrayLike
from msptools.backend import get_backend
import numpy as np

def hankel_plus(n: int, x: float, derivative: bool = False) -> complex:
    """Compute the spherical Hankel function of the first kind."""
    return sph_jn(n, x, derivative) + 1j * sph_yn(n, x, derivative)

def hankel_1st_kind(n: int, x: float, derivative: bool = False) -> complex:
    """Compute the spherical Hankel function of the first kind."""
    return sph_jn(n, x, derivative) + 1j * sph_yn(n, x, derivative)

def hankel_2nd_kind(n: int, x: float, derivative: bool = False) -> complex:
    """Compute the spherical Hankel function of the second kind."""
    return sph_jn(n, x, derivative) - 1j * sph_yn(n, x, derivative)

def tE_n_coefficient(n: int, x_m: float, m: complex) -> complex:
    """
    Compute the Mie coefficient tE_n for an spherical particle.
    
    Parameters
    ----------
    n : int
        The order of the Mie coefficient.
    x_m : float
        The size parameter of the core (k_m * radius_core).
    m : complex
        The ratio of the refractive indices of the particle and surrounding medium.
    
    Returns
    -------
    complex
        The Mie coefficient tE_n for the core-shell particle.
    """
    
    x_p = x_m * m
    
    t11 = m**2 * sph_jn(n,x_p) * (sph_jn(n,x_m) + x_m * sph_jn(n,x_m,derivative=True))
    t12 = sph_jn(n,x_m) * (sph_jn(n,x_p) + x_p * sph_jn(n,x_p,derivative=True))
    t21 = m**2 * sph_jn(n,x_p) * (hankel_1st_kind(n,x_m) + x_m * hankel_1st_kind(n,x_m,derivative=True))
    t22 = hankel_1st_kind(n,x_m) * (sph_jn(n,x_p) + x_p * sph_jn(n,x_p,derivative=True))

    tEn = (t11 - t12) / (t21 - t22) * 1j
    
    return tEn

def tM_n_coefficient(n: int, x_m: float, m: complex) -> complex:
    """
    Compute the Mie coefficient tM_n for an spherical particle.
    
    Parameters
    ----------
    n : int
        The order of the Mie coefficient.
    x_m : float
        The size parameter of the core (k * radius_core).
    m : complex
        The ratio of the refractive indices of the particle and surrounding medium.
    
    Returns
    -------
    complex
        The Mie coefficient tM_n for the core-shell particle.
    """
    
    x_p = x_m * m
    
    t11 = x_m * sph_jn(n,x_p) * sph_jn(n,x_m,derivative=True)
    t12 = x_p * sph_jn(n,x_m) * sph_jn(n,x_p,derivative=True)
    t21 = x_m * sph_jn(n,x_p) * hankel_plus(n,x_m,derivative=True)
    t22 = x_p * hankel_plus(n,x_m) * sph_jn(n,x_p,derivative=True)

    tMn = (t11 - t12) / (t21 - t22) * 1j
    
    return tMn


def eta1_n(n: int, z: complex) -> complex:
    return (sph_jn(n, z) + z * sph_jn(n, z, derivative=True))/z

def eta2_n(n: int, z: complex) -> complex:
    return (hankel_1st_kind(n, z) + z * hankel_1st_kind(n, z, derivative=True))/z

def aden_kerker_An(n: int, x_core: float, x_shell: float, m_1: complex, m_2: complex) -> complex:
    """
    Compute the Aden-Kerker coefficient A_n for a coated sphere.
    
    Parameters
    ----------
    n :
        The order of the coefficient.
    x_core :
        The size parameter of the core (k_m * radius_core).
    x_shell :
        The size parameter of the outer shell radius (k_m * radius_shell).
    m_1 : complex
        The ratio of the refractive indices of the core and surrounding medium.
    m_2 : complex
        The ratio of the refractive indices of the shell and surrounding medium.
    
    Returns
    -------
    complex
        The Aden-Kerker coefficient A_n for the coated sphere.
    """
    
    A_n =   m_2**2 * eta1_n(n, m_1 * x_core) * (
                sph_jn(n, m_2 * x_shell) * hankel_1st_kind(n, m_2 * x_core) - sph_jn(n, m_2 * x_core) * hankel_1st_kind(n, m_2 * x_shell)
        ) + m_1 * m_2 * sph_jn(n, m_1 * x_core) * (
                eta1_n(n, m_2 * x_core) * hankel_1st_kind(n, m_2 * x_shell) - eta2_n(n, m_2 * x_core) * sph_jn(n, m_2 * x_shell)
        )
    
    return A_n

def aden_kerker_Bn(n: int, x_core: float, x_shell: float, m_1: complex, m_2: complex) -> complex:
    """
    Compute the Aden-Kerker coefficient B_n for a coated sphere.
    
    Parameters
    ----------
    n : int
        The order of the coefficient.
    x_core : float
        The size parameter of the core (k_m * radius_core).
    x_shell : float
        The size parameter of the outer shell radius (k_m * radius_shell).
    m_1 : complex
        The ratio of the refractive indices of the core and surrounding medium.
    m_2 : complex
        The ratio of the refractive indices of the shell and surrounding medium.
    
    Returns
    -------
    complex
        The Aden-Kerker coefficient B_n for the coated sphere.
    """

    
    B_n =   m_2 * eta1_n(n, m_1 * x_core) * (
                sph_jn(n, m_2 * x_core) * eta2_n(n, m_2 * x_shell) - hankel_1st_kind(n, m_2 * x_core) * eta1_n(n, m_2 * x_shell)
        ) + m_1 * sph_jn(n, m_1 * x_core) * (
                eta1_n(n, m_2 * x_shell) * eta2_n(n, m_2 * x_core) - eta1_n(n, m_2 * x_core) * eta2_n(n, m_2 * x_shell)
        )
    
    return B_n

def aden_kerker_Cn(n: int, x_core: float, x_shell: float, m_1: complex, m_2: complex) -> complex:
    """
    Compute the Aden-Kerker coefficient C_n for a coated sphere.
    
    Parameters
    ----------
    n : int
        The order of the coefficient.
    x_core : float
        The size parameter of the core (k_m * radius_core).
    x_shell : float
        The size parameter of the outer shell radius (k_m * radius_shell).
    m_1 : complex
        The ratio of the refractive indices of the core and surrounding medium.
    m_2 : complex
        The ratio of the refractive indices of the shell and surrounding medium.

    Returns
    -------
    complex
        The Aden-Kerker coefficient C_n for the coated sphere.
    """
    
    C_n =   m_2**2 * sph_jn(n, m_1 * x_core) * (
                eta1_n(n, m_2 * x_shell) * eta2_n(n, m_2 * x_core) - eta1_n(n, m_2 * x_core) * eta2_n(n, m_2 * x_shell)
        ) + m_1 * m_2 * eta1_n(n, m_1 * x_core) * (
                sph_jn(n, m_2 * x_core) * eta2_n(n, m_2 * x_shell) - hankel_1st_kind(n, m_2 * x_core) * eta1_n(n, m_2 * x_shell)
        )
    
    return C_n

def aden_kerker_Dn(n: int, x_core: float, x_shell: float, m_1: complex, m_2: complex) -> complex:
    """
    Compute the Aden-Kerker coefficient D_n for a coated sphere.
    
    Parameters
    ----------
    n : int
        The order of the coefficient.
    x_core : float
        The size parameter of the core (k_m * radius_core).
    x_shell : float
        The size parameter of the outer shell radius (k_m * radius_shell).
    m_1 : complex
        The ratio of the refractive indices of the core and surrounding medium.
    m_2 : complex
        The ratio of the refractive indices of the shell and surrounding medium.
    
    Returns
    -------
    complex
        The Aden-Kerker coefficient D_n for the coated sphere.
    """
    
    D_n =   m_2* sph_jn(n, m_1 * x_core) * (
                eta1_n(n, m_2 * x_core) * hankel_1st_kind(n, m_2 * x_shell) - sph_jn(n, m_2 * x_shell) * eta2_n(n, m_2 * x_core)
        ) + m_1* eta1_n(n, m_1 * x_core) * (
                sph_jn(n, m_2 * x_shell) * hankel_1st_kind(n, m_2 * x_core) - sph_jn(n, m_2 * x_core) * hankel_1st_kind(n, m_2 * x_shell)
        )
    
    return D_n

def tEn_aden_kerker_coefficient(
    n: int,
    x_core: float,
    x_shell: float,
    m_1: complex,
    m_2: complex,
) -> complex:
    """
    Compute the electric Mie coefficient tE_n for a coated sphere using an Aden-Kerker form.

    Parameters
    ----------
    n : int
        Order of the coefficient.
    x_core : float
        Size parameter of the core, k_m * a.
    x_shell : float
        Size parameter of the outer shell radius, k_m * b.
    m_1 : complex
        The ratio of the refractive indices of the core and surrounding medium.
    m_2 : complex
        The ratio of the refractive indices of the shell and surrounding medium.

    Returns
    -------
    complex
        Electric Mie coefficient for the coated sphere.
    """

    # More standard Aden-Kerker matching uses the coefficient below:
    A_n = aden_kerker_An(n, x_core, x_shell, m_1, m_2)
    
    B_n = aden_kerker_Bn(n, x_core, x_shell, m_1, m_2)
        
    numerator = A_n * eta1_n(n, x_shell) + sph_jn(n, x_shell) * B_n
    denominator = A_n * eta2_n(n, x_shell) + hankel_1st_kind(n, x_shell) * B_n
    
    return  1j * numerator / denominator

def tMn_aden_kerker_coefficient(
    n: int,
    x_core: float,
    x_shell: float,
    m_1: complex,
    m_2: complex
) -> complex:
    """
    Compute the magnetic Mie coefficient tM_n for a coated sphere using an Aden-Kerker form.

    Parameters
    ----------
    n : int
        Order of the coefficient.
    x_core : float
        Size parameter of the core, k_m * a.
    x_shell : float
        Size parameter of the outer shell radius, k * b.
    m_1 : complex
        The ratio of the refractive indices of the core and surrounding medium.
    m_2 : complex
        The ratio of the refractive indices of the shell and surrounding medium.

    Returns
    -------
    complex
        Magnetic Mie coefficient for the coated sphere.
    """

        
    C_n = aden_kerker_Cn(n, x_core, x_shell, m_1, m_2)

    D_n = aden_kerker_Dn(n, x_core, x_shell, m_1, m_2)
        
    numerator = C_n * sph_jn(n, x_shell) + eta1_n(n, x_shell) * D_n
    denominator = C_n * hankel_1st_kind(n, x_shell) + eta2_n(n, x_shell) * D_n

    return 1j * numerator / denominator

def select_multipole_orders_sphere(size_parameter: ArrayLike, m: ArrayLike) -> tuple[int, int]:
    """
    Select the maximum multipole orders to include in the Mie coefficient calculations based on the size parameter.

    Parameters
    ----------
    size_parameter : ArrayLike
        Size parameter of the sphere, k_m * a.
    m : ArrayLike
        Relative refractive index of the particle to the medium (m = n_particle / n_medium).

    Returns
    -------
    tuple
        A tuple containing two arrays: the first with the multipole orders for electric coefficients and the second for magnetic coefficients.
    """
    if np.isscalar(size_parameter):
        xp = np
    else:
        xp = get_backend(size_parameter)
    
    tE1 = tE_n_coefficient(1, size_parameter, m)
    
    N_electric = 1
    N_magnetic = 0
    
    last_tE = False
    last_tM = False
    
    while not last_tE or not last_tM:
        tE_next = tE_n_coefficient(N_electric + 1, size_parameter, m)
        tM_next = tM_n_coefficient(N_magnetic + 1, size_parameter, m)
        
        if (2 * N_electric + 3) * xp.max(tE_next.imag) / (3 * xp.max(tE1.imag)) > 1e-3:
            N_electric += 1
        else:
            last_tE = True
        
        if (2 * N_magnetic + 3) * xp.max(tM_next.imag) / (3 * xp.max(tE1.imag)) > 1e-3:
            N_magnetic += 1
        else:
            last_tM = True
    
    return N_electric, N_magnetic

def select_multipole_orders_core_shell_sphere(size_parameter_core: ArrayLike, size_parameter_shell: ArrayLike, m_1: ArrayLike, m_2: ArrayLike, tol: float = 1e-3) -> tuple[int, int]:
    """
    Select the maximum multipole orders to include in the Mie coefficient calculations for a core-shell sphere based on the size parameters and refractive index ratios.

    Parameters
    ----------
    size_parameter_core : ArrayLike
        Size parameter of the core, k_m * a.
    size_parameter_shell : ArrayLike
        Size parameter of the outer shell radius, k_m * b.
    m_1 : ArrayLike
        Relative refractive index of the core to the medium (m_1 = n_core / n_medium).
    m_2 : ArrayLike
        Relative refractive index of the shell to the medium (m_2 = n_shell / n_medium).
    tol : float, optional
        Tolerance for the convergence criterion.

    Returns
    -------
    tuple
        A tuple containing two arrays: the first with the multipole orders for electric coefficients and the second for magnetic coefficients.
    
    """
    
    if np.isscalar(size_parameter_core):
        xp = np
    else:
        xp = get_backend(size_parameter_core)
    
    tE1 = tEn_aden_kerker_coefficient(1, size_parameter_core, size_parameter_shell, m_1, m_2)
    
    N_electric = 1
    N_magnetic = 0
    
    last_tE = False
    last_tM = False
    
    while not last_tE or not last_tM:
        tE_next = tEn_aden_kerker_coefficient(N_electric + 1, size_parameter_core, size_parameter_shell, m_1, m_2)
        tM_next = tMn_aden_kerker_coefficient(N_magnetic + 1, size_parameter_core, size_parameter_shell, m_1, m_2)
        
        if (2 * N_electric + 3) * xp.max(tE_next.imag) / (3 * xp.max(tE1.imag)) > tol:
            N_electric += 1
        else:
            last_tE = True
        
        if (2 * N_magnetic + 3) * xp.max(tM_next.imag) / (3 * xp.max(tE1.imag)) > tol:
            N_magnetic += 1
        else:
            last_tM = True
    
    return N_electric, N_magnetic

    