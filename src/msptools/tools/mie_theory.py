from scipy.special import spherical_jn as sph_jn
from scipy.special import spherical_yn as sph_yn

def hankel_plus(n: int, x: float, derivative: bool = False) -> complex:
    """Compute the spherical Hankel function of the first kind."""
    return sph_jn(n, x, derivative) + 1j * sph_yn(n, x, derivative)

def hankel_2nd_kind(n: int, x: float, derivative: bool = False) -> complex:
    """Compute the spherical Hankel function of the second kind."""
    return sph_jn(n, x, derivative) + 1j * sph_yn(n, x, derivative)

def tE_n_coefficient(n: int, x: float, eps_p: complex, eps_m: complex) -> complex:
    """
    Compute the Mie coefficient tE_n for an spherical particle.
    
    Parameters
    ----------
    n : int
        The order of the Mie coefficient.
    x : float
        The size parameter of the core (k * radius_core).
    eps_p : complex
        The permittivity of the particle material.
    eps_m : complex
        The permittivity of the surrounding medium.
    
    Returns
    -------
    complex
        The Mie coefficient tE_n for the core-shell particle.
    """
    
    x_p = x * eps_p**0.5
    x_m = x * eps_m**0.5
    
    t11 = eps_p * sph_jn(n,x_p) * (sph_jn(n,x_m) + x_m * sph_jn(n,x_m,derivative=True))
    t12 = eps_m * sph_jn(n,x_m) * (sph_jn(n,x_p) + x_p * sph_jn(n,x_p,derivative=True))
    t21 = eps_m * hankel_plus(n,x_m) * (sph_jn(n,x_p) + x_p * sph_jn(n,x_p,derivative=True))
    t22 = eps_p * sph_jn(n,x_p) * (hankel_plus(n,x_m) + x_m * hankel_plus(n,x_m,derivative=True))

    tEn = -(t11 - t12) / (t21 - t22) * 1j
    
    return tEn

def tM_n_coefficient(n: int, x: float, eps_p: complex, eps_m: complex) -> complex:
    """
    Compute the Mie coefficient tM_n for an spherical particle.
    
    Parameters
    ----------
    n : int
        The order of the Mie coefficient.
    x : float
        The size parameter of the core (k * radius_core).
    eps_p : complex
        The permittivity of the particle material.
    eps_m : complex
        The permittivity of the surrounding medium.
    
    Returns
    -------
    complex
        The Mie coefficient tM_n for the core-shell particle.
    """
    
    x_p = x * eps_p**0.5
    x_m = x * eps_m**0.5
    
    t11 = x_m * sph_jn(n,x_p) * sph_jn(n,x_m,derivative=True)
    t12 = x_p * sph_jn(n,x_m) * sph_jn(n,x_p,derivative=True)
    t21 = x_p *hankel_plus(n,x_m) * sph_jn(n,x_p,derivative=True)
    t22 = x_m * sph_jn(n,x_p) * hankel_plus(n,x_m,derivative=True)

    tMn = -(t11 - t12) / (t21 - t22) * 1j
    
    return tMn


def psi_n(n: int, z: complex, derivative: bool = False) -> complex:
    """Riccati-Bessel function psi_n(z) = z * j_n(z)."""
    return (sph_jn(n, z) + z * sph_jn(n, z, derivative=True))/z

def xi_n(n: int, z: complex) -> complex:
    """Riccati-Hankel function xi_n(z) = z * h_n^(1)(z)."""
    return (hankel_2nd_kind(n, z) + z * hankel_2nd_kind(n, z, derivative=True))/z

def aden_kerker_An(n: int, x_core: float, x_shell: float, eps_core: complex, eps_shell: complex, eps_m: complex) -> complex:
    """
    Compute the Aden-Kerker coefficient A_n for a coated sphere.
    
    Parameters
    ----------
    n : int
        The order of the coefficient.
    x_core : float
        The size parameter of the core (k_m * radius_core).
    x_shell : float
        The size parameter of the outer shell radius (k_m * radius_shell).
    eps_core : complex
        The permittivity of the core material.
    eps_shell : complex
        The permittivity of the shell material.
    eps_m : complex
        The permittivity of the surrounding medium.
    
    Returns
    -------
    complex
        The Aden-Kerker coefficient A_n for the coated sphere.
    """
    
    m1 = (eps_core / eps_m) ** 0.5
    m2 = (eps_shell / eps_m) ** 0.5

    y1 = m1 * x_core
    y2 = m2 * x_shell
    
    A_n =   eps_shell * psi_n(n, y1) * (
                sph_jn(n, y2) * hankel_2nd_kind(n, m2 * x_core) - sph_jn(n, m2 * x_core) * hankel_2nd_kind(n, y2)
        ) + eps_core ** 0.5 * eps_shell **0.5 * sph_jn(n, y1) * (
                psi_n(n, m2 * x_core) * hankel_2nd_kind(n, y2) - xi_n(n, m2 * x_core) * sph_jn(n, y2)
        )
    
    return A_n

def aden_kerker_Bn(n: int, x_core: float, x_shell: float, eps_core: complex, eps_shell: complex, eps_m: complex) -> complex:
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
    eps_core : complex
        The permittivity of the core material.
    eps_shell : complex
        The permittivity of the shell material.
    eps_m : complex
        The permittivity of the surrounding medium.
    
    Returns
    -------
    complex
        The Aden-Kerker coefficient B_n for the coated sphere.
    """
    
    m1 = (eps_core / eps_m) ** 0.5
    m2 = (eps_shell / eps_m) ** 0.5

    y1 = m1 * x_core
    y2 = m2 * x_shell
    
    B_n =   eps_shell**0.5 * psi_n(n, y1) * (
                sph_jn(n, m2 * x_core) * xi_n(n, y2) - hankel_2nd_kind(n, m2 * x_core) * psi_n(n, y2)
        ) + eps_core**0.5 * sph_jn(n, y1) * (
                psi_n(n, y2) * xi_n(n, m2 * x_core) - psi_n(n, m2 * x_core) * xi_n(n, y2)
        )
    
    return B_n

def aden_kerker_Cn(n: int, x_core: float, x_shell: float, eps_core: complex, eps_shell: complex, eps_m: complex) -> complex:
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
    eps_core : complex
        The permittivity of the core material.
    eps_shell : complex
        The permittivity of the shell material.
    eps_m : complex
        The permittivity of the surrounding medium.
    
    Returns
    -------
    complex
        The Aden-Kerker coefficient C_n for the coated sphere.
    """
    
    m1 = (eps_core / eps_m) ** 0.5
    m2 = (eps_shell / eps_m) ** 0.5

    y1 = m1 * x_core
    y2 = m2 * x_shell
    
    C_n =   eps_shell * sph_jn(n, y1) * (
                psi_n(n, y2) * xi_n(n, m2 * x_core) - psi_n(n, m2 * x_core) * xi_n(n, y2)
        ) + eps_core**0.5 * eps_shell**0.5 * psi_n(n, y1) * (
                sph_jn(n, m2 * x_core) * xi_n(n, y2) - hankel_2nd_kind(n, m2 * x_core) * psi_n(n, y2)
        )
    
    return C_n

def aden_kerker_Dn(n: int, x_core: float, x_shell: float, eps_core: complex, eps_shell: complex, eps_m: complex) -> complex:
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
    eps_core : complex
        The permittivity of the core material.
    eps_shell : complex
        The permittivity of the shell material.
    eps_m : complex
        The permittivity of the surrounding medium.
    
    Returns
    -------
    complex
        The Aden-Kerker coefficient D_n for the coated sphere.
    """
    
    m1 = (eps_core / eps_m) ** 0.5
    m2 = (eps_shell / eps_m) ** 0.5

    y1 = m1 * x_core
    y2 = m2 * x_shell
    
    D_n =   eps_shell ** 0.5 * sph_jn(n, y1) * (
                psi_n(n, m2 * x_core) * hankel_2nd_kind(n, y2) - sph_jn(n, y2) * xi_n(n, m2 * x_core)
        ) + eps_core ** 0.5 * sph_jn(n, y1) * (
                sph_jn(n, y2) * hankel_2nd_kind(n, m2 * x_core) - sph_jn(n, m2 * x_core) * hankel_2nd_kind(n, y2)
        )
    
    return D_n

def tEn_aden_kerker_coefficient(
    n: int,
    x_core: float,
    x_shell: float,
    eps_core: complex,
    eps_shell: complex,
    eps_m: complex
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
    eps_core : complex
        Permittivity of the core.
    eps_shell : complex
        Permittivity of the shell.
    eps_m : complex
        Permittivity of the surrounding medium.

    Returns
    -------
    complex
        Electric Mie coefficient for the coated sphere.
    """

    # More standard Aden-Kerker matching uses the coefficient below:
    A_n = aden_kerker_An(n, x_core, x_shell, eps_core, eps_shell, eps_m)
    
    B_n = aden_kerker_Bn(n, x_core, x_shell, eps_core, eps_shell, eps_m)
        
    numerator = A_n * psi_n(n, x_shell) + eps_m**0.5 * sph_jn(n, x_shell) * B_n
    denominator = A_n * xi_n(n, x_shell) + eps_m**0.5 * hankel_2nd_kind(n, x_shell) * B_n
    
    return  1j * numerator / denominator

def tMn_aden_kerker_coefficient(
    n: int,
    x_core: float,
    x_shell: float,
    eps_core: complex,
    eps_shell: complex,
    eps_m: complex
) -> complex:
    """
    Compute the magnetic Mie coefficient tM_n for a coated sphere using an Aden-Kerker form.

    Parameters
    ----------
    n : int
        Order of the coefficient.
    x_core : float
        Size parameter of the core, k * a.
    x_shell : float
        Size parameter of the outer shell radius, k * b.
    eps_core : complex
        Permittivity of the core.
    eps_shell : complex
        Permittivity of the shell.
    eps_m : complex
        Permittivity of the surrounding medium.
        
    Returns
    -------
    complex
        Magnetic Mie coefficient for the coated sphere.
    """

    # More standard Aden-Kerker matching uses the coefficient below:
    
    x_m_core = x_core * eps_m ** 0.5
    x_m_shell = x_shell * eps_m ** 0.5
        
    C_n = aden_kerker_Cn(n, x_m_core, x_m_shell, eps_core, eps_shell, eps_m)

    D_n = aden_kerker_Dn(n, x_m_core, x_m_shell, eps_core, eps_shell, eps_m)
        
    numerator = C_n * sph_jn(n, x_m_shell) + eps_m**0.5 * psi_n(n, x_m_shell) * D_n
    denominator = C_n * hankel_2nd_kind(n, x_m_shell) + eps_m**0.5 * xi_n(n, x_m_shell) * D_n

    return 1j * numerator / denominator

    