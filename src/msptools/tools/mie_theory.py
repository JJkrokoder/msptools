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


def psi_n(n: int, z: complex, derivative: bool = False) -> complex:
    """Riccati-Bessel function psi_n(z) = z * j_n(z)."""
    return (sph_jn(n, z) + z * sph_jn(n, z, derivative=True))/z

def xi_n(n: int, z: complex) -> complex:
    """Riccati-Hankel function xi_n(z) = z * h_n^(1)(z)."""
    return (hankel_2nd_kind(n, z) + z * hankel_2nd_kind(n, z, derivative=True))/z

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
    m1 = (eps_core / eps_m) ** 0.5
    m2 = (eps_shell / eps_m) ** 0.5
    
    n_1 = eps_core ** 0.5
    n_2 = eps_shell ** 0.5
    n_m = eps_m ** 0.5

    y1 = m1 * x_core
    y2 = m2 * x_shell

    # More standard Aden-Kerker matching uses the coefficient below:
    A_n =   eps_shell * psi_n(n, y1) * (
                sph_jn(n, y2) * hankel_2nd_kind(n, m2 * x_core) - sph_jn(n, m2 * x_core) * hankel_2nd_kind(n, y2)
        ) + n_1 * n_2 * sph_jn(n, y1) * (
                psi_n(n, m2 * x_core) * hankel_2nd_kind(n, y2) - xi_n(n, m2 * x_core) * sph_jn(n, y2)
        )
    
    B_n =   n_2 * psi_n(n, y1) * (
                sph_jn(n, m2 * x_core) * xi_n(n, y2) - hankel_2nd_kind(n, m2 * x_core) * psi_n(n, y2)
        ) + n_1 * sph_jn(n, y1) * (
                psi_n(n, y2) * xi_n(n, m2 * x_core) - psi_n(n, m2 * x_core) * xi_n(n, y2)
        )
        
    C_n =   eps_shell * sph_jn(n, y1) * (
                psi_n(n, y2) * xi_n(n, m2 * x_core) - psi_n(n, m2 * x_core) * xi_n(n, y2)
        ) + n_1 * n_2 * psi_n(n, y1) * (
                sph_jn(n, m2 * x_core) * xi_n(n, y2) - hankel_2nd_kind(n, m2 * x_core) * psi_n(n, y2)
        )
    
    D_n =   n_2 * sph_jn(n, y1) * (
                psi_n(n, m2 * x_core) * hankel_2nd_kind(n, y2) - sph_jn(n, y2) * xi_n(n, m2 * x_core)
        ) + n_1 * sph_jn(n, y1) * (
                sph_jn(n, y2) * hankel_2nd_kind(n, m2 * x_core) - sph_jn(n, m2 * x_core) * hankel_2nd_kind(n, y2)
        )
        
    numerator = A_n * psi_n(n, x_shell) + n_m * sph_jn(n, x_shell) * B_n
    denominator = A_n * xi_n(n, x_shell) + n_m * hankel_2nd_kind(n, x_shell) * B_n
    
    # numerator = C_n * sph_jn(n, x_shell) + eps_m * psi_n(n, x_shell) * D_n
    # denominator = C_n * hankel_2nd_kind(n, x_shell) + eps_m * xi_n(n, x_shell) * D_n

    return  1j * numerator / denominator

    