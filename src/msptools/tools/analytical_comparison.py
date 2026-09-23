from scipy.constants import pi
from numpy.typing import ArrayLike
from msptools.backend import get_backend
from msptools.GreenTensor_Electric import (G_0_function, G_1_function,
                                           G_0_derivative_function,
                                           G_1_derivative_function)


def LOF_pair_spheres(
    R: float | ArrayLike,
    alpha_1: complex,
    alpha_2: complex,
    k_medium: float,
    axis: str,
    SingleScattering: bool = False
) -> complex | ArrayLike:
    """
    Computes the net LOF (Lateral Optical Force) between two Rayleigh spheres, normalized to unit field amplitude.
    
    Parameters
    ----------
    R : ArrayLike
        The distance between the two spheres. It can be a single value or an array of values.
    alpha_1 : complex
        The polarizability of the first sphere.
    alpha_2 : complex
        The polarizability of the second sphere.
    k_medium : float
        The wave number in the medium.
    axis : str
        The axis along which the force is computed. It can be 'longitudinal' or 'transverse'.
    SingleScattering : bool, optional
        If True, computes the LOF using only single scattering contributions. Default is False.
    
    Returns
    -------
    complex | ArrayLike
        The LOF between the two spheres.
    
    Notes
    -----
    The LOF is computed using the pairwise Green's tensor and its derivative. The force is calculated based on the polarizabilities of the spheres and the distance between them.
    If SingleScattering is True, only the first-order scattering terms are considered.
    """
    G_0 = G_0_function(R, k_medium) * k_medium**2
    dG_0 = G_0_derivative_function(R, k_medium) * k_medium**2
    
    if axis == 'longitudinal':
        G_1 = G_1_function(R, k_medium) * k_medium**2
        dG_1 = G_1_derivative_function(R, k_medium) * k_medium**2
        G = G_0 + G_1 * R**2
        dG = dG_0 + dG_1 * R**2 + 2 * G_1 * R
    elif axis == 'transverse':
        G = G_0
        dG = dG_0
    else:
        raise ValueError("Axis must be either 'longitudinal' or 'transverse'.")
    
    if SingleScattering:
        force = - dG.imag * (alpha_1 * alpha_2.conjugate()).imag

    else:
        denominator = abs(1 - alpha_1 * alpha_2 * G**2)**2
        numerator = alpha_1 * alpha_2.conjugate() + abs(alpha_2)**2 * alpha_1 * G - abs(alpha_1)**2 * alpha_2 * G
        force = - dG.imag * numerator.imag / denominator

    return force


    
def LOF_pair_spheres_individual(
    R: float | ArrayLike,
    alpha_1: complex,
    alpha_2: complex,
    k_medium: float,
    axis: str,
    SingleScattering: bool = False
) -> tuple[complex | ArrayLike, complex | ArrayLike]:
    """
    Computes the individual contributions to the LOF (Lateral Optical Force) between two Rayleigh spheres, normalized to unit field amplitude.
    
    Parameters
    ----------
    R : ArrayLike
        The distance between the two spheres. It can be a single value or an array of values.
    alpha_1 : complex
        The polarizability of the first sphere.
    alpha_2 : complex
        The polarizability of the second sphere.
    k_medium : float
        The wave number in the medium.
    axis : str
        The axis along which the force is computed. It can be 'longitudinal' or 'transverse'.
    
    Returns
    -------
    tuple[complex | ArrayLike, complex | ArrayLike]
        A tuple containing the individual contributions to the LOF from each sphere.
    
    Notes
    -----
    This function computes the individual contributions to the LOF based on the polarizabilities of the spheres and the distance between them. The contributions are calculated using the pairwise Green's tensor and its derivative.
    """
    G_0 = G_0_function(R, k_medium) * k_medium**2
    dG_0 = G_0_derivative_function(R, k_medium) * k_medium**2
    
    if axis == 'longitudinal':
        G_1 = G_1_function(R, k_medium) * k_medium**2
        dG_1 = G_1_derivative_function(R, k_medium) * k_medium**2
        G = G_0 + G_1 * R**2
        dG = dG_0 + dG_1 * R**2 + 2 * G_1 * R
    elif axis == 'transverse':
        G = G_0
        dG = dG_0
    else:
        raise ValueError("Axis must be either 'longitudinal' or 'transverse'.")
    
    denominator = 2*abs(1 - alpha_1 * alpha_2 * G**2)**2
    
    cross_term = alpha_1 * alpha_2.conjugate() + abs(alpha_2)**2 * alpha_1 * G + abs(alpha_1)**2 * alpha_2.conjugate() * G.conjugate() + abs(alpha_1 * alpha_2 * G)**2
    
    num1 = - dG.conjugate() * cross_term
    num2 = dG.conjugate() * cross_term.conjugate()
    
    force_1 = num1.real / denominator
    force_2 = num2.real / denominator
    
    return force_1, force_2

