from .polarizability_mod import *
from .unit_calcs import *
from .permittivity import permittivity_ridx
from typing import List, Tuple, Self, Callable
import numpy as np

class ParticleType:
    """Class representing a type of particle with specific properties."""

    def compute_polarizability(self, frequency: float, medium_permittivity: float) -> complex:
        """Compute the polarizability of the particle type at a given frequency."""
        raise NotImplementedError("This method should be implemented by subclasses.")

class SphereType(ParticleType):
    """Class representing spherical particles."""

    def __init__(self, material: str, radius: float = 1.0, polarizability: float = None) -> None:
        self.radius = radius
        self.material = material
        if polarizability is not None:
            self.compute_polarizability = lambda frequency, medium_permittivity: polarizability

    def select_computation_method(self, frequency: float) -> Callable[[float, str], float]:
        ... # Implementation of method selection based on material and frequency

    def compute_polarizability(self, frequency: float, medium_permittivity: float) -> complex:
        return CM_with_Radiative_Correction(radius=self.radius,
                                             medium_permittivity=medium_permittivity,
                                             particle_permittivity=permittivity_ridx(frequency, self.material),
                                             wave_number=frequency_to_wavenumber_um(frequency)/1000)
    
    def compute_polarizability_CM(self, frequency: float, medium_permittivity: float) -> float:
        return Clausius_Mossotti(radius=self.radius,
                                 medium_permittivity=medium_permittivity,
                                 particle_permittivity=permittivity_ridx(frequency, self.material))

