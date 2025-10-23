from .polarizability_mod import *
from typing import List, Tuple, Self, Callable
import numpy as np

class ParticleType:
    """Class representing a type of particle with specific properties."""
    def __init__(self, polarizability: float = None) -> None:
        if polarizability is not None:
            self.compute_polarizability = lambda frequency, medium_permittivity: polarizability

    def compute_polarizability(self, frequency: float, medium_permittivity: float) -> complex:
        """Compute the polarizability of the particle type at a given frequency."""
        raise NotImplementedError("This method should be implemented by subclasses.")

class SphereType(ParticleType):
    """Class representing spherical particles."""

    def __init__(self, material: str, radius: float = 1.0, polarizability: float = None) -> None:
        super().__init__(polarizability=polarizability)
        self.radius = radius
        self.material = material
    
    def compute_polarizability(self, frequency: float, medium_permittivity: float) -> complex:
        """PLACEHOLDER: Compute the polarizability of the spherical particle."""
        return 1.0 + 0.0j

