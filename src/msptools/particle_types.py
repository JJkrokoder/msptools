from .polarizability_mod import calculate_sphere_polarizability
from typing import List, Tuple, Self
import numpy as np

class ParticleType:
    """Class representing a type of particle with specific properties."""
    def __init__(self) -> None:
        self.polarizability = None

    def compute_polarizability(self, frequency: float, medium_permittivity: float) -> None:
        """Compute the polarizability of the particle type at a given frequency."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class SphereType(ParticleType):
    """Class representing spherical particles."""

    def __init__(self, material: str, radius: float = 1.0):
        super().__init__()
        self.radius = radius
        self.material = material

    def compute_polarizability(self, frequency: float, medium_permittivity: float) -> None:
        """Compute the polarizability of the spherical particle at a given frequency."""
        