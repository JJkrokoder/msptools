from .polarizability_mod import calculate_sphere_polarizability
from typing import List, Tuple, Self
import numpy as np

class ParticleType:
    """Class representing a type of particle with specific properties."""


class SphereType(ParticleType):
    """Class representing spherical particles."""

    def __init__(self, radius: float = 1.0, material: str = "default"):
        super().__init__()
        self.radius = radius
        self.material = material
