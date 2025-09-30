from .OFO_calculations import *
from .dipole_moments import *
from .polarizability_mod import *


__all__ = [
    "OFO_calculations",
    "dipole_moments",
    "polarizability_mod"
]

def greet():
    print("Welcome to the msptools package!")
    pass

class ParticleType:
    """Class representing a type of particle with specific properties."""

    def __init__(self):
        self.properties = []

class SphereType(ParticleType):
    """Class representing spherical particles."""

    def __init__(self, radius: float, material: str):
        super().__init__()
        radius = radius
        polarizability = polarizability_mod.calculate_sphere_polarizability(radius, material)
        self.properties = {
            "radius": radius,
            "polarizability": polarizability
        }

class ParticleSystem:
    """Class representing a system of particles."""

    def __init__(self, medium_epsilon: float = 1.0):
        self.particles = []
        self.positions = []
        self.medium_epsilon = medium_epsilon

    def add_particles(self, type: ParticleType, positions: np.ndarray):
        """Add particles to the system."""
        self.particles.extend(type.properties*positions.shape[0])
        self.positions.extend(positions)




    
