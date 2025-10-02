from .OFO_calculations import *
from .dipole_moments import *
from .polarizability_mod import *
from .particle_types import *
from typing import List, Self


__all__ = [
    "OFO_calculations",
    "dipole_moments",
    "polarizability_mod",
    "particle_types",
]

class Particles:
    """Class representing a system of particles."""

    def __init__(self: Self, types: ParticleType | List[ParticleType] = SphereType()) -> None:
        """
        Initialize a Particles object by specifying the types of particles in the system.

        Parameters
        ----------
        types :
            ParticleType instances for each particle.
            If a single ParticleType is provided, it is used for all particles.
            If a list of ParticleType instances is provided, the system is supposed to be multi-type.
            If no types are provided, default spherical particles are used.
        """

        self.types = types if isinstance(types, list) else [types]
            




    
