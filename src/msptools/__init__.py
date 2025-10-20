from dataclasses import Field
from .OFO_calculations import *
from .dipole_moments import *
from .polarizability_mod import *
from .particle_types import *
from .particles_mod import *
from typing import List


__all__ = [
    "OFO_calculations",
    "dipole_moments",
    "polarizability_mod",
    "particle_types",
    "particles_mod",
    "permittivity",
]

class System:
    """Class representing a Optical_Forces physical system containing particles."""

    def __init__(self, particledata : ParticleData, field: Field, medium_permittivity: float = 1.0) -> None:
        """
        Initialize a System object by specifying the particles and the field.
        """
        self.particledata = particledata
        self.field = field
        self.medium_permittivity = medium_permittivity

        for type in self.particledata.types:
            type.compute_polarizability(frequency=self.field.frequency, medium_permittivity=self.medium_permittivity)

    



        

