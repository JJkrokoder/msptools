from .OFO_calculations import *
from .dipole_moments import *
from .polarizability_mod import *
from .particle_types import *


__all__ = [
    "OFO_calculations",
    "dipole_moments",
    "polarizability_mod",
    "particle_types",
]

def greet():
    print("Welcome to the msptools package!")
    pass

class Particles:
    """Class representing a system of particles."""

    def __init__(self, type=None, positions=None):
        """
        Initialize the Particles system.

        Parameters
        ----------
        type :
            ParticleType instance for each particle.
        positions :
            List or array of positions for each particle (shape: (N, 3)).
        """

        # Handle types input
        if type is None:
            pass
        elif isinstance(type, ParticleType):
            self.types = [type]
        else:
            raise TypeError("type must be a ParticleType instance")

        if positions is None:
            self.positions = []
        elif isinstance(positions, np.ndarray) and positions.ndim == 2 and positions.shape[1] == 3:
            self.positions = positions.tolist()
        elif isinstance(positions, list) and all(isinstance(pos, (list, tuple)) and len(pos) == 3 for pos in positions):
            self.positions = positions
        else:
            raise ValueError("positions must be a list of 3D coordinates or a (N, 3) numpy array.")
    
        if type and positions:
            self.properties = type.properties * len(positions)     

        

    def add_particles(self, particle_positions: np.ndarray, type: ParticleType = SphereType()):
        """
        Add particles to a ParticleSystem object. The particles are defined by their type and positions.
        If no type is provided, default spherical particles are added.

        Parameters
        ----------
        particle_positions :
            An array of shape (N, 3) representing the positions of N particles in 3D space.
        type :
            An instance of ParticleType defining the properties of the particles to be added.

        """
        items = type.properties.items()
        for item in items:
            self.properties[item[0]] = [item[1]] * particle_positions.shape[0]
        self.positions.extend(particle_positions)




    
