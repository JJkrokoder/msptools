from .OFO_calculations import *
from .dipole_moments import *
from .polarizability_mod import *
from .particle_types import *
from typing import List


__all__ = [
    "OFO_calculations",
    "dipole_moments",
    "polarizability_mod",
    "particle_types",
]


class Particles:
    """Class representing a system of particles."""

    def __init__(self, types: ParticleType | List[ParticleType] | None = None) -> None:
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

        if types is None:
            default_type = SphereType()
            self.types = [default_type]
        elif isinstance(types, ParticleType):
            self.types = [types]
        else:
            self.types = types
        
        self.positions = []
        self.type_assignments = []


    def add_particles(self,
                     positions: np.ndarray | List[float] | List[List[float]],
                     type: ParticleType | None = None) -> None:
        """
        Add particles to the system at specified positions.

        Parameters
        ----------
        positions :
            The position of the particles to add. This can be a 1D-three-element or 2D array-like.
        type :
            The type of the particles to add.

        Notes
        -----
        The positions are stored as 1D numpy arrays.
        """

        if type is None:
            if len(self.types) == 1:
                type = self.types[0]
                type_index = 0
            else:
                raise ValueError("For multi-type systems, the types of the particles must be specified.")
        else:
            if type not in self.types:
                self.types.append(type)
                type_index = len(self.types) - 1
            else:
                type_index = self.types.index(type)

        positions = np.array(positions)
        if positions.ndim == 1 or (positions.ndim == 2 and positions.shape[0] == 1):
            self.positions.append(positions) if positions.ndim == 1 else self.positions.append(positions[0])
            self.type_assignments.append(type_index)
        elif positions.ndim == 2 and positions.shape[0] > 1:
            self.positions.extend(positions)
            self.type_assignments.extend([type_index] * positions.shape[0])
        else:
            raise ValueError("Positions must be a 1D-three-element or 2D array-like.")
        

    def get_positions(self) -> np.ndarray:
        """
        Get the positions of all particles in the system. If there is only one particle, returns a 1D array.

        Returns
        -------
        np.ndarray
            An array of shape (N, 3) where N is the number of particles.
        """

        positions = np.array(self.positions)

        if positions.shape[0] == 1:
            return positions[0]
        
        return positions
    
    def get_position(self, index: int) -> np.ndarray:
        """
        Get the position of a specific particle by its index.

        Parameters
        ----------
        index :
            The index of the particle whose position is to be retrieved.

        Returns
        -------
        np.ndarray
            A 1D array representing the position of the specified particle.
        """

        return self.positions[index]
    
    def clean_positions(self) -> None:
        """
        Remove all particles' positions from the system.
        """

        self.positions = []

