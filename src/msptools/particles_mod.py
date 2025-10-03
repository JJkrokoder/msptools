from .particle_types import ParticleType, SphereType
from typing import List
import numpy as np


class ParticleData:
    """Class representing a system of particles."""

    def __init__(self, types: ParticleType | List[ParticleType]) -> None:
        """
        Initialize a ParticleData object by specifying the types of particles in the system.

        Parameters
        ----------
        types :
            ParticleType instances for each particle.
            If a single ParticleType is provided, it is used for all particles.
            If a list of ParticleType instances is provided, the system is supposed to be multi-type.
            If no types are provided, default spherical particles are used.
        """

        if types is None:
            raise ValueError("Particle type must be specified when adding particles.")
        elif isinstance(types, list):
            self.types = types
        else:
            self.types = [types]

        self.positions = []
        self.type_assignments = []
        self.polarizabilities = []


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
            if len(self.types) == 0:
                raise ValueError("At least one particle type must be specified before adding particles.")
            elif len(self.types) == 1:
                type_index = 0
            else:
                raise ValueError("Particle type must be specified when adding particles to a multi-type system.")
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


    def _calculate_polarizabilities(self, frequency: float) -> None:
        """
        Calculate the polarizabilities of all particles in the system at a given frequency.

        Parameters
        ----------
        frequency :
            The frequency at which to calculate the polarizabilities.
        """

        for particle in range(len(self.positions)):
            type_index = self.type_assignments[particle]
            particle_type = self.types[type_index]
            polarizability = particle_type.polarizability
            self.polarizabilities.append(polarizability)


