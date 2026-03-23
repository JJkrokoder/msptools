from typing import List
from .tools.unit_calcs import get_multiplier_nanometers
from .backend import get_backend
from numpy.typing import ArrayLike


class Particles:
    """Class representing a system of particles."""

    def __init__(self, xp, dim: int = 3) -> None:
        """
        Initialize a Particles object.
        
        Parameters
        ----------
        xp :
            The array library to use (e.g., numpy or cupy).
        """
        self.xp = xp
        self.dim = dim
        self.positions = self.xp.empty((0, self.dim))
        self.polarizabilities = self.xp.empty((0, self.dim, self.dim))


    def add_particles(self,
                     positions: ArrayLike,
                     polarizabilities: ArrayLike) -> None:
        """
        Add particles to the system at specified positions and with specified polarizabilities.

        Parameters
        ----------
        positions :
            The position of the particles to add.
        polarizabilities :
            The polarizabilities of the particles to add.
        positions_unit :
            The unit of the positions provided.
        """

        self.positions = self.xp.concatenate((self.positions, positions))
        self.polarizabilities = self.xp.concatenate((self.polarizabilities, polarizabilities))

    def clean_particles(self) -> None:
        """
        Remove all particles' data from the system.
        """

        self.positions = self.xp.empty((0, self.dim))
        self.polarizabilities = self.xp.empty((0, self.dim, self.dim))  # Reset polarizabilities to an empty array with shape (0, dim, dim)


    def _calculate_polarizabilities(self) -> None:
        """
        Calculate the polarizabilities of all particles in the system given that their types polarizabilities are known.
        """

        for particle in range(len(self.positions)):
            type_index = self.type_assignments[particle]
            particle_type = self.types[type_index]
            polarizability = particle_type.compute_polarizability()
            self.polarizabilities = self.xp.concatenate((self.polarizabilities, polarizability))
            

    def set_position(self, index: int, position: List[float]) -> None:
        """
        Set the position of a particle at a specified index.

        Parameters
        ----------
        index :
            The index of the particle to set the position for.
        position :
            The new position of the particle. This can be a 1D-three-element array-like.
        """
        
        self.positions[index] = position

