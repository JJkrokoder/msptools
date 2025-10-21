from typing import List
import numpy as np


class Particles:
    """Class representing a system of particles."""

    def __init__(self) -> None:
        """
        Initialize a Particles object.
        """

        self.positions = []
        self.polarizabilities = []


    def add_particles(self,
                     positions: List[List[float]],
                     polarizabilities: complex | List[complex]) -> None:
        """
        Add particles to the system at specified positions and with specified polarizabilities.

        Parameters
        ----------
        positions :
            The position of the particles to add.
        polarizabilities :
            The polarizabilities of the particles to add.
        """

        self.positions.extend(positions)

        if isinstance(polarizabilities, list):
            self.polarizabilities.extend(polarizabilities)
        else:
            self.polarizabilities.extend([polarizabilities] * len(positions))

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


    def _calculate_polarizabilities(self) -> None:
        """
        Calculate the polarizabilities of all particles in the system given that their types polarizabilities are known.
        """

        for particle in range(len(self.positions)):
            type_index = self.type_assignments[particle]
            particle_type = self.types[type_index]
            polarizability = particle_type.compute_polarizability()
            self.polarizabilities.append(polarizability)


