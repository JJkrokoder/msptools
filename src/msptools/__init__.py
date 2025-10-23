from dataclasses import Field
import types
from .OFO_calculations import *
from .dipole_moments import *
from .polarizability_mod import *
from .particle_types import *
from .particles_mod import *
from .permittivity import *
from .field_mod import *
from .unit_calcs import *
from typing import List


__all__ = [
    "OFO_calculations",
    "dipole_moments",
    "polarizability_mod",
    "particle_types",
    "particles_mod",
    "permittivity",
    "field_mod",
    "unit_calcs",
]

class System:
    """Class representing a Optical_Forces physical system containing particles."""

    def __init__(self, particle_types : ParticleType | List[ParticleType], field: Field, positions_unit: str, medium_permittivity: float = 1.0) -> None:
        """
        Initialize a System object by specifying the particle types, the field and the medium permittivity.
        """
        if not isinstance(particle_types, list):
            particle_types = [particle_types]
        self.particle_types = particle_types
        self.field = field
        self.medium_permittivity = medium_permittivity
        self.positions_unit = positions_unit
        self.particles = Particles()
    
    def add_particles(self,
                     positions: np.ndarray | List[float] | List[List[float]],
                     particle_type: ParticleType | None = None) -> None:
        """
        Add particles to the system at specified positions.

        Parameters
        ----------
        positions :
            The position of the particles to add. This can be a 1D-three-element or 2D array-like.
        type :
            The type of the particles to add. If not specified, and there is only one type in the system, that type will be used.
        """

        if particle_type is None and len(self.particle_types) > 1:
            raise ValueError("When adding particles to a multi-type system, the 'particle_type' parameter must be specified.")
        else:
            particle_type = self.particle_types[0]

        if particle_type is not None and particle_type not in self.particle_types:
            raise ValueError("The specified particle type is not part of the system's types.")

        positions = np.array(positions)* get_multiplier_nanometers(self.positions_unit)
        if positions.ndim == 1:
            positions = [[pos] for pos in positions.flatten().tolist()]
        elif positions.ndim == 2:
            positions = positions.tolist() 
        else:
            raise ValueError("Positions must be a 1D-three-element or 2D array-like.")

        polarizability = particle_type.compute_polarizability(self.field.frequency, self.medium_permittivity)
        self.particles.add_particles(positions=positions, polarizabilities=polarizability)


class ForceCalculator:
    """Class to compute optical forces on particles in a System."""
    
    def __init__(self, system: System) -> None:
        """
        Initialize a ForceCalculator object by specifying the System.
        """
        self.system = system


    def compute_forces(self, positions : np.ndarray | List[float] | List[List[float]]) -> np.ndarray:
        """
        Compute the optical forces on particles at specified positions.

        Parameters
        ----------
        positions :
            The position of the particles to compute forces on. This can be a 1D-three-element or 2D array-like.

        Returns
        -------
        np.ndarray
            The computed optical forces on the particles.
        """
        
        positions = np.array(positions)

        if positions.ndim == 1:
            positions = np.array([positions.flatten().tolist()])
        elif positions.ndim != 2:
            raise ValueError("Positions must be a 1D-three-element or 2D array-like.")

        E_field = self.system.field.get_field_in_positions(positions)
        E_grad = self.system.field.get_field_gradient_in_positions(positions)
        dipole_moments = calculate_dipole_moments_linear(self.system.particles.polarizabilities, E_field)
        forces = calculate_forces_eppgrad(self.system.medium_permittivity, dipole_moments, E_grad)

        return forces


