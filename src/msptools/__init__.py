from dataclasses import Field
import types
from .OFO_calculations import *
from .dipole_moments import *
from .polarizability_mod import *
from .particle_types import *
from .particles_mod import *
from .permittivity import *
from .field_mod import *
from .tools.unit_calcs import *
from .GreenTensor_Electric import *
from .MSP import *
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
    "GreenTensor_Electric",
    "MSP"
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
        self.field.set_medium_permittivity(medium_permittivity)
        self.medium_permittivity = medium_permittivity
        self.positions_unit = positions_unit
        self.particles = Particles()
        self.medium_wave_number_nm = frequency_to_wavenumber_nm(self.field.get_frequency()) * np.sqrt(self.medium_permittivity)

        for ptype in self.particle_types:
            ptype.compute_polarizability(frequency = self.field.get_frequency(), medium_permittivity=self.medium_permittivity)
    
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

        polarizability = particle_type.polarizability
        self.particles.add_particles(positions=positions, polarizabilities=polarizability)
    
    def get_field_in_particles(self) -> np.ndarray:
        """
        Get the electric field at specified positions by solving the Multiple Scattering Problem (MSP).

        Returns
        -------
        np.ndarray
            The electric field at the specified positions.
        """
        

        external_field = self.field.get_external_field_in_positions(self.particles.get_positions())
        green_tensor = construct_green_tensor(self.particles.get_positions(), self.medium_wave_number_nm)
        polarizabilities_array = np.array(self.particles.polarizabilities)
        field_solution = solve_MSP_from_arrays(polarizabilities=polarizabilities_array,
                                   external_field=external_field,
                                   wave_number=self.medium_wave_number_nm,
                                   green_tensor=green_tensor)
        return field_solution
    
    def get_field_gradient_in_particles(self, current_field: np.ndarray) -> np.ndarray:
        """
        Get the electric field gradient at specified positions by solving the Multiple Scattering Problem (MSP) for the gradient.

        Returns
        -------
        np.ndarray
            The electric field gradient at the specified positions.
        """
        
        external_gradient = self.field.get_external_gradient_in_positions(self.particles.get_positions())
        green_tensor_derivative = construct_green_tensor_gradient(self.particles.get_positions(), self.medium_wave_number_nm)
        dipole_moments = calculate_dipole_moments_linear(self.particles.polarizabilities,
                                                         current_field) 
        gradient_solution = MSP_gradient_from_arrays(dipole_moments=dipole_moments,
                                                     external_gradient=external_gradient,
                                                     wave_number=self.medium_wave_number_nm,
                                                     green_tensor_derivative=green_tensor_derivative)
        return gradient_solution
    
    def set_position(self, index: int, position: np.ndarray[int, 3] | List[float]) -> None:
        """
        Set the position of a particle at a specified index.

        Parameters
        ----------
        index :
            The index of the particle to set the position for.

        position :
            The new position of the particle. This can be a 1D-three-element array-like.
        """
        position = np.array(position)* get_multiplier_nanometers(self.positions_unit)
        if position.ndim != 1 or position.shape[0] != 3:
            raise ValueError("Position must be a 1D-three-element array-like.")
        self.particles.set_position(index, position.tolist())


class ForceCalculator:
    """Class to compute optical forces on particles in a System."""
    
    def __init__(self, system: System) -> None:
        """
        Initialize a ForceCalculator object by specifying the System.
        """
        self.system = system


    def compute_forces(self) -> np.ndarray:
        """
        Compute the optical forces on particles at specified positions.

        Returns
        -------
        np.ndarray
            The computed optical forces on the particles.
        """

        E_field = self.system.get_field_in_particles()
        E_grad = self.system.get_field_gradient_in_particles(E_field)
        dipole_moments = calculate_dipole_moments_linear(self.system.particles.polarizabilities, E_field)
        forces = calculate_forces_eppgrad(self.system.medium_permittivity, dipole_moments, E_grad)

        return forces


