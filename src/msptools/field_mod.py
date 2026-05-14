from typing import List
from .backend import get_backend
from .tools.unit_calcs import *
from .tools.field_tools import *
import numpy as np
from numpy.typing import ArrayLike
from scipy.constants import pi
from abc import ABC, abstractmethod
from dataclasses import dataclass


class Field(ABC):
    """Class representing an electromagnetic field."""
    
    @abstractmethod
    def get_external_field_in_positions(self, positions: ArrayLike) -> ArrayLike:
        """
        Abstract method to get the external electric field at specified positions.

        Parameters
        ----------
        positions :
            The positions at which to evaluate the external field. Asumed to be in nanometers (nm).
        Returns
        -------
        ArrayLike
            The external electric field at the specified positions.
        """
        pass
    
    @abstractmethod
    def get_external_gradient_in_positions(self, positions: ArrayLike) -> ArrayLike:
        """
        Abstract method to get the external electric field gradient at specified positions.

        Parameters
        ----------
        positions :
            The positions at which to evaluate the external field gradient. Asumed to be in nanometers (nm).

        Returns
        -------
        ArrayLike
            The external electric field gradient at the specified positions.
        """
        pass      

    
@dataclass(frozen=True)
class PlaneWaveField(Field):
    """Class representing a plane wave electromagnetic field."""
    
    direction: ArrayLike
    amplitude: float | complex
    polarization: ArrayLike
    wavelength_nm : float

    def __post_init__(self) -> None:
        xp = get_backend(self.direction)

        object.__setattr__(self, "direction", self.direction / xp.linalg.norm(self.direction))
        object.__setattr__(self, "polarization", self.polarization / xp.linalg.norm(self.polarization))

    def get_external_field_in_positions(self, positions: ArrayLike, medium_permittivity: float) -> ArrayLike:
        
        return plane_wave_function(
            direction=self.direction,
            amplitude_vec=self.amplitude * self.polarization,
            positions=positions, 
            k_magnitude=pi*2/self.wavelength_nm * medium_permittivity**0.5
        )
    
    def get_external_gradient_in_positions(self, positions: ArrayLike, medium_permittivity: float) -> ArrayLike:
        
        return plane_wave_gradient(
            direction=self.direction,
            amplitude_vec=self.amplitude * self.polarization,
            positions=positions, 
            k_magnitude=pi*2/self.wavelength_nm * medium_permittivity**0.5
        )
     
@dataclass(frozen=True)   
class StandingWaveField(Field):
    """Class representing a standing wave electromagnetic field."""
    
    direction: ArrayLike
    amplitude: float | complex
    polarization: ArrayLike
    wavelength_nm : float
    
    def __post_init__(self) -> None:
        xp = get_backend(self.direction)

        object.__setattr__(self, "direction", self.direction / xp.linalg.norm(self.direction))
        object.__setattr__(self, "polarization", self.polarization / xp.linalg.norm(self.polarization))
        
    def get_external_field_in_positions(self, positions: ArrayLike, medium_permittivity: float) -> ArrayLike:
        
        return standing_wave_function(
            direction=self.direction,
            amplitude_vec=self.amplitude * self.polarization,
            positions=positions, 
            k_magnitude=pi*2/self.wavelength_nm * medium_permittivity**0.5
        )

    def get_external_gradient_in_positions(self, positions: ArrayLike, medium_permittivity: float) -> ArrayLike:
        
        return standing_wave_gradient(
            direction=self.direction,
            amplitude_vec=self.amplitude * self.polarization,
            positions=positions, 
            k_magnitude=pi*2/self.wavelength_nm * medium_permittivity**0.5
        )

