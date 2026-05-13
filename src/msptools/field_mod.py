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
        direction = xp.asarray(self.direction)
        polarization = xp.asarray(self.polarization)

        object.__setattr__(self, "direction", direction / xp.linalg.norm(direction))
        object.__setattr__(self, "polarization", polarization / xp.linalg.norm(polarization))

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
     
    
class StandingWaveField(Field):
    """Class representing a standing wave electromagnetic field."""
    
    def __init__(self,
                 direction: ArrayLike,
                 amplitude: float | complex,
                 polarization: ArrayLike,
                 **kwargs) -> None:
        """
        Initialize a StandingWaveField object by specifying its direction, amplitude and frequency or wavelength.

        Parameters
        ----------
        direction :
            The propagation direction of the standing wave as a 3-element list. It is normalized by default.
        amplitude :
            The amplitude of the standing wave.
        polarization :
            The polarization vector of the standing wave. It is normalized by default.
        frequency :
            The frequency of the standing wave.
        frequency_unit :
            The unit of the frequency.
        wavelength :
            The wavelength of the standing wave.
        wavelength_unit :
            The unit of the wavelength.

        Notes
        -----
        positions are considered to be in same units as wavelength (default nm).
        """

        super().__init__(**kwargs)
        self.amplitude = amplitude
        xp = get_backend(direction)
        self.polarization = polarization / xp.linalg.norm(xp.asarray(polarization))
        self.direction = xp.asarray(direction) / xp.linalg.norm(xp.asarray(direction))

        if hasattr(self, 'medium_permittivity'):
            wave_number_nm_medium = self.wave_number_um/1000 * self.medium_permittivity**0.5
        else:
            wave_number_nm_medium = self.wave_number_um/1000  # Convert um^-1 to nm^-1
        
        self.external_field_function = lambda positions: standing_wave_function(
            direction=self.direction,
            amplitude_vec=self.amplitude * self.polarization,
            positions=positions, 
            k_magnitude=wave_number_nm_medium
        )

        self.external_gradient_function = lambda positions: standing_wave_gradient(
            direction=self.direction,
            amplitude_vec=self.amplitude * self.polarization,
            positions=positions, 
            k_magnitude=wave_number_nm_medium
        )

