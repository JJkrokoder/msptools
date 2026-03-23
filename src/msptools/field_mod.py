from typing import List
from .backend import get_backend
from .tools.unit_calcs import *
from .tools.field_tools import *
import numpy as np
from numpy.typing import ArrayLike
from scipy.constants import pi


class Field:
    """Class representing an electromagnetic field."""
    
    def __init__(self, **kwargs ) -> None:
        """
        Initialize a Field object by specifying its frequency or wavelength.
        
        Parameters
        ----------
        frequency :
            The frequency of the field.
        frequency_unit :
            The unit of the frequency.
        wavelength :
            The wavelength of the field.
        wavelength_unit :
            The unit of the wavelength.
        """
        frequency = kwargs.get("frequency", None)
        wavelength = kwargs.get("wavelength", None)
        frequency_unit = kwargs.get("frequency_unit", None)
        wavelength_unit = kwargs.get("wavelength_unit", None)

        if frequency is None and wavelength is None:
            raise ValueError("Either 'frequency' or 'wavelength' must be specified.")
        elif frequency is not None and wavelength is not None:
            raise ValueError("Only one of 'frequency' or 'wavelength' should be specified.")
        elif frequency is not None:
            if frequency_unit is None:
                raise ValueError("'frequency' specified but 'frequency_unit' is None.")
            else:
                self.frequency_eV = frequency_to_eV(frequency, frequency_unit)
                self.wave_number_um = frequency_to_wavenumber_um(self.frequency_eV)
                self.wavelength_nm = float(2*pi*1000/self.wave_number_um)
        else:
            if wavelength_unit is None:
                raise ValueError("'wavelength' specified but 'wavelength_unit' is None.")
            else:  
                wavelength_nm = wavelength_to_nm(wavelength, wavelength_unit)
                self.wavelength_nm = wavelength_nm
                self.frequency_eV = nm_to_eV(wavelength_nm)
                self.wave_number_um = float(2*pi*1000/self.wavelength_nm)

    def __str__(self):
        return f"Field: frequency = {self.frequency:.4f} eV, wavelength = {self.wavelength_nm:.2f} nm"


    def get_external_field_in_positions(self, positions: ArrayLike) -> ArrayLike:
        """
        Method to get the external electric field at specified positions.

        Parameters
        ----------
        positions :
            The positions at which to evaluate the external field. Asumed to be in nanometers (nm).
        Returns
        -------
        ArrayLike
            The external electric field at the specified positions.
        """
        return self.external_field_function(positions)
    
    def get_external_gradient_in_positions(self, positions: ArrayLike) -> ArrayLike:
        """
        Method to get the external electric field gradient at specified positions.

        Parameters
        ----------
        positions :
            The positions at which to evaluate the external field gradient. Asumed to be in nanometers (nm).

        Returns
        -------
        ArrayLike
            The external electric field gradient at the specified positions.
        """
        if self.external_gradient_function is None:
            raise NotImplementedError("The method 'get_external_gradient_in_positions' must be implemented in subclasses.")
        else:
            return self.external_gradient_function(positions)
    
    def set_medium_permittivity(self, medium_permittivity: float) -> None:
        """
        Method to set the medium permittivity for the field.

        Parameters
        ----------
        medium_permittivity :
            The permittivity of the medium in which the field propagates.
        """
        self.medium_permittivity = medium_permittivity
        

    

class PlaneWaveField(Field):
    """Class representing a plane wave electromagnetic field."""
    
    def __init__(self,
                 direction: ArrayLike,
                 amplitude: float | complex,
                 polarization: ArrayLike,
                 **kwargs) -> None:
        """
        Initialize a PlaneWaveField object by specifying its direction, amplitude and frequency or wavelength.

        Parameters
        ----------
        direction :
            The propagation direction of the plane wave as a 3-element list. It is normalized by default.
        amplitude :
            The amplitude of the plane wave.
        polarization :
            The polarization vector of the plane wave. It is normalized by default.
        frequency :
            The frequency of the plane wave.
        frequency_unit :
            The unit of the frequency.
        wavelength :
            The wavelength of the plane wave.
        wavelength_unit :
            The unit of the wavelength.

        Notes
        -----
        positions are considered to be in same units as wavelength (default nm).
        """

        super().__init__(**kwargs)
        xp = get_backend(direction)
        self.amplitude = amplitude
        self.polarization = polarization / xp.linalg.norm(xp.asarray(polarization))
        self.direction = xp.asarray(direction) / xp.linalg.norm(xp.asarray(direction))

        if hasattr(self, 'medium_permittivity'):
            wave_number_nm_medium = self.wave_number_um/1000 * self.medium_permittivity**0.5
        else:
            wave_number_nm_medium = self.wave_number_um/1000  # Convert um^-1 to nm^-1
        
        self.external_field_function = lambda positions: plane_wave_function(
            direction=self.direction,
            amplitude_vec=self.amplitude * self.polarization,
            positions=positions, 
            k_magnitude=wave_number_nm_medium
        )

        self.external_gradient_function = lambda positions: plane_wave_gradient(
            direction=self.direction,
            amplitude_vec=self.amplitude * self.polarization,
            positions=positions, 
            k_magnitude=wave_number_nm_medium
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

