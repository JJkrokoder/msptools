from typing import List
import numpy as np
from .tools.unit_calcs import *
from .tools.field_tools import *


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
                self.wavelength_nm = 2*np.pi*1000/self.wave_number_um
        else:
            if wavelength_unit is None:
                raise ValueError("'wavelength' specified but 'wavelength_unit' is None.")
            else:  
                wavelength_nm = wavelength_to_nm(wavelength, wavelength_unit)
                self.wavelength_nm = wavelength_nm
                self.frequency_eV = nm_to_eV(wavelength_nm)
                self.wave_number_um = 2*np.pi*1000/self.wavelength_nm

    def __str__(self):
        return f"Field: frequency = {self.frequency:.4f} eV, wavelength = {self.wavelength_nm:.2f} nm"
    
    def get_frequency(self) -> float:
        """
        Method to get the frequency of the field in eV.

        Returns
        -------
        float
            The frequency of the field in eV.
        """
        return self.frequency_eV
    
    def get_wavelength(self) -> float:
        """
        Method to get the wavelength of the field in nanometers (nm).

        Returns
        -------
        float
            The wavelength of the field in nanometers (nm).
        """
        return self.wavelength_nm


    def get_external_field_in_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        Method to get the external electric field at specified positions.

        Parameters
        ----------
        positions :
            The positions at which to evaluate the external field. Asumed to be in nanometers (nm).
        Returns
        -------
        np.ndarray
            The external electric field at the specified positions.
        """
        if self.external_field_function is None:
            raise NotImplementedError("The method 'get_external_field_in_positions' must be implemented in subclasses.")
        else:
            return self.external_field_function(positions)
    
    def get_external_gradient_in_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        Method to get the external electric field gradient at specified positions.

        Parameters
        ----------
        positions :
            The positions at which to evaluate the external field gradient. Asumed to be in nanometers (nm).

        Returns
        -------
        np.ndarray
            The external electric field gradient at the specified positions.
        """
        if self.external_gradient_function is None:
            raise NotImplementedError("The method 'get_external_gradient_in_positions' must be implemented in subclasses.")
        else:
            return self.external_gradient_function(positions)
    

class PlaneWaveField(Field):
    """Class representing a plane wave electromagnetic field."""
    
    def __init__(self,
                 direction: List[float] | np.ndarray,
                 amplitude: float | complex,
                 polarization: List[float] | np.ndarray,
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
        """

        super().__init__(**kwargs)
        self.amplitude = amplitude
        self.polarization = np.array(polarization) / np.linalg.norm(polarization)
        self.direction = np.array(direction) / np.linalg.norm(direction)
        
        self.external_field_function = lambda positions: plane_wave_function(
            direction=self.direction,
            amplitude_vec=self.amplitude * self.polarization,
            positions=positions,
            k_magnitude=self.wave_number_um / 1000  # convert from um^-1 to nm^-1
        )
    
    def get_direction(self) -> np.ndarray:
        """
        Method to get the propagation direction of the plane wave.

        Returns
        -------
        np.ndarray
            The normalized propagation direction of the plane wave.
        """
        return self.direction
    
    def get_polarization(self) -> np.ndarray:
        """
        Method to get the polarization vector of the plane wave.

        Returns
        -------
        np.ndarray
            The normalized polarization vector of the plane wave.
        """
        return self.polarization
    
    def get_amplitude(self) -> float | complex:
        """
        Method to get the amplitude of the plane wave.

        Returns
        -------
        float | complex
            The amplitude of the plane wave.
        """
        return self.amplitude
    


