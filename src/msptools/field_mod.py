from typing import List
import numpy as np
from .unit_calcs import *

def plane_wave_function(direction: np.ndarray,
                        amplitude: np.ndarray,
                        positions: np.ndarray,
                        k_magnitude: float) -> np.ndarray:
    """
    Calculate the electric field of a plane wave at given positions.

    Parameters
    ----------
    direction :
        The propagation direction of the plane wave as a 3-element list or array.
        It is assumed to be normalized.
    amplitude :
        The amplitude of the plane wave.
    position :
        The position at which to evaluate the field.
    k_magnitude :
        The magnitude of the wave vector.

    Returns
    -------
    np.ndarray
        The electric field at specified positions.

    Notes
    -----
    The electric field of a plane wave is given by:
    E(r) = A * exp(i * k · r)
    where A is the amplitude, k is the wave vector, and r is the position vector
    - positions and k_magnitude should be in consistent units.
    """
    k_vector = direction * k_magnitude

    electric_field = amplitude * np.exp(1j * (k_vector @ positions.T))
    return electric_field


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
                self.frequency = frequency_to_eV(frequency, frequency_unit)
        else:
            if wavelength_unit is None:
                raise ValueError("'wavelength' specified but 'wavelength_unit' is None.")
            else:  
                wavelength_nm = wavelength_to_nm(wavelength, wavelength_unit)
                self.frequency = nm_to_eV(wavelength_nm)
        self.frequency = frequency
        self.wave_number_um = frequency_to_wavenumber_um(self.frequency)
        self.wavelength_nm = 2*np.pi*1000/self.wave_number_um
    
    def get_field_in_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        Method to get the total electric field at specified positions.
        
        Parameters
        ----------
        positions :
            The positions at which to evaluate the field.

        Returns
        -------
        np.ndarray
            The electric field at the specified positions.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

class PlaneWaveField(Field):
    """Class representing a plane wave electromagnetic field."""
    
    def __init__(self,
                 direction: List[float] | np.ndarray,
                 amplitude: np.ndarray,
                 **kwargs) -> None:
        """
        Initialize a PlaneWaveField object by specifying its direction, amplitude and frequency or wavelength.

        Parameters
        ----------
        direction :
            The propagation direction of the plane wave as a 3-element list. It is normalized by default.
        amplitude :
            The amplitude of the plane wave.
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
        self.direction = np.array(direction) / np.linalg.norm(direction)
        self.external_field_function = None
    
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
        return plane_wave_function(self.direction, self.amplitude, positions, self.wave_number_um / 1e3)
