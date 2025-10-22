from typing import List
import numpy as np
from .unit_calcs import *

class Field:
    """Class representing an electromagnetic field."""
    
    def __init__(self) -> None:
        """
        Initialize a Field object.
        """

class PlaneWaveField(Field):
    """Class representing a plane wave electromagnetic field."""
    
    def __init__(self,
                 direction: List[float] | np.ndarray,
                 amplitude: float | complex = 1.0,
                 **kwargs) -> None:
        """
        Initialize a PlaneWaveField object by specifying its direction, amplitude and frequency or wavelength.

        Parameters
        ----------
        direction :
            The propagation direction of the plane wave as a 3-element list.
        amplitude :
            The amplitude of the plane wave. Default is 1.0.
        frequency :
            The frequency of the plane wave.
        frequency_unit :
            The unit of the frequency.
        wavelength :
            The wavelength of the plane wave.
        wavelength_unit :
            The unit of the wavelength.
        """

        super().__init__()
        
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
        self.amplitude = amplitude
        self.direction = direction