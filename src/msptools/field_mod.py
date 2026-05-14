from typing import List, Tuple
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
    def evaluate(self, positions: ArrayLike, medium_permittivity: float) -> ArrayLike:
        """
        Abstract method to get the external electric field at specified positions.

        Parameters
        ----------
        positions :
            The positions at which to evaluate the external field. Asumed to be in nanometers (nm).
        medium_permittivity :
            The permittivity of the medium.
        Returns
        -------
        ArrayLike
            The external electric field at the specified positions.
        """
        pass
    
    @abstractmethod
    def evaluate_gradient(self, positions: ArrayLike, medium_permittivity: float) -> ArrayLike:
        """
        Abstract method to get the external electric field gradient at specified positions.

        Parameters
        ----------
        positions :
            The positions at which to evaluate the external field gradient. Asumed to be in nanometers (nm).
        medium_permittivity :
            The permittivity of the medium.

        Returns
        -------
        ArrayLike
            The external electric field gradient at the specified positions.
        """
        pass
    
    def __add__(self, other: Field) -> Field:
        return SumField((self, other)).simplify()
    
    def __mul__(self, scalar: float | complex) -> Field:
        return ScaledField(self, scalar).simplify()
    
    __rmul__ = __mul__

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

    def evaluate(self, positions: ArrayLike, medium_permittivity: float) -> ArrayLike:
        
        return plane_wave_function(
            direction=self.direction,
            amplitude_vec=self.amplitude * self.polarization,
            positions=positions, 
            k_magnitude=pi*2/self.wavelength_nm * medium_permittivity**0.5
        )
    
    def evaluate_gradient(self, positions: ArrayLike, medium_permittivity: float) -> ArrayLike:
        
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
        
    def evaluate(self, positions: ArrayLike, medium_permittivity: float) -> ArrayLike:
        
        return standing_wave_function(
            direction=self.direction,
            amplitude_vec=self.amplitude * self.polarization,
            positions=positions, 
            k_magnitude=pi*2/self.wavelength_nm * medium_permittivity**0.5
        )

    def evaluate_gradient(self, positions: ArrayLike, medium_permittivity: float) -> ArrayLike:
        
        return standing_wave_gradient(
            direction=self.direction,
            amplitude_vec=self.amplitude * self.polarization,
            positions=positions, 
            k_magnitude=pi*2/self.wavelength_nm * medium_permittivity**0.5
        )


@dataclass(frozen=True)
class SumField(Field):
    """Class representing the sum of multiple electromagnetic fields."""
    
    fields: Tuple[Field]

    def evaluate(self, positions: ArrayLike, medium_permittivity: float) -> ArrayLike:
        result = 0
        for field in self.fields:
            result += field.evaluate(positions, medium_permittivity)
        return result

    def evaluate_gradient(self, positions: ArrayLike, medium_permittivity: float) -> ArrayLike:
        result = 0
        for field in self.fields:
            result += field.evaluate_gradient(positions, medium_permittivity)
        return result

    def simplify(self) -> Field:
        
        flat_terms = []
        for field in self.fields:
            if isinstance(field, SumField):
                flat_terms.extend(field.fields)
            else:
                flat_terms.append(field)
                
        if len(flat_terms) == 1:
            return flat_terms[0]
        
        return SumField(tuple(flat_terms))
        
@dataclass(frozen=True)
class ScaledField(Field):
    """Class representing a scaled electromagnetic field."""
    
    field: Field
    scalar: float | complex

    def evaluate(self, positions: ArrayLike, medium_permittivity: float) -> ArrayLike:
        return self.scalar * self.field.evaluate(positions, medium_permittivity)

    def evaluate_gradient(self, positions: ArrayLike, medium_permittivity: float) -> ArrayLike:
        return self.scalar * self.field.evaluate_gradient(positions, medium_permittivity)

    def simplify(self) -> Field:
        if self.scalar == 1:
            return self.field
        elif isinstance(self.field, ScaledField):
            return ScaledField(self.field.field, self.scalar * self.field.scalar).simplify()
        elif isinstance(self.field, PlaneWaveField):
            return PlaneWaveField(
                direction=self.field.direction,
                amplitude=self.scalar * self.field.amplitude,
                polarization=self.field.polarization,
                wavelength_nm=self.field.wavelength_nm
            )
        elif isinstance(self.field, StandingWaveField):
            return StandingWaveField(
                direction=self.field.direction,
                amplitude=self.scalar * self.field.amplitude,
                polarization=self.field.polarization,
                wavelength_nm=self.field.wavelength_nm
            )
        else:
            return self