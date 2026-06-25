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
    def evaluate(self, positions: ArrayLike) -> ArrayLike:
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
    def evaluate_gradient(self, positions: ArrayLike) -> ArrayLike:
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

    
    def __add__(self, other):
        return SumField((self, other)).simplify()
    
    def __mul__(self, scalar: float | complex):
        return ScaledField(self, scalar).simplify()
    
    __rmul__ = __mul__
    
    def translate(self, displacement: ArrayLike):
        return TranslatedField(self, displacement).simplify()    
    
    def __neg__(self):
        return ScaledField(self, -1).simplify()

    def __sub__(self, other):
        return self + (-other)

    def simplify(self):
        return self

@dataclass(frozen=True)
class PlaneWaveField(Field):
    """Class representing a plane wave electromagnetic field."""
    
    direction: ArrayLike
    amplitude: float | complex
    polarization: ArrayLike
    medium_wavelength_nm : float

    def __post_init__(self) -> None:
        xp = get_backend(self.direction)

        object.__setattr__(self, "direction", self.direction / xp.linalg.norm(self.direction))
        object.__setattr__(self, "polarization", self.polarization / xp.linalg.norm(self.polarization))
        object.__setattr__(self, "k_vector", 2 * pi * self.direction / self.medium_wavelength_nm)

    def evaluate(self, positions: ArrayLike) -> ArrayLike:
        
        return plane_wave_function(
            direction=self.direction,
            amplitude_vec=self.amplitude * self.polarization,
            positions=positions, 
            k_magnitude=pi*2/self.medium_wavelength_nm
        )
    
    def evaluate_gradient(self, positions: ArrayLike) -> ArrayLike:
        
        return plane_wave_gradient(
            direction=self.direction,
            amplitude_vec=self.amplitude * self.polarization,
            positions=positions, 
            k_magnitude=pi*2/self.medium_wavelength_nm
        )
    
    def translate(self, displacement: ArrayLike) -> Field:
        xp = get_backend(self.direction)
        phase_shift = xp.exp(-1j * xp.dot(self.k_vector, displacement))
        return PlaneWaveField(
            direction=self.direction,
            amplitude=self.amplitude * phase_shift,
            polarization=self.polarization,
            medium_wavelength_nm=self.medium_wavelength_nm
        )
     
@dataclass(frozen=True)   
class StandingWaveField(Field):
    """Class representing a standing wave electromagnetic field."""
    
    direction: ArrayLike
    amplitude: float | complex
    polarization: ArrayLike
    medium_wavelength_nm : float
    
    def __post_init__(self) -> None:
        xp = get_backend(self.direction)

        object.__setattr__(self, "direction", self.direction / xp.linalg.norm(self.direction))
        object.__setattr__(self, "polarization", self.polarization / xp.linalg.norm(self.polarization))
        
    def evaluate(self, positions: ArrayLike) -> ArrayLike:
        
        return standing_wave_function(
            direction=self.direction,
            amplitude_vec=self.amplitude * self.polarization,
            positions=positions, 
            k_magnitude=pi*2/self.medium_wavelength_nm
        )

    def evaluate_gradient(self, positions: ArrayLike) -> ArrayLike:
        
        return standing_wave_gradient(
            direction=self.direction,
            amplitude_vec=self.amplitude * self.polarization,
            positions=positions, 
            k_magnitude=pi*2/self.medium_wavelength_nm
        )


@dataclass(frozen=True)
class SumField(Field):
    """Class representing the sum of multiple electromagnetic fields."""
    
    fields: Tuple[Field, ...]
    
    @property
    def medium_wavelength_nm(self):

        wavelengths = [
            f.medium_wavelength_nm
            for f in self.fields
            if getattr(f, "medium_wavelength_nm", None) is not None
        ]

        if not wavelengths:
            return None

        first = wavelengths[0]

        if all(np.isclose(w, first) for w in wavelengths[1:]):
            return first

        return None
            

    def evaluate(self, positions: ArrayLike) -> ArrayLike:
        result = 0
        for field in self.fields:
            result += field.evaluate(positions)
        return result

    def evaluate_gradient(self, positions: ArrayLike) -> ArrayLike:
        result = 0
        for field in self.fields:
            result += field.evaluate_gradient(positions)
        return result

    def simplify(self) -> Field:
        
        flat_terms = []
        for field in self.fields:
            field = field.simplify()
            if isinstance(field, SumField):
                flat_terms.extend(field.fields)
            else:
                flat_terms.append(field)
                
        if len(flat_terms) == 1:
            return flat_terms[0]
        
        return SumField(tuple(flat_terms))
    
    def translate(self, displacement: ArrayLike) -> Field:
        translated_fields = tuple(field.translate(displacement) for field in self.fields)
        return SumField(translated_fields).simplify()
        
@dataclass(frozen=True)
class ScaledField(Field):
    """Class representing a scaled electromagnetic field."""
    
    field: Field
    scalar: float | complex
    
    @property
    def medium_wavelength_nm(self):
        return self.field.medium_wavelength_nm

    def evaluate(self, positions: ArrayLike) -> ArrayLike:
        return self.scalar * self.field.evaluate(positions)

    def evaluate_gradient(self, positions: ArrayLike) -> ArrayLike:
        return self.scalar * self.field.evaluate_gradient(positions)

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
                medium_wavelength_nm=self.field.medium_wavelength_nm
            )
        elif isinstance(self.field, StandingWaveField):
            return StandingWaveField(
                direction=self.field.direction,
                amplitude=self.scalar * self.field.amplitude,
                polarization=self.field.polarization,
                medium_wavelength_nm=self.field.medium_wavelength_nm
            )
        elif isinstance(self.field, SumField):
            return SumField(tuple(ScaledField(f, self.scalar).simplify() for f in self.field.fields)).simplify()
        else:
            return self
        
    def translate(self, displacement: ArrayLike) -> Field:
        return ScaledField(self.field.translate(displacement), self.scalar).simplify()
        
@dataclass(frozen=True)
class TranslatedField(Field):
    """Class representing a translated electromagnetic field."""
    
    field: Field
    displacement: ArrayLike
    
    @property
    def medium_wavelength_nm(self):
        return getattr(self.field, "medium_wavelength_nm", None)

    def evaluate(self, positions: ArrayLike) -> ArrayLike:
        return self.field.evaluate(positions - self.displacement)
    
    def evaluate_gradient(self, positions: ArrayLike) -> ArrayLike:
        return self.field.evaluate_gradient(positions - self.displacement)
    
    def simplify(self) -> Field:
        if isinstance(self.field, TranslatedField):
            return TranslatedField(self.field.field, self.field.displacement + self.displacement).simplify()
        else:
            return self
        

