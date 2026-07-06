from typing import List, Tuple
from .backend import get_backend
from .tools.unit_calcs import *
from .tools.field_tools import *
import numpy as np
from numpy.typing import ArrayLike
from scipy.constants import pi, c
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass(frozen=True)
class MonochromaticData:
    """Class representing the spectra data of an electromagnetic monochromatic field."""
    
    vacuum_wavelength_nm: float
    medium_permittivity: float
    
    @property
    def refractive_index(self):
        return self.medium_permittivity**0.5
    @property
    def medium_wavelength_nm(self):
        return self.vacuum_wavelength_nm / self.refractive_index
    @property
    def angular_frequency(self):
        return 2 * pi * c / (self.vacuum_wavelength_nm * 1e-9)
    

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

        Returns
        -------
        ArrayLike
            The external electric field gradient at the specified positions.
        """
        pass

    @abstractmethod
    def evaluate_double_gradient(self, positions: ArrayLike) -> ArrayLike:
        """
        Abstract method to get the external electric field double gradient at specified positions.

        Parameters
        ----------
        positions :
            The positions at which to evaluate the external field double gradient. Asumed to be in nanometers (nm).

        Returns
        -------
        ArrayLike
            The external electric field double gradient at the specified positions.
        """
        pass

    @property
    def monochromatic_data(self):
        return None

    def eval_complex_field_grad(self, positions: ArrayLike) -> ArrayLike:
        """
        Evaluate the complex field-gradient term ∇E* · E at specified positions.

        Parameters
        ----------
        positions :
            The positions at which to evaluate the complex field-gradient term. Asumed to be in nanometers (nm).

        Returns
        -------
        ArrayLike
            The complex field-gradient term at the specified positions.
        """
        E = self.evaluate(positions)
        grad_E = self.evaluate_gradient(positions)
        return np.einsum('...ij,...j->...i', np.conjugate(grad_E), E)
    
    def eval_curl(self, positions: ArrayLike) -> ArrayLike:
        """
        Evaluate the curl of the electric field at specified positions.

        Parameters
        ----------
        positions :
            The positions at which to evaluate the curl of the electric field. Asumed to be in nanometers (nm).
        
        Returns
        -------
        ArrayLike
            The curl of the electric field at the specified positions.
        """

        grad_E = self.evaluate_gradient(positions)
        xp = get_backend(grad_E)

        curl_E = xp.empty_like(grad_E[..., 0, :])
        curl_E[..., 0] = grad_E[..., 1, 2] - grad_E[..., 2, 1]
        curl_E[..., 1] = grad_E[..., 2, 0] - grad_E[..., 0, 2]
        curl_E[..., 2] = grad_E[..., 0, 1] - grad_E[..., 1, 0]

        return curl_E

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
    
    def evaluate_magnetic(self, positions):
        curl = self.evaluate_curl(positions)
        if self.monochromatic_data:
            return curl / (1j * self.monochromatic_data.angular_frequency)
        else:
            raise ValueError("Monochromatic data is not available for this field.")

@dataclass(frozen=True)
class PlaneWaveSuperposition(Field):
    """Class representing a superposition of plane wave electromagnetic fields."""
    
    fields : Tuple[PlaneWaveField, ...]
    
    def __post_init__(self):
        xp = get_backend(self.fields[0].direction)
        k_vecs = xp.array([field.k_vector for field in self.fields])
        amp_vecs = xp.array([field.amplitude * field.polarization for field in self.fields])
        object.__setattr__(self, "k_vecs", k_vecs)
        object.__setattr__(self, "amp_vecs", amp_vecs)
        object.__setattr__(self, "cross_terms", xp.einsum('ij,ik->ijk', 1j*k_vecs, amp_vecs))
        object.__setattr__(self, "double_cross_terms", xp.einsum('ijk,il->ijkl', -xp.einsum('ij,ik->ijk', k_vecs, k_vecs), amp_vecs))
        set_monochromatic_data = [field.monochromatic_data for field in self.fields]
        object.__setattr__(self, "_monochromatic_data",
                           set_monochromatic_data[0] if all(d == set_monochromatic_data[0] for d in set_monochromatic_data) else None)
        
    @property
    def monochromatic_data(self):
        return self._monochromatic_data
    
    def evaluate(self, positions):
        xp = get_backend(positions)
        phases = xp.einsum('ij,...j->...i', self.k_vecs, positions)
        field_sum = xp.einsum('ij,...i->...j', self.amp_vecs, xp.exp(1j * phases))
        return field_sum
    
    def evaluate_gradient(self, positions):
        xp = get_backend(positions)
        phases = xp.einsum('ij,...j->...i', self.k_vecs, positions)
        cross_term = self.cross_terms
        grad_sum = xp.einsum('ijk,...i->...jk', cross_term, xp.exp(1j * phases))
        return grad_sum

    def evaluate_double_gradient(self, positions):
        xp = get_backend(positions)
        phases = xp.einsum('ij,...j->...i', self.k_vecs, positions)
        double_cross_term = self.double_cross_terms
        double_grad_sum = xp.einsum('ijkl,...i->...jkl', double_cross_term, xp.exp(1j * phases))
        return double_grad_sum


@dataclass(frozen=True)
class PlaneWaveField(Field):
    """Class representing a plane wave electromagnetic field."""
    
    vacuum_wavelength_nm: float
    medium_permittivity: float
    direction: ArrayLike
    amplitude: float | complex
    polarization: ArrayLike
    
    @property
    def monochromatic_data(self):
        return MonochromaticData(
            vacuum_wavelength_nm=self.vacuum_wavelength_nm,
            medium_permittivity=self.medium_permittivity
        )

    def __post_init__(self) -> None:
        xp = get_backend(self.direction)
        object.__setattr__(self, "direction", self.direction / xp.linalg.norm(self.direction))
        object.__setattr__(self, "polarization", self.polarization / xp.linalg.norm(self.polarization))
    
    @property
    def k_vector(self):
        return (2 * pi / self.monochromatic_data.medium_wavelength_nm) * self.direction
    
    def evaluate(self, positions: ArrayLike) -> ArrayLike:
        
        return plane_wave_function(
            direction=self.direction,
            amplitude_vec=self.amplitude * self.polarization,
            positions=positions, 
            k_magnitude=pi*2/self.monochromatic_data.medium_wavelength_nm
        )
    
    def evaluate_gradient(self, positions: ArrayLike) -> ArrayLike:
        
        return plane_wave_gradient(
            direction=self.direction,
            amplitude_vec=self.amplitude * self.polarization,
            positions=positions, 
            k_magnitude=pi*2/self.monochromatic_data.medium_wavelength_nm
        )
        
    def evaluate_double_gradient(self, positions: ArrayLike) -> ArrayLike:
        return plane_wave_double_gradient(
            direction=self.direction,
            amplitude_vec=self.amplitude * self.polarization,
            positions=positions, 
            k_magnitude=pi*2/self.monochromatic_data.medium_wavelength_nm
        )
    
    def translate(self, displacement: ArrayLike) -> Field:
        xp = get_backend(self.direction)
        phase_shift = xp.exp(-1j * xp.dot(self.k_vector, displacement))
        return PlaneWaveField(
            direction=self.direction,
            amplitude=self.amplitude * phase_shift,
            polarization=self.polarization,
            vacuum_wavelength_nm=self.vacuum_wavelength_nm,
            medium_permittivity=self.medium_permittivity
        )
     
@dataclass(frozen=True)   
class StandingWaveField(Field):
    """Class representing a standing wave electromagnetic field."""
    
    vacuum_wavelength_nm: float
    medium_permittivity: float
    direction: ArrayLike
    amplitude: float | complex
    polarization: ArrayLike
    
    @property
    def monochromatic_data(self):
        return MonochromaticData(
            vacuum_wavelength_nm=self.vacuum_wavelength_nm,
            medium_permittivity=self.medium_permittivity
        )
    
    def __post_init__(self) -> None:
        xp = get_backend(self.direction)

        object.__setattr__(self, "direction", self.direction / xp.linalg.norm(self.direction))
        object.__setattr__(self, "polarization", self.polarization / xp.linalg.norm(self.polarization))
        
    def evaluate(self, positions: ArrayLike) -> ArrayLike:
        
        return standing_wave_function(
            direction=self.direction,
            amplitude_vec=self.amplitude * self.polarization,
            positions=positions, 
            k_magnitude=pi*2/self.monochromatic_data.medium_wavelength_nm
        )

    def evaluate_gradient(self, positions: ArrayLike) -> ArrayLike:
        
        return standing_wave_gradient(
            direction=self.direction,
            amplitude_vec=self.amplitude * self.polarization,
            positions=positions, 
            k_magnitude=pi*2/self.monochromatic_data.medium_wavelength_nm
        )
    
    def evaluate_double_gradient(self, positions: ArrayLike) -> ArrayLike:
        return standing_wave_double_gradient(
            direction=self.direction,
            amplitude_vec=self.amplitude * self.polarization,
            positions=positions, 
            k_magnitude=pi*2/self.monochromatic_data.medium_wavelength_nm
        )


@dataclass(frozen=True)
class SumField(Field):
    """Class representing the sum of multiple electromagnetic fields."""
    
    fields: Tuple[Field, ...] 
    
    @property
    def monochromatic_data(self):

        data = [f.monochromatic_data for f in self.fields]

        if any(d is None for d in data):
            return None

        first = data[0]

        if all(d == first for d in data):
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
    
    def evaluate_double_gradient(self, positions: ArrayLike) -> ArrayLike:
        result = 0
        for field in self.fields:
            result += field.evaluate_double_gradient(positions)
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
        
        if all(isinstance(field, PlaneWaveField) for field in flat_terms):
            return PlaneWaveSuperposition(tuple(flat_terms))
        
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
    def monochromatic_data(self):
        return self.field.monochromatic_data

    def evaluate(self, positions: ArrayLike) -> ArrayLike:
        return self.scalar * self.field.evaluate(positions)

    def evaluate_gradient(self, positions: ArrayLike) -> ArrayLike:
        return self.scalar * self.field.evaluate_gradient(positions)
    
    def evaluate_double_gradient(self, positions: ArrayLike) -> ArrayLike:
        return self.scalar * self.field.evaluate_double_gradient(positions)

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
                vacuum_wavelength_nm=self.field.vacuum_wavelength_nm,
                medium_permittivity=self.field.medium_permittivity
            )
        elif isinstance(self.field, StandingWaveField):
            return StandingWaveField(
                direction=self.field.direction,
                amplitude=self.scalar * self.field.amplitude,
                polarization=self.field.polarization,
                vacuum_wavelength_nm=self.field.vacuum_wavelength_nm,
                medium_permittivity=self.field.medium_permittivity
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
    def monochromatic_data(self):
        return self.field.monochromatic_data
    
    @property
    def medium_wavelength_nm(self):
        return getattr(self.field, "medium_wavelength_nm", None)

    def evaluate(self, positions: ArrayLike) -> ArrayLike:
        return self.field.evaluate(positions - self.displacement)
    
    def evaluate_gradient(self, positions: ArrayLike) -> ArrayLike:
        return self.field.evaluate_gradient(positions - self.displacement)
    
    def evaluate_double_gradient(self, positions: ArrayLike) -> ArrayLike:
        return self.field.evaluate_double_gradient(positions - self.displacement)
    
    def simplify(self) -> Field:
        if isinstance(self.field, TranslatedField):
            return TranslatedField(self.field.field, self.field.displacement + self.displacement).simplify()
        else:
            return self
        

