from .polarizability_mod import *
from .tools.unit_calcs import *
from .permittivity import permittivity_ridx

class ParticleType:
    """Class representing a type of particle with specific properties."""

    def compute_polarizability(self, frequency: float, medium_permittivity: float) -> complex:
        """Compute the polarizability of the particle type at a given frequency."""
        raise NotImplementedError("This method should be implemented by subclasses.")

class SphereType(ParticleType):
    """Class representing spherical particles."""

    def __init__(self, material: str, radius: float, radius_unit: str, polarizability: complex = None) -> None:
        self.radius = radius
        self.radius_unit = radius_unit
        self.material = material
        if polarizability is not None:
            self.compute_polarizability = lambda frequency, medium_permittivity: polarizability

    def compute_polarizability(self, frequency: float, medium_permittivity: float, dim: int = 3) -> complex:
        scalar_polarizability = Mie_electric_dipole_polarizability(radius=self.radius,
                                  medium_permittivity=medium_permittivity,
                                  particle_permittivity=permittivity_ridx(frequency, self.material),
                                  wave_number=frequency_to_wavenumber_nm(frequency))
        self.polarizability = scalar_polarizability

class CoreShellType(ParticleType):
    """Class representing core-shell particles."""

    def __init__(self, material_core: str, material_shell: str, radius_core: float, radius_shell: float, radius_unit: str, polarizability: complex = None) -> None:
        """
        Initialize a core-shell particle type.
        
        Parameters
        ----------
        material_core :
            The material of the core.
        material_shell : 
            The material of the shell.
        radius_core :
            The radius of the core.
        radius_shell :
            The radius of the shell (including the core).
        radius_unit :
            The unit of the radius (e.g., 'nm', 'm').
        """
        self.material_core = material_core
        self.material_shell = material_shell
        self.radius_core = radius_core
        self.radius_shell = radius_shell
        self.radius_unit = radius_unit
        if polarizability is not None:
            self.compute_polarizability = lambda frequency, medium_permittivity: polarizability


    def compute_polarizability(self, frequency: float, medium_permittivity: float) -> complex:
        scalar_polarizability = Core_Shell_Clausius_Mossotti(radius_core=self.radius_core,
                                  radius_shell=self.radius_shell,
                                  medium_permittivity=medium_permittivity,
                                  particle_permittivity_core=permittivity_ridx(frequency, self.material_core),
                                  particle_permittivity_shell=permittivity_ridx(frequency, self.material_shell))
        self.polarizability = scalar_polarizability
        
    

