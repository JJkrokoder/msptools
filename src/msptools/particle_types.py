from .polarizability_mod import calculate_sphere_polarizability

class ParticleType:
    """Class representing a type of particle with specific properties."""

    def __init__(self):
        self.properties = []

class SphereType(ParticleType):
    """Class representing spherical particles."""

    def __init__(self, radius: float = 1.0, material: str = "default"):
        super().__init__()
        radius = radius
        if material == "default":
            polarizability = 1.0
        else:
            polarizability = calculate_sphere_polarizability(radius, material)
        self.properties = {
            "radius": radius,
            "polarizability": polarizability
        }