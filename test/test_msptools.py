import pytest
import msptools as msp
import numpy as np


class TestSystem:

    water_epsilon = 1.75

    def initialize_default(self):
        particles = msp.Particles()
        # check that the object is initialized correctly
        assert isinstance(particles, msp.Particles), "The object is not an instance of Particles class"
        assert len(particles.types) == 1, "The default system should have one particle type"
        assert isinstance(particles.types[0], msp.SphereType), "The default particle type should be SphereType"


