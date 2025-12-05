import pytest
import msptools as msp
import numpy as np


class TestParticles:
    
    def test_clean_particles(self):
        particles = msp.Particles()
        particles.add_particles([[0.0, 0.0, 0.0]], 1.0)
        particles.add_particles([[1.0, 0.0, 0.0]], 1.0)
        assert len(particles.positions) == 2, "There should be two particles in the system"
        particles.clean_particles()
        assert len(particles.positions) == 0, "Positions should be cleaned and there should be no particles in the system"

    def test_add_particles(self):
        particles = msp.Particles()
        positions = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        polarizabilities = [1.0, 2.0]
        particles.add_particles(positions, polarizabilities)
        assert len(particles.positions) == 2, "There should be two particles in the system"
        assert particles.polarizabilities == polarizabilities, "Polarizabilities should match the input" 