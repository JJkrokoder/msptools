import msptools as msp
import numpy as np


class TestParticles:
    
    def test_clean_particles(self):
        particles = msp.Particles(xp = np)
        particles.positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        particles.polarizabilities = np.array([1.0, 2.0])
        assert len(particles.positions) == 2, "There should be two particles in the system"
        particles.clean_particles()
        assert len(particles.positions) == 0, "Positions should be cleaned and there should be no particles in the system"

    def test_add_particles(self):
        particles = msp.Particles(xp = np)
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        polarizabilities = np.array([1.0, 2.0])
        particles.add_particles(positions, polarizabilities)
        assert len(particles.positions) == 2, "There should be two particles in the system"
        assert len(particles.polarizabilities) == 2, f"There should be two polarizabilities in the system. Found {len(particles.polarizabilities)}"