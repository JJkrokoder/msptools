import pytest
import msptools as msp
import numpy as np

def test_greet(capsys):
    msp.greet()
    captured = capsys.readouterr()
    assert "Welcome to the msptools package!" in captured.out


class TestSystem:

    water_epsilon = 1.75

    def initialize_default(self):
        particles = msp.Particles()
        # check that the object is initialized correctly
        assert isinstance(particles, msp.Particles), "The object is not an instance of Particles class"

    def test_initialize_positions(self):
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        particles = msp.Particles(positions=positions)
        assert len(particles.positions) == len(positions), "The number of positions is not correct"


