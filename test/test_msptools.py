import pytest
import msptools as msp

def test_greet(capsys):
    msp.greet()
    captured = capsys.readouterr()
    assert "Welcome to the msptools package!" in captured.out


class TestParticleType:

    def test_initialization(self):
        particle = msp.ParticleType()
        assert isinstance(particle, msp.ParticleType)
        assert particle.properties == []
    
class TestSphereType:

    def test_initialization(self):
        radius = 2.3
        material = "gold"
        sphere = msp.SphereType(radius, material)
        assert isinstance(sphere, msp.SphereType)
        assert "polarizability" in sphere.properties
        assert sphere.properties["radius"] == radius
        assert isinstance(sphere.properties["polarizability"], complex)
        
