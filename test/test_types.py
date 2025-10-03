import msptools as msp
import numpy as np 

def test_initialize_default():
    my_type = msp.ParticleType()
    assert isinstance(my_type, msp.ParticleType), "The object is not an instance of ParticleType class"

class TestSphereType:
    
    def test_default_properties(self):
        sphere = msp.SphereType(material="default")
        assert sphere.radius == 1.0, "Default radius should be 1.0"
        assert sphere.material == "default", "Default material should be 'default'"

    def test_custom_properties(self):
        sphere = msp.SphereType(radius=2.5, material="custom_material")
        assert sphere.radius == 2.5, "Radius should be set to 2.5"
        assert sphere.material == "custom_material", "Material should be set to 'custom_material'"
