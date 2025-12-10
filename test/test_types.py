import msptools as msp
import numpy as np 

def test_initialize_default():
    my_type = msp.ParticleType()
    assert isinstance(my_type, msp.ParticleType), "The object is not an instance of ParticleType class"

class TestSphereType:

    def test_custom_properties(self):
        sphere = msp.SphereType(radius=2.5, material="custom_material", radius_unit="nm")
        assert sphere.radius == 2.5, "Radius should be set to 2.5"
        assert sphere.radius_unit == "nm"
        assert sphere.material == "custom_material", "Material should be set to 'custom_material'"
