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

    def test_tunable_properties(self):
        tunable_sphere = msp.SphereType(radius=3.0, material="Tunable", radius_unit="nm", tunable_permittivity=2.5)
        assert tunable_sphere.radius == 3.0, "Radius should be set to 3.0"
        assert tunable_sphere.radius_unit == "nm"
        assert tunable_sphere.material == "Tunable", "Material should be set to 'Tunable'"
        assert tunable_sphere.tunable_permittivity == 2.5, "Tunable permittivity should be set to 2.5"

    def test_missing_tunable_permittivity(self):
        try:
            _ = msp.SphereType(radius=3.0, material="Tunable", radius_unit="nm")
            assert False, "Expected ValueError for missing tunable_permittivity"
        except ValueError as e:
            assert str(e) == "For 'Tunable' material, 'tunable_permittivity' must be provided.", f"Unexpected error message: {str(e)}"