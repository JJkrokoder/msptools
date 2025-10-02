import pytest
import msptools as msp
import numpy as np

def test_initialize_default():
    particles = msp.Particles()
    # check that the object is initialized correctly
    assert isinstance(particles, msp.Particles), "The object is not an instance of Particles class"
    assert len(particles.types) == 1, "The default system should have one particle type"
    assert isinstance(particles.types[0], msp.SphereType), "The default particle type should be SphereType"

def test_initialize_single_type():
    custom_type = msp.SphereType(radius=2.0, material="custom_material")
    particles = msp.Particles(types=custom_type)
    assert len(particles.types) == 1, "The system should have one particle type"

def test_clean_positions():
        particles = msp.Particles()
        particles.add_particles([0.0, 0.0, 0.0])
        particles.add_particles([1.0, 0.0, 0.0])
        assert len(particles.positions) == 2, "There should be two particles in the system"
        particles.clean_positions()
        assert len(particles.positions) == 0, "Positions should be cleaned and there should be no particles in the system"

def test_add_1particle_formats():
    particles = msp.Particles()

    x_position = 2.0

    first_particle_position_examples = (np.array([x_position, 0.0, 0.0]),
                                        np.array([[x_position, 0.0, 0.0]]),
                                        [x_position, 0.0, 0.0],
                                        [[x_position, 0.0, 0.0]])
    
    for first_particle_position in first_particle_position_examples:
        
        particles.clean_positions()
        particles.add_particles(first_particle_position)

        fpp_array = np.array(first_particle_position)
        if fpp_array.ndim == 2:
            fpp_array = fpp_array[0]

        assert len(particles.positions) == 1, "There should be one particle in the system"
        particle_position = particles.positions[0]
        assert np.array_equal(particle_position, fpp_array), "The position of the first particle is incorrect"
        

def test_add_2particles_formats():
    particles = msp.Particles()
    
    x1_position = 2.0
    x2_position = 3.0

    two_particles_position_examples = (np.array([[x1_position, 0.0, 0.0],
                                                [x2_position, 0.0, 0.0]]),
                                        [ [x1_position, 0.0, 0.0],
                                            [x2_position, 0.0, 0.0] ])
    
    for two_particles_position in two_particles_position_examples:
        
        particles.clean_positions()
        particles.add_particles(two_particles_position)

        tp_array = np.array(two_particles_position)

        assert len(particles.positions) == 2, "There should be two particles in the system"
        particle_positions = particles.get_positions()
        particle1_position = particles.get_position(0)
        particle2_position = particles.get_position(1)
        assert np.array_equal(particle1_position, tp_array[0]), "The position of the first particle is incorrect"
        assert np.array_equal(particle2_position, tp_array[1]), "The position of the second particle is incorrect"
        assert np.array_equal(particle_positions, tp_array), "The positions of the particles are incorrect"