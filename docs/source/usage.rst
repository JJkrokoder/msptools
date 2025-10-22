==========================
Usage
==========================

This page provides instructions and examples for using the ``msptools`` package.

Basic Usage
===========

Import the package in your Python script:

.. code-block:: python

    import msptools

Create Particle Types:

.. code-block:: python

    type1 = msptools.ParticleType(radius=50e-9, material='Si')

Create an External Field:
.. code-block:: python

    ext_field = msptools.PlaneWave(
    wavelength=532e-9,
    amplitude=1.0,
    polarization=[0, 0, 1],
    wave_vector=[1, 0, 0]
    )

Initialize System:
.. code-block:: python

    system = msptools.System(particle_types=[type1], field=ext_field, medium_permittivity=1.0)

Add particles to the system:
.. code-block:: python

    system.add_particles(positions=[[0, 0, 0], [200e-9, 0, 0]], particle_type=type1)

Initialize Force Calculator:
.. code-block:: python

    force_calculator = msptools.ForceCalculator(system)
Compute Optical Forces:
.. code-block:: python
    forces = force_calculator.compute_forces(system.get_positions())

Example Workflow
----------------

.. code-block:: python

    import msptools

    # Create Particle Types
    type1 = msptools.ParticleType(radius=50e-9, material='Si')
    type2 = msptools.ParticleType(radius=75e-9, material='Au') 

    # Create an External Field
    ext_field = msptools.PlaneWave(
        wavelength=532e-9,
        amplitude=1.0,
        polarization=[0, 0, 1],
        wave_vector=[1, 0, 0]
    )

    # Initialize System
    system = msptools.System(particle_types=[type1, type2], field=ext_field, medium_permittivity=1.0)

    # Add particles to the system
    system.add_particles(positions=[[0, 0, 0], [200e-9, 0, 0]], particle_type=type1)
    system.add_particles(positions=[[400e-9, 0, 0]], particle_type=type2) 

    # Initialize Force Calculator
    force_calculator = msptools.ForceCalculator(system)

    # Compute Optical Forces
    forces = force_calculator.compute_forces(system.get_positions())
    
