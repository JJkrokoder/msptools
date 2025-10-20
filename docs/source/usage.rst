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

Create a Particles with some types:

.. code-block:: python

    particles = msptools.ParticleData([type1])

Add particles to the system:

.. code-block:: python

    particles.add_particles(positions=[[0, 0, 0], [200e-9, 0, 0]], type=type1)


Example Workflow
----------------

.. code-block:: python

    import msptools

    # Create particle types
    typeSi = msptools.ParticleType(radius=50e-9, material='Si')
    typeAu = msptools.ParticleType(radius=100e-9, material='Au')

    # Create a particle system
    particles = msptools.ParticleData([typeSi, typeAu])
    particles.add_particles(positions=[0, 0, 0], type=typeSi)
    particles.add_particles(positions=[[200e-9, 0, 0],[100e-9, 100e-9, 0]], type=typeAu)
