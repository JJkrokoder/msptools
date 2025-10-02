==========================
Usage
==========================

This page provides instructions and examples for using the ``msptools`` package.

Basic Usage
===========

Import the package in your Python script:

.. code-block:: python

    import msptools

Main Features
=============

- Optical Particle Types management: You can create types of particles with specific properties such as radius, material ...
- Particle System management: You can create systems of particles, add or remove particles ...
- External Field customization: You can define the external field acting on the particle system.
- Force Calculations: You can compute the optical forces acting on each particle in the system. Thus, given
    the external field and the particle properties, the Multiple Scattering Problem (MSP) is solved and the forces
    are computed.

Example Workflow
================

.. code-block:: python

    import msptools

    # Create particle types
    typeSi = msptools.ParticleType(radius=50e-9, material='Si')
    typeAu = msptools.ParticleType(radius=100e-9, material='Au')

    # Create a particle system
    system = msptools.ParticleSystem([typeSi, typeAu])
    system.add_particles(positions=[0, 0, 0], type=typeSi)
    system.add_particles(positions=[[200e-9, 0, 0],[100e-9, 100e-9, 0]], type=typeAu)


Repository
=============

For more examples and detailed code, please refer to the GitHub repository <https://github.com/JJkrokoder/msptools/tree/main>.

Support
=======

If you encounter issues, please open an issue on GitHub or contact the maintainers.