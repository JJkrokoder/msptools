.. docs documentation master file, created by
   sphinx-quickstart on Tue Jul  1 16:58:33 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

msptools
==================

**Welcome to the msptools documentation!**

The msptools package provides python tools for Optical Forces calculations in
particle systems where the Multiple Scattering Problem (MSP) is relevant.

It includes modules for calculating time-average Optical Forces based on the equation:

.. math::

   \mathbf{F} = \frac{\epsilon_m}{2}\Re(\mathbf{p}\cdot \nabla \mathbf{E}^* )

Where:

- :math:`\mathbf{F}` is the optical force,
- :math:`\epsilon_m` is the medium permittivity,
- :math:`\mathbf{p}` is the particle dipole moment,
- :math:`\mathbf{E}` is the electric field.

.. note::

   This equation is also valid if vectors containing the dipole moments and electric field of 
   each particle are used.


Main Features
-------------

- Optical Particle Types management: You can create types of particles with specific properties such as radius, material ...
- Particle System management: You can create systems of particles, add or remove particles ...
- External Field customization: You can define the external field acting on the particle system.
- Force Calculations: You can compute the optical forces acting on each particle in the system. Thus, given the external field and the particle properties, the Multiple Scattering Problem (MSP) is solved and the forces are computed.

Repository
----------

For more examples and detailed code, please refer to the GitHub repository <https://github.com/JJkrokoder/msptools/tree/main>.

Support
--------

If you encounter issues, please open an issue on GitHub or contact the maintainers.

Overview
---------


.. toctree::
   :maxdepth: 2:
   :caption: Basics:

   introduction
   installation
   usage
   API





