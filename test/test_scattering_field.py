from msptools.observables.scattering import compute_scattering_field
import numpy as np
from scipy.constants import pi
from cmath import exp

def fibonacci_sphere(samples: int, radius: float) -> np.ndarray:
    """
    Generate points uniformly distributed on the surface of a sphere using the Fibonacci lattice method.

    Parameters
    ----------
    samples : int
        The number of points to generate on the sphere.
    radius : float
        The radius of the sphere.

    Returns
    -------
    np.ndarray
        An array of shape (samples, 3) containing the Cartesian coordinates of the points on the sphere.
    """
    points = []
    phi = pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y

        points.append((x * radius, y * radius, z * radius))

    return np.array(points)

class TestScatteringFieldComputation:
    
    ref_k = 2 * np.pi  # Reference wave number for testing
    
    def test_single_dipole_zero_parallel_farfield(self):
        # Test with a single dipole at the origin
        particle_positions = np.array([[0.0, 0.0, 0.0]])
        particle_dipoles = np.array([[1.0, 0.0, 0.0]])
        positions = np.array([[1000.0, 0.0, 0.0]])
        k_magnitude = self.ref_k
        
        scattering_field = compute_scattering_field( positions, particle_positions, particle_dipoles, k_magnitude)
        
        # Expected values can be computed analytically or from a reference implementation
        expected_field = np.array([0.0, 0.0, 0.0])
        
        assert np.allclose(scattering_field, expected_field, atol=1e-4), "Scattering field does not match expected values."
    
    def test_superposition_of_two_dipoles(self):
        # Test with two dipoles at different positions
        particle_positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        particle_dipoles = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
        positions = np.array([[2.0, 2.0, 2.0], [3.0, 0.0, 3.0]])
        k_magnitude = self.ref_k
        
        scattering_field = compute_scattering_field(positions, particle_positions, particle_dipoles, k_magnitude)
        
        Esca_1 = compute_scattering_field(positions, particle_positions[0:1], particle_dipoles[0:1], k_magnitude)
        Esca_2 = compute_scattering_field(positions, particle_positions[1:2], particle_dipoles[1:2], k_magnitude)
        expected_field = Esca_1 + Esca_2
        
        assert np.allclose(scattering_field, expected_field, atol=1e-4), "Scattering field does not match expected values."        
    
    def test_farfield_limit(self):
        # Test the far-field limit where the distance is much larger than the wavelength
        particle_positions = np.array([[0.0, 0.0, 0.0]])
        particle_dipoles = np.array([[1.0, 0.0, 0.0]])
        positions = np.array([[10000.0, 10000.0, 10000.0]])
        k_magnitude = self.ref_k
        
        scattering_field = compute_scattering_field(positions, particle_positions, particle_dipoles, k_magnitude)
        
        # In the far-field limit, the field is k^2 *e^(ikr)/(4*pi*r) * r_hat x (r_hat x p)
        r_vec = positions - particle_positions[0]
        r_norm = np.linalg.norm(r_vec)
        r_hat = r_vec / r_norm
        cross_product = np.cross(r_hat, np.cross(particle_dipoles[0], r_hat))
        expected_field = (k_magnitude**2) * cross_product / (4 * pi * r_norm) * exp(1j * k_magnitude * r_norm)
        assert np.allclose(scattering_field, expected_field, rtol=1e-4), "Scattering field does not match expected far-field behavior."
    
    def test_sphere_intensity_integral(self):
        # Test the integral of the intensity over a sphere surrounding a single dipole
        particle_positions = np.array([[0.0, 0.0, 0.0]])
        particle_dipoles = np.array([[1.0, 0.0, 0.0]])
        k_magnitude = self.ref_k
        
        # Sample points on a sphere of radius R
        R = 100.0
        num_points = 1000
        
        positions = fibonacci_sphere(num_points, R)
        
        scattering_field = compute_scattering_field(positions, particle_positions, particle_dipoles, k_magnitude)
        
        # Compute intensity and integrate over the sphere
        intensity = np.abs(scattering_field)**2
        dOmega = 4 * np.pi * R**2 / num_points  # Solid angle element for uniform sampling
        total_intensity_integral = np.sum(intensity) * dOmega
        
        # Theoretical value for a dipole in free space
        expected_integral = (k_magnitude**4 / (6 * pi)) * np.linalg.norm(particle_dipoles[0])**2
        
        assert np.isclose(total_intensity_integral, expected_integral, rtol=1e-2), "Integrated intensity does not match theoretical value."
    
    def test_k_scaling(self):
        # Test that scaling the wave number scales the scattering field appropriately
        particle_positions = np.array([[0.0, 0.0, 0.0]])
        particle_dipoles = np.array([[1.0, 0.0, 0.0]])
        positions = np.array([[10000.0, 10000.0, 10000.0]])
        
        R = np.linalg.norm(positions - particle_positions[0])
        k_magnitude_1 = self.ref_k
        k_magnitude_2 = 2.25 * self.ref_k
        
        scattering_field_1 = compute_scattering_field(positions, particle_positions, particle_dipoles, k_magnitude_1)
        scattering_field_2 = compute_scattering_field(positions, particle_positions, particle_dipoles, k_magnitude_2)
        
        factor = (k_magnitude_2 / k_magnitude_1)**2*exp(1j * (k_magnitude_2 - k_magnitude_1) * R)
        
        assert np.allclose(scattering_field_2, factor * scattering_field_1, rtol=1e-4), "Scattering field does not scale correctly with wave number."
