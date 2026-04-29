from msptools.tools.mie_theory import (tE_n_coefficient, 
                                       tEn_aden_kerker_coefficient, 
                                       tM_n_coefficient, 
                                       tMn_aden_kerker_coefficient,
                                       select_multipole_orders_sphere)
import numpy as np
import pytest

def large_x_approximation_tE1(x, m):
    num = m*np.cos(m*x)*np.sin(x) - np.cos(x)*np.sin(m*x)
    denom = 1j*m*np.cos(m*x) + np.sin(m*x)
    return 1j*(-np.exp(-1j*x)*num/denom)

def small_x_cm_approximation_tE1(x, m):
    return 2/3*(m**2 - 1)/(m**2 + 2)*x**3

def small_x_cm_approximation_tM1(x, m):
    return (1/45)*x**5*(m**2 - 1)

def large_x_approximation_tM1(x, m):
    num = np.cos(m*x)*np.sin(x) - m*np.cos(x)*np.sin(m*x)
    denom = 1j*np.cos(m*x) + m*np.sin(m*x)
    return 1j*(-np.exp(-1j*x)*num/denom)


class Test_Mie_Coefficients:

    eps_particle = 2.5 + 1j
    eps_medium = 1.33**2
    m =eps_particle**0.5 / eps_medium**0.5
    n_list = [1, 2, 3]
    x_list = [0.1, 1.0, 10.0]
    
    def test_tE_n_coefficient_imaginary_positive(self):
        for n in self.n_list:
            for x_m in self.x_list:
                tE_n = tE_n_coefficient(n, x_m, self.m)
                assert tE_n.imag >= 0, f"Imaginary part of tE_n should be positive for physical Mie coefficients, got {tE_n.imag:.4f}"
        
    def test_tM_n_coefficient_imaginary_positive(self):
        for n in self.n_list:
            for x_m in self.x_list:
                tM_n = tM_n_coefficient(n, x_m, self.m)
                assert tM_n.imag >= 0, f"Imaginary part of tM_n should be positive for physical Mie coefficients, got {tM_n.imag:.4f}"
                
    def test_tE_n_bigger_imaginary_than_abs(self):
        for n in self.n_list:
            for x_m in self.x_list:
                tE_n = tE_n_coefficient(n, x_m, self.m)
                assert tE_n.imag >= abs(tE_n)**2, f"Imaginary part of tE_n should be greater than or equal to real part for physical Mie coefficients, got {tE_n.imag:.4f} vs abs(tE_n)**2 = {abs(tE_n)**2:.4f}"
    
    def test_tM_n_bigger_imaginary_than_abs(self):
        for n in self.n_list:
            for x_m in self.x_list:
                tM_n = tM_n_coefficient(n, x_m, self.m)
                assert tM_n.imag >= abs(tM_n)**2, f"Imaginary part of tM_n should be greater than or equal to real part for physical Mie coefficients, got {tM_n.imag:.4f} vs abs(tM_n)**2 = {abs(tM_n)**2:.4f}"
    
    def test_tE_n_imaginary_smaller_than_one(self):
        for n in self.n_list:
            for x_m in self.x_list:
                tE_n = tE_n_coefficient(n, x_m, self.m)
                assert tE_n.imag <= 1, f"Imaginary part of tE_n should be less than or equal to 1 for physical Mie coefficients, got {tE_n.imag:.4f}"
    
    def test_tM_n_imaginary_smaller_than_one(self):
        for n in self.n_list:
            for x_m in self.x_list:
                tM_n = tM_n_coefficient(n, x_m, self.m)
                assert tM_n.imag <= 1, f"Imaginary part of tM_n should be less than or equal to 1 for physical Mie coefficients, got {tM_n.imag:.4f}"
    
    def test_tEn_real_permittivity(self):
        for n in self.n_list:
            for x_m in self.x_list:
                tE_n = tE_n_coefficient(n, x_m, self.m.real)
                assert np.isclose(tE_n.imag, abs(tE_n)**2, atol=1e-6), f"For purely real permittivity, imaginary part of tE_n should equal the square of the absolute value, got {tE_n.imag:.4f} vs abs(tE_n)**2 = {abs(tE_n)**2:.4f}"
    
    def test_tMn_real_permittivity(self):
        for n in self.n_list:
            for x_m in self.x_list:
                tM_n = tM_n_coefficient(n, x_m, self.m.real)
                assert np.isclose(tM_n.imag, abs(tM_n)**2, atol=1e-6), f"For purely real permittivity, imaginary part of tM_n should equal the square of the absolute value, got {tM_n.imag:.4f} vs abs(tM_n)**2 = {abs(tM_n)**2:.4f}"
    
    def test_tE1_large_x_approximation(self):
        x = np.linspace(2900, 2910, 100)
        tE1 = tE_n_coefficient(1, x, self.m)
        tE1_large_x_approx = large_x_approximation_tE1(x, self.m)
        
        assert np.allclose(tE1.imag, tE1_large_x_approx.imag, rtol=1e-3, atol=1e-2), f"Imaginary part of tE1 does not match large x approximation. Got max relative error of {100*np.max(np.abs(tE1.imag - tE1_large_x_approx.imag) / np.abs(tE1.imag)):.4f} %"
        assert np.allclose(tE1.real, tE1_large_x_approx.real, rtol=1e-3, atol=1e-2), f"Real part of tE1 does not match large x approximation. Got max relative error of {100*np.max(np.abs(tE1.real - tE1_large_x_approx.real) / np.abs(tE1.real)):.4f} %"
    
    def test_tE1_small_x_approximation(self):
        x = np.linspace(0.01, 0.1, 100)
        tE1 = tE_n_coefficient(1, x, self.m)
        tE1_small_x_approx = small_x_cm_approximation_tE1(x, self.m)
        
        assert np.allclose(tE1.imag, tE1_small_x_approx.imag, rtol=1e-3, atol=1e-6), f"Imaginary part of tE1 does not match small x approximation. Got max relative error of {100*np.max(np.abs(tE1.imag - tE1_small_x_approx.imag) / np.abs(tE1.imag)):.4f} %"
        assert np.allclose(tE1.real, tE1_small_x_approx.real, rtol=1e-3, atol=1e-6), f"Real part of tE1 does not match small x approximation. Got max relative error of {100*np.max(np.abs(tE1.real - tE1_small_x_approx.real) / np.abs(tE1.real)):.4f} %"
    
    def test_tM1_small_x_approximation(self):
        x = np.linspace(0.01, 0.1, 100)
        tM1 = tM_n_coefficient(1, x, self.m)
        tM1_small_x_approx = small_x_cm_approximation_tM1(x, self.m)
        
        assert np.allclose(tM1.imag, tM1_small_x_approx.imag, rtol=1e-3, atol=1e-6), f"Imaginary part of tM1 does not match small x approximation. Got max relative error of {100*np.max(np.abs(tM1.imag - tM1_small_x_approx.imag) / np.abs(tM1.imag)):.4f} %"
        assert np.allclose(tM1.real, tM1_small_x_approx.real, rtol=1e-3, atol=1e-6), f"Real part of tM1 does not match small x approximation. Got max relative error of {100*np.max(np.abs(tM1.real - tM1_small_x_approx.real) / np.abs(tM1.real)):.4f} %"
        
    def test_tM1_large_x_approximation(self):
        x = np.linspace(2900, 2910, 100)
        tM1 = tM_n_coefficient(1, x, self.m)
        tM1_large_x_approx = large_x_approximation_tM1(x, self.m)
        
        assert np.allclose(tM1.imag, tM1_large_x_approx.imag, rtol=1e-3, atol=1e-2), f"Imaginary part of tM1 does not match large x approximation. Got max relative error of {100*np.max(np.abs(tM1.imag - tM1_large_x_approx.imag) / np.abs(tM1.imag)):.4f} %"
        assert np.allclose(tM1.real, tM1_large_x_approx.real, rtol=1e-3, atol=1e-2), f"Real part of tM1 does not match large x approximation. Got max relative error of {100*np.max(np.abs(tM1.real - tM1_large_x_approx.real) / np.abs(tM1.real)):.4f} %"

class Test_Aden_Kerker_Coefficient:
    
    n_values = [1, 2, 3]
    x_core = 1.0
    x_shell = 2.0
    eps_core = 1.8 + 0.5j
    eps_shell = 1.5
    eps_m = 1.2
    x_c_m = x_core * eps_m**0.5
    x_s_m = x_shell * eps_m**0.5
    m_c = eps_core**0.5 / eps_m**0.5
    m_s = eps_shell**0.5 / eps_m**0.5
    
    def test_tEn_aden_kerker_imaginary_greater_than_abs(self):
        shell_radii = np.linspace(1.0, 10.0, 5)*self.x_c_m
        for n in self.n_values:
            coeff_aden_kerker = tEn_aden_kerker_coefficient(n, self.x_c_m, shell_radii, self.m_c, self.m_s)
            assert np.all(coeff_aden_kerker.imag >= abs(coeff_aden_kerker)**2), f"Imaginary part of A_n should be greater than or equal to the square of the absolute value for physical coefficients, got {coeff_aden_kerker.imag:.4f} vs abs(A_n)**2 = {abs(coeff_aden_kerker)**2:.4f}"
    
    def test_tMn_aden_kerker_imaginary_greater_than_abs(self):
        shell_radii = np.linspace(1.0, 10.0, 5)*self.x_c_m
        for n in self.n_values:
            coeff_aden_kerker = tMn_aden_kerker_coefficient(n, self.x_c_m, shell_radii, self.m_c, self.m_s)
            assert np.all(coeff_aden_kerker.imag >= abs(coeff_aden_kerker)**2), f"Imaginary part of B_n should be greater than or equal to the square of the absolute value for physical coefficients, got {coeff_aden_kerker.imag:.4f} vs abs(B_n)**2 = {abs(coeff_aden_kerker)**2:.4f}"
    
    def test_tEn_aden_kerker_imaginary_smaller_than_one(self):
        shell_radii = np.linspace(1.0, 10.0, 5)*self.x_c_m
        for n in self.n_values:
            coeff_aden_kerker = tEn_aden_kerker_coefficient(n, self.x_c_m, shell_radii, self.m_c, self.m_s)
            assert np.all(coeff_aden_kerker.imag <= 1), f"Imaginary part of A_n should be less than or equal to 1 for physical coefficients, got {coeff_aden_kerker.imag:.4f}"
    
    def test_tMn_aden_kerker_imaginary_smaller_than_one(self):
        shell_radii = np.linspace(1.0, 10.0, 5)*self.x_c_m
        for n in self.n_values:
            coeff_aden_kerker = tMn_aden_kerker_coefficient(n, self.x_c_m, shell_radii, self.m_c, self.m_s)
            assert np.all(coeff_aden_kerker.imag <= 1), f"Imaginary part of B_n should be less than or equal to 1 for physical coefficients, got {coeff_aden_kerker.imag:.4f}"
        
    def test_tEn_aden_kerker_imaginary_real_permittivity(self):
        shell_radii = np.linspace(1.0, 10.0, 5)*self.x_c_m
        for n in self.n_values:
            coeff_aden_kerker = tEn_aden_kerker_coefficient(n, self.x_c_m, shell_radii, self.m_c.real, self.m_s.real)
            assert np.allclose(coeff_aden_kerker.imag, abs(coeff_aden_kerker)**2, atol=1e-6), f"For purely real permittivity, imaginary part of A_n should equal the square of the absolute value, got {coeff_aden_kerker.imag:.4f} vs abs(A_n)**2 = {abs(coeff_aden_kerker)**2:.4f}"
            
    def test_tMn_aden_kerker_imaginary_real_permittivity(self):
        shell_radii = np.linspace(1.0, 10.0, 5)*self.x_c_m
        for n in self.n_values:
            coeff_aden_kerker = tMn_aden_kerker_coefficient(n, self.x_c_m, shell_radii, self.m_c.real, self.m_s.real)
            assert np.allclose(coeff_aden_kerker.imag, abs(coeff_aden_kerker)**2, atol=1e-6), f"For purely real permittivity, imaginary part of B_n should equal the square of the absolute value, got {coeff_aden_kerker.imag:.4f} vs abs(B_n)**2 = {abs(coeff_aden_kerker)**2:.4f}"
    
    def test_tE_n_coefficient_consistency_pure_core_radius(self):
        
        for n in self.n_values:
            coeff_aden_kerker = tEn_aden_kerker_coefficient(n, self.x_c_m, self.x_c_m, self.m_c, self.m_s)
            coeff_Mie_pure_core = tE_n_coefficient(n, self.x_c_m, self.m_c)
            
            assert np.isclose(coeff_aden_kerker, coeff_Mie_pure_core, rtol=1e-6), f"AK {coeff_aden_kerker:.4f} does not match Mie for pure core {coeff_Mie_pure_core:.4f}, n={n}"
        
    def test_tE_n_coefficient_consistency_pure_shell_radius(self):
        
        for n in self.n_values:
            coeff_aden_kerker = tEn_aden_kerker_coefficient(n, 1e-10, self.x_s_m, self.m_s, self.m_s)
            coeff_Mie_pure_shell = tE_n_coefficient(n, self.x_s_m, self.m_s)
            
            assert np.isclose(coeff_aden_kerker, coeff_Mie_pure_shell, rtol=1e-6), f"AK {coeff_aden_kerker:.4f} does not match Mie for pure shell {coeff_Mie_pure_shell:.4f}, n={n}"

    def test_tE_n_coefficient_consistency_core_equals_shell_permittivity(self):
        for n in self.n_values:
            coeff_aden_kerker = tEn_aden_kerker_coefficient(n, self.x_c_m, self.x_s_m, self.m_c, self.m_c)
            coeff_Mie_pure_core = tE_n_coefficient(n, self.x_s_m, self.m_c)
            
            assert np.isclose(coeff_aden_kerker, coeff_Mie_pure_core, rtol=1e-6), f"AK {coeff_aden_kerker:.4f} does not match Mie for core equals shell permittivity {coeff_Mie_pure_core:.4f}, n={n}"

    def test_tM_n_coefficient_consistency_pure_core(self):
        
        for n in self.n_values:
            coeff_aden_kerker = tMn_aden_kerker_coefficient(n, self.x_c_m, self.x_c_m, self.m_c, self.m_s)
            coeff_Mie_pure_core = tM_n_coefficient(n, self.x_c_m, self.m_c)
            
            assert np.isclose(coeff_aden_kerker, coeff_Mie_pure_core, rtol=1e-6), f"AK {coeff_aden_kerker:.4f} does not match Mie for pure core {coeff_Mie_pure_core:.4f}, n={n}"

    def test_tM_n_coefficient_consistency_pure_shell(self):
        
        for n in self.n_values:
            coeff_aden_kerker = tMn_aden_kerker_coefficient(n, 1e-10, self.x_s_m, self.m_s, self.m_s)
            coeff_Mie_pure_shell = tM_n_coefficient(n, self.x_s_m, self.m_s)
            
            assert np.isclose(coeff_aden_kerker, coeff_Mie_pure_shell, rtol=1e-6), f"AK {coeff_aden_kerker:.4f} does not match Mie for pure shell {coeff_Mie_pure_shell:.4f}, n={n}"


class Test_order_Selection:
    def test_select_multipole_orders_sphere_small_limit(self):
        size_parameter = 0.0001  # Very small particle
        medium_permittivity = 1.2
        particle_permittivity = np.array([2.0, 4.0])  # Example permittivity values
        m = (particle_permittivity / medium_permittivity)**0.5
        
        N_electric, N_magnetic = select_multipole_orders_sphere(size_parameter, m)
        
        assert N_electric == 1, f"Expected 1 electric multipole order for small sphere, got {N_electric}"
        assert N_magnetic == 0, f"Expected 0 magnetic multipole orders for small sphere, got {N_magnetic}"
