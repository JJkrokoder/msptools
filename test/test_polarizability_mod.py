import numpy as np
from scipy.special import spherical_jn as sph_jn
from scipy.special import spherical_yn as sph_yn
from msptools.polarizability_mod import (Core_Shell_Clausius_Mossotti,
                                            polarizability_to_matrix, 
                                            Clausius_Mossotti,
                                            Mie_size_dipole_approximation,
                                            Mie_electric_dipole_polarizability,
                                            Aden_Kerker_core_shell_polarizability,
                                            compute_sphere_polarizability_DA,
                                            Mie_electric_quadrupole_polarizability)
from msptools.tools.unit_calcs import nm_to_eV, frequency_to_wavenumber_nm
from msptools.permittivity import permittivity_ridx


def test_Clausius_Mossotti():
    radius = 0.08  # um
    medium_permittivity = 1
    e1 = -2.5676
    e2 = 3.6391
    particle_permittivity = e1 + e2 * 1j  # e.g., gold at 500 nm

    alpha = Clausius_Mossotti(radius, medium_permittivity, particle_permittivity)
    expected_alpha = 4 * np.pi * radius**3 * (particle_permittivity - medium_permittivity) / (particle_permittivity + 2 * medium_permittivity)
    
    assert np.isclose(alpha, expected_alpha), f"Expected {expected_alpha}, got {alpha}"

class Test_Core_Shell:
    
    e1 = -2.5676
    e2 = 3.6391
    medium_permittivity = 1.35**2
    
    def test_Core_Shell_Clausius_Mossotti_no_shell(self):
        radius_core = 0.08  # um
        radius_shell = 0.08 + 1e-10  # um, tiny shell
        medium_permittivity = self.medium_permittivity
        
        particle_permittivity_core = self.e1  
        particle_permittivity_shell = self.e2  

        alpha = Core_Shell_Clausius_Mossotti(radius_core, radius_shell, medium_permittivity, particle_permittivity_core, particle_permittivity_shell)
        expected_alpha = Clausius_Mossotti(radius_core, medium_permittivity, particle_permittivity_core)
        
        assert np.isclose(alpha, expected_alpha), f"Expected {expected_alpha}, got {alpha}"
    
    def test_Core_Shell_Clausius_Mossotti_no_core(self):
        radius_core = 1e-10  # um, tiny core
        radius_shell = 0.08  # um
        medium_permittivity = self.medium_permittivity
        
        particle_permittivity_core = self.e1  
        particle_permittivity_shell = self.e2  

        alpha = Core_Shell_Clausius_Mossotti(radius_core, radius_shell, medium_permittivity, particle_permittivity_core, particle_permittivity_shell)
        expected_alpha = Clausius_Mossotti(radius_shell, medium_permittivity, particle_permittivity_shell)
        
        assert np.isclose(alpha, expected_alpha), f"Expected {expected_alpha}, got {alpha}"
    
    def test_Aden_Kerker_small_particle_consistency(self):
        radius_core = 0.08  # um
        radius_shell = 0.16
        wave_number = 2 * np.pi / 1000
        medium_permittivity = self.medium_permittivity
        
        particle_permittivity_core = self.e1  
        particle_permittivity_shell = self.e2  

        alpha_ak = Aden_Kerker_core_shell_polarizability(radius_core, radius_shell, medium_permittivity, particle_permittivity_core, particle_permittivity_shell, wave_number) 
        alpha_cm = Core_Shell_Clausius_Mossotti(radius_core, radius_shell, medium_permittivity, particle_permittivity_core, particle_permittivity_shell)
        
        assert np.isclose(alpha_ak, alpha_cm, rtol=1e-6), f"Expected {alpha_cm}, got {alpha_ak}"
    
    def test_Aden_Kerker_pure_core_consistency(self):
        radius_core = 0.08  # um
        radius_shell = 0.08 # um, no shell
        wave_number = 2 * np.pi / 0.5
        medium_permittivity = self.medium_permittivity
        
        particle_permittivity_core = self.e1  
        particle_permittivity_shell = self.e2  

        alpha_ak = Aden_Kerker_core_shell_polarizability(radius_core, radius_shell, medium_permittivity, particle_permittivity_core, particle_permittivity_shell, wave_number) 
        expected_alpha = Mie_electric_dipole_polarizability(radius_core, medium_permittivity, particle_permittivity_core, wave_number)
        
        assert np.isclose(alpha_ak, expected_alpha, rtol=1e-6), f"Expected {expected_alpha}, got {alpha_ak}"
    
    def test_Aden_Kerker_pure_shell_consistency(self):
        radius_core = 1e-10  # um, no core
        radius_shell = 0.08 # um, no core
        wave_number = 2 * np.pi / 0.5
        medium_permittivity = self.medium_permittivity
        
        particle_permittivity_core = self.e1  
        particle_permittivity_shell = self.e2  

        alpha_ak = Aden_Kerker_core_shell_polarizability(radius_core, radius_shell, medium_permittivity, particle_permittivity_core, particle_permittivity_shell, wave_number) 
        expected_alpha = Mie_electric_dipole_polarizability(radius_shell, medium_permittivity, particle_permittivity_shell, wave_number)
        
        assert np.isclose(alpha_ak, expected_alpha, rtol=1e-6), f"Expected {expected_alpha}, got {alpha_ak}"   
    

def test_negative_real_polarizability_at_532nm_Au():
    radius = 60  # um
    medium_permittivity = 1.33**2
    wavelength_nm = 532  # nm
    frequency_eV = nm_to_eV(wavelength_nm)
    particle_permittivity = permittivity_ridx(frequency_eV, 'Au')

    alpha = Mie_size_dipole_approximation(radius, medium_permittivity, particle_permittivity, frequency_to_wavenumber_nm(frequency_eV))
    
    assert alpha.real < 0, f"Expected negative real part of polarizability for Au, got {alpha.real}"

def test_Zhou_data_with_Mie_dipole_approximation():
    radius = 80  # nm
    medium_permittivity = 1.33**2
    wavelengths_nm = np.array([400.9234828496042, 412.0052770448549, 422.16358839050133, 431.39841688654354, 439.70976253298153, 448.94459102902374, 457.2559366754617, 465.5672823218997, 472.9551451187335, 479.41952506596306, 486.8073878627968, 494.19525065963063, 503.43007915567284, 509.8944591029024, 515.4353562005277, 521.8997361477573, 529.2875989445911, 536.6754617414248, 544.0633245382586, 551.4511873350923, 558.8390501319261, 566.2269129287599, 573.6147757255936, 579.155672823219, 584.6965699208444, 589.3139841688654, 593.9313984168865, 597.6253298153034, 601.3192612137203, 605.0131926121372, 608.7071240105541, 611.4775725593668, 615.1715039577837, 617.9419525065963, 620.7124010554089, 623.4828496042217, 625.3298153034301, 628.1002638522427, 631.7941952506596, 634.5646437994723, 637.335092348285, 640.1055408970976, 642.8759894459104, 645.646437994723, 648.4168865435356, 651.1873350923483, 653.957783641161, 657.6517150395778, 661.3456464379947, 664.1160949868074, 667.8100263852243, 670.5804749340369, 673.3509234828496, 676.1213720316623, 679.8153034300792, 682.5857519788918, 686.2796833773087, 689.9736147757255, 694.5910290237467, 698.2849604221635, 702.9023746701847, 708.4432717678101, 713.9841688654353, 720.448548812665, 727.8364116094987, 736.1477572559368, 744.4591029023748, 751.8469656992085, 761.0817941952507, 770.3166226912929, 780.4749340369393, 790.6332453825858, 799.8680738786279, 808.1794195250659, 817.4142480211083, 828.4960422163588, 838.6543535620053, 848.8126649076517, 858.9709762532982, 870.9762532981531, 882.0580474934037, 894.0633245382586, 905.1451187335092, 915.3034300791556, 927.3087071240105, 941.1609498680739, 953.1662269129288, 966.0949868073878, 979.023746701847, 993.7994722955145, 1007.6517150395779, 1021.5039577836412, 1039.9736147757258, 1055.6728232189973, 1072.2955145118735, 1086.1477572559365, 1100.9234828496042])  # nm
    frequency_eV = nm_to_eV(wavelengths_nm)
    particle_permittivity = permittivity_ridx(frequency_eV, 'Au')

    alpha_nm = Mie_size_dipole_approximation(radius, medium_permittivity, particle_permittivity, frequency_to_wavenumber_nm(frequency_eV))
    alpha_real_zhou = (1e-23)*np.array([61.97183098591546, 59.15492957746477, 53.521126760563334, 50.704225352112644, 47.887323943661954, 42.25352112676052, 36.61971830985914, 28.169014084507012, 14.084507042253506, -2.8169014084507467, -25.35211267605638, -47.88732394366201, -76.05633802816902, -101.4084507042254, -123.94366197183103, -143.66197183098592, -160.56338028169017, -174.64788732394368, -185.9154929577465, -188.73239436619718, -191.5492957746479, -185.9154929577465, -174.64788732394368, -160.56338028169017, -140.84507042253523, -121.12676056338029, -98.59154929577466, -78.87323943661977, -53.52112676056339, -28.16901408450707, 0, 28.169014084507012, 59.15492957746477, 87.32394366197178, 112.67605633802816, 140.84507042253517, 169.01408450704218, 197.1830985915492, 228.169014084507, 261.9718309859154, 295.7746478873239, 332.394366197183, 369.0140845070422, 400, 433.8028169014084, 467.6056338028168, 501.4084507042253, 538.0281690140845, 571.8309859154929, 611.2676056338028, 647.8873239436618, 676.056338028169, 704.2253521126759, 729.5774647887324, 757.7464788732393, 783.0985915492956, 811.2676056338028, 839.4366197183099, 864.7887323943662, 887.3239436619717, 912.676056338028, 935.2112676056338, 954.9295774647887, 974.6478873239435, 994.3661971830984, 1008.450704225352, 1016.9014084507041, 1019.7183098591549, 1022.5352112676055, 1022.5352112676055, 1019.7183098591549, 1014.0845070422533, 1011.2676056338028, 1002.8169014084506, 997.1830985915492, 988.732394366197, 980.2816901408451, 969.0140845070421, 960.5633802816901, 949.2957746478871, 940.8450704225352, 929.5774647887324, 921.1267605633802, 912.676056338028, 904.2253521126759, 895.7746478873239, 887.3239436619717, 878.8732394366195, 870.4225352112676, 861.9718309859154, 853.5211267605632, 845.0704225352113, 836.6197183098591, 828.1690140845069, 822.5352112676055, 816.9014084507041, 811.2676056338028])
    
    assert np.allclose(alpha_nm.real*(1e-9)**3, alpha_real_zhou, atol=1e-5), "Mie dipole approximation does not match Zhou data within tolerance."


def test_Mie_dipole_for_small_radius():
    radius = 0.05  # nm, very small particle
    medium_permittivity = 1.33**2
    wavelength_nm = 1600  # nm
    frequency_eV = nm_to_eV(wavelength_nm)
    particle_permittivity = permittivity_ridx(frequency_eV, 'Au')
    size_parameter = frequency_to_wavenumber_nm(frequency_eV) * radius

    alpha_mie_approx = Mie_size_dipole_approximation(radius, medium_permittivity, particle_permittivity, frequency_to_wavenumber_nm(frequency_eV))
    alpha_mie = Mie_electric_dipole_polarizability(radius, medium_permittivity, particle_permittivity, frequency_to_wavenumber_nm(frequency_eV))
    alpha_cm = Clausius_Mossotti(radius, medium_permittivity, particle_permittivity)

    assert np.isclose(alpha_mie_approx.real, alpha_mie.real), f"Mie dipole approx real part {alpha_mie_approx.real:.2f} not close to Mie electric dipole {alpha_mie.real:.2f}. rerror: {abs(alpha_mie_approx.real - alpha_mie.real)/abs(alpha_mie.real):.2e}, aerror: {abs(alpha_mie_approx.real - alpha_mie.real):.2e}, size_param^4: {size_parameter**4:.2e}"
    assert np.isclose(alpha_mie_approx.imag, alpha_mie.imag), f"Mie dipole approx imag part {alpha_mie_approx.imag:.2f} not close to Mie electric dipole {alpha_mie.imag:.2f}. rerror: {abs(alpha_mie_approx.imag - alpha_mie.imag)/abs(alpha_mie.imag):.2e}, aerror: {abs(alpha_mie_approx.imag - alpha_mie.imag):.2e}, size_param^4: {size_parameter**4:.2e}"
    assert np.isclose(alpha_mie.real, alpha_cm.real), f"Mie electric dipole real part {alpha_mie.real:.2f} not close to Clausius-Mossotti real part {alpha_cm.real:.2f}. rerror: {abs(alpha_mie.real - alpha_cm.real)/abs(alpha_cm.real):.2e}, aerror: {abs(alpha_mie.real - alpha_cm.real):.2e}"
    assert np.isclose(alpha_mie.imag, alpha_cm.imag), f"Mie electric dipole imag part {alpha_mie.imag:.2f} not close to Clausius-Mossotti imag part {alpha_cm.imag:.2f}. rerror: {abs(alpha_mie.imag - alpha_cm.imag)/abs(alpha_cm.imag):.2e}, aerror: {abs(alpha_mie.imag - alpha_cm.imag):.2e}"

def test_one_polarizability_to_matrix():
    polarizability = 1.0 + 0.5j
    num_particles = 1
    expected_matrix = np.array([[1.0 + 0.5j, 0, 0],
                                [0, 1.0 + 0.5j, 0],
                                [0, 0, 1.0 + 0.5j]])
    
    result_matrix = polarizability_to_matrix(polarizability, num_particles, 3, np)
    
    assert np.allclose(result_matrix, expected_matrix), f"Expected {expected_matrix}, got {result_matrix}"

class Test_spher_pol_function:
    
    medium_permittivity = 1.33**2
    particle_material = 'Au'
    
    def test_sphere_polarizability_spectra_shape(self):
        radius_nm = 100
        wavelengths_nm = np.linspace(400, 800, 10)

        polarizabilities = compute_sphere_polarizability_DA(radius_nm, self.medium_permittivity, self.particle_material, wavelengths_nm)

        assert isinstance(polarizabilities, np.ndarray), f"Expected ndarray, got {type(polarizabilities)}"
        assert polarizabilities.shape == (len(wavelengths_nm),), f"Expected shape ({len(wavelengths_nm)},), got {polarizabilities.shape}"
        
    def test_sphere_polarizability_DA_consistency_with_Mie_dipole(self):
        radius_nm = 5
        wavelengths_nm = np.array([400.9234828496042, 412.0052770448549, 422.16358839050133, 431.39841688654354, 439.70976253298153, 448.94459102902374, 457.2559366754617, 465.5672823218997, 472.9551451187335, 479.41952506596306])  # nm

        polarizabilities_DA = compute_sphere_polarizability_DA(radius_nm, self.medium_permittivity, self.particle_material, wavelengths_nm)
        
        for i in range(len(wavelengths_nm)):
            frequency_eV = nm_to_eV(wavelengths_nm[i])
            particle_permittivity = permittivity_ridx(frequency_eV, self.particle_material)
            expected_alpha = Mie_electric_dipole_polarizability(radius_nm, self.medium_permittivity, particle_permittivity, frequency_to_wavenumber_nm(frequency_eV))
            approximated_alpha = Mie_size_dipole_approximation(radius_nm, self.medium_permittivity, particle_permittivity, frequency_to_wavenumber_nm(frequency_eV))
            assert np.isclose(polarizabilities_DA[i], expected_alpha, rtol=1e-4, atol=1e-5), f"At wavelength {wavelengths_nm[i]} nm: Expected {expected_alpha}, got {polarizabilities_DA[i]}"
            assert np.isclose(polarizabilities_DA[i], approximated_alpha, rtol=1e-4, atol=100), f"At wavelength {wavelengths_nm[i]} nm: Expected approximated {approximated_alpha}, got {polarizabilities_DA[i]}"
    
    def test_plasmon_resonance_peak(self):
        radius_nm = 50
        wavelengths_nm = np.linspace(400, 800, 100)  # nm

        polarizabilities_DA = compute_sphere_polarizability_DA(radius_nm, self.medium_permittivity, self.particle_material, wavelengths_nm)
        
        resonance_index = np.argmax(polarizabilities_DA.imag)
        resonance_wavelength = wavelengths_nm[resonance_index]
        
        assert 500 < resonance_wavelength < 600, f"Expected plasmon resonance between 500-600 nm for Au, got {resonance_wavelength} nm"

class Test_sphere_quadrupole_polarizability:
    
    medium_permittivity = 1.33**2
    m_1_test = 1.5/medium_permittivity**0.5
    radius_1 = 50
    
    def test_function_small_particle_limit(self):
        radius_nm = np.array([1, 2, 3, 4, 5])  # nm
        wavelengths_nm = 1000  # nm

        polarizabilities_quad = Mie_electric_quadrupole_polarizability(radius_nm,
                                                                       self.medium_permittivity,
                                                                       self.m_1_test**2*self.medium_permittivity,
                                                                       frequency_to_wavenumber_nm(nm_to_eV(wavelengths_nm)))
        
        quad_pol_small = 8/3 * np.pi * radius_nm**5 * (self.m_1_test**2 - 1) / (2*self.m_1_test**2 + 3)
        
        assert np.allclose(polarizabilities_quad, quad_pol_small, rtol=1e-3), f"Expected {quad_pol_small}, got {polarizabilities_quad}"
    
    
    def test_same_permittivity_no_polarizability(self):
        wavelengths_nm = 1000  # nm

        polarizabilities_quad = Mie_electric_quadrupole_polarizability(self.radius_1,
                                                                       self.medium_permittivity,
                                                                       self.medium_permittivity,
                                                                       frequency_to_wavenumber_nm(nm_to_eV(wavelengths_nm)))
        
        assert np.isclose(polarizabilities_quad, 0), f"Expected 0 polarizability for same permittivity, got {polarizabilities_quad}"
    
    def test_k_scaling(self):
        k_magnitudes = np.array([2 * np.pi / wl for wl in np.linspace(400, 800, 10)])*self.medium_permittivity**0.5
        x_constant = 0.1
        radii = x_constant / k_magnitudes  # Keep x constant

        polarizabilities_quad = Mie_electric_quadrupole_polarizability(radii,
                                                                       self.medium_permittivity,
                                                                       self.m_1_test**2*self.medium_permittivity,
                                                                       k_magnitudes)
        
        expected_scaling = (k_magnitudes[0]/k_magnitudes)**5
        
        computed_scaling = polarizabilities_quad / polarizabilities_quad[0]  # Normalize to first value
        
        assert np.allclose(computed_scaling, expected_scaling, rtol=1e-3), f"Expected scaling {expected_scaling}, got {computed_scaling}"
        
    