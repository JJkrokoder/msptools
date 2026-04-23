from msptools.tools.mie_theory import tE_n_coefficient, tEn_aden_kerker_coefficient, tM_n_coefficient, tMn_aden_kerker_coefficient
import numpy as np


class Test_Aden_Kerker_Coefficient:
    
    n = 1
    x_core = 1
    x_shell = 2
    eps_core = 1.5
    eps_shell = 1.5
    eps_m = n**2
    
    def test_tE_n_coefficient_consistency(self):
        
        coeff_aden_kerker = tEn_aden_kerker_coefficient(self.n,self.x_core, self.x_core, self.eps_core, self.eps_shell, self.eps_m)
        coeff_Mie_pure_core = tE_n_coefficient(self.n, self.x_core, self.eps_core, self.eps_m)
        
        assert np.isclose(coeff_aden_kerker, coeff_Mie_pure_core, rtol=1e-6), f"Aden-Kerker coefficient {coeff_aden_kerker} does not match standard Mie coefficient for pure core {coeff_Mie_pure_core}"

    def test_tM_n_coefficient_consistency(self):
        
        coeff_aden_kerker = tMn_aden_kerker_coefficient(self.n,self.x_core, self.x_core, self.eps_core, self.eps_shell, self.eps_m)
        coeff_Mie_pure_core = tM_n_coefficient(self.n, self.x_core, self.eps_core, self.eps_m)
        
        assert np.isclose(coeff_aden_kerker, coeff_Mie_pure_core, rtol=1e-6), f"Aden-Kerker coefficient {coeff_aden_kerker} does not match standard Mie coefficient for pure core {coeff_Mie_pure_core}"
