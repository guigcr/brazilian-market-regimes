"""Verifica a variância HAC e a preservação das lacunas na comparação."""

import unittest

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac

from comparar_har_ewma import diebold_mariano


class DieboldMariano(unittest.TestCase):
    def test_erro_hac_coincide_com_newey_west_em_serie_contigua(self):
        diferencas = pd.Series([-0.2, -0.1, 0.4, 0.3, -0.1, -0.3, -0.5, 0.1])
        modelo = OLS(diferencas.to_numpy(), np.ones(len(diferencas))).fit()
        esperado = np.sqrt(cov_hac(modelo, nlags=2, use_correction=False)[0, 0])
        resultado = diebold_mariano(diferencas, 2)
        self.assertAlmostEqual(resultado["erro_padrao"], esperado)

    def test_lacuna_nao_aproxima_residuos_de_pregoes_distantes(self):
        diferencas = pd.Series([-0.4, np.nan, 0.2])
        resultado = diebold_mariano(diferencas, 1)
        self.assertAlmostEqual(resultado["erro_padrao"], np.sqrt(.3**2 + .3**2) / 2)
        self.assertLess(resultado["p_unilateral_modelo_menor"], .5)

    def test_rejeita_perdas_identicas_sem_variancia_estimavel(self):
        with self.assertRaisesRegex(ValueError, "degenerada"):
            diebold_mariano(pd.Series([0., 0., 0.]), 1)
