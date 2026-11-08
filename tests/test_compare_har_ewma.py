"""Check HAC variance and preservation of calendar gaps."""

import unittest

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac

from compare_har_ewma import diebold_mariano


class DieboldMariano(unittest.TestCase):
    def test_error_hac_matches_com_newey_west_em_series_contiguous(self):
        differences = pd.Series([-0.2, -0.1, 0.4, 0.3, -0.1, -0.3, -0.5, 0.1])
        model = OLS(differences.to_numpy(), np.ones(len(differences))).fit()
        expected = np.sqrt(cov_hac(model, nlags=2, use_correction=False)[0, 0])
        result = diebold_mariano(differences, 2)
        self.assertAlmostEqual(result["standard_error"], expected)

    def test_gap_does_not_joins_residuals_de_sessions_distant(self):
        differences = pd.Series([-0.4, np.nan, 0.2])
        result = diebold_mariano(differences, 1)
        self.assertAlmostEqual(result["standard_error"], np.sqrt(.3**2 + .3**2) / 2)
        self.assertLess(result["one_sided_p_model_lower"], .5)

    def test_rejects_losses_identical_sem_variance_estimable(self):
        with self.assertRaisesRegex(ValueError, "degenerate"):
            diebold_mariano(pd.Series([0., 0., 0.]), 1)
