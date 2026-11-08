"""Check timing cutoffs in the volatility lab's HAR and XGBoost cell."""

from pathlib import Path
import unittest

import nbformat
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class VolatilityCausality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "BOVA11_Volatility_Research_Lab.ipynb"
        nb = nbformat.read(path, as_version=4)
        cls.cell = next(c.source for c in nb.cells if "def fatores_har(" in c.source)
        cls.sessions = pd.bdate_range("2024-01-02", periods=170)
        cls.origins = cls.sessions[135:150]
        cls.variance = pd.Series(np.random.default_rng(42).uniform(1e-5, 4e-4, 170),
                                 index=cls.sessions)
        cls.external = pd.Series(np.linspace(100., 130., 170), index=cls.sessions)

    def run_forecast(self, variance, external):
        fits = []

        class EstimadorAuditoria:
            def __init__(self, **kwargs):
                self.model = LinearRegression()

            def fit(self, x, y):
                fits.append((x.copy(), y.copy()))
                self.model.fit(x, y)
                return self

            def predict(self, x):
                return self.model.predict(x)

        ns = {
            "np": np, "pd": pd, "LinearRegression": LinearRegression,
            "XGBRegressor": EstimadorAuditoria, "SEED": 42, "STEP_REFIT": 5,
            "EPS": 1e-10, "ASSET": "BOVA11", "sessions": self.sessions,
            "r": np.sqrt(variance), "variance_obs": variance,
            "target_date": self.sessions.to_series().shift(-1),
            "origins_forecast": self.origins, "forecasts": {},
            "read_snapshot": lambda _: external.to_frame("Close"),
        }
        exec(compile(self.cell, "laboratorio_volatility", "exec"), ns)
        return ns["ml"], ns["extras"], fits

    def test_train_includes_only_targets_known_no_first_close(self):
        _, _, fits = self.run_forecast(self.variance, self.external)
        self.assertEqual(len(fits), 3)
        for start, (x, y) in zip(self.origins[::5], fits):
            position = self.sessions.get_loc(start)
            self.assertEqual(x.index[-1], self.sessions[position - 1])
            self.assertAlmostEqual(y.iloc[-1], self.variance.loc[start] * 1e4)
            self.assertNotIn(start, x.index)

    def test_shock_future_e_quote_externa_do_day_preserve_forecast(self):
        threshold = self.origins[5]
        before, extras, _ = self.run_forecast(self.variance, self.external)
        variance = self.variance.where(self.sessions <= threshold, self.variance * 4)
        external = self.external.where(self.sessions < threshold, self.external * 1.2)
        after, extras_after, _ = self.run_forecast(variance, external)
        pd.testing.assert_frame_equal(extras.loc[:threshold], extras_after.loc[:threshold], check_exact=True)
        pd.testing.assert_frame_equal(before.loc[:threshold], after.loc[:threshold], check_exact=True)
        self.assertFalse(before.loc[before.index > threshold].equals(after.loc[after.index > threshold]))


if __name__ == "__main__":
    unittest.main()
