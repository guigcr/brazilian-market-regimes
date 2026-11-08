"""Check costs, risk conventions, and the standalone regime notebook."""

import ast
from pathlib import Path
import unittest

import nbformat
import numpy as np
import pandas as pd

import intraday_backtest as backtest

ROOT = Path(__file__).resolve().parents[1]


class IntradayBacktest(unittest.TestCase):
    def test_buy_sell_e_cash_com_costs(self):
        weights = pd.DataFrame({"policy": [1.0, -0.5, 0.0]})
        returns = pd.Series([-0.02, -0.04, 0.50])
        metrics, equity, net = backtest.simulate_intraday(
            weights, returns, 0.001, include_details=True
        )
        expected = [-0.022, 0.019, 0.0]
        np.testing.assert_allclose(net["policy"], expected)
        np.testing.assert_allclose(equity["policy"], np.cumprod(1 + np.array(expected)))
        row = metrics.loc["policy"]
        self.assertAlmostEqual(row.drawdown, -0.022)
        self.assertAlmostEqual(row.CVaR_95, 0.022)
        self.assertAlmostEqual(row.total_daily_costs, 0.003)
        self.assertAlmostEqual(row.exposure_mean, 0.5)
        self.assertEqual(row.trades, 2)
        self.assertEqual(row.session_count, 3)

    def test_cvar_com_mass_fractional(self):
        self.assertAlmostEqual(backtest.empirical_cvar([-0.08, -0.02, 0.05], 0.5), 0.06)

    def test_rejects_panel_misaligned(self):
        weights = pd.DataFrame({"policy": [1.0, 1.0]}, index=[1, 2])
        returns = pd.Series([0.01, 0.02], index=[2, 1])
        with self.assertRaisesRegex(ValueError, "chronological panel"):
            backtest.simulate_intraday(weights, returns, 0.001)

    def test_fallback_colab_matches_module(self):
        module = ast.parse(Path(backtest.__file__).read_text(encoding="utf-8"))
        notebook = nbformat.read(ROOT / "notebooks" / "Ibovespa_Risk_Regimes.ipynb", as_version=4)
        cell = next(c for c in notebook.cells if "Fallback generated from intraday_backtest.py" in c.source)
        expected = [ast.dump(n) for n in module.body if isinstance(n, ast.FunctionDef)]
        fallback = [ast.dump(n) for n in ast.walk(ast.parse(cell.source)) if isinstance(n, ast.FunctionDef)]
        self.assertEqual(fallback, expected)


if __name__ == "__main__":
    unittest.main()
