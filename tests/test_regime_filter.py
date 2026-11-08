"""Check causality, gaps, and notebook parity for the regime filter."""

import ast
from pathlib import Path
import unittest

import nbformat
import numpy as np

from regime_filter import filter_regimes

ROOT = Path(__file__).resolve().parents[1]


class RegimeFilter(unittest.TestCase):
    parameters = ([0.5, -0.2], [4.0, 1.0], [[0.7, 0.3], [0.1, 0.9]], [0.6, 0.4])

    def test_future_does_not_change_probabilities_previous(self):
        original = filter_regimes([0.2, np.nan, 1.1, -0.4, 0.0], *self.parameters)
        modified = filter_regimes([0.2, np.nan, 1.1, 4.0, -6.0], *self.parameters)
        for before, after in zip(original[:2], modified[:2]):
            np.testing.assert_array_equal(before[:3], after[:3])
            self.assertFalse(np.array_equal(before[3:], after[3:]))

    def test_gap_advances_transition_sem_update_por_price(self):
        current, future, means, variances, transition = filter_regimes([0.2, np.nan], *self.parameters)
        np.testing.assert_array_equal(variances, [1., 4.])
        np.testing.assert_array_equal(means, [-0.2, 0.5])
        np.testing.assert_array_equal(current[1], future[0])
        np.testing.assert_allclose(future[1], current[1] @ transition)
        np.testing.assert_allclose(future.sum(axis=1), 1.)

    def test_copy_do_notebook_matches_o_module(self):
        module = ast.parse((ROOT / "src" / "regime_filter.py").read_text(encoding="utf-8"))
        nb = nbformat.read(ROOT / "notebooks" / "Ibovespa_Risk_Regimes.ipynb", as_version=4)
        cell = next(c for c in nb.cells if "Fallback generated from regime_filter.py" in c.source)
        expected = next(n for n in module.body if isinstance(n, ast.FunctionDef))
        copy = next(n for n in ast.walk(ast.parse(cell.source))
                     if isinstance(n, ast.FunctionDef) and n.name == "filter_regimes")
        self.assertEqual(ast.dump(copy), ast.dump(expected))


if __name__ == "__main__":
    unittest.main()
