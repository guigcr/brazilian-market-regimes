"""Check tail metrics, calendar gaps, and Colab parity."""
import ast
from pathlib import Path
import unittest

import nbformat
import numpy as np
import pandas as pd

import risk_statistics as risk

ROOT = Path(__file__).resolve().parents[1]


class RiskStatistics(unittest.TestCase):
    def test_interval_respects_segments_separate(self):
        sessions = pd.bdate_range("2026-01-05",periods=5)
        adjacent = risk.session_adjacency(sessions,sessions[[0,1,3,4]])
        np.testing.assert_array_equal(adjacent,[True,False,True])
        # Cada segmento constant preserva seu peso em toda resampling.
        np.testing.assert_allclose(risk.paired_mean_interval([0,0,1,1],adjacent),[.5,.5])
        pairs,p = risk.violation_clustering([False,True,True,False],adjacent,rep=50)
        self.assertEqual(pairs,0)
        self.assertEqual(p,1.)

    def test_fz0_e_order_do_cvar(self):
        losses = [.02,.06]
        v,e = [.03,.03],[.05,.05]
        expected = np.array([0,.03])/(.05*.05)+.03/.05+np.log(.05)-1
        np.testing.assert_allclose(risk.fz0_loss(losses,v,e,.05),expected)
        with self.assertRaisesRegex(ValueError,"CVaR"):
            risk.fz0_loss(losses,v,[.01,.01],.05)

    def test_fallbacks_colab_match_module(self):
        functions = {n.name:ast.dump(n) for n in ast.parse(Path(risk.__file__).read_text()).body
                   if isinstance(n,ast.FunctionDef)}
        for name in ["Ibovespa_Risk_Regimes.ipynb","BOVA11_Volatility_Research_Lab.ipynb"]:
            with self.subTest(notebook=name):
                nb = nbformat.read(ROOT/"notebooks"/name,as_version=4)
                cell = next(c for c in nb.cells if "Fallback generated from risk_statistics.py" in c.source)
                found = {n.name:ast.dump(n) for n in ast.walk(ast.parse(cell.source))
                               if isinstance(n,ast.FunctionDef) and n.name in functions}
                expected = set(functions) if "Volatility" in name else {"session_adjacency","paired_mean_interval"}
                self.assertEqual(set(found),expected)
                for function in expected:
                    self.assertEqual(found[function],functions[function])


if __name__=="__main__":
    unittest.main()
