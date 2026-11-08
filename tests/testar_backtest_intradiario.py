"""Custos, convenções de risco e autonomia do notebook de regimes."""

import ast
from pathlib import Path
import unittest

import nbformat
import numpy as np
import pandas as pd

import backtest_intradiario as backtest

RAIZ = Path(__file__).resolve().parents[1]


class BacktestIntradiario(unittest.TestCase):
    def test_compra_venda_e_caixa_com_custos(self):
        pesos = pd.DataFrame({"politica": [1.0, -0.5, 0.0]})
        retornos = pd.Series([-0.02, -0.04, 0.50])
        metricas, capital, liquido = backtest.simular_intradiario(
            pesos, retornos, 0.001, detalhar=True
        )
        esperado = [-0.022, 0.019, 0.0]
        np.testing.assert_allclose(liquido["politica"], esperado)
        np.testing.assert_allclose(capital["politica"], np.cumprod(1 + np.array(esperado)))
        linha = metricas.loc["politica"]
        self.assertAlmostEqual(linha.drawdown, -0.022)
        self.assertAlmostEqual(linha.CVaR_95, 0.022)
        self.assertAlmostEqual(linha.soma_custos_diarios, 0.003)
        self.assertAlmostEqual(linha.exposicao_media, 0.5)
        self.assertEqual(linha.operacoes, 2)
        self.assertEqual(linha.n_pregoes, 3)

    def test_cvar_com_massa_fracionaria(self):
        self.assertAlmostEqual(backtest.cvar_empirico([-0.08, -0.02, 0.05], 0.5), 0.06)

    def test_recusa_painel_desalinhado(self):
        pesos = pd.DataFrame({"politica": [1.0, 1.0]}, index=[1, 2])
        retornos = pd.Series([0.01, 0.02], index=[2, 1])
        with self.assertRaisesRegex(ValueError, "painel cronológico"):
            backtest.simular_intradiario(pesos, retornos, 0.001)

    def test_fallback_colab_espelha_modulo(self):
        modulo = ast.parse(Path(backtest.__file__).read_text(encoding="utf-8"))
        notebook = nbformat.read(RAIZ / "notebooks" / "Ibovespa_Regimes_de_Risco.ipynb", as_version=4)
        celula = next(c for c in notebook.cells if "Fallback gerado a partir de backtest_intradiario.py" in c.source)
        esperado = [ast.dump(n) for n in modulo.body if isinstance(n, ast.FunctionDef)]
        fallback = [ast.dump(n) for n in ast.walk(ast.parse(celula.source)) if isinstance(n, ast.FunctionDef)]
        self.assertEqual(fallback, esperado)


if __name__ == "__main__":
    unittest.main()
