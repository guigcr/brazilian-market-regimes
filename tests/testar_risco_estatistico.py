"""Métricas de cauda, lacunas de calendário e paridade Colab."""
import ast
from pathlib import Path
import unittest

import nbformat
import numpy as np
import pandas as pd

import risco_estatistico as risco

RAIZ = Path(__file__).resolve().parents[1]


class RiscoEstatistico(unittest.TestCase):
    def test_intervalo_respeita_segmentos_separados(self):
        sessoes = pd.bdate_range("2026-01-05",periods=5)
        adjacentes = risco.adjacencia_sessoes(sessoes,sessoes[[0,1,3,4]])
        np.testing.assert_array_equal(adjacentes,[True,False,True])
        # Cada segmento constante preserva seu peso em toda reamostragem.
        np.testing.assert_allclose(risco.intervalo_media_pareada([0,0,1,1],adjacentes),[.5,.5])
        pares,p = risco.agrupamento_violacoes([False,True,True,False],adjacentes,rep=50)
        self.assertEqual(pares,0)
        self.assertEqual(p,1.)

    def test_fz0_e_ordem_do_cvar(self):
        perdas = [.02,.06]
        v,e = [.03,.03],[.05,.05]
        esperado = np.array([0,.03])/(.05*.05)+.03/.05+np.log(.05)-1
        np.testing.assert_allclose(risco.perda_fz0(perdas,v,e,.05),esperado)
        with self.assertRaisesRegex(ValueError,"CVaR"):
            risco.perda_fz0(perdas,v,[.01,.01],.05)

    def test_fallbacks_colab_espelham_modulo(self):
        funcoes = {n.name:ast.dump(n) for n in ast.parse(Path(risco.__file__).read_text()).body
                   if isinstance(n,ast.FunctionDef)}
        for nome in ["Ibovespa_Regimes_de_Risco.ipynb","BOVA11_Volatility_Research_Lab.ipynb"]:
            with self.subTest(notebook=nome):
                nb = nbformat.read(RAIZ/"notebooks"/nome,as_version=4)
                celula = next(c for c in nb.cells if "Fallback gerado a partir de risco_estatistico.py" in c.source)
                encontrados = {n.name:ast.dump(n) for n in ast.walk(ast.parse(celula.source))
                               if isinstance(n,ast.FunctionDef) and n.name in funcoes}
                esperados = set(funcoes) if "Volatility" in nome else {"adjacencia_sessoes","intervalo_media_pareada"}
                self.assertEqual(set(encontrados),esperados)
                for funcao in esperados:
                    self.assertEqual(encontrados[funcao],funcoes[funcao])


if __name__=="__main__":
    unittest.main()
