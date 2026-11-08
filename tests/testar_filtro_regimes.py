"""Causalidade, lacunas e paridade do filtro do notebook."""

import ast
from pathlib import Path
import unittest

import nbformat
import numpy as np

from filtro_regimes import filtrar

RAIZ = Path(__file__).resolve().parents[1]


class FiltroRegimes(unittest.TestCase):
    parametros = ([0.5, -0.2], [4.0, 1.0], [[0.7, 0.3], [0.1, 0.9]], [0.6, 0.4])

    def test_futuro_nao_altera_probabilidades_anteriores(self):
        original = filtrar([0.2, np.nan, 1.1, -0.4, 0.0], *self.parametros)
        alterada = filtrar([0.2, np.nan, 1.1, 4.0, -6.0], *self.parametros)
        for antes, depois in zip(original[:2], alterada[:2]):
            np.testing.assert_array_equal(antes[:3], depois[:3])
            self.assertFalse(np.array_equal(antes[3:], depois[3:]))

    def test_lacuna_avanca_transicao_sem_atualizar_por_preco(self):
        atual, futura, medias, variancias, transicao = filtrar([0.2, np.nan], *self.parametros)
        np.testing.assert_array_equal(variancias, [1., 4.])
        np.testing.assert_array_equal(medias, [-0.2, 0.5])
        np.testing.assert_array_equal(atual[1], futura[0])
        np.testing.assert_allclose(futura[1], atual[1] @ transicao)
        np.testing.assert_allclose(futura.sum(axis=1), 1.)

    def test_copia_do_notebook_espelha_o_modulo(self):
        modulo = ast.parse((RAIZ / "src" / "filtro_regimes.py").read_text(encoding="utf-8"))
        nb = nbformat.read(RAIZ / "notebooks" / "Ibovespa_Regimes_de_Risco.ipynb", as_version=4)
        celula = next(c for c in nb.cells if "Fallback gerado a partir de filtro_regimes.py" in c.source)
        esperado = next(n for n in modulo.body if isinstance(n, ast.FunctionDef))
        copia = next(n for n in ast.walk(ast.parse(celula.source))
                     if isinstance(n, ast.FunctionDef) and n.name == "filtrar")
        self.assertEqual(ast.dump(copia), ast.dump(esperado))


if __name__ == "__main__":
    unittest.main()
