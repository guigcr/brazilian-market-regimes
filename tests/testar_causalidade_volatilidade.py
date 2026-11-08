"""Audita os cortes temporais da célula de HAR e XGBoost do laboratório."""

from pathlib import Path
import unittest

import nbformat
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class CausalidadeVolatilidade(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        caminho = Path(__file__).resolve().parents[1] / "notebooks" / "BOVA11_Volatility_Research_Lab.ipynb"
        nb = nbformat.read(caminho, as_version=4)
        cls.celula = next(c.source for c in nb.cells if "def fatores_har(" in c.source)
        cls.sessoes = pd.bdate_range("2024-01-02", periods=170)
        cls.origens = cls.sessoes[135:150]
        cls.variancia = pd.Series(np.random.default_rng(42).uniform(1e-5, 4e-4, 170),
                                 index=cls.sessoes)
        cls.externo = pd.Series(np.linspace(100., 130., 170), index=cls.sessoes)

    def executar(self, variancia, externo):
        ajustes = []

        class EstimadorAuditoria:
            def __init__(self, **kwargs):
                self.modelo = LinearRegression()

            def fit(self, x, y):
                ajustes.append((x.copy(), y.copy()))
                self.modelo.fit(x, y)
                return self

            def predict(self, x):
                return self.modelo.predict(x)

        ns = {
            "np": np, "pd": pd, "LinearRegression": LinearRegression,
            "XGBRegressor": EstimadorAuditoria, "SEED": 42, "PASSO_REAJUSTE": 5,
            "EPS": 1e-10, "ATIVO": "BOVA11", "sessoes": self.sessoes,
            "r": np.sqrt(variancia), "variancia_obs": variancia,
            "data_alvo": self.sessoes.to_series().shift(-1),
            "origens_previsao": self.origens, "previsoes": {},
            "ler": lambda _: externo.to_frame("Close"),
        }
        exec(compile(self.celula, "laboratorio_volatilidade", "exec"), ns)
        return ns["ml"], ns["extras"], ajustes

    def test_treino_inclui_apenas_alvos_conhecidos_no_primeiro_fechamento(self):
        _, _, ajustes = self.executar(self.variancia, self.externo)
        self.assertEqual(len(ajustes), 3)
        for inicio, (x, y) in zip(self.origens[::5], ajustes):
            posicao = self.sessoes.get_loc(inicio)
            self.assertEqual(x.index[-1], self.sessoes[posicao - 1])
            self.assertAlmostEqual(y.iloc[-1], self.variancia.loc[inicio] * 1e4)
            self.assertNotIn(inicio, x.index)

    def test_choque_futuro_e_cotacao_externa_do_dia_preservam_previsao(self):
        corte = self.origens[5]
        antes, extras, _ = self.executar(self.variancia, self.externo)
        variancia = self.variancia.where(self.sessoes <= corte, self.variancia * 4)
        externo = self.externo.where(self.sessoes < corte, self.externo * 1.2)
        depois, extras_depois, _ = self.executar(variancia, externo)
        pd.testing.assert_frame_equal(extras.loc[:corte], extras_depois.loc[:corte], check_exact=True)
        pd.testing.assert_frame_equal(antes.loc[:corte], depois.loc[:corte], check_exact=True)
        self.assertFalse(antes.loc[antes.index > corte].equals(depois.loc[depois.index > corte]))


if __name__ == "__main__":
    unittest.main()
