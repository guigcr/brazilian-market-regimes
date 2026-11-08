"""Casos de fronteira das janelas do backtest legado."""

import unittest

import pandas as pd

from backtest_legado import walk_forward


class BacktestLegado(unittest.TestCase):
    def setUp(self):
        self.datas = pd.bdate_range("2025-01-02", periods=14)
        self.barras = pd.DataFrame({
            "Open": [100 + i for i in range(14)],
            "Close": [110 + i for i in range(14)],
        }, index=self.datas)
        self.vol = pd.Series(0.2, index=self.datas)

    def executar(self, quedas):
        retornos = pd.Series(0.0, index=self.datas)
        retornos.iloc[quedas] = -0.03
        return walk_forward(
            self.barras, retornos, self.vol, horizonte=3, janela_treino=6,
            janela_teste=2, lim_queda=-0.02, pct_vol=0.9, min_eventos=1,
            custo_rt=0.002,
        )

    def test_posicao_atravessa_janelas_de_teste(self):
        janelas, trades = self.executar(list(range(14)))
        self.assertEqual(janelas.trades_teste.iloc[:2].tolist(), [1, 1])
        self.assertEqual(trades.data_sinal.tolist(), [self.datas[6], self.datas[9]])
        self.assertEqual(trades.data_entrada.tolist(), [self.datas[7], self.datas[10]])
        self.assertEqual(trades.data_saida.tolist(), [self.datas[9], self.datas[12]])
        self.assertAlmostEqual(trades.liquido.iloc[0], 119 / 107 - 1 - 0.002)

    def test_purga_alvo_que_fecha_no_primeiro_dia_de_teste(self):
        janelas, trades = self.executar([3, 6])
        self.assertEqual(janelas.eventos_treino.iloc[0], 0)
        self.assertEqual(janelas.sinal.iloc[0], 0)
        self.assertTrue(trades.empty)


if __name__ == "__main__":
    unittest.main()
