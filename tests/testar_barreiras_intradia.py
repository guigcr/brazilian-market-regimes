"""Regressões da rotulagem causal e da política de corte."""
import unittest

import pandas as pd

from barreiras_intradia import avaliar_eventos, limite_sem_censura, simular_evento


class BarreirasIntradia(unittest.TestCase):
    def setUp(self):
        self.sessoes = pd.bdate_range("2026-01-05",periods=9)
        self.barras = pd.DataFrame({"Open":100.,"High":100.5,"Low":99.5,"Close":100.},
                                   index=self.sessoes)
        self.corte = self.sessoes[7]

    def avaliar(self,lado=1):
        return simular_evento(self.barras,self.sessoes,self.sessoes[0],lado,self.corte)

    def test_saida_conhecida_antes_de_lacuna(self):
        self.barras.loc[self.sessoes[1],"High"] = 102.
        self.barras = self.barras.drop(self.sessoes[2])
        evento = self.avaliar()
        self.assertEqual(evento["status"],"encerrado")
        self.assertEqual(evento["saida"],self.sessoes[1])
        self.assertAlmostEqual(evento["liquido"],.012)

    def test_lacuna_distingue_entrada_e_censura(self):
        self.barras = self.barras.drop(self.sessoes[2])
        evento = self.avaliar()
        self.assertEqual(evento["status"],"censurado")
        self.assertEqual(evento["data_interrupcao"],self.sessoes[2])
        self.assertNotIn("liquido",evento)
        self.barras = self.barras.drop(self.sessoes[1])
        self.assertEqual(self.avaliar()["status"],"entrada_indisponivel")

    def test_stop_tem_prioridade_com_ambas_barreiras(self):
        self.barras.loc[self.sessoes[1],["High","Low"]] = [103.,97.]
        for lado in [1,-1]:
            with self.subTest(lado=lado):
                evento = self.avaliar(lado)
                self.assertEqual(evento["motivo"],"stop")
                self.assertAlmostEqual(evento["liquido"],-.013)

    def test_anexar_dados_depois_do_corte_nao_altera_resultado(self):
        previsoes = pd.Series([1.,1.],index=self.sessoes[[0,2]])
        self.barras.loc[self.sessoes[3],"High"] = 103.
        curto = avaliar_eventos(self.barras.loc[self.barras.index<self.corte],
                                self.sessoes,previsoes,self.corte)
        completo = avaliar_eventos(self.barras,self.sessoes,previsoes,self.corte)
        self.assertEqual(curto[0],completo[0])
        pd.testing.assert_frame_equal(curto[1],completo[1])
        self.assertEqual(completo[1].status.tolist(),["encerrado","fora_da_janela"])

    def test_painel_para_antes_da_primeira_barra_invalida(self):
        self.barras.loc[self.sessoes[4],"Open"] = 0.
        self.assertEqual(limite_sem_censura(self.barras,self.sessoes,self.corte),self.sessoes[4])


if __name__=="__main__":
    unittest.main()
