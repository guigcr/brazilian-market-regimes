"""Regression checks for causal labels and cutoff rules."""
import unittest

import pandas as pd

from intraday_barriers import evaluate_events, uncensored_barrier, simulate_event


class IntradayBarriers(unittest.TestCase):
    def setUp(self):
        self.sessions = pd.bdate_range("2026-01-05",periods=9)
        self.bars = pd.DataFrame({"Open":100.,"High":100.5,"Low":99.5,"Close":100.},
                                   index=self.sessions)
        self.threshold = self.sessions[7]

    def evaluate(self,side=1):
        return simulate_event(self.bars,self.sessions,self.sessions[0],side,self.threshold)

    def test_output_known_before_de_gap(self):
        self.bars.loc[self.sessions[1],"High"] = 102.
        self.bars = self.bars.drop(self.sessions[2])
        event = self.evaluate()
        self.assertEqual(event["status"],"closed")
        self.assertEqual(event["output"],self.sessions[1])
        self.assertAlmostEqual(event["net"],.012)

    def test_gap_distinguishes_entry_e_censoring(self):
        self.bars = self.bars.drop(self.sessions[2])
        event = self.evaluate()
        self.assertEqual(event["status"],"censored")
        self.assertEqual(event["interruption_date"],self.sessions[2])
        self.assertNotIn("net",event)
        self.bars = self.bars.drop(self.sessions[1])
        self.assertEqual(self.evaluate()["status"],"entry_unavailable")

    def test_stop_tem_priority_com_both_barriers(self):
        self.bars.loc[self.sessions[1],["High","Low"]] = [103.,97.]
        for side in [1,-1]:
            with self.subTest(side=side):
                event = self.evaluate(side)
                self.assertEqual(event["reason"],"stop")
                self.assertAlmostEqual(event["net"],-.013)

    def test_append_data_after_do_threshold_does_not_change_result(self):
        forecasts = pd.Series([1.,1.],index=self.sessions[[0,2]])
        self.bars.loc[self.sessions[3],"High"] = 103.
        short = evaluate_events(self.bars.loc[self.bars.index<self.threshold],
                                self.sessions,forecasts,self.threshold)
        full = evaluate_events(self.bars,self.sessions,forecasts,self.threshold)
        self.assertEqual(short[0],full[0])
        pd.testing.assert_frame_equal(short[1],full[1])
        self.assertEqual(full[1].status.tolist(),["closed","outside_window"])

    def test_panel_para_before_da_first_barra_invalida(self):
        self.bars.loc[self.sessions[4],"Open"] = 0.
        self.assertEqual(uncensored_barrier(self.bars,self.sessions,self.threshold),self.sessions[4])


if __name__=="__main__":
    unittest.main()
