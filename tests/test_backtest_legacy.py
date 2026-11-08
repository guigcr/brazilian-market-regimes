"""Check fold boundaries in the legacy backtest."""

import unittest

import pandas as pd

from legacy_backtest import walk_forward


class LegacyBacktest(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2025-01-02", periods=14)
        self.bars = pd.DataFrame({
            "Open": [100 + i for i in range(14)],
            "Close": [110 + i for i in range(14)],
        }, index=self.dates)
        self.vol = pd.Series(0.2, index=self.dates)

    def run_backtest(self, down_moves):
        returns = pd.Series(0.0, index=self.dates)
        returns.iloc[down_moves] = -0.03
        return walk_forward(
            self.bars, returns, self.vol, horizon=3, training_window=6,
            test_window=2, downside_threshold=-0.02, volatility_percentile=0.9, min_events=1,
            round_trip_cost=0.002,
        )

    def test_position_spans_windows_de_test(self):
        windows, trades = self.run_backtest(list(range(14)))
        self.assertEqual(windows.trades_test.iloc[:2].tolist(), [1, 1])
        self.assertEqual(trades.signal_date.tolist(), [self.dates[6], self.dates[9]])
        self.assertEqual(trades.entry_date.tolist(), [self.dates[7], self.dates[10]])
        self.assertEqual(trades.exit_date.tolist(), [self.dates[9], self.dates[12]])
        self.assertAlmostEqual(trades.net.iloc[0], 119 / 107 - 1 - 0.002)

    def test_purges_target_que_closes_no_first_day_de_test(self):
        windows, trades = self.run_backtest([3, 6])
        self.assertEqual(windows.events_train.iloc[0], 0)
        self.assertEqual(windows.signal.iloc[0], 0)
        self.assertTrue(trades.empty)


if __name__ == "__main__":
    unittest.main()
