"""Walk-forward backtest of the legacy Ibovespa downside study."""

import numpy as np
import pandas as pd


def declust(positions, horizon):
    """Select origins whose trades can be held without overlapping."""
    selected = []
    next_index = 0
    for position in positions:
        if position >= next_index:
            selected.append(position)
            next_index = position + horizon
    return np.array(selected, dtype=int)


def walk_forward(bars, returns, volatility, *, horizon, training_window,
                 test_window, downside_threshold, volatility_percentile, min_events, round_trip_cost,
                 use_volatility=True):
    """Trade from the next open through the horizon close with one position at a time."""
    dates = bars.index
    if (not isinstance(dates, pd.DatetimeIndex) or not dates.is_unique
        or not dates.is_monotonic_increasing or not dates.equals(returns.index)
        or not dates.equals(volatility.index)):
        raise ValueError("Bars, returns, and volatility must share the same chronological index.")
    if not 1 <= horizon < training_window or test_window < 1 or min_events < 1:
        raise ValueError("Invalid horizon, windows, or minimum event count.")
    if not 0 <= volatility_percentile <= 1 or not np.isfinite(round_trip_cost) or round_trip_cost < 0:
        raise ValueError("Invalid volatility percentile or cost.")

    future_returns = (bars["Close"].shift(-horizon) / bars["Open"].shift(-1) - 1).to_numpy()
    r_v, v_v = returns.to_numpy(), volatility.to_numpy()
    valid = np.isfinite(future_returns)
    down_moves = r_v < downside_threshold
    trades, folds = [], []
    next_index_origin = 0
    for start in range(0, len(dates) - training_window - test_window + 1, test_window):
        end_train = start + training_window
        end_test = end_train + test_window
        limit_vol = np.nanquantile(v_v[start:end_train], volatility_percentile)
        candidates = down_moves & valid
        if use_volatility:
            candidates &= v_v >= limit_vol

        # A training target must be observed before the test window.
        sample_train = candidates[start:end_train - horizon]
        positions = declust(np.flatnonzero(sample_train) + start, horizon)
        mean_train = future_returns[positions].mean() if len(positions) else np.nan
        side = int(np.sign(mean_train)) if len(positions) >= min_events else 0

        net_returns = []
        for origin in np.flatnonzero(candidates[end_train:end_test]) + end_train:
            if side == 0 or origin < next_index_origin:
                continue
            raw = side * future_returns[origin]
            net = raw - round_trip_cost
            output = origin + horizon
            trades.append({
                "signal_date": dates[origin], "entry_date": dates[origin + 1],
                "exit_date": dates[output], "side": side, "raw": raw, "net": net,
            })
            net_returns.append(net)
            # Entry occurs one session after the origin, including across fold boundaries.
            next_index_origin = output

        folds.append({
            "train": f"{dates[start].year}-{dates[end_train - 1].year}",
            "test": f"{dates[end_train].year}-{dates[end_test - 1].year}",
            "limit_vol": limit_vol,
            "events_train": len(positions),
            "E[R] train": mean_train,
            "signal": side,
            "trades_test": len(net_returns),
            "mean_net_return": np.mean(net_returns) if net_returns else np.nan,
            f"benchmark (long {horizon}d, all sessions)": np.nanmean(future_returns[end_train:end_test]),
        })

    columns = ["signal_date", "entry_date", "exit_date", "side", "raw", "net"]
    return pd.DataFrame(folds), pd.DataFrame(trades, columns=columns)
