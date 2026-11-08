"""Open-to-close simulation with signed exposures."""

import numpy as np
import pandas as pd


def equity_drawdown(equity):
    """Measure drawdown from the peak, including initial equity of one."""
    return equity / equity.cummax().clip(lower=1) - 1


def empirical_cvar(returns, tail=0.05):
    """Average the worst losses with fractional weight at the tail boundary."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("CVaR requires a nonempty vector of finite returns.")
    if not 0 < tail < 1:
        raise ValueError("The tail probability must be between zero and one.")
    losses = np.sort(-values)[::-1]
    mass = len(losses) * tail
    integers = int(mass)
    return (losses[:integers].sum() + (mass - integers) * losses[integers]) / mass


def simulate_intraday(weights, returns, cost_per_side, *, include_details=False):
    """Use aligned exposures and returns, including sessions held in cash."""
    if (weights.empty or not weights.index.equals(returns.index)
        or not weights.index.is_unique or not weights.index.is_monotonic_increasing
        or not weights.columns.is_unique):
        raise ValueError("Weights and returns must share a chronological panel with unique dates.")
    if not np.isfinite(cost_per_side) or cost_per_side < 0:
        raise ValueError("Cost per side must be finite and nonnegative.")
    if not np.isfinite(weights.to_numpy(dtype=float)).all() or not np.isfinite(returns).all():
        raise ValueError("Weights and returns must be finite for every session.")

    raw = weights.mul(returns, axis=0)
    costs = 2 * cost_per_side * weights.abs()
    net = raw - costs
    equity = (1 + net).cumprod()
    deviation = net.std()
    exposure = weights.abs().mean()
    metrics = pd.DataFrame({
        "net_return": equity.iloc[-1] - 1,
        "vol_annual": deviation * np.sqrt(252),
        "sharpe": np.sqrt(252) * net.mean() / deviation.replace(0, np.nan),
        "drawdown": equity_drawdown(equity).min(),
        "VaR_95": -net.quantile(0.05),
        "CVaR_95": net.apply(empirical_cvar),
        "exposure_mean": exposure,
        "mean_turnover": 2 * exposure,
        "trades": weights.ne(0).sum(),
    })
    if include_details:
        metrics["gross_return"] = (1 + raw).prod() - 1
        metrics["total_daily_costs"] = costs.sum()
        metrics["session_count"] = len(weights)
    return metrics, equity, net
