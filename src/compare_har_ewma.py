"""Compare eight volatility models on common out-of-sample dates."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from market_calendar import expected_sessions
from risk_statistics import session_adjacency, paired_mean_interval

EWMA = "EWMA 0.94"
HAR = "HAR · daily proxy"
MODELS = ("Historical 21", EWMA, "GARCH(1,1)", "EGARCH", "GJR-GARCH",
           "HMM 2 states", HAR, "XGBoost")
TARGET = "proxy_var_observed"
LAG_PRINCIPAL = 10
LAGS = (0, 5, 10, 20)
BLOCKS = (5, 10, 20)


def diebold_mariano(differences, lags):
    """Asymptotic DM test with Bartlett weights and lags in exchange sessions."""
    observed = differences.dropna()
    n = len(observed)
    if not 0 <= lags < n or n < 2 or not np.isfinite(observed).all():
        raise ValueError("Invalid sample or lag count for the DM test.")
    mean = float(observed.mean())
    # Missing sessions contribute zero to products of centered residuals.
    residuals = differences.sub(mean).fillna(0).to_numpy()
    sum_hac = np.dot(residuals, residuals)
    for k in range(1, lags + 1):
        sum_hac += 2 * (1 - k / (lags + 1)) * np.dot(residuals[k:], residuals[:-k])
    if sum_hac <= 0:
        raise ValueError("The loss differential has degenerate HAC variance.")
    standard_error = float(np.sqrt(sum_hac) / n)
    dm = mean / standard_error
    return {"dm": dm, "standard_error": standard_error,
            "two_sided_p": float(2 * norm.sf(abs(dm))),
            "one_sided_p_model_lower": float(norm.cdf(dm))}


def compare(data):
    origins = pd.DatetimeIndex(data.origin)
    targets = pd.DatetimeIndex(data.target_date)
    if (data.empty or origins.hasnans or targets.hasnans or not origins.is_unique
        or not targets.is_unique or not targets.is_monotonic_increasing):
        raise ValueError("The comparison requires unique, valid, ordered origins and targets.")
    sessions = expected_sessions("BVMF", origins.min(), targets.max())
    pos_o, pos_a = sessions.get_indexer(origins), sessions.get_indexer(targets)
    if (pos_o < 0).any() or (pos_a < 0).any() or not (pos_a - pos_o == 1).all():
        raise ValueError("Each target must be the session immediately following its forecast origin.")

    common = data.set_index(targets)[[TARGET, *MODELS]].dropna()
    if (len(common) <= max(LAGS) or not np.isfinite(common).all().all()
        or common[TARGET].lt(0).any() or common[list(MODELS)].le(0).any().any()):
        raise ValueError("Insufficient common observations or invalid variances.")
    h = common[list(MODELS)]
    losses = np.log(h) + h.rdiv(common[TARGET], axis=0)
    grid = sessions[sessions.slice_indexer(common.index.min(), common.index.max())]
    adjacent = session_adjacency(grid, common.index)
    differences = losses.subtract(losses[EWMA], axis=0)
    tests = []
    for name in MODELS:
        if name == EWMA:
            continue
        d = differences[name]
        lower, upper = paired_mean_interval(d, adjacent, block=LAG_PRINCIPAL)
        tests.append({"model": name, "qlike_difference": float(d.mean()),
                       "bootstrap_ci95": [float(lower), float(upper)],
                       **diebold_mariano(d.reindex(grid), LAG_PRINCIPAL)})
    for field in ("two_sided_p", "one_sided_p_model_lower"):
        adjusted = multipletests([t[field] for t in tests], method="holm")[1]
        for test, value in zip(tests, adjusted):
            test[field + "_holm"] = float(value)

    har = differences[HAR]
    ranking = losses.mean().sort_values()
    return {
        "common_count": len(common), "first_target": str(common.index.min().date()),
        "last_target": str(common.index.max().date()),
        "reference": EWMA, "lowest_observed_qlike": ranking.index[0],
        "mean_qlike": ranking.to_dict(),
        "models_beating_ewma_two_sided_holm_dm_at_5pct": [
            t["model"] for t in tests
            if t["qlike_difference"] < 0 and t["two_sided_p_holm"] < .05
        ],
        "primary_hac_lags": LAG_PRINCIPAL, "bootstrap_replicas": 3000,
        "bootstrap_seed": 42, "tests_vs_ewma": tests,
        "har_sensitivity_hac": {k: diebold_mariano(har.reindex(grid), k) for k in LAGS},
        "har_sensitivity_bootstrap": {
            k: paired_mean_interval(har, adjacent, block=k).tolist() for k in BLOCKS
        },
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("forecasts", type=Path, help="The volatility lab's next_session_forecasts.csv file.")
    args = cli.parse_args()
    raw = args.forecasts.read_bytes()
    data = pd.read_csv(BytesIO(raw), parse_dates=["origin", "target_date"])
    result = {"source_sha256": sha256(raw).hexdigest(), **compare(data)}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
