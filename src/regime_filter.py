"""Causal Gaussian filter for risk regimes."""

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm


def filter_regimes(series, mean, var, A, initial):
    """Sort states by variance and return current and next-session probabilities."""
    observations = np.asarray(series, dtype=float)
    mean, var = np.asarray(mean, dtype=float), np.asarray(var, dtype=float)
    A, initial = np.asarray(A, dtype=float), np.asarray(initial, dtype=float)
    k = var.size
    if (observations.ndim != 1 or np.isinf(observations).any() or not k
        or mean.shape != (k,) or var.shape != (k,)
        or A.shape != (k, k) or initial.shape != (k,)):
        raise ValueError("Invalid dimensions or observations for the regime filter.")
    if (not np.isfinite(mean).all() or not np.isfinite(var).all() or (var <= 0).any()
        or not np.isfinite(A).all() or (A < 0).any() or not np.allclose(A.sum(axis=1), 1)
        or not np.isfinite(initial).all() or (initial < 0).any() or not np.isclose(initial.sum(), 1)):
        raise ValueError("Invalid Gaussian distributions or transition probabilities.")

    order = np.argsort(var)
    mean, var = mean[order], var[order]
    A = A[np.ix_(order, order)]
    prior = initial[order].copy()
    hoje = np.empty((len(observations), k))
    amanha = np.empty_like(hoje)
    dispersion = np.sqrt(var)
    for i, value in enumerate(observations):
        if pd.notna(value):
            logpeso = np.log(np.maximum(prior, 1e-300)) + norm.logpdf(value, mean, dispersion)
            prior = np.exp(logpeso - logsumexp(logpeso))
        hoje[i] = prior
        prior = prior @ A
        amanha[i] = prior
    assert np.allclose(amanha.sum(axis=1), 1)
    return hoje, amanha, mean, var, A
