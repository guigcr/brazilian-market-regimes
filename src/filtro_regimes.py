"""Filtro gaussiano causal usado pelo motor de regimes de risco."""

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm


def filtrar(serie, media, var, A, inicial):
    """Ordena estados por variância e devolve probabilidades atuais e de D+1."""
    observacoes = np.asarray(serie, dtype=float)
    media, var = np.asarray(media, dtype=float), np.asarray(var, dtype=float)
    A, inicial = np.asarray(A, dtype=float), np.asarray(inicial, dtype=float)
    k = var.size
    if (observacoes.ndim != 1 or np.isinf(observacoes).any() or not k
        or media.shape != (k,) or var.shape != (k,)
        or A.shape != (k, k) or inicial.shape != (k,)):
        raise ValueError("Dimensões ou observações inválidas para o filtro de regimes.")
    if (not np.isfinite(media).all() or not np.isfinite(var).all() or (var <= 0).any()
        or not np.isfinite(A).all() or (A < 0).any() or not np.allclose(A.sum(axis=1), 1)
        or not np.isfinite(inicial).all() or (inicial < 0).any() or not np.isclose(inicial.sum(), 1)):
        raise ValueError("Distribuições gaussianas ou probabilidades de transição inválidas.")

    ordem = np.argsort(var)
    media, var = media[ordem], var[ordem]
    A = A[np.ix_(ordem, ordem)]
    prior = inicial[ordem].copy()
    hoje = np.empty((len(observacoes), k))
    amanha = np.empty_like(hoje)
    dispersao = np.sqrt(var)
    for i, valor in enumerate(observacoes):
        if pd.notna(valor):
            logpeso = np.log(np.maximum(prior, 1e-300)) + norm.logpdf(valor, media, dispersao)
            prior = np.exp(logpeso - logsumexp(logpeso))
        hoje[i] = prior
        prior = prior @ A
        amanha[i] = prior
    assert np.allclose(amanha.sum(axis=1), 1)
    return hoje, amanha, media, var, A
