"""Simulação de exposições assinadas da abertura ao fechamento."""

import numpy as np
import pandas as pd


def drawdown_capital(capital):
    """Mede a queda desde o pico, incluindo o capital inicial unitário."""
    return capital / capital.cummax().clip(lower=1) - 1


def cvar_empirico(retornos, cauda=0.05):
    """Média das piores perdas, com massa fracionária na fronteira."""
    valores = np.asarray(retornos, dtype=float)
    if valores.ndim != 1 or not valores.size or not np.isfinite(valores).all():
        raise ValueError("O CVaR exige um vetor de retornos finitos e não vazio.")
    if not 0 < cauda < 1:
        raise ValueError("A probabilidade de cauda deve ficar entre zero e um.")
    perdas = np.sort(-valores)[::-1]
    massa = len(perdas) * cauda
    inteiras = int(massa)
    return (perdas[:inteiras].sum() + (massa - inteiras) * perdas[inteiras]) / massa


def simular_intradiario(pesos, retornos, custo_lado, *, detalhar=False):
    """Usa exposições e retornos do mesmo painel, inclusive dias em caixa."""
    if (pesos.empty or not pesos.index.equals(retornos.index)
        or not pesos.index.is_unique or not pesos.index.is_monotonic_increasing
        or not pesos.columns.is_unique):
        raise ValueError("Pesos e retornos devem compartilhar um painel cronológico sem duplicatas.")
    if not np.isfinite(custo_lado) or custo_lado < 0:
        raise ValueError("O custo por lado deve ser finito e não negativo.")
    if not np.isfinite(pesos.to_numpy(dtype=float)).all() or not np.isfinite(retornos).all():
        raise ValueError("Pesos e retornos devem ser finitos em todos os pregões.")

    bruto = pesos.mul(retornos, axis=0)
    custos = 2 * custo_lado * pesos.abs()
    liquido = bruto - custos
    capital = (1 + liquido).cumprod()
    desvio = liquido.std()
    exposicao = pesos.abs().mean()
    metricas = pd.DataFrame({
        "retorno_liquido": capital.iloc[-1] - 1,
        "vol_anual": desvio * np.sqrt(252),
        "sharpe": np.sqrt(252) * liquido.mean() / desvio.replace(0, np.nan),
        "drawdown": drawdown_capital(capital).min(),
        "VaR_95": -liquido.quantile(0.05),
        "CVaR_95": liquido.apply(cvar_empirico),
        "exposicao_media": exposicao,
        "turnover_medio": 2 * exposicao,
        "operacoes": pesos.ne(0).sum(),
    })
    if detalhar:
        metricas["retorno_bruto"] = (1 + bruto).prod() - 1
        metricas["soma_custos_diarios"] = custos.sum()
        metricas["n_pregoes"] = len(pesos)
    return metricas, capital, liquido
