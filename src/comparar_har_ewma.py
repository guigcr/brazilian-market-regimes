"""Compara os oito modelos de volatilidade nas mesmas datas fora do treino."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from calendario_mercado import sessoes_esperadas
from risco_estatistico import adjacencia_sessoes, intervalo_media_pareada

EWMA = "EWMA 0,94"
HAR = "HAR · proxy diária"
MODELOS = ("Histórica 21", EWMA, "GARCH(1,1)", "EGARCH", "GJR-GARCH",
           "HMM 2 estados", HAR, "XGBoost")
ALVO = "proxy_var_observada"
LAG_PRINCIPAL = 10
LAGS = (0, 5, 10, 20)
BLOCOS = (5, 10, 20)


def diebold_mariano(diferencas, lags):
    """DM assintótico com pesos de Bartlett e lags em pregões de calendário."""
    observadas = diferencas.dropna()
    n = len(observadas)
    if not 0 <= lags < n or n < 2 or not np.isfinite(observadas).all():
        raise ValueError("Amostra ou número de defasagens inválido para DM.")
    media = float(observadas.mean())
    # Um pregão ausente não contribui para os produtos de resíduos centrados.
    residuos = diferencas.sub(media).fillna(0).to_numpy()
    soma_hac = np.dot(residuos, residuos)
    for k in range(1, lags + 1):
        soma_hac += 2 * (1 - k / (lags + 1)) * np.dot(residuos[k:], residuos[:-k])
    if soma_hac <= 0:
        raise ValueError("O diferencial de perdas tem variância HAC degenerada.")
    erro_padrao = float(np.sqrt(soma_hac) / n)
    dm = media / erro_padrao
    return {"dm": dm, "erro_padrao": erro_padrao,
            "p_bilateral": float(2 * norm.sf(abs(dm))),
            "p_unilateral_modelo_menor": float(norm.cdf(dm))}


def comparar(dados):
    origens = pd.DatetimeIndex(dados.origem)
    alvos = pd.DatetimeIndex(dados.data_alvo)
    if (dados.empty or origens.hasnans or alvos.hasnans or not origens.is_unique
        or not alvos.is_unique or not alvos.is_monotonic_increasing):
        raise ValueError("A comparação exige origens e alvos únicos, válidos e em ordem.")
    sessoes = sessoes_esperadas("BVMF", origens.min(), alvos.max())
    pos_o, pos_a = sessoes.get_indexer(origens), sessoes.get_indexer(alvos)
    if (pos_o < 0).any() or (pos_a < 0).any() or not (pos_a - pos_o == 1).all():
        raise ValueError("Cada alvo deve ser o pregão seguinte à origem.")

    comum = dados.set_index(alvos)[[ALVO, *MODELOS]].dropna()
    if (len(comum) <= max(LAGS) or not np.isfinite(comum).all().all()
        or comum[ALVO].lt(0).any() or comum[list(MODELOS)].le(0).any().any()):
        raise ValueError("Amostra comum insuficiente ou variâncias inválidas.")
    h = comum[list(MODELOS)]
    perdas = np.log(h) + h.rdiv(comum[ALVO], axis=0)
    grade = sessoes[sessoes.slice_indexer(comum.index.min(), comum.index.max())]
    adjacentes = adjacencia_sessoes(grade, comum.index)
    diferencas = perdas.subtract(perdas[EWMA], axis=0)
    testes = []
    for nome in MODELOS:
        if nome == EWMA:
            continue
        d = diferencas[nome]
        inferior, superior = intervalo_media_pareada(d, adjacentes, bloco=LAG_PRINCIPAL)
        testes.append({"modelo": nome, "diferenca_qlike": float(d.mean()),
                       "ic95_bootstrap": [float(inferior), float(superior)],
                       **diebold_mariano(d.reindex(grade), LAG_PRINCIPAL)})
    for campo in ("p_bilateral", "p_unilateral_modelo_menor"):
        ajustados = multipletests([t[campo] for t in testes], method="holm")[1]
        for teste, valor in zip(testes, ajustados):
            teste[campo + "_holm"] = float(valor)

    har = diferencas[HAR]
    ranking = perdas.mean().sort_values()
    return {
        "n_comum": len(comum), "primeiro_alvo": str(comum.index.min().date()),
        "ultimo_alvo": str(comum.index.max().date()),
        "referencia": EWMA, "melhor_qlike_observado": ranking.index[0],
        "qlike_medio": ranking.to_dict(),
        "modelos_superiores_ao_ewma_dm_holm_bilateral_5pct": [
            t["modelo"] for t in testes
            if t["diferenca_qlike"] < 0 and t["p_bilateral_holm"] < .05
        ],
        "lags_hac_principal": LAG_PRINCIPAL, "bootstrap_replicas": 3000,
        "bootstrap_semente": 42, "testes_vs_ewma": testes,
        "har_sensibilidade_hac": {k: diebold_mariano(har.reindex(grade), k) for k in LAGS},
        "har_sensibilidade_bootstrap": {
            k: intervalo_media_pareada(har, adjacentes, bloco=k).tolist() for k in BLOCOS
        },
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("previsoes", type=Path, help="CSV previsoes_D1.csv do laboratório.")
    args = cli.parse_args()
    bruto = args.previsoes.read_bytes()
    dados = pd.read_csv(BytesIO(bruto), parse_dates=["origem", "data_alvo"])
    resultado = {"fonte_sha256": sha256(bruto).hexdigest(), **comparar(dados)}
    print(json.dumps(resultado, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
