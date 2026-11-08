"""Backtest walk-forward do estudo legado de quedas no Ibovespa."""

import numpy as np
import pandas as pd


def declust(posicoes, horizonte):
    """Seleciona origens cujas operações podem ser mantidas sem sobreposição."""
    selecionadas = []
    proxima = 0
    for posicao in posicoes:
        if posicao >= proxima:
            selecionadas.append(posicao)
            proxima = posicao + horizonte
    return np.array(selecionadas, dtype=int)


def walk_forward(barras, retornos, volatilidade, *, horizonte, janela_treino,
                 janela_teste, lim_queda, pct_vol, min_eventos, custo_rt,
                 usar_vol=True):
    """Opera da abertura seguinte ao fechamento do horizonte, com posição única."""
    datas = barras.index
    if (not isinstance(datas, pd.DatetimeIndex) or not datas.is_unique
        or not datas.is_monotonic_increasing or not datas.equals(retornos.index)
        or not datas.equals(volatilidade.index)):
        raise ValueError("Barras, retornos e volatilidade devem ter o mesmo índice cronológico.")
    if not 1 <= horizonte < janela_treino or janela_teste < 1 or min_eventos < 1:
        raise ValueError("Horizonte, janelas ou mínimo de eventos inválidos.")
    if not 0 <= pct_vol <= 1 or not np.isfinite(custo_rt) or custo_rt < 0:
        raise ValueError("Percentil de volatilidade ou custo inválido.")

    futuros = (barras["Close"].shift(-horizonte) / barras["Open"].shift(-1) - 1).to_numpy()
    r_v, v_v = retornos.to_numpy(), volatilidade.to_numpy()
    validos = np.isfinite(futuros)
    quedas = r_v < lim_queda
    trades, folds = [], []
    proxima_origem = 0
    for inicio in range(0, len(datas) - janela_treino - janela_teste + 1, janela_teste):
        fim_treino = inicio + janela_treino
        fim_teste = fim_treino + janela_teste
        limite_vol = np.nanquantile(v_v[inicio:fim_treino], pct_vol)
        candidatos = quedas & validos
        if usar_vol:
            candidatos &= v_v >= limite_vol

        # No treino, o fechamento que realiza o alvo precisa anteceder o teste.
        amostra_treino = candidatos[inicio:fim_treino - horizonte]
        posicoes = declust(np.flatnonzero(amostra_treino) + inicio, horizonte)
        media_treino = futuros[posicoes].mean() if len(posicoes) else np.nan
        lado = int(np.sign(media_treino)) if len(posicoes) >= min_eventos else 0

        liquidos = []
        for origem in np.flatnonzero(candidatos[fim_treino:fim_teste]) + fim_treino:
            if lado == 0 or origem < proxima_origem:
                continue
            bruto = lado * futuros[origem]
            liquido = bruto - custo_rt
            saida = origem + horizonte
            trades.append({
                "data_sinal": datas[origem], "data_entrada": datas[origem + 1],
                "data_saida": datas[saida], "lado": lado, "bruto": bruto, "liquido": liquido,
            })
            liquidos.append(liquido)
            # A próxima entrada é uma sessão depois da origem, inclusive entre janelas.
            proxima_origem = saida

        folds.append({
            "treino": f"{datas[inicio].year}-{datas[fim_treino - 1].year}",
            "teste": f"{datas[fim_treino].year}-{datas[fim_teste - 1].year}",
            "limite_vol": limite_vol,
            "eventos_treino": len(posicoes),
            "E[R] treino": media_treino,
            "sinal": lado,
            "trades_teste": len(liquidos),
            "ret_liq_medio": np.mean(liquidos) if liquidos else np.nan,
            f"benchmark (long {horizonte}d, todos os dias)": np.nanmean(futuros[fim_treino:fim_teste]),
        })

    colunas = ["data_sinal", "data_entrada", "data_saida", "lado", "bruto", "liquido"]
    return pd.DataFrame(folds), pd.DataFrame(trades, columns=colunas)
