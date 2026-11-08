"""Variáveis causais compartilhadas pelo estudo e pelo paper trading."""
import numpy as np
import pandas as pd
from scipy.special import logsumexp


def construir_features(ibov, sessoes, carregar, congelado):
    """Calcula as variáveis do modelo congelado em ordem causal."""
    r = np.log(ibov.Close).diff()
    X = pd.DataFrame(index=sessoes)
    for k in [0,1,2,3,4]:
        X[f"ret_lag{k}"] = r.shift(k)
    for w in [5,10,21,63]:
        X[f"ret_{w}"] = np.log(ibov.Close).diff(w)
        X[f"vol_{w}"] = r.rolling(w).std()
        X[f"dist_media_{w}"] = np.log(ibov.Close/ibov.Close.rolling(w).mean())
    X["gap"] = np.log(ibov.Open/ibov.Close.shift())
    X["intradia"] = np.log(ibov.Close/ibov.Open)
    X["amplitude"] = np.log(ibov.High/ibov.Low)
    X["dia"] = sessoes.day_of_week
    locais = list(X.columns)

    # Parâmetros estimados nos 252 retornos iniciais e preservados no snapshot.
    estado_hmm = congelado["hmm"]
    mu = np.asarray(estado_hmm["media"])
    var = np.asarray(estado_hmm["variancia"])
    prior = np.asarray(estado_hmm["inicial"])
    transicao = np.asarray(estado_hmm["transicao"])
    regime = []
    for obs in r.dropna().to_numpy()*100:
        logw = np.log(np.maximum(prior,1e-300))-.5*(np.log(var)+(obs-mu)**2/var)
        prior = np.exp(logw-logsumexp(logw)) @ transicao
        regime.append(prior[np.argmax(var)])
    X["hmm_vol_alta"] = pd.Series(regime,index=r.dropna().index)
    estado_garch = congelado["garch"]
    omega, alpha, beta = (estado_garch[k] for k in ["omega","alpha","beta"])
    sigma2 = estado_garch["variancia_inicial"]
    vars_previstas = []
    for obs in r.dropna().to_numpy()*100:
        sigma2 = omega + alpha*obs**2 + beta*sigma2
        vars_previstas.append(sigma2)
    X["garch_vol_prox"] = pd.Series(np.sqrt(vars_previstas)/100,index=r.dropna().index)
    datas_fontes = pd.DataFrame(index=sessoes)

    for nome in ["sp500","dolar","vix","petroleo","ewz"]:
        preco = carregar(nome).Close.dropna()
        if not np.isfinite(preco).all() or not preco.gt(0).all():
            raise ValueError(f"Fechamentos inválidos em {nome}.")
        ext = pd.DataFrame({f"{nome}_ret{w}":np.log(preco).diff(w) for w in [1,5,21]})
        ext["disponivel"] = ext.index
        alinhado = pd.merge_asof(pd.DataFrame(index=sessoes),ext,left_index=True,right_index=True,
                                 allow_exact_matches=False,tolerance=pd.Timedelta(days=7))
        if not (alinhado.disponivel.dropna() < alinhado.disponivel.dropna().index).all():
            raise ValueError(f"Disponibilidade futura em {nome}.")
        datas_fontes[nome] = alinhado.disponivel
        X = X.join(alinhado.drop(columns="disponivel"))

    return r, X, locais, datas_fontes
