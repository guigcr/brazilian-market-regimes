"""Causal features shared by the research and paper-trading workflows."""
import numpy as np
import pandas as pd
from scipy.special import logsumexp


def build_features(ibov, sessions, load, frozen):
    """Build features for the frozen model in chronological order."""
    r = np.log(ibov.Close).diff()
    X = pd.DataFrame(index=sessions)
    for k in [0,1,2,3,4]:
        X[f"ret_lag{k}"] = r.shift(k)
    for w in [5,10,21,63]:
        X[f"ret_{w}"] = np.log(ibov.Close).diff(w)
        X[f"vol_{w}"] = r.rolling(w).std()
        X[f"dist_mean_{w}"] = np.log(ibov.Close/ibov.Close.rolling(w).mean())
    X["gap"] = np.log(ibov.Open/ibov.Close.shift())
    X["intraday"] = np.log(ibov.Close/ibov.Open)
    X["range"] = np.log(ibov.High/ibov.Low)
    X["day"] = sessions.day_of_week
    local = list(X.columns)

    # Parameters fitted on the initial 252 returns and stored in the snapshot.
    state_hmm = frozen["hmm"]
    mu = np.asarray(state_hmm["mean"])
    var = np.asarray(state_hmm["variance"])
    prior = np.asarray(state_hmm["initial"])
    transition = np.asarray(state_hmm["transition"])
    regime = []
    for obs in r.dropna().to_numpy()*100:
        logw = np.log(np.maximum(prior,1e-300))-.5*(np.log(var)+(obs-mu)**2/var)
        prior = np.exp(logw-logsumexp(logw)) @ transition
        regime.append(prior[np.argmax(var)])
    X["hmm_high_volatility"] = pd.Series(regime,index=r.dropna().index)
    state_garch = frozen["garch"]
    omega, alpha, beta = (state_garch[k] for k in ["omega","alpha","beta"])
    sigma2 = state_garch["variance_initial"]
    vars_predicted = []
    for obs in r.dropna().to_numpy()*100:
        sigma2 = omega + alpha*obs**2 + beta*sigma2
        vars_predicted.append(sigma2)
    X["garch_next_volatility"] = pd.Series(np.sqrt(vars_predicted)/100,index=r.dropna().index)
    dates_sources = pd.DataFrame(index=sessions)

    for name in ["sp500","usd","vix","oil","ewz"]:
        price = load(name).Close.dropna()
        if not np.isfinite(price).all() or not price.gt(0).all():
            raise ValueError(f"Invalid closing prices for {name}.")
        ext = pd.DataFrame({f"{name}_ret{w}":np.log(price).diff(w) for w in [1,5,21]})
        ext["available"] = ext.index
        aligned = pd.merge_asof(pd.DataFrame(index=sessions),ext,left_index=True,right_index=True,
                                 allow_exact_matches=False,tolerance=pd.Timedelta(days=7))
        if not (aligned.available.dropna() < aligned.available.dropna().index).all():
            raise ValueError(f"Future data availability for {name}.")
        dates_sources[name] = aligned.available
        X = X.join(aligned.drop(columns="available"))

    return r, X, local, dates_sources
