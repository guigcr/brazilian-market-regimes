# Brazilian Market Regime Engine

An empirical study of next-session volatility and risk regimes in BOVA11, an ETF that tracks the Ibovespa. The project asks whether a more accurate volatility forecast helps size positions once trading costs are included.

[Research paper (PDF)](artigo/Artigo_Cientifico_BOVA11.pdf) · [LaTeX source](artigo/Artigo_Cientifico_BOVA11.tex)

## Project structure

| Directory | Contents |
|---|---|
| [notebooks](notebooks/) | Data cleaning, volatility forecasting, regime models, and directional trading experiments |
| [src](src/) | Reusable Python modules for calendars, data cleaning, risk metrics, filtering, and backtesting |
| [tests](tests/) | Regression tests for risk calculations, time alignment, and execution logic |
| [artigo](artigo/) | Research paper and editable LaTeX source |
| [analise](analise/) | Aggregate results of the reproducible HAR–EWMA comparison |

| Notebook | Purpose |
|---|---|
| [Untitled93](notebooks/Untitled93.ipynb) | Ibovespa data cleaning and a legacy walk-forward backtest |
| [BOVA11 Volatility Research Lab](notebooks/BOVA11_Volatility_Research_Lab.ipynb) | Volatility, drawdown, VaR, and Expected Shortfall |
| [Ibovespa Risk Regimes](notebooks/Ibovespa_Regimes_de_Risco.ipynb) | HMM, Markov switching, and regime-based position sizing |
| [Cross-Market Features and Meta-Labeling](notebooks/Ibovespa_Mercados_e_MetaLabeling.ipynb) | Lagged market features, barrier labels, and an exploratory replay of a frozen signal |

## Run the tests

The reference environment is Python 3.14.10 on macOS ARM.

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests -p 'testar_*.py' -q
```

XGBoost and LightGBM may require OpenMP on macOS.

```sh
brew install libomp
export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib"
```

## Run the notebooks

The public source package reads historical snapshots and the frozen estimator from a separate local copy of the research project. Set `IBOV_PESQUISA_LOCAL` to the directory containing the original notebooks and `dados/ibovespa_bruto.csv`. Start Jupyter from the same terminal after setting both environment variables. Open a notebook in `notebooks/` and run its cells in order.

```sh
export IBOV_PESQUISA_LOCAL="/path/to/your/private/ibovespa"
```

The included regression suite runs without the private snapshots. Some module names, output columns, and historical filenames retain their original Portuguese names to keep the saved research interfaces stable.

## Findings

- Daily HAR had the lowest QLIKE over 437 common targets. Its mean loss difference against EWMA was −0.0645. The 95% block-bootstrap interval was [−0.1457, +0.0027], and the two-sided Diebold–Mariano p-value was 0.7543 after Holm adjustment across seven comparisons.
- Intraday EWMA forecast opening-to-close variance more accurately than close-to-close EWMA on 430 common sessions. Its QLIKE was 0.0677 lower. Its higher average exposure coincided with a net return 7.59 percentage points lower after costs.
- The frozen directional signal had seven gains in ten historical trades and returned +2.94% net in the simulation. At comparable exposure, fixed sizing achieved nearly the same drawdown as regime sizing and a higher net return.

The results describe an exploratory historical sample. Realized-volatility models based on intraday bars and prospective execution remain separate research stages.

## Reproduce the HAR–EWMA comparison

Run the script against the next-session forecasts exported by the volatility notebook:

```sh
python src/comparar_har_ewma.py \
  "$IBOV_PESQUISA_LOCAL/resultados_risco/BOVA11/previsoes_D1.csv"
```

The output records the source hash, the common sample, the QLIKE ranking, Newey–West standard errors, Holm-adjusted p-values, and sensitivity to HAC bandwidth and bootstrap block length. [Archived aggregate results](analise/comparacao_har_ewma.json)

A confirmatory study would specify the forecast horizon, training window, refit schedule, evaluation window, and primary test before observing new targets. Forecasts would be recorded before the targets become available and compared on common dates.
