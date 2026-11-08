"""Conservative cleaning without imputing prices, volume, or missing sessions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from market_calendar import expected_sessions

PRICES = ['Open', 'High', 'Low', 'Close']

def clean_data(raw, start='2000-01-01', end='2026-01-01'):
    if raw is None or raw.empty:
        raise ValueError('Cannot validate an empty data source.')
    data = raw.copy(deep=True)
    if isinstance(data.columns, pd.MultiIndex):
        # Accept one asset regardless of the column level order.
        levels = [i for i in range(data.columns.nlevels)
                  if set(PRICES + ['Volume']).issubset(set(data.columns.get_level_values(i))) ]
        if len(levels) != 1:
            raise ValueError('Ambiguous column layout.')
        i = levels[0]
        if data.columns.droplevel(i).nunique() != 1:
            raise ValueError('Expected exactly one asset.')
        data.columns = data.columns.get_level_values(i)
    if not data.columns.is_unique:
        raise ValueError('Duplicate columns.')
    if not set(PRICES + ['Volume']).issubset(data.columns):
        raise ValueError('Required OHLCV columns are missing.')
    dates = pd.to_datetime(data.index, errors='raise')
    if dates.tz is not None:
        dates = dates.tz_convert('America/Sao_Paulo').tz_localize(None)
    if dates.isna().any():
        raise ValueError('The data source contains missing dates.')
    normalized_times = pd.Series(dates != dates.normalize(), index=dates.normalize()).groupby(level=0).any()
    data.index = dates.normalize()
    data.index.name = 'Date'
    data = data.sort_index(kind='stable')
    duplicates = data.loc[data.index.duplicated(keep=False)]
    if any(len(g.drop_duplicates()) > 1 for _, g in duplicates.groupby(level=0)):
        raise ValueError('Duplicate dates have conflicting values. Review the source.')
    n_duplicates = int(data.index.duplicated().sum())
    data = data.loc[~data.index.duplicated()].copy()
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='raise').astype(float)
    flags = pd.DataFrame(index=data.index)
    flags['time_normalized'] = normalized_times.reindex(data.index, fill_value=False)
    flags['outside_period'] = (data.index < pd.Timestamp(start)) | (data.index >= pd.Timestamp(end))
    flags['data_future'] = data.index > pd.Timestamp.now(tz='America/Sao_Paulo').tz_localize(None).normalize()
    # Use the exchange calendar and leave missing sessions unfilled.
    expected = expected_sessions('BVMF', start, pd.Timestamp(end) - pd.Timedelta(days=1))
    flags['outside_calendar'] = ~data.index.isin(expected)
    flags['price_invalid'] = (~np.isfinite(data[PRICES]) | (data[PRICES] <= 0)).any(axis=1)
    flags['ohlc_inconsistent'] = (
        (data.High < data[PRICES].max(axis=1)) |
        (data.Low > data[PRICES].min(axis=1)))
    flags['volume_zero'] = data.Volume.eq(0)
    flags['volume_invalid'] = ~np.isfinite(data.Volume) | data.Volume.lt(0)
    flags['volume_missing'] = flags.volume_zero | flags.volume_invalid
    if 'Adj Close' in data:
        flags['fit_invalid'] = ~np.isfinite(data['Adj Close']) | data['Adj Close'].le(0)
        data.loc[flags.fit_invalid, 'Adj Close'] = np.nan
    exclude = flags[['outside_period', 'data_future', 'outside_calendar',
                     'price_invalid', 'ohlc_inconsistent']].any(axis=1)
    rejected = data.loc[exclude].join(flags.add_prefix('flag_'))
    data.loc[flags.volume_missing, 'Volume'] = np.nan
    full = data.loc[~exclude].copy()
    missing = expected.difference(full.index)
    calendar = pd.DataFrame({'Date': missing})
    # Retain price rows when volume alone is missing.
    analysis = full.loc['2008-01-01':].copy()
    # Calculate returns only between consecutive exchange sessions.
    # This prevents gaps from being treated as one-session returns.
    positions = pd.Series(np.arange(len(expected)), index=expected).reindex(analysis.index)
    consecutive = positions.diff().eq(1)
    analysis['simple_return'] = analysis.Close.pct_change(fill_method=None).where(consecutive)
    analysis['log_return'] = np.log(analysis.Close / analysis.Close.shift()).where(consecutive)
    analysis['volume_available'] = analysis.Volume.notna()
    # Flag outliers and retain the original observations.
    outliers = analysis.loc[analysis.simple_return.abs() > 0.15].copy()
    annual = full.Volume.groupby(full.index.year).agg(['count', 'median', 'mean', 'max'])
    annual['ratio_median_year_previous'] = annual['median'].div(annual['median'].shift())
    summary = {
        'rows_source': len(raw), 'duplicates_identical_removed': n_duplicates,
        'rows_rejected': len(rejected), 'rows_full': len(full),
        'rows_analysis_since_2008': len(analysis),
        'first_data': str(analysis.index.min()), 'last_data': str(analysis.index.max()),
        'volume_missing_analysis': int(analysis.Volume.isna().sum()),
        'sessions_missing_period_full': len(missing),
        'sessions_missing_since_2008': int((missing >= pd.Timestamp('2008-01-01')).sum()),
        'returns_outliers_flagged': len(outliers),
        'count_flags': {k: int(v) for k, v in flags.sum().items()},
    }
    if analysis.empty:
        raise ValueError('The analysis dataset is empty.')
    assert full.index.is_unique and full.index.is_monotonic_increasing
    assert np.isfinite(full[PRICES]).all().all()
    assert (full[PRICES] > 0).all().all()
    assert np.allclose(analysis.log_return.dropna(), np.log1p(analysis.simple_return.dropna()))
    return dict(full=full, analysis=analysis, flags=flags, rejected=rejected,
                sessions_missing=calendar, outliers=outliers, volume_annual=annual, summary=summary)


def export(result, destination='data'):
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    names = {'full': 'ibovespa_full', 'analysis': 'ibovespa_clean',
             'flags': 'audit_flags', 'rejected': 'rows_rejected',
             'sessions_missing': 'sessions_missing', 'outliers': 'returns_outliers',
             'volume_annual': 'volume_por_year'}
    for key, name in names.items():
        result[key].to_csv(directory / f'{name}.csv', index=key != 'sessions_missing')
    (directory / 'summary_quality.json').write_text(
        json.dumps(result['summary'], indent=2, ensure_ascii=False), encoding='utf-8')
