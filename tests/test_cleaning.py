"""Check cleaning rules that affect returns or flag source defects."""
import numpy as np
import pandas as pd
from ibovespa_cleaning import clean_data

idx = pd.to_datetime(['2008-01-02', '2008-01-03', '2008-01-04'])
b = pd.DataFrame({'Open': [100.,101,102], 'High': [102.,103,104],
                  'Low': [99.,100,101], 'Close': [101.,102,103],
                  'Volume': [10.,0,20]}, index=idx)
def clean(d):
    return clean_data(d, '2008-01-01', '2008-01-05')
r = clean(b)
assert len(r['analysis']) == 3 and pd.isna(r['analysis'].Volume.iloc[1])
assert pd.isna(r['analysis'].simple_return.iloc[0])
assert np.isclose(r['analysis'].simple_return.iloc[1], 102/101-1)
assert b.Volume.iloc[1] == 0  # fonte intacta
assert clean(pd.concat([b.iloc[::-1], b.iloc[:1]]))['summary']['duplicates_identical_removed'] == 1
conflito = b.iloc[:1].copy(); conflito['Close'] = 100
try:
    clean(pd.concat([b, conflito]))
except ValueError:
    pass
else:
    raise AssertionError('Duplicata conflitante foi aceita')
bad = b.copy(); bad.loc[idx[1], 'High'] = 1
assert len(clean(bad)['rejected']) == 1
assert pd.isna(clean(bad)['analysis'].simple_return.iloc[-1])
assert pd.isna(clean(b.drop(idx[1]))['analysis'].simple_return.iloc[-1])
multi = b.copy(); multi.columns = pd.MultiIndex.from_product([b.columns, ['^BVSP']])
assert len(clean(multi)['analysis']) == 3
multi2 = pd.concat({'A': b, 'B': b}, axis=1)
try:
    clean(multi2)
except ValueError:
    pass
else:
    raise AssertionError('Multiple assets were accepted')
print('Cleaning checks passed.')
