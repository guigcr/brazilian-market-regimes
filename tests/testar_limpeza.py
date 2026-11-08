"""Verifica regras que podem alterar retornos ou esconder falhas da fonte."""
import numpy as np
import pandas as pd
from limpeza_ibovespa import limpar_dados

idx = pd.to_datetime(['2008-01-02', '2008-01-03', '2008-01-04'])
b = pd.DataFrame({'Open': [100.,101,102], 'High': [102.,103,104],
                  'Low': [99.,100,101], 'Close': [101.,102,103],
                  'Volume': [10.,0,20]}, index=idx)
def limpar(d):
    return limpar_dados(d, '2008-01-01', '2008-01-05')
r = limpar(b)
assert len(r['analise']) == 3 and pd.isna(r['analise'].Volume.iloc[1])
assert pd.isna(r['analise'].ret_simples.iloc[0])
assert np.isclose(r['analise'].ret_simples.iloc[1], 102/101-1)
assert b.Volume.iloc[1] == 0  # fonte intacta
assert limpar(pd.concat([b.iloc[::-1], b.iloc[:1]]))['resumo']['duplicadas_identicas_removidas'] == 1
conflito = b.iloc[:1].copy(); conflito['Close'] = 100
try:
    limpar(pd.concat([b, conflito]))
except ValueError:
    pass
else:
    raise AssertionError('Duplicata conflitante foi aceita')
bad = b.copy(); bad.loc[idx[1], 'High'] = 1
assert len(limpar(bad)['rejeitadas']) == 1
assert pd.isna(limpar(bad)['analise'].ret_simples.iloc[-1])
assert pd.isna(limpar(b.drop(idx[1]))['analise'].ret_simples.iloc[-1])
multi = b.copy(); multi.columns = pd.MultiIndex.from_product([b.columns, ['^BVSP']])
assert len(limpar(multi)['analise']) == 3
multi2 = pd.concat({'A': b, 'B': b}, axis=1)
try:
    limpar(multi2)
except ValueError:
    pass
else:
    raise AssertionError('Múltiplos ativos foram aceitos')
print('Verificações de limpeza aprovadas.')
