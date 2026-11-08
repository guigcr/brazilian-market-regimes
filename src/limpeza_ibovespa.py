"""Limpeza conservadora: não imputa preços, volume ou pregões ausentes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from calendario_mercado import sessoes_esperadas

PRECOS = ['Open', 'High', 'Low', 'Close']

def limpar_dados(bruto, inicio='2000-01-01', fim='2026-01-01'):
    if bruto is None or bruto.empty:
        raise ValueError('Fonte vazia; não é possível validar os dados.')
    dados = bruto.copy(deep=True)
    if isinstance(dados.columns, pd.MultiIndex):
        # Aceita somente um ativo, independentemente da ordem dos níveis.
        niveis = [i for i in range(dados.columns.nlevels)
                  if set(PRECOS + ['Volume']).issubset(set(dados.columns.get_level_values(i))) ]
        if len(niveis) != 1:
            raise ValueError('Estrutura de colunas ambígua.')
        i = niveis[0]
        if dados.columns.droplevel(i).nunique() != 1:
            raise ValueError('Esperado somente um ativo.')
        dados.columns = dados.columns.get_level_values(i)
    if not dados.columns.is_unique:
        raise ValueError('Colunas duplicadas.')
    if not set(PRECOS + ['Volume']).issubset(dados.columns):
        raise ValueError('Faltam colunas OHLCV obrigatórias.')
    datas = pd.to_datetime(dados.index, errors='raise')
    if datas.tz is not None:
        datas = datas.tz_convert('America/Sao_Paulo').tz_localize(None)
    if datas.isna().any():
        raise ValueError('Datas ausentes na fonte.')
    horarios_normalizados = pd.Series(datas != datas.normalize(), index=datas.normalize()).groupby(level=0).any()
    dados.index = datas.normalize()
    dados.index.name = 'Date'
    dados = dados.sort_index(kind='stable')
    duplicadas = dados.loc[dados.index.duplicated(keep=False)]
    if any(len(g.drop_duplicates()) > 1 for _, g in duplicadas.groupby(level=0)):
        raise ValueError('Datas duplicadas com valores conflitantes; revisar a fonte.')
    n_duplicadas = int(dados.index.duplicated().sum())
    dados = dados.loc[~dados.index.duplicated()].copy()
    for col in dados.columns:
        dados[col] = pd.to_numeric(dados[col], errors='raise').astype(float)
    flags = pd.DataFrame(index=dados.index)
    flags['horario_normalizado'] = horarios_normalizados.reindex(dados.index, fill_value=False)
    flags['fora_periodo'] = (dados.index < pd.Timestamp(inicio)) | (dados.index >= pd.Timestamp(fim))
    flags['data_futura'] = dados.index > pd.Timestamp.now(tz='America/Sao_Paulo').tz_localize(None).normalize()
    # Calendário de pregões, não simples dias úteis. Nenhuma sessão é preenchida.
    esperadas = sessoes_esperadas('BVMF', inicio, pd.Timestamp(fim) - pd.Timedelta(days=1))
    flags['fora_calendario'] = ~dados.index.isin(esperadas)
    flags['preco_invalido'] = (~np.isfinite(dados[PRECOS]) | (dados[PRECOS] <= 0)).any(axis=1)
    flags['ohlc_inconsistente'] = (
        (dados.High < dados[PRECOS].max(axis=1)) |
        (dados.Low > dados[PRECOS].min(axis=1)))
    flags['volume_zero'] = dados.Volume.eq(0)
    flags['volume_invalido'] = ~np.isfinite(dados.Volume) | dados.Volume.lt(0)
    flags['volume_ausente'] = flags.volume_zero | flags.volume_invalido
    if 'Adj Close' in dados:
        flags['ajuste_invalido'] = ~np.isfinite(dados['Adj Close']) | dados['Adj Close'].le(0)
        dados.loc[flags.ajuste_invalido, 'Adj Close'] = np.nan
    excluir = flags[['fora_periodo', 'data_futura', 'fora_calendario',
                     'preco_invalido', 'ohlc_inconsistente']].any(axis=1)
    rejeitadas = dados.loc[excluir].join(flags.add_prefix('flag_'))
    dados.loc[flags.volume_ausente, 'Volume'] = np.nan
    completo = dados.loc[~excluir].copy()
    ausentes = esperadas.difference(completo.index)
    calendario = pd.DataFrame({'Date': ausentes})
    # Base contínua por PREÇOS: não remover linhas apenas por falta de volume.
    analise = completo.loc['2008-01-01':].copy()
    # Retornos somente entre sessões consecutivas do calendário. Impede que uma
    # lacuna de dados seja confundida com um retorno de um pregão.
    posicoes = pd.Series(np.arange(len(esperadas)), index=esperadas).reindex(analise.index)
    consecutivas = posicoes.diff().eq(1)
    analise['ret_simples'] = analise.Close.pct_change(fill_method=None).where(consecutivas)
    analise['ret_log'] = np.log(analise.Close / analise.Close.shift()).where(consecutivas)
    analise['Volume_disponivel'] = analise.Volume.notna()
    # Extremos são sinalizados, nunca apagados automaticamente.
    extremos = analise.loc[analise.ret_simples.abs() > 0.15].copy()
    anual = completo.Volume.groupby(completo.index.year).agg(['count', 'median', 'mean', 'max'])
    anual['razao_mediana_ano_anterior'] = anual['median'].div(anual['median'].shift())
    resumo = {
        'linhas_fonte': len(bruto), 'duplicadas_identicas_removidas': n_duplicadas,
        'linhas_rejeitadas': len(rejeitadas), 'linhas_completo': len(completo),
        'linhas_analise_desde_2008': len(analise),
        'primeira_data': str(analise.index.min()), 'ultima_data': str(analise.index.max()),
        'volume_ausente_analise': int(analise.Volume.isna().sum()),
        'sessoes_ausentes_periodo_completo': len(ausentes),
        'sessoes_ausentes_desde_2008': int((ausentes >= pd.Timestamp('2008-01-01')).sum()),
        'retornos_extremos_sinalizados': len(extremos),
        'contagem_flags': {k: int(v) for k, v in flags.sum().items()},
    }
    if analise.empty:
        raise ValueError('Base de análise vazia.')
    assert completo.index.is_unique and completo.index.is_monotonic_increasing
    assert np.isfinite(completo[PRECOS]).all().all()
    assert (completo[PRECOS] > 0).all().all()
    assert np.allclose(analise.ret_log.dropna(), np.log1p(analise.ret_simples.dropna()))
    return dict(completo=completo, analise=analise, flags=flags, rejeitadas=rejeitadas,
                sessoes_ausentes=calendario, extremos=extremos, volume_anual=anual, resumo=resumo)


def exportar(resultado, destino='dados'):
    pasta = Path(destino)
    pasta.mkdir(parents=True, exist_ok=True)
    nomes = {'completo': 'ibovespa_completo', 'analise': 'ibovespa_limpo',
             'flags': 'auditoria_flags', 'rejeitadas': 'linhas_rejeitadas',
             'sessoes_ausentes': 'sessoes_ausentes', 'extremos': 'retornos_extremos',
             'volume_anual': 'volume_por_ano'}
    for chave, nome in nomes.items():
        resultado[chave].to_csv(pasta / f'{nome}.csv', index=chave != 'sessoes_ausentes')
    (pasta / 'resumo_qualidade.json').write_text(
        json.dumps(resultado['resumo'], indent=2, ensure_ascii=False), encoding='utf-8')
