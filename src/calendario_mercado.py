"""Grades de pregões diários para as bolsas declaradas nas fontes."""

import exchange_calendars as xcals
import pandas as pd


def calendario_e_sessoes(nome, inicio, fim):
    """Retorna o calendário e os rótulos de pregão entre duas datas inclusivas."""
    inicio, fim = pd.Timestamp(inicio), pd.Timestamp(fim)
    if any(pd.isna(dia) or dia.tz is not None or dia != dia.normalize()
           for dia in (inicio, fim)) or inicio > fim:
        raise ValueError("Informe datas locais, sem horário, em ordem crescente.")
    margem = pd.Timedelta(days=15)
    calendario = xcals.get_calendar(nome, start=inicio - margem, end=fim + margem)
    sessoes = calendario.sessions_in_range(inicio, fim)
    if sessoes.tz is not None:
        sessoes = sessoes.tz_localize(None)
    return calendario, sessoes


def sessoes_esperadas(nome, inicio, fim):
    """Uma fonte sem calendário conhecido não tem grade de sessões esperadas."""
    if nome is None:
        return None
    _, sessoes = calendario_e_sessoes(nome, inicio, fim)
    return sessoes
