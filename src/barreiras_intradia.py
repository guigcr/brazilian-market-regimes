"""Rotulagem com barreiras e corte temporal para o estudo de BOVA11."""
import numpy as np
import pandas as pd

OHLC_BARREIRAS = ["Open","High","Low","Close"]


def simular_evento(barras, sessoes, origem, lado, limite, prazo=5, custo_lado=.0015):
    """Lê preços em ordem, após reservar a janela inteira antes do corte."""
    if lado not in (-1,1) or prazo<1 or not np.isfinite(custo_lado) or custo_lado<0:
        raise ValueError("Lado, prazo ou custo inválido.")
    inicio = sessoes.get_loc(origem)+1
    janela = sessoes[inicio:inicio+prazo]
    evento = {"origem":origem,"lado":lado,
              "entrada_data":janela[0] if len(janela) else pd.NaT,
              "vencimento":janela[-1] if len(janela)==prazo else pd.NaT}
    if len(janela)!=prazo or janela[-1]>=limite:
        return evento|{"status":"fora_da_janela","motivo":"prazo_cruza_o_corte"}
    marcas = {}
    for pos,data in enumerate(janela):
        if data not in barras.index:
            valores = np.full(4,np.nan)
        else:
            valores = barras.loc[data,OHLC_BARREIRAS].to_numpy(dtype=float)
        abertura,maxima,minima,fechamento = valores
        if (not np.isfinite(valores).all() or (valores<=0).any() or
            maxima<valores.max() or minima>valores.min()):
            return evento|{"status":"censurado" if pos else "entrada_indisponivel",
                           "motivo":"ohlc_ausente_ou_invalido","data_interrupcao":data}
        if pos==0:
            entrada = abertura
            alvo = entrada*(1+lado*.015)
            stop = entrada*(1-lado*.01)
        pior = minima if lado==1 else maxima
        melhor = maxima if lado==1 else minima
        if lado*(abertura-stop)<=0:
            preco,motivo = abertura,"stop_abertura"
        elif lado*(abertura-alvo)>=0:
            preco,motivo = alvo,"alvo_abertura"
        elif lado*(pior-stop)<=0:
            preco,motivo = stop,"stop"
        elif lado*(melhor-alvo)>=0:
            preco,motivo = alvo,"alvo"
        elif pos==prazo-1:
            preco,motivo = fechamento,"prazo"
        else:
            marcas[data] = lado*(fechamento/entrada-1)-custo_lado
            continue
        liquido = lado*(preco/entrada-1)-2*custo_lado
        marcas[data] = liquido
        return evento|{"status":"encerrado","motivo":motivo,"saida":data,
                       "liquido":liquido,"marcas":marcas}
    raise RuntimeError("A janela elegível deveria terminar em uma saída ou censura.")


def avaliar_eventos(barras,sessoes,probabilidades,limite,prazo=5,custo_lado=.0015):
    """Devolve rótulos conhecidos e auditoria de todas as oportunidades."""
    if not probabilidades.index.is_unique or not np.isfinite(probabilidades).all():
        raise ValueError("As oportunidades exigem origens únicas e valores finitos.")
    encerrados, auditoria = {}, []
    for origem,valor in probabilidades.sort_index().items():
        evento = simular_evento(barras,sessoes,origem,1 if valor>=.5 else -1,
                                limite,prazo,custo_lado)
        if evento["status"]=="encerrado":
            encerrados[origem] = evento
        auditoria.append({k:v for k,v in evento.items() if k!="marcas"})
    return encerrados,pd.DataFrame(auditoria,columns=["origem","lado","entrada_data",
        "vencimento","status","motivo","saida","liquido","data_interrupcao"])


def limite_sem_censura(barras, indice, limite):
    """Encerra o painel antes da primeira sessão com OHLC não confiável."""
    ohlc = barras.reindex(indice)[OHLC_BARREIRAS]
    valido = (np.isfinite(ohlc).all(axis=1)&ohlc.gt(0).all(axis=1)&
              ohlc.High.ge(ohlc.max(axis=1))&ohlc.Low.le(ohlc.min(axis=1)))
    faltantes = ohlc.index[~valido]
    return min(limite,faltantes[0]) if len(faltantes) else limite
