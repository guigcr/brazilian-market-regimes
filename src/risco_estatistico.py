"""Métricas de cauda e inferência pareada para sessões de mercado."""
import numpy as np


def adjacencia_sessoes(sessoes, datas):
    """Marca vizinhos de calendário sem unir os dois lados de uma lacuna."""
    posicoes = sessoes.get_indexer(datas)
    passos = np.diff(posicoes)
    if len(posicoes)==0 or (posicoes<0).any() or (passos<=0).any():
        raise ValueError("A amostra precisa de sessões conhecidas, únicas e em ordem.")
    return passos==1


def intervalo_media_pareada(diferenca,adjacentes,bloco=10,rep=3000,semente=42):
    """Bootstrap circular dentro de cada trecho contínuo, com tamanho preservado."""
    diferenca = np.asarray(diferenca,dtype=float)
    adjacentes = np.asarray(adjacentes,dtype=bool)
    if (diferenca.ndim!=1 or not len(diferenca) or not np.isfinite(diferenca).all()
        or adjacentes.shape!=(len(diferenca)-1,) or bloco<1 or rep<1):
        raise ValueError("Amostra pareada, adjacência ou configuração inválida.")
    rng = np.random.default_rng(semente)
    cortes = np.flatnonzero(~adjacentes)+1
    somas = np.zeros(rep)
    for seg in np.split(diferenca,cortes):
        k = min(bloco,len(seg))
        inicios = rng.integers(len(seg),size=(rep,int(np.ceil(len(seg)/k))))
        indices = ((inicios[:,:,None]+np.arange(k))%len(seg)).reshape(rep,-1)
        somas += seg[indices[:,:len(seg)]].sum(axis=1)
    return np.quantile(somas/len(diferenca),[.025,.975])


def perda_fz0(perda,var,es,alpha):
    """Perda conjunta de quantil e cauda na convenção de perdas positivas."""
    perda,var,es = (np.asarray(x,dtype=float) for x in (perda,var,es))
    if (not 0<alpha<1 or perda.ndim!=1 or not len(perda) or
        perda.shape!=var.shape or perda.shape!=es.shape or not np.isfinite(np.c_[perda,var,es]).all()):
        raise ValueError("FZ0 exige vetores alinhados e finitos e nível de cauda entre zero e um.")
    if (var<=0).any() or (es<var).any():
        raise ValueError("FZ0 nesta implementação exige CVaR >= VaR > 0.")
    return np.maximum(perda-var,0)/(alpha*es)+var/es+np.log(es)-1


def agrupamento_violacoes(violacoes,adjacentes,rep=5000,semente=42):
    """Teste unilateral de excesso de pares de violações contíguas."""
    violacoes = np.asarray(violacoes,dtype=bool)
    adjacentes = np.asarray(adjacentes,dtype=bool)
    if violacoes.ndim!=1 or not len(violacoes) or adjacentes.shape!=(len(violacoes)-1,) or rep<1:
        raise ValueError("Violações, adjacência ou número de repetições inválido.")
    esquerda = np.flatnonzero(adjacentes)
    total = int(violacoes.sum())
    pares = int((violacoes[esquerda]&violacoes[esquerda+1]).sum())
    if len(esquerda)==0 or total<2 or total==len(violacoes):
        return pares,np.nan
    rng = np.random.default_rng(semente)
    extremos = 0
    for _ in range(rep):
        simuladas = np.zeros(len(violacoes),dtype=bool)
        simuladas[rng.choice(len(violacoes),size=total,replace=False)] = True
        extremos += (simuladas[esquerda]&simuladas[esquerda+1]).sum()>=pares
    return pares,(extremos+1)/(rep+1)
