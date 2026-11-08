"""Tail metrics and paired inference for exchange sessions."""
import numpy as np


def session_adjacency(sessions, dates):
    """Mark adjacent exchange sessions while preserving gaps."""
    positions = sessions.get_indexer(dates)
    steps = np.diff(positions)
    if len(positions)==0 or (positions<0).any() or (steps<=0).any():
        raise ValueError("The sample requires known, unique sessions in chronological order.")
    return steps==1


def paired_mean_interval(difference,adjacent,block=10,rep=3000,seed=42):
    """Apply a circular bootstrap within each contiguous segment, preserving its size."""
    difference = np.asarray(difference,dtype=float)
    adjacent = np.asarray(adjacent,dtype=bool)
    if (difference.ndim!=1 or not len(difference) or not np.isfinite(difference).all()
        or adjacent.shape!=(len(difference)-1,) or block<1 or rep<1):
        raise ValueError("Invalid paired sample, adjacency, or configuration.")
    rng = np.random.default_rng(seed)
    thresholds = np.flatnonzero(~adjacent)+1
    sums = np.zeros(rep)
    for segment in np.split(difference,thresholds):
        k = min(block,len(segment))
        starts = rng.integers(len(segment),size=(rep,int(np.ceil(len(segment)/k))))
        indexes = ((starts[:,:,None]+np.arange(k))%len(segment)).reshape(rep,-1)
        sums += segment[indexes[:,:len(segment)]].sum(axis=1)
    return np.quantile(sums/len(difference),[.025,.975])


def fz0_loss(loss,var,es,alpha):
    """Joint quantile and tail loss under the positive-loss convention."""
    loss,var,es = (np.asarray(x,dtype=float) for x in (loss,var,es))
    if (not 0<alpha<1 or loss.ndim!=1 or not len(loss) or
        loss.shape!=var.shape or loss.shape!=es.shape or not np.isfinite(np.c_[loss,var,es]).all()):
        raise ValueError("FZ0 requires aligned finite vectors and a tail level between zero and one.")
    if (var<=0).any() or (es<var).any():
        raise ValueError("This FZ0 implementation requires CVaR >= VaR > 0.")
    return np.maximum(loss-var,0)/(alpha*es)+var/es+np.log(es)-1


def violation_clustering(violations,adjacent,rep=5000,seed=42):
    """One-sided test for excess pairs of adjacent violations."""
    violations = np.asarray(violations,dtype=bool)
    adjacent = np.asarray(adjacent,dtype=bool)
    if violations.ndim!=1 or not len(violations) or adjacent.shape!=(len(violations)-1,) or rep<1:
        raise ValueError("Invalid violations, adjacency, or replication count.")
    esquerda = np.flatnonzero(adjacent)
    total = int(violations.sum())
    pairs = int((violations[esquerda]&violations[esquerda+1]).sum())
    if len(esquerda)==0 or total<2 or total==len(violations):
        return pairs,np.nan
    rng = np.random.default_rng(seed)
    outliers = 0
    for _ in range(rep):
        simulated = np.zeros(len(violations),dtype=bool)
        simulated[rng.choice(len(violations),size=total,replace=False)] = True
        outliers += (simulated[esquerda]&simulated[esquerda+1]).sum()>=pairs
    return pairs,(outliers+1)/(rep+1)
