"""Barrier labeling with a time cutoff for the BOVA11 study."""
import numpy as np
import pandas as pd

OHLC_BARRIERS = ["Open","High","Low","Close"]


def simulate_event(bars, sessions, origin, side, limit, holding_period=5, cost_per_side=.0015):
    """Read prices in order after reserving a full window before the cutoff."""
    if side not in (-1,1) or holding_period<1 or not np.isfinite(cost_per_side) or cost_per_side<0:
        raise ValueError("Invalid trade side, holding period, or cost.")
    start = sessions.get_loc(origin)+1
    window = sessions[start:start+holding_period]
    event = {"origin":origin,"side":side,
              "entry_date":window[0] if len(window) else pd.NaT,
              "expiry":window[-1] if len(window)==holding_period else pd.NaT}
    if len(window)!=holding_period or window[-1]>=limit:
        return event|{"status":"outside_window","reason":"holding_period_crosses_cutoff"}
    marks = {}
    for pos,data in enumerate(window):
        if data not in bars.index:
            values = np.full(4,np.nan)
        else:
            values = bars.loc[data,OHLC_BARRIERS].to_numpy(dtype=float)
        open,maximum,minimum,close = values
        if (not np.isfinite(values).all() or (values<=0).any() or
            maximum<values.max() or minimum>values.min()):
            return event|{"status":"censored" if pos else "entry_unavailable",
                           "reason":"missing_or_invalid_ohlc","interruption_date":data}
        if pos==0:
            entry = open
            target = entry*(1+side*.015)
            stop = entry*(1-side*.01)
        worst = minimum if side==1 else maximum
        best = maximum if side==1 else minimum
        if side*(open-stop)<=0:
            price,reason = open,"stop_open"
        elif side*(open-target)>=0:
            price,reason = target,"target_open"
        elif side*(worst-stop)<=0:
            price,reason = stop,"stop"
        elif side*(best-target)>=0:
            price,reason = target,"target"
        elif pos==holding_period-1:
            price,reason = close,"holding_period"
        else:
            marks[data] = side*(close/entry-1)-cost_per_side
            continue
        net = side*(price/entry-1)-2*cost_per_side
        marks[data] = net
        return event|{"status":"closed","reason":reason,"output":data,
                       "net":net,"marks":marks}
    raise RuntimeError("An eligible window must end with an exit or censoring.")


def evaluate_events(bars,sessions,probabilities,limit,holding_period=5,cost_per_side=.0015):
    """Return observed labels and an audit of every opportunity."""
    if not probabilities.index.is_unique or not np.isfinite(probabilities).all():
        raise ValueError("Opportunities require unique origins and finite values.")
    closed, audit = {}, []
    for origin,value in probabilities.sort_index().items():
        event = simulate_event(bars,sessions,origin,1 if value>=.5 else -1,
                                limit,holding_period,cost_per_side)
        if event["status"]=="closed":
            closed[origin] = event
        audit.append({k:v for k,v in event.items() if k!="marks"})
    return closed,pd.DataFrame(audit,columns=["origin","side","entry_date",
        "expiry","status","reason","output","net","interruption_date"])


def uncensored_barrier(bars, index, limit):
    """End the panel before the first session with unreliable OHLC data."""
    ohlc = bars.reindex(index)[OHLC_BARRIERS]
    valid = (np.isfinite(ohlc).all(axis=1)&ohlc.gt(0).all(axis=1)&
              ohlc.High.ge(ohlc.max(axis=1))&ohlc.Low.le(ohlc.min(axis=1)))
    missing = ohlc.index[~valid]
    return min(limit,missing[0]) if len(missing) else limit
