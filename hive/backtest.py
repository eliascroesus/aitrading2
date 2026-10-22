"""Fast single-position intraday backtester.

Execution model (deliberately conservative):
  * signal evaluated on bar close i -> market entry at bar i+1 open, plus slippage
  * stop and target are checked intrabar; if both are touched on the same bar the
    stop is assumed to fill first; gaps through a stop fill at the bar open
  * stops fill with slippage, targets (limit orders) fill at the target price
  * ``exit_when`` and time exits fill at that bar's close minus slippage
  * position is always flat by ``flat_at`` (prop firms close intraday accounts)
  * commissions per contract round trip; one position at a time
Each trade records MAE/MFE so the prop-firm simulator can check intraday
drawdown limits against open-trade equity, not just closed trades.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Dataset
from .instruments import Instrument

TRADE_FIELDS = ("entry_idx", "exit_idx", "side", "entry", "exit", "qty", "pnl", "mae", "mfe", "reason", "day")
REASONS = ("stop", "target", "exit", "time", "eod")


@dataclass
class Trades:
    entry_idx: np.ndarray
    exit_idx: np.ndarray
    side: np.ndarray  # +1 long / -1 short
    entry: np.ndarray
    exit: np.ndarray
    qty: np.ndarray
    pnl: np.ndarray  # USD net of costs
    mae: np.ndarray  # USD, worst open loss during the trade (<= 0)
    mfe: np.ndarray  # USD, best open profit during the trade (>= 0)
    reason: np.ndarray  # index into REASONS
    day: np.ndarray
    risk: np.ndarray  # USD at risk at entry (stop distance * qty * point value)

    def __len__(self) -> int:
        return len(self.pnl)


def _empty() -> Trades:
    z = np.zeros(0)
    zi = np.zeros(0, np.int64)
    return Trades(zi, zi, zi, z, z, z, z, z, z, zi, zi, z)


def run(ds: Dataset, inst: Instrument, spec, sig_long, sig_short, sig_exit, atr: np.ndarray,
        swing_low: np.ndarray | None = None, swing_high: np.ndarray | None = None,
        max_contracts: float | None = None, slippage_ticks: float = 1.0) -> Trades:
    n = ds.n
    o, h, l, c = ds.o, ds.h, ds.l, ds.c
    tod, day_id, day_end = ds.tod, ds.day_id, ds.day_end
    pv = inst.point_value
    slip = slippage_ticks * inst.tick_size
    comm = inst.commission_rt
    tick = inst.tick_size

    cand_long = np.flatnonzero(sig_long) if sig_long is not None else np.zeros(0, np.int64)
    cand_short = np.flatnonzero(sig_short) if sig_short is not None else np.zeros(0, np.int64)
    if len(cand_long) == 0 and len(cand_short) == 0:
        return _empty()
    cand = np.concatenate([cand_long, cand_short])
    sides = np.concatenate([np.ones(len(cand_long), np.int64), -np.ones(len(cand_short), np.int64)])
    order = np.argsort(cand, kind="stable")
    cand, sides = cand[order], sides[order]
    # drop bars where both directions fire (ambiguous)
    if len(cand_long) and len(cand_short):
        dup = np.zeros(len(cand), bool)
        dup[1:] = cand[1:] == cand[:-1]
        both = dup | np.concatenate([dup[1:], [False]])
        cand, sides = cand[~both], sides[~both]

    # per-day last allowed bar (bar must start before flat_at); exit at its close
    flat_at = spec.flat_at
    max_qty_cap = spec.max_contracts
    if max_contracts is not None:
        max_qty_cap = min(max_qty_cap, int(max_contracts / inst.mini_equiv + 1e-9))
    max_qty_cap = max(int(max_qty_cap), 0)

    rec = {k: [] for k in ("entry_idx", "exit_idx", "side", "entry", "exit", "qty", "pnl", "mae", "mfe", "reason", "day", "risk")}
    next_free = 0
    cur_day = -1
    trades_today = 0
    pnl_today = 0.0
    stop_type, stop_val = spec.stop_type, spec.stop_value
    tgt_type, tgt_val = spec.target_type, spec.target_value
    be_r = spec.breakeven_at_r
    max_hold = max(1, spec.max_hold_bars)
    daily_loss_stop = spec.daily_loss_stop_usd
    daily_profit_stop = spec.daily_profit_stop_usd

    for k in range(len(cand)):
        s = int(cand[k])
        if s < next_free:
            continue
        e = s + 1
        if e >= n or day_id[e] != day_id[s]:
            continue
        t_e = tod[e]
        if t_e < spec.session_start or t_e >= spec.session_end or t_e >= flat_at:
            continue
        d = int(day_id[e])
        if d != cur_day:
            cur_day, trades_today, pnl_today = d, 0, 0.0
        if trades_today >= spec.max_trades_per_day:
            continue
        if daily_loss_stop > 0 and pnl_today <= -daily_loss_stop:
            continue
        if daily_profit_stop > 0 and pnl_today >= daily_profit_stop:
            continue
        side = int(sides[k])
        entry = o[e] + side * slip
        # stop distance in points
        if stop_type == "atr":
            dist = stop_val * atr[s]
        elif stop_type == "pct":
            dist = stop_val / 100.0 * c[s]
        elif stop_type == "points":
            dist = stop_val
        else:  # swing
            ref = swing_low[s] if side > 0 else swing_high[s]
            dist = (entry - ref) if side > 0 else (ref - entry)
            dist += tick
        if not np.isfinite(dist) or dist < 2 * tick:
            continue
        qty = int(spec.risk_usd // (dist * pv + comm))
        qty = min(qty, max_qty_cap)
        if qty < 1:
            continue
        stop_px = entry - side * dist
        if tgt_type == "r":
            tgt_dist = tgt_val * dist
        elif tgt_type == "atr":
            tgt_dist = tgt_val * atr[s]
        elif tgt_type == "pct":
            tgt_dist = tgt_val / 100.0 * c[s]
        elif tgt_type == "points":
            tgt_dist = tgt_val
        else:
            tgt_dist = np.inf
        tgt_px = entry + side * tgt_dist if np.isfinite(tgt_dist) and tgt_dist > 0 else None

        # exit window: same day, bars starting before flat_at, at most max_hold bars
        w = np.arange(e, min(int(day_end[d]), n - 1) + 1)
        w = w[tod[w] < flat_at]
        if len(w) == 0:
            continue
        time_cut = len(w) > max_hold
        w = w[:max_hold]
        wl, wh = l[w], h[w]
        if side > 0:
            orig_hit = wl <= stop_px
            tgt_hit = wh >= tgt_px if tgt_px is not None else np.zeros(len(w), bool)
        else:
            orig_hit = wh >= stop_px
            tgt_hit = wl <= tgt_px if tgt_px is not None else np.zeros(len(w), bool)

        # breakeven: once price reaches be_r * dist in favour, the stop moves to entry from the next bar
        stop_hit = orig_hit
        if be_r > 0:
            be_lvl = entry + side * be_r * dist
            be_reached = (wh >= be_lvl) if side > 0 else (wl <= be_lvl)
            if be_reached.any():
                kb = int(np.argmax(be_reached))
                be_mask = np.zeros(len(w), bool)
                be_mask[kb + 1:] = (wl[kb + 1:] <= entry) if side > 0 else (wh[kb + 1:] >= entry)
                stop_hit = orig_hit | be_mask

        if sig_exit is not None:
            ex_hit = sig_exit[w].copy()
            ex_hit[0] = False  # the exit rule never fires on the entry bar itself
        else:
            ex_hit = None

        i_stop = int(np.argmax(stop_hit)) if stop_hit.any() else len(w)
        i_tgt = int(np.argmax(tgt_hit)) if tgt_hit.any() else len(w)
        i_ex = int(np.argmax(ex_hit)) if ex_hit is not None and ex_hit.any() else len(w)
        i_first = min(i_stop, i_tgt, i_ex)

        if i_first == len(w):
            j = int(w[-1])
            px = c[j] - side * slip
            reason = 3 if time_cut else 4
        else:
            j = int(w[i_first])
            if i_stop == i_first:
                level = stop_px if orig_hit[i_stop] else entry
                # a gap through the stop fills at the open
                px = (min(o[j], level) - slip) if side > 0 else (max(o[j], level) + slip)
                reason = 0
            elif i_tgt == i_first:
                if j == e:
                    px = tgt_px
                else:
                    px = max(o[j], tgt_px) if side > 0 else min(o[j], tgt_px)
                reason = 1
            else:
                px = c[j] - side * slip
                reason = 2

        if side > 0:
            worst = min(float(np.min(l[e:j + 1])), px)
            best = float(np.max(h[e:j + 1]))
            mae = (worst - entry) * pv * qty
            mfe = (best - entry) * pv * qty
        else:
            worst = max(float(np.max(h[e:j + 1])), px)
            best = float(np.min(l[e:j + 1]))
            mae = (entry - worst) * pv * qty
            mfe = (entry - best) * pv * qty
        pnl = side * (px - entry) * pv * qty - comm * qty
        mae = min(mae, 0.0) - comm * qty
        mfe = max(mfe, 0.0)

        rec["entry_idx"].append(e)
        rec["exit_idx"].append(j)
        rec["side"].append(side)
        rec["entry"].append(entry)
        rec["exit"].append(px)
        rec["qty"].append(qty)
        rec["pnl"].append(pnl)
        rec["mae"].append(mae)
        rec["mfe"].append(mfe)
        rec["reason"].append(reason)
        rec["day"].append(d)
        rec["risk"].append(dist * pv * qty)
        trades_today += 1
        pnl_today += pnl
        next_free = j + 1

    if not rec["pnl"]:
        return _empty()
    return Trades(
        entry_idx=np.asarray(rec["entry_idx"], np.int64),
        exit_idx=np.asarray(rec["exit_idx"], np.int64),
        side=np.asarray(rec["side"], np.int64),
        entry=np.asarray(rec["entry"]),
        exit=np.asarray(rec["exit"]),
        qty=np.asarray(rec["qty"], np.float64),
        pnl=np.asarray(rec["pnl"]),
        mae=np.asarray(rec["mae"]),
        mfe=np.asarray(rec["mfe"]),
        reason=np.asarray(rec["reason"], np.int64),
        day=np.asarray(rec["day"], np.int64),
        risk=np.asarray(rec["risk"]),
    )


def daily_records(trades: Trades, n_days: int) -> dict[str, np.ndarray]:
    """Per-day P&L, intraday low/high of cumulative equity (incl. open-trade excursion), trade counts."""
    pnl = np.zeros(n_days)
    low = np.zeros(n_days)
    high = np.zeros(n_days)
    count = np.zeros(n_days, np.int64)
    if len(trades):
        np.add.at(pnl, trades.day, trades.pnl)
        np.add.at(count, trades.day, 1)
        # cumulative closed pnl before each trade within its day
        order = np.argsort(trades.entry_idx, kind="stable")
        days = trades.day[order]
        p = trades.pnl[order]
        cum_after = np.zeros(len(p))
        running = 0.0
        prev_day = -1
        before = np.zeros(len(p))
        for i in range(len(p)):
            if days[i] != prev_day:
                running = 0.0
                prev_day = days[i]
            before[i] = running
            running += p[i]
            cum_after[i] = running
        lows = before + trades.mae[order]
        highs = before + trades.mfe[order]
        np.minimum.at(low, days, np.minimum(lows, cum_after))
        np.maximum.at(high, days, np.maximum(highs, cum_after))
    return {"pnl": pnl, "low": low, "high": high, "count": count}
