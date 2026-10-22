"""Prop-firm evaluation rules and the challenge simulator.

The simulator replays a strategy's daily results (P&L plus intraday equity
low/high including open-trade excursion) through a firm's rulebook: profit
target, max loss (static, end-of-day trailing or intraday trailing), daily loss
limit, consistency rule, minimum trading days and a time limit.  Pass
probability is estimated two ways:

* ``hist``: start a challenge on every historical day and replay what actually followed
* ``mc``:   block-bootstrap Monte Carlo (5-day blocks keep volatility clustering)

PRESETS are approximations of public rulebooks and firms change them often:
check the firm's current rules and edit or add a preset (or set PROP_PRESET=custom
with PROP_* overrides in .env).
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class PropRules:
    key: str
    name: str
    start_balance: float
    profit_target: float
    max_loss: float
    trailing: str  # "static" | "eod" | "intraday"
    trail_locks_at_start: bool  # trailing floor stops rising once it reaches the starting balance
    daily_loss_limit: float | None
    dll_fails_account: bool  # True: breaching the DLL fails the challenge; False: trading just halts for the day
    consistency_pct: float | None  # best day must be <= pct * total profit to pass
    min_days: int
    horizon_days: int  # how many trading days one attempt is simulated for
    max_contracts: float  # in full-size (mini) contract equivalents; micros count 0.1
    fee_usd: float

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        if self.trailing == "static":
            max_loss = f"max loss ${self.max_loss:,.0f} (static floor)"
        else:
            lock = ", stops trailing once it reaches the starting balance" if self.trail_locks_at_start else ""
            max_loss = f"max loss ${self.max_loss:,.0f} ({'end-of-day' if self.trailing == 'eod' else 'intraday'} trailing{lock})"
        parts = [f"{self.name}: ${self.start_balance:,.0f} account, profit target +${self.profit_target:,.0f}", max_loss]
        if self.daily_loss_limit:
            parts.append(f"daily loss limit ${self.daily_loss_limit:,.0f} ({'fails the account' if self.dll_fails_account else 'halts trading for the day'})")
        if self.consistency_pct:
            parts.append(f"consistency: best day <= {self.consistency_pct:.0%} of total profit")
        if self.min_days:
            parts.append(f"min {self.min_days} trading days")
        parts.append(f"max {self.max_contracts:g} contracts (micros count 0.1)")
        parts.append(f"simulated attempt length {self.horizon_days} trading days")
        return "; ".join(parts)


PRESETS: dict[str, PropRules] = {
    "topstep_50k": PropRules("topstep_50k", "Topstep 50K Trading Combine", 50_000, 3_000, 2_000, "eod", True, 1_000, False, 0.5, 2, 30, 5, 49),
    "topstep_100k": PropRules("topstep_100k", "Topstep 100K Trading Combine", 100_000, 6_000, 3_000, "eod", True, 2_000, False, 0.5, 2, 30, 10, 99),
    "topstep_150k": PropRules("topstep_150k", "Topstep 150K Trading Combine", 150_000, 9_000, 4_500, "eod", True, 3_000, False, 0.5, 2, 30, 15, 149),
    "apex_50k": PropRules("apex_50k", "Apex 50K Evaluation", 50_000, 3_000, 2_500, "intraday", True, None, False, None, 1, 30, 10, 167),
    "apex_100k": PropRules("apex_100k", "Apex 100K Evaluation", 100_000, 6_000, 3_000, "intraday", True, None, False, None, 1, 30, 14, 207),
    "tradeify_50k": PropRules("tradeify_50k", "Tradeify-style 50K (EOD trailing)", 50_000, 3_000, 2_000, "eod", True, 1_250, False, 0.4, 1, 30, 4, 99),
    "ftmo_100k": PropRules("ftmo_100k", "FTMO-style 100K (10% target, 5% daily, 10% max)", 100_000, 10_000, 10_000, "static", False, 5_000, True, None, 4, 30, 10, 540),
}


def get_rules(key: str) -> PropRules:
    key = (key or "topstep_50k").lower()
    if key == "custom":
        env = os.getenv
        return PropRules(
            "custom",
            env("PROP_NAME", "Custom challenge"),
            float(env("PROP_START_BALANCE", "50000")),
            float(env("PROP_PROFIT_TARGET", "3000")),
            float(env("PROP_MAX_LOSS", "2000")),
            env("PROP_TRAILING", "eod"),
            env("PROP_TRAIL_LOCKS", "true").lower() == "true",
            float(env("PROP_DAILY_LOSS", "0")) or None,
            env("PROP_DLL_FAILS", "false").lower() == "true",
            float(env("PROP_CONSISTENCY", "0")) or None,
            int(env("PROP_MIN_DAYS", "1")),
            int(env("PROP_HORIZON_DAYS", "30")),
            float(env("PROP_MAX_CONTRACTS", "5")),
            float(env("PROP_FEE", "0")),
        )
    if key not in PRESETS:
        raise KeyError(f"unknown PROP_PRESET '{key}'. Options: {', '.join(PRESETS)}, custom")
    return PRESETS[key]


# status codes
ACTIVE, PASSED, FAIL_MAXLOSS, FAIL_DLL = 0, 1, 2, 3


def simulate(rules: PropRules, pnl: np.ndarray, low: np.ndarray, high: np.ndarray, traded: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised challenge simulation. Inputs are (sims, days) arrays. Returns status, days_to_pass, final balance."""
    sims, days = pnl.shape
    start = rules.start_balance
    bal = np.full(sims, start, dtype=np.float64)
    peak = bal.copy()
    floor = np.full(sims, start - rules.max_loss)
    status = np.zeros(sims, np.int8)
    days_to_pass = np.full(sims, -1, np.int32)
    best_day = np.zeros(sims)
    days_traded = np.zeros(sims, np.int32)
    dll = rules.daily_loss_limit
    for t in range(days):
        act = status == ACTIVE
        if not act.any():
            break
        p = pnl[:, t]
        lo = low[:, t]
        hi = high[:, t]
        if dll:
            breach = lo <= -dll
            if rules.dll_fails_account:
                status[act & breach] = FAIL_DLL
                act = status == ACTIVE
            else:  # trading halts at the limit: the day closes at -dll
                p = np.where(breach, -dll, p)
                lo = np.maximum(lo, -dll)
                hi = np.where(breach, np.minimum(hi, 0.0), hi)
        if rules.trailing == "intraday":
            # worst case ordering: the intraday peak happens before the intraday low
            ipeak = np.maximum(peak, bal + hi)
            ifloor = ipeak - rules.max_loss
            if rules.trail_locks_at_start:
                ifloor = np.minimum(ifloor, start)
            hit = (bal + lo) <= ifloor
            peak = np.where(act, ipeak, peak)
        else:
            hit = (bal + lo) <= floor
        status[act & hit] = FAIL_MAXLOSS
        act = status == ACTIVE
        bal = np.where(act, bal + p, bal)
        days_traded += (act & traded[:, t]).astype(np.int32)
        best_day = np.where(act, np.maximum(best_day, p), best_day)
        if rules.trailing in ("eod", "intraday"):
            peak = np.where(act, np.maximum(peak, bal), peak)
            new_floor = peak - rules.max_loss
            if rules.trail_locks_at_start:
                new_floor = np.minimum(new_floor, start)
            floor = np.where(act, np.maximum(floor, new_floor), floor)
        failed = act & (bal <= floor)
        status[failed] = FAIL_MAXLOSS
        act = status == ACTIVE
        profit = bal - start
        cons_ok = np.ones(sims, bool) if not rules.consistency_pct else best_day <= rules.consistency_pct * profit
        passed = act & (profit >= rules.profit_target) & (days_traded >= rules.min_days) & cons_ok
        status[passed] = PASSED
        days_to_pass[passed] = t + 1
    return status, days_to_pass, bal


def _summarise(status: np.ndarray, days_to_pass: np.ndarray, final_bal: np.ndarray, rules: PropRules) -> dict:
    n = max(len(status), 1)
    passed = status == PASSED
    d = days_to_pass[passed]
    return {
        "pass": float(passed.mean()) if len(status) else 0.0,
        "fail_maxloss": float((status == FAIL_MAXLOSS).sum() / n),
        "fail_dll": float((status == FAIL_DLL).sum() / n),
        "timeout": float((status == ACTIVE).sum() / n),
        "median_days": float(np.median(d)) if len(d) else None,
        "mean_days": float(np.mean(d)) if len(d) else None,
        "avg_final_pnl": float(np.mean(final_bal - rules.start_balance)) if len(final_bal) else 0.0,
        "n": int(len(status)),
    }


def pass_probability_mc(rules: PropRules, day: dict[str, np.ndarray], rng: np.random.Generator, sims: int = 400, block: int = 5) -> dict:
    pnl, low, high, traded = day["pnl"], day["low"], day["high"], day["count"] > 0
    D = len(pnl)
    if D == 0:
        return _summarise(np.zeros(0, np.int8), np.zeros(0, np.int32), np.zeros(0), rules)
    T = rules.horizon_days
    nb = math.ceil(T / block)
    starts = rng.integers(0, D, size=(sims, nb))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % D).reshape(sims, nb * block)[:, :T]
    st, dtp, fb = simulate(rules, pnl[idx], low[idx], high[idx], traded[idx])
    return _summarise(st, dtp, fb, rules)


def pass_probability_hist(rules: PropRules, day: dict[str, np.ndarray]) -> dict:
    """Start an attempt on every day; attempts running past the data end are padded with no-trade days."""
    pnl, low, high, traded = day["pnl"], day["low"], day["high"], day["count"] > 0
    D = len(pnl)
    if D == 0:
        return _summarise(np.zeros(0, np.int8), np.zeros(0, np.int32), np.zeros(0), rules)
    T = rules.horizon_days
    last_start = max(1, D - min(T, D) // 2)  # allow partial windows near the end, but not tiny ones
    idx = np.arange(last_start)[:, None] + np.arange(T)[None, :]
    valid = idx < D
    idx = np.minimum(idx, D - 1)
    z = np.zeros_like(idx, dtype=np.float64)
    st, dtp, fb = simulate(
        rules,
        np.where(valid, pnl[idx], z),
        np.where(valid, low[idx], z),
        np.where(valid, high[idx], z),
        np.where(valid, traded[idx], False),
    )
    return _summarise(st, dtp, fb, rules)
