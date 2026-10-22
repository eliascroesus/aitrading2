"""Seed strategies (from the trading-strategy research) and a procedural strategy generator.

Seeds give the evolver real starting material covering the main families:
opening-range breakouts, liquidity sweeps (ICT / turtle soup), fair value gaps,
VWAP trend, intraday momentum, mean reversion and classic breakouts.  The
procedural generator lets the hive keep inventing strategies with no LLM.
"""
from __future__ import annotations

import numpy as np

from .strategy import FAMILIES, StrategySpec, fill_template, random_filter_pair

SEEDS: list[dict] = [
    dict(name="ORB 15m + VWAP", family="orb",
         long_entry="close > or_high(15) and close > vwap and tod < 690",
         short_entry="close < or_low(15) and close < vwap and tod < 690",
         stop_type="atr", stop_value=1.2, target_type="r", target_value=2.0, max_trades_per_day=1,
         rationale="Opening-range breakout (Crabel; Zarattini & Aziz) filtered by session VWAP."),
    dict(name="5m opening candle ORB", family="orb",
         long_entry="bar == 0 and close > open", short_entry="bar == 0 and close < open",
         stop_type="atr", stop_value=1.0, target_type="r", target_value=4.0, max_trades_per_day=1, session_start=570,
         rationale="Zarattini/Aziz 5-minute ORB: trade in the direction of the first candle, let it run."),
    dict(name="PDH/PDL liquidity sweep", family="liquidity_sweep",
         long_entry="sweep_pdl() and tod < 720", short_entry="sweep_pdh() and tod < 720",
         stop_type="swing", stop_value=3, target_type="r", target_value=2.0, max_trades_per_day=1,
         rationale="ICT / turtle-soup style: fade a raid on the previous day's high or low."),
    dict(name="Opening-range sweep reversal", family="liquidity_sweep",
         long_entry="sweep_or_low(30) and tod < 780", short_entry="sweep_or_high(30) and tod < 780",
         stop_type="swing", stop_value=4, target_type="r", target_value=2.0, max_trades_per_day=1,
         rationale="Sweep of the 30-minute opening range that closes back inside."),
    dict(name="Silver-bullet FVG", family="fvg",
         long_entry="tod >= 600 and tod < 660 and within(sweep_low(12), 6) and in_bull_fvg(6)",
         short_entry="tod >= 600 and tod < 660 and within(sweep_high(12), 6) and in_bear_fvg(6)",
         stop_type="swing", stop_value=6, target_type="r", target_value=2.0, max_trades_per_day=1,
         rationale="ICT Silver Bullet: 10-11am window, liquidity sweep then retrace into a fair value gap."),
    dict(name="FVG trend continuation", family="fvg",
         long_entry="in_bull_fvg(8) and ema(close, 20) > ema(close, 50) and close > vwap",
         short_entry="in_bear_fvg(8) and ema(close, 20) < ema(close, 50) and close < vwap",
         stop_type="atr", stop_value=1.5, target_type="r", target_value=2.0, max_trades_per_day=2,
         rationale="Buy retracements into bullish FVGs in an uptrend (and the mirror for shorts)."),
    dict(name="VWAP trend cross", family="vwap",
         long_entry="crossover(close, vwap) and tod >= 600", short_entry="crossunder(close, vwap) and tod >= 600",
         exit_when="", stop_type="atr", stop_value=2.0, target_type="none", target_value=0, max_trades_per_day=3,
         rationale="Zarattini & Aziz VWAP trend: hold the side of VWAP price is on."),
    dict(name="Last-half-hour momentum", family="momentum",
         long_entry="tod >= 925 and tod < 930 and open_ret(30) > 0", short_entry="tod >= 925 and tod < 930 and open_ret(30) < 0",
         stop_type="atr", stop_value=2.0, target_type="none", target_value=0, session_start=900, session_end=945, max_trades_per_day=1,
         rationale="Gao, Han, Li & Zhou intraday momentum: the first half-hour return predicts the last half-hour."),
    dict(name="RSI(2) intraday reversion", family="mean_reversion",
         long_entry="rsi(close, 2) < 8 and close > ema(close, 100)", short_entry="rsi(close, 2) > 92 and close < ema(close, 100)",
         exit_when="", stop_type="atr", stop_value=2.0, target_type="atr", target_value=1.5, max_trades_per_day=3,
         rationale="Connors RSI(2) pullback inside the prevailing trend."),
    dict(name="Bollinger fade in chop", family="mean_reversion",
         long_entry="close < bb_lower(close, 20, 2.5) and adx(14) < 20", short_entry="close > bb_upper(close, 20, 2.5) and adx(14) < 20",
         stop_type="atr", stop_value=1.5, target_type="r", target_value=1.5, max_trades_per_day=3,
         rationale="Fade 2.5-sigma band pokes when ADX says the market is ranging."),
    dict(name="Displacement BOS", family="breakout",
         long_entry="bos_up(24) and displacement(1.2) and close > vwap", short_entry="bos_down(24) and displacement(1.2) and close < vwap",
         stop_type="atr", stop_value=1.5, target_type="r", target_value=3.0, max_trades_per_day=2,
         rationale="Break of structure confirmed by a displacement candle (SMC)."),
    dict(name="EMA pullback trend", family="trend",
         long_entry="ema(close, 9) > ema(close, 21) and low <= ema(close, 21) and close > ema(close, 21) and adx(14) > 20",
         short_entry="ema(close, 9) < ema(close, 21) and high >= ema(close, 21) and close < ema(close, 21) and adx(14) > 20",
         stop_type="atr", stop_value=1.5, target_type="r", target_value=2.0, max_trades_per_day=2,
         rationale="Classic moving-average pullback with an ADX trend filter."),
    dict(name="Gap fade", family="gap",
         long_entry="bar == 0 and gap < -0.003 and gap > -0.012", short_entry="bar == 0 and gap > 0.003 and gap < 0.012",
         stop_type="atr", stop_value=2.0, target_type="r", target_value=1.5, max_trades_per_day=1, session_start=570,
         rationale="Small opening gaps tend to fill the same day."),
    dict(name="NR7 inside-bar breakout", family="breakout",
         long_entry="within(nr(7), 2) and bos_up(7) and close > vwap", short_entry="within(nr(7), 2) and bos_down(7) and close < vwap",
         stop_type="atr", stop_value=1.0, target_type="r", target_value=2.5, max_trades_per_day=2,
         rationale="Crabel volatility contraction (NR7) followed by expansion."),
    dict(name="Engulfing at VWAP", family="candle",
         long_entry="engulf_bull() and low <= vwap and close > vwap", short_entry="engulf_bear() and high >= vwap and close < vwap",
         stop_type="swing", stop_value=2, target_type="r", target_value=2.0, max_trades_per_day=2,
         rationale="Engulfing candle reclaiming VWAP."),
]

# procedural entry templates: (family, long template, short template)
ENTRY_TEMPLATES = [
    ("orb", "close > or_high({m})", "close < or_low({m})"),
    ("orb", "crossover(close, or_high({m}))", "crossunder(close, or_low({m}))"),
    ("liquidity_sweep", "sweep_low({p})", "sweep_high({p})"),
    ("liquidity_sweep", "sweep_pdl()", "sweep_pdh()"),
    ("liquidity_sweep", "sweep_or_low({m})", "sweep_or_high({m})"),
    ("liquidity_sweep", "within(sweep_low({p}), {w}) and bos_up({q})", "within(sweep_high({p}), {w}) and bos_down({q})"),
    ("fvg", "in_bull_fvg({w})", "in_bear_fvg({w})"),
    ("fvg", "fvg_bull() and displacement({k})", "fvg_bear() and displacement({k})"),
    ("fvg", "within(sweep_low({p}), {w}) and in_bull_fvg({w})", "within(sweep_high({p}), {w}) and in_bear_fvg({w})"),
    ("vwap", "crossover(close, vwap)", "crossunder(close, vwap)"),
    ("vwap", "low <= vwap and close > vwap and close > open", "high >= vwap and close < vwap and close < open"),
    ("trend", "crossover(ema(close, {a}), ema(close, {b}))", "crossunder(ema(close, {a}), ema(close, {b}))"),
    ("trend", "close > ema(close, {b}) and crossover(close, ema(close, {a}))", "close < ema(close, {b}) and crossunder(close, ema(close, {a}))"),
    ("mean_reversion", "rsi(close, {r}) < {lo}", "rsi(close, {r}) > {hi}"),
    ("mean_reversion", "close < bb_lower(close, {p}, {k})", "close > bb_upper(close, {p}, {k})"),
    ("mean_reversion", "zscore(close, {p}) < -{k}", "zscore(close, {p}) > {k}"),
    ("breakout", "bos_up({p})", "bos_down({p})"),
    ("breakout", "close > highest(high, {p}) - 0.0001 * close and displacement({k})", "close < lowest(low, {p}) + 0.0001 * close and displacement({k})"),
    ("momentum", "macd_hist(close, 12, 26, 9) > 0 and crossover(macd(close, 12, 26), macd_signal(close, 12, 26, 9))", "macd_hist(close, 12, 26, 9) < 0 and crossunder(macd(close, 12, 26), macd_signal(close, 12, 26, 9))"),
    ("momentum", "roc(close, {p}) > {pct}", "roc(close, {p}) < -{pct}"),
    ("candle", "pin_bull({k}) and low < lowest(low, {p}) + 0.25 * atr(14)", "pin_bear({k}) and high > highest(high, {p}) - 0.25 * atr(14)"),
    ("candle", "engulf_bull()", "engulf_bear()"),
    ("gap", "bar == 0 and gap < -{g}", "bar == 0 and gap > {g}"),
    ("session", "tod >= {t} and close > session_high(570, {t2})", "tod >= {t} and close < session_low(570, {t2})"),
]


def _params(rng: np.random.Generator) -> dict:
    a = int(rng.choice([5, 8, 9, 10, 12, 20]))
    t2 = int(rng.choice([600, 630, 660]))
    return {
        "m": int(rng.choice([5, 10, 15, 30, 45, 60])),
        "p": int(rng.choice([5, 8, 10, 12, 20, 30, 50])),
        "q": int(rng.choice([3, 5, 8])),
        "w": int(rng.choice([3, 4, 6, 8, 12])),
        "k": round(float(rng.choice([1.0, 1.2, 1.5, 2.0, 2.5])), 2),
        "a": a,
        "b": int(a * rng.choice([2, 3, 4, 5])),
        "r": int(rng.choice([2, 3, 4, 7, 14])),
        "lo": int(rng.choice([5, 10, 15, 20, 30])),
        "hi": int(rng.choice([70, 80, 85, 90, 95])),
        "pct": round(float(rng.choice([0.1, 0.2, 0.3, 0.5])), 2),
        "g": round(float(rng.choice([0.002, 0.003, 0.005, 0.008])), 4),
        "t": t2,
        "t2": t2,
    }


def random_strategy(rng: np.random.Generator, instruments: list[str], timeframes: list[int], author: str = "generator") -> StrategySpec:
    family, lt, st = ENTRY_TEMPLATES[int(rng.integers(len(ENTRY_TEMPLATES)))]
    p = _params(rng)
    long_e, short_e = lt.format(**p), st.format(**p)
    for _ in range(int(rng.choice([0, 1, 1, 2]))):
        fl, fs = random_filter_pair(rng)
        long_e, short_e = f"{long_e} and {fl}", f"{short_e} and {fs}"
    side = rng.choice(["both", "both", "long", "short"])
    stop_type = str(rng.choice(["atr", "atr", "swing", "pct"]))
    stop_value = {"atr": float(rng.choice([0.75, 1.0, 1.5, 2.0, 3.0])), "swing": float(rng.choice([2, 3, 5, 8])), "pct": float(rng.choice([0.1, 0.2, 0.3, 0.5]))}[stop_type]
    target_type = str(rng.choice(["r", "r", "r", "atr", "none"]))
    target_value = {"r": float(rng.choice([1.0, 1.5, 2.0, 3.0, 4.0])), "atr": float(rng.choice([1.0, 2.0, 3.0])), "none": 0.0}[target_type]
    spec = StrategySpec(
        name=f"{family.replace('_', ' ').title()} #{int(rng.integers(1000, 9999))}",
        family=family if family in FAMILIES else "other",
        instrument=str(rng.choice(instruments)),
        timeframe=int(rng.choice(timeframes)),
        long_entry=long_e if side in ("both", "long") else "",
        short_entry=short_e if side in ("both", "short") else "",
        stop_type=stop_type,
        stop_value=stop_value,
        target_type=target_type,
        target_value=target_value,
        breakeven_at_r=float(rng.choice([0.0, 0.0, 1.0])),
        max_hold_bars=int(rng.choice([6, 12, 24, 48, 96])),
        session_start=int(rng.choice([570, 575, 585, 600, 630])),
        session_end=int(rng.choice([660, 720, 840, 900, 945])),
        flat_at=int(rng.choice([900, 940, 955])),
        risk_usd=float(rng.choice([150, 200, 250, 300, 400, 500, 600])),
        max_trades_per_day=int(rng.choice([1, 1, 2, 3, 4])),
        daily_loss_stop_usd=float(rng.choice([0, 0, 500, 800])),
        daily_profit_stop_usd=float(rng.choice([0, 0, 1000, 1500])),
        rationale=f"Procedural {family} template",
        author=author,
        origin="random",
    )
    return spec.normalized(instruments, timeframes)


def seed_strategies(instruments: list[str], timeframes: list[int]) -> list[StrategySpec]:
    out = []
    for d in SEEDS:
        for inst in instruments:
            spec = StrategySpec.from_dict({**d, "instrument": inst, "timeframe": timeframes[0], "author": "seed", "origin": "seed"}, instruments, timeframes)
            spec.name = f"{d['name']} [{inst}]"
            if not spec.errors():
                out.append(spec)
    return out
