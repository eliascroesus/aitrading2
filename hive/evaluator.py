"""Strategy evaluation: backtest -> daily records -> prop-challenge pass probability -> fitness.

Runs inside worker processes (see ``init_worker``) so thousands of strategies
can be scored in parallel.  Fitness only ever sees the train and validation
segments; the holdout segment is computed for display and never used for selection.
"""
from __future__ import annotations

import math
import pickle
import zlib

import numpy as np

from . import backtest, lang, propfirm
from .data import Dataset
from .instruments import get_instrument
from .strategy import StrategySpec

_DATASETS: dict[str, Dataset] = {}
_ENGINES: dict[str, lang.Engine] = {}
_RULES: propfirm.PropRules | None = None
_RATIOS: dict[str, float] = {}
_MC_SIMS = 400


def init_worker(dataset_files: dict[str, str], rules: dict, ratios: dict[str, float], mc_sims: int = 400) -> None:
    global _RULES, _RATIOS, _MC_SIMS
    for key, path in dataset_files.items():
        with open(path, "rb") as fh:
            _DATASETS[key] = pickle.load(fh)
    _ENGINES.clear()
    _RULES = propfirm.PropRules(**rules)
    _RATIOS = dict(ratios)
    _MC_SIMS = mc_sims


def init_local(datasets: dict[str, Dataset], rules: propfirm.PropRules, ratios: dict[str, float], mc_sims: int = 400) -> None:
    """Same as init_worker but for in-process use (tests, live trading)."""
    global _RULES, _RATIOS, _MC_SIMS
    _DATASETS.clear()
    _DATASETS.update(datasets)
    _ENGINES.clear()
    _RULES = rules
    _RATIOS = dict(ratios)
    _MC_SIMS = mc_sims


def engine_for(key: str) -> lang.Engine:
    eng = _ENGINES.get(key)
    if eng is None:
        eng = lang.Engine(_DATASETS[key])
        _ENGINES[key] = eng
    return eng


def _segment_days(ds: Dataset, seg: str) -> tuple[int, int]:
    return ds.splits[seg]


def _seg_metrics(tr: backtest.Trades, day_rec: dict, d0: int, d1: int, rules: propfirm.PropRules, rng: np.random.Generator) -> dict:
    mask = (tr.day >= d0) & (tr.day < d1)
    pnl = tr.pnl[mask]
    risk = tr.risk[mask]
    days = {k: v[d0:d1] for k, v in day_rec.items()}
    n = int(mask.sum())
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    dp = days["pnl"]
    equity = np.cumsum(dp)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0.0], equity])) - np.concatenate([[0.0], equity]))) if len(equity) else 0.0
    sd = float(np.std(dp)) if len(dp) > 1 else 0.0
    mc = propfirm.pass_probability_mc(rules, days, rng, sims=_MC_SIMS)
    hist = propfirm.pass_probability_hist(rules, days)
    return {
        "days": int(d1 - d0),
        "trades": n,
        "net": float(pnl.sum()),
        "win_rate": float(len(wins) / n) if n else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0),
        "expectancy": float(pnl.mean()) if n else 0.0,
        "avg_r": float(np.mean(pnl / np.where(risk > 0, risk, 1.0))) if n else 0.0,
        "max_dd": dd,
        "sharpe": float(np.mean(dp) / sd * math.sqrt(252)) if sd > 0 else 0.0,
        "worst_day": float(dp.min()) if len(dp) else 0.0,
        "best_day": float(dp.max()) if len(dp) else 0.0,
        "trade_days": int((days["count"] > 0).sum()),
        "pass_mc": mc["pass"],
        "pass_hist": hist["pass"],
        "fail_maxloss": mc["fail_maxloss"],
        "fail_dll": mc["fail_dll"],
        "timeout": mc["timeout"],
        "median_days": mc["median_days"],
        "hist_median_days": hist["median_days"],
    }


def fitness(train: dict, val: dict, rules: propfirm.PropRules, folds: list[float] | None = None) -> float:
    """Robust pass probability in [0, 1].

    Geometric mean of the Monte Carlo pass rates on three consecutive train folds and the
    validation segment, so a strategy has to pass in every period rather than one lucky stretch.
    Discounted for too few trades and for losing validation P&L, with a small weight on speed."""
    rates = list(folds) if folds else [train["pass_mc"]]
    rates.append(val["pass_mc"])
    eps = 0.02
    base = max(0.0, math.exp(sum(math.log(max(p, 0.0) + eps) for p in rates) / len(rates)) - eps)
    trade_factor = min(1.0, train["trades"] / 40.0) * min(1.0, val["trades"] / 12.0)
    profit_factor = 1.0 if val["net"] > 0 else 0.5
    md = val["median_days"] or rules.horizon_days
    speed = 1.0 - min(1.0, md / rules.horizon_days)
    # tiny tie-breaker on expectancy so zero-pass strategies still rank (gives the GA a gradient)
    tie = 0.002 * math.tanh((train["expectancy"] + val["expectancy"]) / 200.0)
    return float(min(1.0, max(0.0, base * trade_factor * profit_factor * (0.94 + 0.06 * speed) + tie)))


def _equity_curve(ds: Dataset, day_pnl: np.ndarray, max_points: int = 260) -> list[list[float]]:
    eq = np.cumsum(day_pnl)
    n = len(eq)
    if n == 0:
        return []
    step = max(1, math.ceil(n / max_points))
    idx = np.unique(np.concatenate([np.arange(0, n, step), [n - 1]]))
    return [[int(ds.day_ts[i]), round(float(eq[i]), 2)] for i in idx]


def evaluate(spec_dict: dict) -> dict:
    """Evaluate one strategy. Never raises: errors come back as {'ok': False, 'error': ...}."""
    try:
        spec = StrategySpec.from_dict(spec_dict)
        uid = spec.uid
        errs = spec.errors()
        if errs:
            return {"ok": False, "uid": uid, "error": "; ".join(errs)}
        key = f"{spec.instrument}_{spec.timeframe}"
        if key not in _DATASETS:
            return {"ok": False, "uid": uid, "error": f"no data loaded for {spec.instrument} {spec.timeframe}m (available: {', '.join(sorted(_DATASETS))})"}
        ds = _DATASETS[key]
        eng = engine_for(key)
        inst = get_instrument(spec.instrument, _RATIOS)
        rules = _RULES
        sig_long = eng.signal(spec.long_entry) if spec.long_entry else None
        sig_short = eng.signal(spec.short_entry) if spec.short_entry else None
        sig_exit = eng.signal(spec.exit_when) if spec.exit_when else None
        atr = eng.series("atr(14)")
        swing_low = swing_high = None
        if spec.stop_type == "swing":
            n = int(spec.stop_value)
            swing_low = eng.series(f"lowest(low, {n})")
            swing_high = eng.series(f"highest(high, {n})")
        trades = backtest.run(ds, inst, spec, sig_long, sig_short, sig_exit, atr, swing_low, swing_high,
                              max_contracts=rules.max_contracts)
        day_rec = backtest.daily_records(trades, ds.n_days)
        rng = np.random.default_rng(zlib.crc32(uid.encode()))
        segs = {name: _seg_metrics(trades, day_rec, *_segment_days(ds, name), rules, rng) for name in ("train", "val", "test")}
        d0, d1 = ds.splits["train"]
        edges = np.linspace(d0, d1, 4).astype(int)
        folds = [
            propfirm.pass_probability_mc(rules, {k: v[a:b] for k, v in day_rec.items()}, rng, sims=max(100, _MC_SIMS // 2))["pass"]
            for a, b in zip(edges[:-1], edges[1:])
        ]
        fit = fitness(segs["train"], segs["val"], rules, folds)
        reasons = np.bincount(trades.reason, minlength=len(backtest.REASONS)) if len(trades) else np.zeros(len(backtest.REASONS), int)
        return {
            "ok": True,
            "uid": uid,
            "fitness": fit,
            "folds": folds,
            "train": segs["train"],
            "val": segs["val"],
            "test": segs["test"],
            "trades": int(len(trades)),
            "long_share": float((trades.side > 0).mean()) if len(trades) else 0.0,
            "avg_qty": float(trades.qty.mean()) if len(trades) else 0.0,
            "avg_hold_bars": float((trades.exit_idx - trades.entry_idx + 1).mean()) if len(trades) else 0.0,
            "exit_reasons": {r: int(x) for r, x in zip(backtest.REASONS, reasons)},
            "equity": _equity_curve(ds, day_rec["pnl"]),
            "split_ts": [int(ds.day_ts[ds.splits["val"][0]]) if ds.splits["val"][0] < ds.n_days else None,
                         int(ds.day_ts[ds.splits["test"][0]]) if ds.splits["test"][0] < ds.n_days else None],
        }
    except lang.ExprError as exc:
        return {"ok": False, "uid": spec_dict.get("uid", "?"), "error": str(exc)}
    except Exception as exc:  # a broken strategy must never kill a worker
        return {"ok": False, "uid": spec_dict.get("uid", "?"), "error": f"{type(exc).__name__}: {exc}"}


def evaluate_many(spec_dicts: list[dict]) -> list[dict]:
    return [evaluate(d) for d in spec_dicts]
