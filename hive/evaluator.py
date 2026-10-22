"""Strategy evaluation: backtest -> daily records -> prop-challenge pass probability -> fitness.

Runs inside worker processes (see ``init_worker``) so thousands of strategies
can be scored in parallel.  Fitness only ever sees the train and validation
segments; the holdout segment is computed for display and never used for selection.
"""
from __future__ import annotations

import ast
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
        "pass_fast": mc["pass_fast"],
        "pass_hist": hist["pass"],
        **_luck_fields(rules, days, rng),
        "fail_maxloss": mc["fail_maxloss"],
        "fail_dll": mc["fail_dll"],
        "timeout": mc["timeout"],
        "median_days": mc["median_days"],
        "hist_median_days": hist["median_days"],
    }


def _luck_fields(rules: propfirm.PropRules, days: dict, rng: np.random.Generator) -> dict:
    luck = propfirm.luck_baseline(rules, days, rng)
    return {"luck": luck["pass"], "luck_fast": luck["pass_fast"]}


def complexity(spec: StrategySpec) -> int:
    """Number of conditions/indicators per side: a proxy for how easily a rule can be curve-fitted."""
    exprs = [e for e in (spec.long_entry, spec.short_entry) if e]
    if not exprs:
        return 0
    total = 0
    for e in exprs:
        tree = lang.parse(e)
        total += sum(isinstance(n, (ast.Compare, ast.Call)) for n in ast.walk(tree))
    return round(total / len(exprs))


def fitness(train: dict, val: dict, rules: propfirm.PropRules, folds: list[float] | None = None, n_conditions: int = 0) -> float:
    """Robust pass probability in [0, 1].

    Geometric mean of the Monte Carlo pass rates on three consecutive train folds and the
    validation segment (pass within PROP_TARGET_DAYS when set), so a strategy has to pass in
    every period rather than one lucky stretch. Discounted for too few trades, losing validation
    P&L and rules with many conditions (easier to curve-fit), with a small weight on speed."""
    key = "pass_fast" if rules.target_days else "pass_mc"
    rates = list(folds) if folds else [train[key]]
    rates.append(val[key])
    eps = 0.02
    base = max(0.0, math.exp(sum(math.log(max(p, 0.0) + eps) for p in rates) / len(rates)) - eps)
    trade_factor = min(1.0, train["trades"] / 40.0) * min(1.0, val["trades"] / 12.0)
    profit_factor = 1.0 if val["net"] > 0 else 0.5
    simplicity = 0.985 ** max(0, n_conditions - 4)
    md = val["median_days"] or rules.horizon_days
    speed = 1.0 - min(1.0, md / rules.horizon_days)
    # tiny tie-breaker on expectancy so zero-pass strategies still rank (gives the GA a gradient)
    tie = 0.002 * math.tanh((train["expectancy"] + val["expectancy"]) / 200.0)
    return float(min(1.0, max(0.0, base * trade_factor * profit_factor * simplicity * (0.94 + 0.06 * speed) + tie)))


def _equity_curve(ds: Dataset, day_pnl: np.ndarray, max_points: int = 260) -> list[list[float]]:
    eq = np.cumsum(day_pnl)
    n = len(eq)
    if n == 0:
        return []
    step = max(1, math.ceil(n / max_points))
    idx = np.unique(np.concatenate([np.arange(0, n, step), [n - 1]]))
    return [[int(ds.day_ts[i]), round(float(eq[i]), 2)] for i in idx]


def _backtest_spec(spec: StrategySpec) -> tuple[Dataset, backtest.Trades, dict]:
    key = f"{spec.instrument}_{spec.timeframe}"
    ds = _DATASETS[key]
    eng = engine_for(key)
    inst = get_instrument(spec.instrument, _RATIOS)
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
                          max_contracts=_RULES.max_contracts)
    return ds, trades, backtest.daily_records(trades, ds.n_days)


def _day_seg_metrics(day: dict, d0: int, d1: int, rules: propfirm.PropRules, rng: np.random.Generator) -> dict:
    """Challenge metrics for a slice of combined daily records (used for strategy combos)."""
    days = {k: v[d0:d1] for k, v in day.items()}
    dp = days["pnl"]
    trades = int(days["count"].sum())
    net = float(dp.sum())
    equity = np.concatenate([[0.0], np.cumsum(dp)])
    mc = propfirm.pass_probability_mc(rules, days, rng, sims=_MC_SIMS)
    hist = propfirm.pass_probability_hist(rules, days)
    traded = days["count"] > 0
    return {
        "days": int(d1 - d0), "trades": trades, "net": net,
        "expectancy": net / trades if trades else 0.0,
        "win_days": float((dp[traded] > 0).mean()) if traded.any() else 0.0,
        "max_dd": float(np.max(np.maximum.accumulate(equity) - equity)) if len(dp) else 0.0,
        "worst_day": float(dp.min()) if len(dp) else 0.0, "best_day": float(dp.max()) if len(dp) else 0.0,
        "pass_mc": mc["pass"], "pass_fast": mc["pass_fast"], "pass_hist": hist["pass"],
        "fail_maxloss": mc["fail_maxloss"], "fail_dll": mc["fail_dll"], "timeout": mc["timeout"],
        "median_days": mc["median_days"], **_luck_fields(rules, days, rng),
    }


COMBO_SCALES = (0.4, 0.55, 0.7, 0.85, 1.0)


def evaluate_combo(spec_dicts: list[dict]) -> dict:
    """Score several strategies traded together on one account (each on its own instrument).

    Daily results are added up (intraday lows are summed, which assumes the worst moments coincide),
    then every member's size is scaled by a common factor; the scale with the best fitness is kept."""
    try:
        members = []
        for d in spec_dicts:
            spec = StrategySpec.from_dict(d)
            errs = spec.errors()
            if errs:
                return {"ok": False, "error": "; ".join(errs)}
            members.append((spec, *_backtest_spec(spec)))
        dates = [(ds.day_ts // 86400).astype(np.int64) for _, ds, _, _ in members]
        all_dates = np.unique(np.concatenate(dates))
        n = len(all_dates)
        parts = []
        for (spec, ds, trades, rec), dd in zip(members, dates):
            idx = np.searchsorted(all_dates, dd)
            aligned = {k: np.zeros(n) for k in ("pnl", "low", "high")}
            aligned["count"] = np.zeros(n, np.int64)
            for k in aligned:
                aligned[k][idx] = rec[k]
            parts.append(aligned)
        rules = _RULES
        a, b = int(n * 0.6), int(n * 0.8)
        edges = np.linspace(0, a, 4).astype(int)
        fold_key = "pass_fast" if rules.target_days else "pass"
        uid_key = "+".join(sorted(StrategySpec.from_dict(d).uid for d in spec_dicts))
        best = None
        for scale in COMBO_SCALES:
            rng = np.random.default_rng(zlib.crc32(f"{uid_key}:{scale}".encode()))
            day = {k: scale * sum(p[k] for p in parts) for k in ("pnl", "low", "high")}
            day["count"] = sum(p["count"] for p in parts)
            segs = {"train": _day_seg_metrics(day, 0, a, rules, rng), "val": _day_seg_metrics(day, a, b, rules, rng),
                    "test": _day_seg_metrics(day, b, n, rules, rng)}
            folds = [propfirm.pass_probability_mc(rules, {k: v[x:y] for k, v in day.items()}, rng,
                                                  sims=max(100, _MC_SIMS // 2))[fold_key]
                     for x, y in zip(edges[:-1], edges[1:])]
            fit = fitness(segs["train"], segs["val"], rules, folds)
            if best is None or fit > best["fitness"]:
                best = {"fitness": fit, "scale": scale, "folds": folds, **segs, "day": day}
        traded = [p["pnl"][p["count"] > 0] for p in parts]
        pnl_mat = np.vstack([p["pnl"] for p in parts])
        corr = float(np.nanmean([np.corrcoef(pnl_mat[i], pnl_mat[j])[0, 1]
                                 for i in range(len(parts)) for j in range(i + 1, len(parts))])) if len(parts) > 1 else 1.0
        eq = np.cumsum(best.pop("day")["pnl"])
        step = max(1, math.ceil(n / 260))
        idx = np.unique(np.concatenate([np.arange(0, n, step), [n - 1]]))
        return {
            "ok": True, "key": uid_key, "members": [StrategySpec.from_dict(d).uid for d in spec_dicts],
            "corr": corr if np.isfinite(corr) else 0.0, "fast_days": rules.fast_days,
            "member_trades": [int(len(t)) for t in traded],
            "equity": [[int(all_dates[i] * 86400), round(float(eq[i]), 2)] for i in idx],
            "split_ts": [int(all_dates[a] * 86400) if a < n else None, int(all_dates[b] * 86400) if b < n else None],
            **best,
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def evaluate_combos(groups: list[list[dict]]) -> list[dict]:
    return [evaluate_combo(g) for g in groups]


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
        rules = _RULES
        ds, trades, day_rec = _backtest_spec(spec)
        rng = np.random.default_rng(zlib.crc32(uid.encode()))
        segs = {name: _seg_metrics(trades, day_rec, *_segment_days(ds, name), rules, rng) for name in ("train", "val", "test")}
        d0, d1 = ds.splits["train"]
        edges = np.linspace(d0, d1, 4).astype(int)
        fold_key = "pass_fast" if rules.target_days else "pass"
        folds = [
            propfirm.pass_probability_mc(rules, {k: v[a:b] for k, v in day_rec.items()}, rng, sims=max(100, _MC_SIMS // 2))[fold_key]
            for a, b in zip(edges[:-1], edges[1:])
        ]
        n_cond = complexity(spec)
        fit = fitness(segs["train"], segs["val"], rules, folds, n_cond)
        reasons = np.bincount(trades.reason, minlength=len(backtest.REASONS)) if len(trades) else np.zeros(len(backtest.REASONS), int)
        return {
            "ok": True,
            "uid": uid,
            "fitness": fit,
            "folds": folds,
            "conditions": n_cond,
            "fast_days": rules.fast_days,
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
