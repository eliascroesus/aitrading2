"""Strategy specification plus the genetic operators (mutation / crossover) the evolver uses."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields

import numpy as np

from . import lang

FAMILIES = [
    "orb", "liquidity_sweep", "fvg", "vwap", "trend", "mean_reversion", "breakout",
    "momentum", "gap", "session", "candle", "hybrid", "other",
]
STOP_TYPES = ("atr", "pct", "points", "swing")
TARGET_TYPES = ("r", "atr", "pct", "points", "none")
TRADING_FIELDS = (
    "instrument", "timeframe", "long_entry", "short_entry", "exit_when", "stop_type", "stop_value",
    "target_type", "target_value", "breakeven_at_r", "max_hold_bars", "session_start", "session_end",
    "flat_at", "risk_usd", "max_contracts", "max_trades_per_day", "daily_loss_stop_usd", "daily_profit_stop_usd",
)


@dataclass
class StrategySpec:
    name: str
    family: str
    instrument: str
    timeframe: int
    long_entry: str
    short_entry: str = ""
    exit_when: str = ""
    stop_type: str = "atr"
    stop_value: float = 1.5
    target_type: str = "r"
    target_value: float = 2.0
    breakeven_at_r: float = 0.0
    max_hold_bars: int = 48
    session_start: int = 575
    session_end: int = 900
    flat_at: int = 955
    risk_usd: float = 300.0
    max_contracts: int = 50
    max_trades_per_day: int = 2
    daily_loss_stop_usd: float = 0.0
    daily_profit_stop_usd: float = 0.0
    rationale: str = ""
    author: str = ""
    parents: str = ""
    origin: str = ""

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_dict(cls, d: dict, instruments: list[str] | None = None, timeframes: list[int] | None = None) -> "StrategySpec":
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in d.items() if k in known}
        spec = cls(**{**{"name": "unnamed", "family": "other", "instrument": (instruments or ["MNQ"])[0], "timeframe": (timeframes or [5])[0], "long_entry": ""}, **data})
        return spec.normalized(instruments, timeframes)

    def normalized(self, instruments: list[str] | None = None, timeframes: list[int] | None = None) -> "StrategySpec":
        s = StrategySpec(**asdict(self))
        s.name = str(s.name or "unnamed")[:80]
        s.family = s.family if s.family in FAMILIES else "other"
        s.instrument = str(s.instrument).upper()
        if instruments and s.instrument not in instruments:
            s.instrument = instruments[0]
        s.timeframe = int(round(float(s.timeframe)))
        if timeframes and s.timeframe not in timeframes:
            s.timeframe = min(timeframes, key=lambda t: abs(t - s.timeframe))
        s.long_entry = (s.long_entry or "").strip()
        s.short_entry = (s.short_entry or "").strip()
        s.exit_when = (s.exit_when or "").strip()
        s.stop_type = s.stop_type if s.stop_type in STOP_TYPES else "atr"
        s.target_type = s.target_type if s.target_type in TARGET_TYPES else "r"
        lim = {"atr": (0.1, 20.0), "pct": (0.02, 5.0), "points": (0.01, 10000.0), "swing": (1, 100)}[s.stop_type]
        s.stop_value = float(np.clip(float(s.stop_value), *lim))
        if s.stop_type == "swing":
            s.stop_value = float(int(round(s.stop_value)))
        tlim = {"r": (0.2, 20.0), "atr": (0.1, 40.0), "pct": (0.02, 10.0), "points": (0.01, 20000.0), "none": (0.0, 0.0)}[s.target_type]
        s.target_value = float(np.clip(float(s.target_value), *tlim))
        s.breakeven_at_r = float(np.clip(float(s.breakeven_at_r), 0.0, 10.0))
        s.max_hold_bars = int(np.clip(int(round(float(s.max_hold_bars))), 1, 400))
        s.session_start = int(np.clip(int(s.session_start), 0, 1439))
        s.session_end = int(np.clip(int(s.session_end), s.session_start + 1, 1440))
        s.flat_at = int(np.clip(int(s.flat_at), s.session_start + 1, 1440))
        s.risk_usd = float(np.clip(float(s.risk_usd), 10.0, 100_000.0))
        s.max_contracts = int(np.clip(int(s.max_contracts), 1, 1000))
        s.max_trades_per_day = int(np.clip(int(s.max_trades_per_day), 1, 50))
        s.daily_loss_stop_usd = float(max(0.0, float(s.daily_loss_stop_usd)))
        s.daily_profit_stop_usd = float(max(0.0, float(s.daily_profit_stop_usd)))
        s.rationale = str(s.rationale or "")[:1200]
        s.stop_value = round(s.stop_value, 3)
        s.target_value = round(s.target_value, 3)
        s.breakeven_at_r = round(s.breakeven_at_r, 2)
        s.risk_usd = round(s.risk_usd, 0)
        s.daily_loss_stop_usd = round(s.daily_loss_stop_usd, 0)
        s.daily_profit_stop_usd = round(s.daily_profit_stop_usd, 0)
        return s

    def errors(self) -> list[str]:
        errs = []
        if not self.long_entry and not self.short_entry:
            errs.append("strategy needs long_entry and/or short_entry")
        for label, expr in (("long_entry", self.long_entry), ("short_entry", self.short_entry), ("exit_when", self.exit_when)):
            if expr:
                msg = lang.validate(expr)
                if msg:
                    errs.append(f"{label}: {msg}")
        return errs

    @property
    def uid(self) -> str:
        payload = json.dumps({k: getattr(self, k) for k in TRADING_FIELDS}, sort_keys=True, default=str)
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["uid"] = self.uid
        return d

    @property
    def sides(self) -> str:
        if self.long_entry and self.short_entry:
            return "both"
        return "long" if self.long_entry else "short"


# ---------------------------------------------------------------------- genetic operators

FILTER_PAIRS = [
    ("close > vwap", "close < vwap"),
    ("close > ema(close, {p})", "close < ema(close, {p})"),
    ("ema(close, {a}) > ema(close, {b})", "ema(close, {a}) < ema(close, {b})"),
    ("adx(14) > {adx}", "adx(14) > {adx}"),
    ("adx(14) < {adx}", "adx(14) < {adx}"),
    ("tod < {t}", "tod < {t}"),
    ("tod >= {t}", "tod >= {t}"),
    ("atr(14) > sma(atr(14), {p})", "atr(14) > sma(atr(14), {p})"),
    ("atr(14) < sma(atr(14), {p})", "atr(14) < sma(atr(14), {p})"),
    ("close > day_open", "close < day_open"),
    ("close > prev_close", "close < prev_close"),
    ("rsi(close, 14) > 50", "rsi(close, 14) < 50"),
    ("gap > 0", "gap < 0"),
    ("dow != 0", "dow != 0"),
    ("dow != 4", "dow != 4"),
    ("volume > {k} * sma(volume, 20)", "volume > {k} * sma(volume, 20)"),
    ("close > open", "close < open"),
    ("close > prev_high", "close < prev_low"),
    ("macd_hist(close, 12, 26, 9) > 0", "macd_hist(close, 12, 26, 9) < 0"),
    ("close > sma(close, {p})", "close < sma(close, {p})"),
]


def fill_template(tpl: str, rng: np.random.Generator) -> str:
    a = int(rng.choice([5, 8, 9, 10, 12, 13, 20]))
    params = {
        "p": int(rng.choice([10, 20, 30, 50, 100, 200])),
        "a": a,
        "b": int(a * rng.choice([2, 2.5, 3, 4])),
        "adx": int(rng.choice([15, 18, 20, 25, 30])),
        "t": int(rng.choice([600, 630, 660, 690, 720, 780, 840, 900])),
        "k": round(float(rng.choice([1.2, 1.5, 2.0, 2.5])), 2),
    }
    return tpl.format(**params)


def random_filter_pair(rng: np.random.Generator) -> tuple[str, str]:
    long_t, short_t = FILTER_PAIRS[int(rng.integers(len(FILTER_PAIRS)))]
    seed = int(rng.integers(1 << 30))
    return fill_template(long_t, np.random.default_rng(seed)), fill_template(short_t, np.random.default_rng(seed))


def _perturb(value: float, kind: str, rng: np.random.Generator) -> float:
    if kind == lang.P:
        return float(int(np.clip(round(value * math.exp(rng.normal(0, 0.3))) + rng.choice([-1, 0, 1]), 1, 500)))
    if kind == lang.M:
        return float(rng.choice([5, 10, 15, 20, 30, 45, 60, 90]))
    if kind == lang.T:
        return float(int(np.clip(value + 5 * int(rng.integers(-6, 7)), 0, 1440)))
    if kind == lang.F:
        return round(float(np.clip(value * math.exp(rng.normal(0, 0.25)), 0.01, 1000)), 3)
    if kind == "I":
        return float(max(0, int(value) + int(rng.choice([-1, 1]))))
    if abs(value) < 1e-9:
        return round(float(rng.normal(0, 0.002)), 5)
    if abs(value) < 0.05:
        return round(float(value * math.exp(rng.normal(0, 0.35))), 5)
    new = value * math.exp(rng.normal(0, 0.15))
    return round(new, 2) if abs(new) < 100 else float(round(new))


def _constants_in_tree(tree: ast.AST) -> list[tuple[ast.Constant, str]]:
    """Numeric constants with their role: P/M/T/F (function arguments), T (compared with tod) or C (comparison)."""
    out: list[tuple[ast.Constant, str]] = []
    claimed: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in lang.FUNCTIONS:
            kinds = lang.FUNCTIONS[node.func.id][0]
            for arg, kind in zip(node.args, kinds):
                if kind != lang.S and isinstance(arg, ast.Constant):
                    out.append((arg, kind))
                    claimed.add(id(arg))
        elif isinstance(node, ast.Compare):
            names = {n.id for n in [node.left, *node.comparators] if isinstance(n, ast.Name)}
            kind = lang.T if "tod" in names else ("I" if names & {"bar", "bars_left", "dow", "hour", "minute"} else "C")
            for n in [node.left, *node.comparators]:
                if isinstance(n, ast.Constant) and not isinstance(n.value, bool) and id(n) not in claimed:
                    out.append((n, kind))
                    claimed.add(id(n))
    return out


def _split_and(expr: str) -> list[str]:
    tree = ast.parse(expr, mode="eval").body
    if isinstance(tree, ast.BoolOp) and isinstance(tree.op, ast.And):
        return [ast.unparse(v) for v in tree.values]
    return [ast.unparse(tree)]


def _join_and(parts: list[str]) -> str:
    unique = list(dict.fromkeys(p.strip() for p in parts if p.strip()))
    return " and ".join(f"({p})" if " or " in p else p for p in unique)


def mutate(spec: StrategySpec, rng: np.random.Generator, instruments: list[str], timeframes: list[int]) -> StrategySpec:
    for _ in range(8):
        child = _mutate_once(StrategySpec(**asdict(spec)), rng, instruments, timeframes)
        child = child.normalized(instruments, timeframes)
        if not child.errors() and child.uid != spec.uid:
            child.parents = spec.uid
            child.origin = "mutation"
            child.author = "evolver"
            child.rationale = ""
            child.name = _child_name(spec.name)
            return child
    return spec


def root_name(name: str) -> str:
    return name.split(" ·")[0].split(" ⨯ ")[0].strip()[:44]


def _child_name(name: str) -> str:
    return f"{root_name(name)} ·{hashlib.sha1(str(np.random.random()).encode()).hexdigest()[:3]}"


def _mutate_once(s: StrategySpec, rng: np.random.Generator, instruments: list[str], timeframes: list[int]) -> StrategySpec:
    op = rng.choice(
        ["const", "exit", "session", "risk", "filter_add", "filter_remove", "direction", "timeframe", "instrument"],
        p=[0.34, 0.2, 0.08, 0.14, 0.1, 0.06, 0.03, 0.03, 0.02],
    )
    if op == "const":
        targets = [k for k in ("long_entry", "short_entry", "exit_when") if getattr(s, k)]
        key = targets[int(rng.integers(len(targets)))]
        old = getattr(s, key)
        consts = _constants_in_tree(ast.parse(old, mode="eval"))
        if consts:
            idx = int(rng.integers(len(consts)))
            node, kind = consts[idx]
            new_val = _perturb(float(node.value), kind, rng)
            if kind in (lang.P, lang.M, lang.T, "I"):
                new_val = int(new_val)
            # apply to the chosen expression, and mirror onto the paired side when it has the same constant
            setattr(s, key, _replace_const(old, idx, new_val))
            pair = {"long_entry": "short_entry", "short_entry": "long_entry"}.get(key)
            if pair and getattr(s, pair) and kind in (lang.P, lang.M, lang.T):
                other = getattr(s, pair)
                oc = _constants_in_tree(ast.parse(other, mode="eval"))
                for j, (n2, k2) in enumerate(oc):
                    if k2 == kind and n2.value == node.value:
                        setattr(s, pair, _replace_const(other, j, new_val))
                        break
    elif op == "exit":
        sub = rng.choice(["stop_v", "stop_t", "tgt_v", "tgt_t", "be", "hold", "exit_rule"])
        if sub == "stop_v":
            s.stop_value = s.stop_value * math.exp(rng.normal(0, 0.25)) if s.stop_type != "swing" else s.stop_value + int(rng.integers(-3, 4))
        elif sub == "stop_t":
            s.stop_type = str(rng.choice(STOP_TYPES))
            s.stop_value = {"atr": 1.5, "pct": 0.25, "points": 20.0, "swing": 5}[s.stop_type] * math.exp(rng.normal(0, 0.3))
        elif sub == "tgt_v":
            s.target_value = max(0.2, s.target_value * math.exp(rng.normal(0, 0.25))) if s.target_type != "none" else 2.0
            if s.target_type == "none":
                s.target_type = "r"
        elif sub == "tgt_t":
            s.target_type = str(rng.choice(TARGET_TYPES))
            s.target_value = {"r": 2.0, "atr": 3.0, "pct": 0.5, "points": 40.0, "none": 0.0}[s.target_type] * math.exp(rng.normal(0, 0.3))
        elif sub == "be":
            s.breakeven_at_r = 0.0 if s.breakeven_at_r > 0 and rng.random() < 0.4 else float(rng.choice([0.5, 0.75, 1.0, 1.5]))
        elif sub == "hold":
            s.max_hold_bars = int(max(1, s.max_hold_bars * math.exp(rng.normal(0, 0.4))))
        else:
            if s.exit_when and rng.random() < 0.4:
                s.exit_when = ""
            else:
                s.exit_when = str(rng.choice([
                    "crossunder(close, vwap)" if s.sides != "short" else "crossover(close, vwap)",
                    "crossunder(close, ema(close, 9))" if s.sides != "short" else "crossover(close, ema(close, 9))",
                    "rsi(close, 2) > 90" if s.sides != "short" else "rsi(close, 2) < 10",
                    "bars_left < 3",
                ]))
    elif op == "session":
        which = rng.choice(["start", "end", "flat"])
        delta = 15 * int(rng.integers(-4, 5))
        if which == "start":
            s.session_start += delta
        elif which == "end":
            s.session_end += delta
        else:
            s.flat_at = int(np.clip(s.flat_at + delta, 600, 959))
    elif op == "risk":
        sub = rng.choice(["risk", "trades", "dls", "dps"])
        if sub == "risk":
            s.risk_usd = s.risk_usd * math.exp(rng.normal(0, 0.3))
        elif sub == "trades":
            s.max_trades_per_day = max(1, s.max_trades_per_day + int(rng.choice([-1, 1])))
        elif sub == "dls":
            s.daily_loss_stop_usd = 0.0 if s.daily_loss_stop_usd and rng.random() < 0.3 else float(rng.choice([1, 1.5, 2, 3])) * s.risk_usd
        else:
            s.daily_profit_stop_usd = 0.0 if s.daily_profit_stop_usd and rng.random() < 0.3 else float(rng.choice([1, 2, 3, 4])) * s.risk_usd
    elif op == "filter_add":
        fl, fs = random_filter_pair(rng)
        if s.long_entry:
            s.long_entry = _join_and(_split_and(s.long_entry) + [fl])
        if s.short_entry:
            s.short_entry = _join_and(_split_and(s.short_entry) + [fs])
    elif op == "filter_remove":
        for key in ("long_entry", "short_entry"):
            expr = getattr(s, key)
            if expr:
                parts = _split_and(expr)
                if len(parts) > 1:
                    parts.pop(int(rng.integers(1, len(parts))))
                    setattr(s, key, _join_and(parts))
    elif op == "direction":
        if s.long_entry and s.short_entry:
            if rng.random() < 0.5:
                s.short_entry = ""
            else:
                s.long_entry = ""
    elif op == "timeframe" and len(timeframes) > 1:
        s.timeframe = int(rng.choice([t for t in timeframes if t != s.timeframe]))
    elif op == "instrument" and len(instruments) > 1:
        s.instrument = str(rng.choice([i for i in instruments if i != s.instrument]))
    return s


def _replace_const(expr: str, idx: int, value) -> str:
    tree = ast.parse(expr, mode="eval")
    nodes = _constants_in_tree(tree)
    nodes[idx][0].value = value
    return ast.unparse(tree)


def crossover(a: StrategySpec, b: StrategySpec, rng: np.random.Generator, instruments: list[str], timeframes: list[int]) -> StrategySpec:
    child = StrategySpec(**asdict(a))
    for f in ("stop_type", "stop_value", "target_type", "target_value", "breakeven_at_r", "max_hold_bars", "exit_when"):
        if rng.random() < 0.5:
            setattr(child, f, getattr(b, f))
    for f in ("session_start", "session_end", "flat_at", "risk_usd", "max_trades_per_day", "daily_loss_stop_usd", "daily_profit_stop_usd"):
        if rng.random() < 0.5:
            setattr(child, f, getattr(b, f))
    if rng.random() < 0.25 and b.long_entry and a.long_entry:
        # graft one of b's long clauses onto a's entries
        extra = _split_and(b.long_entry)
        clause = extra[int(rng.integers(len(extra)))]
        child.long_entry = _join_and(_split_and(a.long_entry) + [clause])
    child = child.normalized(instruments, timeframes)
    if child.errors():
        return a
    child.parents = f"{a.uid}+{b.uid}"
    child.origin = "crossover"
    child.author = "evolver"
    child.rationale = ""
    child.name = f"{root_name(a.name)[:30]} ⨯ {root_name(b.name)[:24]}"
    return child
