"""The strategy expression language.

Entry/exit rules are written as Python-style boolean expressions over bar data,
e.g. ``close > or_high(15) and close > vwap and tod < 660``.  Expressions are
parsed with ``ast`` and checked against a whitelist (no attribute access, no
arbitrary calls), then evaluated vectorised with numpy.  Every function only
looks at the current and past bars, so a signal on bar i is known at bar i's
close and the backtester fills it at bar i+1's open: no lookahead.

Sub-expressions are memoised per dataset, so thousands of strategies sharing
``ema(close, 50)`` compute it once per worker.
"""
from __future__ import annotations

import ast
from collections import OrderedDict

import numpy as np
import pandas as pd

from .data import Dataset


class ExprError(ValueError):
    pass


# argument kinds: S = series expression, P = lookback period, M = minutes, T = time of day, F = float constant
S, P, M, T, F = "S", "P", "M", "T", "F"
_BOUNDS = {P: (1, 500), M: (1, 390), T: (0, 1440), F: (0.0, 1000.0)}

FUNCTIONS: dict[str, tuple[tuple[str, ...], str]] = {
    # trend / smoothing
    "sma": ((S, P), "simple moving average of x over n bars"),
    "ema": ((S, P), "exponential moving average"),
    "highest": ((S, P), "highest value of x over the last n bars (including current)"),
    "lowest": ((S, P), "lowest value of x over the last n bars"),
    "shift": ((S, P), "value of x n bars ago"),
    "change": ((S, P), "x - x[n bars ago]"),
    "roc": ((S, P), "percent change of x over n bars"),
    "rising": ((S, P), "x > x[n bars ago]"),
    "falling": ((S, P), "x < x[n bars ago]"),
    "crossover": ((S, S), "a crosses above b on this bar"),
    "crossunder": ((S, S), "a crosses below b on this bar"),
    # oscillators / volatility
    "rsi": ((S, P), "Wilder RSI 0-100"),
    "stoch": ((P,), "stochastic %K 0-100 over n bars"),
    "macd": ((S, P, P), "ema(fast) - ema(slow)"),
    "macd_signal": ((S, P, P, P), "signal line of macd(x, fast, slow), smoothing n"),
    "macd_hist": ((S, P, P, P), "macd - signal"),
    "adx": ((P,), "average directional index 0-100"),
    "plus_di": ((P,), "+DI"),
    "minus_di": ((P,), "-DI"),
    "atr": ((P,), "average true range in points"),
    "stdev": ((S, P), "rolling standard deviation"),
    "zscore": ((S, P), "(x - sma(x,n)) / stdev(x,n)"),
    "bb_upper": ((S, P, F), "Bollinger upper band: sma + k*stdev"),
    "bb_lower": ((S, P, F), "Bollinger lower band: sma - k*stdev"),
    # helpers
    "abs": ((S,), "absolute value"),
    "min": ((S, S), "elementwise minimum"),
    "max": ((S, S), "elementwise maximum"),
    "between": ((S, S, S), "a <= x <= b"),
    "within": ((S, P), "condition was true on any of the last n bars"),
    "bars_since": ((S,), "bars since condition was last true (large if never)"),
    # session structure
    "or_high": ((M,), "opening-range high of the first m minutes (NaN until the range is complete)"),
    "or_low": ((M,), "opening-range low of the first m minutes"),
    "open_ret": ((M,), "return from previous close to the close of the first m minutes (NaN before)"),
    "session_high": ((T, T), "high of today's bars with start<=tod<end (NaN until end)"),
    "session_low": ((T, T), "low of today's bars with start<=tod<end (NaN until end)"),
    # liquidity / ICT-SMC primitives
    "sweep_low": ((P,), "wick below the prior n-bar low but close back above it (sell-side liquidity sweep)"),
    "sweep_high": ((P,), "wick above the prior n-bar high but close back below it (buy-side sweep)"),
    "sweep_pdl": ((), "trades below previous day low and closes back above it"),
    "sweep_pdh": ((), "trades above previous day high and closes back below it"),
    "sweep_or_low": ((M,), "sweep of the m-minute opening-range low"),
    "sweep_or_high": ((M,), "sweep of the m-minute opening-range high"),
    "fvg_bull": ((), "bullish fair value gap just formed: low > high two bars ago"),
    "fvg_bear": ((), "bearish fair value gap just formed: high < low two bars ago"),
    "in_bull_fvg": ((P,), "price retraces into a bullish FVG formed within the last n bars (and still valid)"),
    "in_bear_fvg": ((P,), "price retraces into a bearish FVG formed within the last n bars"),
    "displacement": ((F,), "candle body > k * atr(14)"),
    "bos_up": ((P,), "close breaks above the highest high of the previous n bars"),
    "bos_down": ((P,), "close breaks below the lowest low of the previous n bars"),
    # candles
    "inside_bar": ((), "range inside the previous bar"),
    "engulf_bull": ((), "bullish engulfing candle"),
    "engulf_bear": ((), "bearish engulfing candle"),
    "pin_bull": ((F,), "lower wick >= k * body (hammer)"),
    "pin_bear": ((F,), "upper wick >= k * body (shooting star)"),
    "nr": ((P,), "narrowest range of the last n bars (NR4/NR7)"),
}

VARIABLES: dict[str, str] = {
    "open": "bar open (futures points)",
    "high": "bar high",
    "low": "bar low",
    "close": "bar close",
    "volume": "bar volume (proxy shares)",
    "hl2": "(high+low)/2",
    "hlc3": "(high+low+close)/3",
    "range": "high - low",
    "body": "abs(close - open)",
    "upper_wick": "high - max(open, close)",
    "lower_wick": "min(open, close) - low",
    "vwap": "session VWAP (resets daily)",
    "tod": "bar start time, minutes after midnight New York (09:30=570, 10:00=600, 12:00=720, 15:00=900, 15:55=955)",
    "hour": "bar start hour, New York",
    "minute": "bar start minute",
    "dow": "day of week, 0=Mon .. 4=Fri",
    "bar": "bar number within today's session (0 = first bar)",
    "bars_left": "bars remaining in today's session after this one",
    "day_open": "today's opening price",
    "day_high": "today's high so far (including this bar)",
    "day_low": "today's low so far",
    "prev_high": "previous session high",
    "prev_low": "previous session low",
    "prev_close": "previous session close",
    "pwh": "previous week high",
    "pwl": "previous week low",
    "gap": "(day_open - prev_close) / prev_close, e.g. 0.004 = +0.4% gap",
}

_MAX_NODES = 90
_ALLOWED_CMP = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
_ALLOWED_BIN = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.BitAnd, ast.BitOr)


def parse(expr: str) -> ast.AST:
    """Parse and validate an expression; raises ExprError with a helpful message."""
    if not isinstance(expr, str) or not expr.strip():
        raise ExprError("empty expression")
    if len(expr) > 700:
        raise ExprError("expression too long (max 700 chars)")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"syntax error: {exc.msg}") from None
    count = 0
    func_names = {id(n.func) for n in ast.walk(tree.body) if isinstance(n, ast.Call)}
    for node in ast.walk(tree.body):
        count += 1
        if count > _MAX_NODES:
            raise ExprError("expression too complex")
        if id(node) in func_names:
            continue  # the callee name itself is checked by the Call branch
        _check_node(node)
    return tree.body


def _check_node(node: ast.AST) -> None:
    if isinstance(node, (ast.BoolOp, ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd, ast.Load)):
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.Not, ast.USub, ast.UAdd)):
            raise ExprError("unsupported unary operator")
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BIN):
            raise ExprError(f"unsupported operator {type(node.op).__name__}")
        return
    if isinstance(node, _ALLOWED_BIN):
        return
    if isinstance(node, ast.Compare):
        if not all(isinstance(op, _ALLOWED_CMP) for op in node.ops):
            raise ExprError("unsupported comparison")
        return
    if isinstance(node, _ALLOWED_CMP):
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or isinstance(node.value, (int, float)):
            return
        raise ExprError("only numeric constants are allowed")
    if isinstance(node, ast.Name):
        if node.id in VARIABLES or node.id in ("True", "False"):
            return
        if node.id in FUNCTIONS:
            raise ExprError(f"'{node.id}' is a function; call it like {node.id}(...)")
        raise ExprError(f"unknown variable '{node.id}'")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            name = getattr(node.func, "id", "?")
            raise ExprError(f"unknown function '{name}'")
        if node.keywords:
            raise ExprError(f"{node.func.id}(): keyword arguments are not supported")
        kinds = FUNCTIONS[node.func.id][0]
        if len(node.args) != len(kinds):
            raise ExprError(f"{node.func.id}() takes {len(kinds)} argument(s), got {len(node.args)}")
        for arg, kind in zip(node.args, kinds):
            if kind == S:
                continue
            value = _const_value(arg)
            if value is None:
                raise ExprError(f"{node.func.id}(): argument must be a numeric constant")
            lo, hi = _BOUNDS[kind]
            if kind in (P, M, T) and float(value) != int(value):
                raise ExprError(f"{node.func.id}(): expected an integer, got {value}")
            if not lo <= value <= hi:
                raise ExprError(f"{node.func.id}(): argument {value} out of range [{lo}, {hi}]")
        return
    raise ExprError(f"unsupported syntax: {type(node).__name__}")


def _const_value(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _const_value(node.operand)
        return -inner if inner is not None else None
    return None


def validate(expr: str) -> str | None:
    """Return None if the expression is valid, else an error message."""
    try:
        parse(expr)
        return None
    except ExprError as exc:
        return str(exc)


# --------------------------------------------------------------------------- numeric helpers


def _shift(x: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    out[:n] = np.nan
    out[n:] = x[:-n] if n else x
    return out


def _wilder(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().to_numpy()


def _rolling(x: np.ndarray, n: int, how: str) -> np.ndarray:
    r = pd.Series(x).rolling(n, min_periods=n)
    return getattr(r, how)().to_numpy() if how != "std" else r.std(ddof=0).to_numpy()


def _as_float(x) -> np.ndarray | float:
    if isinstance(x, np.ndarray) and x.dtype == bool:
        return x.astype(np.float64)
    return x


def _as_bool(x, n: int) -> np.ndarray:
    if isinstance(x, np.ndarray):
        if x.dtype == bool:
            return x
        return np.nan_to_num(x, nan=0.0) != 0
    return np.full(n, bool(x) if np.isfinite(x) else False)


class Engine:
    """Evaluates expressions against one Dataset with an LRU cache of sub-expressions."""

    def __init__(self, ds: Dataset, cache_size: int = 192):
        self.ds = ds
        self.n = ds.n
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.cache_size = cache_size
        self._vars: dict[str, np.ndarray] = {}
        self._idx = np.arange(self.n)

    # ---- public API
    def signal(self, expr: str | None) -> np.ndarray | None:
        if not expr or not expr.strip():
            return None
        node = parse(expr)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            return _as_bool(self._eval(node), self.n)

    def series(self, expr: str) -> np.ndarray:
        node = parse(expr)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            out = _as_float(self._eval(node))
        if not isinstance(out, np.ndarray):
            out = np.full(self.n, float(out))
        return out

    # ---- evaluation
    def _eval(self, node: ast.AST):
        if isinstance(node, ast.Constant):
            return float(node.value)
        key = ast.unparse(node)
        hit = self.cache.get(key)
        if hit is not None:
            self.cache.move_to_end(key)
            return hit
        value = self._compute(node)
        if isinstance(value, np.ndarray):
            self.cache[key] = value
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return value

    def _compute(self, node: ast.AST):
        n = self.n
        if isinstance(node, ast.Name):
            if node.id == "True":
                return np.ones(n, bool)
            if node.id == "False":
                return np.zeros(n, bool)
            return self._var(node.id)
        if isinstance(node, ast.UnaryOp):
            val = self._eval(node.operand)
            if isinstance(node.op, ast.Not):
                return ~_as_bool(val, n)
            if isinstance(node.op, ast.USub):
                return -_as_float(val)
            return val
        if isinstance(node, ast.BoolOp):
            vals = [_as_bool(self._eval(v), n) for v in node.values]
            fn = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
            out = vals[0]
            for v in vals[1:]:
                out = fn(out, v)
            return out
        if isinstance(node, ast.BinOp):
            a, b = self._eval(node.left), self._eval(node.right)
            if isinstance(node.op, ast.BitAnd):
                return _as_bool(a, n) & _as_bool(b, n)
            if isinstance(node.op, ast.BitOr):
                return _as_bool(a, n) | _as_bool(b, n)
            a, b = _as_float(a), _as_float(b)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            return np.divide(a, b) if isinstance(a, np.ndarray) or isinstance(b, np.ndarray) else (a / b if b else np.nan)
        if isinstance(node, ast.Compare):
            left = _as_float(self._eval(node.left))
            result = None
            for op, comp in zip(node.ops, node.comparators):
                right = _as_float(self._eval(comp))
                if isinstance(op, ast.Lt):
                    r = left < right
                elif isinstance(op, ast.LtE):
                    r = left <= right
                elif isinstance(op, ast.Gt):
                    r = left > right
                elif isinstance(op, ast.GtE):
                    r = left >= right
                elif isinstance(op, ast.Eq):
                    r = left == right
                else:
                    r = left != right
                r = _as_bool(np.asarray(r) if not isinstance(r, (bool, np.bool_)) else r, n)
                result = r if result is None else (result & r)
                left = right
            return result
        if isinstance(node, ast.Call):
            return self._call(node.func.id, node.args)
        raise ExprError(f"cannot evaluate {type(node).__name__}")

    def _arg_series(self, arg) -> np.ndarray:
        val = _as_float(self._eval(arg))
        if not isinstance(val, np.ndarray):
            val = np.full(self.n, float(val))
        return val

    def _call(self, name: str, args: list) -> np.ndarray:
        kinds = FUNCTIONS[name][0]
        vals = []
        for arg, kind in zip(args, kinds):
            if kind == S:
                vals.append(self._arg_series(arg))
            elif kind == F:
                vals.append(float(_const_value(arg)))
            else:
                vals.append(int(_const_value(arg)))
        ds = self.ds
        h, l, c, o = ds.h, ds.l, ds.c, ds.o
        if name == "sma":
            return _rolling(vals[0], vals[1], "mean")
        if name == "ema":
            return pd.Series(vals[0]).ewm(span=vals[1], adjust=False, min_periods=vals[1]).mean().to_numpy()
        if name == "highest":
            return _rolling(vals[0], vals[1], "max")
        if name == "lowest":
            return _rolling(vals[0], vals[1], "min")
        if name == "shift":
            return _shift(vals[0], vals[1])
        if name == "change":
            return vals[0] - _shift(vals[0], vals[1])
        if name == "roc":
            prev = _shift(vals[0], vals[1])
            return (vals[0] - prev) / prev * 100.0
        if name == "rising":
            return vals[0] > _shift(vals[0], vals[1])
        if name == "falling":
            return vals[0] < _shift(vals[0], vals[1])
        if name == "crossover":
            a, b = vals
            return (a > b) & (_shift(a, 1) <= _shift(b, 1))
        if name == "crossunder":
            a, b = vals
            return (a < b) & (_shift(a, 1) >= _shift(b, 1))
        if name == "rsi":
            x, p = vals
            d = np.diff(x, prepend=np.nan)
            up = _wilder(np.where(d > 0, d, 0.0), p)
            dn = _wilder(np.where(d < 0, -d, 0.0), p)
            rs = up / dn
            out = 100.0 - 100.0 / (1.0 + rs)
            out[(dn == 0) & (up > 0)] = 100.0
            return out
        if name == "stoch":
            p = vals[0]
            ll, hh = _rolling(l, p, "min"), _rolling(h, p, "max")
            return 100.0 * (c - ll) / (hh - ll)
        if name in ("macd", "macd_signal", "macd_hist"):
            x, fast, slow = vals[0], vals[1], vals[2]
            ef = pd.Series(x).ewm(span=fast, adjust=False, min_periods=fast).mean()
            es = pd.Series(x).ewm(span=slow, adjust=False, min_periods=slow).mean()
            line = (ef - es).to_numpy()
            if name == "macd":
                return line
            sig = pd.Series(line).ewm(span=vals[3], adjust=False, min_periods=vals[3]).mean().to_numpy()
            return sig if name == "macd_signal" else line - sig
        if name in ("adx", "plus_di", "minus_di"):
            p = vals[0]
            up = h - _shift(h, 1)
            dn = _shift(l, 1) - l
            pdm = np.where((up > dn) & (up > 0), up, 0.0)
            mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
            atr = _wilder(self._true_range(), p)
            pdi = 100.0 * _wilder(pdm, p) / atr
            mdi = 100.0 * _wilder(mdm, p) / atr
            if name == "plus_di":
                return pdi
            if name == "minus_di":
                return mdi
            dx = 100.0 * np.abs(pdi - mdi) / (pdi + mdi)
            return _wilder(np.nan_to_num(dx, nan=0.0), p)
        if name == "atr":
            return _wilder(self._true_range(), vals[0])
        if name == "stdev":
            return _rolling(vals[0], vals[1], "std")
        if name == "zscore":
            x, p = vals
            return (x - _rolling(x, p, "mean")) / _rolling(x, p, "std")
        if name in ("bb_upper", "bb_lower"):
            x, p, k = vals
            mid, sd = _rolling(x, p, "mean"), _rolling(x, p, "std")
            return mid + k * sd if name == "bb_upper" else mid - k * sd
        if name == "abs":
            return np.abs(vals[0])
        if name == "min":
            return np.fmin(vals[0], vals[1])
        if name == "max":
            return np.fmax(vals[0], vals[1])
        if name == "between":
            x, a, b = vals
            return (x >= a) & (x <= b)
        if name == "within":
            cond = _as_bool(vals[0], self.n).astype(np.float64)
            return _rolling(cond, vals[1], "max") > 0
        if name == "bars_since":
            cond = _as_bool(vals[0], self.n)
            last = np.where(cond, self._idx, -1)
            last = np.maximum.accumulate(last)
            return np.where(last >= 0, self._idx - last, 1e9).astype(np.float64)
        if name in ("or_high", "or_low", "sweep_or_low", "sweep_or_high"):
            m = vals[0]
            mins = self._var("_mins_from_open")
            in_rng = mins < m
            if name in ("or_high", "sweep_or_high"):
                per_day = np.maximum.reduceat(np.where(in_rng, h, -np.inf), ds.day_start)
            else:
                per_day = np.minimum.reduceat(np.where(in_rng, l, np.inf), ds.day_start)
            level = per_day[ds.day_id].astype(np.float64)
            level[in_rng | ~np.isfinite(level)] = np.nan
            if name == "sweep_or_low":
                return (l < level) & (c > level)
            if name == "sweep_or_high":
                return (h > level) & (c < level)
            return level
        if name == "open_ret":
            m = vals[0]
            mins = self._var("_mins_from_open")
            in_rng = mins < m
            count = np.add.reduceat(in_rng.astype(np.int64), ds.day_start)
            last_idx = ds.day_start + np.maximum(count, 1) - 1
            ref = c[last_idx]
            ref = np.where(count > 0, ref, np.nan)[ds.day_id]
            pc = self._var("prev_close")
            out = (ref - pc) / pc
            out[in_rng] = np.nan
            return out
        if name in ("session_high", "session_low"):
            a, b = vals
            tod = ds.tod
            in_rng = (tod >= a) & (tod < b)
            if name == "session_high":
                per_day = np.maximum.reduceat(np.where(in_rng, h, -np.inf), ds.day_start)
            else:
                per_day = np.minimum.reduceat(np.where(in_rng, l, np.inf), ds.day_start)
            level = per_day[ds.day_id].astype(np.float64)
            level[(tod < b) | ~np.isfinite(level)] = np.nan
            return level
        if name in ("sweep_low", "sweep_high"):
            p = vals[0]
            if name == "sweep_low":
                prior = _shift(_rolling(l, p, "min"), 1)
                return (l < prior) & (c > prior)
            prior = _shift(_rolling(h, p, "max"), 1)
            return (h > prior) & (c < prior)
        if name == "sweep_pdl":
            pl = self._var("prev_low")
            return (l < pl) & (c > pl)
        if name == "sweep_pdh":
            ph = self._var("prev_high")
            return (h > ph) & (c < ph)
        if name in ("fvg_bull", "fvg_bear"):
            same = self._var("_same_day_2")
            if name == "fvg_bull":
                return same & (l > _shift(h, 2))
            return same & (h < _shift(l, 2))
        if name in ("in_bull_fvg", "in_bear_fvg"):
            return self._in_fvg(vals[0], bull=name == "in_bull_fvg")
        if name == "displacement":
            return np.abs(c - o) > vals[0] * _wilder(self._true_range(), 14)
        if name == "bos_up":
            return c > _shift(_rolling(h, vals[0], "max"), 1)
        if name == "bos_down":
            return c < _shift(_rolling(l, vals[0], "min"), 1)
        if name == "inside_bar":
            return (h < _shift(h, 1)) & (l > _shift(l, 1))
        if name == "engulf_bull":
            po, pc = _shift(o, 1), _shift(c, 1)
            return (c > o) & (pc < po) & (c >= po) & (o <= pc)
        if name == "engulf_bear":
            po, pc = _shift(o, 1), _shift(c, 1)
            return (c < o) & (pc > po) & (c <= po) & (o >= pc)
        if name in ("pin_bull", "pin_bear"):
            k = vals[0]
            body = np.maximum(np.abs(c - o), 1e-9 + 0.05 * (h - l))
            if name == "pin_bull":
                return (np.minimum(o, c) - l) >= k * body
            return (h - np.maximum(o, c)) >= k * body
        if name == "nr":
            rng = h - l
            return rng <= _rolling(rng, vals[0], "min")
        raise ExprError(f"function {name} not implemented")

    def _true_range(self) -> np.ndarray:
        tr = self._vars.get("_tr")
        if tr is None:
            ds = self.ds
            pc = _shift(ds.c, 1)
            tr = np.fmax(ds.h - ds.l, np.fmax(np.abs(ds.h - pc), np.abs(ds.l - pc)))
            self._vars["_tr"] = tr
        return tr

    def _in_fvg(self, max_age: int, bull: bool) -> np.ndarray:
        ds = self.ds
        h, l, c, day = ds.h, ds.l, ds.c, ds.day_id
        out = np.zeros(self.n, bool)
        top = bot = 0.0
        born = -1
        for i in range(2, self.n):
            if born >= 0:
                if bull:
                    if c[i] < bot or day[i] != day[born]:
                        born = -1
                    elif i - born <= max_age and l[i] <= top and c[i] >= bot:
                        out[i] = True
                else:
                    if c[i] > top or day[i] != day[born]:
                        born = -1
                    elif i - born <= max_age and h[i] >= bot and c[i] <= top:
                        out[i] = True
            if day[i] == day[i - 2]:
                if bull and l[i] > h[i - 2]:
                    top, bot, born = l[i], h[i - 2], i
                elif not bull and h[i] < l[i - 2]:
                    top, bot, born = l[i - 2], h[i], i
        return out

    def _var(self, name: str) -> np.ndarray:
        cached = self._vars.get(name)
        if cached is not None:
            return cached
        ds = self.ds
        o, h, l, c, v = ds.o, ds.h, ds.l, ds.c, ds.v
        if name == "open":
            val = o
        elif name == "high":
            val = h
        elif name == "low":
            val = l
        elif name == "close":
            val = c
        elif name == "volume":
            val = v
        elif name == "hl2":
            val = (h + l) / 2
        elif name == "hlc3":
            val = (h + l + c) / 3
        elif name == "range":
            val = h - l
        elif name == "body":
            val = np.abs(c - o)
        elif name == "upper_wick":
            val = h - np.maximum(o, c)
        elif name == "lower_wick":
            val = np.minimum(o, c) - l
        elif name == "vwap":
            tp = (h + l + c) / 3
            cv = pd.Series(v).groupby(ds.day_id).cumsum().to_numpy()
            cpv = pd.Series(tp * v).groupby(ds.day_id).cumsum().to_numpy()
            val = np.where(cv > 0, cpv / np.where(cv > 0, cv, 1), tp)
        elif name == "tod":
            val = ds.tod.astype(np.float64)
        elif name == "hour":
            val = (ds.tod // 60).astype(np.float64)
        elif name == "minute":
            val = (ds.tod % 60).astype(np.float64)
        elif name == "dow":
            val = ds.dow.astype(np.float64)
        elif name == "bar":
            val = (self._idx - ds.day_start[ds.day_id]).astype(np.float64)
        elif name == "bars_left":
            val = (ds.day_end[ds.day_id] - self._idx).astype(np.float64)
        elif name == "day_open":
            val = o[ds.day_start][ds.day_id]
        elif name == "day_high":
            val = pd.Series(h).groupby(ds.day_id).cummax().to_numpy()
        elif name == "day_low":
            val = pd.Series(l).groupby(ds.day_id).cummin().to_numpy()
        elif name in ("prev_high", "prev_low", "prev_close"):
            if name == "prev_high":
                per_day = np.maximum.reduceat(h, ds.day_start)
            elif name == "prev_low":
                per_day = np.minimum.reduceat(l, ds.day_start)
            else:
                per_day = c[ds.day_end]
            prev = np.concatenate([[np.nan], per_day[:-1]])
            val = prev[ds.day_id]
        elif name in ("pwh", "pwl"):
            starts = np.flatnonzero(np.diff(ds.week_id, prepend=-1) != 0)
            if name == "pwh":
                per_week = np.maximum.reduceat(h, starts)
            else:
                per_week = np.minimum.reduceat(l, starts)
            prev = np.concatenate([[np.nan], per_week[:-1]])
            val = prev[ds.week_id]
        elif name == "gap":
            pc = self._var("prev_close")
            val = (self._var("day_open") - pc) / pc
        elif name == "_mins_from_open":
            val = (ds.tod - ds.tod[ds.day_start][ds.day_id]).astype(np.float64)
        elif name == "_same_day_2":
            d = ds.day_id
            val = np.zeros(self.n, bool)
            val[2:] = d[2:] == d[:-2]
        else:
            raise ExprError(f"unknown variable '{name}'")
        self._vars[name] = val
        return val


def reference_text() -> str:
    """Human/LLM-readable language reference, generated from the whitelist."""
    lines = ["VARIABLES (one value per bar):"]
    for k, desc in VARIABLES.items():
        lines.append(f"  {k}: {desc}")
    lines.append("FUNCTIONS (S = series expression, P = lookback bars 1-500, M = minutes 1-390, T = time of day 0-1440, F = float constant):")
    for k, (kinds, desc) in FUNCTIONS.items():
        sig = ", ".join(kinds)
        lines.append(f"  {k}({sig}): {desc}")
    lines.append(
        "OPERATORS: and, or, not, + - * /, < <= > >= == !=. Numeric constants only (no strings). "
        "Constant arguments (P/M/T/F) must be literal numbers. Booleans used in arithmetic count as 1/0."
    )
    return "\n".join(lines)
