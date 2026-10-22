import numpy as np
import pandas as pd
import pytest

from hive import backtest, evaluator, lang, propfirm, seeds
from hive.data import SyntheticProvider, build_dataset, filter_rth
from hive.instruments import get_instrument
from hive.strategy import StrategySpec, crossover, mutate

INST = get_instrument("MNQ")


@pytest.fixture(scope="module")
def ds():
    raw = filter_rth(SyntheticProvider(seed=3).fetch("QQQ", 5, 120))
    return build_dataset(raw, INST, 5, "synthetic", INST.proxy_ratio)


# ------------------------------------------------------------------ expression language

@pytest.mark.parametrize("expr", [
    "__import__('os').system('ls')",
    "close.real > 1",
    "open('x')",
    "foo > 1",
    "sma(close) > 1",
    "sma(close, 0) > 1",
    "sma(close, 20.5) > 1",
    "close ** 2 > 1",
    "[c for c in close]",
    "lambda: 1",
    "'abc' == close",
])
def test_rejects_unsafe_or_invalid(expr):
    assert lang.validate(expr) is not None


def test_accepts_reference_examples():
    for expr in [
        "close > or_high(15) and close > vwap and tod < 690",
        "sweep_pdl() and within(fvg_bull(), 3) and tod >= 600 and tod < 720",
        "rsi(close, 2) < 10 and close > ema(close, 200) and adx(14) < 25",
        "crossover(ema(close, 9), ema(close, 21)) and close > prev_high",
        "not inside_bar() and (gap > 0.002 or gap < -0.002)",
    ]:
        assert lang.validate(expr) is None, expr


NO_LOOKAHEAD_EXPRS = [
    "ema(close, 20)", "sma(close, 10)", "rsi(close, 14)", "atr(14)", "adx(14)", "vwap", "day_high", "day_low",
    "prev_high", "prev_low", "pwh", "or_high(30)", "or_low(15)", "open_ret(30)", "session_high(570, 630)",
    "sweep_low(10)", "sweep_pdh()", "fvg_bull()", "in_bull_fvg(6)", "in_bear_fvg(6)", "bos_up(20)",
    "displacement(1.5)", "bars_since(fvg_bear())", "within(sweep_high(8), 4)", "macd_hist(close, 12, 26, 9)",
    "bb_lower(close, 20, 2)", "stoch(14)", "nr(7)", "engulf_bull()", "bar", "gap",
]


@pytest.mark.parametrize("expr", NO_LOOKAHEAD_EXPRS)
def test_no_lookahead(ds, expr):
    """A value at bar i must not change when future bars are removed."""
    cut_day = ds.n_days - 15
    cut = int(ds.day_start[cut_day])
    raw = pd.DataFrame(
        {"open": ds.o, "high": ds.h, "low": ds.l, "close": ds.c, "volume": ds.v},
        index=pd.to_datetime(ds.ts, unit="s", utc=True),
    )
    short = build_dataset(raw.iloc[:cut], INST, 5, "synthetic", 1.0)
    full = lang.Engine(ds).series(expr)[:cut]
    part = lang.Engine(short).series(expr)
    both_nan = np.isnan(full) & np.isnan(part)
    assert np.allclose(np.where(both_nan, 0, full), np.where(both_nan, 0, part), equal_nan=False), expr


# ------------------------------------------------------------------ backtester

def _toy_dataset(prices):
    idx = pd.date_range("2025-03-03 14:30", periods=len(prices), freq="5min", tz="UTC")  # 09:30 NY
    o = np.array([p[0] for p in prices], float)
    h = np.array([p[1] for p in prices], float)
    l = np.array([p[2] for p in prices], float)
    c = np.array([p[3] for p in prices], float)
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1000.0}, index=idx)
    return build_dataset(df, INST, 5, "toy", 1.0)


def test_backtest_target_and_costs():
    bars = [(100, 101, 99, 100)] * 3 + [(100, 100.5, 99.5, 100)] + [(100, 104, 99.8, 103)] + [(103, 103, 102, 102)] * 3
    ds = _toy_dataset(bars)
    sig = np.zeros(ds.n, bool)
    sig[3] = True  # enter at bar 4 open (100 + 1 tick)
    spec = StrategySpec(name="t", family="other", instrument="MNQ", timeframe=5, long_entry="close > 0",
                        stop_type="points", stop_value=2.0, target_type="r", target_value=1.0, risk_usd=100,
                        max_contracts=50, session_start=570, session_end=960, flat_at=960)
    tr = backtest.run(ds, INST, spec, sig, None, None, np.full(ds.n, 1.0))
    assert len(tr) == 1
    entry = 100 + INST.tick_size
    qty = int(100 // (2.0 * INST.point_value + INST.commission_rt))
    assert tr.qty[0] == qty
    assert tr.reason[0] == 1  # target
    assert tr.exit[0] == pytest.approx(entry + 2.0)
    assert tr.pnl[0] == pytest.approx(2.0 * INST.point_value * qty - INST.commission_rt * qty)


def test_backtest_stop_first_when_both_touched():
    bars = [(100, 100.5, 99.5, 100)] * 4 + [(100, 105, 95, 100)] + [(100, 100.5, 99.5, 100)] * 3
    ds = _toy_dataset(bars)
    sig = np.zeros(ds.n, bool)
    sig[3] = True
    spec = StrategySpec(name="t", family="other", instrument="MNQ", timeframe=5, long_entry="close > 0",
                        stop_type="points", stop_value=2.0, target_type="r", target_value=1.0, risk_usd=100,
                        session_start=570, session_end=960, flat_at=960)
    tr = backtest.run(ds, INST, spec, sig, None, None, np.full(ds.n, 1.0))
    assert tr.reason[0] == 0 and tr.pnl[0] < 0


# ------------------------------------------------------------------ prop-firm rules

def _days(pnl, low=None, high=None):
    pnl = np.asarray(pnl, float)
    return {"pnl": pnl, "low": np.minimum(pnl, 0) if low is None else np.asarray(low, float),
            "high": np.maximum(pnl, 0) if high is None else np.asarray(high, float), "count": np.ones(len(pnl), int)}


def _run(rules, d):
    shape = (1, len(d["pnl"]))
    return propfirm.simulate(rules, d["pnl"].reshape(shape), d["low"].reshape(shape), d["high"].reshape(shape),
                             (d["count"] > 0).reshape(shape))


def test_steady_winner_passes_on_target_day():
    st, days, _ = _run(propfirm.get_rules("topstep_50k"), _days([500] * 10))
    assert st[0] == propfirm.PASSED and days[0] == 6


def test_consistency_rule_delays_pass():
    # one +3000 day is > 50% of the profit, so it needs more profit before passing
    st, days, _ = _run(propfirm.get_rules("topstep_50k"), _days([3000, 100, 100, 3000]))
    assert st[0] == propfirm.PASSED and days[0] == 4


def test_eod_trailing_floor():
    rules = propfirm.get_rules("topstep_50k")  # $2k EOD trailing max loss
    # +1500 lifts the floor to 49,500; sliding back to 49,400 breaches it (a static 48,000 floor would survive)
    st, _, bal = _run(rules, _days([1500, -900, -900, -300]))
    assert st[0] == propfirm.FAIL_MAXLOSS and bal[0] == pytest.approx(49_700)
    static = propfirm.PropRules(**{**rules.to_dict(), "trailing": "static"})
    st2, _, _ = _run(static, _days([1500, -900, -900, -300]))
    assert st2[0] == propfirm.ACTIVE


def test_daily_loss_limit_halts_not_fails_topstep():
    rules = propfirm.get_rules("topstep_50k")
    st, _, bal = _run(rules, _days([-1500, 0], low=[-1500, 0]))
    assert st[0] == propfirm.ACTIVE and bal[0] == pytest.approx(49_000)


def test_ftmo_daily_loss_fails():
    rules = propfirm.get_rules("ftmo_100k")
    st, _, _ = _run(rules, _days([-5200], low=[-5200]))
    assert st[0] == propfirm.FAIL_DLL


# ------------------------------------------------------------------ strategies + evaluation

def test_mutations_and_crossovers_stay_valid():
    rng = np.random.default_rng(0)
    base = seeds.seed_strategies(["MNQ", "MES", "MGC"], [5, 15])
    assert len(base) == len(seeds.SEEDS) * 3
    for i in range(300):
        child = mutate(base[i % len(base)], rng, ["MNQ", "MES", "MGC"], [5, 15])
        assert not child.errors(), child
        x = crossover(base[i % len(base)], base[(i * 7) % len(base)], rng, ["MNQ", "MES", "MGC"], [5, 15])
        assert not x.errors(), x
    for _ in range(200):
        assert not seeds.random_strategy(rng, ["MNQ", "MES"], [5]).errors()


def test_evaluate_all_seeds(ds):
    evaluator.init_local({ds.key: ds}, propfirm.get_rules("topstep_50k"), {}, mc_sims=100)
    for spec in seeds.seed_strategies(["MNQ"], [5]):
        res = evaluator.evaluate(spec.to_dict())
        assert res["ok"], res
        assert 0.0 <= res["fitness"] <= 1.0
        assert len(res["folds"]) == 3
        for seg in ("train", "val", "test"):
            assert 0.0 <= res[seg]["pass_mc"] <= 1.0


def test_evaluate_reports_bad_expression(ds):
    evaluator.init_local({ds.key: ds}, propfirm.get_rules("topstep_50k"), {}, mc_sims=50)
    res = evaluator.evaluate({"name": "x", "family": "other", "instrument": "MNQ", "timeframe": 5, "long_entry": "close > supertrend(3)"})
    assert not res["ok"] and "supertrend" in res["error"]


# ------------------------------------------------------------------ robustness / luck / speed

def test_neighbors_are_valid_small_variants():
    from hive.strategy import neighbors
    rng = np.random.default_rng(1)
    for base in seeds.seed_strategies(["MNQ"], [5]):
        nb = neighbors(base, rng, 6)
        assert nb, base.name
        for s in nb:
            assert not s.errors()
            assert s.uid != base.uid
            assert (s.instrument, s.timeframe, s.family) == (base.instrument, base.timeframe, base.family)
            assert s.origin == "neighbor"


def test_luck_baseline_separates_edge_from_volatility():
    rng = np.random.default_rng(0)
    rules = propfirm.get_rules("topstep_50k")
    noise = rng.normal(0, 600, 400)
    zero_edge = _days(noise)
    edge = _days(noise + 250)
    assert propfirm.luck_baseline(rules, edge, np.random.default_rng(1)) < propfirm.pass_probability_mc(rules, edge, np.random.default_rng(1))["pass"] - 0.2
    luck0 = propfirm.luck_baseline(rules, zero_edge, np.random.default_rng(2))
    pass0 = propfirm.pass_probability_mc(rules, zero_edge, np.random.default_rng(2))["pass"]
    assert abs(luck0 - pass0) < 0.15


def test_target_days_from_env(monkeypatch):
    monkeypatch.setenv("PROP_TARGET_DAYS", "7")
    rules = propfirm.get_rules("topstep_50k")
    assert rules.target_days == 7 and rules.fast_days == 7
    monkeypatch.setenv("PROP_TARGET_DAYS", "")
    assert propfirm.get_rules("topstep_50k").target_days is None


def test_evaluate_reports_luck_speed_and_complexity(ds):
    evaluator.init_local({ds.key: ds}, propfirm.get_rules("topstep_50k"), {}, mc_sims=100)
    res = evaluator.evaluate(seeds.seed_strategies(["MNQ"], [5])[0].to_dict())
    assert res["ok"]
    for seg in ("train", "val", "test"):
        assert 0.0 <= res[seg]["luck"] <= 1.0
        assert 0.0 <= res[seg]["pass_fast"] <= res[seg]["pass_mc"] + 1e-9
    assert res["conditions"] >= 1 and res["fast_days"] == 10
