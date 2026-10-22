"""Market data: providers (Alpaca, Yahoo, synthetic), caching and the Dataset container.

A Dataset holds intraday bars for one instrument/timeframe as flat numpy arrays,
already rescaled to futures points, plus the per-day bookkeeping the expression
engine and backtester need.  Days are split chronologically into
train (60%) / validation (20%) / holdout (20%).
"""
from __future__ import annotations

import logging
import pickle
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .instruments import Instrument

log = logging.getLogger("hive.data")
NY = "America/New_York"
RTH_OPEN, RTH_CLOSE = 570, 960  # 09:30 and 16:00 New York time, in minutes after midnight


@dataclass
class Dataset:
    instrument: str
    timeframe: int
    source: str
    proxy: str
    ratio: float
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    ts: np.ndarray  # unix seconds of bar open
    tod: np.ndarray  # minutes after midnight New York time of bar open
    dow: np.ndarray  # 0 = Monday
    day_id: np.ndarray  # 0..n_days-1
    week_id: np.ndarray
    day_start: np.ndarray  # first bar index of each day
    day_end: np.ndarray  # last bar index of each day (inclusive)
    day_ts: np.ndarray  # unix seconds of each day's first bar
    splits: dict = field(default_factory=dict)  # name -> (first_day, last_day_exclusive)

    @property
    def n(self) -> int:
        return len(self.c)

    @property
    def n_days(self) -> int:
        return len(self.day_start)

    @property
    def key(self) -> str:
        return f"{self.instrument}_{self.timeframe}"

    def summary(self) -> dict:
        return {
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "source": self.source,
            "proxy": self.proxy,
            "bars": self.n,
            "days": self.n_days,
            "start": datetime.fromtimestamp(int(self.ts[0]), timezone.utc).strftime("%Y-%m-%d"),
            "end": datetime.fromtimestamp(int(self.ts[-1]), timezone.utc).strftime("%Y-%m-%d"),
            "splits": {k: list(map(int, v)) for k, v in self.splits.items()},
        }


def build_dataset(df: pd.DataFrame, inst: Instrument, timeframe: int, source: str, ratio: float) -> Dataset:
    """df: tz-aware DatetimeIndex (bar open), columns open/high/low/close/volume, in proxy price units."""
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    idx = df.index.tz_convert(NY)
    tod = (idx.hour * 60 + idx.minute).to_numpy().astype(np.int32)
    dates = idx.normalize()
    day_codes, day_uniques = pd.factorize(dates)
    day_id = day_codes.astype(np.int32)
    iso = idx.isocalendar()
    week_key = (iso["year"].to_numpy().astype(np.int64) * 100 + iso["week"].to_numpy().astype(np.int64))
    week_id = pd.factorize(week_key)[0].astype(np.int32)
    n_days = int(day_id.max()) + 1 if len(day_id) else 0
    day_start = np.searchsorted(day_id, np.arange(n_days), side="left").astype(np.int64)
    day_end = (np.searchsorted(day_id, np.arange(n_days), side="right") - 1).astype(np.int64)
    ts = (df.index.as_unit("ns").asi8 // 1_000_000_000).astype(np.int64)
    scale = float(ratio)
    ds = Dataset(
        instrument=inst.symbol,
        timeframe=int(timeframe),
        source=source,
        proxy=inst.proxy,
        ratio=scale,
        o=df["open"].to_numpy(np.float64) * scale,
        h=df["high"].to_numpy(np.float64) * scale,
        l=df["low"].to_numpy(np.float64) * scale,
        c=df["close"].to_numpy(np.float64) * scale,
        v=df["volume"].to_numpy(np.float64),
        ts=ts,
        tod=tod,
        dow=idx.dayofweek.to_numpy().astype(np.int32),
        day_id=day_id,
        week_id=week_id,
        day_start=day_start,
        day_end=day_end,
        day_ts=ts[day_start] if n_days else np.zeros(0, np.int64),
    )
    a, b = int(n_days * 0.6), int(n_days * 0.8)
    ds.splits = {"train": (0, a), "val": (a, b), "test": (b, n_days)}
    return ds


def resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes <= 1:
        return df
    out = df.resample(f"{minutes}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["open", "close"])


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    local = df.index.tz_convert(NY)
    tod = local.hour * 60 + local.minute
    mask = (tod >= RTH_OPEN) & (tod < RTH_CLOSE) & (local.dayofweek < 5)
    return df[mask]


# --------------------------------------------------------------------------- providers


class AlpacaProvider:
    name = "alpaca"

    def __init__(self, key: str, secret: str, feed: str = "sip"):
        from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient

        self.client = StockHistoricalDataClient(key, secret)
        self.crypto = CryptoHistoricalDataClient(key, secret)
        self.feed = feed

    def fetch(self, symbol: str, timeframe: int, days: int) -> pd.DataFrame:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        end = datetime.now(timezone.utc) - timedelta(minutes=20)
        start = end - timedelta(days=days)
        if "/" in symbol:  # spot crypto pair, e.g. BTC/USD
            req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(timeframe, TimeFrameUnit.Minute), start=start, end=end)
            df = self.crypto.get_crypto_bars(req).df
            if df is None or df.empty:
                raise RuntimeError(f"no crypto bars returned for {symbol}")
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level=0)
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            log.info("alpaca: %s %smin %d crypto bars", symbol, timeframe, len(df))
            return df
        feeds = [self.feed] + (["iex"] if self.feed != "iex" else [])
        last_err: Exception | None = None
        for feed in feeds:
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame(timeframe, TimeFrameUnit.Minute),
                    start=start,
                    end=end,
                    adjustment=Adjustment.ALL,
                    feed=DataFeed(feed),
                )
                bars = self.client.get_stock_bars(req)
                df = bars.df
                if df is None or df.empty:
                    raise RuntimeError(f"no bars returned for {symbol} ({feed})")
                if isinstance(df.index, pd.MultiIndex):
                    df = df.xs(symbol, level=0)
                df = df[["open", "high", "low", "close", "volume"]].astype(float)
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                log.info("alpaca: %s %smin %d bars via %s feed", symbol, timeframe, len(df), feed)
                return df
            except Exception as exc:  # fall back to the free IEX feed
                last_err = exc
                log.warning("alpaca fetch %s via %s failed: %s", symbol, feed, exc)
        raise RuntimeError(f"Alpaca data fetch failed for {symbol}: {last_err}")


class YahooProvider:
    """Real continuous futures (NQ=F, ES=F, GC=F...) via yfinance. Intraday history is limited (~60 days at 5m)."""

    name = "yahoo"

    def fetch(self, symbol: str, timeframe: int, days: int) -> pd.DataFrame:
        import yfinance as yf  # optional dependency

        interval = {1: "1m", 5: "5m", 15: "15m", 30: "30m", 60: "60m"}[timeframe]
        period = "7d" if timeframe == 1 else ("60d" if timeframe < 60 else "730d")
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty:
            raise RuntimeError(f"yfinance returned no data for {symbol}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].astype(float)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df


class SyntheticProvider:
    """Deterministic synthetic intraday data so the hive can run with no keys.

    Strategies found on synthetic data are meaningless for real trading; it exists
    so the whole pipeline (and UI) can be exercised offline.
    """

    name = "synthetic"
    PROFILE = {  # start price (proxy units), daily vol, daily drift
        "QQQ": (430.0, 0.013, 0.0004),
        "SPY": (520.0, 0.010, 0.0003),
        "GLD": (210.0, 0.009, 0.0004),
        "IWM": (200.0, 0.014, 0.0001),
        "DIA": (390.0, 0.009, 0.0002),
        "FXE": (100.0, 0.005, 0.0),
        "FXB": (123.0, 0.005, 0.0),
        "BTC/USD": (60000.0, 0.030, 0.0010),
        "ETH/USD": (3000.0, 0.038, 0.0008),
    }

    def __init__(self, seed: int = 7):
        self.seed = seed

    def fetch(self, symbol: str, timeframe: int, days: int) -> pd.DataFrame:
        base = self._base_1min(symbol, days)
        return resample(base, timeframe)

    def _base_1min(self, symbol: str, days: int) -> pd.DataFrame:
        price0, dvol, drift = self.PROFILE.get(symbol, (100.0, 0.011, 0.0002))
        rng = np.random.default_rng(zlib.crc32(f"{symbol}:{self.seed}".encode()))
        n_days = max(60, int(days * 252 / 365))
        end = pd.Timestamp.now(tz=NY).normalize()
        sessions = pd.bdate_range(end=end - pd.Timedelta(days=1), periods=n_days, tz=NY)
        minutes = np.arange(RTH_OPEN, RTH_CLOSE)
        m = len(minutes)
        x = (minutes - RTH_OPEN) / (m - 1)
        ushape = 0.6 + 1.6 * (x - 0.5) ** 2 * 4 / 2  # higher vol at open and close
        ushape = ushape / np.sqrt(np.mean(ushape**2))
        per_min = dvol / np.sqrt(m)
        frames = []
        price = price0
        vol_state = 1.0
        for d, day in enumerate(sessions):
            vol_state = float(np.clip(0.94 * vol_state + 0.06 * np.exp(rng.normal(0, 0.45)), 0.4, 3.0))
            # regime of the day: trend day, choppy (slightly mean-reverting) day or plain random walk
            regime = rng.choice(3, p=[0.3, 0.3, 0.4])
            day_drift = rng.normal(drift, dvol * (0.45 if regime == 0 else 0.15))
            revert = 0.004 if regime == 1 else 0.0
            gap = rng.normal(0, dvol * 0.35)
            price *= np.exp(gap)
            eps = rng.standard_t(5, size=m) / np.sqrt(5 / 3)
            rets = np.empty(m)
            dev = 0.0
            for i in range(m):
                r = day_drift / m + per_min * vol_state * ushape[i] * eps[i] - revert * dev
                dev += r
                rets[i] = r
            closes = price * np.exp(np.cumsum(rets))
            opens = np.concatenate([[price], closes[:-1]])
            wick = np.abs(rng.normal(0, per_min * vol_state * 0.6, size=(2, m))) * closes
            highs = np.maximum(opens, closes) + wick[0]
            lows = np.minimum(opens, closes) - wick[1]
            vol = (ushape**2) * rng.lognormal(10.0, 0.4, size=m)
            idx = day + pd.to_timedelta(minutes, unit="m")
            frames.append(pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vol}, index=idx))
            price = closes[-1]
        df = pd.concat(frames)
        df.index = df.index.tz_convert("UTC")
        return df


# --------------------------------------------------------------------------- loading


def make_provider(cfg) -> object:
    source = cfg.resolved_data_source
    if source == "alpaca":
        if not (cfg.alpaca_key and cfg.alpaca_secret):
            raise RuntimeError("DATA_SOURCE=alpaca requires ALPACA_API_KEY and ALPACA_SECRET_KEY")
        return AlpacaProvider(cfg.alpaca_key, cfg.alpaca_secret, cfg.alpaca_feed)
    if source == "yahoo":
        return YahooProvider()
    return SyntheticProvider(cfg.seed)


def load_datasets(cfg, instruments: dict[str, Instrument]) -> tuple[dict[str, Dataset], dict[str, str]]:
    """Fetch (or load cached) bars for every instrument x timeframe. Returns datasets and their pickle paths."""
    provider = make_provider(cfg)
    cache_dir = Path(cfg.data_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, Dataset] = {}
    files: dict[str, str] = {}
    for sym, inst in instruments.items():
        ticker = inst.yahoo if provider.name == "yahoo" else inst.proxy
        ratio = 1.0 if provider.name == "yahoo" else inst.proxy_ratio
        for tf in cfg.timeframes:
            path = cache_dir / f"{provider.name}_{sym}_{tf}m_{cfg.history_days}d.pkl"
            ds = None
            if path.exists() and time.time() - path.stat().st_mtime < 12 * 3600:
                try:
                    with open(path, "rb") as fh:
                        ds = pickle.load(fh)
                except Exception:
                    ds = None
            if ds is None:
                raw = provider.fetch(ticker, tf, cfg.history_days)
                if cfg.rth_only:
                    raw = filter_rth(raw)
                ds = build_dataset(raw, inst, tf, provider.name, ratio)
                if ds.n_days < 20:
                    raise RuntimeError(f"not enough history for {sym} {tf}m ({ds.n_days} days)")
                with open(path, "wb") as fh:
                    pickle.dump(ds, fh, protocol=pickle.HIGHEST_PROTOCOL)
            datasets[ds.key] = ds
            files[ds.key] = str(path)
            log.info("dataset %s: %d bars, %d days (%s)", ds.key, ds.n, ds.n_days, ds.source)
    return datasets, files
