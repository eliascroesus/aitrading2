"""Futures contract specs and the Alpaca ETF proxies used to approximate them.

Alpaca does not offer futures, so historical bars come from a liquid proxy
(ETFs: QQQ for Nasdaq, SPY for S&P 500, GLD for gold, FXE for EUR/USD; spot
crypto: BTC/USD, ETH/USD) and are rescaled into futures points with
``proxy_ratio`` (futures price ~= proxy price * ratio).
P&L is then computed with the real contract point value, so a backtest on QQQ
reads like a backtest on MNQ.  Ratios drift slowly over time; override them with
INSTRUMENT_RATIOS=MNQ:41.3,MES:10.05 in .env when they get stale.
Commissions are approximate round-trip prop-firm fees per contract.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Instrument:
    symbol: str
    name: str
    point_value: float  # USD per 1.0 point per contract
    tick_size: float
    commission_rt: float  # USD round trip per contract
    proxy: str  # Alpaca symbol used for data / paper trading
    proxy_ratio: float  # futures price ~= proxy price * proxy_ratio
    yahoo: str  # Yahoo Finance continuous-futures ticker
    mini_equiv: float = 1.0  # how much of a "max contracts" limit one contract consumes
    asset_class: str = "etf"  # proxy type on Alpaca: "etf" or "crypto"
    market: str = ""  # human-readable market group

    @property
    def tick_value(self) -> float:
        return self.tick_size * self.point_value


INSTRUMENTS: dict[str, Instrument] = {
    # equity indices
    "MNQ": Instrument("MNQ", "Micro E-mini Nasdaq-100", 2.0, 0.25, 0.74, "QQQ", 41.0, "NQ=F", 0.1, market="Nasdaq"),
    "NQ": Instrument("NQ", "E-mini Nasdaq-100", 20.0, 0.25, 2.80, "QQQ", 41.0, "NQ=F", 1.0, market="Nasdaq"),
    "MES": Instrument("MES", "Micro E-mini S&P 500", 5.0, 0.25, 0.74, "SPY", 10.0, "ES=F", 0.1, market="S&P 500"),
    "ES": Instrument("ES", "E-mini S&P 500", 50.0, 0.25, 2.80, "SPY", 10.0, "ES=F", 1.0, market="S&P 500"),
    "M2K": Instrument("M2K", "Micro E-mini Russell 2000", 5.0, 0.10, 0.74, "IWM", 10.0, "RTY=F", 0.1, market="Russell 2000"),
    "MYM": Instrument("MYM", "Micro E-mini Dow", 0.5, 1.0, 0.74, "DIA", 100.0, "YM=F", 0.1, market="Dow"),
    # metals
    "MGC": Instrument("MGC", "Micro Gold", 10.0, 0.10, 1.00, "GLD", 10.9, "GC=F", 0.1, market="Gold"),
    "GC": Instrument("GC", "Gold", 100.0, 0.10, 3.00, "GLD", 10.9, "GC=F", 1.0, market="Gold"),
    # forex (CME currency futures; proxies are the CurrencyShares ETFs)
    "M6E": Instrument("M6E", "Micro EUR/USD", 12500.0, 0.0001, 0.74, "FXE", 0.0109, "6E=F", 0.1, market="Forex EUR/USD"),
    "M6B": Instrument("M6B", "Micro GBP/USD", 6250.0, 0.0001, 0.74, "FXB", 0.0103, "6B=F", 0.1, market="Forex GBP/USD"),
    # crypto (CME micro crypto futures; proxies are Alpaca spot crypto pairs)
    "MBT": Instrument("MBT", "Micro Bitcoin (0.1 BTC)", 0.1, 5.0, 2.50, "BTC/USD", 1.0, "BTC=F", 0.1, "crypto", market="Bitcoin"),
    "MET": Instrument("MET", "Micro Ether (0.1 ETH)", 0.1, 0.5, 0.60, "ETH/USD", 1.0, "ETH=F", 0.1, "crypto", market="Ether"),
}


def get_instrument(symbol: str, ratio_overrides: dict[str, float] | None = None) -> Instrument:
    inst = INSTRUMENTS[symbol.upper()]
    if ratio_overrides and inst.symbol in ratio_overrides:
        inst = replace(inst, proxy_ratio=ratio_overrides[inst.symbol])
    return inst
