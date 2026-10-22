"""Futures contract specs and the Alpaca ETF proxies used to approximate them.

Alpaca does not offer futures, so historical bars come from a liquid ETF proxy
(QQQ for Nasdaq, SPY for S&P 500, GLD for gold, ...) and are rescaled into
futures points with ``proxy_ratio`` (futures price ~= proxy price * ratio).
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

    @property
    def tick_value(self) -> float:
        return self.tick_size * self.point_value


INSTRUMENTS: dict[str, Instrument] = {
    "MNQ": Instrument("MNQ", "Micro E-mini Nasdaq-100", 2.0, 0.25, 0.74, "QQQ", 41.0, "NQ=F", 0.1),
    "NQ": Instrument("NQ", "E-mini Nasdaq-100", 20.0, 0.25, 2.80, "QQQ", 41.0, "NQ=F", 1.0),
    "MES": Instrument("MES", "Micro E-mini S&P 500", 5.0, 0.25, 0.74, "SPY", 10.0, "ES=F", 0.1),
    "ES": Instrument("ES", "E-mini S&P 500", 50.0, 0.25, 2.80, "SPY", 10.0, "ES=F", 1.0),
    "MGC": Instrument("MGC", "Micro Gold", 10.0, 0.10, 1.00, "GLD", 10.9, "GC=F", 0.1),
    "GC": Instrument("GC", "Gold", 100.0, 0.10, 3.00, "GLD", 10.9, "GC=F", 1.0),
    "M2K": Instrument("M2K", "Micro E-mini Russell 2000", 5.0, 0.10, 0.74, "IWM", 10.0, "RTY=F", 0.1),
    "MYM": Instrument("MYM", "Micro E-mini Dow", 0.5, 1.0, 0.74, "DIA", 100.0, "YM=F", 0.1),
    "M6E": Instrument("M6E", "Micro EUR/USD", 12500.0, 0.0001, 0.74, "FXE", 0.0109, "6E=F", 0.1),
}


def get_instrument(symbol: str, ratio_overrides: dict[str, float] | None = None) -> Instrument:
    inst = INSTRUMENTS[symbol.upper()]
    if ratio_overrides and inst.symbol in ratio_overrides:
        inst = replace(inst, proxy_ratio=ratio_overrides[inst.symbol])
    return inst
