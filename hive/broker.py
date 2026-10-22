"""Alpaca paper trading: run a hive strategy live and track a virtual prop-firm challenge on its P&L.

Alpaca has no futures, so the strategy trades the ETF proxy (QQQ for MNQ/NQ,
SPY for MES/ES, GLD for MGC/GC ...).  Share quantity is chosen so one futures
contract maps to ``proxy_ratio x point_value`` shares, which gives the same
dollar P&L per point as the contract.  The challenge (profit target, trailing
max loss, daily loss limit ...) is simulated on account equity relative to the
equity at deployment, divided by PAPER_SIZE_SCALE.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timedelta, timezone

import numpy as np

from . import lang, propfirm
from .data import NY, build_dataset, filter_rth
from .instruments import Instrument
from .strategy import StrategySpec

log = logging.getLogger("hive.paper")


class PaperTrader:
    def __init__(self, cfg, rules: propfirm.PropRules, instruments: dict[str, Instrument], bus, store):
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        self.cfg = cfg
        self.rules = rules
        self.instruments = instruments
        self.bus = bus
        self.store = store
        self.trading = TradingClient(cfg.alpaca_key, cfg.alpaca_secret, paper=cfg.alpaca_paper)
        self.data = StockHistoricalDataClient(cfg.alpaca_key, cfg.alpaca_secret)
        self.scale = max(cfg.paper_scale, 1e-6)
        self.spec: StrategySpec | None = None
        self.task: asyncio.Task | None = None
        self.account: dict = {}
        self.error: str = ""
        self.message: str = "idle"
        self.trades: deque = deque(maxlen=50)
        self.position: dict | None = None
        self.challenge: dict = {}
        self.days: list[dict] = []
        self._last_bar_ts: int | None = None
        self._day: str | None = None
        self._day_trades = 0
        self._day_start_equity = 0.0
        self._day_low = 0.0
        self._day_high = 0.0
        self._entry_equity = None
        saved = store.kv_get("paper_state")
        if saved and saved.get("spec"):
            self._restore(saved)

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        return {
            "connected": bool(self.account) and not self.error,
            "paper": self.cfg.alpaca_paper,
            "account": self.account,
            "error": self.error,
            "message": self.message,
            "deployed": self.spec.to_dict() if self.spec else None,
            "proxy": self.instruments[self.spec.instrument].proxy if self.spec else None,
            "position": self.position,
            "trades": list(self.trades)[-20:],
            "challenge": self.challenge,
            "days": self.days[-40:],
            "scale": self.scale,
        }

    async def refresh_account(self) -> None:
        try:
            acct = await asyncio.to_thread(self.trading.get_account)
            self.account = {
                "number": f"…{str(acct.account_number)[-4:]}",
                "status": str(acct.status).split(".")[-1],
                "equity": float(acct.equity),
                "last_equity": float(acct.last_equity),
                "buying_power": float(acct.buying_power),
                "cash": float(acct.cash),
            }
            self.error = ""
        except Exception as exc:
            self.error = f"Alpaca account error: {exc}"

    # ------------------------------------------------------------------ control
    async def deploy(self, spec_dict: dict) -> None:
        await self.stop(flatten=True)
        await self.refresh_account()
        if self.error:
            raise RuntimeError(self.error)
        self.spec = StrategySpec.from_dict(spec_dict, list(self.instruments), None)
        base = self.account["equity"]
        rules = self.rules
        self.challenge = {
            "status": "active", "start_equity": base, "balance": rules.start_balance, "pnl": 0.0,
            "peak": rules.start_balance, "floor": rules.start_balance - rules.max_loss,
            "target": rules.start_balance + rules.profit_target, "best_day": 0.0, "days_traded": 0,
            "started": time.time(), "today_pnl": 0.0, "fail_reason": "",
        }
        self.days = []
        self._day = None
        self._last_bar_ts = None
        self.trades.clear()
        self._persist()
        self.bus.emit("paper", f"Deployed '{self.spec.name}' to Alpaca {'paper' if self.cfg.alpaca_paper else 'LIVE'} on {self.instruments[self.spec.instrument].proxy}; virtual {rules.name} started.")
        self.task = asyncio.create_task(self._loop(), name="paper")

    async def stop(self, flatten: bool = True) -> None:
        if self.task:
            self.task.cancel()
            self.task = None
        if flatten and self.spec:
            await self._flatten("strategy stopped")
        if self.spec:
            self.bus.emit("paper", f"Paper trading of '{self.spec.name}' stopped.")
        self.spec = None
        self.message = "idle"
        self._persist()

    def resume_if_deployed(self) -> None:
        if self.spec and not self.task:
            self.task = asyncio.create_task(self._loop(), name="paper")

    # ------------------------------------------------------------------ main loop
    async def _loop(self) -> None:
        while self.spec:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("paper tick failed")
                self.error = f"{type(exc).__name__}: {exc}"
                self.bus.emit("error", f"Paper trader: {self.error}", persist=False)
            await asyncio.sleep(15)

    async def _tick(self) -> None:
        await self.refresh_account()
        clock = await asyncio.to_thread(self.trading.get_clock)
        if not clock.is_open:
            self.message = f"market closed, next open {clock.next_open.astimezone(timezone.utc):%Y-%m-%d %H:%M} UTC"
            return
        spec = self.spec
        inst = self.instruments[spec.instrument]
        now_ny = datetime.now(timezone.utc).astimezone(_ny_tz())
        today = now_ny.strftime("%Y-%m-%d")
        tod_now = now_ny.hour * 60 + now_ny.minute
        equity = self.account.get("equity", 0.0)
        self._update_challenge(today, equity)
        if self.challenge.get("status") != "active":
            self.message = f"challenge {self.challenge['status']}: trading halted"
            if self.position:
                await self._flatten("challenge finished")
            return

        await self._sync_position(inst)
        if tod_now >= spec.flat_at:
            if self.position:
                await self._flatten("flat_at reached")
            self.message = "past flat time; flat until tomorrow"
            return

        bars = await asyncio.to_thread(self._recent_bars, inst.proxy, spec.timeframe)
        if bars is None or bars.empty:
            self.message = "waiting for bars"
            return
        ds = build_dataset(bars, inst, spec.timeframe, "alpaca-live", inst.proxy_ratio)
        i = ds.n - 1
        bar_ts = int(ds.ts[i])
        if bar_ts == self._last_bar_ts:
            self.message = f"watching {inst.proxy}: waiting for the next {spec.timeframe}m bar"
            return
        self._last_bar_ts = bar_ts
        eng = lang.Engine(ds)
        long_sig = eng.signal(spec.long_entry) if spec.long_entry else None
        short_sig = eng.signal(spec.short_entry) if spec.short_entry else None
        exit_sig = eng.signal(spec.exit_when) if spec.exit_when else None
        atr = eng.series("atr(14)")

        if self.position:
            if exit_sig is not None and exit_sig[i]:
                await self._flatten("exit_when fired")
            self.message = f"in position {self.position['side']} {self.position['qty']} {inst.proxy}"
            return

        side = 1 if (long_sig is not None and long_sig[i]) else (-1 if (short_sig is not None and short_sig[i]) else 0)
        if side == 0:
            self.message = f"no signal on bar {datetime.fromtimestamp(bar_ts, timezone.utc).astimezone(_ny_tz()):%H:%M}"
            return
        if not (spec.session_start <= tod_now < spec.session_end):
            self.message = "signal outside session window"
            return
        if self._day_trades >= spec.max_trades_per_day:
            self.message = "max trades for today reached"
            return
        today_pnl = self.challenge.get("today_pnl", 0.0)
        if spec.daily_loss_stop_usd and today_pnl <= -spec.daily_loss_stop_usd:
            self.message = "daily loss stop reached"
            return
        if spec.daily_profit_stop_usd and today_pnl >= spec.daily_profit_stop_usd:
            self.message = "daily profit stop reached"
            return
        await self._enter(spec, inst, ds, i, side, atr)

    def _recent_bars(self, symbol: str, tf: int):
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        end = datetime.now(timezone.utc)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(tf, TimeFrameUnit.Minute),
                               start=end - timedelta(days=12), end=end, feed=DataFeed(self.cfg.alpaca_live_feed))
        df = self.data.get_stock_bars(req).df
        if df is None or df.empty:
            return None
        if hasattr(df.index, "levels"):
            df = df.xs(symbol, level=0)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = filter_rth(df)
        # keep only completed bars
        cutoff = end - timedelta(minutes=tf)
        return df[df.index <= cutoff]

    async def _enter(self, spec: StrategySpec, inst: Instrument, ds, i: int, side: int, atr: np.ndarray) -> None:
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest

        c = ds.c[i]
        if spec.stop_type == "atr":
            dist = spec.stop_value * atr[i]
        elif spec.stop_type == "pct":
            dist = spec.stop_value / 100 * c
        elif spec.stop_type == "points":
            dist = spec.stop_value
        else:
            n = int(spec.stop_value)
            ref = np.min(ds.l[max(0, i - n + 1): i + 1]) if side > 0 else np.max(ds.h[max(0, i - n + 1): i + 1])
            dist = abs(c - ref) + inst.tick_size
        if not np.isfinite(dist) or dist <= 0:
            self.message = "invalid stop distance; skipped"
            return
        contracts = int(spec.risk_usd // (dist * inst.point_value + inst.commission_rt))
        contracts = min(contracts, spec.max_contracts, int(self.rules.max_contracts / inst.mini_equiv + 1e-9))
        if contracts < 1:
            self.message = "risk too small for 1 contract; skipped"
            return
        shares = int(round(contracts * inst.proxy_ratio * inst.point_value * self.scale))
        px = c / inst.proxy_ratio
        bp = self.account.get("buying_power", 0.0)
        if bp and shares * px > bp * 0.95:
            shares = int(bp * 0.95 / px)
        if shares < 1:
            self.message = "not enough buying power; skipped"
            return
        stop_px = round((c - side * dist) / inst.proxy_ratio, 2)
        tgt_dist = {"r": spec.target_value * dist, "atr": spec.target_value * atr[i], "pct": spec.target_value / 100 * c,
                    "points": spec.target_value}.get(spec.target_type, 0.0)
        tgt_px = round((c + side * tgt_dist) / inst.proxy_ratio, 2) if tgt_dist and tgt_dist > 0 else None
        kwargs = dict(
            symbol=inst.proxy, qty=shares, side=OrderSide.BUY if side > 0 else OrderSide.SELL, time_in_force=TimeInForce.DAY,
            stop_loss=StopLossRequest(stop_price=stop_px),
        )
        if tgt_px:
            kwargs.update(order_class=OrderClass.BRACKET, take_profit=TakeProfitRequest(limit_price=tgt_px))
        else:
            kwargs.update(order_class=OrderClass.OTO)
        order = await asyncio.to_thread(self.trading.submit_order, MarketOrderRequest(**kwargs))
        self._day_trades += 1
        self._entry_equity = self.account.get("equity")
        rec = {"t": time.time(), "side": "long" if side > 0 else "short", "shares": shares, "contracts_equiv": contracts,
               "ref_price": round(px, 2), "stop": stop_px, "target": tgt_px, "order_id": str(order.id), "pnl": None}
        self.trades.append(rec)
        self.bus.emit("paper", f"PAPER {rec['side'].upper()} {shares} {inst.proxy} (~{contracts} {inst.symbol}) stop {stop_px} target {tgt_px or '-'}")
        self._persist()

    async def _sync_position(self, inst: Instrument) -> None:
        pos = None
        try:
            p = await asyncio.to_thread(self.trading.get_open_position, inst.proxy)
            pos = {"side": "long" if float(p.qty) > 0 else "short", "qty": abs(float(p.qty)), "avg": float(p.avg_entry_price),
                   "unrealized": float(p.unrealized_pl)}
        except Exception:
            pos = None
        if self.position and not pos and self.trades:
            last = self.trades[-1]
            if last.get("pnl") is None and self._entry_equity is not None:
                last["pnl"] = round((self.account.get("equity", 0.0) - self._entry_equity) / self.scale, 2)
                self.bus.emit("paper", f"PAPER trade closed: {last['side']} result ${last['pnl']:,.2f} (challenge units)")
        self.position = pos

    async def _flatten(self, why: str) -> None:
        try:
            await asyncio.to_thread(self.trading.close_all_positions, True)  # dedicated paper account assumed
            self.bus.emit("paper", f"Flattened paper account ({why}).")
        except Exception as exc:
            self.bus.emit("error", f"Paper flatten failed: {exc}", persist=False)

    # ------------------------------------------------------------------ virtual challenge
    def _update_challenge(self, today: str, equity: float) -> None:
        ch = self.challenge
        if not ch or ch.get("status") != "active":
            return
        rules = self.rules
        pnl_total = (equity - ch["start_equity"]) / self.scale
        if self._day != today:
            if self._day is not None:
                self._close_day()
            self._day = today
            self._day_trades = 0
            self._day_start_equity = equity
            self._day_low = self._day_high = 0.0
        day_pnl = (equity - self._day_start_equity) / self.scale
        self._day_low = min(self._day_low, day_pnl)
        self._day_high = max(self._day_high, day_pnl)
        bal = rules.start_balance + pnl_total
        ch["pnl"] = round(pnl_total, 2)
        ch["balance"] = round(bal, 2)
        ch["today_pnl"] = round(day_pnl, 2)
        # live checks
        if rules.trailing == "intraday":
            ch["peak"] = max(ch["peak"], bal)
            floor = ch["peak"] - rules.max_loss
            ch["floor"] = min(floor, rules.start_balance) if rules.trail_locks_at_start else floor
        if bal <= ch["floor"]:
            ch["status"], ch["fail_reason"] = "failed", "max loss breached"
            self.bus.emit("paper", f"Virtual challenge FAILED: balance {bal:,.0f} hit the max-loss floor {ch['floor']:,.0f}")
        elif rules.daily_loss_limit and day_pnl <= -rules.daily_loss_limit and rules.dll_fails_account:
            ch["status"], ch["fail_reason"] = "failed", "daily loss limit breached"
            self.bus.emit("paper", "Virtual challenge FAILED: daily loss limit breached")
        self._persist()

    def _close_day(self) -> None:
        ch, rules = self.challenge, self.rules
        day_pnl = ch.get("today_pnl", 0.0)
        self.days.append({"day": self._day, "pnl": day_pnl, "low": round(self._day_low, 2), "high": round(self._day_high, 2),
                          "trades": self._day_trades})
        if self._day_trades:
            ch["days_traded"] += 1
        ch["best_day"] = max(ch["best_day"], day_pnl)
        bal = ch["balance"]
        if rules.trailing == "eod":
            ch["peak"] = max(ch["peak"], bal)
            floor = ch["peak"] - rules.max_loss
            ch["floor"] = max(ch["floor"], min(floor, rules.start_balance) if rules.trail_locks_at_start else floor)
        profit = bal - rules.start_balance
        cons_ok = not rules.consistency_pct or ch["best_day"] <= rules.consistency_pct * profit
        if profit >= rules.profit_target and ch["days_traded"] >= rules.min_days and cons_ok:
            ch["status"] = "passed"
            self.bus.emit("champion", f"Virtual {rules.name} PASSED on Alpaca paper with '{self.spec.name}' after {len(self.days)} days!")

    def _persist(self) -> None:
        self.store.kv_set("paper_state", {
            "spec": self.spec.to_dict() if self.spec else None, "challenge": self.challenge, "days": self.days,
            "trades": list(self.trades), "day": self._day, "day_start_equity": self._day_start_equity,
            "day_trades": self._day_trades,
        })

    def _restore(self, saved: dict) -> None:
        self.spec = StrategySpec.from_dict(saved["spec"], list(self.instruments), None)
        self.challenge = saved.get("challenge") or {}
        self.days = saved.get("days") or []
        self.trades.extend(saved.get("trades") or [])
        self._day = saved.get("day")
        self._day_start_equity = saved.get("day_start_equity") or 0.0
        self._day_trades = saved.get("day_trades") or 0


def _ny_tz():
    from zoneinfo import ZoneInfo

    return ZoneInfo(NY)

