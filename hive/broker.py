"""Alpaca paper trading: run one strategy or a combo live and track a virtual prop-firm challenge.

Alpaca has no futures, so each strategy trades its instrument's ETF proxy (QQQ for MNQ/NQ,
SPY for MES/ES, GLD for MGC/GC ...).  Share quantity is chosen so one futures contract maps
to ``proxy_ratio x point_value`` shares, which gives the same dollar P&L per point as the
contract.  A combo runs several strategies at once, one per instrument (and so one per
symbol).  The challenge (profit target, trailing max loss, daily loss limit ...) is simulated
on the whole account's equity relative to the equity at deployment, divided by
PAPER_SIZE_SCALE.  Use a dedicated paper account.
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


class Leg:
    """One strategy of the deployment, trading one ETF proxy."""

    def __init__(self, spec: StrategySpec, inst: Instrument):
        self.spec = spec
        self.inst = inst
        self.last_bar_ts: int | None = None
        self.day_trades = 0
        self.position: dict | None = None
        self.open_trade: dict | None = None
        self.message = "waiting"

    def to_dict(self) -> dict:
        return {"name": self.spec.name, "uid": self.spec.uid, "instrument": self.inst.symbol, "proxy": self.inst.proxy,
                "position": self.position, "message": self.message, "day_trades": self.day_trades,
                "risk_usd": self.spec.risk_usd}


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
        self.legs: list[Leg] = []
        self.combo_scale = 1.0
        self.task: asyncio.Task | None = None
        self.account: dict = {}
        self.error = ""
        self.message = "idle"
        self.trades: deque = deque(maxlen=60)
        self.challenge: dict = {}
        self.days: list[dict] = []
        self._day: str | None = None
        self._day_start_equity = 0.0
        self._day_low = 0.0
        self._day_high = 0.0
        saved = store.kv_get("paper_state")
        if saved and saved.get("legs"):
            self._restore(saved)

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        first = self.legs[0] if self.legs else None
        return {
            "connected": bool(self.account) and not self.error,
            "paper": self.cfg.alpaca_paper,
            "account": self.account,
            "error": self.error,
            "message": self.message,
            "deployed": {"name": " + ".join(l.spec.name for l in self.legs)} if self.legs else None,
            "legs": [l.to_dict() for l in self.legs],
            "combo_scale": self.combo_scale,
            "proxy": ", ".join(l.inst.proxy for l in self.legs) if self.legs else None,
            "position": first.position if first and len(self.legs) == 1 else None,
            "trades": list(self.trades)[-25:],
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
    async def deploy(self, spec_dicts: list[dict], scale: float = 1.0) -> None:
        legs = []
        for d in spec_dicts:
            spec = StrategySpec.from_dict(d, list(self.instruments), None)
            spec.risk_usd = round(spec.risk_usd * scale, 0)
            legs.append(Leg(spec, self.instruments[spec.instrument]))
        proxies = [l.inst.proxy for l in legs]
        if len(set(proxies)) != len(proxies):
            raise RuntimeError("each strategy in a combo needs its own instrument")
        await self.stop(flatten=True)
        await self.refresh_account()
        if self.error:
            raise RuntimeError(self.error)
        self.legs = legs
        self.combo_scale = scale
        rules = self.rules
        self.challenge = {
            "status": "active", "start_equity": self.account["equity"], "balance": rules.start_balance, "pnl": 0.0,
            "peak": rules.start_balance, "floor": rules.start_balance - rules.max_loss,
            "target": rules.start_balance + rules.profit_target, "best_day": 0.0, "days_traded": 0,
            "started": time.time(), "today_pnl": 0.0, "fail_reason": "",
        }
        self.days = []
        self._day = None
        self.trades.clear()
        self._persist()
        what = " + ".join(f"'{l.spec.name}' on {l.inst.proxy}" for l in legs)
        self.bus.emit("paper", f"Deployed {what} to Alpaca {'paper' if self.cfg.alpaca_paper else 'LIVE'} (size x{scale}); virtual {rules.name} started.")
        self.task = asyncio.create_task(self._loop(), name="paper")

    async def stop(self, flatten: bool = True) -> None:
        if self.task:
            self.task.cancel()
            self.task = None
        if flatten:
            for leg in self.legs:
                await self._flatten(leg, "strategy stopped")
        if self.legs:
            self.bus.emit("paper", "Paper trading stopped.")
        self.legs = []
        self.message = "idle"
        self._persist()

    def resume_if_deployed(self) -> None:
        if self.legs and not self.task:
            self.task = asyncio.create_task(self._loop(), name="paper")

    # ------------------------------------------------------------------ main loop
    async def _loop(self) -> None:
        while self.legs:
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
        now_ny = datetime.now(timezone.utc).astimezone(_ny_tz())
        self._update_challenge(now_ny.strftime("%Y-%m-%d"), self.account.get("equity", 0.0))
        if self.challenge.get("status") != "active":
            self.message = f"challenge {self.challenge['status']}: trading halted"
            for leg in self.legs:
                if leg.position:
                    await self._flatten(leg, "challenge finished")
            return
        self.message = "running"
        for leg in self.legs:
            try:
                await self._tick_leg(leg, now_ny)
            except Exception as exc:
                leg.message = f"error: {exc}"
                log.exception("leg %s failed", leg.spec.name)

    async def _tick_leg(self, leg: Leg, now_ny: datetime) -> None:
        spec, inst = leg.spec, leg.inst
        tod_now = now_ny.hour * 60 + now_ny.minute
        await self._sync_position(leg)
        if tod_now >= spec.flat_at:
            if leg.position:
                await self._flatten(leg, "flat_at reached")
            leg.message = "past flat time; flat until tomorrow"
            return
        bars = await asyncio.to_thread(self._recent_bars, inst.proxy, spec.timeframe)
        if bars is None or bars.empty:
            leg.message = "waiting for bars"
            return
        ds = build_dataset(bars, inst, spec.timeframe, "alpaca-live", inst.proxy_ratio)
        i = ds.n - 1
        bar_ts = int(ds.ts[i])
        if bar_ts == leg.last_bar_ts:
            return
        leg.last_bar_ts = bar_ts
        eng = lang.Engine(ds)
        long_sig = eng.signal(spec.long_entry) if spec.long_entry else None
        short_sig = eng.signal(spec.short_entry) if spec.short_entry else None
        exit_sig = eng.signal(spec.exit_when) if spec.exit_when else None
        atr = eng.series("atr(14)")
        bar_time = datetime.fromtimestamp(bar_ts, timezone.utc).astimezone(_ny_tz())

        if leg.position:
            if exit_sig is not None and exit_sig[i]:
                await self._flatten(leg, "exit_when fired")
            else:
                leg.message = f"in position {leg.position['side']} {leg.position['qty']:g} {inst.proxy}"
            return
        side = 1 if (long_sig is not None and long_sig[i]) else (-1 if (short_sig is not None and short_sig[i]) else 0)
        if side == 0:
            leg.message = f"no signal on the {bar_time:%H:%M} bar"
            return
        if not (spec.session_start <= tod_now < spec.session_end):
            leg.message = "signal outside session window"
            return
        if leg.day_trades >= spec.max_trades_per_day:
            leg.message = "max trades for today reached"
            return
        today_pnl = self.challenge.get("today_pnl", 0.0)
        if spec.daily_loss_stop_usd and today_pnl <= -spec.daily_loss_stop_usd:
            leg.message = "daily loss stop reached"
            return
        if spec.daily_profit_stop_usd and today_pnl >= spec.daily_profit_stop_usd:
            leg.message = "daily profit stop reached"
            return
        await self._enter(leg, ds, i, side, atr)

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
        return df[df.index <= end - timedelta(minutes=tf)]  # completed bars only

    async def _enter(self, leg: Leg, ds, i: int, side: int, atr: np.ndarray) -> None:
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest

        spec, inst = leg.spec, leg.inst
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
            leg.message = "invalid stop distance; skipped"
            return
        contracts = int(spec.risk_usd // (dist * inst.point_value + inst.commission_rt))
        contracts = min(contracts, spec.max_contracts, int(self.rules.max_contracts / inst.mini_equiv + 1e-9))
        if contracts < 1:
            leg.message = "risk too small for 1 contract; skipped"
            return
        shares = int(round(contracts * inst.proxy_ratio * inst.point_value * self.scale))
        px = c / inst.proxy_ratio
        bp = self.account.get("buying_power", 0.0)
        if bp and shares * px > bp * 0.95:
            shares = int(bp * 0.95 / px)
        if shares < 1:
            leg.message = "not enough buying power; skipped"
            return
        stop_px = round((c - side * dist) / inst.proxy_ratio, 2)
        tgt_dist = {"r": spec.target_value * dist, "atr": spec.target_value * atr[i], "pct": spec.target_value / 100 * c,
                    "points": spec.target_value}.get(spec.target_type, 0.0)
        tgt_px = round((c + side * tgt_dist) / inst.proxy_ratio, 2) if tgt_dist and tgt_dist > 0 else None
        kwargs = dict(symbol=inst.proxy, qty=shares, side=OrderSide.BUY if side > 0 else OrderSide.SELL,
                      time_in_force=TimeInForce.DAY, stop_loss=StopLossRequest(stop_price=stop_px))
        if tgt_px:
            kwargs.update(order_class=OrderClass.BRACKET, take_profit=TakeProfitRequest(limit_price=tgt_px))
        else:
            kwargs.update(order_class=OrderClass.OTO)
        order = await asyncio.to_thread(self.trading.submit_order, MarketOrderRequest(**kwargs))
        leg.day_trades += 1
        rec = {"t": time.time(), "strategy": spec.name, "symbol": inst.proxy, "side": "long" if side > 0 else "short",
               "shares": shares, "contracts_equiv": contracts, "ref_price": round(px, 2), "stop": stop_px, "target": tgt_px,
               "order_id": str(order.id), "entry": None, "pnl": None}
        leg.open_trade = rec
        self.trades.append(rec)
        leg.message = f"entered {rec['side']} {shares} {inst.proxy}"
        self.bus.emit("paper", f"PAPER {rec['side'].upper()} {shares} {inst.proxy} (~{contracts} {inst.symbol}) [{spec.name}] stop {stop_px} target {tgt_px or '-'}")
        self._persist()

    async def _sync_position(self, leg: Leg) -> None:
        symbol = leg.inst.proxy
        try:
            p = await asyncio.to_thread(self.trading.get_open_position, symbol)
            pos = {"side": "long" if float(p.qty) > 0 else "short", "qty": abs(float(p.qty)),
                   "avg": float(p.avg_entry_price), "unrealized": float(p.unrealized_pl)}
        except Exception:
            pos = None
        trade = leg.open_trade
        if pos and trade and trade.get("entry") is None:
            trade["entry"] = pos["avg"]
        if leg.position and not pos and trade and trade.get("pnl") is None:
            exit_px = await asyncio.to_thread(self._last_exit_fill, symbol, trade["side"])
            if exit_px and trade.get("entry"):
                sign = 1 if trade["side"] == "long" else -1
                trade["pnl"] = round(sign * (exit_px - trade["entry"]) * trade["shares"] / self.scale, 2)
            else:
                trade["pnl"] = round(leg.position.get("unrealized", 0.0) / self.scale, 2)
            leg.open_trade = None
            self.bus.emit("paper", f"PAPER trade closed [{leg.spec.name}]: {trade['side']} {symbol} ${trade['pnl']:,.2f} (challenge units)")
            self._persist()
        leg.position = pos

    def _last_exit_fill(self, symbol: str, side: str) -> float | None:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        try:
            orders = self.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[symbol], limit=20, nested=True))
        except Exception:
            return None
        exit_side = "sell" if side == "long" else "buy"
        fills = []
        for o in orders or []:
            for x in [o, *(getattr(o, "legs", None) or [])]:
                when = getattr(x, "filled_at", None) or getattr(x, "updated_at", None)
                if when and getattr(x, "filled_avg_price", None) and str(getattr(x, "side", "")).lower().endswith(exit_side):
                    fills.append((when, float(x.filled_avg_price)))
        return max(fills)[1] if fills else None

    async def _flatten(self, leg: Leg, why: str) -> None:
        """Cancel this strategy's open orders and close its position (other strategies are untouched)."""
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        symbol = leg.inst.proxy
        try:
            open_orders = await asyncio.to_thread(
                self.trading.get_orders, GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol], nested=True))
            for o in open_orders or []:
                await asyncio.to_thread(self.trading.cancel_order_by_id, o.id)
            if leg.position:
                await asyncio.sleep(0.5)
                await asyncio.to_thread(self.trading.close_position, symbol)
            self.bus.emit("paper", f"Flattened {symbol} [{leg.spec.name}] ({why}).")
        except Exception as exc:
            self.bus.emit("error", f"Paper flatten of {symbol} failed: {exc}", persist=False)

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
            for leg in self.legs:
                leg.day_trades = 0
            self._day_start_equity = equity
            self._day_low = self._day_high = 0.0
        day_pnl = (equity - self._day_start_equity) / self.scale
        self._day_low = min(self._day_low, day_pnl)
        self._day_high = max(self._day_high, day_pnl)
        bal = rules.start_balance + pnl_total
        ch["pnl"] = round(pnl_total, 2)
        ch["balance"] = round(bal, 2)
        ch["today_pnl"] = round(day_pnl, 2)
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
        traded = sum(l.day_trades for l in self.legs)
        self.days.append({"day": self._day, "pnl": day_pnl, "low": round(self._day_low, 2), "high": round(self._day_high, 2),
                          "trades": traded})
        if traded:
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
            names = " + ".join(l.spec.name for l in self.legs)
            self.bus.emit("champion", f"Virtual {rules.name} PASSED on Alpaca paper with {names} after {len(self.days)} days!")

    def _persist(self) -> None:
        self.store.kv_set("paper_state", {
            "legs": [l.spec.to_dict() for l in self.legs], "combo_scale": self.combo_scale,
            "challenge": self.challenge, "days": self.days, "trades": list(self.trades), "day": self._day,
            "day_start_equity": self._day_start_equity, "day_trades": [l.day_trades for l in self.legs],
        })

    def _restore(self, saved: dict) -> None:
        for d in saved["legs"]:
            spec = StrategySpec.from_dict(d, list(self.instruments), None)
            if spec.instrument in self.instruments:
                self.legs.append(Leg(spec, self.instruments[spec.instrument]))
        for leg, n in zip(self.legs, saved.get("day_trades") or []):
            leg.day_trades = n
        self.combo_scale = saved.get("combo_scale", 1.0)
        self.challenge = saved.get("challenge") or {}
        self.days = saved.get("days") or []
        self.trades.extend(saved.get("trades") or [])
        self._day = saved.get("day")
        self._day_start_equity = saved.get("day_start_equity") or 0.0


def _ny_tz():
    from zoneinfo import ZoneInfo

    return ZoneInfo(NY)
