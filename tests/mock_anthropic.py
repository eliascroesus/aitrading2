"""A tiny stand-in for the Anthropic Messages API used by the integration test.

It records every request and answers with schema-shaped JSON so the hive's real
SDK calls, prompt plumbing and decision handling can be exercised offline.
"""
from __future__ import annotations

import json
import itertools

from fastapi import FastAPI, Request

app = FastAPI()
REQUESTS: list[dict] = []
_counter = itertools.count()

DRONE_STRATEGIES = [
    {
        "name": "Mock ORB VWAP", "family": "orb", "instrument": "MNQ", "timeframe": 5,
        "long_entry": "close > or_high(15) and close > vwap", "short_entry": "close < or_low(15) and close < vwap",
        "exit_when": "", "stop_type": "atr", "stop_value": 1.2, "target_type": "r", "target_value": 2.0,
        "breakeven_at_r": 0, "max_hold_bars": 40, "session_start": 585, "session_end": 720, "flat_at": 955,
        "risk_usd": 300, "max_contracts": 30, "max_trades_per_day": 1, "daily_loss_stop_usd": 0,
        "daily_profit_stop_usd": 0, "rationale": "mock",
    },
    {
        "name": "Mock sweep", "family": "liquidity_sweep", "instrument": "MES", "timeframe": 5,
        "long_entry": "sweep_pdl() and tod < 720", "short_entry": "", "exit_when": "",
        "stop_type": "swing", "stop_value": 3, "target_type": "r", "target_value": 2.0, "breakeven_at_r": 1,
        "max_hold_bars": 40, "session_start": 575, "session_end": 900, "flat_at": 955, "risk_usd": 250,
        "max_contracts": 30, "max_trades_per_day": 2, "daily_loss_stop_usd": 500, "daily_profit_stop_usd": 0,
        "rationale": "mock",
    },
    {
        "name": "Mock broken", "family": "other", "instrument": "MNQ", "timeframe": 5,
        "long_entry": "close > supertrend(10, 3)", "short_entry": "", "exit_when": "", "stop_type": "atr",
        "stop_value": 1, "target_type": "r", "target_value": 1, "breakeven_at_r": 0, "max_hold_bars": 10,
        "session_start": 575, "session_end": 900, "flat_at": 955, "risk_usd": 200, "max_contracts": 10,
        "max_trades_per_day": 1, "daily_loss_stop_usd": 0, "daily_profit_stop_usd": 0, "rationale": "mock",
    },
]


@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()
    REQUESTS.append({"headers": dict(request.headers), "query": str(request.url.query), "body": body})
    system = body["system"][0]["text"] if isinstance(body.get("system"), list) else str(body.get("system"))
    n = next(_counter)
    if "risk desk" in system:
        payload = {"verdict": "ready_for_paper", "confidence": "medium",
                   "rules_in_plain_english": "Buy the first 5-minute candle direction after 9:35, stop 1 ATR, take 2R, one trade a day.",
                   "why_it_might_work": "Opening order-flow imbalance tends to persist into the first hour.",
                   "risks": ["Proxy ETF data, not real futures", "Only 5 years of history"],
                   "paper_test_checklist": ["Fills within 1 tick of the backtest", "Daily P&L within the simulated range"],
                   "suggested_tweaks": ["Skip FOMC days"]}
    elif "OVERMIND" in system:
        payload = {
            "assessment": f"mock assessment {n}",
            "directive": "Concentrate on opening-range breakouts on MNQ with one trade per day.",
            "spawn": [
                {"name": "Mock ORB", "specialty": "opening range", "mission": "ORB on MNQ", "instrument_focus": "MNQ"},
                {"name": "Mock Sweep", "specialty": "liquidity sweeps", "mission": "sweeps on MES", "instrument_focus": "MES"},
            ],
            "terminate": [],
            "memory": ["Mock lesson: one trade per day reduces max-loss failures."],
            "evolver_focus": "balanced",
        }
    else:
        payload = {"analysis": f"mock drone analysis {n}", "insight": f"mock insight {n % 2}", "strategies": DRONE_STRATEGIES}
    return {
        "id": f"msg_{n}", "type": "message", "role": "assistant", "model": body["model"],
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 1200, "output_tokens": 600, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 3000},
    }


@app.get("/_requests")
async def requests_log():
    return REQUESTS
