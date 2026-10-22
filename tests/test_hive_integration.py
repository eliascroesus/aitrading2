"""End-to-end: short hive runs with Claude replaced by canned structured responses."""
import asyncio
import copy

from hive import llm as llm_mod
from hive.config import Config
from hive.data import load_datasets
from hive.hive import Hive
from hive.instruments import get_instrument
from hive.propfirm import get_rules
from hive.store import Bus, Store
from tests.mock_anthropic import DRONE_STRATEGIES

CALLS = {"overmind": 0, "drone": 0, "review": 0}


async def _fake_json_call(self, *, model, system, user, schema, effort="medium", max_tokens=16000):
    self.usage.calls += 1
    if "risk desk" in system:
        CALLS["review"] += 1
        return {"verdict": "needs_work", "confidence": "medium", "rules_in_plain_english": "x", "why_it_might_work": "y",
                "risks": ["r"], "paper_test_checklist": ["c"], "suggested_tweaks": ["t"]}, 0.01
    if "OVERMIND" in system:
        CALLS["overmind"] += 1
        return {
            "assessment": "test", "directive": "test directive",
            "spawn": [{"name": "Tester", "specialty": "orb", "mission": "test", "instrument_focus": "MNQ"}],
            "terminate": [], "memory": ["test memory"], "evolver_focus": "explore",
        }, 0.01
    CALLS["drone"] += 1
    return {"analysis": "drone test", "insight": "drone insight", "strategies": copy.deepcopy(DRONE_STRATEGIES)}, 0.01


def _run(tmp_path, monkeypatch, seconds, mode="economy", **overrides):
    monkeypatch.setattr(llm_mod.LLM, "json_call", _fake_json_call)
    for k in CALLS:
        CALLS[k] = 0
    cfg = Config(anthropic_api_key="sk-test", workers=2, ga_batch=12, history_days=150, data_source="synthetic",
                 instruments=["MNQ", "MES"], timeframes=[5], data_dir=tmp_path, overmind_interval_sec=5,
                 drone_cooldown_sec=0.2, max_drones=2, claude_mode=mode, **overrides)

    async def scenario():
        instruments = {s: get_instrument(s) for s in cfg.instruments}
        datasets, files = load_datasets(cfg, instruments)
        store = Store(tmp_path / "hive.db")
        bus = Bus(store)
        hive = Hive(cfg, get_rules("topstep_50k"), instruments, datasets, files, store, bus)
        hive.loop_interval = 1.0
        await hive.start()
        await asyncio.sleep(seconds)
        before_order = dict(CALLS)
        hive.operator_order("focus on MES")
        await asyncio.sleep(4)
        snap = hive.snapshot()
        drones = [a for a in hive.agents.values() if a.kind == "drone"]
        rounds = sum(a.rounds for a in drones) + sum(g["rounds"] for g in hive.graveyard)
        reviews = dict(hive.reviews)
        await hive.stop()
        return snap, drones, rounds, list(bus.events), before_order, reviews

    return asyncio.run(scenario())


def test_hive_runs_with_agents(tmp_path, monkeypatch):
    snap, drones, rounds, events, _, _ = _run(tmp_path, monkeypatch, 12, mode="full")
    assert snap["stats"]["tested"] > 50
    assert snap["mode"]["llm"] == "claude"
    assert snap["directive"] == "test directive"
    assert "test memory" in snap["memory"] and "drone insight" in snap["memory"]
    assert rounds >= 1
    assert any(e["kind"] == "spawn" for e in events)
    feedback = [f for d in drones for f in d.recent if f.get("error")]
    assert any("supertrend" in f["error"] for f in feedback) or rounds == 0 or not drones


def test_economy_mode_only_calls_claude_in_bursts(tmp_path, monkeypatch):
    snap, drones, rounds, events, before_order, _ = _run(tmp_path, monkeypatch, 12, burst_after_min=999)
    # one start-up burst: each of the 2 drones thinks once, the Overmind decides once
    assert before_order["drone"] <= 2 and before_order["overmind"] <= 2
    assert snap["stats"]["claude_mode"] == "economy"
    # an operator order triggers another burst
    assert CALLS["drone"] > before_order["drone"] or CALLS["overmind"] > before_order["overmind"]
    assert any("Idea burst" in e["msg"] for e in events)
    assert snap["stats"]["tested"] > 50  # the free evolver kept searching


def test_prop_ready_candidates_get_one_review(tmp_path, monkeypatch):
    snap, _, _, events, _, reviews = _run(tmp_path, monkeypatch, 14, burst_after_min=999, ready_min_pass=0.0,
                                         ready_min_edge=-1.0, ready_max_gap=1.0, ready_min_trades=0)
    assert snap["stats"]["ready_count"] >= 1
    assert any("PROP-READY" in e["msg"] for e in events)
    assert reviews and all(r["verdict"] == "needs_work" for r in reviews.values())
    assert CALLS["review"] == len(reviews)  # never reviews the same candidate twice
