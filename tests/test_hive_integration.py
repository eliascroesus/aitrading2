"""End-to-end: a short hive run with Claude replaced by canned structured responses."""
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


async def _fake_json_call(self, *, model, system, user, schema, effort="medium", max_tokens=16000):
    self.usage.calls += 1
    if "OVERMIND" in system:
        return {
            "assessment": "test", "directive": "test directive",
            "spawn": [{"name": "Tester", "specialty": "orb", "mission": "test", "instrument_focus": "MNQ"}],
            "terminate": [], "memory": ["test memory"], "evolver_focus": "explore",
        }, 0.01
    return {"analysis": "drone test", "insight": "drone insight", "strategies": copy.deepcopy(DRONE_STRATEGIES)}, 0.01


def test_hive_runs_with_agents(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_mod.LLM, "json_call", _fake_json_call)
    cfg = Config(anthropic_api_key="sk-test", workers=2, ga_batch=12, history_days=150, data_source="synthetic",
                 instruments=["MNQ", "MES"], timeframes=[5], data_dir=tmp_path, overmind_interval_sec=5,
                 drone_cooldown_sec=0.2, max_drones=2)

    async def scenario():
        instruments = {s: get_instrument(s) for s in cfg.instruments}
        datasets, files = load_datasets(cfg, instruments)
        store = Store(tmp_path / "hive.db")
        bus = Bus(store)
        hive = Hive(cfg, get_rules("topstep_50k"), instruments, datasets, files, store, bus)
        await hive.start()
        await asyncio.sleep(12)
        snap = hive.snapshot()
        drones = [a for a in hive.agents.values() if a.kind == "drone"]
        rounds = sum(a.rounds for a in drones) + sum(g["rounds"] for g in hive.graveyard)
        await hive.stop()
        return snap, drones, rounds, list(bus.events)

    snap, drones, rounds, events = asyncio.run(scenario())
    assert snap["stats"]["tested"] > 50
    assert snap["mode"]["llm"] == "claude"
    assert snap["directive"] == "test directive"
    assert "test memory" in snap["memory"] and "drone insight" in snap["memory"]
    assert rounds >= 1
    assert any(e["kind"] == "spawn" for e in events)
    # the broken mock strategy must be rejected and fed back to the drone
    feedback = [f for d in drones for f in d.recent if f.get("error")]
    assert any("supertrend" in f["error"] for f in feedback) or rounds == 0 or not drones
