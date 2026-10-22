"""The hive orchestrator: evolver + drones + Overmind, natural selection, leaderboard and live state."""
from __future__ import annotations

import asyncio
import heapq
import logging
import multiprocessing as mp
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np

from . import agents as ag
from . import evaluator, seeds
from .config import Config
from .instruments import Instrument
from .llm import LLM, BudgetExceeded, LLMError
from .propfirm import PropRules
from .store import Bus, Store
from .strategy import StrategySpec, crossover, mutate, neighbors

log = logging.getLogger("hive.core")

STORE_MIN_FITNESS = 0.02  # strategies below this are counted but not written to disk
STORE_MAX_ROWS = 20_000  # database keeps the best this many strategies
SEEN_CAP = 250_000  # how many recent strategy fingerprints are remembered for de-duplication
UNVERIFIED = 0.85  # score multiplier until a strategy has survived the robustness check
ROBUST_NEIGHBORS = 6  # nudged variants evaluated per robustness check
ROBUST_TOP = 30  # strategies that could enter the top this many get checked


def score(rec: dict) -> float:
    """Ranking score: robustness-checked score when available, otherwise discounted fitness."""
    r = rec["result"]
    return r.get("robust_score", r["fitness"] * UNVERIFIED)

DEFAULT_MISSIONS = [
    ("ICT liquidity hunter", "Liquidity sweeps of PDH/PDL, opening-range and session highs/lows into fair value gaps; killzone timing (10:00-11:00 NY silver bullet, NY open).", "MNQ"),
    ("Opening-range & momentum", "Opening range breakouts (5-60 min), first-bar direction, intraday momentum into the close, VWAP trend holds.", "MNQ"),
    ("Chop & reversion", "Mean reversion when the market is ranging: Bollinger/RSI extremes, VWAP deviations, failed breakouts, gap fills.", "MES"),
    ("Gold specialist", "Find what works on micro gold during US hours: breakouts after compression, sweeps, trend days.", "MGC"),
    ("Risk architect", "Take the hall-of-fame leaders and re-engineer exits, breakeven, risk per trade, daily stops and session windows to maximise pass probability.", "any"),
    ("Wildcard", "Unconventional combinations nobody else is trying: candle patterns + structure, time-of-day edges, multi-condition hybrids.", "any"),
]


class Hive:
    def __init__(self, cfg: Config, rules: PropRules, instruments: dict[str, Instrument], datasets: dict, dataset_files: dict,
                 store: Store, bus: Bus):
        self.cfg = cfg
        self.rules = rules
        self.instruments = instruments
        self.datasets = datasets
        self.dataset_files = dataset_files
        self.store = store
        self.bus = bus
        self.inst_list = list(instruments)
        self.tfs = sorted({d.timeframe for d in datasets.values()})
        self.rng = np.random.default_rng(cfg.seed + int(time.time()) % 100000)
        self.datasets_meta = {k: d.summary() for k, d in datasets.items()}
        self.llm = None if cfg.offline else LLM(cfg.anthropic_api_key, cfg.llm_fallbacks, cfg.llm_budget_usd, cfg.llm_concurrency)
        self.pool: ProcessPoolExecutor | None = None
        self.paper = None  # set by the server when Alpaca is configured

        # prompts / schemas are stable for the whole run (prompt caching)
        self.drone_system = ag.drone_system_prompt(rules, instruments, self.datasets_meta, rules.horizon_days)
        self.overmind_system = ag.overmind_system_prompt(rules, instruments, self.datasets_meta, cfg.max_drones,
                                                         cfg.drone_max_strikes, cfg.overmind_patience)
        self.drone_schema = ag.drone_schema(self.inst_list)
        self.overmind_schema = ag.overmind_schema(self.inst_list)

        # state
        self.records: dict[str, dict] = {}
        # dedupe memory is bounded so a run of days never grows without limit
        self.seen: set[str] = store.all_uids()
        self.seen_fifo: deque = deque(self.seen)
        self.pending: set[str] = set()
        counts = store.kv_get("counts")
        self.tested, self.valid = (counts["tested"], counts["valid"]) if counts else store.counts()
        self.niches: dict[str, list] = store.kv_get("niches", {})  # "family / instrument" -> [tested, best fitness]
        self.best: dict | None = None
        self.best_history: list = store.kv_get("best_history", [])
        self.scatter: deque = deque(store.kv_get("scatter", []), maxlen=2400)
        self.agents: dict[str, ag.Agent] = {}
        self.graveyard: list[dict] = []
        self.memory: list[str] = [m["text"] for m in store.load_memory(60)]
        self.directive: str = store.kv_get("directive", "")
        self.operator_notes: list[str] = store.kv_get("operator_notes", [])
        self.post_mortems: list[str] = store.kv_get("post_mortems", [])
        self.evolver_focus: str = store.kv_get("evolver_focus", "balanced")
        self.overmind_gen: int = store.kv_get("overmind_gen", 0)
        self.cycle: int = store.kv_get("cycle", 0)
        self.stagnation = 0
        self.last_stag_check = time.time()
        self.last_cycle_best = 0.0
        self.last_assessment = ""
        self.started = time.time()
        self.prior_elapsed = float(store.kv_get("elapsed", 0.0))
        self.rate: deque = deque(maxlen=120)
        self.running = asyncio.Event()
        self.running.set()
        self.stopping = False
        self.tasks: list[asyncio.Task] = []
        self.drone_tasks: dict[str, asyncio.Task] = {}
        self.wake_overmind = asyncio.Event()

        for rec in store.top(400):
            self.records[rec["spec"]["uid"]] = rec
        if self.records:
            self.best = max(self.records.values(), key=score)
            self.last_cycle_best = score(self.best)
        self.robust_queue: list = []
        self.robust_queued: set[str] = set()
        for a in store.load_agents():
            agent = ag.Agent(**{k: v for k, v in a.items() if k in ag.Agent.__dataclass_fields__})
            if agent.alive:
                agent.status = "idle"
                self.agents[agent.id] = agent
            else:
                self.graveyard.append(agent.to_dict())

    # ------------------------------------------------------------------ lifecycle
    def _make_pool(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=self.cfg.workers, mp_context=mp.get_context("spawn"), initializer=evaluator.init_worker,
            initargs=(self.dataset_files, self.rules.to_dict(), self.cfg.ratio_overrides),
        )

    async def start(self) -> None:
        self.pool = self._make_pool()
        mode = "Claude agents" if self.llm else "OFFLINE procedural agents (set ANTHROPIC_API_KEY for Claude)"
        self.bus.emit("system", f"Hive online: {len(self.datasets)} datasets, {self.cfg.workers} workers, {mode}. Objective: {self.rules.name}.")
        if not self.records:
            await self.evaluate_batch(seeds.seed_strategies(self.inst_list, self.tfs), author="seed")
            self.bus.emit("system", f"Seeded {len(self.records)} strategies from the research library.")
        if not any(a.kind == "overmind" and a.alive for a in self.agents.values()):
            self._spawn_overmind()
        for a in list(self.agents.values()):
            if a.kind == "drone" and a.alive:
                self._start_drone(a)
        self.tasks = [
            asyncio.create_task(self._evolver_loop(), name="evolver"),
            asyncio.create_task(self._robust_loop(), name="robustness"),
            asyncio.create_task(self._overmind_loop(), name="overmind"),
            asyncio.create_task(self._housekeeping_loop(), name="housekeeping"),
        ]

    async def stop(self) -> None:
        self.stopping = True
        self.running.set()
        for t in [*self.tasks, *self.drone_tasks.values()]:
            t.cancel()
        self._persist_counters()
        if self.pool:
            self.pool.shutdown(wait=False, cancel_futures=True)

    def pause(self) -> None:
        self.running.clear()
        self.bus.emit("system", "Hive paused by operator.")

    def resume(self) -> None:
        self.running.set()
        self.bus.emit("system", "Hive resumed.")

    def operator_order(self, text: str) -> None:
        text = text.strip()[:500]
        if not text:
            return
        self.operator_notes.append(text)
        self.operator_notes = self.operator_notes[-10:]
        self.store.kv_set("operator_notes", self.operator_notes)
        self.bus.emit("operator", f"Operator order: {text}")
        self.wake_overmind.set()

    # ------------------------------------------------------------------ evaluation
    async def evaluate_batch(self, specs: list[StrategySpec], author: str = "") -> list[tuple[dict, dict]]:
        loop = asyncio.get_running_loop()
        fresh: list[dict] = []
        dupes: list[tuple[dict, dict]] = []
        batch_uids: set[str] = set()
        for s in specs:
            d = s.to_dict()
            if author and not d.get("author"):
                d["author"] = author
            uid = d["uid"]
            if uid in batch_uids or uid in self.pending:
                continue
            if uid in self.seen:
                prev = self.records.get(uid) or self.store.get(uid)
                if prev:
                    dupes.append((prev["spec"], prev["result"]))
                continue
            batch_uids.add(uid)
            fresh.append(d)
        if not fresh:
            return dupes
        self.pending |= batch_uids
        chunk = 6
        chunks = [fresh[i:i + chunk] for i in range(0, len(fresh), chunk)]
        try:
            outs = await asyncio.gather(*[loop.run_in_executor(self.pool, evaluator.evaluate_many, c) for c in chunks])
        except BrokenProcessPool:
            if self.stopping:
                return dupes
            self.bus.emit("error", "An evaluation worker died; restarting the worker pool.", persist=False)
            old, self.pool = self.pool, self._make_pool()
            old.shutdown(wait=False, cancel_futures=True)
            return dupes
        finally:
            self.pending -= batch_uids
        pairs = [(d, r) for c, o in zip(chunks, outs) for d, r in zip(c, o)]
        self._ingest(pairs)
        return pairs + dupes

    def _ingest(self, pairs: list[tuple[dict, dict]]) -> None:
        now = time.time()
        # only strategies worth looking at again are written to disk; the rest are just counted
        keep = [(d, r) for d, r in pairs
                if r.get("ok") and (r["fitness"] >= STORE_MIN_FITNESS or d.get("origin") in ("llm", "seed"))]
        if keep:
            self.store.save_results(keep)
        batch_fits: list[float] = []
        floor = self._elite_floor()
        cut = self._leader_cut(ROBUST_TOP)
        for spec, res in pairs:
            uid = spec["uid"]
            if uid not in self.seen:
                self.seen.add(uid)
                self.seen_fifo.append(uid)
            self.tested += 1
            if not res.get("ok"):
                continue
            self.valid += 1
            fit = res["fitness"]
            batch_fits.append(fit)
            niche = self.niches.setdefault(f"{spec['family']} / {spec['instrument']}", [0, 0.0])
            niche[0] += 1
            niche[1] = max(niche[1], fit)
            if len(self.records) < 400 or fit * UNVERIFIED > floor:
                self.records[uid] = {"spec": spec, "result": res, "t": now}
                if fit >= 0.05 and fit * UNVERIFIED > cut and uid not in self.robust_queued:
                    prio = fit * (0.5 if spec.get("origin") == "neighbor" else 1.0)
                    heapq.heappush(self.robust_queue, (-prio, uid))
                    self.robust_queued.add(uid)
        if batch_fits:  # thin the scatter: the batch best plus one random sample keeps minutes of history on screen
            picks = {max(batch_fits), batch_fits[int(self.rng.integers(len(batch_fits)))]}
            for f in picks:
                self.scatter.append([round(now, 1), round(f, 4)])
        if len(self.records) > 520:
            top = sorted(self.records.values(), key=score, reverse=True)[:400]
            self.records = {r["spec"]["uid"]: r for r in top}
        if len(self.robust_queue) > 200:
            self.robust_queue = heapq.nsmallest(150, self.robust_queue)
            heapq.heapify(self.robust_queue)
            self.robust_queued = {u for _, u in self.robust_queue}
        self._update_best(now)
        while len(self.seen_fifo) > SEEN_CAP:
            self.seen.discard(self.seen_fifo.popleft())
        self.rate.append((now, self.tested))

    def _elite_floor(self) -> float:
        if len(self.records) < 400:
            return -1.0
        scores = sorted((score(r) for r in self.records.values()), reverse=True)
        return scores[399] if len(scores) > 399 else -1.0

    def _leader_cut(self, n: int) -> float:
        scores = sorted((score(r) for r in self.records.values()), reverse=True)
        return scores[n - 1] if len(scores) >= n else 0.0

    def _update_best(self, now: float) -> None:
        if not self.records:
            return
        top = max(self.records.values(), key=score)
        prev = score(self.best) if self.best else 0.0
        new = score(top)
        changed = self.best is None or top["spec"]["uid"] != self.best["spec"]["uid"]
        self.best = top
        if changed or abs(new - prev) > 1e-9:
            res = top["result"]
            self.best_history.append([round(now, 1), round(new, 4), round(res["val"]["pass_mc"], 4),
                                      round(res["test"]["pass_mc"], 4), self.tested])
        if new > prev + 0.004:
            res, spec = top["result"], top["spec"]
            checked = "robustness-checked" if "robust_score" in res else "robustness check pending"
            self.bus.emit(
                "champion",
                f"New champion: {spec['name']} by {spec.get('author', '?')} | score {new:.3f} ({checked}) | validation pass {res['val']['pass_mc']:.0%} | holdout {res['test']['pass_mc']:.0%}",
                uid=spec["uid"], fitness=new,
            )

    async def _robust_loop(self) -> None:
        """Re-test leading strategies with slightly nudged settings; knife-edge strategies lose their rank."""
        while not self.stopping:
            await self.running.wait()
            if not self.robust_queue:
                await asyncio.sleep(0.5)
                continue
            _, uid = heapq.heappop(self.robust_queue)
            self.robust_queued.discard(uid)
            rec = self.records.get(uid)
            if rec is None or "robust_score" in rec["result"]:
                continue
            try:
                base = StrategySpec.from_dict(rec["spec"], self.inst_list, self.tfs)
                results = await self.evaluate_batch(neighbors(base, self.rng, ROBUST_NEIGHBORS))
                fits = [r["fitness"] for _, r in results if r.get("ok")]
                own = rec["result"]["fitness"]
                med = float(np.median(fits)) if len(fits) >= 3 else 0.0
                rec["result"]["robust_score"] = 0.5 * own + 0.5 * min(own, med)
                rec["result"]["robust"] = {"median": med, "min": float(min(fits)) if fits else 0.0, "n": len(fits)}
                self.store.update_result(uid, rec["result"])
                self._update_best(time.time())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("robustness check failed")
                self.bus.emit("error", f"Robustness check error: {exc}", persist=False)

    def leaders(self, n: int = 25, diverse: bool = False) -> list[dict]:
        ranked = sorted(self.records.values(), key=score, reverse=True)
        if not diverse:
            return ranked[:n]
        out, niche = [], {}
        for r in ranked:
            key = (r["spec"]["family"], r["spec"]["instrument"])
            if niche.get(key, 0) >= 2:
                continue
            niche[key] = niche.get(key, 0) + 1
            out.append(r)
            if len(out) >= n:
                break
        return out

    # ------------------------------------------------------------------ evolver
    def _tournament(self, elite: list[dict], k: int = 3) -> StrategySpec:
        picks = [elite[int(i)] for i in self.rng.integers(0, len(elite), size=k)]
        best = max(picks, key=score)
        return StrategySpec.from_dict(best["spec"], self.inst_list, self.tfs)

    def _parent_pools(self) -> tuple[list[dict], list[list[dict]]]:
        """Global elite plus per-niche elites, so several strategy families keep evolving at once."""
        elite = self.leaders(240)
        by_niche: dict[tuple, list[dict]] = {}
        for r in elite:
            group = by_niche.setdefault((r["spec"]["family"], r["spec"]["instrument"]), [])
            if len(group) < 12:
                group.append(r)
        return elite[:80], list(by_niche.values())

    def _parent(self, elite: list[dict], niches: list[list[dict]]) -> StrategySpec:
        if len(niches) > 1 and self.rng.random() < 0.5:
            return self._tournament(niches[int(self.rng.integers(len(niches)))])
        return self._tournament(elite)

    async def _evolver_loop(self) -> None:
        while not self.stopping:
            await self.running.wait()
            try:
                elite, niches = self._parent_pools()
                explore = {"exploit": 0.06, "balanced": 0.15, "explore": 0.35}.get(self.evolver_focus, 0.15)
                batch: list[StrategySpec] = []
                for _ in range(self.cfg.ga_batch):
                    r = self.rng.random()
                    if len(elite) < 6 or r < explore:
                        batch.append(seeds.random_strategy(self.rng, self.inst_list, self.tfs, author="evolver"))
                    elif r < explore + 0.15:
                        batch.append(crossover(self._parent(elite, niches), self._parent(elite, niches), self.rng, self.inst_list, self.tfs))
                    else:
                        batch.append(mutate(self._parent(elite, niches), self.rng, self.inst_list, self.tfs))
                await self.evaluate_batch(batch, author="evolver")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # keep evolving no matter what
                log.exception("evolver error")
                self.bus.emit("error", f"Evolver error: {exc}", persist=False)
                await asyncio.sleep(2)
            await asyncio.sleep(0.01)

    # ------------------------------------------------------------------ drones
    def _start_drone(self, agent: ag.Agent) -> None:
        self.drone_tasks[agent.id] = asyncio.create_task(self._drone_loop(agent), name=f"drone-{agent.id}")

    def spawn_drone(self, specialty: str, mission: str, focus: str = "any", name: str | None = None, by: str = "") -> ag.Agent | None:
        alive = [a for a in self.agents.values() if a.kind == "drone" and a.alive]
        if len(alive) >= self.cfg.max_drones:
            return None
        focus = focus if focus in self.inst_list else "any"
        agent = ag.new_agent("drone", self.cfg.drone_model if self.llm else "procedural", self.overmind_gen,
                             specialty, mission, focus, name, spawned_by=by)
        self.agents[agent.id] = agent
        self.store.save_agent(agent.to_dict())
        self.bus.emit("spawn", f"{by or 'Hive'} spawned drone {agent.name}: {specialty} (focus {focus})", agent=agent.id)
        self._start_drone(agent)
        return agent

    def kill(self, agent: ag.Agent, reason: str) -> None:
        if not agent.alive:
            return
        agent.status = "dead"
        agent.died = time.time()
        agent.death_reason = reason
        self.graveyard.append(agent.to_dict())
        self.graveyard = self.graveyard[-60:]
        self.store.save_agent(agent.to_dict())
        self.agents.pop(agent.id, None)
        self.bus.emit("death", f"{agent.name} permanently deleted: {reason}", agent=agent.id)
        task = self.drone_tasks.pop(agent.id, None)
        if task and task is not asyncio.current_task():
            task.cancel()

    async def _drone_loop(self, agent: ag.Agent) -> None:
        consecutive_errors = 0
        while not self.stopping and agent.alive:
            await self.running.wait()
            if self.llm and self.llm.over_budget():
                agent.status = "idle"
                agent.last_thought = "Waiting: LLM budget reached (raise HIVE_LLM_BUDGET_USD)."
                await asyncio.sleep(15)
                continue
            agent.status = "thinking"
            try:
                if self.llm:
                    leaders = self.leaders(8, diverse=True)
                    user = ag.drone_user_prompt(agent, self.directive, self.operator_notes, self.memory, leaders,
                                                self.cfg.strategies_per_round, self.cfg.drone_max_strikes)
                    data, cost = await self.llm.json_call(model=agent.model, system=self.drone_system, user=user,
                                                          schema=self.drone_schema, effort=self.cfg.drone_effort)
                    agent.cost_usd += cost
                else:
                    await asyncio.sleep(3 + 4 * self.rng.random())
                    data = self._procedural_drone_round(agent)
                consecutive_errors = 0
            except asyncio.CancelledError:
                raise
            except BudgetExceeded:
                continue
            except LLMError as exc:
                consecutive_errors += 1
                agent.status = "idle"
                agent.last_thought = f"LLM error: {exc}"
                self.bus.emit("error", f"{agent.name}: {exc}", persist=False)
                await asyncio.sleep(min(120, 10 * consecutive_errors))
                continue
            except Exception as exc:
                log.exception("drone error")
                agent.status = "idle"
                self.bus.emit("error", f"{agent.name} crashed a round: {exc}", persist=False)
                await asyncio.sleep(10)
                continue

            agent.last_thought = (data.get("analysis") or "")[:1500]
            self.bus.emit("thought", f"{agent.name}: {agent.last_thought[:400]}", agent=agent.id, persist=False)
            insight = (data.get("insight") or "").strip()
            if insight:
                self._remember(agent.name, insight)

            feedback, specs = [], []
            for raw in (data.get("strategies") or [])[: max(1, self.cfg.strategies_per_round)]:
                spec = StrategySpec.from_dict({**raw, "author": agent.id, "origin": "llm" if self.llm else "procedural"},
                                              self.inst_list, self.tfs)
                errs = spec.errors()
                if errs:
                    feedback.append({"name": spec.name, "error": "; ".join(errs)})
                else:
                    specs.append(spec)
            agent.submitted += len(data.get("strategies") or [])
            agent.valid += len(specs)
            agent.status = "evaluating"
            results = await self.evaluate_batch(specs, author=agent.id) if specs else []
            by_uid = {d["uid"]: r for d, r in results}

            # local search: let the evolver hammer on this round's best idea immediately
            ok = [(s, by_uid[s.uid]) for s in specs if s.uid in by_uid and by_uid[s.uid].get("ok")]
            if ok:
                top_spec = max(ok, key=lambda x: x[1]["fitness"])[0]
                kids = [mutate(top_spec, self.rng, self.inst_list, self.tfs) for _ in range(8)]
                for k in kids:
                    k.author = f"{agent.id}+evo"
                kid_res = await self.evaluate_batch(kids)
                for d, r in kid_res:
                    if r.get("ok"):
                        ok.append((StrategySpec.from_dict(d), r))

            round_best = 0.0
            best_uid = ""
            for s in specs:
                r = by_uid.get(s.uid)
                if r is None:
                    continue
                if not r.get("ok"):
                    feedback.append({"name": s.name, "error": r.get("error", "evaluation failed")})
                    continue
                feedback.append({"name": s.name, "metrics": ag.fmt_metrics(r)})
            for s, r in ok:
                if r["fitness"] > round_best:
                    round_best, best_uid = r["fitness"], s.uid
            agent.recent = feedback[-8:]
            agent.rounds += 1
            if round_best > agent.best_fitness + 0.003:
                agent.best_fitness, agent.best_uid, agent.strikes = round_best, best_uid, 0
            else:
                agent.strikes += 1
            agent.status = "idle"
            self.store.save_agent(agent.to_dict())
            self.bus.emit("round", f"{agent.name} round {agent.rounds}: {len(specs)} valid strategies, round best {round_best:.3f} (own best {agent.best_fitness:.3f}, strikes {agent.strikes}/{self.cfg.drone_max_strikes})",
                          agent=agent.id, persist=False)
            if agent.strikes >= self.cfg.drone_max_strikes:
                self.kill(agent, f"no improvement for {agent.strikes} consecutive rounds (best {agent.best_fitness:.3f})")
                self.wake_overmind.set()
                return
            await asyncio.sleep(self.cfg.drone_cooldown_sec if self.llm else 0.5)

    def _procedural_drone_round(self, agent: ag.Agent) -> dict:
        """Offline stand-in for an LLM drone: templates plus targeted mutations of the leaders."""
        out = []
        leaders = self.leaders(30)
        for i in range(self.cfg.strategies_per_round):
            if leaders and self.rng.random() < 0.5:
                base = StrategySpec.from_dict(leaders[int(self.rng.integers(len(leaders)))]["spec"], self.inst_list, self.tfs)
                spec = mutate(mutate(base, self.rng, self.inst_list, self.tfs), self.rng, self.inst_list, self.tfs)
            else:
                spec = seeds.random_strategy(self.rng, self.inst_list, self.tfs)
            if agent.instrument_focus in self.inst_list:
                spec.instrument = agent.instrument_focus
            spec.name = f"{agent.name.split('-')[0]} {spec.family[:10]} {int(self.rng.integers(100, 999))}"
            out.append(spec.to_dict())
        return {"analysis": f"Procedural round {agent.rounds + 1}: {len(out)} variants around the leaders and fresh templates.", "insight": "", "strategies": out}

    def _remember(self, author: str, text: str) -> None:
        text = text.strip()[:400]
        if not text or text in self.memory:
            return
        self.memory.append(text)
        self.memory = self.memory[-60:]
        self.store.add_memory(author, text)
        self.bus.emit("memory", f"{author} -> hive memory: {text}", persist=False)

    # ------------------------------------------------------------------ overmind
    def _spawn_overmind(self, post_mortem: str = "") -> ag.Agent:
        self.overmind_gen += 1
        self.store.kv_set("overmind_gen", self.overmind_gen)
        om = ag.new_agent("overmind", self.cfg.overmind_model if self.llm else "procedural", self.overmind_gen,
                          specialty="hive operator", mission="Maximise challenge pass probability")
        self.agents[om.id] = om
        self.store.save_agent(om.to_dict())
        self.stagnation = 0
        self.last_stag_check = time.time()
        self.bus.emit("spawn", f"{om.name} has taken control of the hive." + (f" (inherits post-mortem: {post_mortem[:160]})" if post_mortem else ""), agent=om.id)
        return om

    @property
    def overmind(self) -> ag.Agent | None:
        return next((a for a in self.agents.values() if a.kind == "overmind" and a.alive), None)

    def _overmind_state(self) -> dict:
        best = self.best["result"] if self.best else None
        drones = [a.to_dict() for a in self.agents.values() if a.kind == "drone" and a.alive]
        niches = sorted(((k, (v[0], v[1])) for k, v in self.niches.items()), key=lambda x: -x[1][1])[:24]
        return {
            "cycle": self.cycle,
            "elapsed_min": (time.time() - self.started + self.prior_elapsed) / 60,
            "tested": self.tested,
            "best_fitness": score(self.best) if self.best else 0.0,
            "best_val_pass": best["val"]["pass_mc"] if best else 0.0,
            "stagnation": self.stagnation,
            "patience": self.cfg.overmind_patience,
            "spend": self.llm.usage.cost_usd if self.llm else 0.0,
            "post_mortems": self.post_mortems,
            "operator": self.operator_notes,
            "drones": drones,
            "graveyard": [f"{g['name']} ({g['death_reason']})" for g in self.graveyard if g["kind"] == "drone"],
            "niches": niches,
            "leaders": self.leaders(10, diverse=True),
            "memory": self.memory,
            "directive": self.directive,
            "max_drones": self.cfg.max_drones,
        }

    async def _overmind_loop(self) -> None:
        await asyncio.sleep(2)
        while not self.stopping:
            await self.running.wait()
            om = self.overmind or self._spawn_overmind()
            om.status = "thinking"
            self.cycle += 1
            decision = None
            try:
                if self.llm and not self.llm.over_budget():
                    decision, cost = await self.llm.json_call(
                        model=om.model, system=self.overmind_system, user=ag.overmind_user_prompt(self._overmind_state()),
                        schema=self.overmind_schema, effort=self.cfg.overmind_effort,
                    )
                    om.cost_usd += cost
            except asyncio.CancelledError:
                raise
            except LLMError as exc:
                self.bus.emit("error", f"{om.name}: {exc}; falling back to procedural control this cycle", persist=False)
            except Exception as exc:
                log.exception("overmind error")
                self.bus.emit("error", f"{om.name} error: {exc}", persist=False)
            if decision is None:
                decision = self._procedural_overmind()
            self._apply_decision(om, decision)
            om.rounds += 1
            om.status = "idle"

            # natural selection for the Overmind itself: stagnation is counted per full interval,
            # so extra wake-ups (drone deaths, operator orders) never count against it
            best = score(self.best) if self.best else 0.0
            now = time.time()
            if best > self.last_cycle_best + 0.003:
                self.stagnation = 0
                self.last_stag_check = now
                om.best_fitness = best
            elif now - self.last_stag_check >= 0.9 * self.cfg.overmind_interval_sec:
                self.stagnation += 1
                self.last_stag_check = now
            om.strikes = self.stagnation
            self.last_cycle_best = max(self.last_cycle_best, best)
            self.store.save_agent(om.to_dict())
            self.store.kv_set("cycle", self.cycle)
            if self.stagnation >= self.cfg.overmind_patience:
                pm = (f"Overmind {om.generation} was deleted after {self.stagnation} cycles without improvement (best fitness stuck at {best:.3f}). "
                      f"Its last assessment: {self.last_assessment[:300]}")
                self.post_mortems.append(pm)
                self.post_mortems = self.post_mortems[-6:]
                self.store.kv_set("post_mortems", self.post_mortems)
                self.kill(om, f"hive stagnated for {self.stagnation} cycles under its command")
                self._spawn_overmind(pm)
                continue
            # sleep until the next cycle; early wake-ups are allowed but rate limited
            await asyncio.sleep(min(20.0, self.cfg.overmind_interval_sec / 3))
            self.wake_overmind.clear()
            try:
                await asyncio.wait_for(self.wake_overmind.wait(), timeout=self.cfg.overmind_interval_sec)
            except asyncio.TimeoutError:
                pass

    def _procedural_overmind(self) -> dict:
        alive = [a for a in self.agents.values() if a.kind == "drone" and a.alive]
        spawn = []
        used = {a.specialty for a in alive}
        pool = [m for m in DEFAULT_MISSIONS if m[0] not in used] or DEFAULT_MISSIONS
        for i in range(max(0, self.cfg.max_drones - len(alive))):
            spec, mission, focus = pool[i % len(pool)]
            spawn.append({"name": "", "specialty": spec, "mission": mission, "instrument_focus": focus})
        focus = "explore" if self.stagnation >= 2 else ("exploit" if self.stagnation == 0 and self.best else "balanced")
        lead = self.best["spec"]["name"] if self.best else "none"
        return {
            "assessment": f"Procedural control: {len(alive)} drones alive, leader {lead}, stagnation {self.stagnation}.",
            "directive": self.directive or "Beat the hall of fame: higher validation pass rate with enough trades.",
            "spawn": spawn, "terminate": [], "memory": [], "evolver_focus": focus,
        }

    def _apply_decision(self, om: ag.Agent, d: dict) -> None:
        self.last_assessment = d.get("assessment", "")
        om.last_thought = self.last_assessment[:1500]
        self.bus.emit("overmind", f"{om.name}: {self.last_assessment[:500]}", agent=om.id)
        directive = (d.get("directive") or "").strip()
        if directive and directive != self.directive:
            self.directive = directive[:800]
            self.store.kv_set("directive", self.directive)
            self.bus.emit("directive", f"{om.name} directive: {self.directive[:300]}", agent=om.id)
        focus = d.get("evolver_focus")
        if focus in ("exploit", "balanced", "explore") and focus != self.evolver_focus:
            self.evolver_focus = focus
            self.store.kv_set("evolver_focus", focus)
            self.bus.emit("system", f"Evolver switched to {focus} mode.")
        for t in d.get("terminate") or []:
            target = self.agents.get(t.get("agent_id", ""))
            if target and target.kind == "drone" and target.alive:
                self.kill(target, f"terminated by {om.name}: {t.get('reason', '')[:200]}")
        for s in d.get("spawn") or []:
            if not self.spawn_drone(s.get("specialty", ""), s.get("mission", ""), s.get("instrument_focus", "any"),
                                    name=(s.get("name") or None), by=om.name):
                break
        for m in d.get("memory") or []:
            self._remember(om.name, m)

    # ------------------------------------------------------------------ housekeeping / state
    async def _housekeeping_loop(self) -> None:
        ticks = 0
        while not self.stopping:
            await asyncio.sleep(10)
            self._persist_counters()
            ticks += 1
            if ticks % 60 == 0:  # every ~10 minutes keep the database small
                await asyncio.to_thread(self.store.prune, STORE_MAX_ROWS)

    def _persist_counters(self) -> None:
        self.store.kv_set("counts", {"tested": self.tested, "valid": self.valid})
        self.store.kv_set("niches", self.niches)
        self.store.kv_set("best_history", self.best_history[-2000:])
        self.store.kv_set("scatter", list(self.scatter))
        self.store.kv_set("elapsed", self.prior_elapsed + time.time() - self.started)

    def rate_per_min(self) -> float:
        if len(self.rate) < 2:
            return 0.0
        (t0, n0), (t1, n1) = self.rate[0], self.rate[-1]
        return (n1 - n0) / max(t1 - t0, 1e-6) * 60

    @staticmethod
    def summary(rec: dict) -> dict:
        s, r = rec["spec"], rec["result"]
        return {
            "uid": s["uid"], "name": s["name"], "instrument": s["instrument"], "timeframe": s["timeframe"],
            "family": s["family"], "author": s.get("author", ""), "origin": s.get("origin", ""), "sides": "both" if s.get("long_entry") and s.get("short_entry") else ("long" if s.get("long_entry") else "short"),
            "fitness": r["fitness"], "score": score(rec), "robust": "robust_score" in r,
            "robust_median": (r.get("robust") or {}).get("median"),
            "train_pass": r["train"]["pass_mc"], "val_pass": r["val"]["pass_mc"], "test_pass": r["test"]["pass_mc"],
            "test_luck": r["test"].get("luck"), "test_fast": r["test"].get("pass_fast"), "fast_days": r.get("fast_days", 10),
            "test_hist_pass": r["test"]["pass_hist"], "median_days": r["val"]["median_days"],
            "trades": r["trades"], "win_rate": (r["train"]["win_rate"] + r["val"]["win_rate"]) / 2,
            "pf": min(r["val"]["profit_factor"], 99.0), "net": r["train"]["net"] + r["val"]["net"], "test_net": r["test"]["net"],
            "t": rec.get("t", 0),
        }

    def snapshot(self) -> dict:
        best = self.best["result"] if self.best else None
        om = self.overmind
        hist = self.best_history
        if len(hist) > 400:
            step = len(hist) / 400
            hist = [hist[int(i * step)] for i in range(400)] + [hist[-1]]
        return {
            "t": time.time(),
            "status": "running" if self.running.is_set() else "paused",
            "mode": {
                "llm": "claude" if self.llm else "offline",
                "overmind_model": self.cfg.overmind_model, "drone_model": self.cfg.drone_model,
                "data": next(iter(self.datasets_meta.values()))["source"] if self.datasets_meta else "?",
                "preset": self.rules.key, "rules_name": self.rules.name, "rules_text": self.rules.describe(),
                "rules": self.rules.to_dict(),
            },
            "stats": {
                "tested": self.tested, "valid": self.valid, "rate_per_min": round(self.rate_per_min(), 1),
                "best_fitness": score(self.best) if self.best else 0.0,
                "best_robust": bool(best and "robust_score" in best),
                "best_val_pass": best["val"]["pass_mc"] if best else 0.0,
                "best_test_pass": best["test"]["pass_mc"] if best else 0.0,
                "best_test_luck": best["test"].get("luck") if best else None,
                "best_test_fast": best["test"].get("pass_fast") if best else None,
                "fast_days": self.rules.fast_days, "target_days": self.rules.target_days,
                "robust_queue": len(self.robust_queue),
                "best_uid": self.best["spec"]["uid"] if self.best else None,
                "best_name": self.best["spec"]["name"] if self.best else None,
                "elapsed": self.prior_elapsed + time.time() - self.started,
                "overmind_gen": self.overmind_gen, "cycle": self.cycle, "stagnation": self.stagnation,
                "patience": self.cfg.overmind_patience, "max_strikes": self.cfg.drone_max_strikes,
                "drones_alive": sum(1 for a in self.agents.values() if a.kind == "drone" and a.alive),
                "max_drones": self.cfg.max_drones,
                "deaths": len(self.graveyard),
                "spend": round(self.llm.usage.cost_usd, 4) if self.llm else 0.0,
                "budget": self.cfg.llm_budget_usd,
                "llm_calls": self.llm.usage.calls if self.llm else 0,
                "evolver_focus": self.evolver_focus,
                "workers": self.cfg.workers,
            },
            "agents": [a.to_dict() for a in sorted(self.agents.values(), key=lambda a: (a.kind != "overmind", a.born))],
            "graveyard": list(reversed(self.graveyard[-30:])),
            "leaders": [self.summary(r) for r in self.leaders(60)],
            "history": hist,
            "scatter": list(self.scatter)[::2] if len(self.scatter) > 1200 else list(self.scatter),
            "memory": self.memory[-30:],
            "directive": self.directive,
            "operator_notes": self.operator_notes[-5:],
            "datasets": list(self.datasets_meta.values()),
            "paper": self.paper.status() if self.paper else None,
            "overmind": om.to_dict() if om else None,
        }

    async def detail(self, uid: str) -> dict | None:
        rec = self.records.get(uid) or self.store.get(uid)
        if rec is None:
            return None
        if rec["result"].get("ok") and "equity" not in rec["result"]:
            loop = asyncio.get_running_loop()
            res = (await loop.run_in_executor(self.pool, evaluator.evaluate_many, [rec["spec"]]))[0]
            for k in ("robust_score", "robust"):
                if k in rec["result"]:
                    res[k] = rec["result"][k]
            if res.get("ok"):
                rec = {**rec, "result": res}
                self.store.update_result(uid, res)
        return rec
