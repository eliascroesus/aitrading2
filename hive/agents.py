"""Hive agents: the Overmind (operator) and its drones (strategy designers).

Both are Claude instances driven through JSON-schema structured output.
Selection pressure is real: a drone that fails to improve its own best
strategy for N rounds is permanently deleted, and an Overmind whose hive stops
improving for P cycles is deleted and replaced by a fresh Overmind.
"""
from __future__ import annotations

import random
import time
import uuid
from dataclasses import asdict, dataclass, field

from . import lang
from .instruments import Instrument
from .propfirm import PropRules
from .strategy import FAMILIES, STOP_TYPES, TARGET_TYPES

DRONE_NAMES = [
    "Vespa", "Apis", "Bombus", "Melipona", "Xylocopa", "Osmia", "Andrena", "Halictus", "Megachile",
    "Anthophora", "Colletes", "Nomada", "Trigona", "Euglossa", "Ceratina", "Polistes", "Dolichovespula",
    "Scaptotrigona", "Tetragonula", "Lasioglossum", "Hylaeus", "Eucera", "Habropoda", "Panurgus",
]


@dataclass
class Agent:
    id: str
    name: str
    kind: str  # "overmind" | "drone"
    specialty: str
    mission: str
    instrument_focus: str
    model: str
    generation: int
    born: float = field(default_factory=time.time)
    status: str = "idle"  # idle | thinking | evaluating | dead
    died: float | None = None
    death_reason: str = ""
    rounds: int = 0
    submitted: int = 0
    valid: int = 0
    best_fitness: float = 0.0
    best_uid: str = ""
    strikes: int = 0
    cost_usd: float = 0.0
    last_thought: str = ""
    recent: list = field(default_factory=list)  # feedback from the last round
    spawned_by: str = ""

    @property
    def alive(self) -> bool:
        return self.status != "dead"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("recent", None)
        return d


def new_agent(kind: str, model: str, generation: int, specialty: str = "", mission: str = "", focus: str = "any",
              name: str | None = None, spawned_by: str = "") -> Agent:
    short = uuid.uuid4().hex[:5]
    if kind == "overmind":
        name = name or f"Overmind {generation}"
    else:
        name = f"{name.strip()[:28]}·{short[:3]}" if name and name.strip() else f"{random.choice(DRONE_NAMES)}-{short[:3]}"
    return Agent(id=f"{kind[0]}-{short}", name=name[:40], kind=kind, specialty=specialty[:200], mission=mission[:600],
                 instrument_focus=focus, model=model, generation=generation, spawned_by=spawned_by)


# ------------------------------------------------------------------------------- schemas


def strategy_schema(instruments: list[str]) -> dict:
    props = {
        "name": {"type": "string", "description": "short memorable name"},
        "family": {"type": "string", "enum": FAMILIES},
        "instrument": {"type": "string", "enum": instruments},
        "timeframe": {"type": "integer", "description": "bar size in minutes"},
        "long_entry": {"type": "string", "description": "expression; empty string disables longs"},
        "short_entry": {"type": "string", "description": "expression; empty string disables shorts"},
        "exit_when": {"type": "string", "description": "optional exit expression; empty string for none"},
        "stop_type": {"type": "string", "enum": list(STOP_TYPES)},
        "stop_value": {"type": "number"},
        "target_type": {"type": "string", "enum": list(TARGET_TYPES)},
        "target_value": {"type": "number"},
        "breakeven_at_r": {"type": "number"},
        "max_hold_bars": {"type": "integer"},
        "session_start": {"type": "integer"},
        "session_end": {"type": "integer"},
        "flat_at": {"type": "integer"},
        "risk_usd": {"type": "number"},
        "max_contracts": {"type": "integer"},
        "max_trades_per_day": {"type": "integer"},
        "daily_loss_stop_usd": {"type": "number"},
        "daily_profit_stop_usd": {"type": "number"},
        "rationale": {"type": "string"},
    }
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def drone_schema(instruments: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "analysis": {"type": "string", "description": "what you learned from the feedback and what you will try"},
            "insight": {"type": "string", "description": "one reusable lesson for the whole hive, or empty string"},
            "strategies": {"type": "array", "items": strategy_schema(instruments)},
        },
        "required": ["analysis", "insight", "strategies"],
        "additionalProperties": False,
    }


def overmind_schema(instruments: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "assessment": {"type": "string"},
            "directive": {"type": "string", "description": "broadcast to every drone for the next rounds"},
            "spawn": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "specialty": {"type": "string"},
                        "mission": {"type": "string"},
                        "instrument_focus": {"type": "string", "enum": instruments + ["any"]},
                    },
                    "required": ["name", "specialty", "mission", "instrument_focus"],
                    "additionalProperties": False,
                },
            },
            "terminate": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"agent_id": {"type": "string"}, "reason": {"type": "string"}},
                    "required": ["agent_id", "reason"],
                    "additionalProperties": False,
                },
            },
            "memory": {"type": "array", "items": {"type": "string"}},
            "evolver_focus": {"type": "string", "enum": ["exploit", "balanced", "explore"]},
        },
        "required": ["assessment", "directive", "spawn", "terminate", "memory", "evolver_focus"],
        "additionalProperties": False,
    }


# ------------------------------------------------------------------------------- prompts


def _markets_text(instruments: dict[str, Instrument], datasets_meta: dict) -> str:
    lines = []
    for sym, inst in instruments.items():
        tfs = sorted({m["timeframe"] for m in datasets_meta.values() if m["instrument"] == sym})
        meta = next((m for m in datasets_meta.values() if m["instrument"] == sym), None)
        src = f"{meta['source']} data via {meta['proxy']}, {meta['days']} sessions {meta['start']}..{meta['end']}" if meta else ""
        lines.append(
            f"- {sym} ({inst.name}): ${inst.point_value:g}/point, tick {inst.tick_size:g}, ~${inst.commission_rt:.2f} round-trip, "
            f"timeframes {tfs} min; {src}"
        )
    return "\n".join(lines)


def drone_system_prompt(rules: PropRules, instruments: dict[str, Instrument], datasets_meta: dict, horizon: int) -> str:
    speed_line = ""
    if rules.target_days:
        per_day = rules.profit_target / rules.target_days
        cons = f"the consistency rule caps the best day at {rules.consistency_pct:.0%} of total profit, so " if rules.consistency_pct else ""
        speed_line = (
            f"PASS SPEED GOAL: fitness counts only challenges passed within {rules.target_days} trading days. That needs roughly "
            f"${per_day:,.0f}+ net per trading day on average while never touching the max-loss floor: favour strategies that trade most days "
            f"with positive expectancy and controlled daily swings, size risk_usd so a normal losing day costs well under the max loss, "
            f"use daily_profit_stop_usd to bank good days ({cons}several solid days beat one huge one) and daily_loss_stop_usd to cap bad ones. "
            "The hive automatically combines strategies on DIFFERENT instruments into one account, so strong specialists on markets the "
            "leaders do not cover (and that make money on different days) are valuable."
        )
    return f"""You are a strategy-design drone inside HIVEMIND, a swarm of AI agents with one objective: discover intraday futures strategies that pass prop-firm evaluation challenges with the highest possible probability and make money. Nothing else matters.

THE CHALLENGE YOU ARE OPTIMIZING FOR
{rules.describe()}

HOW YOUR STRATEGIES ARE SCORED
Each strategy is backtested on historical bars, then its day-by-day results are replayed through the challenge rules thousands of times (block-bootstrap Monte Carlo, {horizon}-trading-day attempts). Data is split by time into TRAIN (first 60%) and VALIDATION (next 20%); a final 20% HOLDOUT is hidden from you and used to judge the hive.
fitness = geometric mean of the pass rates on THREE consecutive train folds and the validation segment, x trade-count factor (full credit at >=40 train trades and >=12 validation trades), x 0.5 if validation P&L is negative, with a small bonus for passing in fewer days.
Rules with more than 4 conditions per side are discounted 1.5% per extra condition (simple rules generalise better).
ROBUSTNESS CHECK: every strategy that could enter the top 30 is re-run with all its numbers nudged slightly (periods +-10-15%, thresholds +-5-10%, times +-5-10 min, stop/target/risk +-10%). Its ranking score becomes the average of its own fitness and the median of those neighbours; unchecked strategies rank at 85% of fitness. Knife-edge parameter choices therefore lose; broad, stable edges win.
{speed_line}
So: a strategy must pass in every period (not one lucky stretch), survive small parameter changes, trade often enough, and reach +${rules.profit_target:,.0f} before the account hits the max-loss floor.
A zero-edge strategy passes a meaningful share of attempts by luck; the dashboard compares every strategy's holdout pass rate to that luck baseline, so only real edge counts. Variance control (risk per trade vs. the ${rules.max_loss:,.0f} max loss and the daily loss limit), win rate, payoff, frequency and session choice all drive the pass rate. Strategies that only work on one segment score near zero.

MARKETS (bars are regular trading hours 09:30-16:00 New York, prices in futures points)
{_markets_text(instruments, datasets_meta)}

EXECUTION MODEL
- Entry/exit expressions are evaluated on each bar's close; entries fill at the NEXT bar's open plus 1 tick slippage.
- One position at a time. Size = floor(risk_usd / (stop distance in points x point value + commission)), capped by max_contracts and the firm's contract limit. If that is < 1 contract the trade is skipped, so make risk_usd big enough for the stop distance on the instrument you pick.
- Stops and targets are checked intrabar; if both are touched in one bar the stop is assumed first. Gaps through stops fill at the open.
- stop_type: atr (stop_value x atr(14) at the signal bar), pct (% of price), points, swing (stop_value = lookback bars; stop goes beyond the lowest low / highest high of that many bars).
- target_type: r (multiple of the stop distance), atr, pct, points, none (exit by exit_when, max_hold_bars or end of day).
- breakeven_at_r > 0 moves the stop to entry once price has moved that many R in favour (0 = off).
- New entries only when session_start <= bar time < session_end; every position is closed by flat_at (minutes after midnight NY, e.g. 955 = 15:55). max_trades_per_day caps entries; daily_loss_stop_usd / daily_profit_stop_usd stop trading for the day once realised P&L hits them (0 = off).

EXPRESSION LANGUAGE (Python-like boolean expressions over bar data, evaluated vectorised; no lookahead is possible)
{lang.reference_text()}
Examples:
  close > or_high(15) and close > vwap and tod < 690
  sweep_pdl() and within(fvg_bull(), 3) and tod >= 600 and tod < 720
  rsi(close, 2) < 10 and close > ema(close, 200) and adx(14) < 25
  crossover(ema(close, 9), ema(close, 21)) and close > prev_high
Invalid expressions are rejected and the error is returned to you next round, so use only the variables and functions above, with literal numeric constants for P/M/T/F arguments.

OUTPUT
Return JSON matching the schema: a short analysis, one reusable insight for the hive (or empty string) and the list of strategies. Every field is required; use empty strings for unused expressions and 0 for disabled numeric options. Be inventive and quantitative: vary families, instruments, timeframes, sessions, exits and sizing. Learn hard from the feedback."""


def overmind_system_prompt(rules: PropRules, instruments: dict[str, Instrument], datasets_meta: dict, max_drones: int,
                           max_strikes: int, patience: int) -> str:
    return f"""You are the OVERMIND of HIVEMIND, a self-improving swarm of AI agents. Your sole goal: make the hive discover strategies with the highest probability of passing prop-firm challenges and making money, as fast as possible.

CHALLENGE: {rules.describe()}

MARKETS
{_markets_text(instruments, datasets_meta)}

THE HIVE
- Drones: Claude agents you spawn. Each round a drone designs strategies (an expression language over intraday bars: liquidity sweeps, FVGs, opening ranges, VWAP, ICT/SMC concepts, indicators, session timing...) and gets back their backtest + challenge Monte Carlo results. Up to {max_drones} drones can live at once.
- Evolver: a genetic algorithm that continuously mutates and crosses the best strategies (thousands per hour). You steer it with evolver_focus: exploit (refine the leaders), balanced, explore (more random immigrants and diversity).
- Fitness = geometric mean of pass rates on three train folds + validation, x trade-count factor, x simplicity. Leading strategies are then re-tested with nudged parameters (robustness check) and ranked by that robust score. A hidden holdout segment (never used for selection) judges the hive; a large gap between validation and holdout means overfitting, and a holdout pass rate near the zero-edge luck baseline means no real edge.
- When the leaderboard is full of near-identical variants of one idea, that niche is saturated: push drones toward different families, instruments, sessions and timeframes.
- Hive memory: shared insights every drone reads. Add durable lessons with `memory`.

NATURAL SELECTION (enforced automatically)
- A drone that fails to improve its own best fitness for {max_strikes} consecutive rounds is permanently deleted.
- You may terminate any drone early (give agent_id + reason) and spawn replacements with new missions.
- YOU are also mortal: if the hive's best fitness does not improve for {patience} consecutive Overmind cycles, you are permanently deleted and replaced by a new Overmind that inherits only the hive memory and your post-mortem.

EACH CYCLE
Read the state, decide what is working and what is not, then: write a directive for all drones, spawn drones with sharp, differentiated missions (specific families, instruments, sessions, risk profiles, or specific improvements to leading strategies), terminate wasteful drones, record memory. Missions should be concrete and quantitative. Return JSON matching the schema."""


def fmt_metrics(r: dict) -> str:
    t, v = r["train"], r["val"]
    md = v.get("median_days")
    folds = "/".join(f"{x:.0%}" for x in r.get("folds", []))
    return (
        f"fit {r['fitness']:.3f} | pass train {t['pass_mc']:.0%}" + (f" (folds {folds})" if folds else "") + f" val {v['pass_mc']:.0%} | "
        f"trades {t['trades']}/{v['trades']} | win {t['win_rate']:.0%}/{v['win_rate']:.0%} | "
        f"PF {min(t['profit_factor'], 99):.2f}/{min(v['profit_factor'], 99):.2f} | net ${t['net']:,.0f}/${v['net']:,.0f} | "
        f"fail-maxloss {t['fail_maxloss']:.0%}/{v['fail_maxloss']:.0%} timeout {t['timeout']:.0%}/{v['timeout']:.0%}"
        + (f" | median days to pass {md:.0f}" if md else "")
    )


def fmt_spec(s: dict) -> str:
    parts = [f"{s['instrument']} {s['timeframe']}m [{s['family']}]"]
    if s.get("long_entry"):
        parts.append(f"LONG: {s['long_entry']}")
    if s.get("short_entry"):
        parts.append(f"SHORT: {s['short_entry']}")
    if s.get("exit_when"):
        parts.append(f"EXIT: {s['exit_when']}")
    parts.append(
        f"stop {s['stop_type']} {s['stop_value']:g}, target {s['target_type']} {s['target_value']:g}, BE {s['breakeven_at_r']:g}R, "
        f"hold<={s['max_hold_bars']}, session {s['session_start']}-{s['session_end']} flat {s['flat_at']}, risk ${s['risk_usd']:.0f}, "
        f"max {s['max_trades_per_day']}/day, daily stop -${s['daily_loss_stop_usd']:.0f}/+${s['daily_profit_stop_usd']:.0f}"
    )
    return " | ".join(parts)


def drone_user_prompt(agent: Agent, directive: str, operator_notes: list[str], memory: list[str], leaders: list[dict],
                      n_strategies: int, max_strikes: int) -> str:
    lines = [
        f"YOU: {agent.name} ({agent.id}). Specialty: {agent.specialty or 'generalist'}. Instrument focus: {agent.instrument_focus}.",
        f"MISSION: {agent.mission or 'Find strategies with the highest challenge pass probability.'}",
        f"STATUS: round {agent.rounds + 1}, your best fitness {agent.best_fitness:.3f}, strikes {agent.strikes}/{max_strikes} (at {max_strikes} you are permanently deleted; any strategy that beats your best resets strikes).",
        "",
        f"OVERMIND DIRECTIVE: {directive or '(none yet)'}",
    ]
    if operator_notes:
        lines.append("HUMAN OPERATOR ORDERS: " + " | ".join(operator_notes[-5:]))
    if memory:
        lines.append("\nHIVE MEMORY:")
        lines += [f"- {m}" for m in memory[-20:]]
    if leaders:
        lines.append("\nHALL OF FAME (best strategies so far; improve on them or beat them with something new):")
        for i, rec in enumerate(leaders, 1):
            lines.append(f"{i}. {rec['spec']['name']} :: {fmt_spec(rec['spec'])}\n   {fmt_metrics(rec['result'])}")
    if agent.recent:
        lines.append("\nYOUR LAST ROUND:")
        for item in agent.recent:
            if item.get("error"):
                lines.append(f"- {item['name']}: REJECTED -> {item['error']}")
            else:
                lines.append(f"- {item['name']}: {item['metrics']}")
    lines.append(f"\nDesign {n_strategies} new strategies now.")
    return "\n".join(lines)


def overmind_user_prompt(state: dict) -> str:
    s = state
    lines = [
        f"CYCLE {s['cycle']} | elapsed {s['elapsed_min']:.0f} min | strategies tested {s['tested']:,} | best fitness {s['best_fitness']:.3f} "
        f"(best validation pass rate {s['best_val_pass']:.0%}) | stagnant cycles {s['stagnation']}/{s['patience']} | LLM spend ${s['spend']:.2f}",
    ]
    if s.get("post_mortems"):
        lines.append("\nPREDECESSOR POST-MORTEMS:")
        lines += [f"- {p}" for p in s["post_mortems"][-3:]]
    if s.get("operator"):
        lines.append("\nHUMAN OPERATOR ORDERS (highest priority): " + " | ".join(s["operator"][-5:]))
    lines.append("\nLIVING DRONES:")
    if s["drones"]:
        for a in s["drones"]:
            lines.append(
                f"- {a['id']} {a['name']} | specialty: {a['specialty']} | focus {a['instrument_focus']} | rounds {a['rounds']} | "
                f"best {a['best_fitness']:.3f} | strikes {a['strikes']} | valid {a['valid']}/{a['submitted']} | mission: {a['mission']}"
            )
    else:
        lines.append("- none (spawn some)")
    if s.get("graveyard"):
        lines.append("\nRECENTLY DELETED: " + "; ".join(s["graveyard"][-6:]))
    lines.append("\nFAMILY x INSTRUMENT (count tested, best fitness):")
    lines += [f"- {k}: {v[0]} tested, best {v[1]:.3f}" for k, v in s["niches"]]
    lines.append("\nHALL OF FAME:")
    for i, rec in enumerate(s["leaders"], 1):
        lines.append(f"{i}. {rec['spec']['name']} (by {rec['spec'].get('author','?')}) :: {fmt_spec(rec['spec'])}\n   {fmt_metrics(rec['result'])}")
    if s.get("combos"):
        lines.append("\nBEST COMBOS (strategies traded together on one account, each on its own instrument):")
        for c in s["combos"]:
            names = " + ".join(f"{m['name']} [{m['instrument']} {m['timeframe']}m]" for m in c["members"])
            lines.append(f"- {names} | size x{c['scale']} | fit {c['fitness']:.3f} | val pass within {c['fast_days']}d {c['val_fast']:.0%} | daily-P&L correlation {c['corr']:.2f}")
        lines.append("Combos pass faster when members are profitable on different days (low correlation): steer drones toward strong strategies on instruments the leaders do not cover.")
    if s.get("memory"):
        lines.append("\nHIVE MEMORY:")
        lines += [f"- {m}" for m in s["memory"][-20:]]
    lines.append(f"\nYour previous directive: {s.get('directive') or '(none)'}")
    lines.append(f"Decide this cycle. You may have up to {s['max_drones']} living drones (currently {len(s['drones'])}).")
    return "\n".join(lines)
