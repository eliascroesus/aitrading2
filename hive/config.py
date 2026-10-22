"""Runtime configuration, loaded from environment variables / a .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional dependency
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return _env(name, "true" if default else "false").lower() in ("1", "true", "yes", "on")


def _list(name: str, default: str) -> list[str]:
    return [x.strip() for x in _env(name, default).split(",") if x.strip()]


@dataclass
class Config:
    # --- Claude ---
    anthropic_api_key: str | None = None
    overmind_model: str = "claude-opus-5-5"
    drone_model: str = "claude-opus-5-5"
    overmind_effort: str = "high"
    drone_effort: str = "medium"
    llm_fallbacks: bool = True
    llm_budget_usd: float | None = None
    llm_concurrency: int = 4

    # --- hive dynamics ---
    max_drones: int = 4
    strategies_per_round: int = 5
    drone_max_strikes: int = 3
    drone_cooldown_sec: float = 20.0
    overmind_interval_sec: float = 150.0
    overmind_patience: int = 4
    workers: int = 2
    ga_batch: int = 48

    # --- challenge + markets ---
    prop_preset: str = "topstep_50k"
    instruments: list[str] = field(default_factory=lambda: ["MNQ", "MES", "MGC"])
    timeframes: list[int] = field(default_factory=lambda: [5, 15])
    ratio_overrides: dict[str, float] = field(default_factory=dict)

    # --- data ---
    data_source: str = "auto"  # auto | alpaca | yahoo | synthetic
    history_days: int = 730
    rth_only: bool = True

    # --- Alpaca ---
    alpaca_key: str | None = None
    alpaca_secret: str | None = None
    alpaca_paper: bool = True
    alpaca_feed: str = "sip"
    alpaca_live_feed: str = "iex"
    paper_scale: float = 1.0

    # --- server ---
    dashboard_password: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    data_dir: Path = ROOT / "hive_data"
    seed: int = 7

    @property
    def offline(self) -> bool:
        """True when no Anthropic key is configured: drones fall back to procedural generation."""
        return not self.anthropic_api_key

    @property
    def resolved_data_source(self) -> str:
        if self.data_source != "auto":
            return self.data_source
        return "alpaca" if (self.alpaca_key and self.alpaca_secret) else "synthetic"


def load_config() -> Config:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")
    budget = _env("HIVE_LLM_BUDGET_USD")
    ratios: dict[str, float] = {}
    for item in _list("INSTRUMENT_RATIOS", ""):
        if ":" in item:
            sym, val = item.split(":", 1)
            ratios[sym.strip().upper()] = float(val)
    cpu = os.cpu_count() or 2
    return Config(
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
        overmind_model=_env("HIVE_OVERMIND_MODEL", "claude-opus-5-5"),
        drone_model=_env("HIVE_DRONE_MODEL", "claude-opus-5-5"),
        overmind_effort=_env("HIVE_OVERMIND_EFFORT", "high"),
        drone_effort=_env("HIVE_DRONE_EFFORT", "medium"),
        llm_fallbacks=_bool("HIVE_LLM_FALLBACKS", True),
        llm_budget_usd=float(budget) if budget else None,
        llm_concurrency=_int("HIVE_LLM_CONCURRENCY", 4),
        max_drones=_int("HIVE_MAX_DRONES", 4),
        strategies_per_round=_int("HIVE_STRATEGIES_PER_ROUND", 5),
        drone_max_strikes=_int("HIVE_DRONE_MAX_STRIKES", 3),
        drone_cooldown_sec=_float("HIVE_DRONE_COOLDOWN_SEC", 20.0),
        overmind_interval_sec=_float("HIVE_OVERMIND_INTERVAL_SEC", 150.0),
        overmind_patience=_int("HIVE_OVERMIND_PATIENCE", 4),
        workers=_int("HIVE_WORKERS", max(1, cpu // 2)),  # half the cores keeps a laptop usable
        ga_batch=_int("HIVE_GA_BATCH", 48),
        prop_preset=_env("PROP_PRESET", "topstep_50k"),
        instruments=[s.upper() for s in _list("INSTRUMENTS", "MNQ,MES,MGC")],
        timeframes=[int(x) for x in _list("TIMEFRAMES", "5,15")],
        ratio_overrides=ratios,
        data_source=_env("DATA_SOURCE", "auto").lower(),
        history_days=_int("HISTORY_DAYS", 730),
        rth_only=_bool("RTH_ONLY", True),
        alpaca_key=_env("ALPACA_API_KEY"),
        alpaca_secret=_env("ALPACA_SECRET_KEY"),
        alpaca_paper=_bool("ALPACA_PAPER", True),
        alpaca_feed=_env("ALPACA_DATA_FEED", "sip").lower(),
        alpaca_live_feed=_env("ALPACA_LIVE_FEED", "iex").lower(),
        paper_scale=_float("PAPER_SIZE_SCALE", 1.0),
        dashboard_password=_env("DASHBOARD_PASSWORD"),
        host=_env("HOST", "127.0.0.1"),
        port=_int("PORT", 8000),
        data_dir=Path(_env("HIVE_DATA_DIR", str(ROOT / "hive_data"))),
        seed=_int("HIVE_SEED", 7),
    )
