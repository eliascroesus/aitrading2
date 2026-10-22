"""Thin async wrapper around the Anthropic SDK: JSON-schema structured output, cost tracking, fallbacks."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

import anthropic

log = logging.getLogger("hive.llm")

# USD per million tokens (input, output). Cache writes bill at 1.25x input, cache reads at 0.1x.
PRICES = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}
# models that accept the server-side `fallbacks: "default"` parameter
_FALLBACK_MODELS = ("claude-opus", "claude-sonnet-5-5", "claude-fable")
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    pass


class BudgetExceeded(LLMError):
    pass


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost_usd: float = 0.0
    by_model: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
            "cost_usd": round(self.cost_usd, 4),
            "by_model": {k: round(v, 4) for k, v in self.by_model.items()},
        }


class LLM:
    def __init__(self, api_key: str | None, fallbacks: bool = True, budget_usd: float | None = None, concurrency: int = 4):
        self.client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=4, timeout=600.0)
        self.fallbacks = fallbacks
        self.budget = budget_usd
        self.usage = Usage()
        self.sem = asyncio.Semaphore(max(1, concurrency))

    def _cost(self, model: str, u) -> float:
        pin, pout = PRICES.get(model, PRICES["claude-opus-5-5"])
        inp = (u.input_tokens or 0) * pin
        inp += (getattr(u, "cache_creation_input_tokens", 0) or 0) * pin * 1.25
        inp += (getattr(u, "cache_read_input_tokens", 0) or 0) * pin * 0.10
        return (inp + (u.output_tokens or 0) * pout) / 1_000_000

    def _record(self, model: str, u) -> float:
        cost = self._cost(model, u)
        self.usage.calls += 1
        self.usage.input_tokens += u.input_tokens or 0
        self.usage.output_tokens += u.output_tokens or 0
        self.usage.cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
        self.usage.cache_write += getattr(u, "cache_creation_input_tokens", 0) or 0
        self.usage.cost_usd += cost
        self.usage.by_model[model] = self.usage.by_model.get(model, 0.0) + cost
        return cost

    def over_budget(self) -> bool:
        return self.budget is not None and self.usage.cost_usd >= self.budget

    async def json_call(self, *, model: str, system: str, user: str, schema: dict, effort: str = "medium",
                        max_tokens: int = 16000) -> tuple[dict, float]:
        """One request whose reply is JSON matching ``schema``. Returns (data, cost_usd)."""
        if self.over_budget():
            raise BudgetExceeded(f"LLM budget of ${self.budget:.2f} reached")
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            # the long, stable system prompt (language reference, rules) is cached across calls
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        )
        async with self.sem:
            resp = await self._create(kwargs)
        cost = self._record(model, resp.usage)
        if resp.stop_reason == "refusal":
            raise LLMError("model declined the request")
        if resp.stop_reason == "max_tokens":
            raise LLMError("response hit max_tokens before finishing")
        text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), None)
        if not text:
            raise LLMError("no text block in response")
        try:
            return json.loads(text), cost
        except json.JSONDecodeError as exc:
            raise LLMError(f"invalid JSON from model: {exc}") from None

    async def _create(self, kwargs: dict):
        use_fallbacks = self.fallbacks and kwargs["model"].startswith(_FALLBACK_MODELS)
        try:
            if use_fallbacks:
                return await self.client.beta.messages.create(**kwargs, betas=[_FALLBACK_BETA], fallbacks="default")
            return await self.client.messages.create(**kwargs)
        except anthropic.BadRequestError as exc:
            if use_fallbacks and "fallback" in str(exc).lower():
                log.warning("server-side fallbacks rejected (%s); continuing without them", exc)
                self.fallbacks = False
                return await self.client.messages.create(**kwargs)
            raise LLMError(f"bad request: {exc}") from None
        except anthropic.AuthenticationError:
            raise LLMError("Anthropic API key rejected (check ANTHROPIC_API_KEY)") from None
        except anthropic.PermissionDeniedError as exc:
            raise LLMError(f"permission denied: {exc}") from None
        except anthropic.NotFoundError:
            raise LLMError(f"model '{kwargs['model']}' not found") from None
        except anthropic.RateLimitError:
            raise LLMError("rate limited by the Anthropic API (after retries)") from None
        except anthropic.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}: {exc.message}") from None
        except anthropic.APIConnectionError:
            raise LLMError("could not reach the Anthropic API") from None
