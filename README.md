# HIVEMIND

A self-improving swarm of Claude agents whose single objective is finding intraday futures strategies (Nasdaq, S&P 500, gold, and more) with the highest probability of passing prop-firm challenges such as the Topstep 50K Trading Combine. A live dashboard shows it working.

```
                       ┌──────────────────────────────┐
   your orders  ───▶   │  OVERMIND  (Claude Opus 5.5) │  spawns / directs / deletes drones
                       └──────────────┬───────────────┘  deleted itself if the hive stagnates
                                      │ directive + missions
          ┌───────────────┬───────────┴───────┬───────────────┐
          ▼               ▼                   ▼               ▼
      DRONE  …        DRONE  …            DRONE  …        EVOLVER (genetic algorithm,
   designs 5 strategies per round, gets results back,     thousands of mutants / hour)
   3 rounds without improvement = permanently deleted
          │               │                   │               │
          └───────────────┴─────────┬─────────┴───────────────┘
                                    ▼
             EVALUATOR (multi-process)  ── backtest on intraday bars
                                        ── replay day-by-day P&L through the firm's rules
                                        ── Monte Carlo pass probability, train/val/holdout
                                    ▼
             HALL OF FAME + HIVE MEMORY (SQLite)  ──▶  live dashboard (websocket)
                                    ▼
             ALPACA PAPER  ── forward-test the champion, virtual challenge on real fills
```

## Quick start

```bash
git clone https://github.com/eliascroesus/aitrading2.git
cd aitrading2
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # add ANTHROPIC_API_KEY + Alpaca paper keys
python -m hive
```

Open **http://127.0.0.1:8000**. Python 3.10+.

* **No keys at all:** runs on synthetic demo data with procedural agents. Use this to see the machinery. Results on synthetic data mean nothing.
* **`ANTHROPIC_API_KEY`:** the Overmind and drones become Claude agents.
* **`ALPACA_API_KEY` / `ALPACA_SECRET_KEY` (paper):** real market history from Alpaca, plus one-click forward testing of any strategy on your paper account.

## What you see on the dashboard

| Panel | What it shows |
|---|---|
| KPIs | Strategies tested and test rate, best validation pass probability, **holdout** pass rate (the honest number), expected attempts and fees to pass, living and deleted agents, Overmind generation and stagnation, Claude spend |
| Evolution | Every tested strategy as a dot, plus step lines for best fitness and the champion's validation and holdout pass rates over time |
| The Hive | The Overmind and each drone: status (thinking / backtesting), mission, rounds, best fitness, lives left, latest reasoning. ✕ deletes an agent. |
| Leaderboard | Top strategies, sortable; "best per niche" shows one per family/market. Click a row for the full report: segment table, equity curve with train/val/holdout shading, attempt outcomes (pass / max loss / daily limit / timeout), exact rules, deploy and export. |
| Live feed | Champions, spawns, deletions, Overmind directives, drone thoughts, memory, paper trades |
| Alpaca paper | Account, deployed strategy, virtual challenge progress bar (floor → balance → target), trades |
| Hive memory / Graveyard | Lessons the agents share; deleted agents and why they were deleted |
| Command bar | Give the Overmind an order. It is treated as highest priority in the next cycle. |

## How a strategy is judged

1. **Strategy = JSON spec.** It holds long/short entry expressions, an optional exit expression, a stop (ATR / % / points / swing), a target (R / ATR / % / points / none), breakeven, max hold, session window, flat time, risk per trade, trades per day, and daily stop/target. The expression language (`hive/lang.py`) covers price/volume, VWAP, sessions and opening ranges, prior day/week levels, ICT/SMC primitives (`sweep_pdl()`, `sweep_or_high(30)`, `fvg_bull()`, `in_bull_fvg(6)`, `displacement(1.5)`, `bos_up(20)`), indicators (EMA, RSI, ADX, MACD, Bollinger, ATR…) and candles. Expressions are parsed against a whitelist and evaluated vectorised. Every function only uses past bars, and the tests check this.
2. **Backtest** (`hive/backtest.py`). Signal on bar close, fill next open plus 1 tick, intrabar stop/target with stop-first on ties, gaps fill at the open, commissions, flat by `flat_at`, one position at a time. Size = `risk_usd / (stop distance × point value)`, capped by the firm's contract limit.
3. **Challenge simulation** (`hive/propfirm.py`). Each day's P&L, and the intraday equity low/high including open-trade drawdown, is replayed through the rulebook: profit target, max loss (static / end-of-day trailing / intraday trailing), daily loss limit, consistency rule, minimum days and a time limit. Pass probability comes from a 5-day block-bootstrap Monte Carlo and from replaying every possible start day.
4. **Fitness** = geometric mean of the pass rates on three train folds and the validation segment, times a trade-count factor. A strategy has to pass in every period, not just one lucky stretch. The last 20% of history (the **holdout**) is never used for selection; it is the number to trust.

Presets: `topstep_50k/100k/150k`, `apex_50k/100k`, `tradeify_50k`, `ftmo_100k`, or `custom` (see `.env.example`). Firms change their rules often: check the current rulebook and edit the preset.

## Natural selection

* A drone that fails to beat its own best fitness for `HIVE_DRONE_MAX_STRIKES` rounds is permanently deleted. Its strategies stay in the Hall of Fame; the agent does not come back.
* The Overmind can terminate drones and spawn new ones with new missions at any time.
* If the hive's best fitness does not improve for `HIVE_OVERMIND_PATIENCE` Overmind cycles, the Overmind itself is deleted. A new one takes over with only the hive memory and a post-mortem of its predecessor.
* The genetic evolver never stops. Its explore/exploit balance is set by the Overmind.

## Alpaca paper trading

Alpaca does not offer futures, so the deployed strategy trades the ETF proxy: QQQ for MNQ/NQ, SPY for MES/ES, GLD for MGC/GC. Shares are sized so one contract equals `proxy_ratio × point_value` shares, which gives the same dollars per point. Orders are bracket orders (market entry plus stop and target legs). The dashboard tracks a virtual challenge on account equity measured from the moment you deploy.

* Use a **dedicated paper account**: "stop & flatten" closes every position on it.
* `PAPER_SIZE_SCALE=0.1` trades a tenth of the size and rescales the challenge P&L.
* `ALPACA_PAPER=true` is the default. Setting it to false sends real orders.

## Cost and speed

* **Backtesting** runs on your CPU: roughly 2,000 strategies per minute per core on 5-minute data.
* **Claude spend** is shown live; `HIVE_LLM_BUDGET_USD` pauses the agents when the budget is reached. On Opus 5.5 ($4 in / $20 out per million tokens) one drone round costs roughly $0.05–0.20, mostly output and thinking tokens. With 4 drones on a 20 s cooldown, expect tens of dollars per hour. `HIVE_DRONE_MODEL=claude-sonnet-5-5` roughly halves the drone cost, and `claude-haiku-5-5` is ~40× cheaper. You can also raise `HIVE_DRONE_COOLDOWN_SEC` or lower `HIVE_MAX_DRONES`.

## Honest limits

* **Proxy data:** ETF bars cover only US regular hours and approximate futures. Overnight sessions, the London open and roll effects are not in the data. `DATA_SOURCE=yahoo` gives real continuous futures, but only about 60 days of 5-minute history.
* **Overfitting:** searching thousands of strategies finds patterns in noise. A large gap between validation and holdout pass rates is the warning sign. Forward-test on paper before paying for a challenge.
* **Fills:** the fill model is conservative, but it is still a model.
* **Rule details:** firm rule details (how the daily loss limit, trailing drawdown and consistency are computed) differ from the presets in small ways.

## Files

| File | Role |
|---|---|
| `hive/lang.py` | expression language (parser, whitelist, vectorised indicators, ICT/SMC primitives) |
| `hive/backtest.py` | intraday backtester |
| `hive/propfirm.py` | challenge rules, presets, Monte Carlo and historical pass probability |
| `hive/evaluator.py` | worker-process evaluation and fitness |
| `hive/strategy.py`, `hive/seeds.py` | strategy spec, mutation and crossover, research-based seeds, procedural generator |
| `hive/agents.py`, `hive/llm.py` | agent prompts and schemas; Anthropic SDK wrapper (structured JSON output, prompt caching, fallbacks, cost tracking) |
| `hive/hive.py` | orchestrator: evolver, drones, Overmind, natural selection, state |
| `hive/broker.py` | Alpaca paper trader and virtual challenge |
| `hive/server.py`, `hive/static/` | FastAPI + websocket server and the dashboard |
| `tests/` | `pytest tests` (no network needed; Claude is mocked) |

## Research

`docs/trading-strategies-and-top-traders.md` is the research report the seed strategies come from: 100+ strategy concepts (ICT/SMC, CRT, price action, indicators, order flow, sessions, quant) explained for Forex, NQ/ES and gold, ~40 trader profiles, and the evidence on what actually works.
