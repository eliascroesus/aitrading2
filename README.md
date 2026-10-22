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

## Run it for free in GitHub Codespaces (easiest, nothing to install)

1. **Save your keys on GitHub (once):** github.com → your profile picture → **Settings → Codespaces → Secrets → New secret**. Add each of these and give it access to the `aitrading2` repository:
   `ANTHROPIC_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, and optionally `HIVE_LLM_BUDGET_USD` (e.g. `20`).
2. On the repo page click **Code → Codespaces → Create codespace on main**.
3. Wait a couple of minutes. It installs everything, starts the hive, and opens the dashboard in a new tab. If the tab doesn't open, go to the **Ports** tab and click the globe icon next to port 8000.

Free GitHub accounts get about 60 hours a month on this 2-core machine. A codespace stops after a period of inactivity: raise **Settings → Codespaces → Default idle timeout** to 240 minutes, and stop it from github.com/codespaces when you're done. Your data stays in the codespace between sessions. The forwarded port is private to your GitHub login, so no dashboard password is needed.

## Hugging Face Spaces (stays on; Docker Spaces now need a paid plan)

As of late 2026 Hugging Face only offers Docker Spaces on paid plans, so this is no longer free. The setup still works if you upgrade:

1. Create a free account at **huggingface.co**, then **New Space**: pick **Docker** as the SDK, the **Blank** template, the free **CPU basic** hardware (2 vCPU, 16 GB RAM), and set it to **Private**. Note its name, e.g. `yourname/hivemind`.
2. In the Space **Settings → Variables and secrets**, add the secrets `ANTHROPIC_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `DASHBOARD_PASSWORD` and optionally `HIVE_LLM_BUDGET_USD`.
3. On huggingface.co go to **Settings → Access Tokens** and create a token with **Write** access.
4. On this GitHub repo go to **Settings → Secrets and variables → Actions**:
   * **Secrets** tab, new secret `HF_TOKEN` = the token from step 3
   * **Variables** tab, new variable `HF_SPACE` = your Space name, e.g. `yourname/hivemind`
5. Go to **Actions → Deploy to Hugging Face Space → Run workflow**. From then on, every push to `main` redeploys automatically.
6. Open `https://yourname-hivemind.hf.space` (your Space's direct link) and log in with your `DASHBOARD_PASSWORD`.

Free Spaces go to sleep after about 48 hours without visitors and restart on the next visit. Free Spaces have no persistent storage, so a restart begins a fresh hive. Export strategies you want to keep from the strategy drawer.

## Paid alternative: Railway / Render

The `Dockerfile` works on any Docker host. Set the same environment variables plus `DASHBOARD_PASSWORD`, attach a persistent volume at `/data`, and expose port 8000. Railway and Render both deploy straight from this GitHub repo.

## Run it on your own computer (double-click)

1. Install **Python 3.10+** from python.org. On Windows, tick **"Add python.exe to PATH"**.
2. On this repo's GitHub page: **Code → Download ZIP**, then unzip it.
3. Open the folder and double-click:
   * **Mac:** `start-mac.command`. If macOS blocks it, right-click it, choose **Open**, then **Open** again. If that's not offered, use **System Settings → Privacy & Security → Open Anyway**.
   * **Windows:** `start-windows.bat`. If SmartScreen warns you, click **More info → Run anyway**.
4. The first time, it asks for your Anthropic and Alpaca paper keys (paste and press Enter) and a spending cap. It saves them to `.env` in the folder, installs everything into `.venv` in the folder, and opens the dashboard at http://127.0.0.1:8000.

Keep the window open while it runs; close it (or press Ctrl+C) to stop. Next time, just double-click again. It uses half your CPU cores (change with `HIVE_WORKERS` in `.env`) and never uses the GPU.

### Removing it completely

Everything HIVEMIND creates lives inside its folder: the app, its Python packages (`.venv`, about 250 MB), the database and market data (`hive_data`, capped at a few hundred MB) and your keys (`.env`). To remove it all:

1. Close the HIVEMIND window.
2. Delete the folder and empty the Trash / Recycle Bin.

Optionally uninstall Python too (Windows: Settings → Apps → Python; Mac: delete the Python folder in Applications), and delete or regenerate the API keys in the Anthropic and Alpaca dashboards.

### From a terminal instead

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # add ANTHROPIC_API_KEY + Alpaca paper keys
python -m hive
```

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

## Reading the results

| Number | What it means | Good |
|---|---|---|
| **Holdout ≤7d** | Share of challenge attempts passed within the target days (`PROP_TARGET_DAYS`, default 7) on the newest 20% of history, which the hive never selects on. This is the honest headline number. | the higher the better |
| **Holdout 30d** | Share passed at all within a 30-day attempt. | |
| **Luck** | Pass rate the same risk would get with zero edge, from volatility alone. | |
| **Holdout − Luck** | The part of the pass rate that comes from real edge. | under 5 points = no edge; 10–20 = promising; 25+ = strong |
| **Val pass vs Holdout** | A big gap means the strategy is partly memorising history. | gap under ~10 points |
| **✓ Score** | The strategy kept performing when every setting was nudged (robustness check). | ✓ present |
| **≤10d / Days** | Share of attempts passed within 10 trading days / median days to pass. | depends on your goal |

Before paying for a real challenge, deploy the candidate to Alpaca paper and let the virtual challenge run for a few weeks.

**Goal: pass in 7 trading days.** By default the hive optimises for passing within `PROP_TARGET_DAYS=7` trading days, which means making about $430+ a day on average without touching the $2,000 trailing drawdown. Fast and reliable pull against each other: the shorter the target, the more daily swing it takes, and the more often an attempt fails. Set `PROP_TARGET_DAYS` to another number (or empty for the whole 30-day attempt) to change the goal.

**Strategy combos.** Every ~45 seconds the hive combines leading strategies on *different* instruments into one account (for example a Nasdaq strategy plus a gold strategy), sizes them together, and scores the combo on the same 7-day challenge. Combos help when their members make money on different days (low correlation). They appear in their own panel and can be paper traded together with one click; each member trades its own ETF.

## How a strategy is judged

1. **Strategy = JSON spec.** It holds long/short entry expressions, an optional exit expression, a stop (ATR / % / points / swing), a target (R / ATR / % / points / none), breakeven, max hold, session window, flat time, risk per trade, trades per day, and daily stop/target. The expression language (`hive/lang.py`) covers price/volume, VWAP, sessions and opening ranges, prior day/week levels, ICT/SMC primitives (`sweep_pdl()`, `sweep_or_high(30)`, `fvg_bull()`, `in_bull_fvg(6)`, `displacement(1.5)`, `bos_up(20)`), indicators (EMA, RSI, ADX, MACD, Bollinger, ATR…) and candles. Expressions are parsed against a whitelist and evaluated vectorised. Every function only uses past bars, and the tests check this.
2. **Backtest** (`hive/backtest.py`). Signal on bar close, fill next open plus 1 tick, intrabar stop/target with stop-first on ties, gaps fill at the open, commissions, flat by `flat_at`, one position at a time. Size = `risk_usd / (stop distance × point value)`, capped by the firm's contract limit.
3. **Challenge simulation** (`hive/propfirm.py`). Each day's P&L, and the intraday equity low/high including open-trade drawdown, is replayed through the rulebook: profit target, max loss (static / end-of-day trailing / intraday trailing), daily loss limit, consistency rule, minimum days and a time limit. Pass probability comes from a 5-day block-bootstrap Monte Carlo and from replaying every possible start day.
4. **Fitness** = geometric mean of the pass rates on three train folds and the validation segment (pass within `PROP_TARGET_DAYS` when set), times a trade-count factor and a small penalty for rules with many conditions. A strategy has to pass in every period, not just one lucky stretch.
5. **Robustness check.** Every strategy that could reach the top 30 is re-run as 6 variants with every number nudged (periods ±10–15%, thresholds ±5–10%, times ±5–10 min, stop/target/risk ±10%). Its score becomes the average of its own fitness and the variants' median, and unchecked strategies rank at 85%. Knife-edge settings lose.
6. **Holdout and luck.** The last 20% of history is never used for selection. It is shown next to the zero-edge **luck baseline**, so you can see how much of a pass rate is real edge.

The evolver keeps several strategy families evolving at once (half its parents come from per-niche elites), and 5 years of history (`HISTORY_DAYS`) make strategies survive several market regimes.

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
