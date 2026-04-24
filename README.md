# Wealth-First Investing

This repository implements a wealth-first optimization workflow for trading and portfolio allocation. The objective is simple: maximize step-by-step gains when wealth grows, and apply a heavier penalty when wealth falls.

## Service Structure

### 1. Advanced Trading Education

- Focus: institutional-grade market structure, liquidity, and price action.
- Delivery: quant-informed discretionary execution and real-world strategy frameworks.
- Outcome: traders build self-sufficient processes and their own edge.

### 2. Non-Custodial Strategy Access

- Focus: professional strategy access without giving up custody.
- Delivery: copy trading or API-linked execution inside the client's own exchange account.
- Advantage: zero custody risk, no withdrawal rights, transparent execution, aligned incentives.
- Outcome: passive exposure to systematic strategies with full asset ownership.

### 3. Managed Portfolio

- Focus: multi-strategy capital allocation for more stable performance.
- Delivery: trend-following, mean reversion, market-neutral overlays, and active risk management.
- Advantage: portfolio-level optimization instead of single-strategy concentration.
- Outcome: institutional-style risk-adjusted management with stronger drawdown control.

## Wealth-First Logic

For each step in time, the optimizer evaluates the wealth ratio:

`wealth_ratio_t = W(t+1) / W(t) = 1 + portfolio_return_t`

The default piecewise objective is:

`score = -sum(max(wealth_ratio_t - 1, 0)^gain_power) + loss_penalty * sum(max(1 - wealth_ratio_t, 0)^loss_power)`

This produces asymmetric behavior:

- Gains are rewarded.
- Losses are penalized more heavily.
- Rebalancing can be regularized with turnover and weight penalties computed on tradable sleeves, without double-counting the synthetic `CASH` sleeve.

The package also supports a `log_wealth` mode for terminal-growth style optimization.

## Execution Realism

The backtest now models two additional realities:

- Weight drift between rebalance points.
- Trading frictions through `transaction_cost_bps` and `slippage_bps` applied to tradable notional.

This means reported returns are net of a configurable execution-cost model instead of assuming frictionless rebalancing.

The optimizer now uses that same tradable-turnover definition when applying turnover regularization and explicit execution-cost penalties, so allocations into or out of synthetic `CASH` are not overstated during optimization.

Synthetic `CASH` is treated as residual capital, not as a venue order. Trading costs are applied to changes in tradable sleeves, and the execution plan reports target cash separately instead of creating a fake cash order.

The optimizer also supports per-sleeve allocation bounds, so you can cap or floor individual sleeves such as `TREND_FOLLOWING`, `HEDGE_OVERLAY`, or `CASH` without forcing the same range on the entire universe.

When a benchmark stream is available, the optimizer can also score relative wealth versus that benchmark. This lets you penalize benchmark-relative lag while preserving the original absolute wealth objective.

## Why SPY Uses A Cash Sleeve

Testing a single risky asset under a long-only, fully-invested constraint is otherwise trivial. To make SPY optimization meaningful, the code automatically adds a synthetic `CASH` sleeve with zero return by default. The optimizer can then decide how much capital to hold in SPY versus cash at each rebalance.

## Project Layout

- `src/wealth_first/optimizer.py`: objective function and constrained optimizer.
- `src/wealth_first/backtest.py`: rolling walk-forward backtest utilities.
- `src/wealth_first/data.py`: market data download and return preparation.
- `src/wealth_first/cli.py`: command-line entry point for SPY or multi-asset runs.
- `tests/`: offline unit tests using synthetic data.

## Quick Start

Install the package:

```bash
/opt/homebrew/bin/python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

Run the SPY demo:

```bash
.venv/bin/python -m wealth_first.cli --symbols SPY --start 2018-01-01 --lookback 63
```

Or use the convenience launcher:

```bash
.venv/bin/python scripts/run_spy_demo.py
```

## Frontend Deployment

The React board in `frontend/` is now deployment-ready for a hosted frontend that talks to a separately hosted bridge backend.

The frontend reads `import.meta.env.VITE_API_BASE_URL` when that variable is present. If it is not set, the app falls back to same-origin `/api` requests, which is the correct behavior when the frontend and API are served from the same public domain.

For local development, keep using the Vite proxy or set a local env file when you want the browser to talk to a specific bridge origin directly:

```bash
cd frontend
cp .env.development.example .env.development.local
```

Set `VITE_API_BASE_URL` to the public bridge origin, for example:

```bash
VITE_API_BASE_URL=https://your-bridge-host.example.com
```

The bridge backend must allow that frontend origin. For a stable production URL:

```bash
.venv/bin/python -m wealth_first.tradingview_bridge serve \
  --host 0.0.0.0 \
  --port 8000 \
  --allow-origin https://wealth-first-advisor.vercel.app
```

For Vercel preview deployments, use a regex instead of listing every preview hostname:

```bash
.venv/bin/python -m wealth_first.tradingview_bridge serve \
  --host 0.0.0.0 \
  --port 8000 \
  --allow-origin-regex 'https://wealth-first-advisor-.*\.vercel\.app'
```

## Vercel + GitHub

The repository now includes a root `vercel.json` that builds the Vite frontend from `frontend/` and serves `frontend/dist`.

One-time setup:

1. Push this repository to GitHub.
2. Import the GitHub repository into Vercel.
3. In Vercel Project Settings -> Environment Variables, set `VITE_API_BASE_URL` for both Preview and Production.
4. Use the production bridge URL for Production deployments and a staging or preview bridge URL for Preview deployments.
5. If the API is served from the same public origin as the frontend, leave `VITE_API_BASE_URL` unset so the app keeps using same-origin `/api` requests.
6. Redeploy after changing environment variables so Vite rebuilds with the new values.
7. After the first import, every push to the connected branch will trigger an automatic Vercel deployment.

Important constraint: the current FastAPI bridge is stateful because it maintains live webhook, normalization, and execution worker state, so it should stay on a long-lived host such as a VM, container service, or similar backend runtime. Vercel is a good fit for the frontend, not for the current backend worker.

## Backend Deployment

The bridge backend now has an environment-driven runtime entrypoint for long-lived hosts:

```bash
python -m wealth_first.tradingview_bridge serve-env
```

`serve-env` reads deployment settings from environment variables instead of requiring a long CLI command. The main ones are:

- `HOST` or `WEALTH_FIRST_HOST`: bind host. Defaults to `0.0.0.0`.
- `PORT` or `WEALTH_FIRST_PORT`: bind port. Defaults to `8000`.
- `WEALTH_FIRST_EVENT_LOG_PATH`
- `WEALTH_FIRST_OUTPUT_CSV_PATH`
- `WEALTH_FIRST_EXECUTION_LOG_PATH`
- `WEALTH_FIRST_FAILURE_LOG_PATH`
- `WEALTH_FIRST_ARTIFACT_ROOT_PATH`
- `WEALTH_FIRST_ALLOWED_ORIGINS`: comma-separated allowed frontend origins.
- `WEALTH_FIRST_ALLOWED_ORIGIN_REGEX`: preview-origin regex, useful for Vercel preview builds.

The repository also now includes:

- `Dockerfile`: containerized bridge runtime with optional RL research extras still available for future experiments.
- `scripts/start_bridge_runtime.sh`: seeds the persistent artifact volume with the baked-in baseline artifacts and starts `serve-env`.
- `render.yaml`: a Render blueprint that mounts persistent storage at `/var/data` and exposes `/healthz`.

Recommended deployment shape:

1. Deploy the backend container to a long-lived host such as Render, Railway, Fly.io, or a VM.
2. Mount persistent storage at `/var/data` so webhook logs, normalized returns, and generated artifacts survive restarts.
3. Set `WEALTH_FIRST_ALLOWED_ORIGINS` to your Vercel production frontend URL.
4. Set `WEALTH_FIRST_ALLOWED_ORIGIN_REGEX` to match Vercel preview URLs when you want preview builds to call the live bridge.
5. Set `VITE_API_BASE_URL` in Vercel to the public backend URL.

### Render Steps

If you want the fastest concrete backend path, use Render with the included blueprint:

1. In Render, create a new Blueprint and point it at this GitHub repository.
2. Accept the generated `wealth-first-bridge` web service and persistent disk mounted at `/var/data`.
3. Add `WEALTH_FIRST_WEBHOOK_TOKEN` as a secret before the first deploy.
4. If your frontend uses a custom domain instead of `wealth-first-advisor.vercel.app`, update `WEALTH_FIRST_ALLOWED_ORIGINS` and `WEALTH_FIRST_ALLOWED_ORIGIN_REGEX` in Render.
5. Deploy and confirm the service responds on `/healthz`.
6. Copy the Render service URL into Vercel as `VITE_API_BASE_URL` and redeploy the frontend.

The committed `render.yaml` already pins the persistent runtime paths used by the bridge (`/var/data/tradingview_events.jsonl`, `/var/data/tradingview_truth.csv`, `/var/data/artifacts`).

## Example Configuration

```bash
.venv/bin/python -m wealth_first.cli \
  --symbols SPY \
  --start 2015-01-01 \
  --lookback 63 \
  --rebalance-frequency 5 \
  --loss-penalty 10 \
  --loss-power 1.75 \
  --transaction-cost-bps 5 \
  --slippage-bps 5 \
  --turnover-penalty 0.002
```

Add per-sleeve bounds when you need allocation guardrails for specific sleeves:

```bash
.venv/bin/python -m wealth_first.cli \
  --returns-csv data/demo_sleeves.csv \
  --benchmark-column SPY_BENCHMARK \
  --exclude-benchmark-from-universe \
  --lookback 63 \
  --rebalance-frequency 5 \
  --weight-bound TREND_FOLLOWING=0.00:0.50 \
  --weight-bound MEAN_REVERSION=0.00:0.35 \
  --weight-bound HEDGE_OVERLAY=0.00:0.25 \
  --weight-bound CASH=0.10:0.80
```

Per-sleeve bounds are enforced when the optimizer generates target allocations. Market drift can move `ending_weights` outside those ranges between rebalance points, so use the saved `weights.csv` for target-allocation checks and `ending_weights.csv` for post-return drift analysis.

Add benchmark-relative pressure when you want the optimizer to care about alpha, not only absolute wealth preservation:

```bash
.venv/bin/python -m wealth_first.cli \
  --returns-csv data/demo_sleeves.csv \
  --benchmark-column SPY_BENCHMARK \
  --exclude-benchmark-from-universe \
  --lookback 63 \
  --rebalance-frequency 5 \
  --benchmark-gain-reward 0.25 \
  --benchmark-loss-penalty 6.0
```

When a benchmark is present, the backtest summary now reports benchmark total return, relative total return, average active return, tracking error, and information ratio.

Run the offline tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Run the locked `main3` reproducibility suite and drift gate:

```bash
make repro-gate
```

This runs the three locked A/G/I cases, rewrites the suite summary artifacts, and fails if the new results drift beyond the tolerances stored in `artifacts/main3_repro_baseline_locked.json`.

If you only want to refresh the raw repro outputs without enforcing the gate:

```bash
make repro-suite
```

Run the lightweight optimization research audit when you want a current read on where further upside is most realistic:

```bash
make research-audit
```

This reruns the `main3` structural diagnosis, runs the current `main4` stress suite, and writes a summarized investigation report to `artifacts/optimization_research_audit.md`.

Install the optional RL research dependency when you want a Gymnasium environment for policy experiments:

```bash
.venv/bin/python -m pip install -e '.[rl]'
```

That optional extra now includes both Gymnasium and Stable-Baselines3 for the PPO research baseline.

## Execution Adapter Scaffold

For Service 2, use `wealth_first.execution` to convert target weights into an execution plan.

```python
import pandas as pd

from wealth_first.execution import PaperExchangeAdapter, build_execution_plan

target_weights = pd.Series({"SPY": 0.35, "CASH": 0.65})
current_weights = pd.Series({"SPY": 0.60, "CASH": 0.40})

plan = build_execution_plan(
    target_weights=target_weights,
    current_weights=current_weights,
    equity=100_000,
    min_trade_weight=0.01,
)

response = PaperExchangeAdapter().submit_allocation(plan)
print(response)
```

`WebhookExchangeAdapter` is included as a generic HTTP handoff layer for a broker, exchange bridge, or internal execution service.

## Multi-Sleeve Workflow

You can optimize precomputed sleeve returns directly from CSV.

Build the demo sleeves:

```bash
.venv/bin/python scripts/build_demo_sleeves.py --symbol SPY --start 2010-01-01 --output data/demo_sleeves.csv
```

Run the optimizer on those sleeves while keeping SPY as a benchmark only:

```bash
.venv/bin/python -m wealth_first.cli \
  --returns-csv data/demo_sleeves.csv \
  --benchmark-column SPY_BENCHMARK \
  --exclude-benchmark-from-universe \
  --lookback 63 \
  --rebalance-frequency 5 \
  --transaction-cost-bps 5 \
  --slippage-bps 5
```

The demo CSV contains three example sleeves derived from SPY:

- `TREND_FOLLOWING`: long SPY only when medium-term trend is positive.
- `MEAN_REVERSION`: long SPY after a short-term selloff.
- `HEDGE_OVERLAY`: a synthetic hedge sleeve that becomes active during higher-volatility regimes.

These sleeves are demonstrative research inputs, not production trading strategies.

## RL Research Scaffold

The repo now includes a research-only Gymnasium environment in `wealth_first.rl`.

```python
import pandas as pd

from wealth_first.optimizer import WealthFirstConfig
from wealth_first.rl import WealthFirstEnv

returns = pd.read_csv("data/demo_sleeves.csv", index_col=0, parse_dates=True)
benchmark = returns["SPY_BENCHMARK"]
investable_returns = returns.drop(columns=["SPY_BENCHMARK"])

env = WealthFirstEnv(
  investable_returns,
  lookback=20,
  benchmark_returns=benchmark,
  config=WealthFirstConfig(
    include_cash=True,
    transaction_cost_bps=5.0,
    slippage_bps=5.0,
    benchmark_loss_penalty=6.0,
    min_weight_overrides={"CASH": 0.10},
    max_weight_overrides={"HEDGE_OVERLAY": 0.25},
  ),
)
```

The environment reuses the existing turnover, cost, cash-sleeve, and per-sleeve bound logic. Actions are projected onto the feasible simplex before each step, so research policies cannot bypass the same allocation guardrails used by the optimizer.

For a first training baseline, use the built-in PPO runner. It trains on randomized chronological windows from the training split, evaluates on a held-out tail slice, and can optionally normalize observations and rewards through Stable-Baselines3 `VecNormalize`.

For the current SPY-only medium-capacity path, use `main4` and its repro and stress runners instead. They rebuild features from `SPY_BENCHMARK`, evaluate the bounded two-stage policy across walk-forward folds, and write comparison-ready detail and summary artifacts.

```bash
.venv/bin/python scripts/run_main4_repro_suite.py
.venv/bin/python scripts/run_main4_stress_suite.py
```

For this repo's PPO research, treat `gate4 + earlycap before=2 + p005strict + --normalize-observations` as the frozen live baseline. New PPO work should compare targeted branches against that reference instead of reopening generic tuning sweeps. For a reproducible runner, use `scripts/run_ppo_frozen_live_baseline.sh` and override `SEED`, `SPLIT_METHOD`, `OUTPUT_DIR`, or `PYTHON_BIN` through the environment when needed. To summarize saved frozen-baseline artifacts across regime-balanced and chronological runs, use `scripts/evaluate_frozen_ppo_baseline.py`.

Example evaluator command for the current promoted artifact family:

```bash
.venv/bin/python scripts/evaluate_frozen_ppo_baseline.py \
  --regime-prefix ppo_guardrail_t55c25_b25_b002_tc10_gate4_earlycap04_b2_latedefpen005_a2_c18_t64_normobs_walk_2048_seed_ \
  --chrono-prefix ppo_guardrail_t55c25_b25_b002_tc10_chrono_gate4_earlycap04_b2_latedefpen005_a2_c18_t64_normobs_walk_2048_seed_ \
  --output-prefix artifacts/ppo_frozen_live_baseline_eval
```

For a narrower fold/phase postmortem across seeds, use `scripts/postmortem_frozen_ppo_window.py`.

The frozen baseline runner also exposes the early-risk controls through environment overrides. Setting `EARLY_RISK_TURNOVER_CAP=0` on `scripts/run_ppo_frozen_live_baseline.sh` fully blocks early low-cash / high-trend trades that would otherwise push further into trend, which is the narrow lever for the remaining chrono fold-01 issue.

For runtime overlay experiments, keep the current frozen baseline unchanged by default. The only overlay that remained directionally useful in exact replay was a regime-only deep-drawdown mode: apply the trigger `benchmark_regime_drawdown <= -0.10 or pre_trade_mean_reversion >= 0.186` only on the regime-balanced artifact family. On the current active frozen baseline families this lifted mean total return by about `+0.000401` with chrono unchanged and regime about `+0.000803`, but it still left the same residual chrono static-hold-negative rows untouched. Treat that overlay as an optional regime-helping mode, not as the promoted baseline. To replay that mode through the repo evaluator, run `scripts/evaluate_frozen_ppo_baseline.py` with `--runtime-overlay robust-regime-only`.

The runner now supports a data-driven `regime-balanced` split mode that chooses contiguous train, validation, and test windows by searching for chronological boundaries whose volatility, drawdown, and trend regime mix best matches the full sample while preserving time order.

When benchmark data is available, you can also add `--benchmark-relative-observations` to append sleeve-minus-benchmark lookback features to the PPO observation. This is useful when you want the policy to see not just raw sleeve returns, but how each sleeve is behaving versus the benchmark regime it is being judged against.

When you want a more compact regime signal instead of another full lookback block, add `--benchmark-regime-observations`. This appends benchmark cumulative return, benchmark volatility, benchmark drawdown, and each sleeve's cumulative return relative to the benchmark over the lookback window.

That compact regime branch is not promoted over the frozen `gate4 + earlycap before=2 + p005strict + --normalize-observations` baseline. The benchmark-summary trend-preservation gate was retired because it fired too rarely to explain branch behavior.

If you revisit under-trend PPO behavior later, use the state-driven trend-preservation guard instead. It keys off portfolio state that occurs much more often: low pre-trade cash, high pre-trade trend weight, and a projected cash-raising cut in the designated trend sleeve.

One exploratory command for that guard looks like this:

```bash
.venv/bin/python -m wealth_first.train_ppo \
  --returns-csv data/demo_sleeves.csv \
  --benchmark-column SPY_BENCHMARK \
  --exclude-benchmark-from-universe \
  --split-method regime-balanced \
  --lookback 20 \
  --episode-length 126 \
  --benchmark-loss-penalty 6.0 \
  --transaction-cost-bps 5 \
  --slippage-bps 5 \
  --weight-bound CASH=0.10:0.80 \
  --weight-bound HEDGE_OVERLAY=0.00:0.25 \
  --normalize-observations \
  --state-trend-preservation-symbol TREND_FOLLOWING \
  --state-trend-preservation-cash-max-threshold 0.22 \
  --state-trend-preservation-symbol-min-weight 0.62 \
  --state-trend-preservation-max-symbol-reduction 0.02 \
  --normalize-rewards \
  --total-timesteps 20000 \
  --output-dir artifacts/ppo_baseline
```

If you want to inspect the chosen date windows before training, run the same command with `--split-only`. The runner writes `split_windows.json` to the output directory, including start and end dates for train, validation, and test, along with regime coverage and split score diagnostics.

If you want to check stability across more than one holdout, add anchored walk-forward folds. The runner will train one PPO model per fold, keep validation and test separate inside each fold, and write aggregate metrics at the top level.

```bash
.venv/bin/python -m wealth_first.train_ppo \
  --returns-csv data/demo_sleeves.csv \
  --benchmark-column SPY_BENCHMARK \
  --exclude-benchmark-from-universe \
  --split-method regime-balanced \
  --walk-forward-folds 3 \
  --walk-forward-step-rows 285 \
  --lookback 20 \
  --episode-length 126 \
  --benchmark-loss-penalty 6.0 \
  --total-timesteps 2000 \
  --output-dir artifacts/ppo_walk_forward
```

Each fold writes its own `metrics.json`, model artifact, and `split_windows.json`. The top-level output directory contains aggregate validation and test reward summaries across folds.

If PPO is still churning too much, add explicit anti-churn controls during both training and evaluation:

```bash
.venv/bin/python -m wealth_first.train_ppo \
  --returns-csv data/demo_sleeves.csv \
  --benchmark-column SPY_BENCHMARK \
  --exclude-benchmark-from-universe \
  --split-method regime-balanced \
  --walk-forward-folds 3 \
  --lookback 20 \
  --episode-length 126 \
  --action-smoothing 0.5 \
  --no-trade-band 0.05 \
  --total-timesteps 2000 \
  --output-dir artifacts/ppo_walk_forward_smoothed
```

`--action-smoothing` blends the proposed action back toward current weights, while `--no-trade-band` skips a rebalance when projected tradable turnover is too small to justify the friction.

If the PPO policy is still staying too defensive, add an explicit excess-cash penalty. `--cash-weight-penalty` penalizes cash above a target, and `--cash-target-weight` lets you move that target away from the cash floor when you want to test a tighter reserve.

```bash
.venv/bin/python -m wealth_first.train_ppo \
  --returns-csv data/demo_sleeves.csv \
  --benchmark-column SPY_BENCHMARK \
  --exclude-benchmark-from-universe \
  --split-method regime-balanced \
  --lookback 20 \
  --episode-length 126 \
  --action-smoothing 0.5 \
  --no-trade-band 0.05 \
  --cash-weight-penalty 0.005 \
  --cash-target-weight 0.15 \
  --total-timesteps 2000 \
  --output-dir artifacts/ppo_cash_probe
```

Each trained fold now also writes:

- `validation_policy_rollout.csv` and `test_policy_rollout.csv` with step-by-step PPO actions, proposed and executed weights, per-sleeve realized returns, gross return contributions, turnover, trade suppression flags, reward decomposition, costs, wealth, and target weights.
- `validation_optimizer_summary.json` and `test_optimizer_summary.json` with the classical optimizer backtest on the exact same windows.
- `policy_vs_optimizer_comparison.json` with common metrics such as total return, drawdown, turnover, and benchmark-relative performance aligned side by side.

For multi-fold runs, the top-level `metrics.json` now includes mean policy summaries, mean optimizer summaries, and a mean `policy_vs_optimizer_comparison.json` aggregated across folds.

The PPO policy summaries now also report diagnostics such as `average_excess_cash_weight` and `average_cash_weight_penalty_component`, which are useful when a policy still trails the optimizer because it is too defensive rather than too noisy.

If you want to impose stricter sleeve guardrails on PPO without changing the classical optimizer baseline used for comparison, use `--policy-weight-bound`. This applies only to the RL environment, while `--weight-bound` still applies to both PPO and the optimizer baseline.

```bash
.venv/bin/python -m wealth_first.train_ppo \
  --returns-csv data/demo_sleeves.csv \
  --benchmark-column SPY_BENCHMARK \
  --exclude-benchmark-from-universe \
  --split-method regime-balanced \
  --lookback 20 \
  --episode-length 126 \
  --benchmark-loss-penalty 4.0 \
  --action-smoothing 0.5 \
  --no-trade-band 0.05 \
  --weight-bound CASH=0.10:0.80 \
  --weight-bound HEDGE_OVERLAY=0.00:0.25 \
  --policy-weight-bound TREND_FOLLOWING=0.45:0.85 \
  --policy-weight-bound MEAN_REVERSION=0.00:0.25 \
  --policy-weight-bound HEDGE_OVERLAY=0.00:0.05 \
  --policy-weight-bound CASH=0.10:0.35 \
  --total-timesteps 2000 \
  --output-dir artifacts/ppo_guardrail_probe
```

That setup is useful when PPO keeps underweighting trend and overusing hedge or cash, but you still want the optimizer benchmark to reflect the original investable constraints rather than the tighter PPO guardrails.

The PPO baseline is a research step, not a production allocator. It is intended to answer whether the reward shaping, guardrails, and regime splits produce stable out-of-sample behavior before you consider richer features or more complex RL variants.

## TradingView Comparison

If you export strategy returns from TradingView or another charting platform, compare them directly against the Python sleeve CSV before blaming the optimizer.

The official TradingView webhook path is for forward collection of live alerts, not for pulling historical strategy backtests. Keep the wealth-first optimizer as the portfolio engine, and use TradingView as an external truth source for live sleeve events.

Example self-check against identical files:

```bash
.venv/bin/python -m wealth_first.compare \
  --reference-csv data/demo_sleeves.csv \
  --candidate-csv data/demo_sleeves.csv \
  --max-mae 0.0 \
  --min-correlation 0.999999
```

Example with TradingView-style renamed columns:

```bash
.venv/bin/python -m wealth_first.compare \
  --reference-csv data/demo_sleeves.csv \
  --candidate-csv tradingview_export.csv \
  --column-map tv_trend=TREND_FOLLOWING \
  --column-map tv_meanrev=MEAN_REVERSION \
  --column-map tv_hedge=HEDGE_OVERLAY \
  --tolerance 0.000001 \
  --max-mae 0.0005 \
  --min-correlation 0.99 \
  --max-mismatch-rate 0.02
```

The comparison report highlights:

- date mismatches between feeds,
- unmapped columns,
- mean absolute error,
- maximum absolute error,
- correlation,
- per-sleeve mismatch rate.

If the sleeves line up closely and still underperform, the problem is in the sleeve logic. If they do not line up, the problem is in data handling, session timing, or signal implementation.

## TradingView Webhook Bridge

For live forward validation, this repo includes a TradingView webhook receiver and event normalizer.

Run the bridge locally:

```bash
.venv/bin/python -m wealth_first.tradingview_bridge serve \
  --event-log data/tradingview_events.jsonl \
  --output-csv data/tradingview_truth.csv \
  --token my-shared-secret \
  --default-sleeve TRADINGVIEW \
  --aggregate-freq D \
  --base-equity 100000 \
  --failure-log data/tradingview_bridge_failures.jsonl
```

Once the bridge is running, open `http://127.0.0.1:8000/` to use the local dashboard. It polls the bridge in real time, shows recent events/executions/failures, previews the normalized returns CSV, and includes a built-in test webhook composer for localhost iteration.

For the dedicated frontend development loop, run the Vite app on localhost with hot reload:

```bash
npm --prefix frontend install
npm --prefix frontend run dev -- --host 127.0.0.1 --port 5173
```

The Vite app proxies `/api`, `/webhook`, and `/healthz` back to the bridge on `127.0.0.1:8000`, so you can browse `http://127.0.0.1:5173/` while keeping the Python bridge as the single backend process.

To serve the React app directly from FastAPI, build it once:

```bash
npm --prefix frontend run build
```

After that build, `http://127.0.0.1:8000/` serves the compiled React app, `http://127.0.0.1:8000/app` points to the same built frontend explicitly, and `http://127.0.0.1:8000/dashboard` keeps the original static dashboard as a fallback.

The React UI is now focused on bridge operations: live webhook intake, normalized return monitoring, execution telemetry, and a browser-side webhook composer for local testing.

Optional execution handoff modes let the bridge reuse the allocation layer when alerts include target weights:

```bash
.venv/bin/python -m wealth_first.tradingview_bridge serve \
  --event-log data/tradingview_events.jsonl \
  --token my-shared-secret \
  --execution-mode paper \
  --execution-log data/tradingview_execution.jsonl
```

For webhook execution mode, the bridge now probes the destination endpoint at startup and fails immediately if it cannot reach it. You can point the probe at a dedicated health endpoint when the execution service exposes one:

```bash
.venv/bin/python -m wealth_first.tradingview_bridge serve \
  --event-log data/tradingview_events.jsonl \
  --token my-shared-secret \
  --execution-mode webhook \
  --execution-webhook-url https://execution.example.com/orders \
  --execution-probe-url https://execution.example.com/healthz
```

Normalize an existing event log into dated sleeve returns:

```bash
.venv/bin/python -m wealth_first.tradingview_bridge normalize \
  --event-log data/tradingview_events.jsonl \
  --output-csv data/tradingview_truth.csv \
  --aggregate-freq D \
  --base-equity 100000
```

Recommended TradingView strategy setup:

- Use order fill alerts rather than relying only on `alert()` calls for execution events.
- If Pine generates the JSON, put `{{strategy.order.alert_message}}` in the TradingView alert message field and do not use any other TradingView placeholders inside Pine `alert_message`.
- If you type JSON directly into the TradingView alert dialog instead of generating it from Pine, placeholders such as `{{timenow}}`, `{{ticker}}`, and `{{strategy.order.id}}` are valid there.
- If you change the Pine script, inputs, or `alert_message` shape, delete and recreate the TradingView alert. Strategy alerts run from a server-side snapshot and do not automatically pick up later edits.
- A ready-to-paste Pine v6 example lives in `scripts/wealth_first_tv_bridge_example.pine`.

The bundled Pine example now separates the TradingView `strategy` label from the bridge `sleeve` column. Use the `Strategy label` input for the human-facing strategy identifier and `Sleeve / truth-series column` for the column name that will be written into `data/tradingview_truth.csv`.

Example Pine pattern:

```pinescript
//@version=6
//@strategy_alert_message {{strategy.order.alert_message}}
strategy("Wealth-First TV Bridge Example", overlay = true, pyramiding = 0)

if longSignal and strategy.position_size <= 0
  strategy.order(
    "WF_ENTER_LONG",
    strategy.long,
    qty = targetUnits,
    alert_message = buildRebalanceMessage("WF_ENTER_LONG", riskOnWeight, "long")
  )
```

The full example script builds `timestamp`, `ticker`, `equity`, `netprofit`, `event_id`, and `target_weights` from Pine-native values such as `timenow`, `syminfo.ticker`, `strategy.equity`, and `strategy.netprofit`. That is the supported pattern for order-fill alerts, and the emitted `netprofit` lets the bridge derive dated returns when you run it with `--base-equity`. The event identifier now includes the sleeve name and ticker so multiple live probes remain easier to distinguish in the event log.

Operational limits from TradingView’s webhook system matter here:

- Only ports `80` and `443` are supported.
- Webhook requests are cancelled if your endpoint takes longer than about 3 seconds.
- Valid JSON alerts are sent as `application/json`.
- Webhook alerts require TradingView 2FA to be enabled.

This means the receiver should acknowledge quickly, persist raw events first, and only then normalize them into the dated returns CSV used by the comparison workflow.

The bridge now reflects that constraint more closely:

- each accepted event is fingerprinted and duplicate payloads are ignored,
- normalized CSV regeneration and execution handoff are replayed from the persisted event log by a worker thread, so the work survives process restarts,
- normalization or execution failures are written to a failure log and retried with backoff,
- payloads with `target_weights` can optionally trigger an execution handoff through the existing paper or webhook adapter,
- webhook execution mode performs a startup reachability probe instead of waiting for the first live alert to discover a bad endpoint.

Execution handoff payloads should include `target_weights` and, when available, `current_weights` and `equity`. Example JSON body:

```json
{
  "strategy": "ALLOCATOR",
  "event_type": "rebalance",
  "timestamp": "2024-01-03T21:00:00Z",
  "target_weights": {"SPY": 0.55, "TLT": 0.25, "CASH": 0.20},
  "current_weights": {"SPY": 0.40, "TLT": 0.30, "CASH": 0.30},
  "equity": 100000
}
```

For a real TradingView paper-trade smoke test, you have two valid paths.

If you are typing JSON directly into the TradingView alert dialog message field, use a strategy alert with a fully specified JSON body rather than only `timestamp` and `target_weights`. A practical minimum is:

```json
{
  "timestamp": "{{timenow}}",
  "event_id": "{{strategy.order.id}}",
  "event_type": "rebalance",
  "strategy": "LIVE_PROBE",
  "sleeve": "LIVE_PROBE",
  "ticker": "{{ticker}}",
  "equity": 100000,
  "target_weights": {
    "SPY": 0.40,
    "CASH": 0.60
  }
}
```

If you are generating JSON from Pine `alert_message`, use `scripts/wealth_first_tv_bridge_example.pine` and leave the TradingView alert message field at `{{strategy.order.alert_message}}` or the default value prefilled by the `//@strategy_alert_message` annotation.

For a deterministic one-shot order-fill test, the bundled Pine example also exposes three temporary inputs: `Enable force test order`, `Force test side`, and `Force test nonce`. Set `Enable force test order = true`, choose the side you want to emit, and type a fresh nonce value before saving the script. Then recreate the TradingView alert so the server snapshot picks up that nonce and emits exactly one forced order-fill event for it. Clear or change the nonce after the event lands to avoid reusing the same test key on the next alert recreation.

For a two-probe setup, keep separate alerts and separate sleeves for each chart so their normalized returns do not collide:

- SPY probe: use a sleeve such as `LIVE_PROBE_SPY` on your SPY chart to validate the market-hours equity path.
- BTC probe: use a sleeve such as `LIVE_PROBE_BTC` on a 24/7 chart like `BITSTAMP:BTCUSD` to validate that TradingView can still emit live order-fill alerts outside U.S. equity hours.
- You may keep the same `Strategy label` across both charts if you want a shared human-readable name, but the `Sleeve / truth-series column` should stay unique per probe.
- Recreate each alert independently after changing inputs because TradingView stores alert snapshots per chart and symbol.

For normalized truth-series generation, make sure the alert payload includes at least one usable return signal. The bridge accepts `return_value`, `realized_pnl` plus `equity_before`, or cumulative `netprofit`. The bundled Pine example now emits cumulative `netprofit` for that purpose.

Notes for that test:

- put the shared secret in the webhook URL query string, for example `https://your-endpoint.example.com/webhook?token=...`, instead of embedding it in the JSON body,
- prefer `{{timenow}}` over `{{time}}` because `{{time}}` is the bar time, while `{{timenow}}` is the alert fire time,
- `{{strategy.order.id}}` only works in strategy alerts, so use an order-fill or strategy alert, not a generic price alert,
- placeholders such as `{{ticker}}`, `{{timenow}}`, and `{{strategy.order.id}}` are valid in the TradingView alert dialog message field, but not inside Pine `alert_message`,
- if you want dynamic JSON generated from Pine, emit it through `alert_message` in `strategy.entry()`, `strategy.order()`, or `strategy.close()` and set the TradingView alert message to `{{strategy.order.alert_message}}`,
- if you run more than one live TradingView probe, give each chart a unique sleeve name so the bridge writes separate columns into `data/tradingview_truth.csv`.

## Notes

- `yfinance` is used for live SPY download.
- Tests are offline and do not require network access.
- For live trading deployment, replace the data loader with broker or exchange data and use the optimizer outputs as target allocations.