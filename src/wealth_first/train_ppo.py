from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import pandas as pd

from wealth_first.backtest import run_rolling_backtest, summarize_performance
from wealth_first.data_splits import SuggestedTimeSeriesSplit, chronological_train_validation_test_split, generate_walk_forward_splits, suggest_regime_balanced_split
from wealth_first.data import load_returns_csv
from wealth_first.optimizer import WealthFirstConfig
from wealth_first.rl import WealthFirstEnv
from wealth_first.rebalance import compute_execution_cost, compute_tradable_turnover


COMMON_COMPARISON_METRICS = [
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
    "average_step_return",
    "sharpe_like",
    "sortino_like",
    "gross_total_return",
    "average_turnover",
    "total_turnover",
    "average_trading_cost",
    "cost_drag",
    "average_cash_weight",
    "average_risky_weight",
    "benchmark_total_return",
    "relative_total_return",
    "average_active_return",
    "tracking_error",
    "information_ratio",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a PPO baseline against the Wealth-First Gymnasium environment.")
    parser.add_argument("--returns-csv", default="data/demo_sleeves.csv", help="CSV file containing investable sleeves and an optional benchmark column.")
    parser.add_argument("--date-column", default=None, help="Optional date column name when loading returns from CSV.")
    parser.add_argument("--start", default=None, help="Optional inclusive start date filter.")
    parser.add_argument("--end", default=None, help="Optional inclusive end date filter.")
    parser.add_argument("--benchmark-column", default=None, help="Optional benchmark return column used for benchmark-aware rewards.")
    parser.add_argument(
        "--exclude-benchmark-from-universe",
        action="store_true",
        help="Remove the benchmark column from the investable universe while retaining it for reward shaping and evaluation.",
    )
    parser.add_argument("--lookback", type=int, default=20, help="Number of trailing observations exposed to the policy.")
    parser.add_argument("--episode-length", type=int, default=126, help="Episode length in trading periods for sampled training windows.")
    parser.add_argument(
        "--validation-fraction",
        "--eval-fraction",
        dest="validation_fraction",
        type=float,
        default=0.15,
        help="Fraction reserved for validation model selection. `--eval-fraction` is kept as a compatibility alias.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.10, help="Fraction reserved for the final holdout test window.")
    parser.add_argument(
        "--split-method",
        choices=["regime-balanced", "regime", "chronological", "chrono"],
        default="regime-balanced",
        help="How to choose chronological train, validation, and test windows.",
    )
    parser.add_argument("--split-only", action="store_true", help="Identify and save data-driven split windows without training a PPO model.")
    parser.add_argument("--walk-forward-folds", type=int, default=1, help="Number of anchored walk-forward folds to train and evaluate. Use 1 to train only the latest split.")
    parser.add_argument("--walk-forward-step-rows", type=int, default=None, help="Optional step size in rows between walk-forward folds. Defaults to the test window size.")
    parser.add_argument("--total-timesteps", type=int, default=20_000, help="Total PPO training timesteps.")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--normalize-observations", action="store_true", help="Apply VecNormalize observation scaling during training.")
    parser.add_argument(
        "--benchmark-relative-observations",
        action="store_true",
        help="Append sleeve-minus-benchmark lookback features to the PPO observation when benchmark data is available.",
    )
    benchmark_regime_group = parser.add_mutually_exclusive_group()
    benchmark_regime_group.add_argument(
        "--benchmark-regime-observations",
        action="store_true",
        help="Append compact benchmark regime summary features and per-sleeve cumulative relative returns to the PPO observation.",
    )
    benchmark_regime_group.add_argument(
        "--benchmark-regime-summary-only-observations",
        action="store_true",
        help="Append only the compact benchmark regime summary features to the PPO observation.",
    )
    benchmark_regime_group.add_argument(
        "--benchmark-regime-relative-cumulative-only-observations",
        action="store_true",
        help="Append only per-sleeve cumulative relative returns versus the benchmark to the PPO observation.",
    )
    parser.add_argument(
        "--state-trend-preservation-symbol",
        default=None,
        help="Sleeve whose weight reduction is capped when the current portfolio state is already low-cash and trend-heavy.",
    )
    parser.add_argument(
        "--state-trend-preservation-cash-max-threshold",
        type=float,
        default=None,
        help="Maximum pre-trade cash weight required for the state trend-preservation guard to activate.",
    )
    parser.add_argument(
        "--state-trend-preservation-symbol-min-weight",
        type=float,
        default=None,
        help="Minimum pre-trade weight of --state-trend-preservation-symbol required before the guard preserves trend exposure.",
    )
    parser.add_argument(
        "--state-trend-preservation-max-symbol-reduction",
        type=float,
        default=None,
        help="Maximum allowed reduction in --state-trend-preservation-symbol when the state trend-preservation guard is active.",
    )
    parser.add_argument("--normalize-rewards", action="store_true", help="Apply VecNormalize reward scaling during training.")
    parser.add_argument("--loss-penalty", type=float, default=8.0)
    parser.add_argument("--gain-reward", type=float, default=1.0)
    parser.add_argument("--gain-power", type=float, default=1.0)
    parser.add_argument("--loss-power", type=float, default=1.5)
    parser.add_argument("--benchmark-gain-reward", type=float, default=0.0)
    parser.add_argument("--benchmark-loss-penalty", type=float, default=0.0)
    parser.add_argument("--turnover-penalty", type=float, default=0.0)
    parser.add_argument("--weight-reg", type=float, default=0.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--cash-weight-penalty", type=float, default=0.0, help="Penalty applied to cash weight above the configured or inferred cash target.")
    parser.add_argument("--cash-target-weight", type=float, default=None, help="Optional cash target used by --cash-weight-penalty. Defaults to the cash lower bound when omitted.")
    parser.add_argument(
        "--action-smoothing",
        type=float,
        default=0.0,
        help="Blend each projected action back toward current weights. 0 disables smoothing; 1 holds current weights exactly.",
    )
    parser.add_argument(
        "--no-trade-band",
        type=float,
        default=0.0,
        help="Skip a rebalance when projected tradable turnover is below this threshold.",
    )
    parser.add_argument(
        "--max-executed-rebalances",
        type=int,
        default=None,
        help="Optional PPO-only cap on executed rebalances per episode. When reached, later projected trades are suppressed and the portfolio holds current weights.",
    )
    parser.add_argument(
        "--rebalance-cooldown-steps",
        type=int,
        default=None,
        help="Optional PPO-only cooldown after each executed rebalance. When active, later projected trades are suppressed for the next N steps.",
    )
    parser.add_argument(
        "--early-rebalance-risk-penalty",
        type=float,
        default=0.0,
        help="Optional PPO-only fixed penalty applied to early risk-adding rebalances in low-cash / high-risk states.",
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap",
        type=float,
        default=None,
        help=(
            "Optional PPO-only tradable-turnover cap applied to early risk-adding rebalances in low-cash / high-risk states. "
            "Set to 0 to fully block those trades."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-benchmark-drawdown-min-threshold",
        type=float,
        default=None,
        help=(
            "Optional benchmark-aware refinement for the early risk turnover cap only. "
            "Require benchmark regime drawdown to be at least this value before the turnover cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-benchmark-cumulative-return-max-threshold",
        type=float,
        default=None,
        help=(
            "Optional benchmark-aware refinement for the early risk turnover cap only. "
            "Require benchmark regime cumulative return to be at most this value before the turnover cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-min-pre-trade-cash-weight",
        type=float,
        default=None,
        help=(
            "Optional direct refinement for the early risk turnover cap only. "
            "Require the portfolio pre-trade cash weight to be at least this value before the turnover cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-target-cash-min-threshold",
        type=float,
        default=None,
        help=(
            "Optional direct refinement for the early risk turnover cap only. "
            "Require the portfolio target cash weight to be at least this value before the turnover cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-target-cash-max-threshold",
        type=float,
        default=None,
        help=(
            "Optional direct refinement for the early risk turnover cap only. "
            "Require the portfolio target cash weight to be at most this value before the turnover cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-target-trend-min-threshold",
        type=float,
        default=None,
        help=(
            "Optional direct refinement for the early risk turnover cap only. "
            "Require the portfolio target trend weight to be at least this value before the turnover cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-target-trend-max-threshold",
        type=float,
        default=None,
        help=(
            "Optional direct refinement for the early risk turnover cap only. "
            "Require the portfolio target trend weight to be at most this value before the turnover cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-target-mean-reversion-min-threshold",
        type=float,
        default=None,
        help=(
            "Optional direct refinement for the early risk turnover cap only. "
            "Require the portfolio target mean-reversion weight to be at least this value before the turnover cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-target-mean-reversion-max-threshold",
        type=float,
        default=None,
        help=(
            "Optional direct refinement for the early risk turnover cap only. "
            "Require the portfolio target mean-reversion weight to be at most this value before the turnover cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-delta-cash-min-threshold",
        type=float,
        default=None,
        help=(
            "Optional direct refinement for the early risk turnover cap only. "
            "Require the portfolio target cash-weight change to be at least this value before the turnover cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-delta-cash-max-threshold",
        type=float,
        default=None,
        help=(
            "Optional direct refinement for the early risk turnover cap only. "
            "Require the portfolio target cash-weight change to be at most this value before the turnover cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-delta-trend-min-threshold",
        type=float,
        default=None,
        help=(
            "Optional direct refinement for the early risk turnover cap only. "
            "Require the portfolio target trend-weight change to be at least this value before the turnover cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-delta-trend-max-threshold",
        type=float,
        default=None,
        help=(
            "Optional direct refinement for the early risk turnover cap only. "
            "Require the portfolio target trend-weight change to be at most this value before the turnover cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-delta-mean-reversion-min-threshold",
        type=float,
        default=None,
        help=(
            "Optional direct refinement for the early risk turnover cap only. "
            "Require the portfolio target mean-reversion-weight change to be at least this value before the turnover cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-delta-mean-reversion-max-threshold",
        type=float,
        default=None,
        help=(
            "Optional direct refinement for the early risk turnover cap only. "
            "Require the portfolio target mean-reversion-weight change to be at most this value before the turnover cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-proposed-turnover-min-threshold",
        type=float,
        default=None,
        help=(
            "Optional direct refinement for the early risk turnover cap only. "
            "Require the projected tradable turnover to be at least this value before the turnover cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-proposed-turnover-max-threshold",
        type=float,
        default=None,
        help=(
            "Optional direct refinement for the early risk turnover cap only. "
            "Require the projected tradable turnover to be at most this value before the turnover cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-min-pre-trade-relative-wealth-ratio",
        type=float,
        default=None,
        help=(
            "Optional direct refinement for the early risk turnover cap only. "
            "Require the portfolio pre-trade relative wealth ratio versus the benchmark to be at least this value before the turnover cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-max-pre-trade-relative-wealth-ratio",
        type=float,
        default=None,
        help=(
            "Optional direct refinement for the early risk turnover cap only. "
            "Require the portfolio pre-trade relative wealth ratio versus the benchmark to be at most this value before the turnover cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-allow-nonincreasing-risk-symbol",
        action="store_true",
        help=(
            "Optional direct behavior toggle for the early risk turnover cap only. "
            "Allow the configured risk-symbol gate to activate even when the target weight does not exceed the pre-trade weight."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-use-penalty-state-filters",
        action="store_true",
        help=(
            "Optionally let the early risk turnover cap reuse the configured early-risk penalty state filters "
            "such as symbol min/max weights, benchmark drawdown max threshold, and minimum pre-trade relative wealth ratio."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-after",
        type=int,
        default=None,
        help=(
            "Optional turnover-cap-specific start for the early-risk control. "
            "When omitted, the turnover cap reuses --early-rebalance-risk-penalty-after."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-before",
        type=int,
        default=None,
        help=(
            "Optional turnover-cap-specific end for the early-risk control. "
            "When omitted, the turnover cap reuses --early-rebalance-risk-penalty-before."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-max-applications",
        type=int,
        default=None,
        help=(
            "Optional path-memory refinement for the early risk turnover cap only. "
            "Limit how many times the primary turnover cap may apply within an episode."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-secondary-cap",
        type=float,
        default=None,
        help=(
            "Optional two-phase refinement for the early risk turnover cap only. "
            "After the configured application threshold is reached, switch to this secondary turnover cap."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-secondary-after-applications",
        type=int,
        default=None,
        help=(
            "Optional activation threshold for the secondary early risk turnover cap. "
            "Switch once at least this many earlier cap applications have occurred in the episode."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-secondary-benchmark-cumulative-return-min-threshold",
        type=float,
        default=None,
        help=(
            "Optional state condition for the secondary early risk turnover cap only. "
            "Require benchmark cumulative return to be at least this value before the secondary cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-turnover-cap-secondary-max-pre-trade-relative-wealth-ratio",
        type=float,
        default=None,
        help=(
            "Optional catch-up condition for the secondary early risk turnover cap only. "
            "Require pre-trade relative wealth versus the benchmark to be at most this value before the secondary cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-deep-drawdown-turnover-cap",
        type=float,
        default=None,
        help=(
            "Optional PPO-only second tradable-turnover cap for early risk-adding rebalances in deep benchmark drawdowns. "
            "Set to 0 to fully block those trades."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-deep-drawdown-turnover-cap-after",
        type=int,
        default=None,
        help=(
            "Optional start for the deep-drawdown early-risk turnover cap. "
            "When omitted, the deep-drawdown cap starts from the first executed rebalance."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-deep-drawdown-turnover-cap-before",
        type=int,
        default=None,
        help="Apply the deep-drawdown early-risk turnover cap while fewer than N executed rebalances have occurred.",
    )
    parser.add_argument(
        "--early-rebalance-risk-deep-drawdown-turnover-cap-benchmark-drawdown-max-threshold",
        type=float,
        default=None,
        help=(
            "Required benchmark-aware refinement for the deep-drawdown early-risk turnover cap. "
            "Require benchmark regime drawdown to be at most this value before the cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-shallow-drawdown-turnover-cap",
        type=float,
        default=None,
        help=(
            "Optional PPO-only additive tradable-turnover cap for early risk-adding rebalances in shallow benchmark drawdowns. "
            "This stacks on top of the primary early-risk turnover cap and can be set to 0 to fully block those targeted trades."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-shallow-drawdown-turnover-cap-after",
        type=int,
        default=None,
        help=(
            "Optional start for the shallow-drawdown early-risk turnover cap. "
            "For example, 1 with before 3 targets the second and third executed rebalances."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-shallow-drawdown-turnover-cap-before",
        type=int,
        default=None,
        help="Apply the shallow-drawdown early-risk turnover cap while fewer than N executed rebalances have occurred.",
    )
    parser.add_argument(
        "--early-rebalance-risk-shallow-drawdown-turnover-cap-cash-max-threshold",
        type=float,
        default=None,
        help=(
            "Required cash-state refinement for the shallow-drawdown early-risk turnover cap. "
            "Require pre-trade cash weight to be at most this value before the cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-shallow-drawdown-turnover-cap-benchmark-drawdown-min-threshold",
        type=float,
        default=None,
        help=(
            "Required benchmark-aware refinement for the shallow-drawdown early-risk turnover cap. "
            "Require benchmark regime drawdown to be at least this value before the cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-mean-reversion-turnover-cap",
        type=float,
        default=None,
        help=(
            "Optional PPO-only additive tradable-turnover cap for early risk-adding rebalances in mean-reversion-heavy states. "
            "This stacks on top of the primary early-risk turnover cap and can be set to 0 to fully block those targeted trades."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-mean-reversion-action-smoothing",
        type=float,
        default=None,
        help=(
            "Optional PPO-only action smoothing for early risk-adding rebalances in mean-reversion-heavy states. "
            "When triggered, this blends the target weights back toward pre-trade weights before any mean-reversion turnover cap is applied."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-mean-reversion-turnover-cap-after",
        type=int,
        default=None,
        help=(
            "Optional start for the mean-reversion early-risk turnover cap. "
            "For example, 1 with before 3 targets the second and third executed rebalances."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-mean-reversion-turnover-cap-before",
        type=int,
        default=None,
        help="Apply the mean-reversion early-risk turnover cap while fewer than N executed rebalances have occurred.",
    )
    parser.add_argument(
        "--early-rebalance-risk-mean-reversion-turnover-cap-benchmark-cumulative-return-max-threshold",
        type=float,
        default=None,
        help=(
            "Optional benchmark-aware refinement for the mean-reversion early-risk turnover cap only. "
            "When set, this overrides the shared primary early-risk benchmark cumulative-return cutoff for this overlay."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-mean-reversion-turnover-cap-target-mean-reversion-min-threshold",
        type=float,
        default=None,
        help=(
            "Optional mean-reversion target-shape refinement for the mean-reversion early-risk turnover cap. "
            "Require target MEAN_REVERSION weight to be at least this value before the cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-mean-reversion-turnover-cap-pre-trade-mean-reversion-min-threshold",
        type=float,
        default=None,
        help=(
            "Optional state refinement for the mean-reversion early-risk turnover cap. "
            "Require pre-trade MEAN_REVERSION weight to be at least this value before the cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-mean-reversion-turnover-cap-delta-mean-reversion-min-threshold",
        type=float,
        default=None,
        help=(
            "Optional target-shape refinement for the mean-reversion early-risk turnover cap. "
            "Require target MEAN_REVERSION minus pre-trade MEAN_REVERSION to be at least this value before the cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-trend-turnover-cap",
        type=float,
        default=None,
        help=(
            "Optional PPO-only additive tradable-turnover cap for early risk-adding rebalances in trend-heavy states. "
            "This stacks on top of the primary early-risk turnover cap and can be set to 0 to fully block those targeted trades."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-trend-turnover-cap-after",
        type=int,
        default=None,
        help=(
            "Optional start for the trend-following early-risk turnover cap. "
            "For example, 1 with before 3 targets the second and third executed rebalances."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-trend-turnover-cap-before",
        type=int,
        default=None,
        help="Apply the trend-following early-risk turnover cap while fewer than N executed rebalances have occurred.",
    )
    parser.add_argument(
        "--early-rebalance-risk-trend-turnover-cap-benchmark-cumulative-return-max-threshold",
        type=float,
        default=None,
        help=(
            "Optional benchmark-aware refinement for the trend-following early-risk turnover cap only. "
            "When set, this overrides the shared primary early-risk benchmark cumulative-return cutoff for this overlay."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-trend-turnover-cap-target-trend-min-threshold",
        type=float,
        default=None,
        help=(
            "Optional target-shape refinement for the trend-following early-risk turnover cap. "
            "Require target TREND_FOLLOWING weight to be at least this value before the cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-trend-turnover-cap-pre-trade-trend-min-threshold",
        type=float,
        default=None,
        help=(
            "Optional state refinement for the trend-following early-risk turnover cap. "
            "Require pre-trade TREND_FOLLOWING weight to be at least this value before the cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-trend-turnover-cap-delta-trend-min-threshold",
        type=float,
        default=None,
        help=(
            "Optional target-shape refinement for the trend-following early-risk turnover cap. "
            "Require target TREND_FOLLOWING minus pre-trade TREND_FOLLOWING to be at least this value before the cap can activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-repeat-turnover-cap",
        type=float,
        default=None,
        help=(
            "Optional PPO-only tradable-turnover cap for repeated early risk-add trades. "
            "When the previous executed rebalance already spent cash into the configured symbol, "
            "later repeated early adds are capped to this turnover. Set to 0 to fully block them."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-repeat-action-smoothing",
        type=float,
        default=None,
        help=(
            "Optional PPO-only repeat-conditioned action smoothing for repeated early risk-add trades. "
            "When the repeat trigger is met, blend the projected target back toward current weights by this fraction. "
            "0 disables the effect; 1 holds current weights exactly for those triggered trades."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-repeat-turnover-cap-after",
        type=int,
        default=None,
        help=(
            "Optional start for the repeated early risk-add cap. "
            "For example, 1 with before 3 targets the second and third executed rebalances."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-repeat-turnover-cap-before",
        type=int,
        default=None,
        help="Apply the repeated early risk-add cap while fewer than N executed rebalances have occurred.",
    )
    parser.add_argument(
        "--early-rebalance-risk-repeat-symbol",
        default=None,
        help="Sleeve used to detect repeated early risk-add trades for the repeat turnover cap.",
    )
    parser.add_argument(
        "--early-rebalance-risk-repeat-previous-cash-reduction-min",
        type=float,
        default=None,
        help="Minimum cash reduction on the previous executed rebalance required before the repeated early risk-add cap can activate.",
    )
    parser.add_argument(
        "--early-rebalance-risk-repeat-previous-symbol-increase-min",
        type=float,
        default=None,
        help="Minimum increase in --early-rebalance-risk-repeat-symbol on the previous executed rebalance required before the repeated early risk-add cap can activate.",
    )
    parser.add_argument(
        "--early-rebalance-risk-repeat-unrecovered-turnover-cap",
        type=float,
        default=None,
        help=(
            "Optional PPO-only tradable-turnover cap for repeated early risk-add trades that have not yet recovered in relative wealth. "
            "When the previous executed rebalance already spent cash into the configured symbol and the portfolio has not recovered enough versus the benchmark since then, "
            "later repeated early adds are capped to this turnover. Set to 0 to fully block them."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-repeat-unrecovered-turnover-cap-after",
        type=int,
        default=None,
        help=(
            "Optional start for the repeated unrecovered early risk-add cap. "
            "For example, 1 with before 3 targets the second and third executed rebalances."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-repeat-unrecovered-turnover-cap-before",
        type=int,
        default=None,
        help="Apply the repeated unrecovered early risk-add cap while fewer than N executed rebalances have occurred.",
    )
    parser.add_argument(
        "--early-rebalance-risk-repeat-unrecovered-symbol",
        default=None,
        help="Sleeve used to detect repeated unrecovered early risk-add trades for the repeat-unrecovered turnover cap.",
    )
    parser.add_argument(
        "--early-rebalance-risk-repeat-unrecovered-previous-cash-reduction-min",
        type=float,
        default=None,
        help="Minimum cash reduction on the previous executed rebalance required before the repeated unrecovered early risk-add cap can activate.",
    )
    parser.add_argument(
        "--early-rebalance-risk-repeat-unrecovered-previous-symbol-increase-min",
        type=float,
        default=None,
        help="Minimum increase in --early-rebalance-risk-repeat-unrecovered-symbol on the previous executed rebalance required before the repeated unrecovered early risk-add cap can activate.",
    )
    parser.add_argument(
        "--early-rebalance-risk-repeat-unrecovered-min-relative-wealth-recovery",
        type=float,
        default=None,
        help=(
            "Minimum improvement in pre-trade relative wealth ratio versus the previous executed rebalance required to avoid the repeated unrecovered early risk-add cap."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-cumulative-turnover-cap",
        type=float,
        default=None,
        help=(
            "Optional PPO-only tradable-turnover cap for cumulative early risk-add budget breaches. "
            "When the projected cumulative early spend into the configured symbol crosses the configured budget, "
            "later adds are capped to this turnover. Set to 0 to fully block them."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-cumulative-turnover-cap-after",
        type=int,
        default=None,
        help=(
            "Optional start for the cumulative early risk-add cap. "
            "For example, 1 with before 3 targets the second and third executed rebalances."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-cumulative-turnover-cap-before",
        type=int,
        default=None,
        help="Apply the cumulative early risk-add cap while fewer than N executed rebalances have occurred.",
    )
    parser.add_argument(
        "--early-rebalance-risk-cumulative-symbol",
        default=None,
        help="Sleeve used to detect cumulative early risk-add trades for the cumulative turnover cap.",
    )
    parser.add_argument(
        "--early-rebalance-risk-cumulative-cash-reduction-budget",
        type=float,
        default=None,
        help=(
            "Projected cumulative cash reduction budget for the cumulative early risk-add cap. "
            "The cap activates once the projected early path would exceed this budget."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-cumulative-symbol-increase-budget",
        type=float,
        default=None,
        help=(
            "Projected cumulative increase budget for --early-rebalance-risk-cumulative-symbol. "
            "The cap activates once the projected early path would exceed this budget."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-penalty-after",
        type=int,
        default=None,
        help=(
            "Start the early-risk control once at least N executed rebalances have occurred. "
            "For example, 1 with --early-rebalance-risk-penalty-before 3 targets the second and third executed rebalances."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-penalty-before",
        type=int,
        default=None,
        help="Apply --early-rebalance-risk-penalty while fewer than N executed rebalances have occurred. For example, 4 targets the first four executed rebalances.",
    )
    parser.add_argument(
        "--early-rebalance-risk-penalty-cash-max-threshold",
        type=float,
        default=None,
        help="Maximum pre-trade cash weight for the early risk penalty to activate.",
    )
    parser.add_argument(
        "--early-rebalance-risk-penalty-symbol",
        default=None,
        help="Sleeve used to detect early risk-adding trades for --early-rebalance-risk-penalty.",
    )
    parser.add_argument(
        "--early-rebalance-risk-penalty-symbol-min-weight",
        type=float,
        default=None,
        help="Minimum pre-trade weight of --early-rebalance-risk-penalty-symbol required for the early risk penalty to activate.",
    )
    parser.add_argument(
        "--early-rebalance-risk-penalty-symbol-max-weight",
        type=float,
        default=None,
        help="Maximum pre-trade weight of --early-rebalance-risk-penalty-symbol allowed for the early risk penalty to activate.",
    )
    parser.add_argument(
        "--early-rebalance-risk-penalty-benchmark-drawdown-min-threshold",
        type=float,
        default=None,
        help=(
            "Optional benchmark-aware refinement for the early risk penalty. "
            "Require benchmark regime drawdown to be at least this value for the early risk control to activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-penalty-benchmark-drawdown-max-threshold",
        type=float,
        default=None,
        help=(
            "Optional benchmark-aware refinement for the early risk penalty. "
            "Require benchmark regime drawdown to be at most this value for the early risk control to activate."
        ),
    )
    parser.add_argument(
        "--early-rebalance-risk-penalty-min-pre-trade-relative-wealth-ratio",
        type=float,
        default=None,
        help=(
            "Optional benchmark-aware refinement for the early risk penalty. "
            "Require the portfolio pre-trade relative wealth ratio versus the benchmark to be at least this value for the control to activate."
        ),
    )
    parser.add_argument(
        "--early-benchmark-euphoria-penalty",
        type=float,
        default=0.0,
        help="Optional PPO-only fixed penalty applied to early risk-adding trades when benchmark drawdown remains shallow.",
    )
    parser.add_argument(
        "--early-benchmark-euphoria-turnover-cap",
        type=float,
        default=None,
        help=(
            "Optional PPO-only tradable-turnover cap applied to early risk-adding trades when benchmark drawdown remains shallow. "
            "Set to 0 to fully block those trades."
        ),
    )
    parser.add_argument(
        "--early-benchmark-euphoria-before",
        type=int,
        default=None,
        help="Apply the early benchmark-euphoria control while fewer than N executed rebalances have occurred.",
    )
    parser.add_argument(
        "--early-benchmark-euphoria-benchmark-drawdown-min-threshold",
        type=float,
        default=None,
        help="Minimum benchmark regime drawdown required for the early benchmark-euphoria control to activate.",
    )
    parser.add_argument(
        "--early-benchmark-euphoria-symbol",
        default=None,
        help="Sleeve used to detect early benchmark-euphoria risk additions.",
    )
    parser.add_argument(
        "--late-rebalance-penalty",
        type=float,
        default=0.0,
        help="Optional PPO-only fixed penalty applied to executed rebalances after the configured late-rebalance threshold.",
    )
    parser.add_argument(
        "--late-rebalance-penalty-after",
        type=int,
        default=None,
        help="Apply --late-rebalance-penalty starting with the (N+1)th executed rebalance. For example, 4 penalizes the fifth and later executed rebalances.",
    )
    parser.add_argument(
        "--late-rebalance-gate-after",
        type=int,
        default=None,
        help=(
            "Apply a state-conditional late-rebalance gate starting with the (N+1)th executed rebalance. "
            "When active, later projected trades execute only if the configured cash/trend restore condition is met."
        ),
    )
    parser.add_argument(
        "--late-rebalance-gate-cash-threshold",
        type=float,
        default=None,
        help="Minimum pre-trade cash weight required for the late-rebalance gate to allow a trade.",
    )
    parser.add_argument(
        "--late-rebalance-gate-target-cash-min-threshold",
        type=float,
        default=None,
        help="Minimum target cash weight required for the late-rebalance gate to allow a trade.",
    )
    parser.add_argument(
        "--late-rebalance-gate-symbol",
        default=None,
        help="Sleeve that must be increased by the projected late rebalance for the gate to allow the trade.",
    )
    parser.add_argument(
        "--late-rebalance-gate-symbol-max-weight",
        type=float,
        default=None,
        help="Maximum pre-trade weight of --late-rebalance-gate-symbol for the late-rebalance gate to allow a trade.",
    )
    parser.add_argument(
        "--late-rebalance-gate-cash-reduction-max",
        type=float,
        default=None,
        help=(
            "Optional late-gate refinement. When set with --late-rebalance-gate-symbol-increase-max, "
            "later trades are blocked only if they reduce cash by more than this amount and increase the gate symbol by more than the configured max increase."
        ),
    )
    parser.add_argument(
        "--late-rebalance-gate-symbol-increase-max",
        type=float,
        default=None,
        help=(
            "Optional late-gate refinement paired with --late-rebalance-gate-cash-reduction-max. "
            "Later trades are blocked only when both the cash reduction and gate-symbol increase exceed their configured maxima."
        ),
    )
    parser.add_argument(
        "--late-defensive-posture-penalty",
        type=float,
        default=0.0,
        help="Optional PPO-only fixed penalty applied after the configured late window when target posture remains cash-heavy and trend-light.",
    )
    parser.add_argument(
        "--late-defensive-posture-penalty-after",
        type=int,
        default=None,
        help="Start the late defensive posture penalty once at least N executed rebalances have occurred.",
    )
    parser.add_argument(
        "--late-defensive-posture-penalty-cash-min-threshold",
        type=float,
        default=None,
        help="Minimum target cash weight required for the late defensive posture penalty to activate.",
    )
    parser.add_argument(
        "--late-defensive-posture-penalty-symbol",
        default=None,
        help="Sleeve used to detect late defensive posture for --late-defensive-posture-penalty.",
    )
    parser.add_argument(
        "--late-defensive-posture-penalty-symbol-max-weight",
        type=float,
        default=None,
        help="Maximum target weight of --late-defensive-posture-penalty-symbol required for the late defensive posture penalty to activate.",
    )
    parser.add_argument(
        "--late-trend-mean-reversion-conflict-penalty",
        type=float,
        default=0.0,
        help="Optional PPO-only fixed penalty applied after the configured late window when target trend and mean-reversion weights are both elevated.",
    )
    parser.add_argument(
        "--late-trend-mean-reversion-conflict-penalty-after",
        type=int,
        default=None,
        help="Start the late trend/mean-reversion conflict penalty once at least N executed rebalances have occurred.",
    )
    parser.add_argument(
        "--late-trend-mean-reversion-conflict-trend-symbol",
        default=None,
        help="Trend sleeve used to detect late trend/mean-reversion conflict posture.",
    )
    parser.add_argument(
        "--late-trend-mean-reversion-conflict-trend-min-weight",
        type=float,
        default=None,
        help="Minimum target trend weight required for the late trend/mean-reversion conflict penalty to activate.",
    )
    parser.add_argument(
        "--late-trend-mean-reversion-conflict-mean-reversion-symbol",
        default=None,
        help="Mean-reversion sleeve used to detect late trend/mean-reversion conflict posture.",
    )
    parser.add_argument(
        "--late-trend-mean-reversion-conflict-mean-reversion-min-weight",
        type=float,
        default=None,
        help="Minimum target mean-reversion weight required for the late trend/mean-reversion conflict penalty to activate.",
    )
    parser.add_argument(
        "--falsification-trade-budget",
        action="append",
        default=[],
        type=int,
        metavar="N",
        help="Additional capped-trade falsifiers to compute alongside static hold. Each value allows the first N executed rebalances, then forces hold. Defaults to 1, 2, and 3 when omitted.",
    )
    parser.add_argument("--weight-bound", action="append", default=[], metavar="SYMBOL=MIN:MAX")
    parser.add_argument(
        "--policy-weight-bound",
        action="append",
        default=[],
        metavar="SYMBOL=MIN:MAX",
        help="PPO-only sleeve bounds applied to the RL environment without changing the optimizer baseline used for comparison.",
    )
    parser.add_argument("--eval-freq", type=int, default=2_000, help="How often to run evaluation during training.")
    parser.add_argument("--n-eval-episodes", type=int, default=1, help="Number of evaluation episodes for each callback pass.")
    parser.add_argument("--output-dir", default="artifacts/ppo_baseline", help="Directory used for saved models and metrics.")
    parser.add_argument("--tensorboard-log", default=None, help="Optional directory for TensorBoard logs when tensorboard is installed.")
    return parser


def _normalize_split_method_alias(split_method: str) -> str:
    if split_method == "chrono":
        return "chronological"
    if split_method == "regime":
        return "regime-balanced"
    return split_method


def _filter_returns_by_date(returns: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if not isinstance(returns.index, pd.DatetimeIndex):
        return returns

    filtered = returns.sort_index()
    if start:
        filtered = filtered.loc[filtered.index >= pd.Timestamp(start)]
    if end:
        filtered = filtered.loc[filtered.index <= pd.Timestamp(end)]
    if filtered.empty:
        raise ValueError("No return observations remain after applying the requested date filter.")
    return filtered


def _parse_weight_bound_overrides(raw_bounds: list[str]) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    min_overrides: dict[str, float] = {}
    max_overrides: dict[str, float] = {}

    for raw_bound in raw_bounds:
        symbol_text, separator, bounds_text = raw_bound.partition("=")
        if not separator or ":" not in bounds_text:
            raise ValueError(f"Invalid weight bound '{raw_bound}'. Expected SYMBOL=MIN:MAX.")

        minimum_text, maximum_text = bounds_text.split(":", 1)
        symbol = symbol_text.strip()
        if not symbol:
            raise ValueError(f"Invalid weight bound '{raw_bound}'. Sleeve name cannot be empty.")
        if symbol in min_overrides:
            raise ValueError(f"Duplicate weight bound provided for '{symbol}'.")

        min_overrides[symbol] = float(minimum_text.strip())
        max_overrides[symbol] = float(maximum_text.strip())

    return (min_overrides or None, max_overrides or None)


def _merge_weight_bound_overrides(
    base_min_overrides: dict[str, float] | None,
    base_max_overrides: dict[str, float] | None,
    overlay_min_overrides: dict[str, float] | None,
    overlay_max_overrides: dict[str, float] | None,
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    merged_min_overrides = dict(base_min_overrides or {})
    merged_max_overrides = dict(base_max_overrides or {})
    if overlay_min_overrides:
        merged_min_overrides.update(overlay_min_overrides)
    if overlay_max_overrides:
        merged_max_overrides.update(overlay_max_overrides)
    return (merged_min_overrides or None, merged_max_overrides or None)


def _resolve_benchmark_regime_observation_flags(args) -> tuple[bool, bool]:
    if args.benchmark_regime_summary_only_observations:
        return True, False
    if args.benchmark_regime_relative_cumulative_only_observations:
        return False, True
    if args.benchmark_regime_observations:
        return True, True
    return False, False


def _normalize_trade_budgets(raw_budgets: list[int]) -> list[int]:
    normalized_budgets: list[int] = []
    seen_budgets: set[int] = set()
    for raw_budget in raw_budgets:
        budget = int(raw_budget)
        if budget < 1:
            raise ValueError("falsification trade budgets must be positive integers.")
        if budget in seen_budgets:
            continue
        seen_budgets.add(budget)
        normalized_budgets.append(budget)
    return normalized_budgets


def _load_rl_dependencies():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization
    except ImportError as exc:  # pragma: no cover - exercised by runtime validation instead of unit tests.
        raise ImportError(
            "Stable-Baselines3 and its PyTorch dependency are required for PPO training. Install them with `pip install -e '.[rl]'`."
        ) from exc

    return PPO, EvalCallback, evaluate_policy, Monitor, DummyVecEnv, VecNormalize, sync_envs_normalization


def _serialize_scalar(value) -> float | int | str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (float, int, str, bool)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _serialize_series(series: pd.Series) -> dict[str, float | int | str | None]:
    return {str(index): _serialize_scalar(value) for index, value in series.items()}


def _flatten_weight_mapping(prefix: str, weights: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}{symbol}": float(value) for symbol, value in weights.items()}


def _mean_numeric(values: list[object]) -> float | None:
    numeric_values: list[float] = []
    for value in values:
        if value is None or pd.isna(value):
            continue
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not numeric_values:
        return None
    return float(sum(numeric_values) / len(numeric_values))


def _aggregate_metric_maps(metric_maps: list[dict[str, object]]) -> dict[str, float | None]:
    metric_names = sorted({metric_name for metric_map in metric_maps for metric_name in metric_map})
    return {
        metric_name: _mean_numeric([metric_map.get(metric_name) for metric_map in metric_maps])
        for metric_name in metric_names
    }


def _aggregate_named_comparison_maps(
    comparisons: list[dict[str, dict[str, object]]],
    left_field: str,
    right_field: str,
) -> dict[str, dict[str, float | None]]:
    metric_names = sorted({metric_name for comparison in comparisons for metric_name in comparison})
    fields = (left_field, right_field, "difference")
    aggregated: dict[str, dict[str, float | None]] = {}
    for metric_name in metric_names:
        aggregated[metric_name] = {
            field: _mean_numeric([comparison.get(metric_name, {}).get(field) for comparison in comparisons])
            for field in fields
        }
    return aggregated


def _aggregate_comparison_maps(comparisons: list[dict[str, dict[str, object]]]) -> dict[str, dict[str, float | None]]:
    return _aggregate_named_comparison_maps(comparisons, left_field="policy", right_field="optimizer")


def _update_post_return_weights(weights: pd.Series, realized_returns: pd.Series, epsilon: float) -> pd.Series:
    period_values = weights * (1.0 + realized_returns)
    total_value = float(period_values.sum())
    if total_value <= epsilon:
        return weights.copy()
    return period_values / total_value


def _score_step_utility(wealth_ratio: float, config: WealthFirstConfig, gain_reward: float, loss_penalty: float) -> float:
    if config.objective_mode == "piecewise":
        upside = max(wealth_ratio - 1.0, 0.0)
        downside = max(1.0 - wealth_ratio, 0.0)
        return gain_reward * (upside ** config.gain_power) - loss_penalty * (downside ** config.loss_power)

    safe_ratio = max(wealth_ratio, config.epsilon)
    log_growth = float(math.log(safe_ratio))
    upside = max(log_growth, 0.0)
    downside = max(-log_growth, 0.0)
    return gain_reward * upside - loss_penalty * (downside ** config.loss_power)


def _align_frame_to_rollout_records(frame: pd.DataFrame, rollout_records: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.index, pd.DatetimeIndex) and isinstance(rollout_records.index, pd.DatetimeIndex):
        aligned = frame.reindex(rollout_records.index)
    else:
        aligned = frame.copy()
        aligned.index = aligned.index.map(str)
        aligned = aligned.reindex(rollout_records["step_label"].astype(str))
        aligned.index = rollout_records.index
    if aligned.isna().any().any():
        raise ValueError("Could not align return data to the rollout step labels.")
    return aligned


def _align_series_to_rollout_records(series: pd.Series, rollout_records: pd.DataFrame) -> pd.Series:
    if isinstance(series.index, pd.DatetimeIndex) and isinstance(rollout_records.index, pd.DatetimeIndex):
        aligned = series.reindex(rollout_records.index)
    else:
        aligned = series.copy()
        aligned.index = aligned.index.map(str)
        aligned = aligned.reindex(rollout_records["step_label"].astype(str))
        aligned.index = rollout_records.index
    if aligned.isna().any():
        raise ValueError("Could not align benchmark data to the rollout step labels.")
    return aligned


def _extract_rollout_asset_columns(records: pd.DataFrame, prefix: str = "target_weight_") -> pd.Index:
    asset_columns = [column_name[len(prefix) :] for column_name in records.columns if column_name.startswith(prefix)]
    if not asset_columns:
        raise ValueError(f"Rollout records are missing columns with prefix '{prefix}'.")
    return pd.Index(asset_columns, dtype=object)


def _augment_aligned_returns_for_rollout(
    aligned_returns: pd.DataFrame,
    rollout_records: pd.DataFrame,
    cash_symbol: str,
) -> pd.DataFrame:
    asset_columns = _extract_rollout_asset_columns(rollout_records, prefix="target_weight_")
    expanded_returns = aligned_returns.copy()
    for asset_name in asset_columns:
        if asset_name in expanded_returns.columns:
            continue
        if asset_name == cash_symbol:
            expanded_returns[asset_name] = 0.0
            continue
        raise ValueError(f"Rollout records reference asset '{asset_name}' but aligned returns do not include it.")
    return expanded_returns.loc[:, asset_columns]


def _extract_rollout_weight_row(
    row: pd.Series,
    prefix: str,
    asset_columns: pd.Index,
    fallback_prefix: str | None = None,
) -> pd.Series:
    values: dict[str, float] = {}
    for asset_name in asset_columns:
        column_name = f"{prefix}{asset_name}"
        if column_name not in row.index:
            if fallback_prefix is None:
                raise ValueError(f"Rollout records are missing '{column_name}'.")
            column_name = f"{fallback_prefix}{asset_name}"
            if column_name not in row.index:
                raise ValueError(f"Rollout records are missing '{column_name}'.")
        values[str(asset_name)] = float(row[column_name])
    return pd.Series(values, index=asset_columns, dtype=float)


def _extract_rollout_weight_series(records: pd.DataFrame, prefix: str, asset_columns: pd.Index) -> pd.Series:
    if records.empty:
        raise ValueError("Rollout records are empty.")

    return _extract_rollout_weight_row(records.iloc[0], prefix, asset_columns)


def _resolve_cash_target_weight(config: WealthFirstConfig, explicit_cash_target_weight: float | None) -> float | None:
    if not config.include_cash:
        return None
    if explicit_cash_target_weight is not None:
        return float(explicit_cash_target_weight)
    if config.min_weight_overrides and config.cash_symbol in config.min_weight_overrides:
        return float(config.min_weight_overrides[config.cash_symbol])
    return 0.0


def _build_trade_budget_label(max_rebalances: int) -> str:
    return f"trade_budget_{max_rebalances}"


def _simulate_trade_budget_rollout_records(
    policy_rollout_records: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark_returns: pd.Series | None,
    config: WealthFirstConfig,
    max_rebalances: int | None,
    cash_weight_penalty: float = 0.0,
    cash_target_weight: float | None = None,
) -> pd.DataFrame:
    if max_rebalances is not None and max_rebalances < 0:
        raise ValueError("max_rebalances must be non-negative when provided.")
    if policy_rollout_records.empty:
        raise ValueError("Policy rollout records are empty.")

    aligned_returns = _augment_aligned_returns_for_rollout(
        _align_frame_to_rollout_records(returns, policy_rollout_records),
        policy_rollout_records,
        config.cash_symbol,
    )
    aligned_benchmark = (
        _align_series_to_rollout_records(benchmark_returns, policy_rollout_records)
        if benchmark_returns is not None
        else None
    )
    current_weights = _extract_rollout_weight_series(policy_rollout_records, "pre_trade_weight_", aligned_returns.columns)
    resolved_cash_target_weight = _resolve_cash_target_weight(config, cash_target_weight)

    current_wealth = 1.0
    current_benchmark_wealth = 1.0
    peak_wealth = 1.0
    executed_rebalances = 0
    records: list[dict[str, float | str | bool]] = []

    for step_index, (step_label, realized_returns) in enumerate(aligned_returns.iterrows()):
        rollout_row = policy_rollout_records.iloc[step_index]
        pre_trade_weights = current_weights.copy()
        policy_proposed_weights = _extract_rollout_weight_row(
            rollout_row,
            "proposed_weight_",
            aligned_returns.columns,
            fallback_prefix="target_weight_",
        ).rename("proposed_weight")
        policy_target_weights = _extract_rollout_weight_row(
            rollout_row,
            "target_weight_",
            aligned_returns.columns,
        ).rename("target_weight")

        budget_exhausted = max_rebalances is not None and executed_rebalances >= max_rebalances
        if budget_exhausted:
            proposed_weights = policy_target_weights.copy().rename("proposed_weight")
            target_weights = current_weights.copy().rename("target_weight")
        else:
            proposed_weights = policy_proposed_weights.copy().rename("proposed_weight")
            target_weights = policy_target_weights.copy().rename("target_weight")

        proposed_turnover = compute_tradable_turnover(proposed_weights, pre_trade_weights, config.cash_symbol)
        turnover = compute_tradable_turnover(target_weights, pre_trade_weights, config.cash_symbol)
        budget_trade_suppressed = budget_exhausted and proposed_turnover > 1e-12
        execution_cost = compute_execution_cost(turnover, config.transaction_cost_bps, config.slippage_bps)
        gross_return_contributions = (target_weights * realized_returns).rename("gross_return_contribution")
        gross_portfolio_return = float(gross_return_contributions.sum())
        gross_wealth_ratio = max(1.0 + gross_portfolio_return, config.epsilon)
        net_wealth_ratio = max(1.0 - execution_cost, 0.0) * gross_wealth_ratio

        gross_reward_component = _score_step_utility(gross_wealth_ratio, config, config.gain_reward, config.loss_penalty)
        reward_core_component = _score_step_utility(net_wealth_ratio, config, config.gain_reward, config.loss_penalty)
        execution_cost_reward_drag = reward_core_component - gross_reward_component

        benchmark_reward_component = 0.0
        benchmark_return = None
        relative_wealth_ratio = None
        pre_trade_relative_wealth_ratio = None
        if aligned_benchmark is not None:
            benchmark_return = float(aligned_benchmark.iloc[step_index])
            pre_trade_relative_wealth_ratio = current_wealth / max(current_benchmark_wealth, config.epsilon)
            if config.benchmark_gain_reward > 0 or config.benchmark_loss_penalty > 0:
                benchmark_wealth_ratio = max(1.0 + benchmark_return, config.epsilon)
                relative_wealth_ratio = net_wealth_ratio / benchmark_wealth_ratio
                benchmark_reward_component = _score_step_utility(
                    relative_wealth_ratio,
                    config,
                    config.benchmark_gain_reward,
                    config.benchmark_loss_penalty,
                )

        turnover_penalty_component = -config.turnover_penalty * turnover if config.turnover_penalty > 0 else 0.0
        weight_reg_penalty_component = -config.weight_reg * float(target_weights.dot(target_weights)) if config.weight_reg > 0 else 0.0

        cash_weight = float(target_weights.get(config.cash_symbol, 0.0))
        excess_cash_weight = 0.0
        cash_weight_penalty_component = 0.0
        if resolved_cash_target_weight is not None:
            excess_cash_weight = max(cash_weight - resolved_cash_target_weight, 0.0)
            if cash_weight_penalty > 0.0:
                cash_weight_penalty_component = -cash_weight_penalty * excess_cash_weight

        friction_reward_drag = execution_cost_reward_drag + turnover_penalty_component
        reward = reward_core_component + benchmark_reward_component + turnover_penalty_component + weight_reg_penalty_component + cash_weight_penalty_component

        current_wealth *= net_wealth_ratio
        if benchmark_return is not None:
            current_benchmark_wealth *= max(1.0 + benchmark_return, config.epsilon)
        peak_wealth = max(peak_wealth, current_wealth)
        current_weights = _update_post_return_weights(target_weights, realized_returns, config.epsilon)
        ending_weights = current_weights.copy()
        if turnover > 1e-12:
            executed_rebalances += 1

        record: dict[str, float | str | bool] = {
            "step_label": str(step_label),
            "normalized_reward": float(reward),
            "raw_reward": float(reward),
            "gross_portfolio_return": float(gross_portfolio_return),
            "portfolio_return": float(net_wealth_ratio - 1.0),
            "gross_wealth_ratio": float(gross_wealth_ratio),
            "net_wealth_ratio": float(net_wealth_ratio),
            "proposed_turnover": float(proposed_turnover),
            "turnover": float(turnover),
            "trade_suppressed": bool(budget_trade_suppressed),
            "execution_cost": float(execution_cost),
            "wealth": float(current_wealth),
            "gross_reward_component": float(gross_reward_component),
            "reward_core_component": float(reward_core_component),
            "benchmark_reward_component": float(benchmark_reward_component),
            "execution_cost_reward_drag": float(execution_cost_reward_drag),
            "turnover_penalty_component": float(turnover_penalty_component),
            "weight_reg_penalty_component": float(weight_reg_penalty_component),
            "cash_weight": float(cash_weight),
            "excess_cash_weight": float(excess_cash_weight),
            "cash_weight_penalty_component": float(cash_weight_penalty_component),
            "friction_reward_drag": float(friction_reward_drag),
            "rebalances_executed": float(executed_rebalances),
            "rebalance_budget_exhausted": bool(max_rebalances is not None and executed_rebalances >= max_rebalances),
        }
        if max_rebalances is not None:
            record["rebalance_budget"] = float(max_rebalances)
        if benchmark_return is not None:
            record["benchmark_return"] = float(benchmark_return)
        if pre_trade_relative_wealth_ratio is not None:
            record["pre_trade_relative_wealth_ratio"] = float(pre_trade_relative_wealth_ratio)
        if resolved_cash_target_weight is not None:
            record["cash_target_weight"] = float(resolved_cash_target_weight)
        if relative_wealth_ratio is not None:
            record["relative_wealth_ratio"] = float(relative_wealth_ratio)
        record.update(_flatten_weight_mapping("asset_return_", realized_returns.to_dict()))
        record.update(_flatten_weight_mapping("gross_return_contribution_", gross_return_contributions.to_dict()))
        record.update(_flatten_weight_mapping("pre_trade_weight_", pre_trade_weights.to_dict()))
        record.update(_flatten_weight_mapping("proposed_weight_", proposed_weights.to_dict()))
        record.update(_flatten_weight_mapping("target_weight_", target_weights.to_dict()))
        record.update(_flatten_weight_mapping("ending_weight_", ending_weights.to_dict()))
        records.append(record)

    frame = pd.DataFrame(records, index=policy_rollout_records.index.copy())
    frame.index.name = "date"
    return frame


def _simulate_static_hold_rollout_records(
    policy_rollout_records: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark_returns: pd.Series | None,
    config: WealthFirstConfig,
    cash_weight_penalty: float = 0.0,
    cash_target_weight: float | None = None,
) -> pd.DataFrame:
    if policy_rollout_records.empty:
        raise ValueError("Policy rollout records are empty.")

    aligned_returns = _augment_aligned_returns_for_rollout(
        _align_frame_to_rollout_records(returns, policy_rollout_records),
        policy_rollout_records,
        config.cash_symbol,
    )
    aligned_benchmark = (
        _align_series_to_rollout_records(benchmark_returns, policy_rollout_records)
        if benchmark_returns is not None
        else None
    )
    current_weights = _extract_rollout_weight_series(policy_rollout_records, "pre_trade_weight_", aligned_returns.columns)
    initial_target_weights = _extract_rollout_weight_series(policy_rollout_records, "target_weight_", aligned_returns.columns)
    resolved_cash_target_weight = _resolve_cash_target_weight(config, cash_target_weight)

    current_wealth = 1.0
    current_benchmark_wealth = 1.0
    peak_wealth = 1.0
    records: list[dict[str, float | str | bool]] = []

    for step_index, (step_label, realized_returns) in enumerate(aligned_returns.iterrows()):
        pre_trade_weights = current_weights.copy()
        if step_index == 0:
            proposed_weights = initial_target_weights.copy().rename("proposed_weight")
            target_weights = initial_target_weights.copy().rename("target_weight")
        else:
            proposed_weights = current_weights.copy().rename("proposed_weight")
            target_weights = current_weights.copy().rename("target_weight")

        proposed_turnover = compute_tradable_turnover(proposed_weights, pre_trade_weights, config.cash_symbol)
        turnover = compute_tradable_turnover(target_weights, pre_trade_weights, config.cash_symbol)
        execution_cost = compute_execution_cost(turnover, config.transaction_cost_bps, config.slippage_bps)
        gross_return_contributions = (target_weights * realized_returns).rename("gross_return_contribution")
        gross_portfolio_return = float(gross_return_contributions.sum())
        gross_wealth_ratio = max(1.0 + gross_portfolio_return, config.epsilon)
        net_wealth_ratio = max(1.0 - execution_cost, 0.0) * gross_wealth_ratio

        gross_reward_component = _score_step_utility(gross_wealth_ratio, config, config.gain_reward, config.loss_penalty)
        reward_core_component = _score_step_utility(net_wealth_ratio, config, config.gain_reward, config.loss_penalty)
        execution_cost_reward_drag = reward_core_component - gross_reward_component

        benchmark_reward_component = 0.0
        benchmark_return = None
        relative_wealth_ratio = None
        pre_trade_relative_wealth_ratio = None
        if aligned_benchmark is not None:
            benchmark_return = float(aligned_benchmark.iloc[step_index])
            pre_trade_relative_wealth_ratio = current_wealth / max(current_benchmark_wealth, config.epsilon)
            if config.benchmark_gain_reward > 0 or config.benchmark_loss_penalty > 0:
                benchmark_wealth_ratio = max(1.0 + benchmark_return, config.epsilon)
                relative_wealth_ratio = net_wealth_ratio / benchmark_wealth_ratio
                benchmark_reward_component = _score_step_utility(
                    relative_wealth_ratio,
                    config,
                    config.benchmark_gain_reward,
                    config.benchmark_loss_penalty,
                )

        turnover_penalty_component = -config.turnover_penalty * turnover if config.turnover_penalty > 0 else 0.0
        weight_reg_penalty_component = -config.weight_reg * float(target_weights.dot(target_weights)) if config.weight_reg > 0 else 0.0

        cash_weight = float(target_weights.get(config.cash_symbol, 0.0))
        excess_cash_weight = 0.0
        cash_weight_penalty_component = 0.0
        if resolved_cash_target_weight is not None:
            excess_cash_weight = max(cash_weight - resolved_cash_target_weight, 0.0)
            if cash_weight_penalty > 0.0:
                cash_weight_penalty_component = -cash_weight_penalty * excess_cash_weight

        friction_reward_drag = execution_cost_reward_drag + turnover_penalty_component
        reward = reward_core_component + benchmark_reward_component + turnover_penalty_component + weight_reg_penalty_component + cash_weight_penalty_component

        current_wealth *= net_wealth_ratio
        if benchmark_return is not None:
            current_benchmark_wealth *= max(1.0 + benchmark_return, config.epsilon)
        peak_wealth = max(peak_wealth, current_wealth)
        current_weights = _update_post_return_weights(target_weights, realized_returns, config.epsilon)
        ending_weights = current_weights.copy()

        record: dict[str, float | str | bool] = {
            "step_label": str(step_label),
            "normalized_reward": float(reward),
            "raw_reward": float(reward),
            "gross_portfolio_return": float(gross_portfolio_return),
            "portfolio_return": float(net_wealth_ratio - 1.0),
            "gross_wealth_ratio": float(gross_wealth_ratio),
            "net_wealth_ratio": float(net_wealth_ratio),
            "proposed_turnover": float(proposed_turnover),
            "turnover": float(turnover),
            "trade_suppressed": False,
            "execution_cost": float(execution_cost),
            "wealth": float(current_wealth),
            "gross_reward_component": float(gross_reward_component),
            "reward_core_component": float(reward_core_component),
            "benchmark_reward_component": float(benchmark_reward_component),
            "execution_cost_reward_drag": float(execution_cost_reward_drag),
            "turnover_penalty_component": float(turnover_penalty_component),
            "weight_reg_penalty_component": float(weight_reg_penalty_component),
            "cash_weight": float(cash_weight),
            "excess_cash_weight": float(excess_cash_weight),
            "cash_weight_penalty_component": float(cash_weight_penalty_component),
            "friction_reward_drag": float(friction_reward_drag),
        }
        if benchmark_return is not None:
            record["benchmark_return"] = float(benchmark_return)
        if pre_trade_relative_wealth_ratio is not None:
            record["pre_trade_relative_wealth_ratio"] = float(pre_trade_relative_wealth_ratio)
        if resolved_cash_target_weight is not None:
            record["cash_target_weight"] = float(resolved_cash_target_weight)
        if relative_wealth_ratio is not None:
            record["relative_wealth_ratio"] = float(relative_wealth_ratio)
        record.update(_flatten_weight_mapping("asset_return_", realized_returns.to_dict()))
        record.update(_flatten_weight_mapping("gross_return_contribution_", gross_return_contributions.to_dict()))
        record.update(_flatten_weight_mapping("pre_trade_weight_", pre_trade_weights.to_dict()))
        record.update(_flatten_weight_mapping("proposed_weight_", proposed_weights.to_dict()))
        record.update(_flatten_weight_mapping("target_weight_", target_weights.to_dict()))
        record.update(_flatten_weight_mapping("ending_weight_", ending_weights.to_dict()))
        records.append(record)

    frame = pd.DataFrame(records, index=policy_rollout_records.index.copy())
    frame.index.name = "date"
    return frame


def _simulate_passive_interval_outcome(
    initial_weights: pd.Series,
    interval_returns: pd.DataFrame,
    config: WealthFirstConfig,
    initial_turnover: float = 0.0,
    interval_benchmark_returns: pd.Series | None = None,
) -> dict[str, float | pd.Series | None]:
    if interval_returns.empty:
        raise ValueError("Interval returns are empty.")

    current_weights = initial_weights.copy().astype(float)
    wealth = 1.0
    gross_wealth = 1.0
    benchmark_wealth = 1.0 if interval_benchmark_returns is not None else None
    first_step_return: float | None = None
    first_step_gross_return: float | None = None
    first_step_execution_cost = compute_execution_cost(initial_turnover, config.transaction_cost_bps, config.slippage_bps)

    for step_index, (_, realized_returns) in enumerate(interval_returns.iterrows()):
        execution_cost = first_step_execution_cost if step_index == 0 else 0.0
        gross_portfolio_return = float((current_weights * realized_returns).sum())
        gross_wealth_ratio = max(1.0 + gross_portfolio_return, config.epsilon)
        net_wealth_ratio = max(1.0 - execution_cost, 0.0) * gross_wealth_ratio

        if step_index == 0:
            first_step_return = float(net_wealth_ratio - 1.0)
            first_step_gross_return = float(gross_portfolio_return)

        gross_wealth *= gross_wealth_ratio
        wealth *= net_wealth_ratio
        current_weights = _update_post_return_weights(current_weights, realized_returns, config.epsilon)

        if benchmark_wealth is not None and interval_benchmark_returns is not None:
            benchmark_return = float(interval_benchmark_returns.iloc[step_index])
            benchmark_wealth *= max(1.0 + benchmark_return, config.epsilon)

    outcome: dict[str, float | pd.Series | None] = {
        "total_return": float(wealth - 1.0),
        "gross_total_return": float(gross_wealth - 1.0),
        "first_step_return": float(first_step_return if first_step_return is not None else 0.0),
        "first_step_gross_return": float(first_step_gross_return if first_step_gross_return is not None else 0.0),
        "ending_weights": current_weights,
    }
    if benchmark_wealth is not None:
        outcome["relative_total_return"] = float(wealth / benchmark_wealth - 1.0)
    return outcome


def _compute_rebalance_impact_records(
    policy_rollout_records: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark_returns: pd.Series | None,
    config: WealthFirstConfig,
) -> pd.DataFrame:
    if policy_rollout_records.empty:
        raise ValueError("Policy rollout records are empty.")

    aligned_returns = _augment_aligned_returns_for_rollout(
        _align_frame_to_rollout_records(returns, policy_rollout_records),
        policy_rollout_records,
        config.cash_symbol,
    )
    aligned_benchmark = (
        _align_series_to_rollout_records(benchmark_returns, policy_rollout_records)
        if benchmark_returns is not None
        else None
    )
    executed_indices = [
        row_index
        for row_index, turnover in enumerate(policy_rollout_records["turnover"].astype(float))
        if turnover > 1e-12
    ]
    if not executed_indices:
        return pd.DataFrame(columns=[
            "rebalance_number",
            "step_label",
            "interval_end_label",
            "interval_steps",
            "turnover",
            "proposed_turnover",
            "first_step_return_delta",
            "interval_total_return_delta",
        ])

    records: list[dict[str, float | str | None]] = []
    asset_columns = aligned_returns.columns

    for rebalance_number, start_index in enumerate(executed_indices, start=1):
        end_index = executed_indices[rebalance_number] if rebalance_number < len(executed_indices) else len(policy_rollout_records)
        interval_returns = aligned_returns.iloc[start_index:end_index]
        interval_benchmark = aligned_benchmark.iloc[start_index:end_index] if aligned_benchmark is not None else None
        row = policy_rollout_records.iloc[start_index]
        pre_trade_weights = _extract_rollout_weight_row(row, "pre_trade_weight_", asset_columns)
        target_weights = _extract_rollout_weight_row(row, "target_weight_", asset_columns)

        policy_outcome = _simulate_passive_interval_outcome(
            target_weights,
            interval_returns,
            config,
            initial_turnover=float(row["turnover"]),
            interval_benchmark_returns=interval_benchmark,
        )
        hold_outcome = _simulate_passive_interval_outcome(
            pre_trade_weights,
            interval_returns,
            config,
            initial_turnover=0.0,
            interval_benchmark_returns=interval_benchmark,
        )
        interval_end_label = interval_returns.index[-1]
        record: dict[str, float | str | None] = {
            "rebalance_number": float(rebalance_number),
            "step_label": str(row.get("step_label", interval_returns.index[0])),
            "interval_end_label": str(interval_end_label),
            "interval_steps": float(len(interval_returns)),
            "turnover": float(row["turnover"]),
            "proposed_turnover": float(row.get("proposed_turnover", row["turnover"])),
            "policy_first_step_return": float(policy_outcome["first_step_return"]),
            "hold_first_step_return": float(hold_outcome["first_step_return"]),
            "first_step_return_delta": float(policy_outcome["first_step_return"]) - float(hold_outcome["first_step_return"]),
            "policy_interval_total_return": float(policy_outcome["total_return"]),
            "hold_interval_total_return": float(hold_outcome["total_return"]),
            "interval_total_return_delta": float(policy_outcome["total_return"])
            - float(hold_outcome["total_return"]),
            "policy_interval_gross_total_return": float(policy_outcome["gross_total_return"]),
            "hold_interval_gross_total_return": float(hold_outcome["gross_total_return"]),
            "interval_gross_total_return_delta": float(policy_outcome["gross_total_return"])
            - float(hold_outcome["gross_total_return"]),
        }
        if "relative_total_return" in policy_outcome and "relative_total_return" in hold_outcome:
            record["policy_interval_relative_total_return"] = float(policy_outcome["relative_total_return"])
            record["hold_interval_relative_total_return"] = float(hold_outcome["relative_total_return"])
            record["interval_relative_total_return_delta"] = float(policy_outcome["relative_total_return"]) - float(hold_outcome["relative_total_return"])
        record.update(_flatten_weight_mapping("pre_trade_weight_", pre_trade_weights.to_dict()))
        record.update(_flatten_weight_mapping("target_weight_", target_weights.to_dict()))
        records.append(record)

    frame = pd.DataFrame(records)
    frame.index = pd.Index(range(1, len(frame) + 1), name="rebalance_event")
    return frame


def _summarize_rebalance_impact_records(records: pd.DataFrame) -> pd.Series:
    if records.empty:
        return pd.Series(
            {
                "rebalance_count": 0.0,
                "average_turnover": 0.0,
                "average_interval_steps": 0.0,
                "average_first_step_return_delta": 0.0,
                "first_step_win_rate": 0.0,
                "average_interval_total_return_delta": 0.0,
                "interval_win_rate": 0.0,
            },
            dtype=float,
        )

    summary = pd.Series(dtype=float)
    summary.loc["rebalance_count"] = float(len(records))
    summary.loc["average_turnover"] = float(records["turnover"].astype(float).mean())
    summary.loc["average_proposed_turnover"] = float(records["proposed_turnover"].astype(float).mean())
    summary.loc["average_interval_steps"] = float(records["interval_steps"].astype(float).mean())
    summary.loc["average_first_step_return_delta"] = float(records["first_step_return_delta"].astype(float).mean())
    summary.loc["first_step_win_rate"] = float((records["first_step_return_delta"].astype(float) > 0.0).mean())
    summary.loc["average_interval_total_return_delta"] = float(records["interval_total_return_delta"].astype(float).mean())
    summary.loc["interval_win_rate"] = float((records["interval_total_return_delta"].astype(float) > 0.0).mean())
    summary.loc["average_interval_gross_total_return_delta"] = float(records["interval_gross_total_return_delta"].astype(float).mean())
    if "interval_relative_total_return_delta" in records.columns:
        summary.loc["average_interval_relative_total_return_delta"] = float(records["interval_relative_total_return_delta"].astype(float).mean())

    group_masks = {
        "first_rebalance": records["rebalance_number"].astype(float) == 1.0,
        "second_rebalance": records["rebalance_number"].astype(float) == 2.0,
        "third_rebalance": records["rebalance_number"].astype(float) == 3.0,
        "post_first_rebalances": records["rebalance_number"].astype(float) > 1.0,
        "post_third_rebalances": records["rebalance_number"].astype(float) > 3.0,
    }
    for label, mask in group_masks.items():
        subset = records.loc[mask]
        if subset.empty:
            continue
        summary.loc[f"{label}_count"] = float(len(subset))
        summary.loc[f"{label}_average_interval_total_return_delta"] = float(subset["interval_total_return_delta"].astype(float).mean())
        summary.loc[f"{label}_interval_win_rate"] = float((subset["interval_total_return_delta"].astype(float) > 0.0).mean())

    return summary


def _rollout_policy_records(model, env, deterministic: bool = True) -> pd.DataFrame:
    observation = env.reset()
    records: list[dict[str, float | str]] = []

    while True:
        action, _ = model.predict(observation, deterministic=deterministic)
        observation, rewards, dones, infos = env.step(action)
        info = infos[0]
        record: dict[str, float | str] = {
            "step_label": str(info["step_label"]),
            "normalized_reward": float(rewards[0]),
            "raw_reward": float(info.get("raw_reward", rewards[0])),
            "gross_portfolio_return": float(info["gross_portfolio_return"]),
            "portfolio_return": float(info["portfolio_return"]),
            "gross_wealth_ratio": float(info.get("gross_wealth_ratio", 1.0 + info["gross_portfolio_return"])),
            "net_wealth_ratio": float(info["net_wealth_ratio"]),
            "proposed_turnover": float(info.get("proposed_turnover", info["turnover"])),
            "turnover": float(info["turnover"]),
            "trade_suppressed": bool(info.get("trade_suppressed", False)),
            "rebalance_budget_exhausted": bool(info.get("rebalance_budget_exhausted", False)),
            "rebalance_cooldown_active": bool(info.get("rebalance_cooldown_active", False)),
            "rebalance_cooldown_blocked": bool(info.get("rebalance_cooldown_blocked", False)),
            "early_rebalance_risk_window_active": bool(info.get("early_rebalance_risk_window_active", False)),
            "early_rebalance_risk_turnover_cap_window_active": bool(
                info.get("early_rebalance_risk_turnover_cap_window_active", False)
            ),
            "early_rebalance_risk_turnover_cap_condition_met": bool(
                info.get("early_rebalance_risk_turnover_cap_condition_met", False)
            ),
            "early_rebalance_risk_turnover_cap_applied": bool(info.get("early_rebalance_risk_turnover_cap_applied", False)),
            "early_rebalance_risk_shallow_drawdown_turnover_cap_window_active": bool(
                info.get("early_rebalance_risk_shallow_drawdown_turnover_cap_window_active", False)
            ),
            "early_rebalance_risk_shallow_drawdown_turnover_cap_condition_met": bool(
                info.get("early_rebalance_risk_shallow_drawdown_turnover_cap_condition_met", False)
            ),
            "early_rebalance_risk_shallow_drawdown_turnover_cap_applied": bool(
                info.get("early_rebalance_risk_shallow_drawdown_turnover_cap_applied", False)
            ),
            "early_rebalance_risk_mean_reversion_turnover_cap_window_active": bool(
                info.get("early_rebalance_risk_mean_reversion_turnover_cap_window_active", False)
            ),
            "early_rebalance_risk_mean_reversion_turnover_cap_condition_met": bool(
                info.get("early_rebalance_risk_mean_reversion_turnover_cap_condition_met", False)
            ),
            "early_rebalance_risk_mean_reversion_action_smoothing_applied": bool(
                info.get("early_rebalance_risk_mean_reversion_action_smoothing_applied", False)
            ),
            "early_rebalance_risk_mean_reversion_turnover_cap_applied": bool(
                info.get("early_rebalance_risk_mean_reversion_turnover_cap_applied", False)
            ),
            "early_rebalance_risk_trend_turnover_cap_window_active": bool(
                info.get("early_rebalance_risk_trend_turnover_cap_window_active", False)
            ),
            "early_rebalance_risk_trend_turnover_cap_condition_met": bool(
                info.get("early_rebalance_risk_trend_turnover_cap_condition_met", False)
            ),
            "early_rebalance_risk_trend_turnover_cap_applied": bool(
                info.get("early_rebalance_risk_trend_turnover_cap_applied", False)
            ),
            "early_rebalance_risk_deep_drawdown_turnover_cap_window_active": bool(
                info.get("early_rebalance_risk_deep_drawdown_turnover_cap_window_active", False)
            ),
            "early_rebalance_risk_deep_drawdown_turnover_cap_condition_met": bool(
                info.get("early_rebalance_risk_deep_drawdown_turnover_cap_condition_met", False)
            ),
            "early_rebalance_risk_deep_drawdown_turnover_cap_applied": bool(
                info.get("early_rebalance_risk_deep_drawdown_turnover_cap_applied", False)
            ),
            "early_rebalance_risk_repeat_turnover_cap_window_active": bool(
                info.get("early_rebalance_risk_repeat_turnover_cap_window_active", False)
            ),
            "early_rebalance_risk_repeat_turnover_cap_condition_met": bool(
                info.get("early_rebalance_risk_repeat_turnover_cap_condition_met", False)
            ),
            "early_rebalance_risk_repeat_action_smoothing_applied": bool(
                info.get("early_rebalance_risk_repeat_action_smoothing_applied", False)
            ),
            "early_rebalance_risk_repeat_turnover_cap_applied": bool(
                info.get("early_rebalance_risk_repeat_turnover_cap_applied", False)
            ),
            "early_rebalance_risk_repeat_unrecovered_turnover_cap_window_active": bool(
                info.get("early_rebalance_risk_repeat_unrecovered_turnover_cap_window_active", False)
            ),
            "early_rebalance_risk_repeat_unrecovered_turnover_cap_condition_met": bool(
                info.get("early_rebalance_risk_repeat_unrecovered_turnover_cap_condition_met", False)
            ),
            "early_rebalance_risk_repeat_unrecovered_turnover_cap_applied": bool(
                info.get("early_rebalance_risk_repeat_unrecovered_turnover_cap_applied", False)
            ),
            "early_rebalance_risk_cumulative_turnover_cap_window_active": bool(
                info.get("early_rebalance_risk_cumulative_turnover_cap_window_active", False)
            ),
            "early_rebalance_risk_cumulative_turnover_cap_condition_met": bool(
                info.get("early_rebalance_risk_cumulative_turnover_cap_condition_met", False)
            ),
            "early_rebalance_risk_cumulative_turnover_cap_applied": bool(
                info.get("early_rebalance_risk_cumulative_turnover_cap_applied", False)
            ),
            "early_rebalance_risk_penalty_applied": bool(info.get("early_rebalance_risk_penalty_applied", False)),
            "early_rebalance_risk_condition_met": bool(info.get("early_rebalance_risk_condition_met", False)),
            "early_benchmark_euphoria_window_active": bool(info.get("early_benchmark_euphoria_window_active", False)),
            "early_benchmark_euphoria_turnover_cap_applied": bool(
                info.get("early_benchmark_euphoria_turnover_cap_applied", False)
            ),
            "early_benchmark_euphoria_penalty_applied": bool(info.get("early_benchmark_euphoria_penalty_applied", False)),
            "early_benchmark_euphoria_condition_met": bool(info.get("early_benchmark_euphoria_condition_met", False)),
            "late_rebalance_penalty_applied": bool(info.get("late_rebalance_penalty_applied", False)),
            "late_rebalance_threshold_reached": bool(info.get("late_rebalance_threshold_reached", False)),
            "late_rebalance_gate_active": bool(info.get("late_rebalance_gate_active", False)),
            "late_rebalance_gate_blocked": bool(info.get("late_rebalance_gate_blocked", False)),
            "late_rebalance_gate_condition_met": bool(info.get("late_rebalance_gate_condition_met", False)),
            "late_rebalance_gate_threshold_reached": bool(info.get("late_rebalance_gate_threshold_reached", False)),
            "late_defensive_posture_window_active": bool(info.get("late_defensive_posture_window_active", False)),
            "late_defensive_posture_condition_met": bool(info.get("late_defensive_posture_condition_met", False)),
            "late_defensive_posture_penalty_applied": bool(info.get("late_defensive_posture_penalty_applied", False)),
            "late_trend_mean_reversion_conflict_window_active": bool(
                info.get("late_trend_mean_reversion_conflict_window_active", False)
            ),
            "late_trend_mean_reversion_conflict_condition_met": bool(
                info.get("late_trend_mean_reversion_conflict_condition_met", False)
            ),
            "late_trend_mean_reversion_conflict_penalty_applied": bool(
                info.get("late_trend_mean_reversion_conflict_penalty_applied", False)
            ),
            "state_trend_preservation_window_active": bool(info.get("state_trend_preservation_window_active", False)),
            "state_trend_preservation_condition_met": bool(info.get("state_trend_preservation_condition_met", False)),
            "state_trend_preservation_guard_applied": bool(info.get("state_trend_preservation_guard_applied", False)),
            "executed_rebalances": float(info.get("executed_rebalances", 0.0)),
            "execution_cost": float(info["execution_cost"]),
            "wealth": float(info["wealth"]),
            "gross_reward_component": float(info.get("gross_reward_component", info.get("raw_reward", rewards[0]))),
            "reward_core_component": float(info.get("reward_core_component", info.get("raw_reward", rewards[0]))),
            "benchmark_reward_component": float(info.get("benchmark_reward_component", 0.0)),
            "execution_cost_reward_drag": float(info.get("execution_cost_reward_drag", 0.0)),
            "turnover_penalty_component": float(info.get("turnover_penalty_component", 0.0)),
            "weight_reg_penalty_component": float(info.get("weight_reg_penalty_component", 0.0)),
            "cash_weight": float(info.get("cash_weight", 0.0)),
            "excess_cash_weight": float(info.get("excess_cash_weight", 0.0)),
            "cash_weight_penalty_component": float(info.get("cash_weight_penalty_component", 0.0)),
            "early_rebalance_risk_penalty_component": float(info.get("early_rebalance_risk_penalty_component", 0.0)),
            "early_benchmark_euphoria_penalty_component": float(
                info.get("early_benchmark_euphoria_penalty_component", 0.0)
            ),
            "late_rebalance_penalty_component": float(info.get("late_rebalance_penalty_component", 0.0)),
            "late_defensive_posture_penalty_component": float(info.get("late_defensive_posture_penalty_component", 0.0)),
            "late_trend_mean_reversion_conflict_penalty_component": float(
                info.get("late_trend_mean_reversion_conflict_penalty_component", 0.0)
            ),
            "friction_reward_drag": float(info.get("friction_reward_drag", 0.0)),
        }
        if "benchmark_return" in info:
            record["benchmark_return"] = float(info["benchmark_return"])
        if info.get("cash_target_weight") is not None:
            record["cash_target_weight"] = float(info["cash_target_weight"])
        if info.get("max_executed_rebalances") is not None:
            record["max_executed_rebalances"] = float(info["max_executed_rebalances"])
        if info.get("rebalance_cooldown_steps") is not None:
            record["rebalance_cooldown_steps"] = float(info["rebalance_cooldown_steps"])
        if "rebalance_cooldown_remaining" in info:
            record["rebalance_cooldown_remaining"] = float(info["rebalance_cooldown_remaining"])
        if info.get("early_rebalance_risk_penalty") is not None:
            record["early_rebalance_risk_penalty"] = float(info["early_rebalance_risk_penalty"])
        if info.get("early_rebalance_risk_turnover_cap") is not None:
            record["early_rebalance_risk_turnover_cap"] = float(info["early_rebalance_risk_turnover_cap"])
        if info.get("early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold") is not None:
            record["early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold"] = float(
                info["early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold"]
            )
        if info.get("early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold") is not None:
            record["early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold"] = float(
                info["early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold"]
            )
        if info.get("early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight") is not None:
            record["early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight"] = float(
                info["early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight"]
            )
        if info.get("early_rebalance_risk_turnover_cap_target_cash_min_threshold") is not None:
            record["early_rebalance_risk_turnover_cap_target_cash_min_threshold"] = float(
                info["early_rebalance_risk_turnover_cap_target_cash_min_threshold"]
            )
        if info.get("early_rebalance_risk_turnover_cap_target_cash_max_threshold") is not None:
            record["early_rebalance_risk_turnover_cap_target_cash_max_threshold"] = float(
                info["early_rebalance_risk_turnover_cap_target_cash_max_threshold"]
            )
        if info.get("early_rebalance_risk_turnover_cap_target_trend_min_threshold") is not None:
            record["early_rebalance_risk_turnover_cap_target_trend_min_threshold"] = float(
                info["early_rebalance_risk_turnover_cap_target_trend_min_threshold"]
            )
        if info.get("early_rebalance_risk_turnover_cap_target_trend_max_threshold") is not None:
            record["early_rebalance_risk_turnover_cap_target_trend_max_threshold"] = float(
                info["early_rebalance_risk_turnover_cap_target_trend_max_threshold"]
            )
        if info.get("early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold") is not None:
            record["early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold"] = float(
                info["early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold"]
            )
        if info.get("early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold") is not None:
            record["early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold"] = float(
                info["early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold"]
            )
        if info.get("early_rebalance_risk_turnover_cap_delta_cash_min_threshold") is not None:
            record["early_rebalance_risk_turnover_cap_delta_cash_min_threshold"] = float(
                info["early_rebalance_risk_turnover_cap_delta_cash_min_threshold"]
            )
        if info.get("early_rebalance_risk_turnover_cap_delta_cash_max_threshold") is not None:
            record["early_rebalance_risk_turnover_cap_delta_cash_max_threshold"] = float(
                info["early_rebalance_risk_turnover_cap_delta_cash_max_threshold"]
            )
        if info.get("early_rebalance_risk_turnover_cap_delta_trend_min_threshold") is not None:
            record["early_rebalance_risk_turnover_cap_delta_trend_min_threshold"] = float(
                info["early_rebalance_risk_turnover_cap_delta_trend_min_threshold"]
            )
        if info.get("early_rebalance_risk_turnover_cap_delta_trend_max_threshold") is not None:
            record["early_rebalance_risk_turnover_cap_delta_trend_max_threshold"] = float(
                info["early_rebalance_risk_turnover_cap_delta_trend_max_threshold"]
            )
        if info.get("early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold") is not None:
            record["early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold"] = float(
                info["early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold"]
            )
        if info.get("early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold") is not None:
            record["early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold"] = float(
                info["early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold"]
            )
        if info.get("early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold") is not None:
            record["early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold"] = float(
                info["early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold"]
            )
        if info.get("early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold") is not None:
            record["early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold"] = float(
                info["early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold"]
            )
        if info.get("early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio") is not None:
            record["early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio"] = float(
                info["early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio"]
            )
        if info.get("early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio") is not None:
            record["early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio"] = float(
                info["early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio"]
            )
        record["early_rebalance_risk_turnover_cap_allow_nonincreasing_risk_symbol"] = bool(
            info.get("early_rebalance_risk_turnover_cap_allow_nonincreasing_risk_symbol", False)
        )
        record["early_rebalance_risk_turnover_cap_use_penalty_state_filters"] = bool(
            info.get("early_rebalance_risk_turnover_cap_use_penalty_state_filters", False)
        )
        if info.get("early_rebalance_risk_turnover_cap_after") is not None:
            record["early_rebalance_risk_turnover_cap_after"] = float(info["early_rebalance_risk_turnover_cap_after"])
        if info.get("early_rebalance_risk_turnover_cap_before") is not None:
            record["early_rebalance_risk_turnover_cap_before"] = float(info["early_rebalance_risk_turnover_cap_before"])
        if info.get("early_rebalance_risk_turnover_cap_max_applications") is not None:
            record["early_rebalance_risk_turnover_cap_max_applications"] = float(
                info["early_rebalance_risk_turnover_cap_max_applications"]
            )
        if info.get("early_rebalance_risk_turnover_cap_secondary_cap") is not None:
            record["early_rebalance_risk_turnover_cap_secondary_cap"] = float(
                info["early_rebalance_risk_turnover_cap_secondary_cap"]
            )
        if info.get("early_rebalance_risk_turnover_cap_secondary_after_applications") is not None:
            record["early_rebalance_risk_turnover_cap_secondary_after_applications"] = float(
                info["early_rebalance_risk_turnover_cap_secondary_after_applications"]
            )
        if info.get("early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold") is not None:
            record[
                "early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold"
            ] = float(
                info["early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold"]
            )
        if info.get("early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio") is not None:
            record[
                "early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio"
            ] = float(
                info["early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio"]
            )
        if info.get("early_rebalance_risk_turnover_cap_applications") is not None:
            record["early_rebalance_risk_turnover_cap_applications"] = float(
                info["early_rebalance_risk_turnover_cap_applications"]
            )
        if info.get("early_rebalance_risk_turnover_cap_max_applications_reached") is not None:
            record["early_rebalance_risk_turnover_cap_max_applications_reached"] = bool(
                info["early_rebalance_risk_turnover_cap_max_applications_reached"]
            )
        if info.get("early_rebalance_risk_turnover_cap_secondary_active") is not None:
            record["early_rebalance_risk_turnover_cap_secondary_active"] = bool(
                info["early_rebalance_risk_turnover_cap_secondary_active"]
            )
            record["early_rebalance_risk_turnover_cap_secondary_applied"] = bool(
                info["early_rebalance_risk_turnover_cap_secondary_active"]
                and info.get("early_rebalance_risk_turnover_cap_applied", False)
            )
        if info.get("early_rebalance_risk_turnover_cap_secondary_after_applications_reached") is not None:
            record["early_rebalance_risk_turnover_cap_secondary_after_applications_reached"] = bool(
                info["early_rebalance_risk_turnover_cap_secondary_after_applications_reached"]
            )
        if info.get("early_rebalance_risk_turnover_cap_secondary_state_condition_met") is not None:
            record["early_rebalance_risk_turnover_cap_secondary_state_condition_met"] = bool(
                info["early_rebalance_risk_turnover_cap_secondary_state_condition_met"]
            )
        if info.get("early_rebalance_risk_turnover_cap_effective_cap") is not None:
            record["early_rebalance_risk_turnover_cap_effective_cap"] = float(
                info["early_rebalance_risk_turnover_cap_effective_cap"]
            )
        if info.get("early_rebalance_risk_shallow_drawdown_turnover_cap") is not None:
            record["early_rebalance_risk_shallow_drawdown_turnover_cap"] = float(
                info["early_rebalance_risk_shallow_drawdown_turnover_cap"]
            )
        if info.get("early_rebalance_risk_shallow_drawdown_turnover_cap_after") is not None:
            record["early_rebalance_risk_shallow_drawdown_turnover_cap_after"] = float(
                info["early_rebalance_risk_shallow_drawdown_turnover_cap_after"]
            )
        if info.get("early_rebalance_risk_shallow_drawdown_turnover_cap_before") is not None:
            record["early_rebalance_risk_shallow_drawdown_turnover_cap_before"] = float(
                info["early_rebalance_risk_shallow_drawdown_turnover_cap_before"]
            )
        if info.get("early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold") is not None:
            record["early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold"] = float(
                info["early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold"]
            )
        if info.get("early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold") is not None:
            record["early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold"] = float(
                info["early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold"]
            )
        if info.get("early_rebalance_risk_mean_reversion_turnover_cap") is not None:
            record["early_rebalance_risk_mean_reversion_turnover_cap"] = float(
                info["early_rebalance_risk_mean_reversion_turnover_cap"]
            )
        if info.get("early_rebalance_risk_mean_reversion_action_smoothing") is not None:
            record["early_rebalance_risk_mean_reversion_action_smoothing"] = float(
                info["early_rebalance_risk_mean_reversion_action_smoothing"]
            )
        if info.get("early_rebalance_risk_mean_reversion_turnover_cap_after") is not None:
            record["early_rebalance_risk_mean_reversion_turnover_cap_after"] = float(
                info["early_rebalance_risk_mean_reversion_turnover_cap_after"]
            )
        if info.get("early_rebalance_risk_mean_reversion_turnover_cap_before") is not None:
            record["early_rebalance_risk_mean_reversion_turnover_cap_before"] = float(
                info["early_rebalance_risk_mean_reversion_turnover_cap_before"]
            )
        if info.get("early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold") is not None:
            record["early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold"] = float(
                info["early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold"]
            )
        if info.get("early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold") is not None:
            record["early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold"] = float(
                info["early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold"]
            )
        if info.get("early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold") is not None:
            record["early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold"] = float(
                info["early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold"]
            )
        if info.get("early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold") is not None:
            record["early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold"] = float(
                info["early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold"]
            )
        if info.get("early_rebalance_risk_trend_turnover_cap") is not None:
            record["early_rebalance_risk_trend_turnover_cap"] = float(
                info["early_rebalance_risk_trend_turnover_cap"]
            )
        if info.get("early_rebalance_risk_trend_turnover_cap_after") is not None:
            record["early_rebalance_risk_trend_turnover_cap_after"] = float(
                info["early_rebalance_risk_trend_turnover_cap_after"]
            )
        if info.get("early_rebalance_risk_trend_turnover_cap_before") is not None:
            record["early_rebalance_risk_trend_turnover_cap_before"] = float(
                info["early_rebalance_risk_trend_turnover_cap_before"]
            )
        if info.get("early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold") is not None:
            record["early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold"] = float(
                info["early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold"]
            )
        if info.get("early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold") is not None:
            record["early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold"] = float(
                info["early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold"]
            )
        if info.get("early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold") is not None:
            record["early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold"] = float(
                info["early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold"]
            )
        if info.get("early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold") is not None:
            record["early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold"] = float(
                info["early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold"]
            )
        if info.get("early_rebalance_risk_deep_drawdown_turnover_cap") is not None:
            record["early_rebalance_risk_deep_drawdown_turnover_cap"] = float(
                info["early_rebalance_risk_deep_drawdown_turnover_cap"]
            )
        if info.get("early_rebalance_risk_deep_drawdown_turnover_cap_after") is not None:
            record["early_rebalance_risk_deep_drawdown_turnover_cap_after"] = float(
                info["early_rebalance_risk_deep_drawdown_turnover_cap_after"]
            )
        if info.get("early_rebalance_risk_deep_drawdown_turnover_cap_before") is not None:
            record["early_rebalance_risk_deep_drawdown_turnover_cap_before"] = float(
                info["early_rebalance_risk_deep_drawdown_turnover_cap_before"]
            )
        if info.get("early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold") is not None:
            record["early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold"] = float(
                info["early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold"]
            )
        if info.get("early_rebalance_risk_repeat_turnover_cap") is not None:
            record["early_rebalance_risk_repeat_turnover_cap"] = float(
                info["early_rebalance_risk_repeat_turnover_cap"]
            )
        if info.get("early_rebalance_risk_repeat_action_smoothing") is not None:
            record["early_rebalance_risk_repeat_action_smoothing"] = float(
                info["early_rebalance_risk_repeat_action_smoothing"]
            )
        if info.get("early_rebalance_risk_repeat_turnover_cap_after") is not None:
            record["early_rebalance_risk_repeat_turnover_cap_after"] = float(
                info["early_rebalance_risk_repeat_turnover_cap_after"]
            )
        if info.get("early_rebalance_risk_repeat_turnover_cap_before") is not None:
            record["early_rebalance_risk_repeat_turnover_cap_before"] = float(
                info["early_rebalance_risk_repeat_turnover_cap_before"]
            )
        if info.get("early_rebalance_risk_repeat_symbol") is not None:
            record["early_rebalance_risk_repeat_symbol"] = str(info["early_rebalance_risk_repeat_symbol"])
        if info.get("early_rebalance_risk_repeat_previous_cash_reduction_min") is not None:
            record["early_rebalance_risk_repeat_previous_cash_reduction_min"] = float(
                info["early_rebalance_risk_repeat_previous_cash_reduction_min"]
            )
        if info.get("early_rebalance_risk_repeat_previous_symbol_increase_min") is not None:
            record["early_rebalance_risk_repeat_previous_symbol_increase_min"] = float(
                info["early_rebalance_risk_repeat_previous_symbol_increase_min"]
            )
        if "early_rebalance_risk_repeat_previous_cash_reduction" in info:
            record["early_rebalance_risk_repeat_previous_cash_reduction"] = float(
                info["early_rebalance_risk_repeat_previous_cash_reduction"]
            )
        if "early_rebalance_risk_repeat_previous_symbol_increase" in info:
            record["early_rebalance_risk_repeat_previous_symbol_increase"] = float(
                info["early_rebalance_risk_repeat_previous_symbol_increase"]
            )
        if "early_rebalance_risk_repeat_cash_reduction" in info:
            record["early_rebalance_risk_repeat_cash_reduction"] = float(
                info["early_rebalance_risk_repeat_cash_reduction"]
            )
        if "early_rebalance_risk_repeat_symbol_increase" in info:
            record["early_rebalance_risk_repeat_symbol_increase"] = float(
                info["early_rebalance_risk_repeat_symbol_increase"]
            )
        if info.get("early_rebalance_risk_repeat_unrecovered_turnover_cap") is not None:
            record["early_rebalance_risk_repeat_unrecovered_turnover_cap"] = float(
                info["early_rebalance_risk_repeat_unrecovered_turnover_cap"]
            )
        if info.get("early_rebalance_risk_repeat_unrecovered_turnover_cap_after") is not None:
            record["early_rebalance_risk_repeat_unrecovered_turnover_cap_after"] = float(
                info["early_rebalance_risk_repeat_unrecovered_turnover_cap_after"]
            )
        if info.get("early_rebalance_risk_repeat_unrecovered_turnover_cap_before") is not None:
            record["early_rebalance_risk_repeat_unrecovered_turnover_cap_before"] = float(
                info["early_rebalance_risk_repeat_unrecovered_turnover_cap_before"]
            )
        if info.get("early_rebalance_risk_repeat_unrecovered_symbol") is not None:
            record["early_rebalance_risk_repeat_unrecovered_symbol"] = str(
                info["early_rebalance_risk_repeat_unrecovered_symbol"]
            )
        if info.get("early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min") is not None:
            record["early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min"] = float(
                info["early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min"]
            )
        if info.get("early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min") is not None:
            record["early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min"] = float(
                info["early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min"]
            )
        if info.get("early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery") is not None:
            record["early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery"] = float(
                info["early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery"]
            )
        if (
            "early_rebalance_risk_repeat_unrecovered_previous_cash_reduction" in info
            and info["early_rebalance_risk_repeat_unrecovered_previous_cash_reduction"] is not None
        ):
            record["early_rebalance_risk_repeat_unrecovered_previous_cash_reduction"] = float(
                info["early_rebalance_risk_repeat_unrecovered_previous_cash_reduction"]
            )
        if (
            "early_rebalance_risk_repeat_unrecovered_previous_symbol_increase" in info
            and info["early_rebalance_risk_repeat_unrecovered_previous_symbol_increase"] is not None
        ):
            record["early_rebalance_risk_repeat_unrecovered_previous_symbol_increase"] = float(
                info["early_rebalance_risk_repeat_unrecovered_previous_symbol_increase"]
            )
        if (
            "early_rebalance_risk_repeat_unrecovered_previous_pre_trade_relative_wealth_ratio" in info
            and info["early_rebalance_risk_repeat_unrecovered_previous_pre_trade_relative_wealth_ratio"] is not None
        ):
            record["early_rebalance_risk_repeat_unrecovered_previous_pre_trade_relative_wealth_ratio"] = float(
                info["early_rebalance_risk_repeat_unrecovered_previous_pre_trade_relative_wealth_ratio"]
            )
        if (
            "early_rebalance_risk_repeat_unrecovered_relative_wealth_recovery" in info
            and info["early_rebalance_risk_repeat_unrecovered_relative_wealth_recovery"] is not None
        ):
            record["early_rebalance_risk_repeat_unrecovered_relative_wealth_recovery"] = float(
                info["early_rebalance_risk_repeat_unrecovered_relative_wealth_recovery"]
            )
        if (
            "early_rebalance_risk_repeat_unrecovered_cash_reduction" in info
            and info["early_rebalance_risk_repeat_unrecovered_cash_reduction"] is not None
        ):
            record["early_rebalance_risk_repeat_unrecovered_cash_reduction"] = float(
                info["early_rebalance_risk_repeat_unrecovered_cash_reduction"]
            )
        if (
            "early_rebalance_risk_repeat_unrecovered_symbol_increase" in info
            and info["early_rebalance_risk_repeat_unrecovered_symbol_increase"] is not None
        ):
            record["early_rebalance_risk_repeat_unrecovered_symbol_increase"] = float(
                info["early_rebalance_risk_repeat_unrecovered_symbol_increase"]
            )
        if info.get("early_rebalance_risk_cumulative_turnover_cap") is not None:
            record["early_rebalance_risk_cumulative_turnover_cap"] = float(
                info["early_rebalance_risk_cumulative_turnover_cap"]
            )
        if info.get("early_rebalance_risk_cumulative_turnover_cap_after") is not None:
            record["early_rebalance_risk_cumulative_turnover_cap_after"] = float(
                info["early_rebalance_risk_cumulative_turnover_cap_after"]
            )
        if info.get("early_rebalance_risk_cumulative_turnover_cap_before") is not None:
            record["early_rebalance_risk_cumulative_turnover_cap_before"] = float(
                info["early_rebalance_risk_cumulative_turnover_cap_before"]
            )
        if info.get("early_rebalance_risk_cumulative_symbol") is not None:
            record["early_rebalance_risk_cumulative_symbol"] = str(info["early_rebalance_risk_cumulative_symbol"])
        if info.get("early_rebalance_risk_cumulative_cash_reduction_budget") is not None:
            record["early_rebalance_risk_cumulative_cash_reduction_budget"] = float(
                info["early_rebalance_risk_cumulative_cash_reduction_budget"]
            )
        if info.get("early_rebalance_risk_cumulative_symbol_increase_budget") is not None:
            record["early_rebalance_risk_cumulative_symbol_increase_budget"] = float(
                info["early_rebalance_risk_cumulative_symbol_increase_budget"]
            )
        if "early_rebalance_risk_cumulative_prior_cash_reduction" in info:
            record["early_rebalance_risk_cumulative_prior_cash_reduction"] = float(
                info["early_rebalance_risk_cumulative_prior_cash_reduction"]
            )
        if "early_rebalance_risk_cumulative_prior_symbol_increase" in info:
            record["early_rebalance_risk_cumulative_prior_symbol_increase"] = float(
                info["early_rebalance_risk_cumulative_prior_symbol_increase"]
            )
        if "early_rebalance_risk_cumulative_cash_reduction" in info:
            record["early_rebalance_risk_cumulative_cash_reduction"] = float(
                info["early_rebalance_risk_cumulative_cash_reduction"]
            )
        if "early_rebalance_risk_cumulative_symbol_increase" in info:
            record["early_rebalance_risk_cumulative_symbol_increase"] = float(
                info["early_rebalance_risk_cumulative_symbol_increase"]
            )
        if info.get("early_rebalance_risk_penalty_after") is not None:
            record["early_rebalance_risk_penalty_after"] = float(info["early_rebalance_risk_penalty_after"])
        if info.get("early_rebalance_risk_penalty_before") is not None:
            record["early_rebalance_risk_penalty_before"] = float(info["early_rebalance_risk_penalty_before"])
        if info.get("early_rebalance_risk_penalty_cash_max_threshold") is not None:
            record["early_rebalance_risk_penalty_cash_max_threshold"] = float(
                info["early_rebalance_risk_penalty_cash_max_threshold"]
            )
        if info.get("early_rebalance_risk_penalty_symbol") is not None:
            record["early_rebalance_risk_penalty_symbol"] = str(info["early_rebalance_risk_penalty_symbol"])
        if info.get("early_rebalance_risk_penalty_symbol_min_weight") is not None:
            record["early_rebalance_risk_penalty_symbol_min_weight"] = float(
                info["early_rebalance_risk_penalty_symbol_min_weight"]
            )
        if info.get("early_rebalance_risk_penalty_symbol_max_weight") is not None:
            record["early_rebalance_risk_penalty_symbol_max_weight"] = float(
                info["early_rebalance_risk_penalty_symbol_max_weight"]
            )
        if info.get("early_rebalance_risk_penalty_benchmark_drawdown_min_threshold") is not None:
            record["early_rebalance_risk_penalty_benchmark_drawdown_min_threshold"] = float(
                info["early_rebalance_risk_penalty_benchmark_drawdown_min_threshold"]
            )
        if info.get("early_rebalance_risk_penalty_benchmark_drawdown_max_threshold") is not None:
            record["early_rebalance_risk_penalty_benchmark_drawdown_max_threshold"] = float(
                info["early_rebalance_risk_penalty_benchmark_drawdown_max_threshold"]
            )
        if info.get("early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio") is not None:
            record["early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio"] = float(
                info["early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio"]
            )
        if info.get("early_benchmark_euphoria_penalty") is not None:
            record["early_benchmark_euphoria_penalty"] = float(info["early_benchmark_euphoria_penalty"])
        if info.get("early_benchmark_euphoria_turnover_cap") is not None:
            record["early_benchmark_euphoria_turnover_cap"] = float(info["early_benchmark_euphoria_turnover_cap"])
        if info.get("early_benchmark_euphoria_before") is not None:
            record["early_benchmark_euphoria_before"] = float(info["early_benchmark_euphoria_before"])
        if info.get("early_benchmark_euphoria_benchmark_drawdown_min_threshold") is not None:
            record["early_benchmark_euphoria_benchmark_drawdown_min_threshold"] = float(
                info["early_benchmark_euphoria_benchmark_drawdown_min_threshold"]
            )
        if info.get("early_benchmark_euphoria_symbol") is not None:
            record["early_benchmark_euphoria_symbol"] = str(info["early_benchmark_euphoria_symbol"])
        if info.get("late_rebalance_penalty") is not None:
            record["late_rebalance_penalty"] = float(info["late_rebalance_penalty"])
        if info.get("late_rebalance_penalty_after") is not None:
            record["late_rebalance_penalty_after"] = float(info["late_rebalance_penalty_after"])
        if info.get("late_rebalance_gate_after") is not None:
            record["late_rebalance_gate_after"] = float(info["late_rebalance_gate_after"])
        if info.get("late_rebalance_gate_cash_threshold") is not None:
            record["late_rebalance_gate_cash_threshold"] = float(info["late_rebalance_gate_cash_threshold"])
        if info.get("late_rebalance_gate_target_cash_min_threshold") is not None:
            record["late_rebalance_gate_target_cash_min_threshold"] = float(
                info["late_rebalance_gate_target_cash_min_threshold"]
            )
        if info.get("late_rebalance_gate_symbol") is not None:
            record["late_rebalance_gate_symbol"] = str(info["late_rebalance_gate_symbol"])
        if info.get("late_rebalance_gate_symbol_max_weight") is not None:
            record["late_rebalance_gate_symbol_max_weight"] = float(info["late_rebalance_gate_symbol_max_weight"])
        if info.get("late_rebalance_gate_cash_reduction_max") is not None:
            record["late_rebalance_gate_cash_reduction_max"] = float(info["late_rebalance_gate_cash_reduction_max"])
        if info.get("late_rebalance_gate_symbol_increase_max") is not None:
            record["late_rebalance_gate_symbol_increase_max"] = float(info["late_rebalance_gate_symbol_increase_max"])
        if "late_rebalance_gate_cash_reduction" in info:
            record["late_rebalance_gate_cash_reduction"] = float(info["late_rebalance_gate_cash_reduction"])
        if "late_rebalance_gate_symbol_increase" in info:
            record["late_rebalance_gate_symbol_increase"] = float(info["late_rebalance_gate_symbol_increase"])
        if "late_rebalance_gate_refinement_condition_met" in info:
            record["late_rebalance_gate_refinement_condition_met"] = bool(
                info["late_rebalance_gate_refinement_condition_met"]
            )
        if info.get("late_defensive_posture_penalty") is not None:
            record["late_defensive_posture_penalty"] = float(info["late_defensive_posture_penalty"])
        if info.get("late_defensive_posture_penalty_after") is not None:
            record["late_defensive_posture_penalty_after"] = float(info["late_defensive_posture_penalty_after"])
        if info.get("late_defensive_posture_penalty_cash_min_threshold") is not None:
            record["late_defensive_posture_penalty_cash_min_threshold"] = float(
                info["late_defensive_posture_penalty_cash_min_threshold"]
            )
        if info.get("late_defensive_posture_penalty_symbol") is not None:
            record["late_defensive_posture_penalty_symbol"] = str(info["late_defensive_posture_penalty_symbol"])
        if info.get("late_defensive_posture_penalty_symbol_max_weight") is not None:
            record["late_defensive_posture_penalty_symbol_max_weight"] = float(
                info["late_defensive_posture_penalty_symbol_max_weight"]
            )
        if info.get("late_trend_mean_reversion_conflict_penalty") is not None:
            record["late_trend_mean_reversion_conflict_penalty"] = float(
                info["late_trend_mean_reversion_conflict_penalty"]
            )
        if info.get("late_trend_mean_reversion_conflict_penalty_after") is not None:
            record["late_trend_mean_reversion_conflict_penalty_after"] = float(
                info["late_trend_mean_reversion_conflict_penalty_after"]
            )
        if info.get("late_trend_mean_reversion_conflict_trend_symbol") is not None:
            record["late_trend_mean_reversion_conflict_trend_symbol"] = str(
                info["late_trend_mean_reversion_conflict_trend_symbol"]
            )
        if info.get("late_trend_mean_reversion_conflict_trend_min_weight") is not None:
            record["late_trend_mean_reversion_conflict_trend_min_weight"] = float(
                info["late_trend_mean_reversion_conflict_trend_min_weight"]
            )
        if info.get("late_trend_mean_reversion_conflict_mean_reversion_symbol") is not None:
            record["late_trend_mean_reversion_conflict_mean_reversion_symbol"] = str(
                info["late_trend_mean_reversion_conflict_mean_reversion_symbol"]
            )
        if info.get("late_trend_mean_reversion_conflict_mean_reversion_min_weight") is not None:
            record["late_trend_mean_reversion_conflict_mean_reversion_min_weight"] = float(
                info["late_trend_mean_reversion_conflict_mean_reversion_min_weight"]
            )
        if info.get("state_trend_preservation_symbol") is not None:
            record["state_trend_preservation_symbol"] = str(info["state_trend_preservation_symbol"])
        if info.get("state_trend_preservation_cash_max_threshold") is not None:
            record["state_trend_preservation_cash_max_threshold"] = float(
                info["state_trend_preservation_cash_max_threshold"]
            )
        if info.get("state_trend_preservation_symbol_min_weight") is not None:
            record["state_trend_preservation_symbol_min_weight"] = float(
                info["state_trend_preservation_symbol_min_weight"]
            )
        if info.get("state_trend_preservation_max_symbol_reduction") is not None:
            record["state_trend_preservation_max_symbol_reduction"] = float(
                info["state_trend_preservation_max_symbol_reduction"]
            )
        if "benchmark_regime_cumulative_return" in info:
            record["benchmark_regime_cumulative_return"] = float(info["benchmark_regime_cumulative_return"])
        if "benchmark_regime_drawdown" in info:
            record["benchmark_regime_drawdown"] = float(info["benchmark_regime_drawdown"])
        if "pre_trade_relative_wealth_ratio" in info:
            record["pre_trade_relative_wealth_ratio"] = float(info["pre_trade_relative_wealth_ratio"])
        if "relative_wealth_ratio" in info:
            record["relative_wealth_ratio"] = float(info["relative_wealth_ratio"])
        record.update(_flatten_weight_mapping("asset_return_", info.get("asset_returns", {})))
        record.update(_flatten_weight_mapping("gross_return_contribution_", info.get("gross_return_contributions", {})))
        record.update(_flatten_weight_mapping("pre_trade_weight_", info.get("pre_trade_weights", {})))
        record.update(_flatten_weight_mapping("proposed_weight_", info.get("proposed_weights", {})))
        record.update(_flatten_weight_mapping("target_weight_", info.get("target_weights", {})))
        record.update(_flatten_weight_mapping("ending_weight_", info.get("ending_weights", {})))
        records.append(record)
        if bool(dones[0]):
            break

    frame = pd.DataFrame(records)
    if not frame.empty:
        converted_index = pd.to_datetime(frame["step_label"], errors="coerce")
        if converted_index.notna().all():
            frame.index = converted_index
        else:
            frame.index = frame["step_label"].astype(str)
        frame.index.name = "date"
    return frame


def _summarize_policy_rollout_records(
    records: pd.DataFrame,
    cash_symbol: str = "CASH",
    periods_per_year: int = 252,
    initial_wealth: float = 1.0,
) -> pd.Series:
    if records.empty:
        raise ValueError("Policy rollout records are empty.")

    portfolio_returns = records["portfolio_return"].astype(float)
    summary = summarize_performance(
        portfolio_returns=portfolio_returns,
        initial_wealth=initial_wealth,
        periods_per_year=periods_per_year,
    )
    gross_wealth_index = initial_wealth * (1.0 + records["gross_portfolio_return"].astype(float)).cumprod()
    wealth_index = initial_wealth * (1.0 + portfolio_returns).cumprod()
    summary.loc["gross_total_return"] = float(gross_wealth_index.iloc[-1] / initial_wealth - 1.0)
    summary.loc["average_turnover"] = float(records["turnover"].astype(float).mean())
    summary.loc["total_turnover"] = float(records["turnover"].astype(float).sum())
    summary.loc["average_trading_cost"] = float(records["execution_cost"].astype(float).mean())
    summary.loc["cost_drag"] = float(gross_wealth_index.iloc[-1] / initial_wealth - wealth_index.iloc[-1] / initial_wealth)
    executed_rebalances = records["turnover"].astype(float) > 1e-12
    summary.loc["rebalance_count"] = float(executed_rebalances.sum())
    summary.loc["rebalance_rate"] = float(executed_rebalances.mean())
    summary.loc["average_executed_turnover"] = float(records.loc[executed_rebalances, "turnover"].astype(float).mean()) if executed_rebalances.any() else 0.0
    summary.loc["average_raw_reward"] = float(records["raw_reward"].astype(float).mean())
    summary.loc["total_raw_reward"] = float(records["raw_reward"].astype(float).sum())

    reward_component_columns = {
        "average_gross_reward_component": "gross_reward_component",
        "average_reward_core_component": "reward_core_component",
        "average_benchmark_reward_component": "benchmark_reward_component",
        "average_execution_cost_reward_drag": "execution_cost_reward_drag",
        "average_turnover_penalty_component": "turnover_penalty_component",
        "average_weight_reg_penalty_component": "weight_reg_penalty_component",
        "average_early_rebalance_risk_penalty_component": "early_rebalance_risk_penalty_component",
        "average_early_benchmark_euphoria_penalty_component": "early_benchmark_euphoria_penalty_component",
        "average_late_rebalance_penalty_component": "late_rebalance_penalty_component",
        "average_late_defensive_posture_penalty_component": "late_defensive_posture_penalty_component",
        "average_late_trend_mean_reversion_conflict_penalty_component": "late_trend_mean_reversion_conflict_penalty_component",
        "average_friction_reward_drag": "friction_reward_drag",
    }
    for summary_name, column_name in reward_component_columns.items():
        if column_name in records.columns:
            summary.loc[summary_name] = float(records[column_name].astype(float).mean())

    if "proposed_turnover" in records.columns:
        summary.loc["average_proposed_turnover"] = float(records["proposed_turnover"].astype(float).mean())
        summary.loc["average_turnover_reduction"] = float(
            (records["proposed_turnover"].astype(float) - records["turnover"].astype(float)).mean()
        )
    if "trade_suppressed" in records.columns:
        summary.loc["trade_suppression_rate"] = float(records["trade_suppressed"].astype(float).mean())
    if "rebalance_budget_exhausted" in records.columns:
        summary.loc["rebalance_budget_exhaustion_rate"] = float(records["rebalance_budget_exhausted"].astype(float).mean())
    if "rebalance_cooldown_active" in records.columns:
        summary.loc["rebalance_cooldown_active_rate"] = float(records["rebalance_cooldown_active"].astype(float).mean())
    if "rebalance_cooldown_blocked" in records.columns:
        summary.loc["rebalance_cooldown_block_rate"] = float(records["rebalance_cooldown_blocked"].astype(float).mean())
    if "early_rebalance_risk_window_active" in records.columns:
        summary.loc["early_rebalance_risk_window_rate"] = float(records["early_rebalance_risk_window_active"].astype(float).mean())
    if "early_rebalance_risk_turnover_cap_window_active" in records.columns:
        summary.loc["early_rebalance_risk_turnover_cap_window_rate"] = float(
            records["early_rebalance_risk_turnover_cap_window_active"].astype(float).mean()
        )
    if "early_rebalance_risk_turnover_cap_condition_met" in records.columns:
        summary.loc["early_rebalance_risk_turnover_cap_condition_rate"] = float(
            records["early_rebalance_risk_turnover_cap_condition_met"].astype(float).mean()
        )
    if "early_rebalance_risk_turnover_cap_applied" in records.columns:
        summary.loc["early_rebalance_risk_turnover_cap_application_rate"] = float(
            records["early_rebalance_risk_turnover_cap_applied"].astype(float).mean()
        )
    if "early_rebalance_risk_turnover_cap_max_applications_reached" in records.columns:
        summary.loc["early_rebalance_risk_turnover_cap_max_applications_reached_rate"] = float(
            records["early_rebalance_risk_turnover_cap_max_applications_reached"].astype(float).mean()
        )
    if "early_rebalance_risk_turnover_cap_secondary_active" in records.columns:
        summary.loc["early_rebalance_risk_turnover_cap_secondary_active_rate"] = float(
            records["early_rebalance_risk_turnover_cap_secondary_active"].astype(float).mean()
        )
    if "early_rebalance_risk_turnover_cap_secondary_applied" in records.columns:
        summary.loc["early_rebalance_risk_turnover_cap_secondary_application_rate"] = float(
            records["early_rebalance_risk_turnover_cap_secondary_applied"].astype(float).mean()
        )
    if "early_rebalance_risk_turnover_cap_secondary_after_applications_reached" in records.columns:
        summary.loc["early_rebalance_risk_turnover_cap_secondary_after_applications_reached_rate"] = float(
            records["early_rebalance_risk_turnover_cap_secondary_after_applications_reached"].astype(float).mean()
        )
    if "early_rebalance_risk_turnover_cap_secondary_state_condition_met" in records.columns:
        summary.loc["early_rebalance_risk_turnover_cap_secondary_state_condition_met_rate"] = float(
            records["early_rebalance_risk_turnover_cap_secondary_state_condition_met"].astype(float).mean()
        )
    if "early_rebalance_risk_shallow_drawdown_turnover_cap_window_active" in records.columns:
        summary.loc["early_rebalance_risk_shallow_drawdown_turnover_cap_window_rate"] = float(
            records["early_rebalance_risk_shallow_drawdown_turnover_cap_window_active"].astype(float).mean()
        )
    if "early_rebalance_risk_shallow_drawdown_turnover_cap_condition_met" in records.columns:
        summary.loc["early_rebalance_risk_shallow_drawdown_turnover_cap_condition_rate"] = float(
            records["early_rebalance_risk_shallow_drawdown_turnover_cap_condition_met"].astype(float).mean()
        )
    if "early_rebalance_risk_shallow_drawdown_turnover_cap_applied" in records.columns:
        summary.loc["early_rebalance_risk_shallow_drawdown_turnover_cap_application_rate"] = float(
            records["early_rebalance_risk_shallow_drawdown_turnover_cap_applied"].astype(float).mean()
        )
    if "early_rebalance_risk_mean_reversion_turnover_cap_window_active" in records.columns:
        summary.loc["early_rebalance_risk_mean_reversion_turnover_cap_window_rate"] = float(
            records["early_rebalance_risk_mean_reversion_turnover_cap_window_active"].astype(float).mean()
        )
    if "early_rebalance_risk_mean_reversion_turnover_cap_condition_met" in records.columns:
        summary.loc["early_rebalance_risk_mean_reversion_turnover_cap_condition_rate"] = float(
            records["early_rebalance_risk_mean_reversion_turnover_cap_condition_met"].astype(float).mean()
        )
    if "early_rebalance_risk_mean_reversion_action_smoothing_applied" in records.columns:
        summary.loc["early_rebalance_risk_mean_reversion_action_smoothing_application_rate"] = float(
            records["early_rebalance_risk_mean_reversion_action_smoothing_applied"].astype(float).mean()
        )
    if "early_rebalance_risk_mean_reversion_turnover_cap_applied" in records.columns:
        summary.loc["early_rebalance_risk_mean_reversion_turnover_cap_application_rate"] = float(
            records["early_rebalance_risk_mean_reversion_turnover_cap_applied"].astype(float).mean()
        )
    if "early_rebalance_risk_trend_turnover_cap_window_active" in records.columns:
        summary.loc["early_rebalance_risk_trend_turnover_cap_window_rate"] = float(
            records["early_rebalance_risk_trend_turnover_cap_window_active"].astype(float).mean()
        )
    if "early_rebalance_risk_trend_turnover_cap_condition_met" in records.columns:
        summary.loc["early_rebalance_risk_trend_turnover_cap_condition_rate"] = float(
            records["early_rebalance_risk_trend_turnover_cap_condition_met"].astype(float).mean()
        )
    if "early_rebalance_risk_trend_turnover_cap_applied" in records.columns:
        summary.loc["early_rebalance_risk_trend_turnover_cap_application_rate"] = float(
            records["early_rebalance_risk_trend_turnover_cap_applied"].astype(float).mean()
        )
    if "early_rebalance_risk_deep_drawdown_turnover_cap_window_active" in records.columns:
        summary.loc["early_rebalance_risk_deep_drawdown_turnover_cap_window_rate"] = float(
            records["early_rebalance_risk_deep_drawdown_turnover_cap_window_active"].astype(float).mean()
        )
    if "early_rebalance_risk_deep_drawdown_turnover_cap_condition_met" in records.columns:
        summary.loc["early_rebalance_risk_deep_drawdown_turnover_cap_condition_rate"] = float(
            records["early_rebalance_risk_deep_drawdown_turnover_cap_condition_met"].astype(float).mean()
        )
    if "early_rebalance_risk_deep_drawdown_turnover_cap_applied" in records.columns:
        summary.loc["early_rebalance_risk_deep_drawdown_turnover_cap_application_rate"] = float(
            records["early_rebalance_risk_deep_drawdown_turnover_cap_applied"].astype(float).mean()
        )
    if "early_rebalance_risk_repeat_turnover_cap_window_active" in records.columns:
        summary.loc["early_rebalance_risk_repeat_turnover_cap_window_rate"] = float(
            records["early_rebalance_risk_repeat_turnover_cap_window_active"].astype(float).mean()
        )
    if "early_rebalance_risk_repeat_turnover_cap_condition_met" in records.columns:
        summary.loc["early_rebalance_risk_repeat_turnover_cap_condition_rate"] = float(
            records["early_rebalance_risk_repeat_turnover_cap_condition_met"].astype(float).mean()
        )
    if "early_rebalance_risk_repeat_action_smoothing_applied" in records.columns:
        summary.loc["early_rebalance_risk_repeat_action_smoothing_application_rate"] = float(
            records["early_rebalance_risk_repeat_action_smoothing_applied"].astype(float).mean()
        )
    if "early_rebalance_risk_repeat_turnover_cap_applied" in records.columns:
        summary.loc["early_rebalance_risk_repeat_turnover_cap_application_rate"] = float(
            records["early_rebalance_risk_repeat_turnover_cap_applied"].astype(float).mean()
        )
    if "early_rebalance_risk_repeat_unrecovered_turnover_cap_window_active" in records.columns:
        summary.loc["early_rebalance_risk_repeat_unrecovered_turnover_cap_window_rate"] = float(
            records["early_rebalance_risk_repeat_unrecovered_turnover_cap_window_active"].astype(float).mean()
        )
    if "early_rebalance_risk_repeat_unrecovered_turnover_cap_condition_met" in records.columns:
        summary.loc["early_rebalance_risk_repeat_unrecovered_turnover_cap_condition_rate"] = float(
            records["early_rebalance_risk_repeat_unrecovered_turnover_cap_condition_met"].astype(float).mean()
        )
    if "early_rebalance_risk_repeat_unrecovered_turnover_cap_applied" in records.columns:
        summary.loc["early_rebalance_risk_repeat_unrecovered_turnover_cap_application_rate"] = float(
            records["early_rebalance_risk_repeat_unrecovered_turnover_cap_applied"].astype(float).mean()
        )
    if "early_rebalance_risk_cumulative_turnover_cap_window_active" in records.columns:
        summary.loc["early_rebalance_risk_cumulative_turnover_cap_window_rate"] = float(
            records["early_rebalance_risk_cumulative_turnover_cap_window_active"].astype(float).mean()
        )
    if "early_rebalance_risk_cumulative_turnover_cap_condition_met" in records.columns:
        summary.loc["early_rebalance_risk_cumulative_turnover_cap_condition_rate"] = float(
            records["early_rebalance_risk_cumulative_turnover_cap_condition_met"].astype(float).mean()
        )
    if "early_rebalance_risk_cumulative_turnover_cap_applied" in records.columns:
        summary.loc["early_rebalance_risk_cumulative_turnover_cap_application_rate"] = float(
            records["early_rebalance_risk_cumulative_turnover_cap_applied"].astype(float).mean()
        )
    if "early_rebalance_risk_penalty_applied" in records.columns:
        summary.loc["early_rebalance_risk_penalty_application_rate"] = float(
            records["early_rebalance_risk_penalty_applied"].astype(float).mean()
        )
    if "early_rebalance_risk_condition_met" in records.columns:
        summary.loc["early_rebalance_risk_condition_rate"] = float(
            records["early_rebalance_risk_condition_met"].astype(float).mean()
        )
    if "early_benchmark_euphoria_window_active" in records.columns:
        summary.loc["early_benchmark_euphoria_window_rate"] = float(
            records["early_benchmark_euphoria_window_active"].astype(float).mean()
        )
    if "early_benchmark_euphoria_turnover_cap_applied" in records.columns:
        summary.loc["early_benchmark_euphoria_turnover_cap_application_rate"] = float(
            records["early_benchmark_euphoria_turnover_cap_applied"].astype(float).mean()
        )
    if "early_benchmark_euphoria_penalty_applied" in records.columns:
        summary.loc["early_benchmark_euphoria_penalty_application_rate"] = float(
            records["early_benchmark_euphoria_penalty_applied"].astype(float).mean()
        )
    if "early_benchmark_euphoria_condition_met" in records.columns:
        summary.loc["early_benchmark_euphoria_condition_rate"] = float(
            records["early_benchmark_euphoria_condition_met"].astype(float).mean()
        )
    if "late_rebalance_penalty_applied" in records.columns:
        summary.loc["late_rebalance_penalty_application_rate"] = float(records["late_rebalance_penalty_applied"].astype(float).mean())
    if "late_rebalance_threshold_reached" in records.columns:
        summary.loc["late_rebalance_threshold_rate"] = float(records["late_rebalance_threshold_reached"].astype(float).mean())
    if "late_rebalance_gate_active" in records.columns:
        summary.loc["late_rebalance_gate_active_rate"] = float(records["late_rebalance_gate_active"].astype(float).mean())
    if "late_rebalance_gate_blocked" in records.columns:
        summary.loc["late_rebalance_gate_block_rate"] = float(records["late_rebalance_gate_blocked"].astype(float).mean())
    if "late_rebalance_gate_condition_met" in records.columns:
        summary.loc["late_rebalance_gate_condition_rate"] = float(records["late_rebalance_gate_condition_met"].astype(float).mean())
    if "late_rebalance_gate_threshold_reached" in records.columns:
        summary.loc["late_rebalance_gate_threshold_rate"] = float(records["late_rebalance_gate_threshold_reached"].astype(float).mean())
    if "late_defensive_posture_window_active" in records.columns:
        summary.loc["late_defensive_posture_window_rate"] = float(
            records["late_defensive_posture_window_active"].astype(float).mean()
        )
    if "late_defensive_posture_condition_met" in records.columns:
        summary.loc["late_defensive_posture_condition_rate"] = float(
            records["late_defensive_posture_condition_met"].astype(float).mean()
        )
    if "late_defensive_posture_penalty_applied" in records.columns:
        summary.loc["late_defensive_posture_penalty_application_rate"] = float(
            records["late_defensive_posture_penalty_applied"].astype(float).mean()
        )
    if "late_trend_mean_reversion_conflict_window_active" in records.columns:
        summary.loc["late_trend_mean_reversion_conflict_window_rate"] = float(
            records["late_trend_mean_reversion_conflict_window_active"].astype(float).mean()
        )
    if "late_trend_mean_reversion_conflict_condition_met" in records.columns:
        summary.loc["late_trend_mean_reversion_conflict_condition_rate"] = float(
            records["late_trend_mean_reversion_conflict_condition_met"].astype(float).mean()
        )
    if "late_trend_mean_reversion_conflict_penalty_applied" in records.columns:
        summary.loc["late_trend_mean_reversion_conflict_penalty_application_rate"] = float(
            records["late_trend_mean_reversion_conflict_penalty_applied"].astype(float).mean()
        )
    if "state_trend_preservation_window_active" in records.columns:
        summary.loc["state_trend_preservation_window_rate"] = float(
            records["state_trend_preservation_window_active"].astype(float).mean()
        )
    if "state_trend_preservation_condition_met" in records.columns:
        summary.loc["state_trend_preservation_condition_rate"] = float(
            records["state_trend_preservation_condition_met"].astype(float).mean()
        )
    if "state_trend_preservation_guard_applied" in records.columns:
        summary.loc["state_trend_preservation_guard_application_rate"] = float(
            records["state_trend_preservation_guard_applied"].astype(float).mean()
        )
    if "executed_rebalances" in records.columns:
        summary.loc["final_executed_rebalances"] = float(records["executed_rebalances"].astype(float).iloc[-1])
    if "early_rebalance_risk_turnover_cap_applications" in records.columns:
        summary.loc["final_early_rebalance_risk_turnover_cap_applications"] = float(
            records["early_rebalance_risk_turnover_cap_applications"].astype(float).iloc[-1]
        )
    if "max_executed_rebalances" in records.columns:
        non_null_budget = records["max_executed_rebalances"].dropna()
        summary.loc["max_executed_rebalances"] = float(non_null_budget.iloc[0]) if not non_null_budget.empty else None
    if "rebalance_cooldown_remaining" in records.columns:
        summary.loc["final_rebalance_cooldown_remaining"] = float(records["rebalance_cooldown_remaining"].astype(float).iloc[-1])
    if "rebalance_cooldown_steps" in records.columns:
        non_null_cooldown = records["rebalance_cooldown_steps"].dropna()
        summary.loc["rebalance_cooldown_steps"] = float(non_null_cooldown.iloc[0]) if not non_null_cooldown.empty else None
    if "early_rebalance_risk_penalty" in records.columns:
        non_null_early_penalty = records["early_rebalance_risk_penalty"].dropna()
        summary.loc["early_rebalance_risk_penalty"] = (
            float(non_null_early_penalty.iloc[0]) if not non_null_early_penalty.empty else None
        )
    if "early_rebalance_risk_turnover_cap" in records.columns:
        non_null_early_turnover_cap = records["early_rebalance_risk_turnover_cap"].dropna()
        summary.loc["early_rebalance_risk_turnover_cap"] = (
            float(non_null_early_turnover_cap.iloc[0]) if not non_null_early_turnover_cap.empty else None
        )
    if "early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold" in records.columns:
        non_null_early_turnover_cap_drawdown = records[
            "early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold"] = (
            float(non_null_early_turnover_cap_drawdown.iloc[0]) if not non_null_early_turnover_cap_drawdown.empty else None
        )
    if "early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold" in records.columns:
        non_null_early_turnover_cap_cumret = records[
            "early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold"] = (
            float(non_null_early_turnover_cap_cumret.iloc[0]) if not non_null_early_turnover_cap_cumret.empty else None
        )
    if "early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight" in records.columns:
        non_null_early_turnover_cap_cash_min = records[
            "early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight"] = (
            float(non_null_early_turnover_cap_cash_min.iloc[0])
            if not non_null_early_turnover_cap_cash_min.empty
            else None
        )
    if "early_rebalance_risk_turnover_cap_target_cash_min_threshold" in records.columns:
        non_null_early_turnover_cap_target_cash_min = records[
            "early_rebalance_risk_turnover_cap_target_cash_min_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_target_cash_min_threshold"] = (
            float(non_null_early_turnover_cap_target_cash_min.iloc[0])
            if not non_null_early_turnover_cap_target_cash_min.empty
            else None
        )
    if "early_rebalance_risk_turnover_cap_target_cash_max_threshold" in records.columns:
        non_null_early_turnover_cap_target_cash_max = records[
            "early_rebalance_risk_turnover_cap_target_cash_max_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_target_cash_max_threshold"] = (
            float(non_null_early_turnover_cap_target_cash_max.iloc[0])
            if not non_null_early_turnover_cap_target_cash_max.empty
            else None
        )
    if "early_rebalance_risk_turnover_cap_target_trend_min_threshold" in records.columns:
        non_null_early_turnover_cap_target_trend_min = records[
            "early_rebalance_risk_turnover_cap_target_trend_min_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_target_trend_min_threshold"] = (
            float(non_null_early_turnover_cap_target_trend_min.iloc[0])
            if not non_null_early_turnover_cap_target_trend_min.empty
            else None
        )
    if "early_rebalance_risk_turnover_cap_target_trend_max_threshold" in records.columns:
        non_null_early_turnover_cap_target_trend_max = records[
            "early_rebalance_risk_turnover_cap_target_trend_max_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_target_trend_max_threshold"] = (
            float(non_null_early_turnover_cap_target_trend_max.iloc[0])
            if not non_null_early_turnover_cap_target_trend_max.empty
            else None
        )
    if "early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold" in records.columns:
        non_null_early_turnover_cap_target_mean_reversion_min = records[
            "early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold"] = (
            float(non_null_early_turnover_cap_target_mean_reversion_min.iloc[0])
            if not non_null_early_turnover_cap_target_mean_reversion_min.empty
            else None
        )
    if "early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold" in records.columns:
        non_null_early_turnover_cap_target_mean_reversion_max = records[
            "early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold"] = (
            float(non_null_early_turnover_cap_target_mean_reversion_max.iloc[0])
            if not non_null_early_turnover_cap_target_mean_reversion_max.empty
            else None
        )
    if "early_rebalance_risk_turnover_cap_delta_cash_min_threshold" in records.columns:
        non_null_early_turnover_cap_delta_cash_min = records[
            "early_rebalance_risk_turnover_cap_delta_cash_min_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_delta_cash_min_threshold"] = (
            float(non_null_early_turnover_cap_delta_cash_min.iloc[0])
            if not non_null_early_turnover_cap_delta_cash_min.empty
            else None
        )
    if "early_rebalance_risk_turnover_cap_delta_cash_max_threshold" in records.columns:
        non_null_early_turnover_cap_delta_cash_max = records[
            "early_rebalance_risk_turnover_cap_delta_cash_max_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_delta_cash_max_threshold"] = (
            float(non_null_early_turnover_cap_delta_cash_max.iloc[0])
            if not non_null_early_turnover_cap_delta_cash_max.empty
            else None
        )
    if "early_rebalance_risk_turnover_cap_delta_trend_min_threshold" in records.columns:
        non_null_early_turnover_cap_delta_trend_min = records[
            "early_rebalance_risk_turnover_cap_delta_trend_min_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_delta_trend_min_threshold"] = (
            float(non_null_early_turnover_cap_delta_trend_min.iloc[0])
            if not non_null_early_turnover_cap_delta_trend_min.empty
            else None
        )
    if "early_rebalance_risk_turnover_cap_delta_trend_max_threshold" in records.columns:
        non_null_early_turnover_cap_delta_trend_max = records[
            "early_rebalance_risk_turnover_cap_delta_trend_max_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_delta_trend_max_threshold"] = (
            float(non_null_early_turnover_cap_delta_trend_max.iloc[0])
            if not non_null_early_turnover_cap_delta_trend_max.empty
            else None
        )
    if "early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold" in records.columns:
        non_null_early_turnover_cap_delta_mean_reversion_min = records[
            "early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold"] = (
            float(non_null_early_turnover_cap_delta_mean_reversion_min.iloc[0])
            if not non_null_early_turnover_cap_delta_mean_reversion_min.empty
            else None
        )
    if "early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold" in records.columns:
        non_null_early_turnover_cap_delta_mean_reversion_max = records[
            "early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold"] = (
            float(non_null_early_turnover_cap_delta_mean_reversion_max.iloc[0])
            if not non_null_early_turnover_cap_delta_mean_reversion_max.empty
            else None
        )
    if "early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold" in records.columns:
        non_null_early_turnover_cap_proposed_turnover_min = records[
            "early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold"] = (
            float(non_null_early_turnover_cap_proposed_turnover_min.iloc[0])
            if not non_null_early_turnover_cap_proposed_turnover_min.empty
            else None
        )
    if "early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold" in records.columns:
        non_null_early_turnover_cap_proposed_turnover_max = records[
            "early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold"] = (
            float(non_null_early_turnover_cap_proposed_turnover_max.iloc[0])
            if not non_null_early_turnover_cap_proposed_turnover_max.empty
            else None
        )
    if "early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio" in records.columns:
        non_null_early_turnover_cap_relative_wealth = records[
            "early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio"] = (
            float(non_null_early_turnover_cap_relative_wealth.iloc[0])
            if not non_null_early_turnover_cap_relative_wealth.empty
            else None
        )
    if "early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio" in records.columns:
        non_null_early_turnover_cap_relative_wealth_max = records[
            "early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio"] = (
            float(non_null_early_turnover_cap_relative_wealth_max.iloc[0])
            if not non_null_early_turnover_cap_relative_wealth_max.empty
            else None
        )
    if "early_rebalance_risk_turnover_cap_allow_nonincreasing_risk_symbol" in records.columns:
        non_null_early_turnover_cap_allow_nonincreasing = records[
            "early_rebalance_risk_turnover_cap_allow_nonincreasing_risk_symbol"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_allow_nonincreasing_risk_symbol"] = (
            bool(non_null_early_turnover_cap_allow_nonincreasing.iloc[0])
            if not non_null_early_turnover_cap_allow_nonincreasing.empty
            else None
        )
    if "early_rebalance_risk_turnover_cap_use_penalty_state_filters" in records.columns:
        non_null_early_turnover_cap_state_filters = records[
            "early_rebalance_risk_turnover_cap_use_penalty_state_filters"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_use_penalty_state_filters"] = (
            bool(non_null_early_turnover_cap_state_filters.iloc[0])
            if not non_null_early_turnover_cap_state_filters.empty
            else None
        )
    if "early_rebalance_risk_turnover_cap_after" in records.columns:
        non_null_early_turnover_cap_after = records["early_rebalance_risk_turnover_cap_after"].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_after"] = (
            float(non_null_early_turnover_cap_after.iloc[0]) if not non_null_early_turnover_cap_after.empty else None
        )
    if "early_rebalance_risk_turnover_cap_before" in records.columns:
        non_null_early_turnover_cap_before = records["early_rebalance_risk_turnover_cap_before"].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_before"] = (
            float(non_null_early_turnover_cap_before.iloc[0]) if not non_null_early_turnover_cap_before.empty else None
        )
    if "early_rebalance_risk_turnover_cap_max_applications" in records.columns:
        non_null_early_turnover_cap_max_applications = records[
            "early_rebalance_risk_turnover_cap_max_applications"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_max_applications"] = (
            float(non_null_early_turnover_cap_max_applications.iloc[0])
            if not non_null_early_turnover_cap_max_applications.empty
            else None
        )
    if "early_rebalance_risk_turnover_cap_secondary_cap" in records.columns:
        non_null_early_turnover_cap_secondary_cap = records[
            "early_rebalance_risk_turnover_cap_secondary_cap"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_secondary_cap"] = (
            float(non_null_early_turnover_cap_secondary_cap.iloc[0])
            if not non_null_early_turnover_cap_secondary_cap.empty
            else None
        )
    if "early_rebalance_risk_turnover_cap_secondary_after_applications" in records.columns:
        non_null_early_turnover_cap_secondary_after_applications = records[
            "early_rebalance_risk_turnover_cap_secondary_after_applications"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_secondary_after_applications"] = (
            float(non_null_early_turnover_cap_secondary_after_applications.iloc[0])
            if not non_null_early_turnover_cap_secondary_after_applications.empty
            else None
        )
    if "early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold" in records.columns:
        non_null_secondary_benchmark_cumulative_return_min_threshold = records[
            "early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold"] = (
            float(non_null_secondary_benchmark_cumulative_return_min_threshold.iloc[0])
            if not non_null_secondary_benchmark_cumulative_return_min_threshold.empty
            else None
        )
    if "early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio" in records.columns:
        non_null_secondary_max_pre_trade_relative_wealth_ratio = records[
            "early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio"
        ].dropna()
        summary.loc["early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio"] = (
            float(non_null_secondary_max_pre_trade_relative_wealth_ratio.iloc[0])
            if not non_null_secondary_max_pre_trade_relative_wealth_ratio.empty
            else None
        )
    if "early_rebalance_risk_shallow_drawdown_turnover_cap" in records.columns:
        non_null_shallow_drawdown_cap = records["early_rebalance_risk_shallow_drawdown_turnover_cap"].dropna()
        summary.loc["early_rebalance_risk_shallow_drawdown_turnover_cap"] = (
            float(non_null_shallow_drawdown_cap.iloc[0]) if not non_null_shallow_drawdown_cap.empty else None
        )
    if "early_rebalance_risk_shallow_drawdown_turnover_cap_after" in records.columns:
        non_null_shallow_drawdown_cap_after = records[
            "early_rebalance_risk_shallow_drawdown_turnover_cap_after"
        ].dropna()
        summary.loc["early_rebalance_risk_shallow_drawdown_turnover_cap_after"] = (
            float(non_null_shallow_drawdown_cap_after.iloc[0])
            if not non_null_shallow_drawdown_cap_after.empty
            else None
        )
    if "early_rebalance_risk_shallow_drawdown_turnover_cap_before" in records.columns:
        non_null_shallow_drawdown_cap_before = records[
            "early_rebalance_risk_shallow_drawdown_turnover_cap_before"
        ].dropna()
        summary.loc["early_rebalance_risk_shallow_drawdown_turnover_cap_before"] = (
            float(non_null_shallow_drawdown_cap_before.iloc[0])
            if not non_null_shallow_drawdown_cap_before.empty
            else None
        )
    if "early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold" in records.columns:
        non_null_shallow_drawdown_cap_cash = records[
            "early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold"] = (
            float(non_null_shallow_drawdown_cap_cash.iloc[0])
            if not non_null_shallow_drawdown_cap_cash.empty
            else None
        )
    if "early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold" in records.columns:
        non_null_shallow_drawdown_cap_drawdown = records[
            "early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold"] = (
            float(non_null_shallow_drawdown_cap_drawdown.iloc[0])
            if not non_null_shallow_drawdown_cap_drawdown.empty
            else None
        )
    if "early_rebalance_risk_mean_reversion_turnover_cap" in records.columns:
        non_null_mean_reversion_cap = records["early_rebalance_risk_mean_reversion_turnover_cap"].dropna()
        summary.loc["early_rebalance_risk_mean_reversion_turnover_cap"] = (
            float(non_null_mean_reversion_cap.iloc[0]) if not non_null_mean_reversion_cap.empty else None
        )
    if "early_rebalance_risk_mean_reversion_action_smoothing" in records.columns:
        non_null_mean_reversion_action_smoothing = records[
            "early_rebalance_risk_mean_reversion_action_smoothing"
        ].dropna()
        summary.loc["early_rebalance_risk_mean_reversion_action_smoothing"] = (
            float(non_null_mean_reversion_action_smoothing.iloc[0])
            if not non_null_mean_reversion_action_smoothing.empty
            else None
        )
    if "early_rebalance_risk_mean_reversion_turnover_cap_after" in records.columns:
        non_null_mean_reversion_cap_after = records[
            "early_rebalance_risk_mean_reversion_turnover_cap_after"
        ].dropna()
        summary.loc["early_rebalance_risk_mean_reversion_turnover_cap_after"] = (
            float(non_null_mean_reversion_cap_after.iloc[0])
            if not non_null_mean_reversion_cap_after.empty
            else None
        )
    if "early_rebalance_risk_mean_reversion_turnover_cap_before" in records.columns:
        non_null_mean_reversion_cap_before = records[
            "early_rebalance_risk_mean_reversion_turnover_cap_before"
        ].dropna()
        summary.loc["early_rebalance_risk_mean_reversion_turnover_cap_before"] = (
            float(non_null_mean_reversion_cap_before.iloc[0])
            if not non_null_mean_reversion_cap_before.empty
            else None
        )
    if "early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold" in records.columns:
        non_null_mean_reversion_benchmark_cumulative_return_max_threshold = records[
            "early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold"] = (
            float(non_null_mean_reversion_benchmark_cumulative_return_max_threshold.iloc[0])
            if not non_null_mean_reversion_benchmark_cumulative_return_max_threshold.empty
            else None
        )
    if "early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold" in records.columns:
        non_null_mean_reversion_target_threshold = records[
            "early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold"] = (
            float(non_null_mean_reversion_target_threshold.iloc[0])
            if not non_null_mean_reversion_target_threshold.empty
            else None
        )
    if "early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold" in records.columns:
        non_null_mean_reversion_pre_trade_threshold = records[
            "early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold"] = (
            float(non_null_mean_reversion_pre_trade_threshold.iloc[0])
            if not non_null_mean_reversion_pre_trade_threshold.empty
            else None
        )
    if "early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold" in records.columns:
        non_null_mean_reversion_delta_threshold = records[
            "early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold"] = (
            float(non_null_mean_reversion_delta_threshold.iloc[0])
            if not non_null_mean_reversion_delta_threshold.empty
            else None
        )
    if "early_rebalance_risk_trend_turnover_cap" in records.columns:
        non_null_trend_cap = records["early_rebalance_risk_trend_turnover_cap"].dropna()
        summary.loc["early_rebalance_risk_trend_turnover_cap"] = (
            float(non_null_trend_cap.iloc[0]) if not non_null_trend_cap.empty else None
        )
    if "early_rebalance_risk_trend_turnover_cap_after" in records.columns:
        non_null_trend_cap_after = records["early_rebalance_risk_trend_turnover_cap_after"].dropna()
        summary.loc["early_rebalance_risk_trend_turnover_cap_after"] = (
            float(non_null_trend_cap_after.iloc[0]) if not non_null_trend_cap_after.empty else None
        )
    if "early_rebalance_risk_trend_turnover_cap_before" in records.columns:
        non_null_trend_cap_before = records["early_rebalance_risk_trend_turnover_cap_before"].dropna()
        summary.loc["early_rebalance_risk_trend_turnover_cap_before"] = (
            float(non_null_trend_cap_before.iloc[0]) if not non_null_trend_cap_before.empty else None
        )
    if "early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold" in records.columns:
        non_null_trend_benchmark_cumulative_return_max_threshold = records[
            "early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold"] = (
            float(non_null_trend_benchmark_cumulative_return_max_threshold.iloc[0])
            if not non_null_trend_benchmark_cumulative_return_max_threshold.empty
            else None
        )
    if "early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold" in records.columns:
        non_null_trend_target_threshold = records[
            "early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold"] = (
            float(non_null_trend_target_threshold.iloc[0]) if not non_null_trend_target_threshold.empty else None
        )
    if "early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold" in records.columns:
        non_null_trend_pre_trade_threshold = records[
            "early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold"] = (
            float(non_null_trend_pre_trade_threshold.iloc[0])
            if not non_null_trend_pre_trade_threshold.empty
            else None
        )
    if "early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold" in records.columns:
        non_null_trend_delta_threshold = records[
            "early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold"] = (
            float(non_null_trend_delta_threshold.iloc[0]) if not non_null_trend_delta_threshold.empty else None
        )
    if "early_rebalance_risk_deep_drawdown_turnover_cap" in records.columns:
        non_null_deep_drawdown_cap = records["early_rebalance_risk_deep_drawdown_turnover_cap"].dropna()
        summary.loc["early_rebalance_risk_deep_drawdown_turnover_cap"] = (
            float(non_null_deep_drawdown_cap.iloc[0]) if not non_null_deep_drawdown_cap.empty else None
        )
    if "early_rebalance_risk_deep_drawdown_turnover_cap_after" in records.columns:
        non_null_deep_drawdown_cap_after = records[
            "early_rebalance_risk_deep_drawdown_turnover_cap_after"
        ].dropna()
        summary.loc["early_rebalance_risk_deep_drawdown_turnover_cap_after"] = (
            float(non_null_deep_drawdown_cap_after.iloc[0]) if not non_null_deep_drawdown_cap_after.empty else None
        )
    if "early_rebalance_risk_deep_drawdown_turnover_cap_before" in records.columns:
        non_null_deep_drawdown_cap_before = records[
            "early_rebalance_risk_deep_drawdown_turnover_cap_before"
        ].dropna()
        summary.loc["early_rebalance_risk_deep_drawdown_turnover_cap_before"] = (
            float(non_null_deep_drawdown_cap_before.iloc[0]) if not non_null_deep_drawdown_cap_before.empty else None
        )
    if "early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold" in records.columns:
        non_null_deep_drawdown_cap_threshold = records[
            "early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold"] = (
            float(non_null_deep_drawdown_cap_threshold.iloc[0])
            if not non_null_deep_drawdown_cap_threshold.empty
            else None
        )
    if "early_rebalance_risk_repeat_turnover_cap" in records.columns:
        non_null_repeat_cap = records["early_rebalance_risk_repeat_turnover_cap"].dropna()
        summary.loc["early_rebalance_risk_repeat_turnover_cap"] = (
            float(non_null_repeat_cap.iloc[0]) if not non_null_repeat_cap.empty else None
        )
    if "early_rebalance_risk_repeat_action_smoothing" in records.columns:
        non_null_repeat_action_smoothing = records["early_rebalance_risk_repeat_action_smoothing"].dropna()
        summary.loc["early_rebalance_risk_repeat_action_smoothing"] = (
            float(non_null_repeat_action_smoothing.iloc[0])
            if not non_null_repeat_action_smoothing.empty
            else None
        )
    if "early_rebalance_risk_repeat_turnover_cap_after" in records.columns:
        non_null_repeat_after = records["early_rebalance_risk_repeat_turnover_cap_after"].dropna()
        summary.loc["early_rebalance_risk_repeat_turnover_cap_after"] = (
            float(non_null_repeat_after.iloc[0]) if not non_null_repeat_after.empty else None
        )
    if "early_rebalance_risk_repeat_turnover_cap_before" in records.columns:
        non_null_repeat_before = records["early_rebalance_risk_repeat_turnover_cap_before"].dropna()
        summary.loc["early_rebalance_risk_repeat_turnover_cap_before"] = (
            float(non_null_repeat_before.iloc[0]) if not non_null_repeat_before.empty else None
        )
    if "early_rebalance_risk_repeat_symbol" in records.columns:
        non_null_repeat_symbol = records["early_rebalance_risk_repeat_symbol"].dropna()
        summary.loc["early_rebalance_risk_repeat_symbol"] = (
            str(non_null_repeat_symbol.iloc[0]) if not non_null_repeat_symbol.empty else None
        )
    if "early_rebalance_risk_repeat_previous_cash_reduction_min" in records.columns:
        non_null_repeat_cash = records["early_rebalance_risk_repeat_previous_cash_reduction_min"].dropna()
        summary.loc["early_rebalance_risk_repeat_previous_cash_reduction_min"] = (
            float(non_null_repeat_cash.iloc[0]) if not non_null_repeat_cash.empty else None
        )
    if "early_rebalance_risk_repeat_previous_symbol_increase_min" in records.columns:
        non_null_repeat_symbol_increase = records[
            "early_rebalance_risk_repeat_previous_symbol_increase_min"
        ].dropna()
        summary.loc["early_rebalance_risk_repeat_previous_symbol_increase_min"] = (
            float(non_null_repeat_symbol_increase.iloc[0])
            if not non_null_repeat_symbol_increase.empty
            else None
        )
    if "early_rebalance_risk_repeat_unrecovered_turnover_cap" in records.columns:
        non_null_repeat_unrecovered_cap = records[
            "early_rebalance_risk_repeat_unrecovered_turnover_cap"
        ].dropna()
        summary.loc["early_rebalance_risk_repeat_unrecovered_turnover_cap"] = (
            float(non_null_repeat_unrecovered_cap.iloc[0])
            if not non_null_repeat_unrecovered_cap.empty
            else None
        )
    if "early_rebalance_risk_repeat_unrecovered_turnover_cap_after" in records.columns:
        non_null_repeat_unrecovered_after = records[
            "early_rebalance_risk_repeat_unrecovered_turnover_cap_after"
        ].dropna()
        summary.loc["early_rebalance_risk_repeat_unrecovered_turnover_cap_after"] = (
            float(non_null_repeat_unrecovered_after.iloc[0])
            if not non_null_repeat_unrecovered_after.empty
            else None
        )
    if "early_rebalance_risk_repeat_unrecovered_turnover_cap_before" in records.columns:
        non_null_repeat_unrecovered_before = records[
            "early_rebalance_risk_repeat_unrecovered_turnover_cap_before"
        ].dropna()
        summary.loc["early_rebalance_risk_repeat_unrecovered_turnover_cap_before"] = (
            float(non_null_repeat_unrecovered_before.iloc[0])
            if not non_null_repeat_unrecovered_before.empty
            else None
        )
    if "early_rebalance_risk_repeat_unrecovered_symbol" in records.columns:
        non_null_repeat_unrecovered_symbol = records[
            "early_rebalance_risk_repeat_unrecovered_symbol"
        ].dropna()
        summary.loc["early_rebalance_risk_repeat_unrecovered_symbol"] = (
            str(non_null_repeat_unrecovered_symbol.iloc[0])
            if not non_null_repeat_unrecovered_symbol.empty
            else None
        )
    if "early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min" in records.columns:
        non_null_repeat_unrecovered_cash = records[
            "early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min"
        ].dropna()
        summary.loc["early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min"] = (
            float(non_null_repeat_unrecovered_cash.iloc[0])
            if not non_null_repeat_unrecovered_cash.empty
            else None
        )
    if "early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min" in records.columns:
        non_null_repeat_unrecovered_symbol_increase = records[
            "early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min"
        ].dropna()
        summary.loc["early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min"] = (
            float(non_null_repeat_unrecovered_symbol_increase.iloc[0])
            if not non_null_repeat_unrecovered_symbol_increase.empty
            else None
        )
    if "early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery" in records.columns:
        non_null_repeat_unrecovered_recovery = records[
            "early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery"
        ].dropna()
        summary.loc["early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery"] = (
            float(non_null_repeat_unrecovered_recovery.iloc[0])
            if not non_null_repeat_unrecovered_recovery.empty
            else None
        )
    if "early_rebalance_risk_cumulative_turnover_cap" in records.columns:
        non_null_cumulative_cap = records["early_rebalance_risk_cumulative_turnover_cap"].dropna()
        summary.loc["early_rebalance_risk_cumulative_turnover_cap"] = (
            float(non_null_cumulative_cap.iloc[0]) if not non_null_cumulative_cap.empty else None
        )
    if "early_rebalance_risk_cumulative_turnover_cap_after" in records.columns:
        non_null_cumulative_after = records["early_rebalance_risk_cumulative_turnover_cap_after"].dropna()
        summary.loc["early_rebalance_risk_cumulative_turnover_cap_after"] = (
            float(non_null_cumulative_after.iloc[0]) if not non_null_cumulative_after.empty else None
        )
    if "early_rebalance_risk_cumulative_turnover_cap_before" in records.columns:
        non_null_cumulative_before = records["early_rebalance_risk_cumulative_turnover_cap_before"].dropna()
        summary.loc["early_rebalance_risk_cumulative_turnover_cap_before"] = (
            float(non_null_cumulative_before.iloc[0]) if not non_null_cumulative_before.empty else None
        )
    if "early_rebalance_risk_cumulative_symbol" in records.columns:
        non_null_cumulative_symbol = records["early_rebalance_risk_cumulative_symbol"].dropna()
        summary.loc["early_rebalance_risk_cumulative_symbol"] = (
            str(non_null_cumulative_symbol.iloc[0]) if not non_null_cumulative_symbol.empty else None
        )
    if "early_rebalance_risk_cumulative_cash_reduction_budget" in records.columns:
        non_null_cumulative_cash_budget = records[
            "early_rebalance_risk_cumulative_cash_reduction_budget"
        ].dropna()
        summary.loc["early_rebalance_risk_cumulative_cash_reduction_budget"] = (
            float(non_null_cumulative_cash_budget.iloc[0]) if not non_null_cumulative_cash_budget.empty else None
        )
    if "early_rebalance_risk_cumulative_symbol_increase_budget" in records.columns:
        non_null_cumulative_symbol_budget = records[
            "early_rebalance_risk_cumulative_symbol_increase_budget"
        ].dropna()
        summary.loc["early_rebalance_risk_cumulative_symbol_increase_budget"] = (
            float(non_null_cumulative_symbol_budget.iloc[0])
            if not non_null_cumulative_symbol_budget.empty
            else None
        )
    if "early_rebalance_risk_penalty_after" in records.columns:
        non_null_early_after = records["early_rebalance_risk_penalty_after"].dropna()
        summary.loc["early_rebalance_risk_penalty_after"] = (
            float(non_null_early_after.iloc[0]) if not non_null_early_after.empty else None
        )
    if "early_rebalance_risk_penalty_before" in records.columns:
        non_null_early_before = records["early_rebalance_risk_penalty_before"].dropna()
        summary.loc["early_rebalance_risk_penalty_before"] = (
            float(non_null_early_before.iloc[0]) if not non_null_early_before.empty else None
        )
    if "early_rebalance_risk_penalty_cash_max_threshold" in records.columns:
        non_null_early_cash = records["early_rebalance_risk_penalty_cash_max_threshold"].dropna()
        summary.loc["early_rebalance_risk_penalty_cash_max_threshold"] = (
            float(non_null_early_cash.iloc[0]) if not non_null_early_cash.empty else None
        )
    if "early_rebalance_risk_penalty_symbol" in records.columns:
        non_null_early_symbol = records["early_rebalance_risk_penalty_symbol"].dropna()
        summary.loc["early_rebalance_risk_penalty_symbol"] = (
            str(non_null_early_symbol.iloc[0]) if not non_null_early_symbol.empty else None
        )
    if "early_rebalance_risk_penalty_symbol_min_weight" in records.columns:
        non_null_early_symbol_min = records["early_rebalance_risk_penalty_symbol_min_weight"].dropna()
        summary.loc["early_rebalance_risk_penalty_symbol_min_weight"] = (
            float(non_null_early_symbol_min.iloc[0]) if not non_null_early_symbol_min.empty else None
        )
    if "early_rebalance_risk_penalty_symbol_max_weight" in records.columns:
        non_null_early_symbol_max = records["early_rebalance_risk_penalty_symbol_max_weight"].dropna()
        summary.loc["early_rebalance_risk_penalty_symbol_max_weight"] = (
            float(non_null_early_symbol_max.iloc[0]) if not non_null_early_symbol_max.empty else None
        )
    if "early_rebalance_risk_penalty_benchmark_drawdown_min_threshold" in records.columns:
        non_null_early_drawdown = records[
            "early_rebalance_risk_penalty_benchmark_drawdown_min_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_penalty_benchmark_drawdown_min_threshold"] = (
            float(non_null_early_drawdown.iloc[0]) if not non_null_early_drawdown.empty else None
        )
    if "early_rebalance_risk_penalty_benchmark_drawdown_max_threshold" in records.columns:
        non_null_early_drawdown_max = records[
            "early_rebalance_risk_penalty_benchmark_drawdown_max_threshold"
        ].dropna()
        summary.loc["early_rebalance_risk_penalty_benchmark_drawdown_max_threshold"] = (
            float(non_null_early_drawdown_max.iloc[0]) if not non_null_early_drawdown_max.empty else None
        )
    if "early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio" in records.columns:
        non_null_early_relative_wealth = records[
            "early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio"
        ].dropna()
        summary.loc["early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio"] = (
            float(non_null_early_relative_wealth.iloc[0]) if not non_null_early_relative_wealth.empty else None
        )
    if "early_benchmark_euphoria_penalty" in records.columns:
        non_null_early_euphoria_penalty = records["early_benchmark_euphoria_penalty"].dropna()
        summary.loc["early_benchmark_euphoria_penalty"] = (
            float(non_null_early_euphoria_penalty.iloc[0]) if not non_null_early_euphoria_penalty.empty else None
        )
    if "early_benchmark_euphoria_turnover_cap" in records.columns:
        non_null_early_euphoria_cap = records["early_benchmark_euphoria_turnover_cap"].dropna()
        summary.loc["early_benchmark_euphoria_turnover_cap"] = (
            float(non_null_early_euphoria_cap.iloc[0]) if not non_null_early_euphoria_cap.empty else None
        )
    if "early_benchmark_euphoria_before" in records.columns:
        non_null_early_euphoria_before = records["early_benchmark_euphoria_before"].dropna()
        summary.loc["early_benchmark_euphoria_before"] = (
            float(non_null_early_euphoria_before.iloc[0]) if not non_null_early_euphoria_before.empty else None
        )
    if "early_benchmark_euphoria_benchmark_drawdown_min_threshold" in records.columns:
        non_null_early_euphoria_drawdown = records[
            "early_benchmark_euphoria_benchmark_drawdown_min_threshold"
        ].dropna()
        summary.loc["early_benchmark_euphoria_benchmark_drawdown_min_threshold"] = (
            float(non_null_early_euphoria_drawdown.iloc[0]) if not non_null_early_euphoria_drawdown.empty else None
        )
    if "early_benchmark_euphoria_symbol" in records.columns:
        non_null_early_euphoria_symbol = records["early_benchmark_euphoria_symbol"].dropna()
        summary.loc["early_benchmark_euphoria_symbol"] = (
            str(non_null_early_euphoria_symbol.iloc[0]) if not non_null_early_euphoria_symbol.empty else None
        )
    if "late_rebalance_penalty" in records.columns:
        non_null_penalty = records["late_rebalance_penalty"].dropna()
        summary.loc["late_rebalance_penalty"] = float(non_null_penalty.iloc[0]) if not non_null_penalty.empty else None
    if "late_rebalance_penalty_after" in records.columns:
        non_null_penalty_after = records["late_rebalance_penalty_after"].dropna()
        summary.loc["late_rebalance_penalty_after"] = float(non_null_penalty_after.iloc[0]) if not non_null_penalty_after.empty else None
    if "late_rebalance_gate_after" in records.columns:
        non_null_gate_after = records["late_rebalance_gate_after"].dropna()
        summary.loc["late_rebalance_gate_after"] = float(non_null_gate_after.iloc[0]) if not non_null_gate_after.empty else None
    if "late_rebalance_gate_cash_threshold" in records.columns:
        non_null_gate_cash = records["late_rebalance_gate_cash_threshold"].dropna()
        summary.loc["late_rebalance_gate_cash_threshold"] = float(non_null_gate_cash.iloc[0]) if not non_null_gate_cash.empty else None
    if "late_rebalance_gate_target_cash_min_threshold" in records.columns:
        non_null_gate_target_cash = records["late_rebalance_gate_target_cash_min_threshold"].dropna()
        summary.loc["late_rebalance_gate_target_cash_min_threshold"] = (
            float(non_null_gate_target_cash.iloc[0]) if not non_null_gate_target_cash.empty else None
        )
    if "late_rebalance_gate_symbol" in records.columns:
        non_null_gate_symbol = records["late_rebalance_gate_symbol"].dropna()
        summary.loc["late_rebalance_gate_symbol"] = str(non_null_gate_symbol.iloc[0]) if not non_null_gate_symbol.empty else None
    if "late_rebalance_gate_symbol_max_weight" in records.columns:
        non_null_gate_symbol_max = records["late_rebalance_gate_symbol_max_weight"].dropna()
        summary.loc["late_rebalance_gate_symbol_max_weight"] = (
            float(non_null_gate_symbol_max.iloc[0]) if not non_null_gate_symbol_max.empty else None
        )
    if "late_rebalance_gate_cash_reduction_max" in records.columns:
        non_null_gate_cash_reduction_max = records["late_rebalance_gate_cash_reduction_max"].dropna()
        summary.loc["late_rebalance_gate_cash_reduction_max"] = (
            float(non_null_gate_cash_reduction_max.iloc[0]) if not non_null_gate_cash_reduction_max.empty else None
        )
    if "late_rebalance_gate_symbol_increase_max" in records.columns:
        non_null_gate_symbol_increase_max = records["late_rebalance_gate_symbol_increase_max"].dropna()
        summary.loc["late_rebalance_gate_symbol_increase_max"] = (
            float(non_null_gate_symbol_increase_max.iloc[0]) if not non_null_gate_symbol_increase_max.empty else None
        )
    if "late_rebalance_gate_refinement_condition_met" in records.columns:
        summary.loc["late_rebalance_gate_refinement_condition_rate"] = float(
            records["late_rebalance_gate_refinement_condition_met"].astype(float).mean()
        )
    if "late_defensive_posture_penalty" in records.columns:
        non_null_defensive_penalty = records["late_defensive_posture_penalty"].dropna()
        summary.loc["late_defensive_posture_penalty"] = (
            float(non_null_defensive_penalty.iloc[0]) if not non_null_defensive_penalty.empty else None
        )
    if "late_defensive_posture_penalty_after" in records.columns:
        non_null_defensive_after = records["late_defensive_posture_penalty_after"].dropna()
        summary.loc["late_defensive_posture_penalty_after"] = (
            float(non_null_defensive_after.iloc[0]) if not non_null_defensive_after.empty else None
        )
    if "late_defensive_posture_penalty_cash_min_threshold" in records.columns:
        non_null_defensive_cash = records["late_defensive_posture_penalty_cash_min_threshold"].dropna()
        summary.loc["late_defensive_posture_penalty_cash_min_threshold"] = (
            float(non_null_defensive_cash.iloc[0]) if not non_null_defensive_cash.empty else None
        )
    if "late_defensive_posture_penalty_symbol" in records.columns:
        non_null_defensive_symbol = records["late_defensive_posture_penalty_symbol"].dropna()
        summary.loc["late_defensive_posture_penalty_symbol"] = (
            str(non_null_defensive_symbol.iloc[0]) if not non_null_defensive_symbol.empty else None
        )
    if "late_defensive_posture_penalty_symbol_max_weight" in records.columns:
        non_null_defensive_symbol_max = records["late_defensive_posture_penalty_symbol_max_weight"].dropna()
        summary.loc["late_defensive_posture_penalty_symbol_max_weight"] = (
            float(non_null_defensive_symbol_max.iloc[0]) if not non_null_defensive_symbol_max.empty else None
        )
    if "late_trend_mean_reversion_conflict_penalty" in records.columns:
        non_null_conflict_penalty = records["late_trend_mean_reversion_conflict_penalty"].dropna()
        summary.loc["late_trend_mean_reversion_conflict_penalty"] = (
            float(non_null_conflict_penalty.iloc[0]) if not non_null_conflict_penalty.empty else None
        )
    if "late_trend_mean_reversion_conflict_penalty_after" in records.columns:
        non_null_conflict_after = records["late_trend_mean_reversion_conflict_penalty_after"].dropna()
        summary.loc["late_trend_mean_reversion_conflict_penalty_after"] = (
            float(non_null_conflict_after.iloc[0]) if not non_null_conflict_after.empty else None
        )
    if "late_trend_mean_reversion_conflict_trend_symbol" in records.columns:
        non_null_conflict_trend_symbol = records["late_trend_mean_reversion_conflict_trend_symbol"].dropna()
        summary.loc["late_trend_mean_reversion_conflict_trend_symbol"] = (
            str(non_null_conflict_trend_symbol.iloc[0]) if not non_null_conflict_trend_symbol.empty else None
        )
    if "late_trend_mean_reversion_conflict_trend_min_weight" in records.columns:
        non_null_conflict_trend_min = records["late_trend_mean_reversion_conflict_trend_min_weight"].dropna()
        summary.loc["late_trend_mean_reversion_conflict_trend_min_weight"] = (
            float(non_null_conflict_trend_min.iloc[0]) if not non_null_conflict_trend_min.empty else None
        )
    if "late_trend_mean_reversion_conflict_mean_reversion_symbol" in records.columns:
        non_null_conflict_mr_symbol = records[
            "late_trend_mean_reversion_conflict_mean_reversion_symbol"
        ].dropna()
        summary.loc["late_trend_mean_reversion_conflict_mean_reversion_symbol"] = (
            str(non_null_conflict_mr_symbol.iloc[0]) if not non_null_conflict_mr_symbol.empty else None
        )
    if "late_trend_mean_reversion_conflict_mean_reversion_min_weight" in records.columns:
        non_null_conflict_mr_min = records[
            "late_trend_mean_reversion_conflict_mean_reversion_min_weight"
        ].dropna()
        summary.loc["late_trend_mean_reversion_conflict_mean_reversion_min_weight"] = (
            float(non_null_conflict_mr_min.iloc[0]) if not non_null_conflict_mr_min.empty else None
        )
    if "state_trend_preservation_symbol" in records.columns:
        non_null_trend_symbol = records["state_trend_preservation_symbol"].dropna()
        summary.loc["state_trend_preservation_symbol"] = (
            str(non_null_trend_symbol.iloc[0]) if not non_null_trend_symbol.empty else None
        )
    if "state_trend_preservation_cash_max_threshold" in records.columns:
        non_null_trend_cash = records["state_trend_preservation_cash_max_threshold"].dropna()
        summary.loc["state_trend_preservation_cash_max_threshold"] = (
            float(non_null_trend_cash.iloc[0]) if not non_null_trend_cash.empty else None
        )
    if "state_trend_preservation_symbol_min_weight" in records.columns:
        non_null_trend_min_weight = records["state_trend_preservation_symbol_min_weight"].dropna()
        summary.loc["state_trend_preservation_symbol_min_weight"] = (
            float(non_null_trend_min_weight.iloc[0]) if not non_null_trend_min_weight.empty else None
        )
    if "state_trend_preservation_max_symbol_reduction" in records.columns:
        non_null_trend_max_reduction = records["state_trend_preservation_max_symbol_reduction"].dropna()
        summary.loc["state_trend_preservation_max_symbol_reduction"] = (
            float(non_null_trend_max_reduction.iloc[0]) if not non_null_trend_max_reduction.empty else None
        )
    if "excess_cash_weight" in records.columns:
        summary.loc["average_excess_cash_weight"] = float(records["excess_cash_weight"].astype(float).mean())
    if "cash_weight_penalty_component" in records.columns:
        summary.loc["average_cash_weight_penalty_component"] = float(records["cash_weight_penalty_component"].astype(float).mean())
    for column in records.columns:
        if column.startswith("target_weight_"):
            summary.loc[f"average_{column}"] = float(records[column].astype(float).mean())
        elif column.startswith("gross_return_contribution_"):
            summary.loc[f"average_{column}"] = float(records[column].astype(float).mean())

    cash_column = f"target_weight_{cash_symbol}"
    if cash_column in records.columns:
        summary.loc["average_cash_weight"] = float(records[cash_column].astype(float).mean())
        summary.loc["average_risky_weight"] = float(1.0 - records[cash_column].astype(float).mean())

    if "benchmark_return" in records.columns:
        benchmark_returns = records["benchmark_return"].astype(float)
        benchmark_wealth_index = initial_wealth * (1.0 + benchmark_returns).cumprod()
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = float(active_returns.std(ddof=0) * (periods_per_year ** 0.5))
        annualized_active_return = float(active_returns.mean() * periods_per_year)
        summary.loc["benchmark_total_return"] = float(benchmark_wealth_index.iloc[-1] / initial_wealth - 1.0)
        summary.loc["relative_total_return"] = float(wealth_index.iloc[-1] / benchmark_wealth_index.iloc[-1] - 1.0)
        summary.loc["average_active_return"] = float(active_returns.mean())
        summary.loc["tracking_error"] = tracking_error
        summary.loc["information_ratio"] = annualized_active_return / tracking_error if tracking_error > 0 else None

    return summary


def _build_named_metric_comparison(
    left_summary: pd.Series,
    right_summary: pd.Series,
    left_label: str,
    right_label: str,
) -> dict[str, dict[str, float | None]]:
    comparison: dict[str, dict[str, float | None]] = {}
    for metric in COMMON_COMPARISON_METRICS:
        left_value = left_summary.get(metric)
        right_value = right_summary.get(metric)
        left_serialized = _serialize_scalar(left_value)
        right_serialized = _serialize_scalar(right_value)
        difference = None
        if left_serialized is not None and right_serialized is not None:
            difference = float(left_serialized) - float(right_serialized)
        comparison[metric] = {
            left_label: left_serialized,
            right_label: right_serialized,
            "difference": difference,
        }
    return comparison


def _build_common_metric_comparison(
    policy_summary: pd.Series,
    optimizer_summary: pd.Series,
) -> dict[str, dict[str, float | None]]:
    return _build_named_metric_comparison(policy_summary, optimizer_summary, left_label="policy", right_label="optimizer")


def _save_rollout_artifacts(
    output_dir: Path,
    prefix: str,
    label: str,
    records: pd.DataFrame,
    summary: pd.Series,
) -> None:
    records.to_csv(output_dir / f"{prefix}_{label}_rollout.csv", index=True)
    (output_dir / f"{prefix}_{label}_summary.json").write_text(json.dumps(_serialize_series(summary), indent=2), encoding="utf-8")


def _save_backtest_artifacts(
    output_dir: Path,
    prefix: str,
    result,
) -> None:
    result.weights.to_csv(output_dir / f"{prefix}_optimizer_weights.csv")
    result.ending_weights.to_csv(output_dir / f"{prefix}_optimizer_ending_weights.csv")
    result.portfolio_returns.to_csv(output_dir / f"{prefix}_optimizer_portfolio_returns.csv", header=True)
    result.wealth_index.to_csv(output_dir / f"{prefix}_optimizer_wealth_index.csv", header=True)
    result.turnover.to_csv(output_dir / f"{prefix}_optimizer_turnover.csv", header=True)
    result.trading_costs.to_csv(output_dir / f"{prefix}_optimizer_trading_costs.csv", header=True)
    (output_dir / f"{prefix}_optimizer_summary.json").write_text(json.dumps(_serialize_series(result.summary), indent=2), encoding="utf-8")


def _save_rebalance_impact_artifacts(
    output_dir: Path,
    prefix: str,
    records: pd.DataFrame,
    summary: pd.Series,
) -> None:
    records.to_csv(output_dir / f"{prefix}_rebalance_impacts.csv", index=True)
    (output_dir / f"{prefix}_rebalance_impact_summary.json").write_text(json.dumps(_serialize_series(summary), indent=2), encoding="utf-8")


def _split_to_metadata(split: SuggestedTimeSeriesSplit) -> dict[str, object]:
    return {
        "method": split.method,
        "score": split.score,
        "search_step": split.search_step,
        "target_rows": split.target_rows,
        "regime_coverage": split.regime_coverage,
        "regime_distance": split.regime_distance,
        "train": {
            "start": str(split.train.start_label),
            "end": str(split.train.end_label),
            "rows": split.train.rows,
        },
        "validation": {
            "start": str(split.validation.start_label),
            "end": str(split.validation.end_label),
            "rows": split.validation.rows,
        },
        "test": {
            "start": str(split.test.start_label),
            "end": str(split.test.end_label),
            "rows": split.test.rows,
        },
    }


def _slice_returns_for_split(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series | None,
    split: SuggestedTimeSeriesSplit,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series | None, pd.Series | None, pd.Series | None]:
    train_returns = returns.iloc[split.train.start_index : split.train.end_index].copy()
    validation_returns = returns.iloc[split.validation.start_index : split.validation.end_index].copy()
    test_returns = returns.iloc[split.test.start_index : split.test.end_index].copy()
    train_benchmark = benchmark_returns.iloc[split.train.start_index : split.train.end_index].copy() if benchmark_returns is not None else None
    validation_benchmark = benchmark_returns.iloc[split.validation.start_index : split.validation.end_index].copy() if benchmark_returns is not None else None
    test_benchmark = benchmark_returns.iloc[split.test.start_index : split.test.end_index].copy() if benchmark_returns is not None else None
    return train_returns, validation_returns, test_returns, train_benchmark, validation_benchmark, test_benchmark


def _run_single_fold(
    split: SuggestedTimeSeriesSplit,
    returns: pd.DataFrame,
    benchmark_returns: pd.Series | None,
    args,
    optimizer_config: WealthFirstConfig,
    policy_config: WealthFirstConfig,
    output_dir: Path,
    dependencies,
) -> dict[str, object]:
    PPO, EvalCallback, evaluate_policy, Monitor, DummyVecEnv, VecNormalize, sync_envs_normalization = dependencies
    benchmark_regime_summary_observations, benchmark_regime_relative_cumulative_observations = (
        _resolve_benchmark_regime_observation_flags(args)
    )
    train_returns, validation_returns, test_returns, train_benchmark, validation_benchmark, test_benchmark = _slice_returns_for_split(
        returns,
        benchmark_returns,
        split,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "best_model"
    eval_log_dir = output_dir / "eval_logs"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log = str(Path(args.tensorboard_log)) if args.tensorboard_log else None
    (output_dir / "split_windows.json").write_text(json.dumps(_split_to_metadata(split), indent=2), encoding="utf-8")

    def build_train_env():
        return Monitor(
            WealthFirstEnv(
                train_returns,
                lookback=args.lookback,
                config=policy_config,
                benchmark_returns=train_benchmark,
                benchmark_relative_observations=args.benchmark_relative_observations,
                benchmark_regime_observations=args.benchmark_regime_observations,
                benchmark_regime_summary_observations=benchmark_regime_summary_observations,
                benchmark_regime_relative_cumulative_observations=benchmark_regime_relative_cumulative_observations,
                episode_length=args.episode_length,
                random_episode_start=True,
                action_smoothing=args.action_smoothing,
                no_trade_band=args.no_trade_band,
                max_executed_rebalances=args.max_executed_rebalances,
                rebalance_cooldown_steps=args.rebalance_cooldown_steps,
                early_rebalance_risk_penalty=args.early_rebalance_risk_penalty,
                early_rebalance_risk_turnover_cap=args.early_rebalance_risk_turnover_cap,
                early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold=args.early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold,
                early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold=args.early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold,
                early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight=args.early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight,
                early_rebalance_risk_turnover_cap_target_cash_min_threshold=args.early_rebalance_risk_turnover_cap_target_cash_min_threshold,
                early_rebalance_risk_turnover_cap_target_cash_max_threshold=args.early_rebalance_risk_turnover_cap_target_cash_max_threshold,
                early_rebalance_risk_turnover_cap_target_trend_min_threshold=args.early_rebalance_risk_turnover_cap_target_trend_min_threshold,
                early_rebalance_risk_turnover_cap_target_trend_max_threshold=args.early_rebalance_risk_turnover_cap_target_trend_max_threshold,
                early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold=args.early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold,
                early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold=args.early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold,
                early_rebalance_risk_turnover_cap_delta_cash_min_threshold=args.early_rebalance_risk_turnover_cap_delta_cash_min_threshold,
                early_rebalance_risk_turnover_cap_delta_cash_max_threshold=args.early_rebalance_risk_turnover_cap_delta_cash_max_threshold,
                early_rebalance_risk_turnover_cap_delta_trend_min_threshold=args.early_rebalance_risk_turnover_cap_delta_trend_min_threshold,
                early_rebalance_risk_turnover_cap_delta_trend_max_threshold=args.early_rebalance_risk_turnover_cap_delta_trend_max_threshold,
                early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold=args.early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold,
                early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold=args.early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold,
                early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold=args.early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold,
                early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold=args.early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold,
                early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio=args.early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio,
                early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio=args.early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio,
                early_rebalance_risk_turnover_cap_allow_nonincreasing_risk_symbol=args.early_rebalance_risk_turnover_cap_allow_nonincreasing_risk_symbol,
                early_rebalance_risk_turnover_cap_use_penalty_state_filters=args.early_rebalance_risk_turnover_cap_use_penalty_state_filters,
                early_rebalance_risk_turnover_cap_after=args.early_rebalance_risk_turnover_cap_after,
                early_rebalance_risk_turnover_cap_before=args.early_rebalance_risk_turnover_cap_before,
                early_rebalance_risk_turnover_cap_max_applications=args.early_rebalance_risk_turnover_cap_max_applications,
                early_rebalance_risk_turnover_cap_secondary_cap=args.early_rebalance_risk_turnover_cap_secondary_cap,
                early_rebalance_risk_turnover_cap_secondary_after_applications=args.early_rebalance_risk_turnover_cap_secondary_after_applications,
                early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold=args.early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold,
                early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio=args.early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio,
                early_rebalance_risk_shallow_drawdown_turnover_cap=args.early_rebalance_risk_shallow_drawdown_turnover_cap,
                early_rebalance_risk_shallow_drawdown_turnover_cap_after=args.early_rebalance_risk_shallow_drawdown_turnover_cap_after,
                early_rebalance_risk_shallow_drawdown_turnover_cap_before=args.early_rebalance_risk_shallow_drawdown_turnover_cap_before,
                early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold=args.early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold,
                early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold=args.early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold,
                early_rebalance_risk_mean_reversion_turnover_cap=args.early_rebalance_risk_mean_reversion_turnover_cap,
                early_rebalance_risk_mean_reversion_action_smoothing=args.early_rebalance_risk_mean_reversion_action_smoothing,
                early_rebalance_risk_mean_reversion_turnover_cap_after=args.early_rebalance_risk_mean_reversion_turnover_cap_after,
                early_rebalance_risk_mean_reversion_turnover_cap_before=args.early_rebalance_risk_mean_reversion_turnover_cap_before,
                early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold=args.early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold,
                early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold=args.early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold,
                early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold=args.early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold,
                early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold=args.early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold,
                early_rebalance_risk_trend_turnover_cap=args.early_rebalance_risk_trend_turnover_cap,
                early_rebalance_risk_trend_turnover_cap_after=args.early_rebalance_risk_trend_turnover_cap_after,
                early_rebalance_risk_trend_turnover_cap_before=args.early_rebalance_risk_trend_turnover_cap_before,
                early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold=args.early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold,
                early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold=args.early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold,
                early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold=args.early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold,
                early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold=args.early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold,
                early_rebalance_risk_deep_drawdown_turnover_cap=args.early_rebalance_risk_deep_drawdown_turnover_cap,
                early_rebalance_risk_deep_drawdown_turnover_cap_after=args.early_rebalance_risk_deep_drawdown_turnover_cap_after,
                early_rebalance_risk_deep_drawdown_turnover_cap_before=args.early_rebalance_risk_deep_drawdown_turnover_cap_before,
                early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold=args.early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold,
                early_rebalance_risk_repeat_turnover_cap=args.early_rebalance_risk_repeat_turnover_cap,
                early_rebalance_risk_repeat_action_smoothing=args.early_rebalance_risk_repeat_action_smoothing,
                early_rebalance_risk_repeat_turnover_cap_after=args.early_rebalance_risk_repeat_turnover_cap_after,
                early_rebalance_risk_repeat_turnover_cap_before=args.early_rebalance_risk_repeat_turnover_cap_before,
                early_rebalance_risk_repeat_symbol=args.early_rebalance_risk_repeat_symbol,
                early_rebalance_risk_repeat_previous_cash_reduction_min=args.early_rebalance_risk_repeat_previous_cash_reduction_min,
                early_rebalance_risk_repeat_previous_symbol_increase_min=args.early_rebalance_risk_repeat_previous_symbol_increase_min,
                early_rebalance_risk_repeat_unrecovered_turnover_cap=args.early_rebalance_risk_repeat_unrecovered_turnover_cap,
                early_rebalance_risk_repeat_unrecovered_turnover_cap_after=args.early_rebalance_risk_repeat_unrecovered_turnover_cap_after,
                early_rebalance_risk_repeat_unrecovered_turnover_cap_before=args.early_rebalance_risk_repeat_unrecovered_turnover_cap_before,
                early_rebalance_risk_repeat_unrecovered_symbol=args.early_rebalance_risk_repeat_unrecovered_symbol,
                early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min=args.early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min,
                early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min=args.early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min,
                early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery=args.early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery,
                early_rebalance_risk_cumulative_turnover_cap=args.early_rebalance_risk_cumulative_turnover_cap,
                early_rebalance_risk_cumulative_turnover_cap_after=args.early_rebalance_risk_cumulative_turnover_cap_after,
                early_rebalance_risk_cumulative_turnover_cap_before=args.early_rebalance_risk_cumulative_turnover_cap_before,
                early_rebalance_risk_cumulative_symbol=args.early_rebalance_risk_cumulative_symbol,
                early_rebalance_risk_cumulative_cash_reduction_budget=args.early_rebalance_risk_cumulative_cash_reduction_budget,
                early_rebalance_risk_cumulative_symbol_increase_budget=args.early_rebalance_risk_cumulative_symbol_increase_budget,
                early_rebalance_risk_penalty_after=args.early_rebalance_risk_penalty_after,
                early_rebalance_risk_penalty_before=args.early_rebalance_risk_penalty_before,
                early_rebalance_risk_penalty_cash_max_threshold=args.early_rebalance_risk_penalty_cash_max_threshold,
                early_rebalance_risk_penalty_symbol=args.early_rebalance_risk_penalty_symbol,
                early_rebalance_risk_penalty_symbol_min_weight=args.early_rebalance_risk_penalty_symbol_min_weight,
                early_rebalance_risk_penalty_symbol_max_weight=args.early_rebalance_risk_penalty_symbol_max_weight,
                early_rebalance_risk_penalty_benchmark_drawdown_min_threshold=args.early_rebalance_risk_penalty_benchmark_drawdown_min_threshold,
                early_rebalance_risk_penalty_benchmark_drawdown_max_threshold=args.early_rebalance_risk_penalty_benchmark_drawdown_max_threshold,
                early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio=args.early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio,
                early_benchmark_euphoria_penalty=args.early_benchmark_euphoria_penalty,
                early_benchmark_euphoria_turnover_cap=args.early_benchmark_euphoria_turnover_cap,
                early_benchmark_euphoria_before=args.early_benchmark_euphoria_before,
                early_benchmark_euphoria_benchmark_drawdown_min_threshold=args.early_benchmark_euphoria_benchmark_drawdown_min_threshold,
                early_benchmark_euphoria_symbol=args.early_benchmark_euphoria_symbol,
                late_rebalance_penalty=args.late_rebalance_penalty,
                late_rebalance_penalty_after=args.late_rebalance_penalty_after,
                late_rebalance_gate_after=args.late_rebalance_gate_after,
                late_rebalance_gate_cash_threshold=args.late_rebalance_gate_cash_threshold,
                late_rebalance_gate_target_cash_min_threshold=args.late_rebalance_gate_target_cash_min_threshold,
                late_rebalance_gate_symbol=args.late_rebalance_gate_symbol,
                late_rebalance_gate_symbol_max_weight=args.late_rebalance_gate_symbol_max_weight,
                late_rebalance_gate_cash_reduction_max=args.late_rebalance_gate_cash_reduction_max,
                late_rebalance_gate_symbol_increase_max=args.late_rebalance_gate_symbol_increase_max,
                late_defensive_posture_penalty=args.late_defensive_posture_penalty,
                late_defensive_posture_penalty_after=args.late_defensive_posture_penalty_after,
                late_defensive_posture_penalty_cash_min_threshold=args.late_defensive_posture_penalty_cash_min_threshold,
                late_defensive_posture_penalty_symbol=args.late_defensive_posture_penalty_symbol,
                late_defensive_posture_penalty_symbol_max_weight=args.late_defensive_posture_penalty_symbol_max_weight,
                late_trend_mean_reversion_conflict_penalty=args.late_trend_mean_reversion_conflict_penalty,
                late_trend_mean_reversion_conflict_penalty_after=args.late_trend_mean_reversion_conflict_penalty_after,
                late_trend_mean_reversion_conflict_trend_symbol=args.late_trend_mean_reversion_conflict_trend_symbol,
                late_trend_mean_reversion_conflict_trend_min_weight=args.late_trend_mean_reversion_conflict_trend_min_weight,
                late_trend_mean_reversion_conflict_mean_reversion_symbol=args.late_trend_mean_reversion_conflict_mean_reversion_symbol,
                late_trend_mean_reversion_conflict_mean_reversion_min_weight=args.late_trend_mean_reversion_conflict_mean_reversion_min_weight,
                state_trend_preservation_symbol=args.state_trend_preservation_symbol,
                state_trend_preservation_cash_max_threshold=args.state_trend_preservation_cash_max_threshold,
                state_trend_preservation_symbol_min_weight=args.state_trend_preservation_symbol_min_weight,
                state_trend_preservation_max_symbol_reduction=args.state_trend_preservation_max_symbol_reduction,
                cash_weight_penalty=args.cash_weight_penalty,
                cash_target_weight=args.cash_target_weight,
            )
        )

    def build_validation_env():
        return Monitor(
            WealthFirstEnv(
                validation_returns,
                lookback=args.lookback,
                config=policy_config,
                benchmark_returns=validation_benchmark,
                benchmark_relative_observations=args.benchmark_relative_observations,
                benchmark_regime_observations=args.benchmark_regime_observations,
                benchmark_regime_summary_observations=benchmark_regime_summary_observations,
                benchmark_regime_relative_cumulative_observations=benchmark_regime_relative_cumulative_observations,
                episode_length=None,
                random_episode_start=False,
                action_smoothing=args.action_smoothing,
                no_trade_band=args.no_trade_band,
                max_executed_rebalances=args.max_executed_rebalances,
                rebalance_cooldown_steps=args.rebalance_cooldown_steps,
                early_rebalance_risk_penalty=args.early_rebalance_risk_penalty,
                early_rebalance_risk_turnover_cap=args.early_rebalance_risk_turnover_cap,
                early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold=args.early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold,
                early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold=args.early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold,
                early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight=args.early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight,
                early_rebalance_risk_turnover_cap_target_cash_min_threshold=args.early_rebalance_risk_turnover_cap_target_cash_min_threshold,
                early_rebalance_risk_turnover_cap_target_cash_max_threshold=args.early_rebalance_risk_turnover_cap_target_cash_max_threshold,
                early_rebalance_risk_turnover_cap_target_trend_min_threshold=args.early_rebalance_risk_turnover_cap_target_trend_min_threshold,
                early_rebalance_risk_turnover_cap_target_trend_max_threshold=args.early_rebalance_risk_turnover_cap_target_trend_max_threshold,
                early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold=args.early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold,
                early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold=args.early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold,
                early_rebalance_risk_turnover_cap_delta_cash_min_threshold=args.early_rebalance_risk_turnover_cap_delta_cash_min_threshold,
                early_rebalance_risk_turnover_cap_delta_cash_max_threshold=args.early_rebalance_risk_turnover_cap_delta_cash_max_threshold,
                early_rebalance_risk_turnover_cap_delta_trend_min_threshold=args.early_rebalance_risk_turnover_cap_delta_trend_min_threshold,
                early_rebalance_risk_turnover_cap_delta_trend_max_threshold=args.early_rebalance_risk_turnover_cap_delta_trend_max_threshold,
                early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold=args.early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold,
                early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold=args.early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold,
                early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold=args.early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold,
                early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold=args.early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold,
                early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio=args.early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio,
                early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio=args.early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio,
                early_rebalance_risk_turnover_cap_allow_nonincreasing_risk_symbol=args.early_rebalance_risk_turnover_cap_allow_nonincreasing_risk_symbol,
                early_rebalance_risk_turnover_cap_use_penalty_state_filters=args.early_rebalance_risk_turnover_cap_use_penalty_state_filters,
                early_rebalance_risk_turnover_cap_after=args.early_rebalance_risk_turnover_cap_after,
                early_rebalance_risk_turnover_cap_before=args.early_rebalance_risk_turnover_cap_before,
                early_rebalance_risk_turnover_cap_max_applications=args.early_rebalance_risk_turnover_cap_max_applications,
                early_rebalance_risk_turnover_cap_secondary_cap=args.early_rebalance_risk_turnover_cap_secondary_cap,
                early_rebalance_risk_turnover_cap_secondary_after_applications=args.early_rebalance_risk_turnover_cap_secondary_after_applications,
                early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold=args.early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold,
                early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio=args.early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio,
                early_rebalance_risk_shallow_drawdown_turnover_cap=args.early_rebalance_risk_shallow_drawdown_turnover_cap,
                early_rebalance_risk_shallow_drawdown_turnover_cap_after=args.early_rebalance_risk_shallow_drawdown_turnover_cap_after,
                early_rebalance_risk_shallow_drawdown_turnover_cap_before=args.early_rebalance_risk_shallow_drawdown_turnover_cap_before,
                early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold=args.early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold,
                early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold=args.early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold,
                early_rebalance_risk_mean_reversion_turnover_cap=args.early_rebalance_risk_mean_reversion_turnover_cap,
                early_rebalance_risk_mean_reversion_action_smoothing=args.early_rebalance_risk_mean_reversion_action_smoothing,
                early_rebalance_risk_mean_reversion_turnover_cap_after=args.early_rebalance_risk_mean_reversion_turnover_cap_after,
                early_rebalance_risk_mean_reversion_turnover_cap_before=args.early_rebalance_risk_mean_reversion_turnover_cap_before,
                early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold=args.early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold,
                early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold=args.early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold,
                early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold=args.early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold,
                early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold=args.early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold,
                early_rebalance_risk_trend_turnover_cap=args.early_rebalance_risk_trend_turnover_cap,
                early_rebalance_risk_trend_turnover_cap_after=args.early_rebalance_risk_trend_turnover_cap_after,
                early_rebalance_risk_trend_turnover_cap_before=args.early_rebalance_risk_trend_turnover_cap_before,
                early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold=args.early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold,
                early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold=args.early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold,
                early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold=args.early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold,
                early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold=args.early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold,
                early_rebalance_risk_deep_drawdown_turnover_cap=args.early_rebalance_risk_deep_drawdown_turnover_cap,
                early_rebalance_risk_deep_drawdown_turnover_cap_after=args.early_rebalance_risk_deep_drawdown_turnover_cap_after,
                early_rebalance_risk_deep_drawdown_turnover_cap_before=args.early_rebalance_risk_deep_drawdown_turnover_cap_before,
                early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold=args.early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold,
                early_rebalance_risk_repeat_turnover_cap=args.early_rebalance_risk_repeat_turnover_cap,
                early_rebalance_risk_repeat_action_smoothing=args.early_rebalance_risk_repeat_action_smoothing,
                early_rebalance_risk_repeat_turnover_cap_after=args.early_rebalance_risk_repeat_turnover_cap_after,
                early_rebalance_risk_repeat_turnover_cap_before=args.early_rebalance_risk_repeat_turnover_cap_before,
                early_rebalance_risk_repeat_symbol=args.early_rebalance_risk_repeat_symbol,
                early_rebalance_risk_repeat_previous_cash_reduction_min=args.early_rebalance_risk_repeat_previous_cash_reduction_min,
                early_rebalance_risk_repeat_previous_symbol_increase_min=args.early_rebalance_risk_repeat_previous_symbol_increase_min,
                early_rebalance_risk_repeat_unrecovered_turnover_cap=args.early_rebalance_risk_repeat_unrecovered_turnover_cap,
                early_rebalance_risk_repeat_unrecovered_turnover_cap_after=args.early_rebalance_risk_repeat_unrecovered_turnover_cap_after,
                early_rebalance_risk_repeat_unrecovered_turnover_cap_before=args.early_rebalance_risk_repeat_unrecovered_turnover_cap_before,
                early_rebalance_risk_repeat_unrecovered_symbol=args.early_rebalance_risk_repeat_unrecovered_symbol,
                early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min=args.early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min,
                early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min=args.early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min,
                early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery=args.early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery,
                early_rebalance_risk_cumulative_turnover_cap=args.early_rebalance_risk_cumulative_turnover_cap,
                early_rebalance_risk_cumulative_turnover_cap_after=args.early_rebalance_risk_cumulative_turnover_cap_after,
                early_rebalance_risk_cumulative_turnover_cap_before=args.early_rebalance_risk_cumulative_turnover_cap_before,
                early_rebalance_risk_cumulative_symbol=args.early_rebalance_risk_cumulative_symbol,
                early_rebalance_risk_cumulative_cash_reduction_budget=args.early_rebalance_risk_cumulative_cash_reduction_budget,
                early_rebalance_risk_cumulative_symbol_increase_budget=args.early_rebalance_risk_cumulative_symbol_increase_budget,
                early_rebalance_risk_penalty_after=args.early_rebalance_risk_penalty_after,
                early_rebalance_risk_penalty_before=args.early_rebalance_risk_penalty_before,
                early_rebalance_risk_penalty_cash_max_threshold=args.early_rebalance_risk_penalty_cash_max_threshold,
                early_rebalance_risk_penalty_symbol=args.early_rebalance_risk_penalty_symbol,
                early_rebalance_risk_penalty_symbol_min_weight=args.early_rebalance_risk_penalty_symbol_min_weight,
                early_rebalance_risk_penalty_symbol_max_weight=args.early_rebalance_risk_penalty_symbol_max_weight,
                early_rebalance_risk_penalty_benchmark_drawdown_min_threshold=args.early_rebalance_risk_penalty_benchmark_drawdown_min_threshold,
                early_rebalance_risk_penalty_benchmark_drawdown_max_threshold=args.early_rebalance_risk_penalty_benchmark_drawdown_max_threshold,
                early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio=args.early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio,
                early_benchmark_euphoria_penalty=args.early_benchmark_euphoria_penalty,
                early_benchmark_euphoria_turnover_cap=args.early_benchmark_euphoria_turnover_cap,
                early_benchmark_euphoria_before=args.early_benchmark_euphoria_before,
                early_benchmark_euphoria_benchmark_drawdown_min_threshold=args.early_benchmark_euphoria_benchmark_drawdown_min_threshold,
                early_benchmark_euphoria_symbol=args.early_benchmark_euphoria_symbol,
                late_rebalance_penalty=args.late_rebalance_penalty,
                late_rebalance_penalty_after=args.late_rebalance_penalty_after,
                late_rebalance_gate_after=args.late_rebalance_gate_after,
                late_rebalance_gate_cash_threshold=args.late_rebalance_gate_cash_threshold,
                late_rebalance_gate_target_cash_min_threshold=args.late_rebalance_gate_target_cash_min_threshold,
                late_rebalance_gate_symbol=args.late_rebalance_gate_symbol,
                late_rebalance_gate_symbol_max_weight=args.late_rebalance_gate_symbol_max_weight,
                late_rebalance_gate_cash_reduction_max=args.late_rebalance_gate_cash_reduction_max,
                late_rebalance_gate_symbol_increase_max=args.late_rebalance_gate_symbol_increase_max,
                late_defensive_posture_penalty=args.late_defensive_posture_penalty,
                late_defensive_posture_penalty_after=args.late_defensive_posture_penalty_after,
                late_defensive_posture_penalty_cash_min_threshold=args.late_defensive_posture_penalty_cash_min_threshold,
                late_defensive_posture_penalty_symbol=args.late_defensive_posture_penalty_symbol,
                late_defensive_posture_penalty_symbol_max_weight=args.late_defensive_posture_penalty_symbol_max_weight,
                late_trend_mean_reversion_conflict_penalty=args.late_trend_mean_reversion_conflict_penalty,
                late_trend_mean_reversion_conflict_penalty_after=args.late_trend_mean_reversion_conflict_penalty_after,
                late_trend_mean_reversion_conflict_trend_symbol=args.late_trend_mean_reversion_conflict_trend_symbol,
                late_trend_mean_reversion_conflict_trend_min_weight=args.late_trend_mean_reversion_conflict_trend_min_weight,
                late_trend_mean_reversion_conflict_mean_reversion_symbol=args.late_trend_mean_reversion_conflict_mean_reversion_symbol,
                late_trend_mean_reversion_conflict_mean_reversion_min_weight=args.late_trend_mean_reversion_conflict_mean_reversion_min_weight,
                state_trend_preservation_symbol=args.state_trend_preservation_symbol,
                state_trend_preservation_cash_max_threshold=args.state_trend_preservation_cash_max_threshold,
                state_trend_preservation_symbol_min_weight=args.state_trend_preservation_symbol_min_weight,
                state_trend_preservation_max_symbol_reduction=args.state_trend_preservation_max_symbol_reduction,
                cash_weight_penalty=args.cash_weight_penalty,
                cash_target_weight=args.cash_target_weight,
            )
        )

    def build_test_env():
        return Monitor(
            WealthFirstEnv(
                test_returns,
                lookback=args.lookback,
                config=policy_config,
                benchmark_returns=test_benchmark,
                benchmark_relative_observations=args.benchmark_relative_observations,
                benchmark_regime_observations=args.benchmark_regime_observations,
                benchmark_regime_summary_observations=benchmark_regime_summary_observations,
                benchmark_regime_relative_cumulative_observations=benchmark_regime_relative_cumulative_observations,
                episode_length=None,
                random_episode_start=False,
                action_smoothing=args.action_smoothing,
                no_trade_band=args.no_trade_band,
                max_executed_rebalances=args.max_executed_rebalances,
                rebalance_cooldown_steps=args.rebalance_cooldown_steps,
                early_rebalance_risk_penalty=args.early_rebalance_risk_penalty,
                early_rebalance_risk_turnover_cap=args.early_rebalance_risk_turnover_cap,
                early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold=args.early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold,
                early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold=args.early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold,
                early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight=args.early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight,
                early_rebalance_risk_turnover_cap_target_cash_min_threshold=args.early_rebalance_risk_turnover_cap_target_cash_min_threshold,
                early_rebalance_risk_turnover_cap_target_cash_max_threshold=args.early_rebalance_risk_turnover_cap_target_cash_max_threshold,
                early_rebalance_risk_turnover_cap_target_trend_min_threshold=args.early_rebalance_risk_turnover_cap_target_trend_min_threshold,
                early_rebalance_risk_turnover_cap_target_trend_max_threshold=args.early_rebalance_risk_turnover_cap_target_trend_max_threshold,
                early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold=args.early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold,
                early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold=args.early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold,
                early_rebalance_risk_turnover_cap_delta_cash_min_threshold=args.early_rebalance_risk_turnover_cap_delta_cash_min_threshold,
                early_rebalance_risk_turnover_cap_delta_cash_max_threshold=args.early_rebalance_risk_turnover_cap_delta_cash_max_threshold,
                early_rebalance_risk_turnover_cap_delta_trend_min_threshold=args.early_rebalance_risk_turnover_cap_delta_trend_min_threshold,
                early_rebalance_risk_turnover_cap_delta_trend_max_threshold=args.early_rebalance_risk_turnover_cap_delta_trend_max_threshold,
                early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold=args.early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold,
                early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold=args.early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold,
                early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold=args.early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold,
                early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold=args.early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold,
                early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio=args.early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio,
                early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio=args.early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio,
                early_rebalance_risk_turnover_cap_allow_nonincreasing_risk_symbol=args.early_rebalance_risk_turnover_cap_allow_nonincreasing_risk_symbol,
                early_rebalance_risk_turnover_cap_use_penalty_state_filters=args.early_rebalance_risk_turnover_cap_use_penalty_state_filters,
                early_rebalance_risk_turnover_cap_after=args.early_rebalance_risk_turnover_cap_after,
                early_rebalance_risk_turnover_cap_before=args.early_rebalance_risk_turnover_cap_before,
                early_rebalance_risk_turnover_cap_max_applications=args.early_rebalance_risk_turnover_cap_max_applications,
                early_rebalance_risk_turnover_cap_secondary_cap=args.early_rebalance_risk_turnover_cap_secondary_cap,
                early_rebalance_risk_turnover_cap_secondary_after_applications=args.early_rebalance_risk_turnover_cap_secondary_after_applications,
                early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold=args.early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold,
                early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio=args.early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio,
                early_rebalance_risk_shallow_drawdown_turnover_cap=args.early_rebalance_risk_shallow_drawdown_turnover_cap,
                early_rebalance_risk_shallow_drawdown_turnover_cap_after=args.early_rebalance_risk_shallow_drawdown_turnover_cap_after,
                early_rebalance_risk_shallow_drawdown_turnover_cap_before=args.early_rebalance_risk_shallow_drawdown_turnover_cap_before,
                early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold=args.early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold,
                early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold=args.early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold,
                early_rebalance_risk_mean_reversion_turnover_cap=args.early_rebalance_risk_mean_reversion_turnover_cap,
                early_rebalance_risk_mean_reversion_action_smoothing=args.early_rebalance_risk_mean_reversion_action_smoothing,
                early_rebalance_risk_mean_reversion_turnover_cap_after=args.early_rebalance_risk_mean_reversion_turnover_cap_after,
                early_rebalance_risk_mean_reversion_turnover_cap_before=args.early_rebalance_risk_mean_reversion_turnover_cap_before,
                early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold=args.early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold,
                early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold=args.early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold,
                early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold=args.early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold,
                early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold=args.early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold,
                early_rebalance_risk_trend_turnover_cap=args.early_rebalance_risk_trend_turnover_cap,
                early_rebalance_risk_trend_turnover_cap_after=args.early_rebalance_risk_trend_turnover_cap_after,
                early_rebalance_risk_trend_turnover_cap_before=args.early_rebalance_risk_trend_turnover_cap_before,
                early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold=args.early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold,
                early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold=args.early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold,
                early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold=args.early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold,
                early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold=args.early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold,
                early_rebalance_risk_deep_drawdown_turnover_cap=args.early_rebalance_risk_deep_drawdown_turnover_cap,
                early_rebalance_risk_deep_drawdown_turnover_cap_after=args.early_rebalance_risk_deep_drawdown_turnover_cap_after,
                early_rebalance_risk_deep_drawdown_turnover_cap_before=args.early_rebalance_risk_deep_drawdown_turnover_cap_before,
                early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold=args.early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold,
                early_rebalance_risk_repeat_turnover_cap=args.early_rebalance_risk_repeat_turnover_cap,
                early_rebalance_risk_repeat_action_smoothing=args.early_rebalance_risk_repeat_action_smoothing,
                early_rebalance_risk_repeat_turnover_cap_after=args.early_rebalance_risk_repeat_turnover_cap_after,
                early_rebalance_risk_repeat_turnover_cap_before=args.early_rebalance_risk_repeat_turnover_cap_before,
                early_rebalance_risk_repeat_symbol=args.early_rebalance_risk_repeat_symbol,
                early_rebalance_risk_repeat_previous_cash_reduction_min=args.early_rebalance_risk_repeat_previous_cash_reduction_min,
                early_rebalance_risk_repeat_previous_symbol_increase_min=args.early_rebalance_risk_repeat_previous_symbol_increase_min,
                early_rebalance_risk_repeat_unrecovered_turnover_cap=args.early_rebalance_risk_repeat_unrecovered_turnover_cap,
                early_rebalance_risk_repeat_unrecovered_turnover_cap_after=args.early_rebalance_risk_repeat_unrecovered_turnover_cap_after,
                early_rebalance_risk_repeat_unrecovered_turnover_cap_before=args.early_rebalance_risk_repeat_unrecovered_turnover_cap_before,
                early_rebalance_risk_repeat_unrecovered_symbol=args.early_rebalance_risk_repeat_unrecovered_symbol,
                early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min=args.early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min,
                early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min=args.early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min,
                early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery=args.early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery,
                early_rebalance_risk_cumulative_turnover_cap=args.early_rebalance_risk_cumulative_turnover_cap,
                early_rebalance_risk_cumulative_turnover_cap_after=args.early_rebalance_risk_cumulative_turnover_cap_after,
                early_rebalance_risk_cumulative_turnover_cap_before=args.early_rebalance_risk_cumulative_turnover_cap_before,
                early_rebalance_risk_cumulative_symbol=args.early_rebalance_risk_cumulative_symbol,
                early_rebalance_risk_cumulative_cash_reduction_budget=args.early_rebalance_risk_cumulative_cash_reduction_budget,
                early_rebalance_risk_cumulative_symbol_increase_budget=args.early_rebalance_risk_cumulative_symbol_increase_budget,
                early_rebalance_risk_penalty_after=args.early_rebalance_risk_penalty_after,
                early_rebalance_risk_penalty_before=args.early_rebalance_risk_penalty_before,
                early_rebalance_risk_penalty_cash_max_threshold=args.early_rebalance_risk_penalty_cash_max_threshold,
                early_rebalance_risk_penalty_symbol=args.early_rebalance_risk_penalty_symbol,
                early_rebalance_risk_penalty_symbol_min_weight=args.early_rebalance_risk_penalty_symbol_min_weight,
                early_rebalance_risk_penalty_symbol_max_weight=args.early_rebalance_risk_penalty_symbol_max_weight,
                early_rebalance_risk_penalty_benchmark_drawdown_min_threshold=args.early_rebalance_risk_penalty_benchmark_drawdown_min_threshold,
                early_rebalance_risk_penalty_benchmark_drawdown_max_threshold=args.early_rebalance_risk_penalty_benchmark_drawdown_max_threshold,
                early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio=args.early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio,
                early_benchmark_euphoria_penalty=args.early_benchmark_euphoria_penalty,
                early_benchmark_euphoria_turnover_cap=args.early_benchmark_euphoria_turnover_cap,
                early_benchmark_euphoria_before=args.early_benchmark_euphoria_before,
                early_benchmark_euphoria_benchmark_drawdown_min_threshold=args.early_benchmark_euphoria_benchmark_drawdown_min_threshold,
                early_benchmark_euphoria_symbol=args.early_benchmark_euphoria_symbol,
                late_rebalance_penalty=args.late_rebalance_penalty,
                late_rebalance_penalty_after=args.late_rebalance_penalty_after,
                late_rebalance_gate_after=args.late_rebalance_gate_after,
                late_rebalance_gate_cash_threshold=args.late_rebalance_gate_cash_threshold,
                late_rebalance_gate_target_cash_min_threshold=args.late_rebalance_gate_target_cash_min_threshold,
                late_rebalance_gate_symbol=args.late_rebalance_gate_symbol,
                late_rebalance_gate_symbol_max_weight=args.late_rebalance_gate_symbol_max_weight,
                late_rebalance_gate_cash_reduction_max=args.late_rebalance_gate_cash_reduction_max,
                late_rebalance_gate_symbol_increase_max=args.late_rebalance_gate_symbol_increase_max,
                late_defensive_posture_penalty=args.late_defensive_posture_penalty,
                late_defensive_posture_penalty_after=args.late_defensive_posture_penalty_after,
                late_defensive_posture_penalty_cash_min_threshold=args.late_defensive_posture_penalty_cash_min_threshold,
                late_defensive_posture_penalty_symbol=args.late_defensive_posture_penalty_symbol,
                late_defensive_posture_penalty_symbol_max_weight=args.late_defensive_posture_penalty_symbol_max_weight,
                late_trend_mean_reversion_conflict_penalty=args.late_trend_mean_reversion_conflict_penalty,
                late_trend_mean_reversion_conflict_penalty_after=args.late_trend_mean_reversion_conflict_penalty_after,
                late_trend_mean_reversion_conflict_trend_symbol=args.late_trend_mean_reversion_conflict_trend_symbol,
                late_trend_mean_reversion_conflict_trend_min_weight=args.late_trend_mean_reversion_conflict_trend_min_weight,
                late_trend_mean_reversion_conflict_mean_reversion_symbol=args.late_trend_mean_reversion_conflict_mean_reversion_symbol,
                late_trend_mean_reversion_conflict_mean_reversion_min_weight=args.late_trend_mean_reversion_conflict_mean_reversion_min_weight,
                state_trend_preservation_symbol=args.state_trend_preservation_symbol,
                state_trend_preservation_cash_max_threshold=args.state_trend_preservation_cash_max_threshold,
                state_trend_preservation_symbol_min_weight=args.state_trend_preservation_symbol_min_weight,
                state_trend_preservation_max_symbol_reduction=args.state_trend_preservation_max_symbol_reduction,
                cash_weight_penalty=args.cash_weight_penalty,
                cash_target_weight=args.cash_target_weight,
            )
        )

    train_env = DummyVecEnv([build_train_env])
    validation_env = DummyVecEnv([build_validation_env])
    test_env = DummyVecEnv([build_test_env])
    if args.normalize_observations or args.normalize_rewards:
        train_env = VecNormalize(
            train_env,
            norm_obs=args.normalize_observations,
            norm_reward=args.normalize_rewards,
            clip_obs=10.0,
        )
        validation_env = VecNormalize(
            validation_env,
            norm_obs=args.normalize_observations,
            norm_reward=args.normalize_rewards,
            clip_obs=10.0,
            training=False,
        )
        test_env = VecNormalize(
            test_env,
            norm_obs=args.normalize_observations,
            norm_reward=args.normalize_rewards,
            clip_obs=10.0,
            training=False,
        )

    eval_callback = EvalCallback(
        validation_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        eval_freq=max(args.eval_freq, 1),
        n_eval_episodes=max(args.n_eval_episodes, 1),
        deterministic=True,
    )
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=tensorboard_log,
    )
    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback, progress_bar=False)

    final_model_path = output_dir / "ppo_final_model"
    model.save(final_model_path)
    if args.normalize_observations or args.normalize_rewards:
        train_env.save(str(output_dir / "vecnormalize.pkl"))
        sync_envs_normalization(train_env, validation_env)
        sync_envs_normalization(train_env, test_env)

    validation_rollout = _rollout_policy_records(model, validation_env, deterministic=True)
    test_rollout = _rollout_policy_records(model, test_env, deterministic=True)
    validation_policy_summary = _summarize_policy_rollout_records(
        validation_rollout,
        cash_symbol=policy_config.cash_symbol,
    )
    test_policy_summary = _summarize_policy_rollout_records(
        test_rollout,
        cash_symbol=policy_config.cash_symbol,
    )
    validation_rebalance_impact_records = _compute_rebalance_impact_records(
        validation_rollout,
        validation_returns,
        validation_benchmark,
        policy_config,
    )
    test_rebalance_impact_records = _compute_rebalance_impact_records(
        test_rollout,
        test_returns,
        test_benchmark,
        policy_config,
    )
    validation_rebalance_impact_summary = _summarize_rebalance_impact_records(validation_rebalance_impact_records)
    test_rebalance_impact_summary = _summarize_rebalance_impact_records(test_rebalance_impact_records)
    validation_static_hold_rollout = _simulate_static_hold_rollout_records(
        validation_rollout,
        validation_returns,
        validation_benchmark,
        policy_config,
        cash_weight_penalty=args.cash_weight_penalty,
        cash_target_weight=args.cash_target_weight,
    )
    test_static_hold_rollout = _simulate_static_hold_rollout_records(
        test_rollout,
        test_returns,
        test_benchmark,
        policy_config,
        cash_weight_penalty=args.cash_weight_penalty,
        cash_target_weight=args.cash_target_weight,
    )
    validation_static_hold_summary = _summarize_policy_rollout_records(
        validation_static_hold_rollout,
        cash_symbol=policy_config.cash_symbol,
    )
    test_static_hold_summary = _summarize_policy_rollout_records(
        test_static_hold_rollout,
        cash_symbol=policy_config.cash_symbol,
    )
    trade_budget_falsification: dict[str, dict[str, object]] = {}
    for trade_budget in args.falsification_trade_budget:
        label = _build_trade_budget_label(trade_budget)
        validation_trade_budget_rollout = _simulate_trade_budget_rollout_records(
            validation_rollout,
            validation_returns,
            validation_benchmark,
            policy_config,
            max_rebalances=trade_budget,
            cash_weight_penalty=args.cash_weight_penalty,
            cash_target_weight=args.cash_target_weight,
        )
        test_trade_budget_rollout = _simulate_trade_budget_rollout_records(
            test_rollout,
            test_returns,
            test_benchmark,
            policy_config,
            max_rebalances=trade_budget,
            cash_weight_penalty=args.cash_weight_penalty,
            cash_target_weight=args.cash_target_weight,
        )
        validation_trade_budget_summary = _summarize_policy_rollout_records(
            validation_trade_budget_rollout,
            cash_symbol=policy_config.cash_symbol,
        )
        test_trade_budget_summary = _summarize_policy_rollout_records(
            test_trade_budget_rollout,
            cash_symbol=policy_config.cash_symbol,
        )
        _save_rollout_artifacts(output_dir, "validation", label, validation_trade_budget_rollout, validation_trade_budget_summary)
        _save_rollout_artifacts(output_dir, "test", label, test_trade_budget_rollout, test_trade_budget_summary)
        trade_budget_comparison = {
            "validation": _build_named_metric_comparison(
                validation_policy_summary,
                validation_trade_budget_summary,
                left_label="policy",
                right_label=label,
            ),
            "test": _build_named_metric_comparison(
                test_policy_summary,
                test_trade_budget_summary,
                left_label="policy",
                right_label=label,
            ),
        }
        trade_budget_falsification[label] = {
            "budget": trade_budget,
            "validation_summary": _serialize_series(validation_trade_budget_summary),
            "test_summary": _serialize_series(test_trade_budget_summary),
            "comparison": trade_budget_comparison,
        }
        (output_dir / f"policy_vs_{label}_comparison.json").write_text(
            json.dumps(trade_budget_comparison, indent=2),
            encoding="utf-8",
        )

    optimizer_validation_result = run_rolling_backtest(
        validation_returns,
        lookback=args.lookback,
        rebalance_frequency=1,
        config=optimizer_config,
        benchmark_returns=validation_benchmark,
    )
    optimizer_test_result = run_rolling_backtest(
        test_returns,
        lookback=args.lookback,
        rebalance_frequency=1,
        config=optimizer_config,
        benchmark_returns=test_benchmark,
    )
    _save_rollout_artifacts(output_dir, "validation", "policy", validation_rollout, validation_policy_summary)
    _save_rebalance_impact_artifacts(output_dir, "validation", validation_rebalance_impact_records, validation_rebalance_impact_summary)
    _save_rollout_artifacts(output_dir, "validation", "static_hold", validation_static_hold_rollout, validation_static_hold_summary)
    _save_rollout_artifacts(output_dir, "test", "policy", test_rollout, test_policy_summary)
    _save_rebalance_impact_artifacts(output_dir, "test", test_rebalance_impact_records, test_rebalance_impact_summary)
    _save_rollout_artifacts(output_dir, "test", "static_hold", test_static_hold_rollout, test_static_hold_summary)
    _save_backtest_artifacts(output_dir, "validation", optimizer_validation_result)
    _save_backtest_artifacts(output_dir, "test", optimizer_test_result)

    validation_mean_reward, validation_std_reward = evaluate_policy(
        model,
        validation_env,
        n_eval_episodes=max(args.n_eval_episodes, 1),
        deterministic=True,
    )
    test_mean_reward, test_std_reward = evaluate_policy(
        model,
        test_env,
        n_eval_episodes=max(args.n_eval_episodes, 1),
        deterministic=True,
    )
    metrics = {
        "mean_validation_reward": float(validation_mean_reward),
        "std_validation_reward": float(validation_std_reward),
        "mean_test_reward": float(test_mean_reward),
        "std_test_reward": float(test_std_reward),
        "train_rows": int(len(train_returns)),
        "validation_rows": int(len(validation_returns)),
        "test_rows": int(len(test_returns)),
        "train_columns": list(train_returns.columns),
        "benchmark_column": args.benchmark_column,
        "total_timesteps": int(args.total_timesteps),
        "lookback": int(args.lookback),
        "episode_length": int(args.episode_length),
        "benchmark_relative_observations": bool(args.benchmark_relative_observations),
        "benchmark_regime_observations": bool(
            benchmark_regime_summary_observations or benchmark_regime_relative_cumulative_observations
        ),
        "benchmark_regime_summary_observations": bool(benchmark_regime_summary_observations),
        "benchmark_regime_relative_cumulative_observations": bool(
            benchmark_regime_relative_cumulative_observations
        ),
        "split_method": split.method,
        "split_score": split.score,
        "action_smoothing": float(args.action_smoothing),
        "no_trade_band": float(args.no_trade_band),
        "max_executed_rebalances": args.max_executed_rebalances,
        "rebalance_cooldown_steps": args.rebalance_cooldown_steps,
        "late_rebalance_penalty": float(args.late_rebalance_penalty),
        "late_rebalance_penalty_after": args.late_rebalance_penalty_after,
        "late_rebalance_gate_after": args.late_rebalance_gate_after,
        "late_rebalance_gate_cash_threshold": args.late_rebalance_gate_cash_threshold,
        "late_rebalance_gate_target_cash_min_threshold": args.late_rebalance_gate_target_cash_min_threshold,
        "early_rebalance_risk_turnover_cap_target_cash_min_threshold": args.early_rebalance_risk_turnover_cap_target_cash_min_threshold,
        "early_rebalance_risk_turnover_cap_target_cash_max_threshold": args.early_rebalance_risk_turnover_cap_target_cash_max_threshold,
        "early_rebalance_risk_turnover_cap_target_trend_min_threshold": args.early_rebalance_risk_turnover_cap_target_trend_min_threshold,
        "early_rebalance_risk_turnover_cap_target_trend_max_threshold": args.early_rebalance_risk_turnover_cap_target_trend_max_threshold,
        "early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold": args.early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold,
        "early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold": args.early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold,
        "early_rebalance_risk_turnover_cap_delta_cash_min_threshold": args.early_rebalance_risk_turnover_cap_delta_cash_min_threshold,
        "early_rebalance_risk_turnover_cap_delta_cash_max_threshold": args.early_rebalance_risk_turnover_cap_delta_cash_max_threshold,
        "early_rebalance_risk_turnover_cap_delta_trend_min_threshold": args.early_rebalance_risk_turnover_cap_delta_trend_min_threshold,
        "early_rebalance_risk_turnover_cap_delta_trend_max_threshold": args.early_rebalance_risk_turnover_cap_delta_trend_max_threshold,
        "early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold": args.early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold,
        "early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold": args.early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold,
        "early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold": args.early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold,
        "early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold": args.early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold,
        "late_rebalance_gate_symbol": args.late_rebalance_gate_symbol,
        "late_rebalance_gate_symbol_max_weight": args.late_rebalance_gate_symbol_max_weight,
        "late_rebalance_gate_cash_reduction_max": args.late_rebalance_gate_cash_reduction_max,
        "late_rebalance_gate_symbol_increase_max": args.late_rebalance_gate_symbol_increase_max,
        "state_trend_preservation_symbol": args.state_trend_preservation_symbol,
        "state_trend_preservation_cash_max_threshold": args.state_trend_preservation_cash_max_threshold,
        "state_trend_preservation_symbol_min_weight": args.state_trend_preservation_symbol_min_weight,
        "state_trend_preservation_max_symbol_reduction": args.state_trend_preservation_max_symbol_reduction,
        "early_rebalance_risk_repeat_turnover_cap": args.early_rebalance_risk_repeat_turnover_cap,
        "early_rebalance_risk_repeat_turnover_cap_after": args.early_rebalance_risk_repeat_turnover_cap_after,
        "early_rebalance_risk_repeat_turnover_cap_before": args.early_rebalance_risk_repeat_turnover_cap_before,
        "early_rebalance_risk_repeat_symbol": args.early_rebalance_risk_repeat_symbol,
        "early_rebalance_risk_repeat_previous_cash_reduction_min": args.early_rebalance_risk_repeat_previous_cash_reduction_min,
        "early_rebalance_risk_repeat_previous_symbol_increase_min": args.early_rebalance_risk_repeat_previous_symbol_increase_min,
        "early_rebalance_risk_repeat_unrecovered_turnover_cap": args.early_rebalance_risk_repeat_unrecovered_turnover_cap,
        "early_rebalance_risk_repeat_unrecovered_turnover_cap_after": args.early_rebalance_risk_repeat_unrecovered_turnover_cap_after,
        "early_rebalance_risk_repeat_unrecovered_turnover_cap_before": args.early_rebalance_risk_repeat_unrecovered_turnover_cap_before,
        "early_rebalance_risk_repeat_unrecovered_symbol": args.early_rebalance_risk_repeat_unrecovered_symbol,
        "early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min": args.early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min,
        "early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min": args.early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min,
        "early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery": args.early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery,
        "early_rebalance_risk_cumulative_turnover_cap": args.early_rebalance_risk_cumulative_turnover_cap,
        "early_rebalance_risk_cumulative_turnover_cap_after": args.early_rebalance_risk_cumulative_turnover_cap_after,
        "early_rebalance_risk_cumulative_turnover_cap_before": args.early_rebalance_risk_cumulative_turnover_cap_before,
        "early_rebalance_risk_cumulative_symbol": args.early_rebalance_risk_cumulative_symbol,
        "early_rebalance_risk_cumulative_cash_reduction_budget": args.early_rebalance_risk_cumulative_cash_reduction_budget,
        "early_rebalance_risk_cumulative_symbol_increase_budget": args.early_rebalance_risk_cumulative_symbol_increase_budget,
        "cash_weight_penalty": float(args.cash_weight_penalty),
        "cash_target_weight": args.cash_target_weight,
        "optimizer_min_weight_overrides": optimizer_config.min_weight_overrides,
        "optimizer_max_weight_overrides": optimizer_config.max_weight_overrides,
        "policy_min_weight_overrides": policy_config.min_weight_overrides,
        "policy_max_weight_overrides": policy_config.max_weight_overrides,
        "validation_policy_summary": _serialize_series(validation_policy_summary),
        "validation_rebalance_impact_summary": _serialize_series(validation_rebalance_impact_summary),
        "validation_static_hold_summary": _serialize_series(validation_static_hold_summary),
        "validation_optimizer_summary": _serialize_series(optimizer_validation_result.summary),
        "test_policy_summary": _serialize_series(test_policy_summary),
        "test_rebalance_impact_summary": _serialize_series(test_rebalance_impact_summary),
        "test_static_hold_summary": _serialize_series(test_static_hold_summary),
        "test_optimizer_summary": _serialize_series(optimizer_test_result.summary),
        "comparison": {
            "validation": _build_common_metric_comparison(validation_policy_summary, optimizer_validation_result.summary),
            "test": _build_common_metric_comparison(test_policy_summary, optimizer_test_result.summary),
        },
        "falsification_comparison": {
            "validation": _build_named_metric_comparison(validation_policy_summary, validation_static_hold_summary, left_label="policy", right_label="static_hold"),
            "test": _build_named_metric_comparison(test_policy_summary, test_static_hold_summary, left_label="policy", right_label="static_hold"),
        },
        "trade_budget_falsification": trade_budget_falsification,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "policy_vs_optimizer_comparison.json").write_text(
        json.dumps(metrics["comparison"], indent=2),
        encoding="utf-8",
    )
    (output_dir / "policy_vs_static_hold_comparison.json").write_text(
        json.dumps(metrics["falsification_comparison"], indent=2),
        encoding="utf-8",
    )
    (output_dir / "policy_vs_trade_budget_comparisons.json").write_text(
        json.dumps({label: data["comparison"] for label, data in trade_budget_falsification.items()}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "rebalance_impact_summary.json").write_text(
        json.dumps(
            {
                "validation": metrics["validation_rebalance_impact_summary"],
                "test": metrics["test_rebalance_impact_summary"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved PPO baseline artifacts to {output_dir}")
    print(f"Validation reward: {validation_mean_reward:.6f} +/- {validation_std_reward:.6f}")
    print(f"Test reward: {test_mean_reward:.6f} +/- {test_std_reward:.6f}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.split_method = _normalize_split_method_alias(args.split_method)
    if args.walk_forward_folds <= 0:
        raise ValueError("walk_forward_folds must be positive.")
    if not args.falsification_trade_budget:
        args.falsification_trade_budget = [1, 2, 3]
    args.falsification_trade_budget = _normalize_trade_budgets(args.falsification_trade_budget)
    dependencies = _load_rl_dependencies()

    returns = load_returns_csv(args.returns_csv, date_column=args.date_column)
    returns = _filter_returns_by_date(returns, start=args.start, end=args.end)

    benchmark_returns = None
    if args.benchmark_column:
        if args.benchmark_column not in returns.columns:
            raise ValueError(f"Benchmark column '{args.benchmark_column}' was not found in the returns CSV.")
        benchmark_returns = returns[args.benchmark_column].copy()
        if args.exclude_benchmark_from_universe:
            returns = returns.drop(columns=[args.benchmark_column])

    if args.split_method == "regime-balanced":
        suggested_split = suggest_regime_balanced_split(
            returns,
            benchmark_returns=benchmark_returns,
            lookback=args.lookback,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
        )
    else:
        suggested_split = chronological_train_validation_test_split(
            returns,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            lookback=args.lookback,
        )

    walk_forward_splits = generate_walk_forward_splits(
        returns,
        benchmark_returns=benchmark_returns,
        lookback=args.lookback,
        validation_rows=suggested_split.validation.rows,
        test_rows=suggested_split.test.rows,
        step_rows=args.walk_forward_step_rows,
        max_splits=args.walk_forward_folds,
    )
    min_weight_overrides, max_weight_overrides = _parse_weight_bound_overrides(args.weight_bound)
    policy_min_weight_overrides, policy_max_weight_overrides = _parse_weight_bound_overrides(args.policy_weight_bound)
    config = WealthFirstConfig(
        loss_penalty=args.loss_penalty,
        gain_reward=args.gain_reward,
        gain_power=args.gain_power,
        loss_power=args.loss_power,
        benchmark_gain_reward=args.benchmark_gain_reward,
        benchmark_loss_penalty=args.benchmark_loss_penalty,
        turnover_penalty=args.turnover_penalty,
        weight_reg=args.weight_reg,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        include_cash=True,
        min_weight_overrides=min_weight_overrides,
        max_weight_overrides=max_weight_overrides,
    )
    merged_policy_min_weight_overrides, merged_policy_max_weight_overrides = _merge_weight_bound_overrides(
        min_weight_overrides,
        max_weight_overrides,
        policy_min_weight_overrides,
        policy_max_weight_overrides,
    )
    policy_config = replace(
        config,
        min_weight_overrides=merged_policy_min_weight_overrides,
        max_weight_overrides=merged_policy_max_weight_overrides,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_metadata = {
        "primary_split": _split_to_metadata(suggested_split),
        "walk_forward_folds": [
            _split_to_metadata(split)
            for split in walk_forward_splits
        ],
        "walk_forward_fold_count": len(walk_forward_splits),
    }
    (output_dir / "split_windows.json").write_text(json.dumps(split_metadata, indent=2), encoding="utf-8")
    if args.split_only:
        print(f"Saved split metadata to {output_dir / 'split_windows.json'}")
        return 0

    fold_metrics: list[dict[str, object]] = []
    multiple_folds = len(walk_forward_splits) > 1
    for fold_index, split in enumerate(walk_forward_splits, start=1):
        fold_output_dir = output_dir / f"fold_{fold_index:02d}" if multiple_folds else output_dir
        metrics = _run_single_fold(
            split,
            returns,
            benchmark_returns,
            args,
            config,
            policy_config,
            fold_output_dir,
            dependencies,
        )
        metrics["fold_index"] = fold_index
        metrics["split"] = _split_to_metadata(split)
        fold_metrics.append(metrics)

    if multiple_folds:
        aggregated_comparison = {
            "validation": _aggregate_comparison_maps([metric["comparison"]["validation"] for metric in fold_metrics]),
            "test": _aggregate_comparison_maps([metric["comparison"]["test"] for metric in fold_metrics]),
        }
        aggregated_falsification_comparison = {
            "validation": _aggregate_named_comparison_maps(
                [metric["falsification_comparison"]["validation"] for metric in fold_metrics],
                left_field="policy",
                right_field="static_hold",
            ),
            "test": _aggregate_named_comparison_maps(
                [metric["falsification_comparison"]["test"] for metric in fold_metrics],
                left_field="policy",
                right_field="static_hold",
            ),
        }
        aggregated_trade_budget_falsification: dict[str, dict[str, object]] = {}
        aggregated_trade_budget_comparisons: dict[str, dict[str, dict[str, float | None]]] = {}
        for trade_budget in args.falsification_trade_budget:
            label = _build_trade_budget_label(trade_budget)
            aggregated_trade_budget_comparison = {
                "validation": _aggregate_named_comparison_maps(
                    [metric["trade_budget_falsification"][label]["comparison"]["validation"] for metric in fold_metrics],
                    left_field="policy",
                    right_field=label,
                ),
                "test": _aggregate_named_comparison_maps(
                    [metric["trade_budget_falsification"][label]["comparison"]["test"] for metric in fold_metrics],
                    left_field="policy",
                    right_field=label,
                ),
            }
            aggregated_trade_budget_comparisons[label] = aggregated_trade_budget_comparison
            aggregated_trade_budget_falsification[label] = {
                "budget": trade_budget,
                "mean_validation_summary": _aggregate_metric_maps(
                    [metric["trade_budget_falsification"][label]["validation_summary"] for metric in fold_metrics]
                ),
                "mean_test_summary": _aggregate_metric_maps(
                    [metric["trade_budget_falsification"][label]["test_summary"] for metric in fold_metrics]
                ),
                "mean_comparison": aggregated_trade_budget_comparison,
            }
        aggregate_metrics = {
            "walk_forward_fold_count": len(fold_metrics),
            "mean_validation_reward": float(pd.Series([metric["mean_validation_reward"] for metric in fold_metrics]).mean()),
            "std_validation_reward": float(pd.Series([metric["mean_validation_reward"] for metric in fold_metrics]).std(ddof=0)),
            "mean_test_reward": float(pd.Series([metric["mean_test_reward"] for metric in fold_metrics]).mean()),
            "std_test_reward": float(pd.Series([metric["mean_test_reward"] for metric in fold_metrics]).std(ddof=0)),
            "benchmark_relative_observations": bool(args.benchmark_relative_observations),
            "benchmark_regime_observations": bool(
                args.benchmark_regime_observations
                or args.benchmark_regime_summary_only_observations
                or args.benchmark_regime_relative_cumulative_only_observations
            ),
            "benchmark_regime_summary_observations": bool(
                args.benchmark_regime_observations or args.benchmark_regime_summary_only_observations
            ),
            "benchmark_regime_relative_cumulative_observations": bool(
                args.benchmark_regime_observations or args.benchmark_regime_relative_cumulative_only_observations
            ),
            "action_smoothing": float(args.action_smoothing),
            "no_trade_band": float(args.no_trade_band),
            "max_executed_rebalances": args.max_executed_rebalances,
            "rebalance_cooldown_steps": args.rebalance_cooldown_steps,
            "late_rebalance_penalty": float(args.late_rebalance_penalty),
            "late_rebalance_penalty_after": args.late_rebalance_penalty_after,
            "late_rebalance_gate_after": args.late_rebalance_gate_after,
            "late_rebalance_gate_cash_threshold": args.late_rebalance_gate_cash_threshold,
            "late_rebalance_gate_target_cash_min_threshold": args.late_rebalance_gate_target_cash_min_threshold,
            "early_rebalance_risk_turnover_cap_target_cash_min_threshold": args.early_rebalance_risk_turnover_cap_target_cash_min_threshold,
            "early_rebalance_risk_turnover_cap_target_cash_max_threshold": args.early_rebalance_risk_turnover_cap_target_cash_max_threshold,
            "early_rebalance_risk_turnover_cap_target_trend_min_threshold": args.early_rebalance_risk_turnover_cap_target_trend_min_threshold,
            "early_rebalance_risk_turnover_cap_target_trend_max_threshold": args.early_rebalance_risk_turnover_cap_target_trend_max_threshold,
            "early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold": args.early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold,
            "early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold": args.early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold,
            "early_rebalance_risk_turnover_cap_delta_cash_min_threshold": args.early_rebalance_risk_turnover_cap_delta_cash_min_threshold,
            "early_rebalance_risk_turnover_cap_delta_cash_max_threshold": args.early_rebalance_risk_turnover_cap_delta_cash_max_threshold,
            "early_rebalance_risk_turnover_cap_delta_trend_min_threshold": args.early_rebalance_risk_turnover_cap_delta_trend_min_threshold,
            "early_rebalance_risk_turnover_cap_delta_trend_max_threshold": args.early_rebalance_risk_turnover_cap_delta_trend_max_threshold,
            "early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold": args.early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold,
            "early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold": args.early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold,
            "early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold": args.early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold,
            "early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold": args.early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold,
            "late_rebalance_gate_symbol": args.late_rebalance_gate_symbol,
            "late_rebalance_gate_symbol_max_weight": args.late_rebalance_gate_symbol_max_weight,
            "late_rebalance_gate_cash_reduction_max": args.late_rebalance_gate_cash_reduction_max,
            "late_rebalance_gate_symbol_increase_max": args.late_rebalance_gate_symbol_increase_max,
            "cash_weight_penalty": float(args.cash_weight_penalty),
            "cash_target_weight": args.cash_target_weight,
            "policy_min_weight_overrides": policy_config.min_weight_overrides,
            "policy_max_weight_overrides": policy_config.max_weight_overrides,
            "mean_validation_policy_summary": _aggregate_metric_maps([metric["validation_policy_summary"] for metric in fold_metrics]),
            "mean_validation_rebalance_impact_summary": _aggregate_metric_maps([metric["validation_rebalance_impact_summary"] for metric in fold_metrics]),
            "mean_validation_static_hold_summary": _aggregate_metric_maps([metric["validation_static_hold_summary"] for metric in fold_metrics]),
            "mean_validation_optimizer_summary": _aggregate_metric_maps([metric["validation_optimizer_summary"] for metric in fold_metrics]),
            "mean_test_policy_summary": _aggregate_metric_maps([metric["test_policy_summary"] for metric in fold_metrics]),
            "mean_test_rebalance_impact_summary": _aggregate_metric_maps([metric["test_rebalance_impact_summary"] for metric in fold_metrics]),
            "mean_test_static_hold_summary": _aggregate_metric_maps([metric["test_static_hold_summary"] for metric in fold_metrics]),
            "mean_test_optimizer_summary": _aggregate_metric_maps([metric["test_optimizer_summary"] for metric in fold_metrics]),
            "mean_comparison": aggregated_comparison,
            "mean_falsification_comparison": aggregated_falsification_comparison,
            "trade_budget_falsification": aggregated_trade_budget_falsification,
            "fold_metrics": fold_metrics,
            "split_method": suggested_split.method,
        }
        (output_dir / "metrics.json").write_text(json.dumps(aggregate_metrics, indent=2), encoding="utf-8")
        (output_dir / "policy_vs_optimizer_comparison.json").write_text(json.dumps(aggregated_comparison, indent=2), encoding="utf-8")
        (output_dir / "policy_vs_static_hold_comparison.json").write_text(json.dumps(aggregated_falsification_comparison, indent=2), encoding="utf-8")
        (output_dir / "policy_vs_trade_budget_comparisons.json").write_text(
            json.dumps(aggregated_trade_budget_comparisons, indent=2),
            encoding="utf-8",
        )
        (output_dir / "rebalance_impact_summary.json").write_text(
            json.dumps(
                {
                    "validation": aggregate_metrics["mean_validation_rebalance_impact_summary"],
                    "test": aggregate_metrics["mean_test_rebalance_impact_summary"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved walk-forward PPO artifacts to {output_dir}")
        print(f"Mean validation reward across folds: {aggregate_metrics['mean_validation_reward']:.6f}")
        print(f"Mean test reward across folds: {aggregate_metrics['mean_test_reward']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())