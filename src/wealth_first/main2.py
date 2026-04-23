from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from wealth_first.backtest import run_rolling_backtest, summarize_performance
from wealth_first.data import load_returns_csv
from wealth_first.data_splits import SuggestedTimeSeriesSplit, chronological_train_validation_test_split, generate_walk_forward_splits, suggest_regime_balanced_split
from wealth_first.optimizer import WealthFirstConfig

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised in environments without RL extras.
    torch = None
    nn = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TORCH_AVAILABLE = torch is not None and nn is not None


def _main2_json_ready(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _main2_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_main2_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


def _append_main2_progress_event(progress_log_path: Path | None, event_type: str, **fields: Any) -> None:
    if progress_log_path is None:
        return

    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event_type": event_type,
        "timestamp": datetime.now(UTC).isoformat(),
        **fields,
    }
    with progress_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_main2_json_ready(payload), sort_keys=True) + "\n")


@dataclass(frozen=True)
class Main2Config:
    lookback: int = 5
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0
    action_smoothing: float = 0.5
    no_trade_band: float = 0.02
    initial_spy_weight: float = 1.0
    min_spy_weight: float = 0.65
    max_spy_weight: float = 1.0
    trend_floor_min_spy_weight: float | None = 0.9
    trend_floor_lookback: int = 63
    trend_floor_return_threshold: float = 0.08
    trend_floor_drawdown_threshold: float = -0.10
    participation_floor_min_spy_weight: float | None = 0.85
    participation_floor_lookback: int = 126
    participation_floor_return_threshold: float = 0.02
    participation_floor_ma_gap_threshold: float = 0.02
    participation_floor_drawdown_threshold: float = -0.05
    recovery_floor_min_spy_weight: float | None = 0.8
    recovery_floor_long_lookback: int = 126
    recovery_floor_drawdown_threshold: float = -0.10
    recovery_floor_short_lookback: int = 21
    recovery_floor_return_threshold: float = 0.01
    recovery_floor_ma_gap_threshold: float = 0.0
    early_crack_floor_min_spy_weight: float | None = None
    early_crack_long_lookback: int = 126
    early_crack_long_return_threshold: float = 0.10
    early_crack_long_ma_gap_threshold: float = 0.04
    early_crack_long_drawdown_threshold: float = -0.03
    early_crack_trend_ma_gap_threshold: float = 0.02
    early_crack_short_lookback: int = 21
    early_crack_short_return_threshold: float = 0.0
    early_crack_short_ma_gap_threshold: float = 0.0
    early_crack_short_drawdown_threshold: float = -0.02
    constructive_crack_cap_min_spy_weight: float | None = None
    constructive_crack_cap_recent_constructive_lookback: int = 15
    constructive_crack_cap_long_lookback: int = 126
    constructive_crack_cap_long_return_threshold: float = 0.10
    constructive_crack_cap_long_ma_gap_threshold: float = 0.03
    constructive_crack_cap_long_drawdown_threshold: float = -0.04
    constructive_crack_cap_short_lookback: int = 21
    constructive_crack_cap_short_return_threshold: float = 0.0
    constructive_crack_cap_short_ma_gap_threshold: float = 0.0
    constructive_crack_cap_short_drawdown_threshold: float = -0.015
    constructive_crack_cap_current_trend_return_cap: float = 0.05
    constructive_crack_cap_current_trend_ma_gap_cap: float = 0.015
    constructive_crack_cap_current_long_return_min: float = 0.09
    constructive_crack_cap_current_long_return_max: float = 0.15
    constructive_crack_cap_sticky_days: int = 0
    action_bins: int = 21
    train_steps: int = 50_000
    episode_length: int = 252
    learning_rate: float = 3e-4
    batch_size: int = 256
    replay_capacity: int = 200_000
    learning_starts: int = 1_000
    target_update_interval: int = 500
    gradient_steps_per_update: int = 4
    hidden_size: int = 128
    quantiles: int = 31
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    validation_eval_interval: int = 250
    validation_slices: int = 1
    validation_stability_penalty: float = 0.0
    # Crisis filter: when set to a negative value (e.g. -0.20), checkpoint scoring skips
    # validation sub-windows where SPY drawdown from its local peak exceeds this threshold.
    # This prevents over-tuning the checkpoint on extreme crisis regimes (e.g. COVID crash)
    # that are absent from the test period.  0.0 = disabled (uses all of validation).
    validation_crisis_drawdown_threshold: float = 0.0
    loss_penalty: float = 1.5
    loss_power: float = 1.0
    gain_reward: float = 1.0
    # Optimizer baseline config — kept independent from the DRL reward shaping params above
    # so that the optimizer is a meaningful market-timing baseline regardless of DRL tuning.
    optimizer_loss_penalty: float = 8.0
    optimizer_lookback: int = 21
    benchmark_outperformance_reward: float = 0.5
    benchmark_shortfall_penalty: float = 3.0
    drawdown_penalty: float = 0.0
    bear_oversample_weight: float = 1.5
    validation_checkpoint_metric: str = "sharpe_ratio"  # "total_return" | "sharpe_ratio" | "relative_total_return"
    # Option A: OOD bear cap — when 252d drawdown falls below this threshold (e.g. -0.15),
    # hard-cap SPY weight at ood_bear_max_spy_weight.  0.0 = disabled.
    ood_bear_drawdown_threshold: float = 0.0
    ood_bear_max_spy_weight: float = 1.0
    # Option B: Bear episode truncation — end training episodes early once SPY has fallen
    # more than this fraction from the episode start (e.g. -0.15).  0.0 = disabled.
    bear_episode_spy_return_threshold: float = 0.0
    # Option C: Disable dynamic floors during training so the agent experiences the full
    # return consequences of its chosen weight (floors are still applied at eval/test time).
    # Prevents the floor from acting as a "free" target instead of a minimum guardrail.
    disable_floors_in_training: bool = True


@dataclass(frozen=True)
class Main2WindowMetrics:
    total_return: float
    relative_total_return: float
    average_cash_weight: float
    average_spy_weight: float
    average_turnover: float
    rebalance_count: int
    portfolio_returns: pd.Series


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("main2 requires torch. Install the repository with the optional RL dependencies.")


def _filter_returns_by_date(returns: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    filtered = returns.copy()
    if start is not None:
        filtered = filtered.loc[filtered.index >= pd.Timestamp(start)]
    if end is not None:
        filtered = filtered.loc[filtered.index <= pd.Timestamp(end)]
    if filtered.empty:
        raise ValueError("No rows remain after applying the requested date filter.")
    return filtered


def load_spy_source_returns(
    returns_csv: str | Path,
    benchmark_column: str,
    date_column: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    returns = load_returns_csv(returns_csv, date_column=date_column)
    returns = _filter_returns_by_date(returns, start=start, end=end)
    if benchmark_column not in returns.columns:
        raise ValueError(f"Benchmark column '{benchmark_column}' was not found in the returns CSV.")
    spy_returns = returns[benchmark_column].astype(float).rename("SPY")
    if spy_returns.empty:
        raise ValueError("SPY benchmark series is empty after filtering.")
    return spy_returns


def build_main2_feature_frame(spy_returns: pd.Series) -> pd.DataFrame:
    ordered_returns = pd.Series(spy_returns, copy=True).astype(float).sort_index()
    if ordered_returns.empty:
        raise ValueError("SPY returns are empty.")

    price = (1.0 + ordered_returns).cumprod().rename("price")
    lagged_price = price.shift(1)
    lagged_returns = ordered_returns.shift(1)

    def _rolling_total_return(window: int) -> pd.Series:
        return ((1.0 + ordered_returns).rolling(window, min_periods=1).apply(np.prod, raw=True) - 1.0).shift(1)

    def _rolling_vol(window: int) -> pd.Series:
        return ordered_returns.rolling(window, min_periods=2).std(ddof=0).shift(1)

    def _rolling_drawdown(window: int) -> pd.Series:
        rolling_peak = lagged_price.rolling(window, min_periods=1).max()
        return lagged_price / rolling_peak - 1.0

    def _ma_gap(window: int) -> pd.Series:
        rolling_mean = lagged_price.rolling(window, min_periods=1).mean()
        return lagged_price / rolling_mean - 1.0

    features = pd.DataFrame(
        {
            "ret_1": lagged_returns,
            "ret_5": _rolling_total_return(5),
            "ret_10": _rolling_total_return(10),
            "ret_21": _rolling_total_return(21),
            "ret_63": _rolling_total_return(63),
            "ret_252": _rolling_total_return(252),
            "vol_5": _rolling_vol(5),
            "vol_21": _rolling_vol(21),
            "vol_63": _rolling_vol(63),
            "drawdown_21": _rolling_drawdown(21),
            "drawdown_63": _rolling_drawdown(63),
            "drawdown_252": _rolling_drawdown(252),
            "ma_gap_10": _ma_gap(10),
            "ma_gap_21": _ma_gap(21),
            "ma_gap_63": _ma_gap(63),
        },
        index=ordered_returns.index,
    )
    return features.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def compute_main2_trailing_path_state(spy_returns: pd.Series, row_index: int, lookback: int) -> dict[str, float]:
    if lookback <= 0:
        raise ValueError("lookback must be positive.")
    if row_index < 0 or row_index > len(spy_returns):
        raise ValueError("row_index is outside the SPY return series.")

    history_end = row_index
    history_start = max(0, history_end - lookback)
    history = spy_returns.iloc[history_start:history_end].astype(float)
    if history.empty:
        return {
            "total_return": 0.0,
            "current_drawdown": 0.0,
            "max_drawdown": 0.0,
            "ma_gap": 0.0,
        }

    trailing_wealth = (1.0 + history).cumprod()
    running_peak = trailing_wealth.cummax()
    current_drawdown = float(trailing_wealth.iloc[-1] / running_peak.iloc[-1] - 1.0)
    max_drawdown = float((trailing_wealth / running_peak - 1.0).min())
    total_return = float(trailing_wealth.iloc[-1] - 1.0)

    trailing_price = (1.0 + spy_returns.astype(float)).cumprod()
    history_prices = trailing_price.iloc[history_start:history_end]
    if history_prices.empty:
        ma_gap = 0.0
    else:
        lagged_price = float(trailing_price.iloc[history_end - 1])
        ma_gap = float(lagged_price / float(history_prices.mean()) - 1.0)

    return {
        "total_return": total_return,
        "current_drawdown": current_drawdown,
        "max_drawdown": max_drawdown,
        "ma_gap": ma_gap,
    }


def _find_recent_constructive_crack_reference_row_index(
    spy_returns: pd.Series,
    row_index: int,
    config: Main2Config,
) -> int | None:
    search_start = max(0, row_index - max(config.constructive_crack_cap_recent_constructive_lookback, 0))
    for candidate_row_index in range(row_index, search_start - 1, -1):
        candidate_state_63 = compute_main2_trailing_path_state(
            spy_returns,
            row_index=candidate_row_index,
            lookback=max(config.trend_floor_lookback, 1),
        )
        candidate_state_long = compute_main2_trailing_path_state(
            spy_returns,
            row_index=candidate_row_index,
            lookback=max(config.constructive_crack_cap_long_lookback, 1),
        )
        if (
            candidate_state_63["total_return"] >= config.trend_floor_return_threshold
            and candidate_state_63["current_drawdown"] >= config.trend_floor_drawdown_threshold
            and candidate_state_long["total_return"] >= config.constructive_crack_cap_long_return_threshold
            and candidate_state_long["ma_gap"] >= config.constructive_crack_cap_long_ma_gap_threshold
            and candidate_state_long["current_drawdown"] >= config.constructive_crack_cap_long_drawdown_threshold
        ):
            return candidate_row_index
    return None


def compute_main2_dynamic_floor_components(spy_returns: pd.Series, row_index: int, config: Main2Config) -> dict[str, float | bool]:
    base_min_spy_weight = float(config.min_spy_weight)
    state_21 = compute_main2_trailing_path_state(spy_returns, row_index=row_index, lookback=21)
    state_63 = compute_main2_trailing_path_state(spy_returns, row_index=row_index, lookback=max(config.trend_floor_lookback, 1))
    state_participation = compute_main2_trailing_path_state(
        spy_returns,
        row_index=row_index,
        lookback=max(config.participation_floor_lookback, 1),
    )
    state_recovery_long = compute_main2_trailing_path_state(
        spy_returns,
        row_index=row_index,
        lookback=max(config.recovery_floor_long_lookback, 1),
    )
    state_recovery_short = compute_main2_trailing_path_state(
        spy_returns,
        row_index=row_index,
        lookback=max(config.recovery_floor_short_lookback, 1),
    )
    state_early_crack_long = compute_main2_trailing_path_state(
        spy_returns,
        row_index=row_index,
        lookback=max(config.early_crack_long_lookback, 1),
    )
    state_early_crack_short = compute_main2_trailing_path_state(
        spy_returns,
        row_index=row_index,
        lookback=max(config.early_crack_short_lookback, 1),
    )
    state_constructive_crack_long = compute_main2_trailing_path_state(
        spy_returns,
        row_index=row_index,
        lookback=max(config.constructive_crack_cap_long_lookback, 1),
    )
    state_constructive_crack_short = compute_main2_trailing_path_state(
        spy_returns,
        row_index=row_index,
        lookback=max(config.constructive_crack_cap_short_lookback, 1),
    )

    dynamic_min_spy_weight = base_min_spy_weight

    trend_floor_active = False
    early_crack_active = False
    trend_floor_weight = config.trend_floor_min_spy_weight
    effective_trend_floor_min_spy_weight = base_min_spy_weight
    if trend_floor_weight is not None and trend_floor_weight > dynamic_min_spy_weight:
        trend_floor_active = (
            state_63["total_return"] >= config.trend_floor_return_threshold
            and state_63["current_drawdown"] >= config.trend_floor_drawdown_threshold
        )
        if trend_floor_active:
            effective_trend_floor_min_spy_weight = min(float(trend_floor_weight), config.max_spy_weight)
            early_crack_floor_weight = config.early_crack_floor_min_spy_weight
            if early_crack_floor_weight is not None and early_crack_floor_weight < effective_trend_floor_min_spy_weight:
                early_crack_active = (
                    state_63["ma_gap"] >= config.early_crack_trend_ma_gap_threshold
                    and state_early_crack_long["total_return"] >= config.early_crack_long_return_threshold
                    and state_early_crack_long["ma_gap"] >= config.early_crack_long_ma_gap_threshold
                    and state_early_crack_long["current_drawdown"] >= config.early_crack_long_drawdown_threshold
                    and state_early_crack_short["total_return"] <= config.early_crack_short_return_threshold
                    and state_early_crack_short["ma_gap"] <= config.early_crack_short_ma_gap_threshold
                    and state_early_crack_short["current_drawdown"] <= config.early_crack_short_drawdown_threshold
                )
                if early_crack_active:
                    effective_trend_floor_min_spy_weight = max(
                        base_min_spy_weight,
                        min(float(early_crack_floor_weight), config.max_spy_weight),
                    )
            dynamic_min_spy_weight = max(dynamic_min_spy_weight, effective_trend_floor_min_spy_weight)

    participation_floor_active = False
    participation_floor_weight = config.participation_floor_min_spy_weight
    if participation_floor_weight is not None and participation_floor_weight > dynamic_min_spy_weight:
        participation_floor_active = (
            state_participation["total_return"] >= config.participation_floor_return_threshold
            and state_participation["ma_gap"] >= config.participation_floor_ma_gap_threshold
            and state_63["current_drawdown"] >= config.participation_floor_drawdown_threshold
        )
        if participation_floor_active:
            dynamic_min_spy_weight = max(dynamic_min_spy_weight, min(float(participation_floor_weight), config.max_spy_weight))

    recovery_floor_active = False
    recovery_floor_weight = config.recovery_floor_min_spy_weight
    if recovery_floor_weight is not None and recovery_floor_weight > dynamic_min_spy_weight:
        recovery_floor_active = (
            state_recovery_long["current_drawdown"] <= config.recovery_floor_drawdown_threshold
            and state_recovery_short["total_return"] >= config.recovery_floor_return_threshold
            and state_recovery_short["ma_gap"] >= config.recovery_floor_ma_gap_threshold
        )
        if recovery_floor_active:
            dynamic_min_spy_weight = max(dynamic_min_spy_weight, min(float(recovery_floor_weight), config.max_spy_weight))

    constructive_crack_cap_active = False
    constructive_crack_cap_sticky_active = False
    constructive_crack_recent_constructive_active = False
    constructive_crack_recent_constructive_days_ago = float("nan")
    constructive_crack_cap_weight = config.constructive_crack_cap_min_spy_weight
    if constructive_crack_cap_weight is not None and constructive_crack_cap_weight < dynamic_min_spy_weight:
        constructive_crack_cap_candidate = (
            state_constructive_crack_short["total_return"] <= config.constructive_crack_cap_short_return_threshold
            and state_constructive_crack_short["ma_gap"] <= config.constructive_crack_cap_short_ma_gap_threshold
            and state_constructive_crack_short["current_drawdown"] <= config.constructive_crack_cap_short_drawdown_threshold
            and state_63["total_return"] <= config.constructive_crack_cap_current_trend_return_cap
            and state_63["ma_gap"] <= config.constructive_crack_cap_current_trend_ma_gap_cap
            and state_constructive_crack_long["total_return"] >= config.constructive_crack_cap_current_long_return_min
            and state_constructive_crack_long["total_return"] <= config.constructive_crack_cap_current_long_return_max
        )
        if constructive_crack_cap_candidate:
            recent_constructive_row_index = _find_recent_constructive_crack_reference_row_index(
                spy_returns,
                row_index=row_index,
                config=config,
            )
            if recent_constructive_row_index is not None:
                constructive_crack_recent_constructive_active = True
                constructive_crack_recent_constructive_days_ago = float(row_index - recent_constructive_row_index)
                constructive_crack_cap_active = True
                constructive_crack_cap_weight = min(float(constructive_crack_cap_weight), config.max_spy_weight)
                dynamic_min_spy_weight = max(
                    base_min_spy_weight,
                    min(dynamic_min_spy_weight, constructive_crack_cap_weight),
                )
        # Sticky: if the cap didn't fire today but fired within the last N days, keep it active.
        if not constructive_crack_cap_active and config.constructive_crack_cap_sticky_days > 0:
            sticky_config = replace(config, constructive_crack_cap_sticky_days=0)
            for prior_offset in range(1, config.constructive_crack_cap_sticky_days + 1):
                prior_row = row_index - prior_offset
                if prior_row < 0:
                    break
                prior_fc = compute_main2_dynamic_floor_components(spy_returns, prior_row, sticky_config)
                if prior_fc["constructive_crack_cap_active"]:
                    constructive_crack_cap_sticky_active = True
                    constructive_crack_cap_active = True
                    dynamic_min_spy_weight = max(
                        base_min_spy_weight,
                        min(dynamic_min_spy_weight, float(config.constructive_crack_cap_min_spy_weight)),
                    )
                    break

    return {
        "dynamic_min_spy_weight": float(dynamic_min_spy_weight),
        "effective_trend_floor_min_spy_weight": float(effective_trend_floor_min_spy_weight),
        "trend_floor_active": trend_floor_active,
        "early_crack_active": early_crack_active,
        "participation_floor_active": participation_floor_active,
        "recovery_floor_active": recovery_floor_active,
        "constructive_crack_cap_active": constructive_crack_cap_active,
        "constructive_crack_cap_sticky_active": constructive_crack_cap_sticky_active,
        "constructive_crack_recent_constructive_active": constructive_crack_recent_constructive_active,
        "constructive_crack_recent_constructive_days_ago": float(constructive_crack_recent_constructive_days_ago),
        "ret_21": float(state_21["total_return"]),
        "dd_21": float(state_21["current_drawdown"]),
        "ma_gap_21": float(state_21["ma_gap"]),
        "ret_63": float(state_63["total_return"]),
        "dd_63": float(state_63["current_drawdown"]),
        "ma_gap_63": float(state_63["ma_gap"]),
        "participation_ret": float(state_participation["total_return"]),
        "participation_dd": float(state_participation["current_drawdown"]),
        "participation_ma_gap": float(state_participation["ma_gap"]),
        "recovery_long_dd": float(state_recovery_long["current_drawdown"]),
        "recovery_short_ret": float(state_recovery_short["total_return"]),
        "recovery_short_ma_gap": float(state_recovery_short["ma_gap"]),
        "early_crack_long_ret": float(state_early_crack_long["total_return"]),
        "early_crack_long_dd": float(state_early_crack_long["current_drawdown"]),
        "early_crack_long_ma_gap": float(state_early_crack_long["ma_gap"]),
        "early_crack_short_ret": float(state_early_crack_short["total_return"]),
        "early_crack_short_dd": float(state_early_crack_short["current_drawdown"]),
        "early_crack_short_ma_gap": float(state_early_crack_short["ma_gap"]),
        "constructive_crack_long_ret": float(state_constructive_crack_long["total_return"]),
        "constructive_crack_long_dd": float(state_constructive_crack_long["current_drawdown"]),
        "constructive_crack_long_ma_gap": float(state_constructive_crack_long["ma_gap"]),
        "constructive_crack_short_ret": float(state_constructive_crack_short["total_return"]),
        "constructive_crack_short_dd": float(state_constructive_crack_short["current_drawdown"]),
        "constructive_crack_short_ma_gap": float(state_constructive_crack_short["ma_gap"]),
    }


def normalize_main2_features(
    features: pd.DataFrame,
    train_end_index: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if train_end_index < 0 or train_end_index >= len(features):
        raise ValueError("train_end_index is outside the feature frame.")

    train_slice = features.iloc[: train_end_index + 1]
    means = train_slice.mean(axis=0)
    stds = train_slice.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    normalized = (features - means) / stds
    return normalized.fillna(0.0), means, stds


class Main2SPYEnv:
    def __init__(
        self,
        normalized_features: pd.DataFrame,
        spy_returns: pd.Series,
        start_index: int,
        end_index: int,
        config: Main2Config,
        random_episode_start: bool,
    ) -> None:
        if start_index < 0 or end_index >= len(normalized_features) or start_index > end_index:
            raise ValueError("Invalid main2 environment window bounds.")
        if not normalized_features.index.equals(spy_returns.index):
            raise ValueError("Features and SPY returns must share the same index.")

        self.features = normalized_features.astype(np.float32)
        self.spy_returns = pd.Series(spy_returns, copy=True).astype(float)
        self.start_index = int(start_index)
        self.end_index = int(end_index)
        self.config = config
        self.random_episode_start = random_episode_start
        self.action_grid = np.linspace(config.min_spy_weight, config.max_spy_weight, config.action_bins, dtype=np.float32)
        self._rng = np.random.default_rng(0)
        self.current_index = self.start_index
        self.episode_end_index = self.end_index
        self.current_spy_weight = float(np.clip(config.initial_spy_weight, config.min_spy_weight, config.max_spy_weight))
        self.current_wealth = 1.0
        self.max_wealth = 1.0
        self._ep_spy_cum: float = 1.0
        # Precompute raw 252-day trailing drawdown for the OOD bear cap (Option A).
        _price = (1.0 + spy_returns).cumprod()
        _peak = _price.rolling(252, min_periods=1).max()
        self._raw_dd252 = (_price / _peak - 1.0).to_numpy(dtype=np.float64)

    @property
    def observation_size(self) -> int:
        return self.config.lookback * self.features.shape[1] + 3

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self.random_episode_start:
            latest_start = max(self.start_index, self.end_index - self.config.episode_length + 1)
            candidate_starts = np.arange(self.start_index, latest_start + 1)
            if self.config.bear_oversample_weight > 0.0 and len(candidate_starts) > 1:
                # Weight each start by 1 + bear_oversample_weight * |drawdown at that day|.
                # Drawdown is computed from the running peak of SPY returns in the training window.
                cum = np.cumprod(1.0 + self.spy_returns.iloc[self.start_index: latest_start + 1].to_numpy())
                peak = np.maximum.accumulate(cum)
                drawdown = cum / peak - 1.0  # <= 0
                weights = 1.0 + self.config.bear_oversample_weight * np.abs(np.minimum(drawdown, 0.0))
                weights = weights / weights.sum()
                self.current_index = int(self._rng.choice(candidate_starts, p=weights))
            else:
                self.current_index = int(self._rng.integers(self.start_index, latest_start + 1))
            self.episode_end_index = min(self.end_index, self.current_index + self.config.episode_length - 1)
        else:
            self.current_index = self.start_index
            self.episode_end_index = self.end_index

        self.current_spy_weight = float(np.clip(self.config.initial_spy_weight, self.config.min_spy_weight, self.config.max_spy_weight))
        self.current_wealth = 1.0
        self.max_wealth = 1.0
        self._ep_spy_cum = 1.0
        return self._build_observation(self.current_index)

    def step(self, action_index: int) -> tuple[np.ndarray, float, bool, dict[str, float | int | str]]:
        selected_weight = float(self.action_grid[int(action_index)])
        smoothed_weight = self.current_spy_weight + (1.0 - self.config.action_smoothing) * (selected_weight - self.current_spy_weight)
        # Option C: during training episodes, bypass dynamic floors so the agent experiences
        # the real return consequences of its chosen weight. Floors still apply at eval/test.
        if self.random_episode_start and self.config.disable_floors_in_training:
            dynamic_min_spy_weight = self.config.min_spy_weight
        else:
            dynamic_min_spy_weight = self._dynamic_min_spy_weight(self.current_index)
        # Option A: OOD bear cap — when 252d drawdown is below the configured threshold,
        # hard-cap effective max SPY weight to limit unrecovered-bear exposure.
        effective_max_spy_weight = self.config.max_spy_weight
        if self.config.ood_bear_drawdown_threshold < 0.0:
            if self._raw_dd252[self.current_index] < self.config.ood_bear_drawdown_threshold:
                effective_max_spy_weight = min(self.config.max_spy_weight, self.config.ood_bear_max_spy_weight)
        target_spy_weight = float(np.clip(smoothed_weight, dynamic_min_spy_weight, effective_max_spy_weight))
        if abs(target_spy_weight - self.current_spy_weight) < self.config.no_trade_band:
            target_spy_weight = float(self.current_spy_weight)

        turnover = abs(target_spy_weight - self.current_spy_weight)
        trading_cost = turnover * (self.config.transaction_cost_bps + self.config.slippage_bps) / 10_000.0
        realized_return = float(self.spy_returns.iloc[self.current_index])
        # Option B: track cumulative SPY return from episode start for early-termination.
        self._ep_spy_cum *= (1.0 + realized_return)
        gross_portfolio_return = target_spy_weight * realized_return
        net_portfolio_return = gross_portfolio_return - trading_cost
        reward = self._score_reward(net_portfolio_return, realized_return)

        wealth_ratio = max(1.0 + net_portfolio_return, 1e-8)
        self.current_wealth *= wealth_ratio
        self.max_wealth = max(self.max_wealth, self.current_wealth)

        risky_value = target_spy_weight * (1.0 + realized_return)
        cash_value = 1.0 - target_spy_weight
        gross_total = risky_value + cash_value
        if gross_total <= 1e-8:
            next_spy_weight = target_spy_weight
        else:
            next_spy_weight = float(np.clip(risky_value / gross_total, 0.0, 1.0))
        self.current_spy_weight = next_spy_weight

        current_label = self.spy_returns.index[self.current_index]
        done = self.current_index >= self.episode_end_index
        # Option B: early-terminate training episodes once SPY has fallen far enough
        # from the episode start so the Q-function experiences bear losses without recovery.
        if (
            self.config.bear_episode_spy_return_threshold < 0.0
            and self.random_episode_start
            and (self._ep_spy_cum - 1.0) < self.config.bear_episode_spy_return_threshold
        ):
            done = True
        self.current_index += 1
        next_observation = np.zeros(self.observation_size, dtype=np.float32) if done else self._build_observation(self.current_index)
        info: dict[str, float | int | str] = {
            "date": str(current_label),
            "target_spy_weight": target_spy_weight,
            "dynamic_min_spy_weight": float(dynamic_min_spy_weight),
            "turnover": float(turnover),
            "trading_cost": float(trading_cost),
            "gross_portfolio_return": float(gross_portfolio_return),
            "net_portfolio_return": float(net_portfolio_return),
            "wealth": float(self.current_wealth),
            "post_return_spy_weight": float(self.current_spy_weight),
        }
        return next_observation, float(reward), done, info

    def _build_observation(self, row_index: int) -> np.ndarray:
        window_start = max(0, row_index - self.config.lookback + 1)
        feature_window = self.features.iloc[window_start : row_index + 1].to_numpy(dtype=np.float32)
        if len(feature_window) < self.config.lookback:
            pad_count = self.config.lookback - len(feature_window)
            pad_row = feature_window[0] if len(feature_window) > 0 else np.zeros(self.features.shape[1], dtype=np.float32)
            pad_block = np.repeat(pad_row.reshape(1, -1), pad_count, axis=0)
            feature_window = np.vstack([pad_block, feature_window])
        extras = np.array(
            [
                float(self.current_spy_weight),
                float(1.0 - self.current_spy_weight),
                float(self.current_wealth / max(self.max_wealth, 1e-8) - 1.0),
            ],
            dtype=np.float32,
        )
        return np.concatenate([feature_window.reshape(-1), extras]).astype(np.float32, copy=False)

    def _score_reward(self, portfolio_return: float, benchmark_return: float) -> float:
        wealth_ratio = max(1.0 + portfolio_return, 1e-8)
        log_growth = math.log(wealth_ratio)
        if log_growth >= 0.0:
            reward = self.config.gain_reward * log_growth
        else:
            downside = -log_growth
            reward = -self.config.loss_penalty * (downside ** self.config.loss_power)

        active_return = portfolio_return - benchmark_return
        if active_return >= 0.0:
            reward += self.config.benchmark_outperformance_reward * active_return
        else:
            reward -= self.config.benchmark_shortfall_penalty * ((-active_return) ** self.config.loss_power)

        if self.config.drawdown_penalty > 0.0:
            drawdown_from_peak = self.current_wealth / max(self.max_wealth, 1e-8) - 1.0
            if drawdown_from_peak < 0.0:
                reward -= self.config.drawdown_penalty * (-drawdown_from_peak)
        return reward

    def _dynamic_min_spy_weight(self, row_index: int) -> float:
        floor_components = compute_main2_dynamic_floor_components(self.spy_returns, row_index=row_index, config=self.config)
        return float(floor_components["dynamic_min_spy_weight"])


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._storage: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self._position = 0

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        payload = (state.copy(), int(action), float(reward), next_state.copy(), bool(done))
        if len(self._storage) < self.capacity:
            self._storage.append(payload)
        else:
            self._storage[self._position] = payload
        self._position = (self._position + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = rng.choice(len(self._storage), size=batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self._storage[int(index)] for index in indices), strict=False)
        return (
            np.stack(states).astype(np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.asarray(dones, dtype=np.float32),
        )


class QuantileQNetwork(nn.Module):
    def __init__(self, observation_size: int, action_count: int, hidden_size: int, quantiles: int) -> None:
        super().__init__()
        self.action_count = action_count
        self.quantiles = quantiles
        self.network = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count * quantiles),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        raw = self.network(observations)
        return raw.view(-1, self.action_count, self.quantiles)

    def q_values(self, observations: torch.Tensor) -> torch.Tensor:
        return self.forward(observations).mean(dim=-1)


def _quantile_huber_loss(td_errors: torch.Tensor, quantiles: int) -> torch.Tensor:
    assert torch is not None
    tau = (torch.arange(quantiles, device=td_errors.device, dtype=torch.float32) + 0.5) / quantiles
    tau = tau.view(1, quantiles, 1)
    abs_error = td_errors.abs()
    huber = torch.where(abs_error <= 1.0, 0.5 * td_errors.pow(2), abs_error - 0.5)
    quantile_weight = torch.abs(tau - (td_errors.detach() < 0.0).float())
    return (quantile_weight * huber).mean()


def evaluate_main2_policy(env: Main2SPYEnv, network: QuantileQNetwork, device: torch.device) -> Main2WindowMetrics:
    assert torch is not None
    was_training = network.training
    network.eval()
    observation = env.reset()
    policy_returns: list[float] = []
    target_spy_weights: list[float] = []
    turnovers: list[float] = []

    while True:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_index = int(network.q_values(observation_tensor).argmax(dim=1).item())
        observation, _, done, info = env.step(action_index)
        policy_returns.append(float(info["net_portfolio_return"]))
        target_spy_weights.append(float(info["target_spy_weight"]))
        turnovers.append(float(info["turnover"]))
        if done:
            break

    portfolio_returns = pd.Series(policy_returns, index=env.spy_returns.index[env.start_index : env.end_index + 1], name="portfolio_return")
    summary = summarize_performance(portfolio_returns)
    benchmark_total_return = float(np.prod(1.0 + env.spy_returns.iloc[env.start_index : env.end_index + 1].to_numpy(dtype=float)) - 1.0)
    relative_total_return = float((1.0 + summary["total_return"]) / (1.0 + benchmark_total_return) - 1.0)
    average_spy_weight = float(np.mean(target_spy_weights)) if target_spy_weights else float(env.config.initial_spy_weight)
    metrics = Main2WindowMetrics(
        total_return=float(summary["total_return"]),
        relative_total_return=relative_total_return,
        average_cash_weight=float(1.0 - average_spy_weight),
        average_spy_weight=average_spy_weight,
        average_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        rebalance_count=int(sum(turnover > 1e-9 for turnover in turnovers)),
        portfolio_returns=portfolio_returns,
    )
    if was_training:
        network.train()
    return metrics


def _validation_window_score(metrics: Main2WindowMetrics, metric: str) -> float:
    if metric == "sharpe_ratio":
        rets = metrics.portfolio_returns.to_numpy(dtype=float)
        std = float(np.std(rets, ddof=1))
        return float(np.mean(rets)) / std * (252 ** 0.5) if std > 1e-9 else 0.0
    if metric == "relative_total_return":
        return float(metrics.relative_total_return)
    return float(metrics.total_return)


def _build_validation_slices(base_env: Main2SPYEnv, slice_count: int) -> list[Main2SPYEnv]:
    if slice_count <= 1:
        return [base_env]

    total_rows = base_env.end_index - base_env.start_index + 1
    if total_rows <= 1:
        return [base_env]

    slice_count = min(slice_count, total_rows)
    edges = np.linspace(base_env.start_index, base_env.end_index + 1, slice_count + 1, dtype=int)
    slices: list[Main2SPYEnv] = []
    for i in range(slice_count):
        start_index = int(edges[i])
        end_index = int(edges[i + 1] - 1)
        if end_index < start_index:
            continue
        slices.append(
            Main2SPYEnv(
                normalized_features=base_env.features,
                spy_returns=base_env.spy_returns,
                start_index=start_index,
                end_index=end_index,
                config=base_env.config,
                random_episode_start=False,
            )
        )
    return slices or [base_env]


def _build_crisis_filtered_validation_slices(
    base_env: Main2SPYEnv,
    drawdown_threshold: float,
    min_segment_rows: int = 21,
) -> list[Main2SPYEnv]:
    """Return validation sub-windows that exclude crisis periods (drawdown < threshold).

    Computes running SPY drawdown from its cumulative peak within the validation window.
    Contiguous stretches where drawdown > threshold become separate slice envs used for
    checkpoint scoring.  If no stretch is long enough, falls back to the full window so
    training always has at least one validation signal.
    """
    val_spy = base_env.spy_returns.iloc[base_env.start_index : base_env.end_index + 1]
    wealth = (1.0 + val_spy).cumprod().to_numpy()
    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / running_max - 1.0
    clean_mask = drawdown > drawdown_threshold

    segments: list[tuple[int, int]] = []  # (offset_start, offset_end) relative to start_index
    in_seg = False
    seg_start = 0
    for k, clean in enumerate(clean_mask):
        if clean and not in_seg:
            seg_start = k
            in_seg = True
        elif not clean and in_seg:
            if k - seg_start >= min_segment_rows:
                segments.append((seg_start, k - 1))
            in_seg = False
    if in_seg:
        seg_len = len(clean_mask) - seg_start
        if seg_len >= min_segment_rows:
            segments.append((seg_start, len(clean_mask) - 1))

    if not segments:
        return [base_env]

    slices: list[Main2SPYEnv] = []
    for off_start, off_end in segments:
        slices.append(
            Main2SPYEnv(
                normalized_features=base_env.features,
                spy_returns=base_env.spy_returns,
                start_index=base_env.start_index + off_start,
                end_index=base_env.start_index + off_end,
                config=base_env.config,
                random_episode_start=False,
            )
        )
    return slices


def train_main2_quantile_agent(
    train_env: Main2SPYEnv,
    validation_env: Main2SPYEnv,
    config: Main2Config,
    seed: int,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    progress_context: dict[str, Any] | None = None,
) -> QuantileQNetwork:
    _require_torch()
    assert torch is not None

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")
    online = QuantileQNetwork(
        observation_size=train_env.observation_size,
        action_count=len(train_env.action_grid),
        hidden_size=config.hidden_size,
        quantiles=config.quantiles,
    ).to(device)
    target = copy.deepcopy(online).to(device)
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(capacity=config.replay_capacity)
    rng = np.random.default_rng(seed)
    if config.validation_crisis_drawdown_threshold < 0.0:
        validation_env_slices = _build_crisis_filtered_validation_slices(
            validation_env, config.validation_crisis_drawdown_threshold
        )
    else:
        validation_env_slices = _build_validation_slices(validation_env, config.validation_slices)

    best_validation_score = float("-inf")
    best_state_dict = copy.deepcopy(online.state_dict())
    observation = train_env.reset(seed=seed)
    episode_reset_count = 0
    latest_loss: float | None = None

    for step in range(1, config.train_steps + 1):
        epsilon_progress = min(1.0, step / max(config.epsilon_decay_steps, 1))
        epsilon = config.epsilon_start + epsilon_progress * (config.epsilon_end - config.epsilon_start)
        if rng.random() < epsilon:
            action_index = int(rng.integers(0, len(train_env.action_grid)))
        else:
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_index = int(online.q_values(observation_tensor).argmax(dim=1).item())

        next_observation, reward, done, _ = train_env.step(action_index)
        replay.add(observation, action_index, reward, next_observation, done)
        observation = next_observation

        if done:
            observation = train_env.reset()
            episode_reset_count += 1

        if len(replay) >= max(config.learning_starts, config.batch_size):
            for _ in range(config.gradient_steps_per_update):
                states, actions, rewards, next_states, dones = replay.sample(config.batch_size, rng)
                states_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
                actions_tensor = torch.as_tensor(actions, dtype=torch.int64, device=device)
                rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
                next_states_tensor = torch.as_tensor(next_states, dtype=torch.float32, device=device)
                dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=device)

                current_quantiles = online(states_tensor)[torch.arange(config.batch_size, device=device), actions_tensor]
                with torch.no_grad():
                    next_actions = online.q_values(next_states_tensor).argmax(dim=1)
                    next_quantiles = target(next_states_tensor)[torch.arange(config.batch_size, device=device), next_actions]
                    target_quantiles = rewards_tensor.unsqueeze(1) + config.gamma * (1.0 - dones_tensor.unsqueeze(1)) * next_quantiles

                td_errors = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
                loss = _quantile_huber_loss(td_errors, config.quantiles)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
                optimizer.step()
                latest_loss = float(loss.item())

        if step % config.target_update_interval == 0:
            target.load_state_dict(online.state_dict())

        if step % config.validation_eval_interval == 0 or step == config.train_steps:
            metric = config.validation_checkpoint_metric
            slice_scores: list[float] = []
            for slice_env in validation_env_slices:
                validation_metrics = evaluate_main2_policy(slice_env, online, device)
                slice_scores.append(_validation_window_score(validation_metrics, metric))

            mean_slice_score = float(np.mean(slice_scores)) if slice_scores else float("-inf")
            if len(slice_scores) > 1 and config.validation_stability_penalty > 0.0:
                score_penalty = config.validation_stability_penalty * float(np.std(slice_scores, ddof=0))
            else:
                score_penalty = 0.0
            val_score = mean_slice_score - score_penalty
            checkpoint_improved = val_score > best_validation_score
            if checkpoint_improved:
                best_validation_score = val_score
                best_state_dict = copy.deepcopy(online.state_dict())

            if progress_callback is not None:
                progress_callback(
                    "training_checkpoint",
                    {
                        **(progress_context or {}),
                        "step": step,
                        "train_steps": config.train_steps,
                        "progress_fraction": step / max(config.train_steps, 1),
                        "epsilon": epsilon,
                        "validation_checkpoint_metric": metric,
                        "validation_score": val_score,
                        "best_validation_score": best_validation_score,
                        "validation_score_penalty": score_penalty,
                        "validation_slice_scores": slice_scores,
                        "checkpoint_improved": checkpoint_improved,
                        "episode_reset_count": episode_reset_count,
                        "replay_size": len(replay),
                        "latest_loss": latest_loss,
                    },
                )

    online.load_state_dict(best_state_dict)
    return online


def _phase_slice(series_or_frame: pd.Series | pd.DataFrame, start_label: object, end_label: object) -> pd.Series | pd.DataFrame:
    return series_or_frame.loc[start_label:end_label]


def summarize_main2_static_hold(spy_returns: pd.Series, start_label: object, end_label: object) -> Main2WindowMetrics:
    phase_returns = _phase_slice(spy_returns, start_label, end_label).astype(float)
    summary = summarize_performance(phase_returns)
    return Main2WindowMetrics(
        total_return=float(summary["total_return"]),
        relative_total_return=0.0,
        average_cash_weight=0.0,
        average_spy_weight=1.0,
        average_turnover=0.0,
        rebalance_count=1,
        portfolio_returns=phase_returns.rename("portfolio_return"),
    )


def summarize_main2_optimizer_baseline(
    optimizer_result,
    start_label: object,
    end_label: object,
    benchmark_total_return: float,
) -> Main2WindowMetrics:
    portfolio_returns = _phase_slice(optimizer_result.portfolio_returns, start_label, end_label).astype(float)
    summary = summarize_performance(portfolio_returns)
    phase_weights = _phase_slice(optimizer_result.weights, start_label, end_label)
    phase_turnover = _phase_slice(optimizer_result.turnover, start_label, end_label)
    return Main2WindowMetrics(
        total_return=float(summary["total_return"]),
        relative_total_return=float((1.0 + summary["total_return"]) / (1.0 + benchmark_total_return) - 1.0),
        average_cash_weight=float(phase_weights["CASH"].mean()) if "CASH" in phase_weights.columns else 0.0,
        average_spy_weight=float(phase_weights["SPY"].mean()) if "SPY" in phase_weights.columns else 0.0,
        average_turnover=float(phase_turnover.mean()),
        rebalance_count=int((phase_turnover > 1e-9).sum()),
        portfolio_returns=portfolio_returns.rename("portfolio_return"),
    )


def build_main2_detail_row(
    split_name: str,
    seed: int,
    fold_number: int,
    phase_name: str,
    phase_start: object,
    phase_end: object,
    policy_metrics: Main2WindowMetrics,
    static_hold_metrics: Main2WindowMetrics,
    optimizer_metrics: Main2WindowMetrics,
) -> dict[str, float | int | str]:
    return {
        "split": split_name,
        "seed": int(seed),
        "fold": f"fold_{fold_number:02d}",
        "phase": phase_name,
        "phase_start": str(phase_start),
        "phase_end": str(phase_end),
        "policy_total_return": policy_metrics.total_return,
        "static_hold_total_return": static_hold_metrics.total_return,
        "optimizer_total_return": optimizer_metrics.total_return,
        "policy_relative_total_return": policy_metrics.relative_total_return,
        "static_hold_relative_total_return": static_hold_metrics.relative_total_return,
        "optimizer_relative_total_return": optimizer_metrics.relative_total_return,
        "policy_cash_weight": policy_metrics.average_cash_weight,
        "static_hold_cash_weight": static_hold_metrics.average_cash_weight,
        "optimizer_cash_weight": optimizer_metrics.average_cash_weight,
        "policy_trend_weight": policy_metrics.average_spy_weight,
        "static_hold_trend_weight": static_hold_metrics.average_spy_weight,
        "policy_turnover": policy_metrics.average_turnover,
        "static_hold_turnover": static_hold_metrics.average_turnover,
        "optimizer_turnover": optimizer_metrics.average_turnover,
        "policy_rebalance_count": float(policy_metrics.rebalance_count),
        "static_hold_rebalance_count": float(static_hold_metrics.rebalance_count),
        "policy_late_def_rate": 0.0,
        "delta_total_return_vs_static_hold": policy_metrics.total_return - static_hold_metrics.total_return,
        "delta_total_return_vs_optimizer": policy_metrics.total_return - optimizer_metrics.total_return,
        "delta_relative_total_return_vs_static_hold": policy_metrics.relative_total_return - static_hold_metrics.relative_total_return,
        "delta_relative_total_return_vs_optimizer": policy_metrics.relative_total_return - optimizer_metrics.relative_total_return,
        "delta_cash_weight_vs_static_hold": policy_metrics.average_cash_weight - static_hold_metrics.average_cash_weight,
        "delta_cash_weight_vs_optimizer": policy_metrics.average_cash_weight - optimizer_metrics.average_cash_weight,
        "delta_trend_weight_vs_static_hold": policy_metrics.average_spy_weight - static_hold_metrics.average_spy_weight,
        "delta_turnover_vs_static_hold": policy_metrics.average_turnover - static_hold_metrics.average_turnover,
        "delta_turnover_vs_optimizer": policy_metrics.average_turnover - optimizer_metrics.average_turnover,
        "delta_rebalance_count_vs_static_hold": float(policy_metrics.rebalance_count - static_hold_metrics.rebalance_count),
    }


def build_main2_eval_summary(detail_frame: pd.DataFrame) -> dict[str, object]:
    overall = {
        "rows": int(len(detail_frame)),
        "policy_beats_static_hold_rows": int((detail_frame["delta_total_return_vs_static_hold"] > 0).sum()),
        "policy_beats_optimizer_rows": int((detail_frame["delta_total_return_vs_optimizer"] > 0).sum()),
        "mean_policy_total_return": float(detail_frame["policy_total_return"].mean()),
        "mean_static_hold_total_return": float(detail_frame["static_hold_total_return"].mean()),
        "mean_optimizer_total_return": float(detail_frame["optimizer_total_return"].mean()),
        "mean_delta_total_return_vs_static_hold": float(detail_frame["delta_total_return_vs_static_hold"].mean()),
        "mean_delta_total_return_vs_optimizer": float(detail_frame["delta_total_return_vs_optimizer"].mean()),
        "mean_delta_cash_weight_vs_static_hold": float(detail_frame["delta_cash_weight_vs_static_hold"].mean()),
        "mean_delta_turnover_vs_static_hold": float(detail_frame["delta_turnover_vs_static_hold"].mean()),
    }

    def _group_summary(group: pd.DataFrame) -> dict[str, object]:
        return {
            "rows": int(len(group)),
            "policy_beats_static_hold_rows": int((group["delta_total_return_vs_static_hold"] > 0).sum()),
            "policy_beats_optimizer_rows": int((group["delta_total_return_vs_optimizer"] > 0).sum()),
            "mean_policy_total_return": float(group["policy_total_return"].mean()),
            "mean_static_hold_total_return": float(group["static_hold_total_return"].mean()),
            "mean_optimizer_total_return": float(group["optimizer_total_return"].mean()),
            "mean_delta_total_return_vs_static_hold": float(group["delta_total_return_vs_static_hold"].mean()),
            "mean_delta_total_return_vs_optimizer": float(group["delta_total_return_vs_optimizer"].mean()),
        }

    negative_rows = detail_frame.loc[detail_frame["delta_total_return_vs_static_hold"] < 0].copy()
    weak_rows = detail_frame.sort_values("delta_total_return_vs_static_hold").head(min(8, len(detail_frame)))
    return {
        "overall": overall,
        "by_split": {str(name): _group_summary(group) for name, group in detail_frame.groupby("split")},
        "by_phase": {str(name): _group_summary(group) for name, group in detail_frame.groupby("phase")},
        "negative_row_diagnosis": {
            "negative_vs_static_hold_rows": int(len(negative_rows)),
            "negative_vs_optimizer_rows": int((detail_frame["delta_total_return_vs_optimizer"] < 0).sum()),
        },
        "weak_rows": {
            "worst_vs_static_hold": weak_rows.to_dict(orient="records"),
        },
    }


def build_policy_total_return_comparison_summary(candidate_detail: pd.DataFrame, current_detail: pd.DataFrame) -> dict[str, object]:
    key_columns = ["split", "seed", "fold", "phase", "phase_start", "phase_end"]
    merged = current_detail.merge(candidate_detail, on=key_columns, suffixes=("_current", "_main2"))
    merged["policy_total_return_diff"] = merged["policy_total_return_main2"] - merged["policy_total_return_current"]
    merged["delta_vs_static_hold_diff"] = merged["delta_total_return_vs_static_hold_main2"] - merged["delta_total_return_vs_static_hold_current"]
    merged["main2_beats_current"] = merged["policy_total_return_diff"] > 0.0
    merged["current_beats_main2"] = merged["policy_total_return_diff"] < 0.0

    return {
        "shared_rows": int(len(merged)),
        "mean_policy_total_return_diff": float(merged["policy_total_return_diff"].mean()),
        "chrono_mean_policy_total_return_diff": float(merged.loc[merged["split"] == "chrono", "policy_total_return_diff"].mean()) if (merged["split"] == "chrono").any() else None,
        "regime_mean_policy_total_return_diff": float(merged.loc[merged["split"] == "regime", "policy_total_return_diff"].mean()) if (merged["split"] == "regime").any() else None,
        "test_mean_policy_total_return_diff": float(merged.loc[merged["phase"] == "test", "policy_total_return_diff"].mean()),
        "validation_mean_policy_total_return_diff": float(merged.loc[merged["phase"] == "validation", "policy_total_return_diff"].mean()),
        "main2_win_rows": int(merged["main2_beats_current"].sum()),
        "current_win_rows": int(merged["current_beats_main2"].sum()),
        "ties": int((merged["policy_total_return_diff"] == 0.0).sum()),
        "largest_main2_wins": merged.sort_values("policy_total_return_diff", ascending=False).head(min(8, len(merged))).loc[
            :, key_columns + ["policy_total_return_current", "policy_total_return_main2", "policy_total_return_diff"]
        ].to_dict(orient="records"),
        "largest_main2_losses": merged.sort_values("policy_total_return_diff", ascending=True).head(min(8, len(merged))).loc[
            :, key_columns + ["policy_total_return_current", "policy_total_return_main2", "policy_total_return_diff"]
        ].to_dict(orient="records"),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the separate main2 SPY-only distributional RL case study.")
    parser.add_argument("--returns-csv", default="data/demo_sleeves.csv", help="CSV file containing the SPY benchmark return stream.")
    parser.add_argument("--benchmark-column", default="SPY_BENCHMARK", help="Column used as the sole SPY return source for main2.")
    parser.add_argument("--date-column", default=None, help="Optional date column when loading the returns CSV.")
    parser.add_argument("--start", default=None, help="Optional inclusive start date filter.")
    parser.add_argument("--end", default=None, help="Optional inclusive end date filter.")
    parser.add_argument("--splits", nargs="+", choices=["regime", "chrono"], default=["regime", "chrono"], help="Split families to evaluate.")
    parser.add_argument("--walk-forward-folds", type=int, default=3, help="Number of anchored walk-forward folds to evaluate per split family.")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27], help="Training seeds for the main2 study.")
    parser.add_argument("--lookback", type=int, default=5, help="Number of stacked feature rows exposed to the policy.")
    parser.add_argument("--action-bins", type=int, default=21, help="Number of discrete SPY target weights in the action grid.")
    parser.add_argument("--train-steps", type=int, default=3_000, help="Distributional RL training steps per fold.")
    parser.add_argument("--episode-length", type=int, default=252, help="Randomized training episode length.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=200_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--target-update-interval", type=int, default=500)
    parser.add_argument("--gradient-steps-per-update", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--quantiles", type=int, default=31)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=2_000)
    parser.add_argument("--validation-eval-interval", type=int, default=250)
    parser.add_argument("--validation-checkpoint-metric", type=str, default="total_return", choices=["total_return", "sharpe_ratio", "relative_total_return"])
    parser.add_argument("--validation-slices", type=int, default=1,
                        help="Number of contiguous sub-slices used for robust validation checkpoint scoring.")
    parser.add_argument("--validation-stability-penalty", type=float, default=0.0,
                        help="Penalty multiplier for validation score dispersion across slices.")
    parser.add_argument("--validation-crisis-drawdown-threshold", type=float, default=0.0,
                        help="Skip checkpoint scoring when SPY validation drawdown exceeds this value (e.g. -0.20). 0.0 = disabled.")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--action-smoothing", type=float, default=0.5)
    parser.add_argument("--no-trade-band", type=float, default=0.05)
    parser.add_argument("--initial-spy-weight", type=float, default=1.0)
    parser.add_argument("--min-spy-weight", type=float, default=0.65)
    parser.add_argument("--max-spy-weight", type=float, default=1.0)
    parser.add_argument("--trend-floor-min-spy-weight", type=float, default=0.9)
    parser.add_argument("--trend-floor-lookback", type=int, default=63)
    parser.add_argument("--trend-floor-return-threshold", type=float, default=0.08)
    parser.add_argument("--trend-floor-drawdown-threshold", type=float, default=-0.10)
    parser.add_argument("--participation-floor-min-spy-weight", type=float, default=0.85)
    parser.add_argument("--participation-floor-lookback", type=int, default=126)
    parser.add_argument("--participation-floor-return-threshold", type=float, default=0.02)
    parser.add_argument("--participation-floor-ma-gap-threshold", type=float, default=0.02)
    parser.add_argument("--participation-floor-drawdown-threshold", type=float, default=-0.05)
    parser.add_argument("--recovery-floor-min-spy-weight", type=float, default=0.8)
    parser.add_argument("--recovery-floor-long-lookback", type=int, default=126)
    parser.add_argument("--recovery-floor-drawdown-threshold", type=float, default=-0.10)
    parser.add_argument("--recovery-floor-short-lookback", type=int, default=21)
    parser.add_argument("--recovery-floor-return-threshold", type=float, default=0.01)
    parser.add_argument("--recovery-floor-ma-gap-threshold", type=float, default=0.0)
    parser.add_argument("--early-crack-floor-min-spy-weight", type=float, default=None)
    parser.add_argument("--early-crack-long-lookback", type=int, default=126)
    parser.add_argument("--early-crack-long-return-threshold", type=float, default=0.10)
    parser.add_argument("--early-crack-long-ma-gap-threshold", type=float, default=0.04)
    parser.add_argument("--early-crack-long-drawdown-threshold", type=float, default=-0.03)
    parser.add_argument("--early-crack-trend-ma-gap-threshold", type=float, default=0.02)
    parser.add_argument("--early-crack-short-lookback", type=int, default=21)
    parser.add_argument("--early-crack-short-return-threshold", type=float, default=0.0)
    parser.add_argument("--early-crack-short-ma-gap-threshold", type=float, default=0.0)
    parser.add_argument("--early-crack-short-drawdown-threshold", type=float, default=-0.02)
    parser.add_argument("--constructive-crack-cap-min-spy-weight", type=float, default=None)
    parser.add_argument("--constructive-crack-cap-recent-constructive-lookback", type=int, default=15)
    parser.add_argument("--constructive-crack-cap-long-lookback", type=int, default=126)
    parser.add_argument("--constructive-crack-cap-long-return-threshold", type=float, default=0.10)
    parser.add_argument("--constructive-crack-cap-long-ma-gap-threshold", type=float, default=0.03)
    parser.add_argument("--constructive-crack-cap-long-drawdown-threshold", type=float, default=-0.04)
    parser.add_argument("--constructive-crack-cap-short-lookback", type=int, default=21)
    parser.add_argument("--constructive-crack-cap-short-return-threshold", type=float, default=0.0)
    parser.add_argument("--constructive-crack-cap-short-ma-gap-threshold", type=float, default=0.0)
    parser.add_argument("--constructive-crack-cap-short-drawdown-threshold", type=float, default=-0.015)
    parser.add_argument("--constructive-crack-cap-current-trend-return-cap", type=float, default=0.05)
    parser.add_argument("--constructive-crack-cap-current-trend-ma-gap-cap", type=float, default=0.015)
    parser.add_argument("--constructive-crack-cap-current-long-return-min", type=float, default=0.09)
    parser.add_argument("--constructive-crack-cap-current-long-return-max", type=float, default=0.15)
    parser.add_argument("--constructive-crack-cap-sticky-days", type=int, default=0)
    parser.add_argument("--gain-reward", type=float, default=1.0)
    parser.add_argument("--loss-penalty", type=float, default=1.5)
    parser.add_argument("--loss-power", type=float, default=1.5)
    parser.add_argument("--optimizer-loss-penalty", type=float, default=8.0,
                        help="loss_penalty for the WealthFirst optimizer baseline (independent of DRL reward shaping).")
    parser.add_argument("--optimizer-lookback", type=int, default=21,
                        help="Estimation window in days for the WealthFirst optimizer baseline.")
    parser.add_argument("--benchmark-outperformance-reward", type=float, default=0.0)
    parser.add_argument("--benchmark-shortfall-penalty", type=float, default=0.0)
    parser.add_argument("--drawdown-penalty", type=float, default=0.0)
    parser.add_argument("--bear-oversample-weight", type=float, default=0.0,
                        help="Up-weight episode starts in drawdown periods. 0=uniform, higher=more bear sampling.")
    parser.add_argument("--ood-bear-drawdown-threshold", type=float, default=0.0,
                        help="252d drawdown level below which SPY weight is hard-capped. 0=disabled, e.g. -0.15.")
    parser.add_argument("--ood-bear-max-spy-weight", type=float, default=1.0,
                        help="Max SPY weight when OOD bear cap is active.")
    parser.add_argument("--bear-episode-spy-return-threshold", type=float, default=0.0,
                        help="End training episodes early when cumulative SPY return from start falls below this. 0=disabled, e.g. -0.15.")
    parser.add_argument("--no-disable-floors-in-training", dest="disable_floors_in_training", action="store_false",
                        help="Apply dynamic floors during training (default: floors are disabled during training and only applied at eval/test time).")
    parser.set_defaults(disable_floors_in_training=True)
    parser.add_argument("--output-prefix", default="artifacts/main2_spy_distributional", help="Output path prefix for detail, summary, and comparison artifacts.")
    parser.add_argument(
        "--compare-detail-csv",
        default="artifacts/ppo_frozen_live_baseline_earlyrisk_cap00215_b4_eval_detail.csv",
        help="Optional current-best eval detail CSV to compare raw policy total return against.",
    )
    parser.add_argument("--progress-log", default=None, help="Optional JSONL path used to emit structured run progress events.")
    return parser


def _config_from_args(args: argparse.Namespace) -> Main2Config:
    return Main2Config(
        lookback=args.lookback,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        action_smoothing=args.action_smoothing,
        no_trade_band=args.no_trade_band,
        initial_spy_weight=args.initial_spy_weight,
        min_spy_weight=args.min_spy_weight,
        max_spy_weight=args.max_spy_weight,
        trend_floor_min_spy_weight=args.trend_floor_min_spy_weight,
        trend_floor_lookback=args.trend_floor_lookback,
        trend_floor_return_threshold=args.trend_floor_return_threshold,
        trend_floor_drawdown_threshold=args.trend_floor_drawdown_threshold,
        participation_floor_min_spy_weight=args.participation_floor_min_spy_weight,
        participation_floor_lookback=args.participation_floor_lookback,
        participation_floor_return_threshold=args.participation_floor_return_threshold,
        participation_floor_ma_gap_threshold=args.participation_floor_ma_gap_threshold,
        participation_floor_drawdown_threshold=args.participation_floor_drawdown_threshold,
        recovery_floor_min_spy_weight=args.recovery_floor_min_spy_weight,
        recovery_floor_long_lookback=args.recovery_floor_long_lookback,
        recovery_floor_drawdown_threshold=args.recovery_floor_drawdown_threshold,
        recovery_floor_short_lookback=args.recovery_floor_short_lookback,
        recovery_floor_return_threshold=args.recovery_floor_return_threshold,
        recovery_floor_ma_gap_threshold=args.recovery_floor_ma_gap_threshold,
        early_crack_floor_min_spy_weight=args.early_crack_floor_min_spy_weight,
        early_crack_long_lookback=args.early_crack_long_lookback,
        early_crack_long_return_threshold=args.early_crack_long_return_threshold,
        early_crack_long_ma_gap_threshold=args.early_crack_long_ma_gap_threshold,
        early_crack_long_drawdown_threshold=args.early_crack_long_drawdown_threshold,
        early_crack_trend_ma_gap_threshold=args.early_crack_trend_ma_gap_threshold,
        early_crack_short_lookback=args.early_crack_short_lookback,
        early_crack_short_return_threshold=args.early_crack_short_return_threshold,
        early_crack_short_ma_gap_threshold=args.early_crack_short_ma_gap_threshold,
        early_crack_short_drawdown_threshold=args.early_crack_short_drawdown_threshold,
        constructive_crack_cap_min_spy_weight=args.constructive_crack_cap_min_spy_weight,
        constructive_crack_cap_recent_constructive_lookback=args.constructive_crack_cap_recent_constructive_lookback,
        constructive_crack_cap_long_lookback=args.constructive_crack_cap_long_lookback,
        constructive_crack_cap_long_return_threshold=args.constructive_crack_cap_long_return_threshold,
        constructive_crack_cap_long_ma_gap_threshold=args.constructive_crack_cap_long_ma_gap_threshold,
        constructive_crack_cap_long_drawdown_threshold=args.constructive_crack_cap_long_drawdown_threshold,
        constructive_crack_cap_short_lookback=args.constructive_crack_cap_short_lookback,
        constructive_crack_cap_short_return_threshold=args.constructive_crack_cap_short_return_threshold,
        constructive_crack_cap_short_ma_gap_threshold=args.constructive_crack_cap_short_ma_gap_threshold,
        constructive_crack_cap_short_drawdown_threshold=args.constructive_crack_cap_short_drawdown_threshold,
        constructive_crack_cap_current_trend_return_cap=args.constructive_crack_cap_current_trend_return_cap,
        constructive_crack_cap_current_trend_ma_gap_cap=args.constructive_crack_cap_current_trend_ma_gap_cap,
        constructive_crack_cap_current_long_return_min=args.constructive_crack_cap_current_long_return_min,
        constructive_crack_cap_current_long_return_max=args.constructive_crack_cap_current_long_return_max,
        constructive_crack_cap_sticky_days=args.constructive_crack_cap_sticky_days,
        action_bins=args.action_bins,
        train_steps=args.train_steps,
        episode_length=args.episode_length,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        learning_starts=args.learning_starts,
        target_update_interval=args.target_update_interval,
        gradient_steps_per_update=args.gradient_steps_per_update,
        hidden_size=args.hidden_size,
        quantiles=args.quantiles,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        validation_eval_interval=args.validation_eval_interval,
        validation_checkpoint_metric=args.validation_checkpoint_metric,
        validation_slices=args.validation_slices,
        validation_stability_penalty=args.validation_stability_penalty,
        validation_crisis_drawdown_threshold=args.validation_crisis_drawdown_threshold,
        loss_penalty=args.loss_penalty,
        loss_power=args.loss_power,
        gain_reward=args.gain_reward,
        optimizer_loss_penalty=args.optimizer_loss_penalty,
        optimizer_lookback=args.optimizer_lookback,
        benchmark_outperformance_reward=args.benchmark_outperformance_reward,
        benchmark_shortfall_penalty=args.benchmark_shortfall_penalty,
        drawdown_penalty=args.drawdown_penalty,
        bear_oversample_weight=args.bear_oversample_weight,
        ood_bear_drawdown_threshold=args.ood_bear_drawdown_threshold,
        ood_bear_max_spy_weight=args.ood_bear_max_spy_weight,
        bear_episode_spy_return_threshold=args.bear_episode_spy_return_threshold,
        disable_floors_in_training=args.disable_floors_in_training,
    )


def _split_family_windows(split_name: str, returns_frame: pd.DataFrame, spy_returns: pd.Series, config: Main2Config, validation_fraction: float, test_fraction: float, walk_forward_folds: int) -> list[SuggestedTimeSeriesSplit]:
    if split_name == "regime":
        suggested_split = suggest_regime_balanced_split(
            returns_frame,
            benchmark_returns=spy_returns,
            lookback=max(config.lookback, 20),
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
    else:
        suggested_split = chronological_train_validation_test_split(
            returns_frame,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            lookback=max(config.lookback, 20),
        )

    return generate_walk_forward_splits(
        returns_frame,
        benchmark_returns=spy_returns,
        lookback=max(config.lookback, 20),
        validation_rows=suggested_split.validation.rows,
        test_rows=suggested_split.test.rows,
        max_splits=walk_forward_folds,
    )


def main(argv: list[str] | None = None) -> int:
    _require_torch()
    assert torch is not None

    args = _build_parser().parse_args(argv)
    config = _config_from_args(args)

    spy_returns = load_spy_source_returns(
        returns_csv=args.returns_csv,
        benchmark_column=args.benchmark_column,
        date_column=args.date_column,
        start=args.start,
        end=args.end,
    )
    returns_frame = spy_returns.to_frame(name="SPY")
    features = build_main2_feature_frame(spy_returns)

    optimizer_baseline = run_rolling_backtest(
        returns_frame,
        lookback=config.optimizer_lookback,
        config=WealthFirstConfig(
            loss_penalty=config.optimizer_loss_penalty,
            gain_reward=config.gain_reward,
            loss_power=config.loss_power,
            transaction_cost_bps=config.transaction_cost_bps,
            slippage_bps=config.slippage_bps,
            include_cash=True,
        ),
        benchmark_symbol="SPY",
    )

    progress_log_path = Path(args.progress_log) if args.progress_log else None
    if progress_log_path is not None:
        if not progress_log_path.is_absolute():
            progress_log_path = PROJECT_ROOT / progress_log_path
        progress_log_path.parent.mkdir(parents=True, exist_ok=True)
        progress_log_path.write_text("", encoding="utf-8")

    split_windows_by_name = {
        split_name: _split_family_windows(
            split_name=split_name,
            returns_frame=returns_frame,
            spy_returns=spy_returns,
            config=config,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            walk_forward_folds=args.walk_forward_folds,
        )
        for split_name in args.splits
    }
    total_folds = sum(len(walk_forward_splits) * len(args.seeds) for walk_forward_splits in split_windows_by_name.values())

    def emit_progress(event_type: str, payload: dict[str, Any]) -> None:
        _append_main2_progress_event(progress_log_path, event_type, **payload)

    _append_main2_progress_event(
        progress_log_path,
        "run_started",
        output_prefix=args.output_prefix,
        splits=args.splits,
        seeds=args.seeds,
        total_folds=total_folds,
        train_steps=config.train_steps,
        validation_eval_interval=config.validation_eval_interval,
        compare_detail_csv=args.compare_detail_csv,
    )

    detail_rows: list[dict[str, float | int | str]] = []
    completed_folds = 0

    try:
        for split_name in args.splits:
            walk_forward_splits = split_windows_by_name[split_name]

            for seed in args.seeds:
                for fold_number, split in enumerate(walk_forward_splits, start=1):
                    fold_started_at = time.monotonic()
                    fold_context = {
                        "split": split_name,
                        "seed": seed,
                        "fold": f"fold_{fold_number:02d}",
                        "fold_number": fold_number,
                        "completed_folds": completed_folds,
                        "total_folds": total_folds,
                        "train_start": split.train.start_label,
                        "train_end": split.train.end_label,
                        "validation_start": split.validation.start_label,
                        "validation_end": split.validation.end_label,
                        "test_start": split.test.start_label,
                        "test_end": split.test.end_label,
                    }
                    _append_main2_progress_event(progress_log_path, "fold_started", **fold_context)

                normalized_features, _, _ = normalize_main2_features(features, train_end_index=split.train.end_index - 1)
                train_env = Main2SPYEnv(
                    normalized_features=normalized_features,
                    spy_returns=spy_returns,
                    start_index=split.train.start_index,
                    end_index=split.train.end_index - 1,
                    config=config,
                    random_episode_start=True,
                )
                validation_env = Main2SPYEnv(
                    normalized_features=normalized_features,
                    spy_returns=spy_returns,
                    start_index=split.validation.start_index,
                    end_index=split.validation.end_index - 1,
                    config=config,
                    random_episode_start=False,
                )
                test_env = Main2SPYEnv(
                    normalized_features=normalized_features,
                    spy_returns=spy_returns,
                    start_index=split.test.start_index,
                    end_index=split.test.end_index - 1,
                    config=config,
                    random_episode_start=False,
                )

                network = train_main2_quantile_agent(
                    train_env=train_env,
                    validation_env=validation_env,
                    config=config,
                    seed=seed,
                    progress_callback=emit_progress,
                    progress_context=fold_context,
                )
                device = torch.device("cpu")

                validation_metrics = evaluate_main2_policy(validation_env, network, device)
                test_metrics = evaluate_main2_policy(test_env, network, device)

                validation_static_hold = summarize_main2_static_hold(
                    spy_returns,
                    start_label=split.validation.start_label,
                    end_label=split.validation.end_label,
                )
                test_static_hold = summarize_main2_static_hold(
                    spy_returns,
                    start_label=split.test.start_label,
                    end_label=split.test.end_label,
                )
                validation_optimizer = summarize_main2_optimizer_baseline(
                    optimizer_baseline,
                    start_label=split.validation.start_label,
                    end_label=split.validation.end_label,
                    benchmark_total_return=validation_static_hold.total_return,
                )
                test_optimizer = summarize_main2_optimizer_baseline(
                    optimizer_baseline,
                    start_label=split.test.start_label,
                    end_label=split.test.end_label,
                    benchmark_total_return=test_static_hold.total_return,
                )

                detail_rows.append(
                    build_main2_detail_row(
                        split_name=split_name,
                        seed=seed,
                        fold_number=fold_number,
                        phase_name="validation",
                        phase_start=split.validation.start_label,
                        phase_end=split.validation.end_label,
                        policy_metrics=validation_metrics,
                        static_hold_metrics=validation_static_hold,
                        optimizer_metrics=validation_optimizer,
                    )
                )
                detail_rows.append(
                    build_main2_detail_row(
                        split_name=split_name,
                        seed=seed,
                        fold_number=fold_number,
                        phase_name="test",
                        phase_start=split.test.start_label,
                        phase_end=split.test.end_label,
                        policy_metrics=test_metrics,
                        static_hold_metrics=test_static_hold,
                        optimizer_metrics=test_optimizer,
                    )
                )
                completed_folds += 1
                fold_completed_context = {
                    **fold_context,
                    "completed_folds": completed_folds,
                }
                _append_main2_progress_event(
                    progress_log_path,
                    "fold_completed",
                    **fold_completed_context,
                    runtime_seconds=round(time.monotonic() - fold_started_at, 3),
                    validation_total_return=validation_metrics.total_return,
                    validation_delta_total_return_vs_static_hold=(
                        validation_metrics.total_return - validation_static_hold.total_return
                    ),
                    validation_delta_total_return_vs_optimizer=(
                        validation_metrics.total_return - validation_optimizer.total_return
                    ),
                    test_total_return=test_metrics.total_return,
                    test_delta_total_return_vs_static_hold=(
                        test_metrics.total_return - test_static_hold.total_return
                    ),
                    test_delta_total_return_vs_optimizer=(
                        test_metrics.total_return - test_optimizer.total_return
                    ),
                    validation_average_spy_weight=validation_metrics.average_spy_weight,
                    test_average_spy_weight=test_metrics.average_spy_weight,
                    validation_average_turnover=validation_metrics.average_turnover,
                    test_average_turnover=test_metrics.average_turnover,
                )
    except Exception as exc:
        _append_main2_progress_event(
            progress_log_path,
            "run_failed",
            message=str(exc),
            error_type=type(exc).__name__,
            completed_folds=completed_folds,
            total_folds=total_folds,
        )
        raise

    detail_frame = pd.DataFrame(detail_rows).sort_values(["split", "seed", "fold", "phase"]).reset_index(drop=True)
    summary = build_main2_eval_summary(detail_frame)

    output_prefix = Path(args.output_prefix)
    if not output_prefix.is_absolute():
        output_prefix = PROJECT_ROOT / output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    detail_path = output_prefix.with_name(f"{output_prefix.name}_detail.csv")
    summary_path = output_prefix.with_name(f"{output_prefix.name}_summary.json")
    detail_frame.round(6).to_csv(detail_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    comparison_summary_path = None
    compare_path = Path(args.compare_detail_csv)
    if not compare_path.is_absolute():
        compare_path = PROJECT_ROOT / compare_path
    if compare_path.exists():
        current_detail = pd.read_csv(compare_path)
        comparison_summary = build_policy_total_return_comparison_summary(detail_frame, current_detail)
        comparison_summary_path = output_prefix.with_name(f"{output_prefix.name}_vs_current_best_summary.json")
        comparison_summary_path.write_text(json.dumps(comparison_summary, indent=2), encoding="utf-8")

    _append_main2_progress_event(
        progress_log_path,
        "run_completed",
        completed_folds=completed_folds,
        total_folds=total_folds,
        detail_path=detail_path,
        summary_path=summary_path,
        comparison_summary_path=comparison_summary_path,
    )

    print(detail_path)
    print(summary_path)
    if comparison_summary_path is not None:
        print(comparison_summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())