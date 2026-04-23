from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised indirectly in tests when gymnasium is absent.
    gym = None
    spaces = None

from wealth_first.data import add_cash_sleeve
from wealth_first.optimizer import WealthFirstConfig, _coerce_returns_frame, _coerce_weight_vector, _project_to_weight_bounds, _resolve_weight_bounds
from wealth_first.rebalance import compute_execution_cost, compute_tradable_turnover

GYMNASIUM_AVAILABLE = gym is not None and spaces is not None
_GymEnvBase = gym.Env if gym is not None else object


def _coerce_benchmark_series(
    benchmark_returns: pd.Series | pd.DataFrame | np.ndarray | None,
    index: pd.Index,
) -> pd.Series | None:
    if benchmark_returns is None:
        return None

    if isinstance(benchmark_returns, pd.DataFrame):
        if benchmark_returns.shape[1] != 1:
            raise ValueError("Benchmark returns must be one-dimensional.")
        benchmark_series = benchmark_returns.iloc[:, 0]
    elif isinstance(benchmark_returns, pd.Series):
        benchmark_series = benchmark_returns
    else:
        benchmark_array = np.asarray(benchmark_returns, dtype=float)
        if benchmark_array.ndim != 1 or benchmark_array.shape[0] != len(index):
            raise ValueError("Benchmark returns must have the same number of observations as the environment index.")
        return pd.Series(benchmark_array, index=index, name="benchmark_return")

    aligned = pd.Series(benchmark_series, copy=True).astype(float).sort_index().reindex(index)
    if aligned.isna().any():
        raise ValueError("Benchmark returns could not be aligned to the environment index.")
    return aligned.rename("benchmark_return")


def _update_post_return_weights(weights: pd.Series, realized_returns: pd.Series, epsilon: float) -> pd.Series:
    period_values = weights * (1.0 + realized_returns)
    total_value = float(period_values.sum())
    if total_value <= epsilon:
        return weights.copy()
    return period_values / total_value


def _cap_weight_transition_turnover(
    target_weights: pd.Series,
    current_weights: pd.Series,
    turnover_limit: float,
    cash_symbol: str,
    epsilon: float,
    iterations: int = 24,
) -> pd.Series:
    current_turnover = compute_tradable_turnover(target_weights, current_weights, cash_symbol)
    if current_turnover <= turnover_limit + epsilon:
        return target_weights.copy()

    low = 0.0
    high = 1.0
    current_vector = current_weights.to_numpy(dtype=float)
    target_vector = target_weights.to_numpy(dtype=float)

    for _ in range(iterations):
        mid = 0.5 * (low + high)
        candidate_vector = current_vector + mid * (target_vector - current_vector)
        candidate_weights = pd.Series(candidate_vector, index=target_weights.index, name=target_weights.name)
        candidate_turnover = compute_tradable_turnover(candidate_weights, current_weights, cash_symbol)
        if candidate_turnover <= turnover_limit:
            low = mid
        else:
            high = mid

    capped_vector = current_vector + low * (target_vector - current_vector)
    return pd.Series(capped_vector, index=target_weights.index, name=target_weights.name)


def _cap_weight_transition_minimum_weight(
    target_weights: pd.Series,
    current_weights: pd.Series,
    symbol_index: int,
    minimum_weight: float,
    epsilon: float,
    iterations: int = 24,
) -> pd.Series:
    current_symbol_weight = float(current_weights.iloc[symbol_index])
    target_symbol_weight = float(target_weights.iloc[symbol_index])
    if target_symbol_weight >= minimum_weight - epsilon:
        return target_weights.copy()
    if current_symbol_weight <= minimum_weight + epsilon:
        return current_weights.copy()

    low = 0.0
    high = 1.0
    current_vector = current_weights.to_numpy(dtype=float)
    target_vector = target_weights.to_numpy(dtype=float)

    for _ in range(iterations):
        mid = 0.5 * (low + high)
        candidate_vector = current_vector + mid * (target_vector - current_vector)
        if float(candidate_vector[symbol_index]) >= minimum_weight:
            low = mid
        else:
            high = mid

    capped_vector = current_vector + low * (target_vector - current_vector)
    return pd.Series(capped_vector, index=target_weights.index, name=target_weights.name)


def _summarize_return_window(window_returns: np.ndarray) -> tuple[float, float, float]:
    cumulative_return = float(np.prod(1.0 + window_returns) - 1.0)
    volatility = float(np.std(window_returns, ddof=0))
    wealth_index = np.cumprod(1.0 + window_returns)
    drawdown = float(np.min(wealth_index / np.maximum.accumulate(wealth_index) - 1.0))
    return cumulative_return, volatility, drawdown


class WealthFirstEnv(_GymEnvBase):
    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: pd.DataFrame | pd.Series | np.ndarray,
        lookback: int = 20,
        config: WealthFirstConfig | None = None,
        benchmark_returns: pd.Series | pd.DataFrame | np.ndarray | None = None,
        benchmark_relative_observations: bool = False,
        benchmark_regime_observations: bool = False,
        benchmark_regime_summary_observations: bool | None = None,
        benchmark_regime_relative_cumulative_observations: bool | None = None,
        initial_weights: pd.Series | np.ndarray | None = None,
        episode_length: int | None = None,
        random_episode_start: bool = False,
        action_smoothing: float = 0.0,
        no_trade_band: float = 0.0,
        max_executed_rebalances: int | None = None,
        rebalance_cooldown_steps: int | None = None,
        early_rebalance_risk_penalty: float = 0.0,
        early_rebalance_risk_turnover_cap: float | None = None,
        early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold: float | None = None,
        early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold: float | None = None,
        early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight: float | None = None,
        early_rebalance_risk_turnover_cap_target_cash_min_threshold: float | None = None,
        early_rebalance_risk_turnover_cap_target_cash_max_threshold: float | None = None,
        early_rebalance_risk_turnover_cap_target_trend_min_threshold: float | None = None,
        early_rebalance_risk_turnover_cap_target_trend_max_threshold: float | None = None,
        early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold: float | None = None,
        early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold: float | None = None,
        early_rebalance_risk_turnover_cap_delta_cash_min_threshold: float | None = None,
        early_rebalance_risk_turnover_cap_delta_cash_max_threshold: float | None = None,
        early_rebalance_risk_turnover_cap_delta_trend_min_threshold: float | None = None,
        early_rebalance_risk_turnover_cap_delta_trend_max_threshold: float | None = None,
        early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold: float | None = None,
        early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold: float | None = None,
        early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold: float | None = None,
        early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold: float | None = None,
        early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio: float | None = None,
        early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio: float | None = None,
        early_rebalance_risk_turnover_cap_allow_nonincreasing_risk_symbol: bool = False,
        early_rebalance_risk_turnover_cap_use_penalty_state_filters: bool = False,
        early_rebalance_risk_turnover_cap_after: int | None = None,
        early_rebalance_risk_turnover_cap_before: int | None = None,
        early_rebalance_risk_turnover_cap_max_applications: int | None = None,
        early_rebalance_risk_turnover_cap_secondary_cap: float | None = None,
        early_rebalance_risk_turnover_cap_secondary_after_applications: int | None = None,
        early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold: float | None = None,
        early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio: float | None = None,
        early_rebalance_risk_deep_drawdown_turnover_cap: float | None = None,
        early_rebalance_risk_deep_drawdown_turnover_cap_after: int | None = None,
        early_rebalance_risk_deep_drawdown_turnover_cap_before: int | None = None,
        early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold: float | None = None,
        early_rebalance_risk_shallow_drawdown_turnover_cap: float | None = None,
        early_rebalance_risk_shallow_drawdown_turnover_cap_after: int | None = None,
        early_rebalance_risk_shallow_drawdown_turnover_cap_before: int | None = None,
        early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold: float | None = None,
        early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold: float | None = None,
        early_rebalance_risk_mean_reversion_turnover_cap: float | None = None,
        early_rebalance_risk_mean_reversion_action_smoothing: float | None = None,
        early_rebalance_risk_mean_reversion_turnover_cap_after: int | None = None,
        early_rebalance_risk_mean_reversion_turnover_cap_before: int | None = None,
        early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold: float | None = None,
        early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold: float | None = None,
        early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold: float | None = None,
        early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold: float | None = None,
        early_rebalance_risk_trend_turnover_cap: float | None = None,
        early_rebalance_risk_trend_turnover_cap_after: int | None = None,
        early_rebalance_risk_trend_turnover_cap_before: int | None = None,
        early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold: float | None = None,
        early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold: float | None = None,
        early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold: float | None = None,
        early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold: float | None = None,
        early_rebalance_risk_repeat_turnover_cap: float | None = None,
        early_rebalance_risk_repeat_action_smoothing: float | None = None,
        early_rebalance_risk_repeat_turnover_cap_after: int | None = None,
        early_rebalance_risk_repeat_turnover_cap_before: int | None = None,
        early_rebalance_risk_repeat_symbol: str | None = None,
        early_rebalance_risk_repeat_previous_cash_reduction_min: float | None = None,
        early_rebalance_risk_repeat_previous_symbol_increase_min: float | None = None,
        early_rebalance_risk_repeat_unrecovered_turnover_cap: float | None = None,
        early_rebalance_risk_repeat_unrecovered_turnover_cap_after: int | None = None,
        early_rebalance_risk_repeat_unrecovered_turnover_cap_before: int | None = None,
        early_rebalance_risk_repeat_unrecovered_symbol: str | None = None,
        early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min: float | None = None,
        early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min: float | None = None,
        early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery: float | None = None,
        early_rebalance_risk_cumulative_turnover_cap: float | None = None,
        early_rebalance_risk_cumulative_turnover_cap_after: int | None = None,
        early_rebalance_risk_cumulative_turnover_cap_before: int | None = None,
        early_rebalance_risk_cumulative_symbol: str | None = None,
        early_rebalance_risk_cumulative_cash_reduction_budget: float | None = None,
        early_rebalance_risk_cumulative_symbol_increase_budget: float | None = None,
        early_rebalance_risk_penalty_after: int | None = None,
        early_rebalance_risk_penalty_before: int | None = None,
        early_rebalance_risk_penalty_cash_max_threshold: float | None = None,
        early_rebalance_risk_penalty_symbol: str | None = None,
        early_rebalance_risk_penalty_symbol_min_weight: float | None = None,
        early_rebalance_risk_penalty_symbol_max_weight: float | None = None,
        early_rebalance_risk_penalty_benchmark_drawdown_min_threshold: float | None = None,
        early_rebalance_risk_penalty_benchmark_drawdown_max_threshold: float | None = None,
        early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio: float | None = None,
        early_benchmark_euphoria_penalty: float = 0.0,
        early_benchmark_euphoria_turnover_cap: float | None = None,
        early_benchmark_euphoria_before: int | None = None,
        early_benchmark_euphoria_benchmark_drawdown_min_threshold: float | None = None,
        early_benchmark_euphoria_symbol: str | None = None,
        late_rebalance_penalty: float = 0.0,
        late_rebalance_penalty_after: int | None = None,
        late_rebalance_gate_after: int | None = None,
        late_rebalance_gate_cash_threshold: float | None = None,
        late_rebalance_gate_target_cash_min_threshold: float | None = None,
        late_rebalance_gate_symbol: str | None = None,
        late_rebalance_gate_symbol_max_weight: float | None = None,
        late_rebalance_gate_cash_reduction_max: float | None = None,
        late_rebalance_gate_symbol_increase_max: float | None = None,
        late_defensive_posture_penalty: float = 0.0,
        late_defensive_posture_penalty_after: int | None = None,
        late_defensive_posture_penalty_cash_min_threshold: float | None = None,
        late_defensive_posture_penalty_symbol: str | None = None,
        late_defensive_posture_penalty_symbol_max_weight: float | None = None,
        late_trend_mean_reversion_conflict_penalty: float = 0.0,
        late_trend_mean_reversion_conflict_penalty_after: int | None = None,
        late_trend_mean_reversion_conflict_trend_symbol: str | None = None,
        late_trend_mean_reversion_conflict_trend_min_weight: float | None = None,
        late_trend_mean_reversion_conflict_mean_reversion_symbol: str | None = None,
        late_trend_mean_reversion_conflict_mean_reversion_min_weight: float | None = None,
        state_trend_preservation_symbol: str | None = None,
        state_trend_preservation_cash_max_threshold: float | None = None,
        state_trend_preservation_symbol_min_weight: float | None = None,
        state_trend_preservation_max_symbol_reduction: float | None = None,
        cash_weight_penalty: float = 0.0,
        cash_target_weight: float | None = None,
    ) -> None:
        if not GYMNASIUM_AVAILABLE:
            raise ImportError("gymnasium is required for WealthFirstEnv. Install the optional RL extra with `pip install -e '.[rl]'`.")
        if lookback <= 0:
            raise ValueError("lookback must be positive.")

        self.config = config or WealthFirstConfig()
        self.lookback = int(lookback)
        self.random_episode_start = bool(random_episode_start)
        if action_smoothing < 0.0 or action_smoothing > 1.0:
            raise ValueError("action_smoothing must be between 0 and 1.")
        if no_trade_band < 0.0:
            raise ValueError("no_trade_band must be non-negative.")
        if max_executed_rebalances is not None and max_executed_rebalances <= 0:
            raise ValueError("max_executed_rebalances must be positive when provided.")
        if rebalance_cooldown_steps is not None and rebalance_cooldown_steps <= 0:
            raise ValueError("rebalance_cooldown_steps must be positive when provided.")
        early_rebalance_risk_control_configured = any(
            value is not None
            for value in (
                early_rebalance_risk_penalty_after,
                early_rebalance_risk_penalty_before,
                early_rebalance_risk_penalty_cash_max_threshold,
                early_rebalance_risk_penalty_symbol,
                early_rebalance_risk_penalty_symbol_min_weight,
                early_rebalance_risk_penalty_symbol_max_weight,
                early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold,
                early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold,
                early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight,
                early_rebalance_risk_turnover_cap_target_cash_min_threshold,
                early_rebalance_risk_turnover_cap_target_cash_max_threshold,
                early_rebalance_risk_turnover_cap_target_trend_min_threshold,
                early_rebalance_risk_turnover_cap_target_trend_max_threshold,
                early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold,
                early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold,
                early_rebalance_risk_turnover_cap_delta_cash_min_threshold,
                early_rebalance_risk_turnover_cap_delta_cash_max_threshold,
                early_rebalance_risk_turnover_cap_delta_trend_min_threshold,
                early_rebalance_risk_turnover_cap_delta_trend_max_threshold,
                early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold,
                early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold,
                early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold,
                early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold,
                early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio,
                early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio,
                early_rebalance_risk_turnover_cap_after,
                early_rebalance_risk_turnover_cap_before,
                early_rebalance_risk_turnover_cap_max_applications,
                early_rebalance_risk_deep_drawdown_turnover_cap_after,
                early_rebalance_risk_deep_drawdown_turnover_cap_before,
                early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold,
                early_rebalance_risk_shallow_drawdown_turnover_cap_after,
                early_rebalance_risk_shallow_drawdown_turnover_cap_before,
                early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold,
                early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold,
                early_rebalance_risk_mean_reversion_turnover_cap_after,
                early_rebalance_risk_mean_reversion_turnover_cap_before,
                early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold,
                early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold,
                early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold,
                early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold,
                early_rebalance_risk_trend_turnover_cap_after,
                early_rebalance_risk_trend_turnover_cap_before,
                early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold,
                early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold,
                early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold,
                early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold,
                early_rebalance_risk_penalty_benchmark_drawdown_min_threshold,
                early_rebalance_risk_penalty_benchmark_drawdown_max_threshold,
                early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio,
            )
        ) or early_rebalance_risk_turnover_cap_use_penalty_state_filters
        if early_rebalance_risk_penalty < 0.0:
            raise ValueError("early_rebalance_risk_penalty must be non-negative.")
        if early_rebalance_risk_turnover_cap is not None and early_rebalance_risk_turnover_cap < 0.0:
            raise ValueError("early_rebalance_risk_turnover_cap must be non-negative when provided.")
        if early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold is not None and (
            early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold < -1.0
            or early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold > 0.0
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold must be between -1 and 0 when provided."
            )
        if (
            early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold is not None
            and early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold <= -1.0
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold must be greater than -1 when provided."
            )
        if early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight is not None and (
            early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight < 0.0
            or early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight > 1.0
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight must be between 0 and 1 when provided."
            )
        if early_rebalance_risk_turnover_cap_target_cash_min_threshold is not None and (
            early_rebalance_risk_turnover_cap_target_cash_min_threshold < 0.0
            or early_rebalance_risk_turnover_cap_target_cash_min_threshold > 1.0
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_target_cash_min_threshold must be between 0 and 1 when provided."
            )
        if early_rebalance_risk_turnover_cap_target_cash_max_threshold is not None and (
            early_rebalance_risk_turnover_cap_target_cash_max_threshold < 0.0
            or early_rebalance_risk_turnover_cap_target_cash_max_threshold > 1.0
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_target_cash_max_threshold must be between 0 and 1 when provided."
            )
        if early_rebalance_risk_turnover_cap_target_trend_min_threshold is not None and (
            early_rebalance_risk_turnover_cap_target_trend_min_threshold < 0.0
            or early_rebalance_risk_turnover_cap_target_trend_min_threshold > 1.0
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_target_trend_min_threshold must be between 0 and 1 when provided."
            )
        if early_rebalance_risk_turnover_cap_target_trend_max_threshold is not None and (
            early_rebalance_risk_turnover_cap_target_trend_max_threshold < 0.0
            or early_rebalance_risk_turnover_cap_target_trend_max_threshold > 1.0
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_target_trend_max_threshold must be between 0 and 1 when provided."
            )
        if early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold is not None and (
            early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold < 0.0
            or early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold > 1.0
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold must be between 0 and 1 when provided."
            )
        if early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold is not None and (
            early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold < 0.0
            or early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold > 1.0
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold must be between 0 and 1 when provided."
            )
        if (
            early_rebalance_risk_turnover_cap_target_trend_min_threshold is not None
            and early_rebalance_risk_turnover_cap_target_trend_max_threshold is not None
            and early_rebalance_risk_turnover_cap_target_trend_min_threshold
            > early_rebalance_risk_turnover_cap_target_trend_max_threshold
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_target_trend_min_threshold must be smaller than or equal to early_rebalance_risk_turnover_cap_target_trend_max_threshold when both are provided."
            )
        if (
            early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold is not None
            and early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold is not None
            and early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold
            > early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold must be smaller than or equal to early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold when both are provided."
            )
        if early_rebalance_risk_turnover_cap_delta_cash_min_threshold is not None and (
            early_rebalance_risk_turnover_cap_delta_cash_min_threshold < -1.0
            or early_rebalance_risk_turnover_cap_delta_cash_min_threshold > 1.0
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_delta_cash_min_threshold must be between -1 and 1 when provided."
            )
        if early_rebalance_risk_turnover_cap_delta_cash_max_threshold is not None and (
            early_rebalance_risk_turnover_cap_delta_cash_max_threshold < -1.0
            or early_rebalance_risk_turnover_cap_delta_cash_max_threshold > 1.0
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_delta_cash_max_threshold must be between -1 and 1 when provided."
            )
        if early_rebalance_risk_turnover_cap_delta_trend_min_threshold is not None and (
            early_rebalance_risk_turnover_cap_delta_trend_min_threshold < -1.0
            or early_rebalance_risk_turnover_cap_delta_trend_min_threshold > 1.0
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_delta_trend_min_threshold must be between -1 and 1 when provided."
            )
        if early_rebalance_risk_turnover_cap_delta_trend_max_threshold is not None and (
            early_rebalance_risk_turnover_cap_delta_trend_max_threshold < -1.0
            or early_rebalance_risk_turnover_cap_delta_trend_max_threshold > 1.0
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_delta_trend_max_threshold must be between -1 and 1 when provided."
            )
        if early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold is not None and (
            early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold < -1.0
            or early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold > 1.0
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold must be between -1 and 1 when provided."
            )
        if early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold is not None and (
            early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold < -1.0
            or early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold > 1.0
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold must be between -1 and 1 when provided."
            )
        if (
            early_rebalance_risk_turnover_cap_delta_trend_min_threshold is not None
            and early_rebalance_risk_turnover_cap_delta_trend_max_threshold is not None
            and early_rebalance_risk_turnover_cap_delta_trend_min_threshold
            > early_rebalance_risk_turnover_cap_delta_trend_max_threshold
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_delta_trend_min_threshold must be smaller than or equal to early_rebalance_risk_turnover_cap_delta_trend_max_threshold when both are provided."
            )
        if (
            early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold is not None
            and early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold is not None
            and early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold
            > early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold must be smaller than or equal to early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold when both are provided."
            )
        if (
            early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold is not None
            and early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold < 0.0
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold must be non-negative when provided."
            )
        if (
            early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold is not None
            and early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold < 0.0
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold must be non-negative when provided."
            )
        if (
            early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold is not None
            and early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold is not None
            and early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold
            > early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold must be smaller than or equal to early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold when both are provided."
            )
        if (
            early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio is not None
            and early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio <= 0.0
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio must be positive when provided."
            )
        if (
            early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio is not None
            and early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio <= 0.0
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio must be positive when provided."
            )
        if (
            early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio is not None
            and early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio is not None
            and early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio
            > early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio must be smaller than or equal to early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio when both are provided."
            )
        if early_rebalance_risk_turnover_cap_after is not None and early_rebalance_risk_turnover_cap_after < 0:
            raise ValueError("early_rebalance_risk_turnover_cap_after must be non-negative when provided.")
        if early_rebalance_risk_turnover_cap_before is not None and early_rebalance_risk_turnover_cap_before < 0:
            raise ValueError("early_rebalance_risk_turnover_cap_before must be non-negative when provided.")
        if (
            early_rebalance_risk_turnover_cap_after is not None
            and early_rebalance_risk_turnover_cap_before is not None
            and early_rebalance_risk_turnover_cap_after >= early_rebalance_risk_turnover_cap_before
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_after must be smaller than early_rebalance_risk_turnover_cap_before when both are provided."
            )
        if (
            early_rebalance_risk_turnover_cap_max_applications is not None
            and early_rebalance_risk_turnover_cap_max_applications < 0
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_max_applications must be non-negative when provided."
            )
        if (
            early_rebalance_risk_turnover_cap_max_applications is not None
            and early_rebalance_risk_turnover_cap is None
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_max_applications requires early_rebalance_risk_turnover_cap when configured."
            )
        if (
            early_rebalance_risk_turnover_cap_secondary_cap is not None
            and early_rebalance_risk_turnover_cap_secondary_cap < 0.0
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_secondary_cap must be non-negative when provided."
            )
        if (
            early_rebalance_risk_turnover_cap_secondary_after_applications is not None
            and early_rebalance_risk_turnover_cap_secondary_after_applications < 0
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_secondary_after_applications must be non-negative when provided."
            )
        if (
            early_rebalance_risk_turnover_cap_secondary_cap is not None
            and early_rebalance_risk_turnover_cap is None
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_secondary_cap requires early_rebalance_risk_turnover_cap when configured."
            )
        if (
            early_rebalance_risk_turnover_cap_secondary_after_applications is not None
            and early_rebalance_risk_turnover_cap is None
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_secondary_after_applications requires early_rebalance_risk_turnover_cap when configured."
            )
        if (
            early_rebalance_risk_turnover_cap_secondary_cap is None
            and early_rebalance_risk_turnover_cap_secondary_after_applications is not None
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_secondary_after_applications requires early_rebalance_risk_turnover_cap_secondary_cap when configured."
            )
        if (
            early_rebalance_risk_turnover_cap_secondary_cap is not None
            and early_rebalance_risk_turnover_cap_secondary_after_applications is None
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_secondary_cap requires early_rebalance_risk_turnover_cap_secondary_after_applications when configured."
            )
        if (
            early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold is not None
            and early_rebalance_risk_turnover_cap is None
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold requires early_rebalance_risk_turnover_cap when configured."
            )
        if (
            early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio is not None
            and early_rebalance_risk_turnover_cap is None
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio requires early_rebalance_risk_turnover_cap when configured."
            )
        if (
            early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold is not None
            and early_rebalance_risk_turnover_cap_secondary_cap is None
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold requires early_rebalance_risk_turnover_cap_secondary_cap when configured."
            )
        if (
            early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio is not None
            and early_rebalance_risk_turnover_cap_secondary_cap is None
        ):
            raise ValueError(
                "early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio requires early_rebalance_risk_turnover_cap_secondary_cap when configured."
            )
        if (
            early_rebalance_risk_deep_drawdown_turnover_cap is not None
            and early_rebalance_risk_deep_drawdown_turnover_cap < 0.0
        ):
            raise ValueError("early_rebalance_risk_deep_drawdown_turnover_cap must be non-negative when provided.")
        if (
            early_rebalance_risk_deep_drawdown_turnover_cap_after is not None
            and early_rebalance_risk_deep_drawdown_turnover_cap_after < 0
        ):
            raise ValueError(
                "early_rebalance_risk_deep_drawdown_turnover_cap_after must be non-negative when provided."
            )
        if (
            early_rebalance_risk_deep_drawdown_turnover_cap_before is not None
            and early_rebalance_risk_deep_drawdown_turnover_cap_before < 0
        ):
            raise ValueError(
                "early_rebalance_risk_deep_drawdown_turnover_cap_before must be non-negative when provided."
            )
        if (
            early_rebalance_risk_deep_drawdown_turnover_cap_after is not None
            and early_rebalance_risk_deep_drawdown_turnover_cap_before is not None
            and early_rebalance_risk_deep_drawdown_turnover_cap_after
            >= early_rebalance_risk_deep_drawdown_turnover_cap_before
        ):
            raise ValueError(
                "early_rebalance_risk_deep_drawdown_turnover_cap_after must be smaller than early_rebalance_risk_deep_drawdown_turnover_cap_before when both are provided."
            )
        if early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold is not None and (
            early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold < -1.0
            or early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold > 0.0
        ):
            raise ValueError(
                "early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold must be between -1 and 0 when provided."
            )
        if (
            early_rebalance_risk_deep_drawdown_turnover_cap is not None
            or early_rebalance_risk_deep_drawdown_turnover_cap_after is not None
            or early_rebalance_risk_deep_drawdown_turnover_cap_before is not None
            or early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold is not None
        ) and (
            early_rebalance_risk_deep_drawdown_turnover_cap is None
            or early_rebalance_risk_deep_drawdown_turnover_cap_before is None
            or early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold is None
        ):
            raise ValueError(
                "early rebalance risk deep-drawdown turnover cap requires cap, before, and benchmark drawdown max threshold when configured."
            )
        if (
            early_rebalance_risk_shallow_drawdown_turnover_cap is not None
            and early_rebalance_risk_shallow_drawdown_turnover_cap < 0.0
        ):
            raise ValueError("early_rebalance_risk_shallow_drawdown_turnover_cap must be non-negative when provided.")
        if (
            early_rebalance_risk_shallow_drawdown_turnover_cap_after is not None
            and early_rebalance_risk_shallow_drawdown_turnover_cap_after < 0
        ):
            raise ValueError(
                "early_rebalance_risk_shallow_drawdown_turnover_cap_after must be non-negative when provided."
            )
        if (
            early_rebalance_risk_shallow_drawdown_turnover_cap_before is not None
            and early_rebalance_risk_shallow_drawdown_turnover_cap_before < 0
        ):
            raise ValueError(
                "early_rebalance_risk_shallow_drawdown_turnover_cap_before must be non-negative when provided."
            )
        if (
            early_rebalance_risk_shallow_drawdown_turnover_cap_after is not None
            and early_rebalance_risk_shallow_drawdown_turnover_cap_before is not None
            and early_rebalance_risk_shallow_drawdown_turnover_cap_after
            >= early_rebalance_risk_shallow_drawdown_turnover_cap_before
        ):
            raise ValueError(
                "early_rebalance_risk_shallow_drawdown_turnover_cap_after must be smaller than early_rebalance_risk_shallow_drawdown_turnover_cap_before when both are provided."
            )
        if early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold is not None and (
            early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold < 0.0
            or early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold > 1.0
        ):
            raise ValueError(
                "early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold must be between 0 and 1 when provided."
            )
        if early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold is not None and (
            early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold < -1.0
            or early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold > 0.0
        ):
            raise ValueError(
                "early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold must be between -1 and 0 when provided."
            )
        if (
            early_rebalance_risk_shallow_drawdown_turnover_cap is not None
            or early_rebalance_risk_shallow_drawdown_turnover_cap_after is not None
            or early_rebalance_risk_shallow_drawdown_turnover_cap_before is not None
            or early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold is not None
            or early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold is not None
        ) and (
            early_rebalance_risk_shallow_drawdown_turnover_cap is None
            or early_rebalance_risk_shallow_drawdown_turnover_cap_before is None
            or early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold is None
            or early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold is None
        ):
            raise ValueError(
                "early rebalance risk shallow-drawdown turnover cap requires cap, before, cash max threshold, and benchmark drawdown min threshold when configured."
            )
        if (
            early_rebalance_risk_mean_reversion_turnover_cap is not None
            and early_rebalance_risk_mean_reversion_turnover_cap < 0.0
        ):
            raise ValueError("early_rebalance_risk_mean_reversion_turnover_cap must be non-negative when provided.")
        if (
            early_rebalance_risk_mean_reversion_action_smoothing is not None
            and (
                early_rebalance_risk_mean_reversion_action_smoothing < 0.0
                or early_rebalance_risk_mean_reversion_action_smoothing > 1.0
            )
        ):
            raise ValueError(
                "early_rebalance_risk_mean_reversion_action_smoothing must be between 0 and 1 when provided."
            )
        if (
            early_rebalance_risk_mean_reversion_turnover_cap_after is not None
            and early_rebalance_risk_mean_reversion_turnover_cap_after < 0
        ):
            raise ValueError(
                "early_rebalance_risk_mean_reversion_turnover_cap_after must be non-negative when provided."
            )
        if (
            early_rebalance_risk_mean_reversion_turnover_cap_before is not None
            and early_rebalance_risk_mean_reversion_turnover_cap_before < 0
        ):
            raise ValueError(
                "early_rebalance_risk_mean_reversion_turnover_cap_before must be non-negative when provided."
            )
        if (
            early_rebalance_risk_mean_reversion_turnover_cap_after is not None
            and early_rebalance_risk_mean_reversion_turnover_cap_before is not None
            and early_rebalance_risk_mean_reversion_turnover_cap_after
            >= early_rebalance_risk_mean_reversion_turnover_cap_before
        ):
            raise ValueError(
                "early_rebalance_risk_mean_reversion_turnover_cap_after must be smaller than early_rebalance_risk_mean_reversion_turnover_cap_before when both are provided."
            )
        if (
            early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold is not None
            and early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold <= -1.0
        ):
            raise ValueError(
                "early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold must be greater than -1 when provided."
            )
        if early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold is not None and (
            early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold < 0.0
            or early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold > 1.0
        ):
            raise ValueError(
                "early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold must be between 0 and 1 when provided."
            )
        if early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold is not None and (
            early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold < 0.0
            or early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold > 1.0
        ):
            raise ValueError(
                "early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold must be between 0 and 1 when provided."
            )
        if early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold is not None and (
            early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold < -1.0
            or early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold > 1.0
        ):
            raise ValueError(
                "early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold must be between -1 and 1 when provided."
            )
        if (
            early_rebalance_risk_mean_reversion_turnover_cap is not None
            or early_rebalance_risk_mean_reversion_action_smoothing is not None
            or early_rebalance_risk_mean_reversion_turnover_cap_after is not None
            or early_rebalance_risk_mean_reversion_turnover_cap_before is not None
            or early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold is not None
            or early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold is not None
            or early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold is not None
        ) and (
            (
                early_rebalance_risk_mean_reversion_turnover_cap is None
                and early_rebalance_risk_mean_reversion_action_smoothing is None
            )
            or early_rebalance_risk_mean_reversion_turnover_cap_before is None
            or (
                early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold is None
                and early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold is None
                and early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold is None
            )
        ):
            raise ValueError(
                "early rebalance risk mean-reversion control requires turnover cap or action smoothing, before, and at least one mean-reversion threshold when configured."
            )
        if (
            early_rebalance_risk_trend_turnover_cap is not None
            and early_rebalance_risk_trend_turnover_cap < 0.0
        ):
            raise ValueError("early_rebalance_risk_trend_turnover_cap must be non-negative when provided.")
        if (
            early_rebalance_risk_trend_turnover_cap_after is not None
            and early_rebalance_risk_trend_turnover_cap_after < 0
        ):
            raise ValueError(
                "early_rebalance_risk_trend_turnover_cap_after must be non-negative when provided."
            )
        if (
            early_rebalance_risk_trend_turnover_cap_before is not None
            and early_rebalance_risk_trend_turnover_cap_before < 0
        ):
            raise ValueError(
                "early_rebalance_risk_trend_turnover_cap_before must be non-negative when provided."
            )
        if (
            early_rebalance_risk_trend_turnover_cap_after is not None
            and early_rebalance_risk_trend_turnover_cap_before is not None
            and early_rebalance_risk_trend_turnover_cap_after >= early_rebalance_risk_trend_turnover_cap_before
        ):
            raise ValueError(
                "early_rebalance_risk_trend_turnover_cap_after must be smaller than early_rebalance_risk_trend_turnover_cap_before when both are provided."
            )
        if (
            early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold is not None
            and early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold <= -1.0
        ):
            raise ValueError(
                "early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold must be greater than -1 when provided."
            )
        if early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold is not None and (
            early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold < 0.0
            or early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold > 1.0
        ):
            raise ValueError(
                "early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold must be between 0 and 1 when provided."
            )
        if early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold is not None and (
            early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold < 0.0
            or early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold > 1.0
        ):
            raise ValueError(
                "early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold must be between 0 and 1 when provided."
            )
        if early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold is not None and (
            early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold < -1.0
            or early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold > 1.0
        ):
            raise ValueError(
                "early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold must be between -1 and 1 when provided."
            )
        if (
            early_rebalance_risk_trend_turnover_cap is not None
            or early_rebalance_risk_trend_turnover_cap_after is not None
            or early_rebalance_risk_trend_turnover_cap_before is not None
            or early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold is not None
            or early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold is not None
            or early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold is not None
        ) and (
            early_rebalance_risk_trend_turnover_cap is None
            or early_rebalance_risk_trend_turnover_cap_before is None
            or (
                early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold is None
                and early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold is None
                and early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold is None
            )
        ):
            raise ValueError(
                "early rebalance risk trend turnover cap requires cap, before, and at least one trend threshold when configured."
            )
        early_rebalance_risk_repeat_control_configured = any(
            value is not None
            for value in (
                early_rebalance_risk_repeat_turnover_cap_after,
                early_rebalance_risk_repeat_turnover_cap_before,
                early_rebalance_risk_repeat_symbol,
                early_rebalance_risk_repeat_previous_cash_reduction_min,
                early_rebalance_risk_repeat_previous_symbol_increase_min,
            )
        )
        if (
            early_rebalance_risk_repeat_turnover_cap is not None
            and early_rebalance_risk_repeat_turnover_cap < 0.0
        ):
            raise ValueError("early_rebalance_risk_repeat_turnover_cap must be non-negative when provided.")
        if (
            early_rebalance_risk_repeat_action_smoothing is not None
            and (
                early_rebalance_risk_repeat_action_smoothing < 0.0
                or early_rebalance_risk_repeat_action_smoothing > 1.0
            )
        ):
            raise ValueError(
                "early_rebalance_risk_repeat_action_smoothing must be between 0 and 1 when provided."
            )
        if (
            early_rebalance_risk_repeat_turnover_cap_after is not None
            and early_rebalance_risk_repeat_turnover_cap_after < 0
        ):
            raise ValueError(
                "early_rebalance_risk_repeat_turnover_cap_after must be non-negative when provided."
            )
        if (
            early_rebalance_risk_repeat_turnover_cap_before is not None
            and early_rebalance_risk_repeat_turnover_cap_before < 0
        ):
            raise ValueError(
                "early_rebalance_risk_repeat_turnover_cap_before must be non-negative when provided."
            )
        if (
            early_rebalance_risk_repeat_turnover_cap_after is not None
            and early_rebalance_risk_repeat_turnover_cap_before is not None
            and early_rebalance_risk_repeat_turnover_cap_after >= early_rebalance_risk_repeat_turnover_cap_before
        ):
            raise ValueError(
                "early_rebalance_risk_repeat_turnover_cap_after must be smaller than early_rebalance_risk_repeat_turnover_cap_before when both are provided."
            )
        if (
            early_rebalance_risk_repeat_previous_cash_reduction_min is not None
            and (
                early_rebalance_risk_repeat_previous_cash_reduction_min < 0.0
                or early_rebalance_risk_repeat_previous_cash_reduction_min > 1.0
            )
        ):
            raise ValueError(
                "early_rebalance_risk_repeat_previous_cash_reduction_min must be between 0 and 1 when provided."
            )
        if (
            early_rebalance_risk_repeat_previous_symbol_increase_min is not None
            and (
                early_rebalance_risk_repeat_previous_symbol_increase_min < 0.0
                or early_rebalance_risk_repeat_previous_symbol_increase_min > 1.0
            )
        ):
            raise ValueError(
                "early_rebalance_risk_repeat_previous_symbol_increase_min must be between 0 and 1 when provided."
            )
        if (
            early_rebalance_risk_repeat_turnover_cap is not None
            or early_rebalance_risk_repeat_action_smoothing is not None
            or early_rebalance_risk_repeat_control_configured
        ) and (
            early_rebalance_risk_repeat_turnover_cap_before is None
            or early_rebalance_risk_repeat_symbol is None
            or early_rebalance_risk_repeat_previous_cash_reduction_min is None
            or early_rebalance_risk_repeat_previous_symbol_increase_min is None
            or (
                early_rebalance_risk_repeat_turnover_cap is None
                and early_rebalance_risk_repeat_action_smoothing is None
            )
        ):
            raise ValueError(
                "early rebalance risk repeat control requires turnover cap or action smoothing, before, symbol, and prior cash/symbol thresholds when configured."
            )
        early_rebalance_risk_repeat_unrecovered_control_configured = any(
            value is not None
            for value in (
                early_rebalance_risk_repeat_unrecovered_turnover_cap_after,
                early_rebalance_risk_repeat_unrecovered_turnover_cap_before,
                early_rebalance_risk_repeat_unrecovered_symbol,
                early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min,
                early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min,
                early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery,
            )
        )
        if (
            early_rebalance_risk_repeat_unrecovered_turnover_cap is not None
            and early_rebalance_risk_repeat_unrecovered_turnover_cap < 0.0
        ):
            raise ValueError(
                "early_rebalance_risk_repeat_unrecovered_turnover_cap must be non-negative when provided."
            )
        if (
            early_rebalance_risk_repeat_unrecovered_turnover_cap_after is not None
            and early_rebalance_risk_repeat_unrecovered_turnover_cap_after < 0
        ):
            raise ValueError(
                "early_rebalance_risk_repeat_unrecovered_turnover_cap_after must be non-negative when provided."
            )
        if (
            early_rebalance_risk_repeat_unrecovered_turnover_cap_before is not None
            and early_rebalance_risk_repeat_unrecovered_turnover_cap_before < 0
        ):
            raise ValueError(
                "early_rebalance_risk_repeat_unrecovered_turnover_cap_before must be non-negative when provided."
            )
        if (
            early_rebalance_risk_repeat_unrecovered_turnover_cap_after is not None
            and early_rebalance_risk_repeat_unrecovered_turnover_cap_before is not None
            and early_rebalance_risk_repeat_unrecovered_turnover_cap_after
            >= early_rebalance_risk_repeat_unrecovered_turnover_cap_before
        ):
            raise ValueError(
                "early_rebalance_risk_repeat_unrecovered_turnover_cap_after must be smaller than early_rebalance_risk_repeat_unrecovered_turnover_cap_before when both are provided."
            )
        if (
            early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min is not None
            and (
                early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min < 0.0
                or early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min > 1.0
            )
        ):
            raise ValueError(
                "early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min must be between 0 and 1 when provided."
            )
        if (
            early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min is not None
            and (
                early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min < 0.0
                or early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min > 1.0
            )
        ):
            raise ValueError(
                "early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min must be between 0 and 1 when provided."
            )
        if (
            early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery is not None
            and early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery < 0.0
        ):
            raise ValueError(
                "early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery must be non-negative when provided."
            )
        if (
            early_rebalance_risk_repeat_unrecovered_turnover_cap is not None
            or early_rebalance_risk_repeat_unrecovered_control_configured
        ) and (
            early_rebalance_risk_repeat_unrecovered_turnover_cap is None
            or early_rebalance_risk_repeat_unrecovered_turnover_cap_before is None
            or early_rebalance_risk_repeat_unrecovered_symbol is None
            or early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min is None
            or early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min is None
            or early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery is None
        ):
            raise ValueError(
                "early rebalance risk repeat unrecovered turnover cap requires cap, before, symbol, prior cash/symbol thresholds, and a minimum relative wealth recovery when configured."
            )
        if (
            (early_rebalance_risk_repeat_turnover_cap is not None or early_rebalance_risk_repeat_control_configured)
            and (
                early_rebalance_risk_repeat_unrecovered_turnover_cap is not None
                or early_rebalance_risk_repeat_unrecovered_control_configured
            )
            and early_rebalance_risk_repeat_symbol is not None
            and early_rebalance_risk_repeat_unrecovered_symbol is not None
            and early_rebalance_risk_repeat_symbol != early_rebalance_risk_repeat_unrecovered_symbol
        ):
            raise ValueError(
                "early rebalance risk repeat turnover cap and repeat unrecovered turnover cap must use the same symbol when both are configured."
            )
        early_rebalance_risk_cumulative_control_configured = any(
            value is not None
            for value in (
                early_rebalance_risk_cumulative_turnover_cap_after,
                early_rebalance_risk_cumulative_turnover_cap_before,
                early_rebalance_risk_cumulative_symbol,
                early_rebalance_risk_cumulative_cash_reduction_budget,
                early_rebalance_risk_cumulative_symbol_increase_budget,
            )
        )
        if (
            early_rebalance_risk_cumulative_turnover_cap is not None
            and early_rebalance_risk_cumulative_turnover_cap < 0.0
        ):
            raise ValueError("early_rebalance_risk_cumulative_turnover_cap must be non-negative when provided.")
        if (
            early_rebalance_risk_cumulative_turnover_cap_after is not None
            and early_rebalance_risk_cumulative_turnover_cap_after < 0
        ):
            raise ValueError(
                "early_rebalance_risk_cumulative_turnover_cap_after must be non-negative when provided."
            )
        if (
            early_rebalance_risk_cumulative_turnover_cap_before is not None
            and early_rebalance_risk_cumulative_turnover_cap_before < 0
        ):
            raise ValueError(
                "early_rebalance_risk_cumulative_turnover_cap_before must be non-negative when provided."
            )
        if (
            early_rebalance_risk_cumulative_turnover_cap_after is not None
            and early_rebalance_risk_cumulative_turnover_cap_before is not None
            and early_rebalance_risk_cumulative_turnover_cap_after
            >= early_rebalance_risk_cumulative_turnover_cap_before
        ):
            raise ValueError(
                "early_rebalance_risk_cumulative_turnover_cap_after must be smaller than early_rebalance_risk_cumulative_turnover_cap_before when both are provided."
            )
        if (
            early_rebalance_risk_cumulative_cash_reduction_budget is not None
            and (
                early_rebalance_risk_cumulative_cash_reduction_budget < 0.0
                or early_rebalance_risk_cumulative_cash_reduction_budget > 1.0
            )
        ):
            raise ValueError(
                "early_rebalance_risk_cumulative_cash_reduction_budget must be between 0 and 1 when provided."
            )
        if (
            early_rebalance_risk_cumulative_symbol_increase_budget is not None
            and (
                early_rebalance_risk_cumulative_symbol_increase_budget < 0.0
                or early_rebalance_risk_cumulative_symbol_increase_budget > 1.0
            )
        ):
            raise ValueError(
                "early_rebalance_risk_cumulative_symbol_increase_budget must be between 0 and 1 when provided."
            )
        if (
            early_rebalance_risk_cumulative_turnover_cap is not None
            or early_rebalance_risk_cumulative_control_configured
        ) and (
            early_rebalance_risk_cumulative_turnover_cap is None
            or early_rebalance_risk_cumulative_turnover_cap_before is None
            or early_rebalance_risk_cumulative_symbol is None
            or (
                early_rebalance_risk_cumulative_cash_reduction_budget is None
                and early_rebalance_risk_cumulative_symbol_increase_budget is None
            )
        ):
            raise ValueError(
                "early rebalance risk cumulative turnover cap requires cap, before, symbol, and at least one cumulative budget when configured."
            )
        if early_rebalance_risk_penalty_after is not None and early_rebalance_risk_penalty_after < 0:
            raise ValueError("early_rebalance_risk_penalty_after must be non-negative when provided.")
        if early_rebalance_risk_penalty_before is not None and early_rebalance_risk_penalty_before < 0:
            raise ValueError("early_rebalance_risk_penalty_before must be non-negative when provided.")
        if (
            early_rebalance_risk_penalty_after is not None
            and early_rebalance_risk_penalty_before is not None
            and early_rebalance_risk_penalty_after >= early_rebalance_risk_penalty_before
        ):
            raise ValueError(
                "early_rebalance_risk_penalty_after must be smaller than early_rebalance_risk_penalty_before when both are provided."
            )
        if early_rebalance_risk_penalty_cash_max_threshold is not None and (
            early_rebalance_risk_penalty_cash_max_threshold < 0.0 or early_rebalance_risk_penalty_cash_max_threshold > 1.0
        ):
            raise ValueError("early_rebalance_risk_penalty_cash_max_threshold must be between 0 and 1 when provided.")
        if early_rebalance_risk_penalty_symbol_min_weight is not None and (
            early_rebalance_risk_penalty_symbol_min_weight < 0.0 or early_rebalance_risk_penalty_symbol_min_weight > 1.0
        ):
            raise ValueError("early_rebalance_risk_penalty_symbol_min_weight must be between 0 and 1 when provided.")
        if early_rebalance_risk_penalty_symbol_max_weight is not None and (
            early_rebalance_risk_penalty_symbol_max_weight < 0.0 or early_rebalance_risk_penalty_symbol_max_weight > 1.0
        ):
            raise ValueError("early_rebalance_risk_penalty_symbol_max_weight must be between 0 and 1 when provided.")
        if early_rebalance_risk_penalty_benchmark_drawdown_min_threshold is not None and (
            early_rebalance_risk_penalty_benchmark_drawdown_min_threshold < -1.0
            or early_rebalance_risk_penalty_benchmark_drawdown_min_threshold > 0.0
        ):
            raise ValueError(
                "early_rebalance_risk_penalty_benchmark_drawdown_min_threshold must be between -1 and 0 when provided."
            )
        if early_rebalance_risk_penalty_benchmark_drawdown_max_threshold is not None and (
            early_rebalance_risk_penalty_benchmark_drawdown_max_threshold < -1.0
            or early_rebalance_risk_penalty_benchmark_drawdown_max_threshold > 0.0
        ):
            raise ValueError(
                "early_rebalance_risk_penalty_benchmark_drawdown_max_threshold must be between -1 and 0 when provided."
            )
        if (
            early_rebalance_risk_penalty_benchmark_drawdown_min_threshold is not None
            and early_rebalance_risk_penalty_benchmark_drawdown_max_threshold is not None
            and early_rebalance_risk_penalty_benchmark_drawdown_min_threshold
            > early_rebalance_risk_penalty_benchmark_drawdown_max_threshold
        ):
            raise ValueError(
                "early_rebalance_risk_penalty_benchmark_drawdown_min_threshold must be smaller than or equal to early_rebalance_risk_penalty_benchmark_drawdown_max_threshold when both are provided."
            )
        if (
            early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio is not None
            and early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio <= 0.0
        ):
            raise ValueError(
                "early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio must be positive when provided."
            )
        if (
            early_rebalance_risk_penalty_symbol_min_weight is not None
            and early_rebalance_risk_penalty_symbol_max_weight is not None
            and early_rebalance_risk_penalty_symbol_min_weight > early_rebalance_risk_penalty_symbol_max_weight
        ):
            raise ValueError(
                "early_rebalance_risk_penalty_symbol_min_weight must be smaller than or equal to early_rebalance_risk_penalty_symbol_max_weight when both are provided."
            )
        if (
            early_rebalance_risk_penalty > 0.0
            or early_rebalance_risk_turnover_cap is not None
            or early_rebalance_risk_deep_drawdown_turnover_cap is not None
            or early_rebalance_risk_shallow_drawdown_turnover_cap is not None
            or early_rebalance_risk_mean_reversion_turnover_cap is not None
            or early_rebalance_risk_control_configured
        ) and (
            (
                early_rebalance_risk_penalty_before is None
                and early_rebalance_risk_turnover_cap_before is None
                and early_rebalance_risk_deep_drawdown_turnover_cap_before is None
                and early_rebalance_risk_shallow_drawdown_turnover_cap_before is None
                and early_rebalance_risk_mean_reversion_turnover_cap_before is None
            )
            or (
                early_rebalance_risk_penalty_cash_max_threshold is None
                and early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold is None
            )
            or early_rebalance_risk_penalty_symbol is None
        ):
            raise ValueError(
                "early rebalance risk control requires before, cash max threshold, and symbol when configured."
            )
        if (
            early_rebalance_risk_penalty > 0.0
            and early_rebalance_risk_penalty_symbol_min_weight is None
            and early_rebalance_risk_penalty_symbol_max_weight is None
        ):
            raise ValueError(
                "positive early rebalance risk penalty requires at least one symbol weight bound when configured."
            )
        early_benchmark_euphoria_control_configured = any(
            value is not None
            for value in (
                early_benchmark_euphoria_before,
                early_benchmark_euphoria_benchmark_drawdown_min_threshold,
                early_benchmark_euphoria_symbol,
            )
        )
        if early_benchmark_euphoria_penalty < 0.0:
            raise ValueError("early_benchmark_euphoria_penalty must be non-negative.")
        if early_benchmark_euphoria_turnover_cap is not None and early_benchmark_euphoria_turnover_cap < 0.0:
            raise ValueError("early_benchmark_euphoria_turnover_cap must be non-negative when provided.")
        if early_benchmark_euphoria_before is not None and early_benchmark_euphoria_before < 0:
            raise ValueError("early_benchmark_euphoria_before must be non-negative when provided.")
        if early_benchmark_euphoria_benchmark_drawdown_min_threshold is not None and (
            early_benchmark_euphoria_benchmark_drawdown_min_threshold < -1.0
            or early_benchmark_euphoria_benchmark_drawdown_min_threshold > 0.0
        ):
            raise ValueError(
                "early_benchmark_euphoria_benchmark_drawdown_min_threshold must be between -1 and 0 when provided."
            )
        if (
            early_benchmark_euphoria_penalty > 0.0
            or early_benchmark_euphoria_turnover_cap is not None
            or early_benchmark_euphoria_control_configured
        ) and (
            early_benchmark_euphoria_before is None
            or early_benchmark_euphoria_benchmark_drawdown_min_threshold is None
            or early_benchmark_euphoria_symbol is None
        ):
            raise ValueError(
                "early benchmark euphoria control requires before, benchmark drawdown threshold, and symbol when configured."
            )
        if late_rebalance_penalty < 0.0:
            raise ValueError("late_rebalance_penalty must be non-negative.")
        if late_rebalance_penalty_after is not None and late_rebalance_penalty_after < 0:
            raise ValueError("late_rebalance_penalty_after must be non-negative when provided.")
        if late_rebalance_penalty > 0.0 and late_rebalance_penalty_after is None:
            raise ValueError("late_rebalance_penalty_after is required when late_rebalance_penalty is positive.")
        late_rebalance_gate_configured = any(
            value is not None
            for value in (
                late_rebalance_gate_after,
                late_rebalance_gate_cash_threshold,
                late_rebalance_gate_target_cash_min_threshold,
                late_rebalance_gate_symbol,
                late_rebalance_gate_symbol_max_weight,
                late_rebalance_gate_cash_reduction_max,
                late_rebalance_gate_symbol_increase_max,
            )
        )
        if late_rebalance_gate_after is not None and late_rebalance_gate_after < 0:
            raise ValueError("late_rebalance_gate_after must be non-negative when provided.")
        if late_rebalance_gate_cash_threshold is not None and (
            late_rebalance_gate_cash_threshold < 0.0 or late_rebalance_gate_cash_threshold > 1.0
        ):
            raise ValueError("late_rebalance_gate_cash_threshold must be between 0 and 1 when provided.")
        if late_rebalance_gate_target_cash_min_threshold is not None and (
            late_rebalance_gate_target_cash_min_threshold < 0.0 or late_rebalance_gate_target_cash_min_threshold > 1.0
        ):
            raise ValueError("late_rebalance_gate_target_cash_min_threshold must be between 0 and 1 when provided.")
        if late_rebalance_gate_symbol_max_weight is not None and (
            late_rebalance_gate_symbol_max_weight < 0.0 or late_rebalance_gate_symbol_max_weight > 1.0
        ):
            raise ValueError("late_rebalance_gate_symbol_max_weight must be between 0 and 1 when provided.")
        if late_rebalance_gate_cash_reduction_max is not None and (
            late_rebalance_gate_cash_reduction_max < 0.0 or late_rebalance_gate_cash_reduction_max > 1.0
        ):
            raise ValueError("late_rebalance_gate_cash_reduction_max must be between 0 and 1 when provided.")
        if late_rebalance_gate_symbol_increase_max is not None and (
            late_rebalance_gate_symbol_increase_max < 0.0 or late_rebalance_gate_symbol_increase_max > 1.0
        ):
            raise ValueError("late_rebalance_gate_symbol_increase_max must be between 0 and 1 when provided.")
        if late_rebalance_gate_configured and (
            late_rebalance_gate_after is None
            or late_rebalance_gate_cash_threshold is None
            or late_rebalance_gate_symbol is None
            or late_rebalance_gate_symbol_max_weight is None
        ):
            raise ValueError(
                "late rebalance gate requires after, cash threshold, symbol, and symbol max weight when configured."
            )
        if (late_rebalance_gate_cash_reduction_max is None) != (late_rebalance_gate_symbol_increase_max is None):
            raise ValueError(
                "late rebalance gate refinement requires both cash reduction max and symbol increase max when configured."
            )
        late_defensive_posture_configured = any(
            value is not None
            for value in (
                late_defensive_posture_penalty_after,
                late_defensive_posture_penalty_cash_min_threshold,
                late_defensive_posture_penalty_symbol,
                late_defensive_posture_penalty_symbol_max_weight,
            )
        )
        state_trend_preservation_configured = any(
            value is not None
            for value in (
                state_trend_preservation_symbol,
                state_trend_preservation_cash_max_threshold,
                state_trend_preservation_symbol_min_weight,
                state_trend_preservation_max_symbol_reduction,
            )
        )
        if late_defensive_posture_penalty < 0.0:
            raise ValueError("late_defensive_posture_penalty must be non-negative.")
        if late_defensive_posture_penalty_after is not None and late_defensive_posture_penalty_after < 0:
            raise ValueError("late_defensive_posture_penalty_after must be non-negative when provided.")
        if late_defensive_posture_penalty_cash_min_threshold is not None and (
            late_defensive_posture_penalty_cash_min_threshold < 0.0
            or late_defensive_posture_penalty_cash_min_threshold > 1.0
        ):
            raise ValueError(
                "late_defensive_posture_penalty_cash_min_threshold must be between 0 and 1 when provided."
            )
        if late_defensive_posture_penalty_symbol_max_weight is not None and (
            late_defensive_posture_penalty_symbol_max_weight < 0.0
            or late_defensive_posture_penalty_symbol_max_weight > 1.0
        ):
            raise ValueError(
                "late_defensive_posture_penalty_symbol_max_weight must be between 0 and 1 when provided."
            )
        if (late_defensive_posture_penalty > 0.0 or late_defensive_posture_configured) and (
            late_defensive_posture_penalty_after is None
            or late_defensive_posture_penalty_cash_min_threshold is None
            or late_defensive_posture_penalty_symbol is None
            or late_defensive_posture_penalty_symbol_max_weight is None
        ):
            raise ValueError(
                "late defensive posture penalty requires after, cash min threshold, symbol, and symbol max weight when configured."
            )
        late_trend_mean_reversion_conflict_configured = any(
            value is not None
            for value in (
                late_trend_mean_reversion_conflict_penalty_after,
                late_trend_mean_reversion_conflict_trend_symbol,
                late_trend_mean_reversion_conflict_trend_min_weight,
                late_trend_mean_reversion_conflict_mean_reversion_symbol,
                late_trend_mean_reversion_conflict_mean_reversion_min_weight,
            )
        )
        if late_trend_mean_reversion_conflict_penalty < 0.0:
            raise ValueError("late_trend_mean_reversion_conflict_penalty must be non-negative.")
        if (
            late_trend_mean_reversion_conflict_penalty_after is not None
            and late_trend_mean_reversion_conflict_penalty_after < 0
        ):
            raise ValueError("late_trend_mean_reversion_conflict_penalty_after must be non-negative when provided.")
        if late_trend_mean_reversion_conflict_trend_min_weight is not None and (
            late_trend_mean_reversion_conflict_trend_min_weight < 0.0
            or late_trend_mean_reversion_conflict_trend_min_weight > 1.0
        ):
            raise ValueError(
                "late_trend_mean_reversion_conflict_trend_min_weight must be between 0 and 1 when provided."
            )
        if late_trend_mean_reversion_conflict_mean_reversion_min_weight is not None and (
            late_trend_mean_reversion_conflict_mean_reversion_min_weight < 0.0
            or late_trend_mean_reversion_conflict_mean_reversion_min_weight > 1.0
        ):
            raise ValueError(
                "late_trend_mean_reversion_conflict_mean_reversion_min_weight must be between 0 and 1 when provided."
            )
        if (
            late_trend_mean_reversion_conflict_penalty > 0.0
            or late_trend_mean_reversion_conflict_configured
        ) and (
            late_trend_mean_reversion_conflict_penalty_after is None
            or late_trend_mean_reversion_conflict_trend_symbol is None
            or late_trend_mean_reversion_conflict_trend_min_weight is None
            or late_trend_mean_reversion_conflict_mean_reversion_symbol is None
            or late_trend_mean_reversion_conflict_mean_reversion_min_weight is None
        ):
            raise ValueError(
                "late trend/mean-reversion conflict penalty requires after, trend symbol/min weight, and mean-reversion symbol/min weight when configured."
            )
        if state_trend_preservation_cash_max_threshold is not None and (
            state_trend_preservation_cash_max_threshold < 0.0
            or state_trend_preservation_cash_max_threshold > 1.0
        ):
            raise ValueError("state_trend_preservation_cash_max_threshold must be between 0 and 1 when provided.")
        if state_trend_preservation_symbol_min_weight is not None and (
            state_trend_preservation_symbol_min_weight < 0.0 or state_trend_preservation_symbol_min_weight > 1.0
        ):
            raise ValueError("state_trend_preservation_symbol_min_weight must be between 0 and 1 when provided.")
        if state_trend_preservation_max_symbol_reduction is not None and (
            state_trend_preservation_max_symbol_reduction <= 0.0 or state_trend_preservation_max_symbol_reduction > 1.0
        ):
            raise ValueError("state_trend_preservation_max_symbol_reduction must be positive and at most 1 when provided.")
        if state_trend_preservation_configured and (
            state_trend_preservation_symbol is None
            or state_trend_preservation_cash_max_threshold is None
            or state_trend_preservation_symbol_min_weight is None
            or state_trend_preservation_max_symbol_reduction is None
        ):
            raise ValueError(
                "state trend preservation guard requires symbol, cash max threshold, symbol min weight, and max symbol reduction when configured."
            )
        if cash_weight_penalty < 0.0:
            raise ValueError("cash_weight_penalty must be non-negative.")
        self.action_smoothing = float(action_smoothing)
        self.no_trade_band = float(no_trade_band)
        self.max_executed_rebalances = int(max_executed_rebalances) if max_executed_rebalances is not None else None
        self.rebalance_cooldown_steps = int(rebalance_cooldown_steps) if rebalance_cooldown_steps is not None else None
        self.early_rebalance_risk_penalty = float(early_rebalance_risk_penalty)
        self.early_rebalance_risk_turnover_cap = (
            float(early_rebalance_risk_turnover_cap) if early_rebalance_risk_turnover_cap is not None else None
        )
        self.early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold = (
            float(early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold)
            if early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold is not None
            else None
        )
        self.early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold = (
            float(early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold)
            if early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold is not None
            else None
        )
        self.early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight = (
            float(early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight)
            if early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight is not None
            else None
        )
        self.early_rebalance_risk_turnover_cap_target_cash_min_threshold = (
            float(early_rebalance_risk_turnover_cap_target_cash_min_threshold)
            if early_rebalance_risk_turnover_cap_target_cash_min_threshold is not None
            else None
        )
        self.early_rebalance_risk_turnover_cap_target_cash_max_threshold = (
            float(early_rebalance_risk_turnover_cap_target_cash_max_threshold)
            if early_rebalance_risk_turnover_cap_target_cash_max_threshold is not None
            else None
        )
        self.early_rebalance_risk_turnover_cap_target_trend_min_threshold = (
            float(early_rebalance_risk_turnover_cap_target_trend_min_threshold)
            if early_rebalance_risk_turnover_cap_target_trend_min_threshold is not None
            else None
        )
        self.early_rebalance_risk_turnover_cap_target_trend_max_threshold = (
            float(early_rebalance_risk_turnover_cap_target_trend_max_threshold)
            if early_rebalance_risk_turnover_cap_target_trend_max_threshold is not None
            else None
        )
        self.early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold = (
            float(early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold)
            if early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold is not None
            else None
        )
        self.early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold = (
            float(early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold)
            if early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold is not None
            else None
        )
        self.early_rebalance_risk_turnover_cap_delta_cash_min_threshold = (
            float(early_rebalance_risk_turnover_cap_delta_cash_min_threshold)
            if early_rebalance_risk_turnover_cap_delta_cash_min_threshold is not None
            else None
        )
        self.early_rebalance_risk_turnover_cap_delta_cash_max_threshold = (
            float(early_rebalance_risk_turnover_cap_delta_cash_max_threshold)
            if early_rebalance_risk_turnover_cap_delta_cash_max_threshold is not None
            else None
        )
        self.early_rebalance_risk_turnover_cap_delta_trend_min_threshold = (
            float(early_rebalance_risk_turnover_cap_delta_trend_min_threshold)
            if early_rebalance_risk_turnover_cap_delta_trend_min_threshold is not None
            else None
        )
        self.early_rebalance_risk_turnover_cap_delta_trend_max_threshold = (
            float(early_rebalance_risk_turnover_cap_delta_trend_max_threshold)
            if early_rebalance_risk_turnover_cap_delta_trend_max_threshold is not None
            else None
        )
        self.early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold = (
            float(early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold)
            if early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold is not None
            else None
        )
        self.early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold = (
            float(early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold)
            if early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold is not None
            else None
        )
        self.early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold = (
            float(early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold)
            if early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold is not None
            else None
        )
        self.early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold = (
            float(early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold)
            if early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold is not None
            else None
        )
        self.early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio = (
            float(early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio)
            if early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio is not None
            else None
        )
        self.early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio = (
            float(early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio)
            if early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio is not None
            else None
        )
        self.early_rebalance_risk_turnover_cap_allow_nonincreasing_risk_symbol = bool(
            early_rebalance_risk_turnover_cap_allow_nonincreasing_risk_symbol
        )
        self.early_rebalance_risk_turnover_cap_use_penalty_state_filters = bool(
            early_rebalance_risk_turnover_cap_use_penalty_state_filters
        )
        self.early_rebalance_risk_turnover_cap_after = (
            int(early_rebalance_risk_turnover_cap_after) if early_rebalance_risk_turnover_cap_after is not None else None
        )
        self.early_rebalance_risk_turnover_cap_before = (
            int(early_rebalance_risk_turnover_cap_before) if early_rebalance_risk_turnover_cap_before is not None else None
        )
        self.early_rebalance_risk_turnover_cap_max_applications = (
            int(early_rebalance_risk_turnover_cap_max_applications)
            if early_rebalance_risk_turnover_cap_max_applications is not None
            else None
        )
        self.early_rebalance_risk_turnover_cap_secondary_cap = (
            float(early_rebalance_risk_turnover_cap_secondary_cap)
            if early_rebalance_risk_turnover_cap_secondary_cap is not None
            else None
        )
        self.early_rebalance_risk_turnover_cap_secondary_after_applications = (
            int(early_rebalance_risk_turnover_cap_secondary_after_applications)
            if early_rebalance_risk_turnover_cap_secondary_after_applications is not None
            else None
        )
        self.early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold = (
            float(early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold)
            if early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold is not None
            else None
        )
        self.early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio = (
            float(early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio)
            if early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio is not None
            else None
        )
        self.early_rebalance_risk_deep_drawdown_turnover_cap = (
            float(early_rebalance_risk_deep_drawdown_turnover_cap)
            if early_rebalance_risk_deep_drawdown_turnover_cap is not None
            else None
        )
        self.early_rebalance_risk_deep_drawdown_turnover_cap_after = (
            int(early_rebalance_risk_deep_drawdown_turnover_cap_after)
            if early_rebalance_risk_deep_drawdown_turnover_cap_after is not None
            else None
        )
        self.early_rebalance_risk_deep_drawdown_turnover_cap_before = (
            int(early_rebalance_risk_deep_drawdown_turnover_cap_before)
            if early_rebalance_risk_deep_drawdown_turnover_cap_before is not None
            else None
        )
        self.early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold = (
            float(early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold)
            if early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold is not None
            else None
        )
        self.early_rebalance_risk_shallow_drawdown_turnover_cap = (
            float(early_rebalance_risk_shallow_drawdown_turnover_cap)
            if early_rebalance_risk_shallow_drawdown_turnover_cap is not None
            else None
        )
        self.early_rebalance_risk_shallow_drawdown_turnover_cap_after = (
            int(early_rebalance_risk_shallow_drawdown_turnover_cap_after)
            if early_rebalance_risk_shallow_drawdown_turnover_cap_after is not None
            else None
        )
        self.early_rebalance_risk_shallow_drawdown_turnover_cap_before = (
            int(early_rebalance_risk_shallow_drawdown_turnover_cap_before)
            if early_rebalance_risk_shallow_drawdown_turnover_cap_before is not None
            else None
        )
        self.early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold = (
            float(early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold)
            if early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold is not None
            else None
        )
        self.early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold = (
            float(early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold)
            if early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold is not None
            else None
        )
        self.early_rebalance_risk_mean_reversion_turnover_cap = (
            float(early_rebalance_risk_mean_reversion_turnover_cap)
            if early_rebalance_risk_mean_reversion_turnover_cap is not None
            else None
        )
        self.early_rebalance_risk_mean_reversion_action_smoothing = (
            float(early_rebalance_risk_mean_reversion_action_smoothing)
            if early_rebalance_risk_mean_reversion_action_smoothing is not None
            else None
        )
        self.early_rebalance_risk_mean_reversion_turnover_cap_after = (
            int(early_rebalance_risk_mean_reversion_turnover_cap_after)
            if early_rebalance_risk_mean_reversion_turnover_cap_after is not None
            else None
        )
        self.early_rebalance_risk_mean_reversion_turnover_cap_before = (
            int(early_rebalance_risk_mean_reversion_turnover_cap_before)
            if early_rebalance_risk_mean_reversion_turnover_cap_before is not None
            else None
        )
        self.early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold = (
            float(early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold)
            if early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold is not None
            else None
        )
        self.early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold = (
            float(early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold)
            if early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold is not None
            else None
        )
        self.early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold = (
            float(early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold)
            if early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold is not None
            else None
        )
        self.early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold = (
            float(early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold)
            if early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold is not None
            else None
        )
        self.early_rebalance_risk_trend_turnover_cap = (
            float(early_rebalance_risk_trend_turnover_cap)
            if early_rebalance_risk_trend_turnover_cap is not None
            else None
        )
        self.early_rebalance_risk_trend_turnover_cap_after = (
            int(early_rebalance_risk_trend_turnover_cap_after)
            if early_rebalance_risk_trend_turnover_cap_after is not None
            else None
        )
        self.early_rebalance_risk_trend_turnover_cap_before = (
            int(early_rebalance_risk_trend_turnover_cap_before)
            if early_rebalance_risk_trend_turnover_cap_before is not None
            else None
        )
        self.early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold = (
            float(early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold)
            if early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold is not None
            else None
        )
        self.early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold = (
            float(early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold)
            if early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold is not None
            else None
        )
        self.early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold = (
            float(early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold)
            if early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold is not None
            else None
        )
        self.early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold = (
            float(early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold)
            if early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold is not None
            else None
        )
        self.early_rebalance_risk_repeat_turnover_cap = (
            float(early_rebalance_risk_repeat_turnover_cap)
            if early_rebalance_risk_repeat_turnover_cap is not None
            else None
        )
        self.early_rebalance_risk_repeat_action_smoothing = (
            float(early_rebalance_risk_repeat_action_smoothing)
            if early_rebalance_risk_repeat_action_smoothing is not None
            else None
        )
        self.early_rebalance_risk_repeat_turnover_cap_after = (
            int(early_rebalance_risk_repeat_turnover_cap_after)
            if early_rebalance_risk_repeat_turnover_cap_after is not None
            else None
        )
        self.early_rebalance_risk_repeat_turnover_cap_before = (
            int(early_rebalance_risk_repeat_turnover_cap_before)
            if early_rebalance_risk_repeat_turnover_cap_before is not None
            else None
        )
        self.early_rebalance_risk_repeat_symbol = early_rebalance_risk_repeat_symbol
        self.early_rebalance_risk_repeat_previous_cash_reduction_min = (
            float(early_rebalance_risk_repeat_previous_cash_reduction_min)
            if early_rebalance_risk_repeat_previous_cash_reduction_min is not None
            else None
        )
        self.early_rebalance_risk_repeat_previous_symbol_increase_min = (
            float(early_rebalance_risk_repeat_previous_symbol_increase_min)
            if early_rebalance_risk_repeat_previous_symbol_increase_min is not None
            else None
        )
        self.early_rebalance_risk_repeat_unrecovered_turnover_cap = (
            float(early_rebalance_risk_repeat_unrecovered_turnover_cap)
            if early_rebalance_risk_repeat_unrecovered_turnover_cap is not None
            else None
        )
        self.early_rebalance_risk_repeat_unrecovered_turnover_cap_after = (
            int(early_rebalance_risk_repeat_unrecovered_turnover_cap_after)
            if early_rebalance_risk_repeat_unrecovered_turnover_cap_after is not None
            else None
        )
        self.early_rebalance_risk_repeat_unrecovered_turnover_cap_before = (
            int(early_rebalance_risk_repeat_unrecovered_turnover_cap_before)
            if early_rebalance_risk_repeat_unrecovered_turnover_cap_before is not None
            else None
        )
        self.early_rebalance_risk_repeat_unrecovered_symbol = (
            early_rebalance_risk_repeat_unrecovered_symbol
        )
        self.early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min = (
            float(early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min)
            if early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min is not None
            else None
        )
        self.early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min = (
            float(early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min)
            if early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min is not None
            else None
        )
        self.early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery = (
            float(early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery)
            if early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery is not None
            else None
        )
        self.early_rebalance_risk_cumulative_turnover_cap = (
            float(early_rebalance_risk_cumulative_turnover_cap)
            if early_rebalance_risk_cumulative_turnover_cap is not None
            else None
        )
        self.early_rebalance_risk_cumulative_turnover_cap_after = (
            int(early_rebalance_risk_cumulative_turnover_cap_after)
            if early_rebalance_risk_cumulative_turnover_cap_after is not None
            else None
        )
        self.early_rebalance_risk_cumulative_turnover_cap_before = (
            int(early_rebalance_risk_cumulative_turnover_cap_before)
            if early_rebalance_risk_cumulative_turnover_cap_before is not None
            else None
        )
        self.early_rebalance_risk_cumulative_symbol = early_rebalance_risk_cumulative_symbol
        self.early_rebalance_risk_cumulative_cash_reduction_budget = (
            float(early_rebalance_risk_cumulative_cash_reduction_budget)
            if early_rebalance_risk_cumulative_cash_reduction_budget is not None
            else None
        )
        self.early_rebalance_risk_cumulative_symbol_increase_budget = (
            float(early_rebalance_risk_cumulative_symbol_increase_budget)
            if early_rebalance_risk_cumulative_symbol_increase_budget is not None
            else None
        )
        self.early_rebalance_risk_penalty_after = (
            int(early_rebalance_risk_penalty_after) if early_rebalance_risk_penalty_after is not None else None
        )
        self.early_rebalance_risk_penalty_before = (
            int(early_rebalance_risk_penalty_before) if early_rebalance_risk_penalty_before is not None else None
        )
        self.early_rebalance_risk_penalty_cash_max_threshold = (
            float(early_rebalance_risk_penalty_cash_max_threshold)
            if early_rebalance_risk_penalty_cash_max_threshold is not None
            else None
        )
        self.early_rebalance_risk_penalty_symbol = early_rebalance_risk_penalty_symbol
        self.early_rebalance_risk_penalty_symbol_min_weight = (
            float(early_rebalance_risk_penalty_symbol_min_weight)
            if early_rebalance_risk_penalty_symbol_min_weight is not None
            else None
        )
        self.early_rebalance_risk_penalty_symbol_max_weight = (
            float(early_rebalance_risk_penalty_symbol_max_weight)
            if early_rebalance_risk_penalty_symbol_max_weight is not None
            else None
        )
        self.early_rebalance_risk_penalty_benchmark_drawdown_min_threshold = (
            float(early_rebalance_risk_penalty_benchmark_drawdown_min_threshold)
            if early_rebalance_risk_penalty_benchmark_drawdown_min_threshold is not None
            else None
        )
        self.early_rebalance_risk_penalty_benchmark_drawdown_max_threshold = (
            float(early_rebalance_risk_penalty_benchmark_drawdown_max_threshold)
            if early_rebalance_risk_penalty_benchmark_drawdown_max_threshold is not None
            else None
        )
        self.early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio = (
            float(early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio)
            if early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio is not None
            else None
        )
        self.early_benchmark_euphoria_penalty = float(early_benchmark_euphoria_penalty)
        self.early_benchmark_euphoria_turnover_cap = (
            float(early_benchmark_euphoria_turnover_cap) if early_benchmark_euphoria_turnover_cap is not None else None
        )
        self.early_benchmark_euphoria_before = (
            int(early_benchmark_euphoria_before) if early_benchmark_euphoria_before is not None else None
        )
        self.early_benchmark_euphoria_benchmark_drawdown_min_threshold = (
            float(early_benchmark_euphoria_benchmark_drawdown_min_threshold)
            if early_benchmark_euphoria_benchmark_drawdown_min_threshold is not None
            else None
        )
        self.early_benchmark_euphoria_symbol = early_benchmark_euphoria_symbol
        self.late_rebalance_penalty = float(late_rebalance_penalty)
        self.late_rebalance_penalty_after = int(late_rebalance_penalty_after) if late_rebalance_penalty_after is not None else None
        self.late_rebalance_gate_after = int(late_rebalance_gate_after) if late_rebalance_gate_after is not None else None
        self.late_rebalance_gate_cash_threshold = (
            float(late_rebalance_gate_cash_threshold) if late_rebalance_gate_cash_threshold is not None else None
        )
        self.late_rebalance_gate_target_cash_min_threshold = (
            float(late_rebalance_gate_target_cash_min_threshold)
            if late_rebalance_gate_target_cash_min_threshold is not None
            else None
        )
        self.late_rebalance_gate_symbol = late_rebalance_gate_symbol
        self.late_rebalance_gate_symbol_max_weight = (
            float(late_rebalance_gate_symbol_max_weight) if late_rebalance_gate_symbol_max_weight is not None else None
        )
        self.late_rebalance_gate_cash_reduction_max = (
            float(late_rebalance_gate_cash_reduction_max)
            if late_rebalance_gate_cash_reduction_max is not None
            else None
        )
        self.late_rebalance_gate_symbol_increase_max = (
            float(late_rebalance_gate_symbol_increase_max)
            if late_rebalance_gate_symbol_increase_max is not None
            else None
        )
        self.late_defensive_posture_penalty = float(late_defensive_posture_penalty)
        self.late_defensive_posture_penalty_after = (
            int(late_defensive_posture_penalty_after) if late_defensive_posture_penalty_after is not None else None
        )
        self.late_defensive_posture_penalty_cash_min_threshold = (
            float(late_defensive_posture_penalty_cash_min_threshold)
            if late_defensive_posture_penalty_cash_min_threshold is not None
            else None
        )
        self.late_defensive_posture_penalty_symbol = late_defensive_posture_penalty_symbol
        self.late_defensive_posture_penalty_symbol_max_weight = (
            float(late_defensive_posture_penalty_symbol_max_weight)
            if late_defensive_posture_penalty_symbol_max_weight is not None
            else None
        )
        self.late_trend_mean_reversion_conflict_penalty = float(late_trend_mean_reversion_conflict_penalty)
        self.late_trend_mean_reversion_conflict_penalty_after = (
            int(late_trend_mean_reversion_conflict_penalty_after)
            if late_trend_mean_reversion_conflict_penalty_after is not None
            else None
        )
        self.late_trend_mean_reversion_conflict_trend_symbol = late_trend_mean_reversion_conflict_trend_symbol
        self.late_trend_mean_reversion_conflict_trend_min_weight = (
            float(late_trend_mean_reversion_conflict_trend_min_weight)
            if late_trend_mean_reversion_conflict_trend_min_weight is not None
            else None
        )
        self.late_trend_mean_reversion_conflict_mean_reversion_symbol = (
            late_trend_mean_reversion_conflict_mean_reversion_symbol
        )
        self.late_trend_mean_reversion_conflict_mean_reversion_min_weight = (
            float(late_trend_mean_reversion_conflict_mean_reversion_min_weight)
            if late_trend_mean_reversion_conflict_mean_reversion_min_weight is not None
            else None
        )
        self.state_trend_preservation_symbol = state_trend_preservation_symbol
        self.state_trend_preservation_cash_max_threshold = (
            float(state_trend_preservation_cash_max_threshold)
            if state_trend_preservation_cash_max_threshold is not None
            else None
        )
        self.state_trend_preservation_symbol_min_weight = (
            float(state_trend_preservation_symbol_min_weight)
            if state_trend_preservation_symbol_min_weight is not None
            else None
        )
        self.state_trend_preservation_max_symbol_reduction = (
            float(state_trend_preservation_max_symbol_reduction)
            if state_trend_preservation_max_symbol_reduction is not None
            else None
        )
        self.cash_weight_penalty = float(cash_weight_penalty)
        self.returns = _coerce_returns_frame(returns).sort_index()
        if self.config.include_cash:
            self.returns = add_cash_sleeve(self.returns, cash_symbol=self.config.cash_symbol)
        if len(self.returns) <= self.lookback:
            raise ValueError("Not enough return observations for the requested lookback.")
        max_episode_steps = len(self.returns) - self.lookback
        if episode_length is not None and (episode_length <= 0 or episode_length > max_episode_steps):
            raise ValueError("episode_length must be positive and leave at least one valid episode window.")
        self.episode_length = int(episode_length) if episode_length is not None else None

        self.benchmark_returns = _coerce_benchmark_series(benchmark_returns, self.returns.index)
        if (self.config.benchmark_gain_reward > 0 or self.config.benchmark_loss_penalty > 0) and self.benchmark_returns is None:
            raise ValueError("Benchmark-aware reward requires benchmark returns.")
        if benchmark_relative_observations and self.benchmark_returns is None:
            raise ValueError("benchmark_relative_observations requires benchmark returns.")
        requested_benchmark_regime_observations = bool(benchmark_regime_observations)
        resolved_benchmark_regime_summary_observations = (
            requested_benchmark_regime_observations
            if benchmark_regime_summary_observations is None
            else bool(benchmark_regime_summary_observations)
        )
        resolved_benchmark_regime_relative_cumulative_observations = (
            requested_benchmark_regime_observations
            if benchmark_regime_relative_cumulative_observations is None
            else bool(benchmark_regime_relative_cumulative_observations)
        )
        if (
            (requested_benchmark_regime_observations
            or resolved_benchmark_regime_summary_observations
            or resolved_benchmark_regime_relative_cumulative_observations)
            and self.benchmark_returns is None
        ):
            raise ValueError("benchmark_regime_observations requires benchmark returns.")
        if (
            self.benchmark_returns is None
            and (
                self.early_benchmark_euphoria_penalty > 0.0
                or self.early_benchmark_euphoria_turnover_cap is not None
                or early_benchmark_euphoria_control_configured
            )
        ):
            raise ValueError("early benchmark euphoria controls require benchmark returns.")
        if (
            self.benchmark_returns is None
            and (
                self.early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold is not None
                or self.early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold is not None
                or self.early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold is not None
                or self.early_rebalance_risk_deep_drawdown_turnover_cap is not None
                or self.early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold is not None
                or self.early_rebalance_risk_repeat_unrecovered_turnover_cap is not None
                or early_rebalance_risk_repeat_unrecovered_control_configured
                or self.early_rebalance_risk_penalty_benchmark_drawdown_min_threshold is not None
                or self.early_rebalance_risk_penalty_benchmark_drawdown_max_threshold is not None
                or self.early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio is not None
            )
        ):
            raise ValueError("benchmark returns are required for benchmark-aware early rebalance risk filters.")
        self.benchmark_relative_observations = bool(benchmark_relative_observations)
        self.benchmark_regime_summary_observations = resolved_benchmark_regime_summary_observations
        self.benchmark_regime_relative_cumulative_observations = (
            resolved_benchmark_regime_relative_cumulative_observations
        )
        self.benchmark_regime_observations = bool(
            self.benchmark_regime_summary_observations
            or self.benchmark_regime_relative_cumulative_observations
        )

        self.lower_bounds, self.upper_bounds = _resolve_weight_bounds(self.returns.columns, self.config)
        self.cash_index = self.returns.columns.get_loc(self.config.cash_symbol) if self.config.cash_symbol in self.returns.columns else None
        if self.cash_index is None:
            if (
                self.cash_weight_penalty > 0.0
                or cash_target_weight is not None
                or late_rebalance_gate_configured
                or late_defensive_posture_configured
                or early_rebalance_risk_repeat_control_configured
                or early_rebalance_risk_repeat_unrecovered_control_configured
                or early_rebalance_risk_cumulative_control_configured
                or state_trend_preservation_configured
            ):
                raise ValueError("Cash penalty controls require the configured cash sleeve to be present in the environment.")
            self.cash_target_weight = None
        else:
            resolved_cash_target = float(self.lower_bounds[self.cash_index]) if cash_target_weight is None else float(cash_target_weight)
            if resolved_cash_target < 0.0 or resolved_cash_target > 1.0:
                raise ValueError("cash_target_weight must be between 0 and 1.")
            self.cash_target_weight = resolved_cash_target
        if self.late_rebalance_gate_symbol is not None:
            if self.late_rebalance_gate_symbol not in self.returns.columns:
                raise ValueError("late_rebalance_gate_symbol must exist in the environment universe.")
            self.late_rebalance_gate_symbol_index = self.returns.columns.get_loc(self.late_rebalance_gate_symbol)
        else:
            self.late_rebalance_gate_symbol_index = None
        if self.late_defensive_posture_penalty_symbol is not None:
            if self.late_defensive_posture_penalty_symbol not in self.returns.columns:
                raise ValueError("late_defensive_posture_penalty_symbol must exist in the environment universe.")
            self.late_defensive_posture_penalty_symbol_index = self.returns.columns.get_loc(
                self.late_defensive_posture_penalty_symbol
            )
        else:
            self.late_defensive_posture_penalty_symbol_index = None
        if self.early_rebalance_risk_penalty_symbol is not None:
            if self.early_rebalance_risk_penalty_symbol not in self.returns.columns:
                raise ValueError("early_rebalance_risk_penalty_symbol must exist in the environment universe.")
            self.early_rebalance_risk_penalty_symbol_index = self.returns.columns.get_loc(self.early_rebalance_risk_penalty_symbol)
        else:
            self.early_rebalance_risk_penalty_symbol_index = None
        if self.early_rebalance_risk_repeat_symbol is not None:
            if self.early_rebalance_risk_repeat_symbol not in self.returns.columns:
                raise ValueError("early_rebalance_risk_repeat_symbol must exist in the environment universe.")
            self.early_rebalance_risk_repeat_symbol_index = self.returns.columns.get_loc(
                self.early_rebalance_risk_repeat_symbol
            )
        else:
            self.early_rebalance_risk_repeat_symbol_index = None
        if self.early_rebalance_risk_repeat_unrecovered_symbol is not None:
            if self.early_rebalance_risk_repeat_unrecovered_symbol not in self.returns.columns:
                raise ValueError(
                    "early_rebalance_risk_repeat_unrecovered_symbol must exist in the environment universe."
                )
            self.early_rebalance_risk_repeat_unrecovered_symbol_index = self.returns.columns.get_loc(
                self.early_rebalance_risk_repeat_unrecovered_symbol
            )
        else:
            self.early_rebalance_risk_repeat_unrecovered_symbol_index = None
        if self.early_rebalance_risk_cumulative_symbol is not None:
            if self.early_rebalance_risk_cumulative_symbol not in self.returns.columns:
                raise ValueError("early_rebalance_risk_cumulative_symbol must exist in the environment universe.")
            self.early_rebalance_risk_cumulative_symbol_index = self.returns.columns.get_loc(
                self.early_rebalance_risk_cumulative_symbol
            )
        else:
            self.early_rebalance_risk_cumulative_symbol_index = None
        early_rebalance_risk_mean_reversion_control_configured = any(
            value is not None
            for value in (
                self.early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold,
                self.early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold,
                self.early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold,
                self.early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold,
                self.early_rebalance_risk_mean_reversion_turnover_cap,
                self.early_rebalance_risk_mean_reversion_action_smoothing,
                self.early_rebalance_risk_mean_reversion_turnover_cap_after,
                self.early_rebalance_risk_mean_reversion_turnover_cap_before,
                self.early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold,
                self.early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold,
                self.early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold,
            )
        )
        if early_rebalance_risk_mean_reversion_control_configured:
            if "MEAN_REVERSION" not in self.returns.columns:
                raise ValueError(
                    "early_rebalance_risk_mean_reversion control requires a MEAN_REVERSION asset in the environment universe."
                )
            self.early_rebalance_risk_mean_reversion_turnover_cap_symbol_index = self.returns.columns.get_loc(
                "MEAN_REVERSION"
            )
        else:
            self.early_rebalance_risk_mean_reversion_turnover_cap_symbol_index = None
        early_rebalance_risk_trend_control_configured = any(
            value is not None
            for value in (
                self.early_rebalance_risk_trend_turnover_cap,
                self.early_rebalance_risk_trend_turnover_cap_after,
                self.early_rebalance_risk_trend_turnover_cap_before,
                self.early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold,
                self.early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold,
                self.early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold,
            )
        )
        if early_rebalance_risk_trend_control_configured:
            if "TREND_FOLLOWING" not in self.returns.columns:
                raise ValueError(
                    "early_rebalance_risk_trend_turnover_cap requires a TREND_FOLLOWING asset in the environment universe."
                )
            self.early_rebalance_risk_trend_turnover_cap_symbol_index = self.returns.columns.get_loc(
                "TREND_FOLLOWING"
            )
        else:
            self.early_rebalance_risk_trend_turnover_cap_symbol_index = None
        if self.early_benchmark_euphoria_symbol is not None:
            if self.early_benchmark_euphoria_symbol not in self.returns.columns:
                raise ValueError("early_benchmark_euphoria_symbol must exist in the environment universe.")
            self.early_benchmark_euphoria_symbol_index = self.returns.columns.get_loc(self.early_benchmark_euphoria_symbol)
        else:
            self.early_benchmark_euphoria_symbol_index = None
        if self.state_trend_preservation_symbol is not None:
            if self.state_trend_preservation_symbol not in self.returns.columns:
                raise ValueError("state_trend_preservation_symbol must exist in the environment universe.")
            self.state_trend_preservation_symbol_index = self.returns.columns.get_loc(
                self.state_trend_preservation_symbol
            )
        else:
            self.state_trend_preservation_symbol_index = None
        if self.late_trend_mean_reversion_conflict_trend_symbol is not None:
            if self.late_trend_mean_reversion_conflict_trend_symbol not in self.returns.columns:
                raise ValueError(
                    "late_trend_mean_reversion_conflict_trend_symbol must exist in the environment universe."
                )
            self.late_trend_mean_reversion_conflict_trend_symbol_index = self.returns.columns.get_loc(
                self.late_trend_mean_reversion_conflict_trend_symbol
            )
        else:
            self.late_trend_mean_reversion_conflict_trend_symbol_index = None
        if self.late_trend_mean_reversion_conflict_mean_reversion_symbol is not None:
            if self.late_trend_mean_reversion_conflict_mean_reversion_symbol not in self.returns.columns:
                raise ValueError(
                    "late_trend_mean_reversion_conflict_mean_reversion_symbol must exist in the environment universe."
                )
            self.late_trend_mean_reversion_conflict_mean_reversion_symbol_index = self.returns.columns.get_loc(
                self.late_trend_mean_reversion_conflict_mean_reversion_symbol
            )
        else:
            self.late_trend_mean_reversion_conflict_mean_reversion_symbol_index = None
        default_weight = 1.0 / self.returns.shape[1]
        initial_weight_vector = _coerce_weight_vector(initial_weights, self.returns.columns, default_weight)
        initial_weight_vector = _project_to_weight_bounds(initial_weight_vector, self.lower_bounds, self.upper_bounds)
        self.initial_weights = pd.Series(initial_weight_vector, index=self.returns.columns, name="weight")

        self.asset_count = len(self.returns.columns)
        self.include_benchmark_features = self.benchmark_returns is not None
        observation_size = self.lookback * self.asset_count + self.asset_count + 2
        if self.include_benchmark_features:
            observation_size += self.lookback
        if self.benchmark_relative_observations:
            observation_size += self.lookback * self.asset_count
        if self.benchmark_regime_summary_observations:
            observation_size += 3
        if self.benchmark_regime_relative_cumulative_observations:
            observation_size += self.asset_count

        self.action_space = spaces.Box(
            low=self.lower_bounds.astype(np.float32),
            high=self.upper_bounds.astype(np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.full(observation_size, -np.inf, dtype=np.float32),
            high=np.full(observation_size, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

        self.current_step = self.lookback
        self.episode_start_step = self.lookback
        self.episode_end_step = len(self.returns)
        self.current_weights = self.initial_weights.copy()
        self.current_wealth = 1.0
        self.current_benchmark_wealth = 1.0
        self.peak_wealth = 1.0
        self.executed_rebalances = 0
        self.early_rebalance_risk_turnover_cap_applications = 0
        self.rebalance_cooldown_remaining = 0
        self.last_executed_rebalance_cash_reduction = 0.0
        self.last_executed_rebalance_symbol_increase = 0.0
        self.last_executed_rebalance_pre_trade_relative_wealth_ratio = None
        self.cumulative_executed_rebalance_cash_reduction = 0.0
        self.cumulative_executed_rebalance_symbol_increase = 0.0

    def _resolve_episode_window(self, options: dict[str, Any] | None = None) -> tuple[int, int]:
        requested_episode_length = self.episode_length
        if options and options.get("episode_length") is not None:
            requested_episode_length = int(options["episode_length"])
        if requested_episode_length is None:
            requested_episode_length = len(self.returns) - self.lookback
        if requested_episode_length <= 0 or requested_episode_length > len(self.returns) - self.lookback:
            raise ValueError("episode_length is infeasible for the available return history.")

        max_start_step = len(self.returns) - requested_episode_length
        if max_start_step < self.lookback:
            raise ValueError("Not enough return observations to build the requested episode window.")

        start_step = self.lookback
        if options and options.get("start_step") is not None:
            start_step = int(options["start_step"])
        elif self.random_episode_start:
            start_step = int(self.np_random.integers(self.lookback, max_start_step + 1))

        if start_step < self.lookback or start_step > max_start_step:
            raise ValueError("start_step is outside the feasible episode window range.")

        end_step = min(start_step + requested_episode_length, len(self.returns))
        return start_step, end_step

    def _score_step_utility(self, wealth_ratio: float, gain_reward: float, loss_penalty: float) -> float:
        if self.config.objective_mode == "piecewise":
            upside = max(wealth_ratio - 1.0, 0.0)
            downside = max(1.0 - wealth_ratio, 0.0)
            return gain_reward * (upside ** self.config.gain_power) - loss_penalty * (downside ** self.config.loss_power)

        safe_ratio = max(wealth_ratio, self.config.epsilon)
        log_growth = float(np.log(safe_ratio))
        upside = max(log_growth, 0.0)
        downside = max(-log_growth, 0.0)
        return gain_reward * upside - loss_penalty * (downside ** self.config.loss_power)

    def _build_observation(self) -> np.ndarray:
        window_end = min(self.current_step, len(self.returns))
        window_start = max(0, window_end - self.lookback)
        returns_frame = self.returns.iloc[window_start:window_end]
        returns_window = returns_frame.to_numpy(dtype=np.float32).reshape(-1)
        observation_parts: list[np.ndarray] = [returns_window]
        if self.include_benchmark_features and self.benchmark_returns is not None:
            benchmark_window = self.benchmark_returns.iloc[window_start:window_end].to_numpy(dtype=np.float32)
            observation_parts.append(benchmark_window)
            if self.benchmark_relative_observations:
                relative_window = returns_frame.sub(self.benchmark_returns.iloc[window_start:window_end], axis=0)
                observation_parts.append(relative_window.to_numpy(dtype=np.float32).reshape(-1))
            if self.benchmark_regime_observations:
                benchmark_window_float = self.benchmark_returns.iloc[window_start:window_end].to_numpy(dtype=float)
                benchmark_cumulative_return, benchmark_volatility, benchmark_drawdown = _summarize_return_window(
                    benchmark_window_float
                )
                asset_cumulative_returns = (1.0 + returns_frame.astype(float)).prod(axis=0) - 1.0
                relative_cumulative_returns = asset_cumulative_returns - benchmark_cumulative_return
                if self.benchmark_regime_summary_observations:
                    observation_parts.append(
                        np.array(
                            [benchmark_cumulative_return, benchmark_volatility, benchmark_drawdown],
                            dtype=np.float32,
                        )
                    )
                if self.benchmark_regime_relative_cumulative_observations:
                    observation_parts.append(relative_cumulative_returns.to_numpy(dtype=np.float32))

        drawdown = self.current_wealth / self.peak_wealth - 1.0
        observation_parts.append(self.current_weights.to_numpy(dtype=np.float32))
        observation_parts.append(np.array([self.current_wealth, drawdown], dtype=np.float32))
        return np.concatenate(observation_parts, dtype=np.float32)

    def _early_rebalance_risk_condition_met(
        self,
        *,
        window_active: bool,
        pre_trade_cash_weight: float,
        target_cash_weight: float,
        proposed_turnover: float | None = None,
        delta_cash_weight: float | None = None,
        pre_trade_risk_symbol_weight: float,
        target_risk_symbol_weight: float,
        delta_risk_symbol_weight: float | None = None,
        target_mean_reversion_weight: float | None = None,
        delta_mean_reversion_weight: float | None = None,
        benchmark_regime_cumulative_return: float | None,
        benchmark_regime_drawdown: float | None,
        pre_trade_relative_wealth_ratio: float | None,
        benchmark_cumulative_return_max_threshold: float | None = None,
        benchmark_drawdown_min_threshold: float | None = None,
        benchmark_drawdown_max_threshold: float | None = None,
        min_pre_trade_cash_weight: float | None = None,
        min_target_cash_weight: float | None = None,
        max_target_cash_weight: float | None = None,
        min_target_risk_symbol_weight: float | None = None,
        max_target_risk_symbol_weight: float | None = None,
        min_target_mean_reversion_weight: float | None = None,
        max_target_mean_reversion_weight: float | None = None,
        min_delta_cash_weight: float | None = None,
        max_delta_cash_weight: float | None = None,
        min_delta_risk_symbol_weight: float | None = None,
        max_delta_risk_symbol_weight: float | None = None,
        min_delta_mean_reversion_weight: float | None = None,
        max_delta_mean_reversion_weight: float | None = None,
        min_proposed_turnover: float | None = None,
        max_proposed_turnover: float | None = None,
        min_pre_trade_relative_wealth_ratio: float | None = None,
        max_pre_trade_relative_wealth_ratio: float | None = None,
        apply_penalty_state_filters: bool = True,
        require_risk_symbol_increase: bool = True,
        cash_max_threshold: float | None = None,
    ) -> bool:
        if not window_active:
            return False
        if min_pre_trade_cash_weight is not None and (
            pre_trade_cash_weight < float(min_pre_trade_cash_weight) - self.config.epsilon
        ):
            return False
        if min_target_cash_weight is not None and (
            target_cash_weight < float(min_target_cash_weight) - self.config.epsilon
        ):
            return False
        if max_target_cash_weight is not None and (
            target_cash_weight > float(max_target_cash_weight) + self.config.epsilon
        ):
            return False
        if min_target_risk_symbol_weight is not None and (
            target_risk_symbol_weight < float(min_target_risk_symbol_weight) - self.config.epsilon
        ):
            return False
        if max_target_risk_symbol_weight is not None and (
            target_risk_symbol_weight > float(max_target_risk_symbol_weight) + self.config.epsilon
        ):
            return False
        if min_target_mean_reversion_weight is not None:
            if target_mean_reversion_weight is None:
                return False
            if target_mean_reversion_weight < float(min_target_mean_reversion_weight) - self.config.epsilon:
                return False
        if max_target_mean_reversion_weight is not None:
            if target_mean_reversion_weight is None:
                return False
            if target_mean_reversion_weight > float(max_target_mean_reversion_weight) + self.config.epsilon:
                return False
        if min_delta_cash_weight is not None:
            if delta_cash_weight is None:
                return False
            if delta_cash_weight < float(min_delta_cash_weight) - self.config.epsilon:
                return False
        if max_delta_cash_weight is not None:
            if delta_cash_weight is None:
                return False
            if delta_cash_weight > float(max_delta_cash_weight) + self.config.epsilon:
                return False
        if min_delta_risk_symbol_weight is not None:
            if delta_risk_symbol_weight is None:
                return False
            if delta_risk_symbol_weight < float(min_delta_risk_symbol_weight) - self.config.epsilon:
                return False
        if max_delta_risk_symbol_weight is not None:
            if delta_risk_symbol_weight is None:
                return False
            if delta_risk_symbol_weight > float(max_delta_risk_symbol_weight) + self.config.epsilon:
                return False
        if min_delta_mean_reversion_weight is not None:
            if delta_mean_reversion_weight is None:
                return False
            if delta_mean_reversion_weight < float(min_delta_mean_reversion_weight) - self.config.epsilon:
                return False
        if max_delta_mean_reversion_weight is not None:
            if delta_mean_reversion_weight is None:
                return False
            if delta_mean_reversion_weight > float(max_delta_mean_reversion_weight) + self.config.epsilon:
                return False
        if min_proposed_turnover is not None:
            if proposed_turnover is None:
                return False
            if proposed_turnover < float(min_proposed_turnover) - self.config.epsilon:
                return False
        if max_proposed_turnover is not None:
            if proposed_turnover is None:
                return False
            if proposed_turnover > float(max_proposed_turnover) + self.config.epsilon:
                return False
        resolved_cash_max_threshold = (
            self.early_rebalance_risk_penalty_cash_max_threshold
            if cash_max_threshold is None
            else float(cash_max_threshold)
        )
        if resolved_cash_max_threshold is None:
            return False
        if pre_trade_cash_weight > float(resolved_cash_max_threshold) + self.config.epsilon:
            return False
        if apply_penalty_state_filters:
            if (
                self.early_rebalance_risk_penalty_symbol_min_weight is not None
                and pre_trade_risk_symbol_weight
                < float(self.early_rebalance_risk_penalty_symbol_min_weight) - self.config.epsilon
            ):
                return False
            if (
                self.early_rebalance_risk_penalty_symbol_max_weight is not None
                and pre_trade_risk_symbol_weight
                > float(self.early_rebalance_risk_penalty_symbol_max_weight) + self.config.epsilon
            ):
                return False
        if require_risk_symbol_increase and (
            target_risk_symbol_weight <= pre_trade_risk_symbol_weight + self.config.epsilon
        ):
            return False
        if benchmark_cumulative_return_max_threshold is not None and (
            benchmark_regime_cumulative_return is None
            or benchmark_regime_cumulative_return
            > float(benchmark_cumulative_return_max_threshold) + self.config.epsilon
        ):
            return False
        if benchmark_drawdown_min_threshold is not None and (
            benchmark_regime_drawdown is None
            or benchmark_regime_drawdown < float(benchmark_drawdown_min_threshold) - self.config.epsilon
        ):
            return False
        if min_pre_trade_relative_wealth_ratio is not None and (
            pre_trade_relative_wealth_ratio is None
            or pre_trade_relative_wealth_ratio < float(min_pre_trade_relative_wealth_ratio) - self.config.epsilon
        ):
            return False
        if max_pre_trade_relative_wealth_ratio is not None and (
            pre_trade_relative_wealth_ratio is None
            or pre_trade_relative_wealth_ratio > float(max_pre_trade_relative_wealth_ratio) + self.config.epsilon
        ):
            return False
        if apply_penalty_state_filters:
            if benchmark_drawdown_max_threshold is not None and (
                benchmark_regime_drawdown is None
                or benchmark_regime_drawdown > float(benchmark_drawdown_max_threshold) + self.config.epsilon
            ):
                return False
        return True

    def _early_rebalance_risk_mean_reversion_condition_met(
        self,
        *,
        base_condition_met: bool,
        pre_trade_mean_reversion_weight: float,
        target_mean_reversion_weight: float,
    ) -> bool:
        if not base_condition_met:
            return False
        if (
            self.early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold is not None
            and target_mean_reversion_weight
            < float(self.early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold)
            - self.config.epsilon
        ):
            return False
        if (
            self.early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold is not None
            and pre_trade_mean_reversion_weight
            < float(self.early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold)
            - self.config.epsilon
        ):
            return False
        delta_mean_reversion = target_mean_reversion_weight - pre_trade_mean_reversion_weight
        if (
            self.early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold is not None
            and delta_mean_reversion
            < float(self.early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold)
            - self.config.epsilon
        ):
            return False
        return True

    def _early_rebalance_risk_trend_condition_met(
        self,
        *,
        base_condition_met: bool,
        pre_trade_trend_weight: float,
        target_trend_weight: float,
    ) -> bool:
        if not base_condition_met:
            return False
        if (
            self.early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold is not None
            and target_trend_weight
            < float(self.early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold)
            - self.config.epsilon
        ):
            return False
        if (
            self.early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold is not None
            and pre_trade_trend_weight
            < float(self.early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold)
            - self.config.epsilon
        ):
            return False
        delta_trend = target_trend_weight - pre_trade_trend_weight
        if (
            self.early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold is not None
            and delta_trend < float(self.early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold) - self.config.epsilon
        ):
            return False
        return True

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.episode_start_step, self.episode_end_step = self._resolve_episode_window(options)
        self.current_step = self.episode_start_step
        self.current_weights = self.initial_weights.copy()
        self.current_wealth = 1.0
        self.current_benchmark_wealth = 1.0
        self.peak_wealth = 1.0
        self.executed_rebalances = 0
        self.early_rebalance_risk_turnover_cap_applications = 0
        self.rebalance_cooldown_remaining = 0
        self.last_executed_rebalance_cash_reduction = 0.0
        self.last_executed_rebalance_symbol_increase = 0.0
        self.last_executed_rebalance_pre_trade_relative_wealth_ratio = None
        self.cumulative_executed_rebalance_cash_reduction = 0.0
        self.cumulative_executed_rebalance_symbol_increase = 0.0
        observation = self._build_observation()
        info = {
            "current_weights": self.current_weights.to_dict(),
            "wealth": self.current_wealth,
            "episode_start_step": self.episode_start_step,
            "episode_end_step": self.episode_end_step,
            "executed_rebalances": self.executed_rebalances,
            "max_executed_rebalances": self.max_executed_rebalances,
            "rebalance_cooldown_steps": self.rebalance_cooldown_steps,
            "rebalance_cooldown_remaining": self.rebalance_cooldown_remaining,
            "early_rebalance_risk_penalty": self.early_rebalance_risk_penalty,
            "early_rebalance_risk_turnover_cap": self.early_rebalance_risk_turnover_cap,
            "early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold": self.early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold,
            "early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold": self.early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold,
            "early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight": self.early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight,
            "early_rebalance_risk_turnover_cap_target_cash_min_threshold": self.early_rebalance_risk_turnover_cap_target_cash_min_threshold,
            "early_rebalance_risk_turnover_cap_target_cash_max_threshold": self.early_rebalance_risk_turnover_cap_target_cash_max_threshold,
            "early_rebalance_risk_turnover_cap_target_trend_min_threshold": self.early_rebalance_risk_turnover_cap_target_trend_min_threshold,
            "early_rebalance_risk_turnover_cap_target_trend_max_threshold": self.early_rebalance_risk_turnover_cap_target_trend_max_threshold,
            "early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold": self.early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold,
            "early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold": self.early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold,
            "early_rebalance_risk_turnover_cap_delta_cash_min_threshold": self.early_rebalance_risk_turnover_cap_delta_cash_min_threshold,
            "early_rebalance_risk_turnover_cap_delta_cash_max_threshold": self.early_rebalance_risk_turnover_cap_delta_cash_max_threshold,
            "early_rebalance_risk_turnover_cap_delta_trend_min_threshold": self.early_rebalance_risk_turnover_cap_delta_trend_min_threshold,
            "early_rebalance_risk_turnover_cap_delta_trend_max_threshold": self.early_rebalance_risk_turnover_cap_delta_trend_max_threshold,
            "early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold": self.early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold,
            "early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold": self.early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold,
            "early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold": self.early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold,
            "early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold": self.early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold,
            "early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio": self.early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio,
            "early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio": self.early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio,
            "early_rebalance_risk_turnover_cap_allow_nonincreasing_risk_symbol": self.early_rebalance_risk_turnover_cap_allow_nonincreasing_risk_symbol,
            "early_rebalance_risk_turnover_cap_use_penalty_state_filters": self.early_rebalance_risk_turnover_cap_use_penalty_state_filters,
            "early_rebalance_risk_turnover_cap_after": self.early_rebalance_risk_turnover_cap_after,
            "early_rebalance_risk_turnover_cap_before": self.early_rebalance_risk_turnover_cap_before,
            "early_rebalance_risk_turnover_cap_max_applications": self.early_rebalance_risk_turnover_cap_max_applications,
            "early_rebalance_risk_turnover_cap_secondary_cap": self.early_rebalance_risk_turnover_cap_secondary_cap,
            "early_rebalance_risk_turnover_cap_secondary_after_applications": self.early_rebalance_risk_turnover_cap_secondary_after_applications,
            "early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold": self.early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold,
            "early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio": self.early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio,
            "early_rebalance_risk_turnover_cap_applications": self.early_rebalance_risk_turnover_cap_applications,
            "early_rebalance_risk_turnover_cap_max_applications_reached": (
                self.early_rebalance_risk_turnover_cap_max_applications is not None
                and self.early_rebalance_risk_turnover_cap_applications
                >= self.early_rebalance_risk_turnover_cap_max_applications
            ),
            "early_rebalance_risk_deep_drawdown_turnover_cap": self.early_rebalance_risk_deep_drawdown_turnover_cap,
            "early_rebalance_risk_deep_drawdown_turnover_cap_after": self.early_rebalance_risk_deep_drawdown_turnover_cap_after,
            "early_rebalance_risk_deep_drawdown_turnover_cap_before": self.early_rebalance_risk_deep_drawdown_turnover_cap_before,
            "early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold": self.early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold,
            "early_rebalance_risk_shallow_drawdown_turnover_cap": self.early_rebalance_risk_shallow_drawdown_turnover_cap,
            "early_rebalance_risk_shallow_drawdown_turnover_cap_after": self.early_rebalance_risk_shallow_drawdown_turnover_cap_after,
            "early_rebalance_risk_shallow_drawdown_turnover_cap_before": self.early_rebalance_risk_shallow_drawdown_turnover_cap_before,
            "early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold": self.early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold,
            "early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold": self.early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold,
            "early_rebalance_risk_mean_reversion_turnover_cap": self.early_rebalance_risk_mean_reversion_turnover_cap,
            "early_rebalance_risk_mean_reversion_action_smoothing": self.early_rebalance_risk_mean_reversion_action_smoothing,
            "early_rebalance_risk_mean_reversion_turnover_cap_after": self.early_rebalance_risk_mean_reversion_turnover_cap_after,
            "early_rebalance_risk_mean_reversion_turnover_cap_before": self.early_rebalance_risk_mean_reversion_turnover_cap_before,
            "early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold": self.early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold,
            "early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold": self.early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold,
            "early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold": self.early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold,
            "early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold": self.early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold,
            "early_rebalance_risk_trend_turnover_cap": self.early_rebalance_risk_trend_turnover_cap,
            "early_rebalance_risk_trend_turnover_cap_after": self.early_rebalance_risk_trend_turnover_cap_after,
            "early_rebalance_risk_trend_turnover_cap_before": self.early_rebalance_risk_trend_turnover_cap_before,
            "early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold": self.early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold,
            "early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold": self.early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold,
            "early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold": self.early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold,
            "early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold": self.early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold,
            "early_rebalance_risk_repeat_turnover_cap": self.early_rebalance_risk_repeat_turnover_cap,
            "early_rebalance_risk_repeat_action_smoothing": self.early_rebalance_risk_repeat_action_smoothing,
            "early_rebalance_risk_repeat_turnover_cap_after": self.early_rebalance_risk_repeat_turnover_cap_after,
            "early_rebalance_risk_repeat_turnover_cap_before": self.early_rebalance_risk_repeat_turnover_cap_before,
            "early_rebalance_risk_repeat_symbol": self.early_rebalance_risk_repeat_symbol,
            "early_rebalance_risk_repeat_previous_cash_reduction_min": self.early_rebalance_risk_repeat_previous_cash_reduction_min,
            "early_rebalance_risk_repeat_previous_symbol_increase_min": self.early_rebalance_risk_repeat_previous_symbol_increase_min,
            "early_rebalance_risk_repeat_unrecovered_turnover_cap": self.early_rebalance_risk_repeat_unrecovered_turnover_cap,
            "early_rebalance_risk_repeat_unrecovered_turnover_cap_after": self.early_rebalance_risk_repeat_unrecovered_turnover_cap_after,
            "early_rebalance_risk_repeat_unrecovered_turnover_cap_before": self.early_rebalance_risk_repeat_unrecovered_turnover_cap_before,
            "early_rebalance_risk_repeat_unrecovered_symbol": self.early_rebalance_risk_repeat_unrecovered_symbol,
            "early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min": self.early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min,
            "early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min": self.early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min,
            "early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery": self.early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery,
            "early_rebalance_risk_cumulative_turnover_cap": self.early_rebalance_risk_cumulative_turnover_cap,
            "early_rebalance_risk_cumulative_turnover_cap_after": self.early_rebalance_risk_cumulative_turnover_cap_after,
            "early_rebalance_risk_cumulative_turnover_cap_before": self.early_rebalance_risk_cumulative_turnover_cap_before,
            "early_rebalance_risk_cumulative_symbol": self.early_rebalance_risk_cumulative_symbol,
            "early_rebalance_risk_cumulative_cash_reduction_budget": self.early_rebalance_risk_cumulative_cash_reduction_budget,
            "early_rebalance_risk_cumulative_symbol_increase_budget": self.early_rebalance_risk_cumulative_symbol_increase_budget,
            "last_executed_rebalance_cash_reduction": self.last_executed_rebalance_cash_reduction,
            "last_executed_rebalance_symbol_increase": self.last_executed_rebalance_symbol_increase,
            "last_executed_rebalance_pre_trade_relative_wealth_ratio": self.last_executed_rebalance_pre_trade_relative_wealth_ratio,
            "cumulative_executed_rebalance_cash_reduction": self.cumulative_executed_rebalance_cash_reduction,
            "cumulative_executed_rebalance_symbol_increase": self.cumulative_executed_rebalance_symbol_increase,
            "early_rebalance_risk_penalty_after": self.early_rebalance_risk_penalty_after,
            "early_rebalance_risk_penalty_before": self.early_rebalance_risk_penalty_before,
            "early_rebalance_risk_penalty_cash_max_threshold": self.early_rebalance_risk_penalty_cash_max_threshold,
            "early_rebalance_risk_penalty_symbol": self.early_rebalance_risk_penalty_symbol,
            "early_rebalance_risk_penalty_symbol_min_weight": self.early_rebalance_risk_penalty_symbol_min_weight,
            "early_rebalance_risk_penalty_symbol_max_weight": self.early_rebalance_risk_penalty_symbol_max_weight,
            "early_rebalance_risk_penalty_benchmark_drawdown_min_threshold": self.early_rebalance_risk_penalty_benchmark_drawdown_min_threshold,
            "early_rebalance_risk_penalty_benchmark_drawdown_max_threshold": self.early_rebalance_risk_penalty_benchmark_drawdown_max_threshold,
            "early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio": self.early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio,
            "early_benchmark_euphoria_penalty": self.early_benchmark_euphoria_penalty,
            "early_benchmark_euphoria_turnover_cap": self.early_benchmark_euphoria_turnover_cap,
            "early_benchmark_euphoria_before": self.early_benchmark_euphoria_before,
            "early_benchmark_euphoria_benchmark_drawdown_min_threshold": self.early_benchmark_euphoria_benchmark_drawdown_min_threshold,
            "early_benchmark_euphoria_symbol": self.early_benchmark_euphoria_symbol,
            "late_rebalance_penalty": self.late_rebalance_penalty,
            "late_rebalance_penalty_after": self.late_rebalance_penalty_after,
            "late_rebalance_gate_after": self.late_rebalance_gate_after,
            "late_rebalance_gate_cash_threshold": self.late_rebalance_gate_cash_threshold,
            "late_rebalance_gate_target_cash_min_threshold": self.late_rebalance_gate_target_cash_min_threshold,
            "late_rebalance_gate_symbol": self.late_rebalance_gate_symbol,
            "late_rebalance_gate_symbol_max_weight": self.late_rebalance_gate_symbol_max_weight,
            "late_defensive_posture_penalty": self.late_defensive_posture_penalty,
            "late_defensive_posture_penalty_after": self.late_defensive_posture_penalty_after,
            "late_defensive_posture_penalty_cash_min_threshold": self.late_defensive_posture_penalty_cash_min_threshold,
            "late_defensive_posture_penalty_symbol": self.late_defensive_posture_penalty_symbol,
            "late_defensive_posture_penalty_symbol_max_weight": self.late_defensive_posture_penalty_symbol_max_weight,
            "late_trend_mean_reversion_conflict_penalty": self.late_trend_mean_reversion_conflict_penalty,
            "late_trend_mean_reversion_conflict_penalty_after": self.late_trend_mean_reversion_conflict_penalty_after,
            "late_trend_mean_reversion_conflict_trend_symbol": self.late_trend_mean_reversion_conflict_trend_symbol,
            "late_trend_mean_reversion_conflict_trend_min_weight": self.late_trend_mean_reversion_conflict_trend_min_weight,
            "late_trend_mean_reversion_conflict_mean_reversion_symbol": self.late_trend_mean_reversion_conflict_mean_reversion_symbol,
            "late_trend_mean_reversion_conflict_mean_reversion_min_weight": self.late_trend_mean_reversion_conflict_mean_reversion_min_weight,
            "state_trend_preservation_symbol": self.state_trend_preservation_symbol,
            "state_trend_preservation_cash_max_threshold": self.state_trend_preservation_cash_max_threshold,
            "state_trend_preservation_symbol_min_weight": self.state_trend_preservation_symbol_min_weight,
            "state_trend_preservation_max_symbol_reduction": self.state_trend_preservation_max_symbol_reduction,
            "benchmark_relative_observations": self.benchmark_relative_observations,
            "benchmark_regime_observations": self.benchmark_regime_observations,
            "benchmark_regime_summary_observations": self.benchmark_regime_summary_observations,
            "benchmark_regime_relative_cumulative_observations": self.benchmark_regime_relative_cumulative_observations,
        }
        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.current_step >= self.episode_end_step:
            raise RuntimeError("Episode is complete. Call reset() before stepping again.")

        action_vector = np.asarray(action, dtype=float)
        if action_vector.shape != (self.asset_count,):
            raise ValueError("Action shape does not match the asset universe.")

        step_label = self.returns.index[self.current_step]
        pre_trade_weights = self.current_weights.copy()
        proposed_weight_vector = _project_to_weight_bounds(action_vector, self.lower_bounds, self.upper_bounds)
        if self.action_smoothing > 0.0:
            proposed_weight_vector = (1.0 - self.action_smoothing) * proposed_weight_vector + self.action_smoothing * pre_trade_weights.to_numpy(dtype=float)
        target_weight_vector = _project_to_weight_bounds(proposed_weight_vector, self.lower_bounds, self.upper_bounds)
        proposed_weights = pd.Series(target_weight_vector, index=self.returns.columns, name="proposed_weight")
        proposed_turnover = compute_tradable_turnover(proposed_weights, self.current_weights, self.config.cash_symbol)
        trade_suppressed = False
        rebalance_budget_exhausted = (
            self.max_executed_rebalances is not None and self.executed_rebalances >= self.max_executed_rebalances
        )
        rebalance_cooldown_active = self.rebalance_cooldown_remaining > 0
        rebalance_cooldown_blocked = False
        pre_trade_cash_weight = float(pre_trade_weights.iloc[self.cash_index]) if self.cash_index is not None else 0.0
        pre_trade_risk_symbol_weight = (
            float(pre_trade_weights.iloc[self.early_rebalance_risk_penalty_symbol_index])
            if self.early_rebalance_risk_penalty_symbol_index is not None
            else 0.0
        )
        pre_trade_repeat_symbol_weight = (
            float(pre_trade_weights.iloc[self.early_rebalance_risk_repeat_symbol_index])
            if self.early_rebalance_risk_repeat_symbol_index is not None
            else 0.0
        )
        pre_trade_repeat_unrecovered_symbol_weight = (
            float(pre_trade_weights.iloc[self.early_rebalance_risk_repeat_unrecovered_symbol_index])
            if self.early_rebalance_risk_repeat_unrecovered_symbol_index is not None
            else 0.0
        )
        pre_trade_cumulative_symbol_weight = (
            float(pre_trade_weights.iloc[self.early_rebalance_risk_cumulative_symbol_index])
            if self.early_rebalance_risk_cumulative_symbol_index is not None
            else 0.0
        )
        pre_trade_mean_reversion_weight = (
            float(pre_trade_weights.iloc[self.early_rebalance_risk_mean_reversion_turnover_cap_symbol_index])
            if self.early_rebalance_risk_mean_reversion_turnover_cap_symbol_index is not None
            else 0.0
        )
        pre_trade_trend_weight = (
            float(pre_trade_weights.iloc[self.early_rebalance_risk_trend_turnover_cap_symbol_index])
            if self.early_rebalance_risk_trend_turnover_cap_symbol_index is not None
            else 0.0
        )
        early_rebalance_risk_after = (
            self.early_rebalance_risk_penalty_after if self.early_rebalance_risk_penalty_after is not None else 0
        )
        early_rebalance_risk_window_active = (
            self.early_rebalance_risk_penalty_before is not None
            and self.executed_rebalances >= early_rebalance_risk_after
            and self.executed_rebalances < self.early_rebalance_risk_penalty_before
            and proposed_turnover > 1e-12
        )
        early_rebalance_risk_turnover_cap_after = (
            self.early_rebalance_risk_turnover_cap_after
            if self.early_rebalance_risk_turnover_cap_after is not None
            else early_rebalance_risk_after
        )
        early_rebalance_risk_turnover_cap_before = (
            self.early_rebalance_risk_turnover_cap_before
            if self.early_rebalance_risk_turnover_cap_before is not None
            else self.early_rebalance_risk_penalty_before
        )
        early_rebalance_risk_turnover_cap_window_active = (
            early_rebalance_risk_turnover_cap_before is not None
            and self.executed_rebalances >= early_rebalance_risk_turnover_cap_after
            and self.executed_rebalances < early_rebalance_risk_turnover_cap_before
            and proposed_turnover > 1e-12
        )
        early_rebalance_risk_deep_drawdown_turnover_cap_after = (
            self.early_rebalance_risk_deep_drawdown_turnover_cap_after
            if self.early_rebalance_risk_deep_drawdown_turnover_cap_after is not None
            else 0
        )
        early_rebalance_risk_deep_drawdown_turnover_cap_window_active = (
            self.early_rebalance_risk_deep_drawdown_turnover_cap_before is not None
            and self.executed_rebalances >= early_rebalance_risk_deep_drawdown_turnover_cap_after
            and self.executed_rebalances < self.early_rebalance_risk_deep_drawdown_turnover_cap_before
            and proposed_turnover > 1e-12
        )
        early_rebalance_risk_condition_met = False
        early_rebalance_risk_turnover_cap_condition_met = False
        early_rebalance_risk_turnover_cap_applied = False
        early_rebalance_risk_turnover_cap_max_applications_reached = (
            self.early_rebalance_risk_turnover_cap_max_applications is not None
            and self.early_rebalance_risk_turnover_cap_applications
            >= self.early_rebalance_risk_turnover_cap_max_applications
        )
        early_rebalance_risk_turnover_cap_secondary_after_applications_reached = (
            self.early_rebalance_risk_turnover_cap_secondary_cap is not None
            and self.early_rebalance_risk_turnover_cap_secondary_after_applications is not None
            and self.early_rebalance_risk_turnover_cap_applications
            >= self.early_rebalance_risk_turnover_cap_secondary_after_applications
        )
        early_rebalance_risk_turnover_cap_secondary_state_condition_met = True
        early_rebalance_risk_turnover_cap_secondary_active = False
        early_rebalance_risk_turnover_cap_effective_cap = self.early_rebalance_risk_turnover_cap
        early_rebalance_risk_deep_drawdown_turnover_cap_condition_met = False
        early_rebalance_risk_deep_drawdown_turnover_cap_applied = False
        early_rebalance_risk_shallow_drawdown_turnover_cap_after = (
            self.early_rebalance_risk_shallow_drawdown_turnover_cap_after
            if self.early_rebalance_risk_shallow_drawdown_turnover_cap_after is not None
            else 0
        )
        early_rebalance_risk_shallow_drawdown_turnover_cap_window_active = (
            self.early_rebalance_risk_shallow_drawdown_turnover_cap_before is not None
            and self.executed_rebalances >= early_rebalance_risk_shallow_drawdown_turnover_cap_after
            and self.executed_rebalances < self.early_rebalance_risk_shallow_drawdown_turnover_cap_before
            and proposed_turnover > 1e-12
        )
        early_rebalance_risk_shallow_drawdown_turnover_cap_condition_met = False
        early_rebalance_risk_shallow_drawdown_turnover_cap_applied = False
        early_rebalance_risk_mean_reversion_turnover_cap_after = (
            self.early_rebalance_risk_mean_reversion_turnover_cap_after
            if self.early_rebalance_risk_mean_reversion_turnover_cap_after is not None
            else 0
        )
        early_rebalance_risk_mean_reversion_turnover_cap_window_active = (
            self.early_rebalance_risk_mean_reversion_turnover_cap_before is not None
            and self.executed_rebalances >= early_rebalance_risk_mean_reversion_turnover_cap_after
            and self.executed_rebalances < self.early_rebalance_risk_mean_reversion_turnover_cap_before
            and proposed_turnover > 1e-12
        )
        early_rebalance_risk_mean_reversion_turnover_cap_condition_met = False
        early_rebalance_risk_mean_reversion_action_smoothing_applied = False
        early_rebalance_risk_mean_reversion_turnover_cap_applied = False
        early_rebalance_risk_trend_turnover_cap_after = (
            self.early_rebalance_risk_trend_turnover_cap_after
            if self.early_rebalance_risk_trend_turnover_cap_after is not None
            else 0
        )
        early_rebalance_risk_trend_turnover_cap_window_active = (
            self.early_rebalance_risk_trend_turnover_cap_before is not None
            and self.executed_rebalances >= early_rebalance_risk_trend_turnover_cap_after
            and self.executed_rebalances < self.early_rebalance_risk_trend_turnover_cap_before
            and proposed_turnover > 1e-12
        )
        early_rebalance_risk_trend_turnover_cap_condition_met = False
        early_rebalance_risk_trend_turnover_cap_applied = False
        early_rebalance_risk_repeat_turnover_cap_after = (
            self.early_rebalance_risk_repeat_turnover_cap_after
            if self.early_rebalance_risk_repeat_turnover_cap_after is not None
            else 0
        )
        early_rebalance_risk_repeat_turnover_cap_window_active = (
            self.early_rebalance_risk_repeat_turnover_cap_before is not None
            and self.executed_rebalances >= early_rebalance_risk_repeat_turnover_cap_after
            and self.executed_rebalances < self.early_rebalance_risk_repeat_turnover_cap_before
            and proposed_turnover > 1e-12
        )
        early_rebalance_risk_repeat_turnover_cap_condition_met = False
        early_rebalance_risk_repeat_action_smoothing_applied = False
        early_rebalance_risk_repeat_turnover_cap_applied = False
        early_rebalance_risk_repeat_previous_cash_reduction = float(self.last_executed_rebalance_cash_reduction)
        early_rebalance_risk_repeat_previous_symbol_increase = float(self.last_executed_rebalance_symbol_increase)
        early_rebalance_risk_repeat_cash_reduction = 0.0
        early_rebalance_risk_repeat_symbol_increase = 0.0
        early_rebalance_risk_repeat_unrecovered_turnover_cap_after = (
            self.early_rebalance_risk_repeat_unrecovered_turnover_cap_after
            if self.early_rebalance_risk_repeat_unrecovered_turnover_cap_after is not None
            else 0
        )
        early_rebalance_risk_repeat_unrecovered_turnover_cap_window_active = (
            self.early_rebalance_risk_repeat_unrecovered_turnover_cap_before is not None
            and self.executed_rebalances >= early_rebalance_risk_repeat_unrecovered_turnover_cap_after
            and self.executed_rebalances < self.early_rebalance_risk_repeat_unrecovered_turnover_cap_before
            and proposed_turnover > 1e-12
        )
        early_rebalance_risk_repeat_unrecovered_turnover_cap_condition_met = False
        early_rebalance_risk_repeat_unrecovered_turnover_cap_applied = False
        early_rebalance_risk_repeat_unrecovered_previous_cash_reduction = float(
            self.last_executed_rebalance_cash_reduction
        )
        early_rebalance_risk_repeat_unrecovered_previous_symbol_increase = float(
            self.last_executed_rebalance_symbol_increase
        )
        early_rebalance_risk_repeat_unrecovered_previous_pre_trade_relative_wealth_ratio = (
            float(self.last_executed_rebalance_pre_trade_relative_wealth_ratio)
            if self.last_executed_rebalance_pre_trade_relative_wealth_ratio is not None
            else None
        )
        early_rebalance_risk_repeat_unrecovered_relative_wealth_recovery = None
        early_rebalance_risk_repeat_unrecovered_cash_reduction = 0.0
        early_rebalance_risk_repeat_unrecovered_symbol_increase = 0.0
        early_rebalance_risk_cumulative_turnover_cap_after = (
            self.early_rebalance_risk_cumulative_turnover_cap_after
            if self.early_rebalance_risk_cumulative_turnover_cap_after is not None
            else 0
        )
        early_rebalance_risk_cumulative_turnover_cap_window_active = (
            self.early_rebalance_risk_cumulative_turnover_cap_before is not None
            and self.executed_rebalances >= early_rebalance_risk_cumulative_turnover_cap_after
            and self.executed_rebalances < self.early_rebalance_risk_cumulative_turnover_cap_before
            and proposed_turnover > 1e-12
        )
        early_rebalance_risk_cumulative_turnover_cap_condition_met = False
        early_rebalance_risk_cumulative_turnover_cap_applied = False
        early_rebalance_risk_cumulative_prior_cash_reduction = float(
            self.cumulative_executed_rebalance_cash_reduction
        )
        early_rebalance_risk_cumulative_prior_symbol_increase = float(
            self.cumulative_executed_rebalance_symbol_increase
        )
        early_rebalance_risk_cumulative_cash_reduction = 0.0
        early_rebalance_risk_cumulative_symbol_increase = 0.0
        pre_trade_benchmark_euphoria_symbol_weight = (
            float(pre_trade_weights.iloc[self.early_benchmark_euphoria_symbol_index])
            if self.early_benchmark_euphoria_symbol_index is not None
            else 0.0
        )
        early_benchmark_euphoria_window_active = False
        early_benchmark_euphoria_condition_met = False
        early_benchmark_euphoria_turnover_cap_applied = False
        late_rebalance_gate_threshold_reached = (
            self.late_rebalance_gate_after is not None and self.executed_rebalances >= self.late_rebalance_gate_after
        )
        late_rebalance_gate_active = False
        late_rebalance_gate_condition_met = False
        late_rebalance_gate_blocked = False
        late_rebalance_gate_refinement_condition_met = False
        late_rebalance_gate_cash_reduction = 0.0
        late_rebalance_gate_symbol_increase = 0.0
        benchmark_regime_cumulative_return = None
        benchmark_regime_drawdown = None
        pre_trade_relative_wealth_ratio = None
        if self.benchmark_returns is not None:
            window_end = min(self.current_step, len(self.benchmark_returns))
            window_start = max(0, window_end - self.lookback)
            benchmark_window = self.benchmark_returns.iloc[window_start:window_end].to_numpy(dtype=float)
            benchmark_regime_cumulative_return, _, benchmark_regime_drawdown = _summarize_return_window(benchmark_window)
            pre_trade_relative_wealth_ratio = self.current_wealth / max(self.current_benchmark_wealth, self.config.epsilon)
        if (
            self.early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold is not None
            and (
                benchmark_regime_cumulative_return is None
                or benchmark_regime_cumulative_return
                < self.early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold
            )
        ):
            early_rebalance_risk_turnover_cap_secondary_state_condition_met = False
        if (
            self.early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio is not None
            and (
                pre_trade_relative_wealth_ratio is None
                or pre_trade_relative_wealth_ratio
                > self.early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio
            )
        ):
            early_rebalance_risk_turnover_cap_secondary_state_condition_met = False
        early_rebalance_risk_turnover_cap_secondary_active = (
            early_rebalance_risk_turnover_cap_secondary_after_applications_reached
            and early_rebalance_risk_turnover_cap_secondary_state_condition_met
        )
        early_rebalance_risk_turnover_cap_effective_cap = (
            self.early_rebalance_risk_turnover_cap_secondary_cap
            if early_rebalance_risk_turnover_cap_secondary_active
            else self.early_rebalance_risk_turnover_cap
        )
        early_benchmark_euphoria_window_active = (
            self.early_benchmark_euphoria_before is not None
            and self.executed_rebalances < self.early_benchmark_euphoria_before
            and proposed_turnover > 1e-12
            and benchmark_regime_drawdown is not None
        )
        if proposed_turnover < self.no_trade_band:
            trade_suppressed = True
            target_weights = self.current_weights.copy().rename("target_weight")
            target_weight_vector = target_weights.to_numpy(dtype=float)
        elif rebalance_budget_exhausted:
            trade_suppressed = True
            target_weights = self.current_weights.copy().rename("target_weight")
            target_weight_vector = target_weights.to_numpy(dtype=float)
        elif rebalance_cooldown_active:
            trade_suppressed = True
            rebalance_cooldown_blocked = proposed_turnover > 1e-12
            target_weights = self.current_weights.copy().rename("target_weight")
            target_weight_vector = target_weights.to_numpy(dtype=float)
        elif late_rebalance_gate_threshold_reached:
            late_rebalance_gate_active = proposed_turnover > 1e-12
            if late_rebalance_gate_active:
                pre_trade_cash_weight = float(pre_trade_weights.iloc[self.cash_index]) if self.cash_index is not None else 0.0
                pre_trade_gate_symbol_weight = (
                    float(pre_trade_weights.iloc[self.late_rebalance_gate_symbol_index])
                    if self.late_rebalance_gate_symbol_index is not None
                    else 0.0
                )
                proposed_cash_weight = float(proposed_weights.iloc[self.cash_index]) if self.cash_index is not None else 0.0
                proposed_gate_symbol_weight = (
                    float(proposed_weights.iloc[self.late_rebalance_gate_symbol_index])
                    if self.late_rebalance_gate_symbol_index is not None
                    else 0.0
                )
                late_rebalance_gate_cash_reduction = pre_trade_cash_weight - proposed_cash_weight
                late_rebalance_gate_symbol_increase = proposed_gate_symbol_weight - pre_trade_gate_symbol_weight
                late_rebalance_gate_refinement_condition_met = True
                late_rebalance_gate_target_cash_condition_met = True
                if (
                    self.late_rebalance_gate_cash_reduction_max is not None
                    and self.late_rebalance_gate_symbol_increase_max is not None
                ):
                    late_rebalance_gate_refinement_condition_met = (
                        late_rebalance_gate_cash_reduction
                        <= float(self.late_rebalance_gate_cash_reduction_max) + self.config.epsilon
                        or late_rebalance_gate_symbol_increase
                        <= float(self.late_rebalance_gate_symbol_increase_max) + self.config.epsilon
                    )
                if self.late_rebalance_gate_target_cash_min_threshold is not None:
                    late_rebalance_gate_target_cash_condition_met = (
                        proposed_cash_weight
                        >= float(self.late_rebalance_gate_target_cash_min_threshold) - self.config.epsilon
                    )
                late_rebalance_gate_condition_met = (
                    pre_trade_cash_weight >= float(self.late_rebalance_gate_cash_threshold)
                    and pre_trade_gate_symbol_weight <= float(self.late_rebalance_gate_symbol_max_weight)
                    and proposed_gate_symbol_weight > pre_trade_gate_symbol_weight + self.config.epsilon
                    and late_rebalance_gate_target_cash_condition_met
                    and late_rebalance_gate_refinement_condition_met
                )
                late_rebalance_gate_blocked = not late_rebalance_gate_condition_met
            if late_rebalance_gate_blocked:
                trade_suppressed = True
                target_weights = self.current_weights.copy().rename("target_weight")
                target_weight_vector = target_weights.to_numpy(dtype=float)
            else:
                target_weights = proposed_weights.rename("target_weight")
        else:
            target_weights = proposed_weights.rename("target_weight")

        initial_target_risk_symbol_weight = (
            float(target_weights.iloc[self.early_rebalance_risk_penalty_symbol_index])
            if self.early_rebalance_risk_penalty_symbol_index is not None
            else 0.0
        )
        initial_target_cash_weight = float(target_weights.iloc[self.cash_index]) if self.cash_index is not None else 0.0
        initial_target_delta_cash_weight = initial_target_cash_weight - pre_trade_cash_weight
        initial_target_mean_reversion_weight = (
            float(target_weights.iloc[self.early_rebalance_risk_mean_reversion_turnover_cap_symbol_index])
            if self.early_rebalance_risk_mean_reversion_turnover_cap_symbol_index is not None
            else 0.0
        )
        initial_target_delta_mean_reversion_weight = (
            initial_target_mean_reversion_weight - pre_trade_mean_reversion_weight
        )
        early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio = (
            self.early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio
        )
        if (
            self.early_rebalance_risk_turnover_cap_use_penalty_state_filters
            and self.early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio is not None
            and (
                early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio is None
                or self.early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio
                > early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio
            )
        ):
            early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio = (
                self.early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio
            )
        early_rebalance_risk_turnover_cap_condition_met = self._early_rebalance_risk_condition_met(
            window_active=early_rebalance_risk_turnover_cap_window_active,
            pre_trade_cash_weight=pre_trade_cash_weight,
            target_cash_weight=initial_target_cash_weight,
            proposed_turnover=proposed_turnover,
            delta_cash_weight=initial_target_delta_cash_weight,
            pre_trade_risk_symbol_weight=pre_trade_risk_symbol_weight,
            target_risk_symbol_weight=initial_target_risk_symbol_weight,
            delta_risk_symbol_weight=initial_target_risk_symbol_weight - pre_trade_risk_symbol_weight,
            benchmark_regime_cumulative_return=benchmark_regime_cumulative_return,
            benchmark_regime_drawdown=benchmark_regime_drawdown,
            pre_trade_relative_wealth_ratio=pre_trade_relative_wealth_ratio,
            benchmark_cumulative_return_max_threshold=self.early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold,
            benchmark_drawdown_min_threshold=self.early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold,
            benchmark_drawdown_max_threshold=(
                self.early_rebalance_risk_penalty_benchmark_drawdown_max_threshold
                if self.early_rebalance_risk_turnover_cap_use_penalty_state_filters
                else None
            ),
            min_pre_trade_cash_weight=self.early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight,
            min_target_cash_weight=self.early_rebalance_risk_turnover_cap_target_cash_min_threshold,
            max_target_cash_weight=self.early_rebalance_risk_turnover_cap_target_cash_max_threshold,
            min_target_risk_symbol_weight=self.early_rebalance_risk_turnover_cap_target_trend_min_threshold,
            max_target_risk_symbol_weight=self.early_rebalance_risk_turnover_cap_target_trend_max_threshold,
            target_mean_reversion_weight=initial_target_mean_reversion_weight,
            min_target_mean_reversion_weight=self.early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold,
            max_target_mean_reversion_weight=self.early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold,
            min_delta_cash_weight=self.early_rebalance_risk_turnover_cap_delta_cash_min_threshold,
            max_delta_cash_weight=self.early_rebalance_risk_turnover_cap_delta_cash_max_threshold,
            min_delta_risk_symbol_weight=self.early_rebalance_risk_turnover_cap_delta_trend_min_threshold,
            max_delta_risk_symbol_weight=self.early_rebalance_risk_turnover_cap_delta_trend_max_threshold,
            delta_mean_reversion_weight=initial_target_delta_mean_reversion_weight,
            min_delta_mean_reversion_weight=self.early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold,
            max_delta_mean_reversion_weight=self.early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold,
            min_proposed_turnover=self.early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold,
            max_proposed_turnover=self.early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold,
            min_pre_trade_relative_wealth_ratio=early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio,
            max_pre_trade_relative_wealth_ratio=self.early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio,
            apply_penalty_state_filters=self.early_rebalance_risk_turnover_cap_use_penalty_state_filters,
            require_risk_symbol_increase=not self.early_rebalance_risk_turnover_cap_allow_nonincreasing_risk_symbol,
        )
        if (
            early_rebalance_risk_turnover_cap_condition_met
            and early_rebalance_risk_turnover_cap_effective_cap is not None
            and not early_rebalance_risk_turnover_cap_max_applications_reached
        ):
            capped_target_weights = _cap_weight_transition_turnover(
                target_weights,
                self.current_weights,
                float(early_rebalance_risk_turnover_cap_effective_cap),
                self.config.cash_symbol,
                self.config.epsilon,
            )
            capped_turnover = compute_tradable_turnover(capped_target_weights, self.current_weights, self.config.cash_symbol)
            uncapped_turnover = compute_tradable_turnover(target_weights, self.current_weights, self.config.cash_symbol)
            if capped_turnover < uncapped_turnover - self.config.epsilon:
                target_weights = capped_target_weights.rename("target_weight")
                target_weight_vector = target_weights.to_numpy(dtype=float)
                early_rebalance_risk_turnover_cap_applied = True
                self.early_rebalance_risk_turnover_cap_applications += 1
                if capped_turnover <= self.config.epsilon:
                    trade_suppressed = True
        early_rebalance_risk_turnover_cap_max_applications_reached = (
            self.early_rebalance_risk_turnover_cap_max_applications is not None
            and self.early_rebalance_risk_turnover_cap_applications
            >= self.early_rebalance_risk_turnover_cap_max_applications
        )

        initial_target_mean_reversion_risk_symbol_weight = (
            float(target_weights.iloc[self.early_rebalance_risk_penalty_symbol_index])
            if self.early_rebalance_risk_penalty_symbol_index is not None
            else 0.0
        )
        initial_target_mean_reversion_weight = (
            float(target_weights.iloc[self.early_rebalance_risk_mean_reversion_turnover_cap_symbol_index])
            if self.early_rebalance_risk_mean_reversion_turnover_cap_symbol_index is not None
            else 0.0
        )
        early_rebalance_risk_mean_reversion_base_condition_met = self._early_rebalance_risk_condition_met(
            window_active=early_rebalance_risk_mean_reversion_turnover_cap_window_active,
            pre_trade_cash_weight=pre_trade_cash_weight,
            target_cash_weight=float(target_weights.iloc[self.cash_index]) if self.cash_index is not None else 0.0,
            pre_trade_risk_symbol_weight=pre_trade_risk_symbol_weight,
            target_risk_symbol_weight=initial_target_mean_reversion_risk_symbol_weight,
            benchmark_regime_cumulative_return=benchmark_regime_cumulative_return,
            benchmark_regime_drawdown=benchmark_regime_drawdown,
            pre_trade_relative_wealth_ratio=pre_trade_relative_wealth_ratio,
            benchmark_cumulative_return_max_threshold=(
                self.early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold
                if self.early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold
                is not None
                else self.early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold
            ),
            benchmark_drawdown_min_threshold=self.early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold,
            benchmark_drawdown_max_threshold=(
                self.early_rebalance_risk_penalty_benchmark_drawdown_max_threshold
                if self.early_rebalance_risk_turnover_cap_use_penalty_state_filters
                else None
            ),
            min_pre_trade_relative_wealth_ratio=(
                self.early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio
                if self.early_rebalance_risk_turnover_cap_use_penalty_state_filters
                else None
            ),
            apply_penalty_state_filters=self.early_rebalance_risk_turnover_cap_use_penalty_state_filters,
        )
        early_rebalance_risk_mean_reversion_turnover_cap_condition_met = (
            self._early_rebalance_risk_mean_reversion_condition_met(
                base_condition_met=early_rebalance_risk_mean_reversion_base_condition_met,
                pre_trade_mean_reversion_weight=pre_trade_mean_reversion_weight,
                target_mean_reversion_weight=initial_target_mean_reversion_weight,
            )
        )
        if (
            early_rebalance_risk_mean_reversion_turnover_cap_condition_met
            and self.early_rebalance_risk_mean_reversion_action_smoothing is not None
        ):
            mean_reversion_smoothed_weight_vector = (
                (1.0 - float(self.early_rebalance_risk_mean_reversion_action_smoothing))
                * target_weights.to_numpy(dtype=float)
                + float(self.early_rebalance_risk_mean_reversion_action_smoothing)
                * self.current_weights.to_numpy(dtype=float)
            )
            mean_reversion_smoothed_target_weights = pd.Series(
                mean_reversion_smoothed_weight_vector,
                index=target_weights.index,
                name="target_weight",
            )
            mean_reversion_smoothed_turnover = compute_tradable_turnover(
                mean_reversion_smoothed_target_weights,
                self.current_weights,
                self.config.cash_symbol,
            )
            mean_reversion_uncapped_turnover = compute_tradable_turnover(
                target_weights,
                self.current_weights,
                self.config.cash_symbol,
            )
            if mean_reversion_smoothed_turnover < mean_reversion_uncapped_turnover - self.config.epsilon:
                target_weights = mean_reversion_smoothed_target_weights
                target_weight_vector = target_weights.to_numpy(dtype=float)
                early_rebalance_risk_mean_reversion_action_smoothing_applied = True
                if mean_reversion_smoothed_turnover <= self.config.epsilon:
                    trade_suppressed = True
        if (
            early_rebalance_risk_mean_reversion_turnover_cap_condition_met
            and self.early_rebalance_risk_mean_reversion_turnover_cap is not None
        ):
            mean_reversion_capped_target_weights = _cap_weight_transition_turnover(
                target_weights,
                self.current_weights,
                float(self.early_rebalance_risk_mean_reversion_turnover_cap),
                self.config.cash_symbol,
                self.config.epsilon,
            )
            mean_reversion_capped_turnover = compute_tradable_turnover(
                mean_reversion_capped_target_weights,
                self.current_weights,
                self.config.cash_symbol,
            )
            mean_reversion_uncapped_turnover = compute_tradable_turnover(
                target_weights,
                self.current_weights,
                self.config.cash_symbol,
            )
            if mean_reversion_capped_turnover < mean_reversion_uncapped_turnover - self.config.epsilon:
                target_weights = mean_reversion_capped_target_weights.rename("target_weight")
                target_weight_vector = target_weights.to_numpy(dtype=float)
                early_rebalance_risk_mean_reversion_turnover_cap_applied = True
                if mean_reversion_capped_turnover <= self.config.epsilon:
                    trade_suppressed = True

        initial_target_trend_risk_symbol_weight = (
            float(target_weights.iloc[self.early_rebalance_risk_penalty_symbol_index])
            if self.early_rebalance_risk_penalty_symbol_index is not None
            else 0.0
        )
        initial_target_trend_weight = (
            float(target_weights.iloc[self.early_rebalance_risk_trend_turnover_cap_symbol_index])
            if self.early_rebalance_risk_trend_turnover_cap_symbol_index is not None
            else 0.0
        )
        early_rebalance_risk_trend_base_condition_met = self._early_rebalance_risk_condition_met(
            window_active=early_rebalance_risk_trend_turnover_cap_window_active,
            pre_trade_cash_weight=pre_trade_cash_weight,
            target_cash_weight=float(target_weights.iloc[self.cash_index]) if self.cash_index is not None else 0.0,
            pre_trade_risk_symbol_weight=pre_trade_risk_symbol_weight,
            target_risk_symbol_weight=initial_target_trend_risk_symbol_weight,
            benchmark_regime_cumulative_return=benchmark_regime_cumulative_return,
            benchmark_regime_drawdown=benchmark_regime_drawdown,
            pre_trade_relative_wealth_ratio=pre_trade_relative_wealth_ratio,
            benchmark_cumulative_return_max_threshold=(
                self.early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold
                if self.early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold
                is not None
                else self.early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold
            ),
            benchmark_drawdown_min_threshold=self.early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold,
            benchmark_drawdown_max_threshold=(
                self.early_rebalance_risk_penalty_benchmark_drawdown_max_threshold
                if self.early_rebalance_risk_turnover_cap_use_penalty_state_filters
                else None
            ),
            min_pre_trade_relative_wealth_ratio=(
                self.early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio
                if self.early_rebalance_risk_turnover_cap_use_penalty_state_filters
                else None
            ),
            apply_penalty_state_filters=self.early_rebalance_risk_turnover_cap_use_penalty_state_filters,
        )
        early_rebalance_risk_trend_turnover_cap_condition_met = self._early_rebalance_risk_trend_condition_met(
            base_condition_met=early_rebalance_risk_trend_base_condition_met,
            pre_trade_trend_weight=pre_trade_trend_weight,
            target_trend_weight=initial_target_trend_weight,
        )
        if (
            early_rebalance_risk_trend_turnover_cap_condition_met
            and self.early_rebalance_risk_trend_turnover_cap is not None
        ):
            trend_capped_target_weights = _cap_weight_transition_turnover(
                target_weights,
                self.current_weights,
                float(self.early_rebalance_risk_trend_turnover_cap),
                self.config.cash_symbol,
                self.config.epsilon,
            )
            trend_capped_turnover = compute_tradable_turnover(
                trend_capped_target_weights,
                self.current_weights,
                self.config.cash_symbol,
            )
            trend_uncapped_turnover = compute_tradable_turnover(
                target_weights,
                self.current_weights,
                self.config.cash_symbol,
            )
            if trend_capped_turnover < trend_uncapped_turnover - self.config.epsilon:
                target_weights = trend_capped_target_weights.rename("target_weight")
                target_weight_vector = target_weights.to_numpy(dtype=float)
                early_rebalance_risk_trend_turnover_cap_applied = True
                if trend_capped_turnover <= self.config.epsilon:
                    trade_suppressed = True

        initial_target_shallow_drawdown_risk_symbol_weight = (
            float(target_weights.iloc[self.early_rebalance_risk_penalty_symbol_index])
            if self.early_rebalance_risk_penalty_symbol_index is not None
            else 0.0
        )
        early_rebalance_risk_shallow_drawdown_turnover_cap_condition_met = self._early_rebalance_risk_condition_met(
            window_active=early_rebalance_risk_shallow_drawdown_turnover_cap_window_active,
            pre_trade_cash_weight=pre_trade_cash_weight,
            target_cash_weight=float(target_weights.iloc[self.cash_index]) if self.cash_index is not None else 0.0,
            pre_trade_risk_symbol_weight=pre_trade_risk_symbol_weight,
            target_risk_symbol_weight=initial_target_shallow_drawdown_risk_symbol_weight,
            benchmark_regime_cumulative_return=benchmark_regime_cumulative_return,
            benchmark_regime_drawdown=benchmark_regime_drawdown,
            pre_trade_relative_wealth_ratio=pre_trade_relative_wealth_ratio,
            benchmark_drawdown_min_threshold=self.early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold,
            apply_penalty_state_filters=False,
            cash_max_threshold=self.early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold,
        )
        if (
            early_rebalance_risk_shallow_drawdown_turnover_cap_condition_met
            and self.early_rebalance_risk_shallow_drawdown_turnover_cap is not None
        ):
            shallow_drawdown_capped_target_weights = _cap_weight_transition_turnover(
                target_weights,
                self.current_weights,
                float(self.early_rebalance_risk_shallow_drawdown_turnover_cap),
                self.config.cash_symbol,
                self.config.epsilon,
            )
            shallow_drawdown_capped_turnover = compute_tradable_turnover(
                shallow_drawdown_capped_target_weights,
                self.current_weights,
                self.config.cash_symbol,
            )
            shallow_drawdown_uncapped_turnover = compute_tradable_turnover(
                target_weights,
                self.current_weights,
                self.config.cash_symbol,
            )
            if shallow_drawdown_capped_turnover < shallow_drawdown_uncapped_turnover - self.config.epsilon:
                target_weights = shallow_drawdown_capped_target_weights.rename("target_weight")
                target_weight_vector = target_weights.to_numpy(dtype=float)
                early_rebalance_risk_shallow_drawdown_turnover_cap_applied = True
                if shallow_drawdown_capped_turnover <= self.config.epsilon:
                    trade_suppressed = True

        initial_target_deep_drawdown_risk_symbol_weight = (
            float(target_weights.iloc[self.early_rebalance_risk_penalty_symbol_index])
            if self.early_rebalance_risk_penalty_symbol_index is not None
            else 0.0
        )
        early_rebalance_risk_deep_drawdown_turnover_cap_condition_met = self._early_rebalance_risk_condition_met(
            window_active=early_rebalance_risk_deep_drawdown_turnover_cap_window_active,
            pre_trade_cash_weight=pre_trade_cash_weight,
            target_cash_weight=float(target_weights.iloc[self.cash_index]) if self.cash_index is not None else 0.0,
            pre_trade_risk_symbol_weight=pre_trade_risk_symbol_weight,
            target_risk_symbol_weight=initial_target_deep_drawdown_risk_symbol_weight,
            benchmark_regime_cumulative_return=benchmark_regime_cumulative_return,
            benchmark_regime_drawdown=benchmark_regime_drawdown,
            pre_trade_relative_wealth_ratio=pre_trade_relative_wealth_ratio,
            benchmark_drawdown_max_threshold=self.early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold,
        )
        if (
            early_rebalance_risk_deep_drawdown_turnover_cap_condition_met
            and self.early_rebalance_risk_deep_drawdown_turnover_cap is not None
        ):
            deep_drawdown_capped_target_weights = _cap_weight_transition_turnover(
                target_weights,
                self.current_weights,
                float(self.early_rebalance_risk_deep_drawdown_turnover_cap),
                self.config.cash_symbol,
                self.config.epsilon,
            )
            deep_drawdown_capped_turnover = compute_tradable_turnover(
                deep_drawdown_capped_target_weights,
                self.current_weights,
                self.config.cash_symbol,
            )
            uncapped_turnover = compute_tradable_turnover(target_weights, self.current_weights, self.config.cash_symbol)
            if deep_drawdown_capped_turnover < uncapped_turnover - self.config.epsilon:
                target_weights = deep_drawdown_capped_target_weights.rename("target_weight")
                target_weight_vector = target_weights.to_numpy(dtype=float)
                early_rebalance_risk_deep_drawdown_turnover_cap_applied = True
                if deep_drawdown_capped_turnover <= self.config.epsilon:
                    trade_suppressed = True

        initial_target_benchmark_euphoria_symbol_weight = (
            float(target_weights.iloc[self.early_benchmark_euphoria_symbol_index])
            if self.early_benchmark_euphoria_symbol_index is not None
            else 0.0
        )
        early_benchmark_euphoria_condition_triggered = (
            early_benchmark_euphoria_window_active
            and benchmark_regime_drawdown is not None
            and benchmark_regime_drawdown
            >= float(self.early_benchmark_euphoria_benchmark_drawdown_min_threshold) - self.config.epsilon
            and initial_target_benchmark_euphoria_symbol_weight
            > pre_trade_benchmark_euphoria_symbol_weight + self.config.epsilon
        )
        if early_benchmark_euphoria_condition_triggered and self.early_benchmark_euphoria_turnover_cap is not None:
            capped_target_weights = _cap_weight_transition_turnover(
                target_weights,
                self.current_weights,
                float(self.early_benchmark_euphoria_turnover_cap),
                self.config.cash_symbol,
                self.config.epsilon,
            )
            capped_turnover = compute_tradable_turnover(capped_target_weights, self.current_weights, self.config.cash_symbol)
            uncapped_turnover = compute_tradable_turnover(target_weights, self.current_weights, self.config.cash_symbol)
            if capped_turnover < uncapped_turnover - self.config.epsilon:
                target_weights = capped_target_weights.rename("target_weight")
                target_weight_vector = target_weights.to_numpy(dtype=float)
                early_benchmark_euphoria_turnover_cap_applied = True
                if capped_turnover <= self.config.epsilon:
                    trade_suppressed = True

        state_trend_preservation_window_active = False
        state_trend_preservation_condition_met = False
        state_trend_preservation_guard_applied = False
        if self.state_trend_preservation_symbol_index is not None and self.cash_index is not None:
            pre_trade_trend_preservation_weight = float(pre_trade_weights.iloc[self.state_trend_preservation_symbol_index])
            current_target_trend_preservation_weight = float(target_weights.iloc[self.state_trend_preservation_symbol_index])
            current_target_cash_weight = float(target_weights.iloc[self.cash_index])
            state_trend_preservation_window_active = (
                compute_tradable_turnover(target_weights, self.current_weights, self.config.cash_symbol) > self.config.epsilon
                and pre_trade_cash_weight <= float(self.state_trend_preservation_cash_max_threshold)
                and pre_trade_trend_preservation_weight >= float(self.state_trend_preservation_symbol_min_weight)
            )
            minimum_trend_preservation_weight = max(
                float(self.state_trend_preservation_symbol_min_weight),
                pre_trade_trend_preservation_weight - float(self.state_trend_preservation_max_symbol_reduction),
            )
            state_trend_preservation_condition_met = (
                state_trend_preservation_window_active
                and current_target_cash_weight > pre_trade_cash_weight + self.config.epsilon
                and current_target_trend_preservation_weight < minimum_trend_preservation_weight - self.config.epsilon
            )
            if state_trend_preservation_condition_met:
                guarded_target_weights = _cap_weight_transition_minimum_weight(
                    target_weights,
                    self.current_weights,
                    self.state_trend_preservation_symbol_index,
                    minimum_trend_preservation_weight,
                    self.config.epsilon,
                )
                if float(guarded_target_weights.iloc[self.state_trend_preservation_symbol_index]) > (
                    current_target_trend_preservation_weight + self.config.epsilon
                ):
                    target_weights = guarded_target_weights.rename("target_weight")
                    target_weight_vector = target_weights.to_numpy(dtype=float)
                    state_trend_preservation_guard_applied = True

        if self.early_rebalance_risk_repeat_symbol_index is not None and self.cash_index is not None:
            current_target_repeat_symbol_weight = float(target_weights.iloc[self.early_rebalance_risk_repeat_symbol_index])
            current_target_cash_weight = float(target_weights.iloc[self.cash_index])
            early_rebalance_risk_repeat_cash_reduction = max(pre_trade_cash_weight - current_target_cash_weight, 0.0)
            early_rebalance_risk_repeat_symbol_increase = max(
                current_target_repeat_symbol_weight - pre_trade_repeat_symbol_weight,
                0.0,
            )
            early_rebalance_risk_repeat_turnover_cap_condition_met = (
                early_rebalance_risk_repeat_turnover_cap_window_active
                and early_rebalance_risk_repeat_previous_cash_reduction
                >= float(self.early_rebalance_risk_repeat_previous_cash_reduction_min) - self.config.epsilon
                and early_rebalance_risk_repeat_previous_symbol_increase
                >= float(self.early_rebalance_risk_repeat_previous_symbol_increase_min) - self.config.epsilon
                and early_rebalance_risk_repeat_cash_reduction > self.config.epsilon
                and early_rebalance_risk_repeat_symbol_increase > self.config.epsilon
            )
            if (
                early_rebalance_risk_repeat_turnover_cap_condition_met
                and self.early_rebalance_risk_repeat_action_smoothing is not None
            ):
                repeat_smoothed_weight_vector = (
                    (1.0 - float(self.early_rebalance_risk_repeat_action_smoothing))
                    * target_weights.to_numpy(dtype=float)
                    + float(self.early_rebalance_risk_repeat_action_smoothing)
                    * self.current_weights.to_numpy(dtype=float)
                )
                repeat_smoothed_target_weights = pd.Series(
                    repeat_smoothed_weight_vector,
                    index=target_weights.index,
                    name="target_weight",
                )
                repeat_smoothed_turnover = compute_tradable_turnover(
                    repeat_smoothed_target_weights,
                    self.current_weights,
                    self.config.cash_symbol,
                )
                uncapped_turnover = compute_tradable_turnover(
                    target_weights,
                    self.current_weights,
                    self.config.cash_symbol,
                )
                if repeat_smoothed_turnover < uncapped_turnover - self.config.epsilon:
                    target_weights = repeat_smoothed_target_weights
                    target_weight_vector = target_weights.to_numpy(dtype=float)
                    early_rebalance_risk_repeat_action_smoothing_applied = True
                    current_target_repeat_symbol_weight = float(
                        target_weights.iloc[self.early_rebalance_risk_repeat_symbol_index]
                    )
                    current_target_cash_weight = float(target_weights.iloc[self.cash_index])
                    early_rebalance_risk_repeat_cash_reduction = max(
                        pre_trade_cash_weight - current_target_cash_weight,
                        0.0,
                    )
                    early_rebalance_risk_repeat_symbol_increase = max(
                        current_target_repeat_symbol_weight - pre_trade_repeat_symbol_weight,
                        0.0,
                    )
                    if repeat_smoothed_turnover <= self.config.epsilon:
                        trade_suppressed = True
            if (
                early_rebalance_risk_repeat_turnover_cap_condition_met
                and self.early_rebalance_risk_repeat_turnover_cap is not None
            ):
                repeat_capped_target_weights = _cap_weight_transition_turnover(
                    target_weights,
                    self.current_weights,
                    float(self.early_rebalance_risk_repeat_turnover_cap),
                    self.config.cash_symbol,
                    self.config.epsilon,
                )
                repeat_capped_turnover = compute_tradable_turnover(
                    repeat_capped_target_weights,
                    self.current_weights,
                    self.config.cash_symbol,
                )
                uncapped_turnover = compute_tradable_turnover(target_weights, self.current_weights, self.config.cash_symbol)
                if repeat_capped_turnover < uncapped_turnover - self.config.epsilon:
                    target_weights = repeat_capped_target_weights.rename("target_weight")
                    target_weight_vector = target_weights.to_numpy(dtype=float)
                    early_rebalance_risk_repeat_turnover_cap_applied = True
                    current_target_repeat_symbol_weight = float(target_weights.iloc[self.early_rebalance_risk_repeat_symbol_index])
                    current_target_cash_weight = float(target_weights.iloc[self.cash_index])
                    early_rebalance_risk_repeat_cash_reduction = max(
                        pre_trade_cash_weight - current_target_cash_weight,
                        0.0,
                    )
                    early_rebalance_risk_repeat_symbol_increase = max(
                        current_target_repeat_symbol_weight - pre_trade_repeat_symbol_weight,
                        0.0,
                    )
                    if repeat_capped_turnover <= self.config.epsilon:
                        trade_suppressed = True

        if self.early_rebalance_risk_repeat_unrecovered_symbol_index is not None and self.cash_index is not None:
            current_target_repeat_unrecovered_symbol_weight = float(
                target_weights.iloc[self.early_rebalance_risk_repeat_unrecovered_symbol_index]
            )
            current_target_cash_weight = float(target_weights.iloc[self.cash_index])
            early_rebalance_risk_repeat_unrecovered_cash_reduction = max(
                pre_trade_cash_weight - current_target_cash_weight,
                0.0,
            )
            early_rebalance_risk_repeat_unrecovered_symbol_increase = max(
                current_target_repeat_unrecovered_symbol_weight - pre_trade_repeat_unrecovered_symbol_weight,
                0.0,
            )
            if (
                pre_trade_relative_wealth_ratio is not None
                and early_rebalance_risk_repeat_unrecovered_previous_pre_trade_relative_wealth_ratio is not None
            ):
                early_rebalance_risk_repeat_unrecovered_relative_wealth_recovery = (
                    pre_trade_relative_wealth_ratio
                    - early_rebalance_risk_repeat_unrecovered_previous_pre_trade_relative_wealth_ratio
                )
            early_rebalance_risk_repeat_unrecovered_turnover_cap_condition_met = (
                early_rebalance_risk_repeat_unrecovered_turnover_cap_window_active
                and early_rebalance_risk_repeat_unrecovered_previous_cash_reduction
                >= float(self.early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min)
                - self.config.epsilon
                and early_rebalance_risk_repeat_unrecovered_previous_symbol_increase
                >= float(self.early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min)
                - self.config.epsilon
                and early_rebalance_risk_repeat_unrecovered_cash_reduction > self.config.epsilon
                and early_rebalance_risk_repeat_unrecovered_symbol_increase > self.config.epsilon
                and early_rebalance_risk_repeat_unrecovered_relative_wealth_recovery is not None
                and early_rebalance_risk_repeat_unrecovered_relative_wealth_recovery
                < float(self.early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery)
                - self.config.epsilon
            )
            if (
                early_rebalance_risk_repeat_unrecovered_turnover_cap_condition_met
                and self.early_rebalance_risk_repeat_unrecovered_turnover_cap is not None
            ):
                repeat_unrecovered_capped_target_weights = _cap_weight_transition_turnover(
                    target_weights,
                    self.current_weights,
                    float(self.early_rebalance_risk_repeat_unrecovered_turnover_cap),
                    self.config.cash_symbol,
                    self.config.epsilon,
                )
                repeat_unrecovered_capped_turnover = compute_tradable_turnover(
                    repeat_unrecovered_capped_target_weights,
                    self.current_weights,
                    self.config.cash_symbol,
                )
                uncapped_turnover = compute_tradable_turnover(
                    target_weights,
                    self.current_weights,
                    self.config.cash_symbol,
                )
                if repeat_unrecovered_capped_turnover < uncapped_turnover - self.config.epsilon:
                    target_weights = repeat_unrecovered_capped_target_weights.rename("target_weight")
                    target_weight_vector = target_weights.to_numpy(dtype=float)
                    early_rebalance_risk_repeat_unrecovered_turnover_cap_applied = True
                    current_target_repeat_unrecovered_symbol_weight = float(
                        target_weights.iloc[self.early_rebalance_risk_repeat_unrecovered_symbol_index]
                    )
                    current_target_cash_weight = float(target_weights.iloc[self.cash_index])
                    early_rebalance_risk_repeat_unrecovered_cash_reduction = max(
                        pre_trade_cash_weight - current_target_cash_weight,
                        0.0,
                    )
                    early_rebalance_risk_repeat_unrecovered_symbol_increase = max(
                        current_target_repeat_unrecovered_symbol_weight - pre_trade_repeat_unrecovered_symbol_weight,
                        0.0,
                    )
                    if repeat_unrecovered_capped_turnover <= self.config.epsilon:
                        trade_suppressed = True

        if self.early_rebalance_risk_cumulative_symbol_index is not None and self.cash_index is not None:
            current_target_cumulative_symbol_weight = float(
                target_weights.iloc[self.early_rebalance_risk_cumulative_symbol_index]
            )
            current_target_cash_weight = float(target_weights.iloc[self.cash_index])
            early_rebalance_risk_cumulative_cash_reduction = max(
                pre_trade_cash_weight - current_target_cash_weight,
                0.0,
            )
            early_rebalance_risk_cumulative_symbol_increase = max(
                current_target_cumulative_symbol_weight - pre_trade_cumulative_symbol_weight,
                0.0,
            )
            cumulative_cash_budget_condition_met = (
                self.early_rebalance_risk_cumulative_cash_reduction_budget is None
                or (
                    early_rebalance_risk_cumulative_cash_reduction > self.config.epsilon
                    and early_rebalance_risk_cumulative_prior_cash_reduction
                    + early_rebalance_risk_cumulative_cash_reduction
                    > float(self.early_rebalance_risk_cumulative_cash_reduction_budget) + self.config.epsilon
                )
            )
            cumulative_symbol_budget_condition_met = (
                self.early_rebalance_risk_cumulative_symbol_increase_budget is None
                or (
                    early_rebalance_risk_cumulative_symbol_increase > self.config.epsilon
                    and early_rebalance_risk_cumulative_prior_symbol_increase
                    + early_rebalance_risk_cumulative_symbol_increase
                    > float(self.early_rebalance_risk_cumulative_symbol_increase_budget) + self.config.epsilon
                )
            )
            early_rebalance_risk_cumulative_turnover_cap_condition_met = (
                early_rebalance_risk_cumulative_turnover_cap_window_active
                and early_rebalance_risk_cumulative_symbol_increase > self.config.epsilon
                and cumulative_cash_budget_condition_met
                and cumulative_symbol_budget_condition_met
            )
            if (
                early_rebalance_risk_cumulative_turnover_cap_condition_met
                and self.early_rebalance_risk_cumulative_turnover_cap is not None
            ):
                cumulative_capped_target_weights = _cap_weight_transition_turnover(
                    target_weights,
                    self.current_weights,
                    float(self.early_rebalance_risk_cumulative_turnover_cap),
                    self.config.cash_symbol,
                    self.config.epsilon,
                )
                cumulative_capped_turnover = compute_tradable_turnover(
                    cumulative_capped_target_weights,
                    self.current_weights,
                    self.config.cash_symbol,
                )
                uncapped_turnover = compute_tradable_turnover(
                    target_weights,
                    self.current_weights,
                    self.config.cash_symbol,
                )
                if cumulative_capped_turnover < uncapped_turnover - self.config.epsilon:
                    target_weights = cumulative_capped_target_weights.rename("target_weight")
                    target_weight_vector = target_weights.to_numpy(dtype=float)
                    early_rebalance_risk_cumulative_turnover_cap_applied = True
                    current_target_cumulative_symbol_weight = float(
                        target_weights.iloc[self.early_rebalance_risk_cumulative_symbol_index]
                    )
                    current_target_cash_weight = float(target_weights.iloc[self.cash_index])
                    early_rebalance_risk_cumulative_cash_reduction = max(
                        pre_trade_cash_weight - current_target_cash_weight,
                        0.0,
                    )
                    early_rebalance_risk_cumulative_symbol_increase = max(
                        current_target_cumulative_symbol_weight - pre_trade_cumulative_symbol_weight,
                        0.0,
                    )
                    if cumulative_capped_turnover <= self.config.epsilon:
                        trade_suppressed = True

        late_defensive_posture_window_active = (
            self.late_defensive_posture_penalty_after is not None
            and self.executed_rebalances >= self.late_defensive_posture_penalty_after
        )
        target_defensive_symbol_weight = (
            float(target_weights.iloc[self.late_defensive_posture_penalty_symbol_index])
            if self.late_defensive_posture_penalty_symbol_index is not None
            else 0.0
        )
        target_cash_weight = float(target_weights.iloc[self.cash_index]) if self.cash_index is not None else 0.0
        late_defensive_posture_condition_met = (
            late_defensive_posture_window_active
            and target_cash_weight >= float(self.late_defensive_posture_penalty_cash_min_threshold)
            and target_defensive_symbol_weight <= float(self.late_defensive_posture_penalty_symbol_max_weight)
        )
        late_trend_mean_reversion_conflict_window_active = (
            self.late_trend_mean_reversion_conflict_penalty_after is not None
            and self.executed_rebalances >= self.late_trend_mean_reversion_conflict_penalty_after
        )
        target_conflict_trend_weight = (
            float(target_weights.iloc[self.late_trend_mean_reversion_conflict_trend_symbol_index])
            if self.late_trend_mean_reversion_conflict_trend_symbol_index is not None
            else 0.0
        )
        target_conflict_mean_reversion_weight = (
            float(target_weights.iloc[self.late_trend_mean_reversion_conflict_mean_reversion_symbol_index])
            if self.late_trend_mean_reversion_conflict_mean_reversion_symbol_index is not None
            else 0.0
        )
        late_trend_mean_reversion_conflict_condition_met = (
            late_trend_mean_reversion_conflict_window_active
            and target_conflict_trend_weight
            >= float(self.late_trend_mean_reversion_conflict_trend_min_weight) - self.config.epsilon
            and target_conflict_mean_reversion_weight
            >= float(self.late_trend_mean_reversion_conflict_mean_reversion_min_weight) - self.config.epsilon
        )

        realized_returns = self.returns.iloc[self.current_step]
        turnover = compute_tradable_turnover(target_weights, self.current_weights, self.config.cash_symbol)
        execution_cost = compute_execution_cost(
            turnover,
            self.config.transaction_cost_bps,
            self.config.slippage_bps,
        )
        gross_return_contributions = (target_weights * realized_returns).rename("gross_return_contribution")
        gross_portfolio_return = float(gross_return_contributions.sum())
        gross_wealth_ratio = max(1.0 + gross_portfolio_return, self.config.epsilon)
        net_wealth_ratio = max(1.0 - execution_cost, 0.0) * max(1.0 + gross_portfolio_return, self.config.epsilon)

        gross_reward_component = self._score_step_utility(gross_wealth_ratio, self.config.gain_reward, self.config.loss_penalty)
        reward_core_component = self._score_step_utility(net_wealth_ratio, self.config.gain_reward, self.config.loss_penalty)
        execution_cost_reward_drag = reward_core_component - gross_reward_component

        benchmark_reward_component = 0.0
        relative_wealth_ratio = None
        benchmark_return = None
        if self.benchmark_returns is not None:
            benchmark_return = float(self.benchmark_returns.iloc[self.current_step])
            if self.config.benchmark_gain_reward > 0 or self.config.benchmark_loss_penalty > 0:
                benchmark_wealth_ratio = max(1.0 + benchmark_return, self.config.epsilon)
                relative_wealth_ratio = net_wealth_ratio / benchmark_wealth_ratio
                benchmark_reward_component = self._score_step_utility(
                    relative_wealth_ratio,
                    self.config.benchmark_gain_reward,
                    self.config.benchmark_loss_penalty,
                )

        turnover_penalty_component = 0.0
        if self.config.turnover_penalty > 0:
            turnover_penalty_component = -self.config.turnover_penalty * turnover

        weight_reg_penalty_component = 0.0
        if self.config.weight_reg > 0:
            weight_reg_penalty_component = -self.config.weight_reg * float(np.dot(target_weight_vector, target_weight_vector))

        cash_weight = float(target_weights.get(self.config.cash_symbol, 0.0))
        excess_cash_weight = 0.0
        cash_weight_penalty_component = 0.0
        if self.cash_target_weight is not None:
            excess_cash_weight = max(cash_weight - self.cash_target_weight, 0.0)
            if self.cash_weight_penalty > 0.0:
                cash_weight_penalty_component = -self.cash_weight_penalty * excess_cash_weight

        friction_reward_drag = execution_cost_reward_drag + turnover_penalty_component
        early_rebalance_risk_penalty_component = 0.0
        early_rebalance_risk_penalty_applied = False
        if early_rebalance_risk_window_active:
            target_risk_symbol_weight = (
                float(target_weights.iloc[self.early_rebalance_risk_penalty_symbol_index])
                if self.early_rebalance_risk_penalty_symbol_index is not None
                else 0.0
            )
            early_rebalance_risk_condition_met = self._early_rebalance_risk_condition_met(
                window_active=True,
                pre_trade_cash_weight=pre_trade_cash_weight,
                target_cash_weight=float(target_weights.iloc[self.cash_index]) if self.cash_index is not None else 0.0,
                pre_trade_risk_symbol_weight=pre_trade_risk_symbol_weight,
                target_risk_symbol_weight=target_risk_symbol_weight,
                benchmark_regime_cumulative_return=benchmark_regime_cumulative_return,
                benchmark_regime_drawdown=benchmark_regime_drawdown,
                pre_trade_relative_wealth_ratio=pre_trade_relative_wealth_ratio,
                benchmark_drawdown_min_threshold=self.early_rebalance_risk_penalty_benchmark_drawdown_min_threshold,
                benchmark_drawdown_max_threshold=self.early_rebalance_risk_penalty_benchmark_drawdown_max_threshold,
                min_pre_trade_relative_wealth_ratio=self.early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio,
                apply_penalty_state_filters=True,
            )
            if early_rebalance_risk_condition_met and self.early_rebalance_risk_penalty > 0.0:
                early_rebalance_risk_penalty_component = -self.early_rebalance_risk_penalty
                early_rebalance_risk_penalty_applied = True
        early_benchmark_euphoria_penalty_component = 0.0
        early_benchmark_euphoria_penalty_applied = False
        if early_benchmark_euphoria_window_active:
            target_benchmark_euphoria_symbol_weight = (
                float(target_weights.iloc[self.early_benchmark_euphoria_symbol_index])
                if self.early_benchmark_euphoria_symbol_index is not None
                else 0.0
            )
            early_benchmark_euphoria_condition_met = (
                benchmark_regime_drawdown is not None
                and benchmark_regime_drawdown
                >= float(self.early_benchmark_euphoria_benchmark_drawdown_min_threshold) - self.config.epsilon
                and target_benchmark_euphoria_symbol_weight
                > pre_trade_benchmark_euphoria_symbol_weight + self.config.epsilon
            )
            if early_benchmark_euphoria_condition_met and self.early_benchmark_euphoria_penalty > 0.0:
                early_benchmark_euphoria_penalty_component = -self.early_benchmark_euphoria_penalty
                early_benchmark_euphoria_penalty_applied = True
        late_rebalance_penalty_component = 0.0
        late_rebalance_penalty_applied = False
        late_rebalance_threshold_reached = (
            turnover > 1e-12
            and self.late_rebalance_penalty_after is not None
            and self.executed_rebalances >= self.late_rebalance_penalty_after
        )
        if late_rebalance_threshold_reached and self.late_rebalance_penalty > 0.0:
            late_rebalance_penalty_component = -self.late_rebalance_penalty
            late_rebalance_penalty_applied = True
        late_defensive_posture_penalty_component = 0.0
        late_defensive_posture_penalty_applied = False
        if late_defensive_posture_condition_met and self.late_defensive_posture_penalty > 0.0:
            late_defensive_posture_penalty_component = -self.late_defensive_posture_penalty
            late_defensive_posture_penalty_applied = True
        late_trend_mean_reversion_conflict_penalty_component = 0.0
        late_trend_mean_reversion_conflict_penalty_applied = False
        if late_trend_mean_reversion_conflict_condition_met and self.late_trend_mean_reversion_conflict_penalty > 0.0:
            late_trend_mean_reversion_conflict_penalty_component = -self.late_trend_mean_reversion_conflict_penalty
            late_trend_mean_reversion_conflict_penalty_applied = True
        reward = (
            reward_core_component
            + benchmark_reward_component
            + turnover_penalty_component
            + weight_reg_penalty_component
            + cash_weight_penalty_component
            + early_rebalance_risk_penalty_component
            + early_benchmark_euphoria_penalty_component
            + late_rebalance_penalty_component
            + late_defensive_posture_penalty_component
            + late_trend_mean_reversion_conflict_penalty_component
        )

        self.current_wealth *= net_wealth_ratio
        if benchmark_return is not None:
            self.current_benchmark_wealth *= max(1.0 + benchmark_return, self.config.epsilon)
        self.peak_wealth = max(self.peak_wealth, self.current_wealth)
        if turnover > 1e-12:
            repeat_memory_symbol_index = self.early_rebalance_risk_repeat_symbol_index
            if repeat_memory_symbol_index is None:
                repeat_memory_symbol_index = self.early_rebalance_risk_repeat_unrecovered_symbol_index
            if repeat_memory_symbol_index is not None and self.cash_index is not None:
                executed_target_cash_weight = float(target_weights.iloc[self.cash_index])
                executed_target_repeat_symbol_weight = float(target_weights.iloc[repeat_memory_symbol_index])
                self.last_executed_rebalance_cash_reduction = max(
                    pre_trade_cash_weight - executed_target_cash_weight,
                    0.0,
                )
                self.last_executed_rebalance_symbol_increase = max(
                    executed_target_repeat_symbol_weight - float(pre_trade_weights.iloc[repeat_memory_symbol_index]),
                    0.0,
                )
                self.last_executed_rebalance_pre_trade_relative_wealth_ratio = pre_trade_relative_wealth_ratio
            if self.early_rebalance_risk_cumulative_symbol_index is not None and self.cash_index is not None:
                executed_target_cash_weight = float(target_weights.iloc[self.cash_index])
                executed_target_cumulative_symbol_weight = float(
                    target_weights.iloc[self.early_rebalance_risk_cumulative_symbol_index]
                )
                self.cumulative_executed_rebalance_cash_reduction += max(
                    pre_trade_cash_weight - executed_target_cash_weight,
                    0.0,
                )
                self.cumulative_executed_rebalance_symbol_increase += max(
                    executed_target_cumulative_symbol_weight - pre_trade_cumulative_symbol_weight,
                    0.0,
                )
            self.executed_rebalances += 1
            if self.rebalance_cooldown_steps is not None:
                self.rebalance_cooldown_remaining = self.rebalance_cooldown_steps
        elif rebalance_cooldown_active and self.rebalance_cooldown_remaining > 0:
            self.rebalance_cooldown_remaining -= 1
        self.current_weights = _update_post_return_weights(target_weights, realized_returns, self.config.epsilon)
        ending_weights = self.current_weights.copy()
        self.current_step += 1
        terminated = self.current_step >= self.episode_end_step
        observation = self._build_observation()
        info = {
            "step_label": str(step_label),
            "pre_trade_weights": pre_trade_weights.to_dict(),
            "proposed_weights": proposed_weights.to_dict(),
            "target_weights": target_weights.to_dict(),
            "ending_weights": ending_weights.to_dict(),
            "proposed_turnover": proposed_turnover,
            "turnover": turnover,
            "trade_suppressed": trade_suppressed,
            "rebalance_budget_exhausted": rebalance_budget_exhausted,
            "rebalance_cooldown_active": rebalance_cooldown_active,
            "rebalance_cooldown_blocked": rebalance_cooldown_blocked,
            "rebalance_cooldown_remaining": self.rebalance_cooldown_remaining,
            "rebalance_cooldown_steps": self.rebalance_cooldown_steps,
            "early_rebalance_risk_window_active": early_rebalance_risk_window_active,
            "early_rebalance_risk_turnover_cap_window_active": early_rebalance_risk_turnover_cap_window_active,
            "early_rebalance_risk_turnover_cap_condition_met": early_rebalance_risk_turnover_cap_condition_met,
            "early_rebalance_risk_turnover_cap_applied": early_rebalance_risk_turnover_cap_applied,
            "early_rebalance_risk_turnover_cap_applications": self.early_rebalance_risk_turnover_cap_applications,
            "early_rebalance_risk_turnover_cap_max_applications_reached": early_rebalance_risk_turnover_cap_max_applications_reached,
            "early_rebalance_risk_turnover_cap_secondary_active": early_rebalance_risk_turnover_cap_secondary_active,
            "early_rebalance_risk_turnover_cap_effective_cap": early_rebalance_risk_turnover_cap_effective_cap,
            "early_rebalance_risk_turnover_cap_secondary_after_applications_reached": early_rebalance_risk_turnover_cap_secondary_after_applications_reached,
            "early_rebalance_risk_turnover_cap_secondary_state_condition_met": early_rebalance_risk_turnover_cap_secondary_state_condition_met,
            "early_rebalance_risk_shallow_drawdown_turnover_cap_window_active": early_rebalance_risk_shallow_drawdown_turnover_cap_window_active,
            "early_rebalance_risk_shallow_drawdown_turnover_cap_condition_met": early_rebalance_risk_shallow_drawdown_turnover_cap_condition_met,
            "early_rebalance_risk_shallow_drawdown_turnover_cap_applied": early_rebalance_risk_shallow_drawdown_turnover_cap_applied,
            "early_rebalance_risk_mean_reversion_turnover_cap_window_active": early_rebalance_risk_mean_reversion_turnover_cap_window_active,
            "early_rebalance_risk_mean_reversion_turnover_cap_condition_met": early_rebalance_risk_mean_reversion_turnover_cap_condition_met,
            "early_rebalance_risk_mean_reversion_action_smoothing_applied": early_rebalance_risk_mean_reversion_action_smoothing_applied,
            "early_rebalance_risk_mean_reversion_turnover_cap_applied": early_rebalance_risk_mean_reversion_turnover_cap_applied,
            "early_rebalance_risk_trend_turnover_cap_window_active": early_rebalance_risk_trend_turnover_cap_window_active,
            "early_rebalance_risk_trend_turnover_cap_condition_met": early_rebalance_risk_trend_turnover_cap_condition_met,
            "early_rebalance_risk_trend_turnover_cap_applied": early_rebalance_risk_trend_turnover_cap_applied,
            "early_rebalance_risk_deep_drawdown_turnover_cap_window_active": early_rebalance_risk_deep_drawdown_turnover_cap_window_active,
            "early_rebalance_risk_deep_drawdown_turnover_cap_condition_met": early_rebalance_risk_deep_drawdown_turnover_cap_condition_met,
            "early_rebalance_risk_deep_drawdown_turnover_cap_applied": early_rebalance_risk_deep_drawdown_turnover_cap_applied,
            "early_rebalance_risk_repeat_turnover_cap_window_active": early_rebalance_risk_repeat_turnover_cap_window_active,
            "early_rebalance_risk_repeat_turnover_cap_condition_met": early_rebalance_risk_repeat_turnover_cap_condition_met,
            "early_rebalance_risk_repeat_action_smoothing_applied": early_rebalance_risk_repeat_action_smoothing_applied,
            "early_rebalance_risk_repeat_turnover_cap_applied": early_rebalance_risk_repeat_turnover_cap_applied,
            "early_rebalance_risk_repeat_unrecovered_turnover_cap_window_active": early_rebalance_risk_repeat_unrecovered_turnover_cap_window_active,
            "early_rebalance_risk_repeat_unrecovered_turnover_cap_condition_met": early_rebalance_risk_repeat_unrecovered_turnover_cap_condition_met,
            "early_rebalance_risk_repeat_unrecovered_turnover_cap_applied": early_rebalance_risk_repeat_unrecovered_turnover_cap_applied,
            "early_rebalance_risk_cumulative_turnover_cap_window_active": early_rebalance_risk_cumulative_turnover_cap_window_active,
            "early_rebalance_risk_cumulative_turnover_cap_condition_met": early_rebalance_risk_cumulative_turnover_cap_condition_met,
            "early_rebalance_risk_cumulative_turnover_cap_applied": early_rebalance_risk_cumulative_turnover_cap_applied,
            "early_rebalance_risk_penalty_applied": early_rebalance_risk_penalty_applied,
            "early_rebalance_risk_condition_met": early_rebalance_risk_condition_met,
            "early_rebalance_risk_penalty": self.early_rebalance_risk_penalty,
            "early_rebalance_risk_turnover_cap": self.early_rebalance_risk_turnover_cap,
            "early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold": self.early_rebalance_risk_turnover_cap_benchmark_drawdown_min_threshold,
            "early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold": self.early_rebalance_risk_turnover_cap_benchmark_cumulative_return_max_threshold,
            "early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight": self.early_rebalance_risk_turnover_cap_min_pre_trade_cash_weight,
            "early_rebalance_risk_turnover_cap_target_cash_min_threshold": self.early_rebalance_risk_turnover_cap_target_cash_min_threshold,
            "early_rebalance_risk_turnover_cap_target_cash_max_threshold": self.early_rebalance_risk_turnover_cap_target_cash_max_threshold,
            "early_rebalance_risk_turnover_cap_target_trend_min_threshold": self.early_rebalance_risk_turnover_cap_target_trend_min_threshold,
            "early_rebalance_risk_turnover_cap_target_trend_max_threshold": self.early_rebalance_risk_turnover_cap_target_trend_max_threshold,
            "early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold": self.early_rebalance_risk_turnover_cap_target_mean_reversion_min_threshold,
            "early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold": self.early_rebalance_risk_turnover_cap_target_mean_reversion_max_threshold,
            "early_rebalance_risk_turnover_cap_delta_cash_min_threshold": self.early_rebalance_risk_turnover_cap_delta_cash_min_threshold,
            "early_rebalance_risk_turnover_cap_delta_cash_max_threshold": self.early_rebalance_risk_turnover_cap_delta_cash_max_threshold,
            "early_rebalance_risk_turnover_cap_delta_trend_min_threshold": self.early_rebalance_risk_turnover_cap_delta_trend_min_threshold,
            "early_rebalance_risk_turnover_cap_delta_trend_max_threshold": self.early_rebalance_risk_turnover_cap_delta_trend_max_threshold,
            "early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold": self.early_rebalance_risk_turnover_cap_delta_mean_reversion_min_threshold,
            "early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold": self.early_rebalance_risk_turnover_cap_delta_mean_reversion_max_threshold,
            "early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold": self.early_rebalance_risk_turnover_cap_proposed_turnover_min_threshold,
            "early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold": self.early_rebalance_risk_turnover_cap_proposed_turnover_max_threshold,
            "early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio": self.early_rebalance_risk_turnover_cap_min_pre_trade_relative_wealth_ratio,
            "early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio": self.early_rebalance_risk_turnover_cap_max_pre_trade_relative_wealth_ratio,
            "early_rebalance_risk_turnover_cap_allow_nonincreasing_risk_symbol": self.early_rebalance_risk_turnover_cap_allow_nonincreasing_risk_symbol,
            "early_rebalance_risk_turnover_cap_use_penalty_state_filters": self.early_rebalance_risk_turnover_cap_use_penalty_state_filters,
            "early_rebalance_risk_turnover_cap_after": self.early_rebalance_risk_turnover_cap_after,
            "early_rebalance_risk_turnover_cap_before": self.early_rebalance_risk_turnover_cap_before,
            "early_rebalance_risk_turnover_cap_max_applications": self.early_rebalance_risk_turnover_cap_max_applications,
            "early_rebalance_risk_turnover_cap_secondary_cap": self.early_rebalance_risk_turnover_cap_secondary_cap,
            "early_rebalance_risk_turnover_cap_secondary_after_applications": self.early_rebalance_risk_turnover_cap_secondary_after_applications,
            "early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold": self.early_rebalance_risk_turnover_cap_secondary_benchmark_cumulative_return_min_threshold,
            "early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio": self.early_rebalance_risk_turnover_cap_secondary_max_pre_trade_relative_wealth_ratio,
            "early_rebalance_risk_shallow_drawdown_turnover_cap": self.early_rebalance_risk_shallow_drawdown_turnover_cap,
            "early_rebalance_risk_shallow_drawdown_turnover_cap_after": self.early_rebalance_risk_shallow_drawdown_turnover_cap_after,
            "early_rebalance_risk_shallow_drawdown_turnover_cap_before": self.early_rebalance_risk_shallow_drawdown_turnover_cap_before,
            "early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold": self.early_rebalance_risk_shallow_drawdown_turnover_cap_cash_max_threshold,
            "early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold": self.early_rebalance_risk_shallow_drawdown_turnover_cap_benchmark_drawdown_min_threshold,
            "early_rebalance_risk_mean_reversion_turnover_cap": self.early_rebalance_risk_mean_reversion_turnover_cap,
            "early_rebalance_risk_mean_reversion_action_smoothing": self.early_rebalance_risk_mean_reversion_action_smoothing,
            "early_rebalance_risk_mean_reversion_turnover_cap_after": self.early_rebalance_risk_mean_reversion_turnover_cap_after,
            "early_rebalance_risk_mean_reversion_turnover_cap_before": self.early_rebalance_risk_mean_reversion_turnover_cap_before,
            "early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold": self.early_rebalance_risk_mean_reversion_turnover_cap_benchmark_cumulative_return_max_threshold,
            "early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold": self.early_rebalance_risk_mean_reversion_turnover_cap_target_mean_reversion_min_threshold,
            "early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold": self.early_rebalance_risk_mean_reversion_turnover_cap_pre_trade_mean_reversion_min_threshold,
            "early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold": self.early_rebalance_risk_mean_reversion_turnover_cap_delta_mean_reversion_min_threshold,
            "early_rebalance_risk_trend_turnover_cap": self.early_rebalance_risk_trend_turnover_cap,
            "early_rebalance_risk_trend_turnover_cap_after": self.early_rebalance_risk_trend_turnover_cap_after,
            "early_rebalance_risk_trend_turnover_cap_before": self.early_rebalance_risk_trend_turnover_cap_before,
            "early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold": self.early_rebalance_risk_trend_turnover_cap_benchmark_cumulative_return_max_threshold,
            "early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold": self.early_rebalance_risk_trend_turnover_cap_target_trend_min_threshold,
            "early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold": self.early_rebalance_risk_trend_turnover_cap_pre_trade_trend_min_threshold,
            "early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold": self.early_rebalance_risk_trend_turnover_cap_delta_trend_min_threshold,
            "early_rebalance_risk_deep_drawdown_turnover_cap": self.early_rebalance_risk_deep_drawdown_turnover_cap,
            "early_rebalance_risk_deep_drawdown_turnover_cap_after": self.early_rebalance_risk_deep_drawdown_turnover_cap_after,
            "early_rebalance_risk_deep_drawdown_turnover_cap_before": self.early_rebalance_risk_deep_drawdown_turnover_cap_before,
            "early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold": self.early_rebalance_risk_deep_drawdown_turnover_cap_benchmark_drawdown_max_threshold,
            "early_rebalance_risk_repeat_turnover_cap": self.early_rebalance_risk_repeat_turnover_cap,
            "early_rebalance_risk_repeat_action_smoothing": self.early_rebalance_risk_repeat_action_smoothing,
            "early_rebalance_risk_repeat_turnover_cap_after": self.early_rebalance_risk_repeat_turnover_cap_after,
            "early_rebalance_risk_repeat_turnover_cap_before": self.early_rebalance_risk_repeat_turnover_cap_before,
            "early_rebalance_risk_repeat_symbol": self.early_rebalance_risk_repeat_symbol,
            "early_rebalance_risk_repeat_previous_cash_reduction_min": self.early_rebalance_risk_repeat_previous_cash_reduction_min,
            "early_rebalance_risk_repeat_previous_symbol_increase_min": self.early_rebalance_risk_repeat_previous_symbol_increase_min,
            "early_rebalance_risk_repeat_previous_cash_reduction": early_rebalance_risk_repeat_previous_cash_reduction,
            "early_rebalance_risk_repeat_previous_symbol_increase": early_rebalance_risk_repeat_previous_symbol_increase,
            "early_rebalance_risk_repeat_cash_reduction": early_rebalance_risk_repeat_cash_reduction,
            "early_rebalance_risk_repeat_symbol_increase": early_rebalance_risk_repeat_symbol_increase,
            "early_rebalance_risk_repeat_unrecovered_turnover_cap": self.early_rebalance_risk_repeat_unrecovered_turnover_cap,
            "early_rebalance_risk_repeat_unrecovered_turnover_cap_after": self.early_rebalance_risk_repeat_unrecovered_turnover_cap_after,
            "early_rebalance_risk_repeat_unrecovered_turnover_cap_before": self.early_rebalance_risk_repeat_unrecovered_turnover_cap_before,
            "early_rebalance_risk_repeat_unrecovered_symbol": self.early_rebalance_risk_repeat_unrecovered_symbol,
            "early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min": self.early_rebalance_risk_repeat_unrecovered_previous_cash_reduction_min,
            "early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min": self.early_rebalance_risk_repeat_unrecovered_previous_symbol_increase_min,
            "early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery": self.early_rebalance_risk_repeat_unrecovered_min_relative_wealth_recovery,
            "early_rebalance_risk_repeat_unrecovered_previous_cash_reduction": early_rebalance_risk_repeat_unrecovered_previous_cash_reduction,
            "early_rebalance_risk_repeat_unrecovered_previous_symbol_increase": early_rebalance_risk_repeat_unrecovered_previous_symbol_increase,
            "early_rebalance_risk_repeat_unrecovered_previous_pre_trade_relative_wealth_ratio": early_rebalance_risk_repeat_unrecovered_previous_pre_trade_relative_wealth_ratio,
            "early_rebalance_risk_repeat_unrecovered_relative_wealth_recovery": early_rebalance_risk_repeat_unrecovered_relative_wealth_recovery,
            "early_rebalance_risk_repeat_unrecovered_cash_reduction": early_rebalance_risk_repeat_unrecovered_cash_reduction,
            "early_rebalance_risk_repeat_unrecovered_symbol_increase": early_rebalance_risk_repeat_unrecovered_symbol_increase,
            "early_rebalance_risk_cumulative_turnover_cap": self.early_rebalance_risk_cumulative_turnover_cap,
            "early_rebalance_risk_cumulative_turnover_cap_after": self.early_rebalance_risk_cumulative_turnover_cap_after,
            "early_rebalance_risk_cumulative_turnover_cap_before": self.early_rebalance_risk_cumulative_turnover_cap_before,
            "early_rebalance_risk_cumulative_symbol": self.early_rebalance_risk_cumulative_symbol,
            "early_rebalance_risk_cumulative_cash_reduction_budget": self.early_rebalance_risk_cumulative_cash_reduction_budget,
            "early_rebalance_risk_cumulative_symbol_increase_budget": self.early_rebalance_risk_cumulative_symbol_increase_budget,
            "early_rebalance_risk_cumulative_prior_cash_reduction": early_rebalance_risk_cumulative_prior_cash_reduction,
            "early_rebalance_risk_cumulative_prior_symbol_increase": early_rebalance_risk_cumulative_prior_symbol_increase,
            "early_rebalance_risk_cumulative_cash_reduction": early_rebalance_risk_cumulative_cash_reduction,
            "early_rebalance_risk_cumulative_symbol_increase": early_rebalance_risk_cumulative_symbol_increase,
            "early_rebalance_risk_penalty_after": self.early_rebalance_risk_penalty_after,
            "early_rebalance_risk_penalty_before": self.early_rebalance_risk_penalty_before,
            "early_rebalance_risk_penalty_cash_max_threshold": self.early_rebalance_risk_penalty_cash_max_threshold,
            "early_rebalance_risk_penalty_symbol": self.early_rebalance_risk_penalty_symbol,
            "early_rebalance_risk_penalty_symbol_min_weight": self.early_rebalance_risk_penalty_symbol_min_weight,
            "early_rebalance_risk_penalty_symbol_max_weight": self.early_rebalance_risk_penalty_symbol_max_weight,
            "early_rebalance_risk_penalty_benchmark_drawdown_min_threshold": self.early_rebalance_risk_penalty_benchmark_drawdown_min_threshold,
            "early_rebalance_risk_penalty_benchmark_drawdown_max_threshold": self.early_rebalance_risk_penalty_benchmark_drawdown_max_threshold,
            "early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio": self.early_rebalance_risk_penalty_min_pre_trade_relative_wealth_ratio,
            "early_benchmark_euphoria_window_active": early_benchmark_euphoria_window_active,
            "early_benchmark_euphoria_turnover_cap_applied": early_benchmark_euphoria_turnover_cap_applied,
            "early_benchmark_euphoria_penalty_applied": early_benchmark_euphoria_penalty_applied,
            "early_benchmark_euphoria_condition_met": early_benchmark_euphoria_condition_met,
            "early_benchmark_euphoria_penalty": self.early_benchmark_euphoria_penalty,
            "early_benchmark_euphoria_turnover_cap": self.early_benchmark_euphoria_turnover_cap,
            "early_benchmark_euphoria_before": self.early_benchmark_euphoria_before,
            "early_benchmark_euphoria_benchmark_drawdown_min_threshold": self.early_benchmark_euphoria_benchmark_drawdown_min_threshold,
            "early_benchmark_euphoria_symbol": self.early_benchmark_euphoria_symbol,
            "late_rebalance_penalty": self.late_rebalance_penalty,
            "late_rebalance_penalty_after": self.late_rebalance_penalty_after,
            "late_rebalance_penalty_applied": late_rebalance_penalty_applied,
            "late_rebalance_threshold_reached": late_rebalance_threshold_reached,
            "late_rebalance_gate_active": late_rebalance_gate_active,
            "late_rebalance_gate_blocked": late_rebalance_gate_blocked,
            "late_rebalance_gate_condition_met": late_rebalance_gate_condition_met,
            "late_rebalance_gate_refinement_condition_met": late_rebalance_gate_refinement_condition_met,
            "late_rebalance_gate_threshold_reached": late_rebalance_gate_threshold_reached,
            "late_rebalance_gate_after": self.late_rebalance_gate_after,
            "late_rebalance_gate_cash_threshold": self.late_rebalance_gate_cash_threshold,
            "late_rebalance_gate_target_cash_min_threshold": self.late_rebalance_gate_target_cash_min_threshold,
            "late_rebalance_gate_symbol": self.late_rebalance_gate_symbol,
            "late_rebalance_gate_symbol_max_weight": self.late_rebalance_gate_symbol_max_weight,
            "late_rebalance_gate_cash_reduction_max": self.late_rebalance_gate_cash_reduction_max,
            "late_rebalance_gate_symbol_increase_max": self.late_rebalance_gate_symbol_increase_max,
            "late_rebalance_gate_cash_reduction": float(late_rebalance_gate_cash_reduction),
            "late_rebalance_gate_symbol_increase": float(late_rebalance_gate_symbol_increase),
            "late_defensive_posture_window_active": late_defensive_posture_window_active,
            "late_defensive_posture_condition_met": late_defensive_posture_condition_met,
            "late_defensive_posture_penalty_applied": late_defensive_posture_penalty_applied,
            "late_defensive_posture_penalty": self.late_defensive_posture_penalty,
            "late_defensive_posture_penalty_after": self.late_defensive_posture_penalty_after,
            "late_defensive_posture_penalty_cash_min_threshold": self.late_defensive_posture_penalty_cash_min_threshold,
            "late_defensive_posture_penalty_symbol": self.late_defensive_posture_penalty_symbol,
            "late_defensive_posture_penalty_symbol_max_weight": self.late_defensive_posture_penalty_symbol_max_weight,
            "late_trend_mean_reversion_conflict_window_active": late_trend_mean_reversion_conflict_window_active,
            "late_trend_mean_reversion_conflict_condition_met": late_trend_mean_reversion_conflict_condition_met,
            "late_trend_mean_reversion_conflict_penalty_applied": late_trend_mean_reversion_conflict_penalty_applied,
            "late_trend_mean_reversion_conflict_penalty": self.late_trend_mean_reversion_conflict_penalty,
            "late_trend_mean_reversion_conflict_penalty_after": self.late_trend_mean_reversion_conflict_penalty_after,
            "late_trend_mean_reversion_conflict_trend_symbol": self.late_trend_mean_reversion_conflict_trend_symbol,
            "late_trend_mean_reversion_conflict_trend_min_weight": self.late_trend_mean_reversion_conflict_trend_min_weight,
            "late_trend_mean_reversion_conflict_mean_reversion_symbol": self.late_trend_mean_reversion_conflict_mean_reversion_symbol,
            "late_trend_mean_reversion_conflict_mean_reversion_min_weight": self.late_trend_mean_reversion_conflict_mean_reversion_min_weight,
            "state_trend_preservation_window_active": state_trend_preservation_window_active,
            "state_trend_preservation_condition_met": state_trend_preservation_condition_met,
            "state_trend_preservation_guard_applied": state_trend_preservation_guard_applied,
            "state_trend_preservation_symbol": self.state_trend_preservation_symbol,
            "state_trend_preservation_cash_max_threshold": self.state_trend_preservation_cash_max_threshold,
            "state_trend_preservation_symbol_min_weight": self.state_trend_preservation_symbol_min_weight,
            "state_trend_preservation_max_symbol_reduction": self.state_trend_preservation_max_symbol_reduction,
            "executed_rebalances": self.executed_rebalances,
            "last_executed_rebalance_pre_trade_relative_wealth_ratio": self.last_executed_rebalance_pre_trade_relative_wealth_ratio,
            "cumulative_executed_rebalance_cash_reduction": self.cumulative_executed_rebalance_cash_reduction,
            "cumulative_executed_rebalance_symbol_increase": self.cumulative_executed_rebalance_symbol_increase,
            "max_executed_rebalances": self.max_executed_rebalances,
            "execution_cost": execution_cost,
            "asset_returns": realized_returns.to_dict(),
            "gross_return_contributions": gross_return_contributions.to_dict(),
            "portfolio_return": net_wealth_ratio - 1.0,
            "gross_portfolio_return": gross_portfolio_return,
            "gross_wealth_ratio": gross_wealth_ratio,
            "net_wealth_ratio": net_wealth_ratio,
            "wealth": self.current_wealth,
            "gross_reward_component": float(gross_reward_component),
            "reward_core_component": float(reward_core_component),
            "benchmark_reward_component": float(benchmark_reward_component),
            "execution_cost_reward_drag": float(execution_cost_reward_drag),
            "turnover_penalty_component": float(turnover_penalty_component),
            "weight_reg_penalty_component": float(weight_reg_penalty_component),
            "cash_weight": float(cash_weight),
            "cash_target_weight": self.cash_target_weight,
            "excess_cash_weight": float(excess_cash_weight),
            "cash_weight_penalty_component": float(cash_weight_penalty_component),
            "early_rebalance_risk_penalty_component": float(early_rebalance_risk_penalty_component),
            "early_benchmark_euphoria_penalty_component": float(early_benchmark_euphoria_penalty_component),
            "late_rebalance_penalty_component": float(late_rebalance_penalty_component),
            "late_defensive_posture_penalty_component": float(late_defensive_posture_penalty_component),
            "late_trend_mean_reversion_conflict_penalty_component": float(
                late_trend_mean_reversion_conflict_penalty_component
            ),
            "friction_reward_drag": float(friction_reward_drag),
            "raw_reward": float(reward),
            "action_smoothing": self.action_smoothing,
            "no_trade_band": self.no_trade_band,
            "max_executed_rebalances": self.max_executed_rebalances,
            "rebalance_cooldown_steps": self.rebalance_cooldown_steps,
            "early_rebalance_risk_penalty": self.early_rebalance_risk_penalty,
            "early_rebalance_risk_turnover_cap": self.early_rebalance_risk_turnover_cap,
            "early_rebalance_risk_penalty_after": self.early_rebalance_risk_penalty_after,
            "early_rebalance_risk_penalty_before": self.early_rebalance_risk_penalty_before,
            "early_benchmark_euphoria_penalty": self.early_benchmark_euphoria_penalty,
            "early_benchmark_euphoria_turnover_cap": self.early_benchmark_euphoria_turnover_cap,
            "early_benchmark_euphoria_before": self.early_benchmark_euphoria_before,
            "late_rebalance_penalty": self.late_rebalance_penalty,
            "late_rebalance_penalty_after": self.late_rebalance_penalty_after,
            "late_trend_mean_reversion_conflict_penalty": self.late_trend_mean_reversion_conflict_penalty,
            "late_trend_mean_reversion_conflict_penalty_after": self.late_trend_mean_reversion_conflict_penalty_after,
            "cash_weight_penalty": self.cash_weight_penalty,
            "episode_start_step": self.episode_start_step,
            "episode_end_step": self.episode_end_step,
        }
        if benchmark_regime_cumulative_return is not None:
            info["benchmark_regime_cumulative_return"] = float(benchmark_regime_cumulative_return)
        if benchmark_regime_drawdown is not None:
            info["benchmark_regime_drawdown"] = float(benchmark_regime_drawdown)
        if benchmark_return is not None:
            info["benchmark_return"] = benchmark_return
        if pre_trade_relative_wealth_ratio is not None:
            info["pre_trade_relative_wealth_ratio"] = float(pre_trade_relative_wealth_ratio)
        if relative_wealth_ratio is not None:
            info["relative_wealth_ratio"] = float(relative_wealth_ratio)
        return observation, float(reward), terminated, False, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None