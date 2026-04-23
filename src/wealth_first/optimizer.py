from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from wealth_first.data import add_cash_sleeve
from wealth_first.rebalance import compute_execution_cost, compute_tradable_turnover


@dataclass(frozen=True)
class WealthFirstConfig:
    loss_penalty: float = 8.0
    gain_reward: float = 1.0
    gain_power: float = 1.0
    loss_power: float = 1.5
    turnover_penalty: float = 0.0
    weight_reg: float = 0.0
    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    objective_mode: str = "piecewise"
    benchmark_gain_reward: float = 0.0
    benchmark_loss_penalty: float = 0.0
    min_weight: float = 0.0
    max_weight: float = 1.0
    min_weight_overrides: Mapping[str, float] | None = None
    max_weight_overrides: Mapping[str, float] | None = None
    include_cash: bool = True
    cash_symbol: str = "CASH"
    epsilon: float = 1e-8


@dataclass(frozen=True)
class OptimizationResult:
    weights: pd.Series
    objective_value: float
    success: bool
    message: str
    iterations: int


def _coerce_returns_frame(returns: pd.DataFrame | pd.Series | np.ndarray) -> pd.DataFrame:
    if isinstance(returns, pd.DataFrame):
        frame = returns.copy()
    elif isinstance(returns, pd.Series):
        column_name = returns.name if returns.name is not None else "asset_1"
        frame = returns.to_frame(name=str(column_name))
    else:
        array = np.asarray(returns, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError("Returns must be one-dimensional or two-dimensional.")
        columns = [f"asset_{index + 1}" for index in range(array.shape[1])]
        frame = pd.DataFrame(array, columns=columns)

    frame = frame.astype(float).dropna(how="any")
    if frame.empty:
        raise ValueError("Returns data is empty after dropping missing values.")
    return frame


def _coerce_weight_vector(
    weights: pd.Series | np.ndarray | None,
    columns: pd.Index,
    default_value: float,
) -> np.ndarray:
    if weights is None:
        return np.full(len(columns), default_value, dtype=float)

    if isinstance(weights, pd.Series):
        aligned = weights.reindex(columns).fillna(0.0).to_numpy(dtype=float)
    else:
        aligned = np.asarray(weights, dtype=float)
        if aligned.ndim != 1 or aligned.shape[0] != len(columns):
            raise ValueError("Weight vector length does not match the number of assets.")

    aligned = np.clip(aligned, 0.0, None)
    total = aligned.sum()
    if total <= 0:
        return np.full(len(columns), default_value, dtype=float)
    return aligned / total


def _resolve_weight_bounds(
    columns: pd.Index,
    config: WealthFirstConfig,
) -> tuple[np.ndarray, np.ndarray]:
    lower_bounds = np.full(len(columns), config.min_weight, dtype=float)
    upper_bounds = np.full(len(columns), config.max_weight, dtype=float)

    override_symbols: set[str] = set()
    if config.min_weight_overrides:
        override_symbols.update(config.min_weight_overrides)
        for index, column in enumerate(columns):
            if column in config.min_weight_overrides:
                lower_bounds[index] = float(config.min_weight_overrides[column])

    if config.max_weight_overrides:
        override_symbols.update(config.max_weight_overrides)
        for index, column in enumerate(columns):
            if column in config.max_weight_overrides:
                upper_bounds[index] = float(config.max_weight_overrides[column])

    unknown_symbols = sorted(symbol for symbol in override_symbols if symbol not in columns)
    if unknown_symbols:
        raise ValueError(f"Weight bounds were provided for unknown sleeves: {', '.join(unknown_symbols)}")
    if (lower_bounds < 0.0).any():
        raise ValueError("Weight lower bounds must be non-negative.")
    if (upper_bounds < lower_bounds).any():
        raise ValueError("Weight lower bounds cannot exceed upper bounds.")
    if float(lower_bounds.sum()) > 1.0 + 1e-9:
        raise ValueError("Weight lower bounds are infeasible because they sum to more than 1.")
    if float(upper_bounds.sum()) < 1.0 - 1e-9:
        raise ValueError("Weight upper bounds are infeasible because they sum to less than 1.")

    return lower_bounds, upper_bounds


def _project_to_weight_bounds(
    weights: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    tolerance: float = 1e-9,
) -> np.ndarray:
    projected = np.clip(np.asarray(weights, dtype=float), lower_bounds, upper_bounds)
    residual = 1.0 - float(projected.sum())

    if residual > tolerance:
        remaining_capacity = upper_bounds - projected
        total_capacity = float(remaining_capacity.sum())
        if total_capacity < residual - tolerance:
            raise ValueError("Weight bounds cannot be satisfied while enforcing the simplex constraint.")
        if total_capacity > 0:
            projected += residual * (remaining_capacity / total_capacity)
    elif residual < -tolerance:
        removable_weight = projected - lower_bounds
        total_removable_weight = float(removable_weight.sum())
        if total_removable_weight < (-residual) - tolerance:
            raise ValueError("Weight bounds cannot be satisfied while enforcing the simplex constraint.")
        if total_removable_weight > 0:
            projected -= (-residual) * (removable_weight / total_removable_weight)

    projected = np.clip(projected, lower_bounds, upper_bounds)
    residual = 1.0 - float(projected.sum())
    if residual > tolerance:
        for index in np.argsort(upper_bounds - projected)[::-1]:
            increment = min(residual, upper_bounds[index] - projected[index])
            if increment <= 0:
                continue
            projected[index] += increment
            residual -= increment
            if residual <= tolerance:
                break
    elif residual < -tolerance:
        for index in np.argsort(projected - lower_bounds)[::-1]:
            decrement = min(-residual, projected[index] - lower_bounds[index])
            if decrement <= 0:
                continue
            projected[index] -= decrement
            residual += decrement
            if residual >= -tolerance:
                break

    if not np.isclose(projected.sum(), 1.0, atol=1e-6):
        raise ValueError("Projected weights do not sum to 1 within tolerance.")
    return projected


def _coerce_benchmark_vector(
    benchmark_returns: pd.Series | pd.DataFrame | np.ndarray | None,
    index: pd.Index,
) -> np.ndarray | None:
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
            raise ValueError("Benchmark returns must have the same number of observations as the optimization window.")
        return benchmark_array

    aligned = pd.Series(benchmark_series, copy=True).astype(float)
    if isinstance(aligned.index, pd.DatetimeIndex) and isinstance(index, pd.DatetimeIndex):
        aligned = aligned.sort_index().reindex(index)
        if aligned.isna().any():
            raise ValueError("Benchmark returns could not be aligned to the optimization window.")
        return aligned.to_numpy(dtype=float)

    benchmark_array = aligned.to_numpy(dtype=float)
    if benchmark_array.shape[0] != len(index):
        raise ValueError("Benchmark returns must have the same number of observations as the optimization window.")
    return benchmark_array


def _score_wealth_ratios(
    wealth_ratios: np.ndarray,
    gain_reward: float,
    loss_penalty: float,
    gain_power: float,
    loss_power: float,
    objective_mode: str,
    epsilon: float,
) -> float:
    if objective_mode == "piecewise":
        upside = np.clip(wealth_ratios - 1.0, 0.0, None)
        downside = np.clip(1.0 - wealth_ratios, 0.0, None)
        score = -gain_reward * np.sum(upside ** gain_power)
        score += loss_penalty * np.sum(downside ** loss_power)
        return float(score)

    if objective_mode == "log_wealth":
        safe_ratios = np.clip(wealth_ratios, epsilon, None)
        log_growth = np.log(safe_ratios)
        upside = np.clip(log_growth, 0.0, None)
        downside = np.clip(-log_growth, 0.0, None)
        score = -gain_reward * np.sum(upside)
        score += loss_penalty * np.sum(downside ** loss_power)
        return float(score)

    raise ValueError(f"Unsupported objective mode: {objective_mode}")


def wealth_first_objective(
    weights: np.ndarray,
    returns: pd.DataFrame | pd.Series | np.ndarray,
    config: WealthFirstConfig,
    previous_weights: np.ndarray | None = None,
    benchmark_returns: pd.Series | pd.DataFrame | np.ndarray | None = None,
) -> float:
    candidate_weights = np.asarray(weights, dtype=float)
    returns_frame = _coerce_returns_frame(returns)
    port_returns = returns_frame.to_numpy() @ candidate_weights
    wealth_ratios = 1.0 + port_returns

    score = _score_wealth_ratios(
        wealth_ratios,
        gain_reward=config.gain_reward,
        loss_penalty=config.loss_penalty,
        gain_power=config.gain_power,
        loss_power=config.loss_power,
        objective_mode=config.objective_mode,
        epsilon=config.epsilon,
    )

    benchmark_vector = _coerce_benchmark_vector(benchmark_returns, returns_frame.index)
    benchmark_is_required = config.benchmark_gain_reward > 0 or config.benchmark_loss_penalty > 0
    if benchmark_is_required and benchmark_vector is None:
        raise ValueError("Benchmark-aware objective requires benchmark returns.")
    if benchmark_vector is not None and benchmark_is_required:
        benchmark_wealth_ratios = np.clip(1.0 + benchmark_vector, config.epsilon, None)
        relative_wealth_ratios = wealth_ratios / benchmark_wealth_ratios
        score += _score_wealth_ratios(
            relative_wealth_ratios,
            gain_reward=config.benchmark_gain_reward,
            loss_penalty=config.benchmark_loss_penalty,
            gain_power=config.gain_power,
            loss_power=config.loss_power,
            objective_mode=config.objective_mode,
            epsilon=config.epsilon,
        )

    if previous_weights is not None:
        current_weight_series = pd.Series(candidate_weights, index=returns_frame.columns, name="current_weight")
        previous_weight_series = pd.Series(np.asarray(previous_weights, dtype=float), index=returns_frame.columns, name="previous_weight")
        tradable_turnover = compute_tradable_turnover(
            current_weight_series,
            previous_weight_series,
            config.cash_symbol,
        )

        if config.turnover_penalty > 0:
            score += config.turnover_penalty * tradable_turnover

        execution_cost = compute_execution_cost(
            tradable_turnover,
            config.transaction_cost_bps,
            config.slippage_bps,
        )
        if execution_cost > 0:
            if config.objective_mode == "piecewise":
                score += config.loss_penalty * (execution_cost ** config.loss_power)
            else:
                trade_wealth_ratio = max(1.0 - execution_cost, config.epsilon)
                trade_log_growth = np.log(trade_wealth_ratio)
                trade_downside = np.clip(-trade_log_growth, 0.0, None)
                score += config.loss_penalty * (trade_downside ** config.loss_power)

    if config.weight_reg > 0:
        score += config.weight_reg * float(np.dot(weights, weights))

    return float(score)


def optimize_weights(
    returns: pd.DataFrame | pd.Series | np.ndarray,
    config: WealthFirstConfig | None = None,
    previous_weights: pd.Series | np.ndarray | None = None,
    initial_weights: pd.Series | np.ndarray | None = None,
    benchmark_returns: pd.Series | pd.DataFrame | np.ndarray | None = None,
) -> OptimizationResult:
    effective_config = config or WealthFirstConfig()
    returns_frame = _coerce_returns_frame(returns)
    if effective_config.include_cash:
        returns_frame = add_cash_sleeve(returns_frame, cash_symbol=effective_config.cash_symbol)

    lower_bounds, upper_bounds = _resolve_weight_bounds(returns_frame.columns, effective_config)

    default_weight = 1.0 / returns_frame.shape[1]
    previous_weight_vector = _coerce_weight_vector(previous_weights, returns_frame.columns, default_weight) if previous_weights is not None else None
    initial_weight_vector = _coerce_weight_vector(
        initial_weights if initial_weights is not None else previous_weights,
        returns_frame.columns,
        default_weight,
    )
    initial_weight_vector = _project_to_weight_bounds(initial_weight_vector, lower_bounds, upper_bounds)

    bounds = tuple((float(lower_bound), float(upper_bound)) for lower_bound, upper_bound in zip(lower_bounds, upper_bounds))
    constraints = ({"type": "eq", "fun": lambda weight_vector: np.sum(weight_vector) - 1.0},)

    result = minimize(
        wealth_first_objective,
        initial_weight_vector,
        args=(returns_frame, effective_config, previous_weight_vector, benchmark_returns),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    optimized_weights = _project_to_weight_bounds(result.x, lower_bounds, upper_bounds)
    weight_series = pd.Series(optimized_weights, index=returns_frame.columns, name="weight")

    return OptimizationResult(
        weights=weight_series,
        objective_value=float(result.fun),
        success=bool(result.success),
        message=str(result.message),
        iterations=int(getattr(result, "nit", 0)),
    )