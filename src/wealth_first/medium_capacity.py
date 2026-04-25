"""Medium-capacity policy: two-stage deviation predictor + bounded weight mapping.

This module implements a model between main3 (low-capacity linear) and a future higher-capacity DRL path.
It learns when to deviate from a baseline allocation, then maps bounded deltas to target weights.

Design:
  - Stage 1: Binary classifier or regressor that predicts deviation signal [-1, +1]
  - Stage 2: Map signal to target weight bounded within [min_spy_weight, max_spy_weight]
  - Fast training on validation windows with optional regularization

This allows the model to learn market timing decisions without exponential parameter search.
The feature-generation family is configurable so mechanism searches can move beyond
small threshold nudges on a single fixed representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MediumCapacityConfig:
    """Configuration for medium-capacity model."""

    min_spy_weight: float = 0.80
    max_spy_weight: float = 1.05
    initial_spy_weight: float = 1.0
    action_smoothing: float = 0.5
    no_trade_band: float = 0.02
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0
    # Ridge L2 regularization for linear stage
    ridge_l2: float = 1.0
    # Optional penalty on validation turnover during scale selection.
    scale_turnover_penalty: float = 0.0
    # Optional explicit bounds on stage-2 signal scale search.
    min_signal_scale: float = -0.75
    max_signal_scale: float = 0.75
    # Stage-1 target construction.
    target_mode: str = "sign"
    # Feature-generation family used by the stage-1 signal model.
    feature_family: str = "baseline"
    # Stage-1 signal model family.
    signal_model_family: str = "single_linear"
    # Drawdown split threshold used by the regime-two-model family.
    regime_drawdown_threshold: float = -0.08
    # Whether to use multiplicative signal or additive
    use_multiplicative_signal: bool = True
    # Quantile for robust tail behavior
    quantile_loss_tau: float | None = None


@dataclass(frozen=True)
class MediumCapacityParams:
    """Learned parameters for the medium-capacity model."""

    # Linear weights for the deviation signal [bias, feature_1, feature_2, ...]
    signal_weights: np.ndarray
    # Learned scaling to map signal to weight delta
    signal_scale: float
    # Training standardization statistics applied at inference time
    feature_mu: np.ndarray
    feature_std: np.ndarray
    # Optional regime-specific weights used by signal_model_family="regime_two_model".
    signal_weights_regime_stress: np.ndarray | None = None
    signal_weights_regime_normal: np.ndarray | None = None


def _build_simple_features(spy_returns: pd.Series) -> pd.DataFrame:
    """Build a minimal, fast feature set for signal prediction."""
    price = (1.0 + spy_returns).cumprod()
    lagged_price = price.shift(1)

    def rolling_ret(window: int) -> pd.Series:
        return ((1.0 + spy_returns).rolling(window, min_periods=1).apply(np.prod, raw=True) - 1.0).shift(1)

    def rolling_vol(window: int) -> pd.Series:
        return spy_returns.rolling(window, min_periods=2).std(ddof=0).shift(1)

    def rolling_dd(window: int) -> pd.Series:
        rolling_peak = lagged_price.rolling(window, min_periods=1).max()
        return lagged_price / rolling_peak - 1.0

    out = pd.DataFrame(
        {
            "ret_21": rolling_ret(21),
            "ret_63": rolling_ret(63),
            "dd_63": rolling_dd(63),
            "vol_21": rolling_vol(21),
        },
        index=spy_returns.index,
    )
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _build_regime_interaction_features(spy_returns: pd.Series) -> pd.DataFrame:
    """Augment the baseline representation with trend, drawdown, and interaction state."""
    base = _build_simple_features(spy_returns)
    price = (1.0 + spy_returns).cumprod()
    lagged_price = price.shift(1)

    def rolling_ret(window: int) -> pd.Series:
        return ((1.0 + spy_returns).rolling(window, min_periods=1).apply(np.prod, raw=True) - 1.0).shift(1)

    def rolling_vol(window: int) -> pd.Series:
        return spy_returns.rolling(window, min_periods=2).std(ddof=0).shift(1)

    def rolling_dd(window: int) -> pd.Series:
        rolling_peak = lagged_price.rolling(window, min_periods=1).max()
        return lagged_price / rolling_peak - 1.0

    ma_21 = lagged_price.rolling(21, min_periods=1).mean()
    ma_63 = lagged_price.rolling(63, min_periods=1).mean()
    vol_21 = rolling_vol(21)
    vol_63 = rolling_vol(63)
    ret_5 = rolling_ret(5)
    ret_126 = rolling_ret(126)
    dd_21 = rolling_dd(21)
    dd_126 = rolling_dd(126)

    out = base.assign(
        ret_5=ret_5,
        ret_126=ret_126,
        dd_21=dd_21,
        dd_126=dd_126,
        vol_63=vol_63,
        trend_gap_21_63=(lagged_price / np.maximum(ma_21, 1e-8)) - (lagged_price / np.maximum(ma_63, 1e-8)),
        vol_ratio_21_63=vol_21 / np.maximum(vol_63, 1e-8) - 1.0,
    )
    out["ret_21_x_dd_63"] = out["ret_21"] * np.abs(out["dd_63"])
    out["ret_63_x_dd_21"] = out["ret_63"] * np.abs(out["dd_21"])
    out["ret_5_x_vol_21"] = out["ret_5"] * out["vol_21"]
    out["trend_gap_x_dd_126"] = out["trend_gap_21_63"] * np.abs(out["dd_126"])
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _build_shock_reversal_features(spy_returns: pd.Series) -> pd.DataFrame:
    """Build a distinct representation for crash-and-rebound style timing behavior."""
    base = _build_simple_features(spy_returns)

    def rolling_ret(window: int) -> pd.Series:
        return ((1.0 + spy_returns).rolling(window, min_periods=1).apply(np.prod, raw=True) - 1.0).shift(1)

    def rolling_vol(window: int) -> pd.Series:
        return spy_returns.rolling(window, min_periods=2).std(ddof=0).shift(1)

    ret_5 = rolling_ret(5)
    ret_10 = rolling_ret(10)
    ret_42 = rolling_ret(42)
    vol_5 = rolling_vol(5)
    vol_21 = rolling_vol(21)

    downside_shock_5 = np.minimum(ret_5, 0.0)
    downside_shock_10 = np.minimum(ret_10, 0.0)
    rebound_gap_5_21 = ret_5 - base["ret_21"]
    rebound_gap_10_42 = ret_10 - ret_42
    vol_spike_5_21 = vol_5 / np.maximum(vol_21, 1e-8) - 1.0

    out = pd.DataFrame(
        {
            "downside_shock_5": downside_shock_5,
            "downside_shock_10": downside_shock_10,
            "rebound_gap_5_21": rebound_gap_5_21,
            "rebound_gap_10_42": rebound_gap_10_42,
            "vol_spike_5_21": vol_spike_5_21,
            "dd_21": base["dd_63"] - base["ret_21"],
            "dd_63": base["dd_63"],
            "ret_21": base["ret_21"],
            "vol_21": base["vol_21"],
        },
        index=spy_returns.index,
    )
    out["shock_x_drawdown"] = out["downside_shock_5"] * np.abs(out["dd_63"])
    out["rebound_x_vol_spike"] = out["rebound_gap_5_21"] * out["vol_spike_5_21"]
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def build_medium_capacity_features(
    spy_returns: pd.Series,
    feature_family: str = "baseline",
) -> pd.DataFrame:
    """Build the requested feature family for the stage-1 signal model."""
    if feature_family == "baseline":
        return _build_simple_features(spy_returns)
    if feature_family == "regime_interactions":
        return _build_regime_interaction_features(spy_returns)
    if feature_family == "shock_reversal":
        return _build_shock_reversal_features(spy_returns)
    raise ValueError(f"Unsupported feature_family: {feature_family}")


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return standardized features and their mean/std."""
    mu = np.mean(x, axis=0)
    std = np.std(x, axis=0, ddof=0)
    std = np.where(std < 1e-8, 1.0, std)
    x_std = (x - mu) / std
    return x_std, mu, std


def _build_train_target(train_returns: np.ndarray, cfg: MediumCapacityConfig) -> np.ndarray:
    """Build the bounded stage-1 target for the linear signal model."""
    if cfg.target_mode == "sign":
        return np.sign(train_returns).astype(float)
    if cfg.target_mode == "tanh_return":
        scale = float(np.std(train_returns, ddof=0))
        scale = max(scale, 1e-8)
        return np.tanh(train_returns / scale).astype(float)
    raise ValueError(f"Unsupported target_mode: {cfg.target_mode}")


def _solve_ridge_weights(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    """Solve ridge regression weights for a design matrix with bias column included."""
    XtX = X.T @ X
    XtX_reg = XtX + l2 * np.eye(XtX.shape[0])
    Xty = X.T @ y
    return np.linalg.solve(XtX_reg, Xty)


def _simulate_signal_path(
    signal_clipped: np.ndarray,
    window_returns: np.ndarray,
    cfg: MediumCapacityConfig,
    signal_scale: float,
) -> dict[str, Any]:
    """Project signal to weights and return performance plus gate diagnostics."""
    target_weights = cfg.initial_spy_weight + signal_scale * signal_clipped
    target_weights = np.clip(target_weights, cfg.min_spy_weight, cfg.max_spy_weight)

    weights = np.empty_like(target_weights)
    weights[0] = cfg.initial_spy_weight
    proposed_steps = np.zeros_like(target_weights)
    gate_suppressed = np.zeros_like(target_weights, dtype=bool)
    for t in range(1, len(target_weights)):
        raw_target_delta = target_weights[t] - weights[t - 1]
        candidate_weight = weights[t - 1] + cfg.action_smoothing * raw_target_delta
        proposed_steps[t] = abs(candidate_weight - weights[t - 1])
        # Evaluate the deadband in target-weight space so smoothing does not
        # implicitly widen the threshold by 1 / action_smoothing.
        if abs(raw_target_delta) < cfg.no_trade_band:
            gate_suppressed[t] = True
            candidate_weight = weights[t - 1]
        weights[t] = float(np.clip(candidate_weight, cfg.min_spy_weight, cfg.max_spy_weight))

    turnovers = np.abs(np.diff(weights, prepend=weights[0]))
    wealth = 1.0
    spy_wealth = 1.0
    relative_daily_returns = np.empty(len(window_returns), dtype=float)
    relative_ratio_path = np.empty(len(window_returns), dtype=float)
    for t, (weight, turnover, ret) in enumerate(zip(weights, turnovers, window_returns, strict=False)):
        cost = turnover * (cfg.transaction_cost_bps + cfg.slippage_bps) / 10_000.0
        policy_step = max(1.0 + weight * ret - cost, 1e-8)
        benchmark_step = max(1.0 + ret, 1e-8)
        wealth *= policy_step
        spy_wealth *= benchmark_step
        relative_daily_returns[t] = policy_step / benchmark_step - 1.0
        relative_ratio_path[t] = wealth / max(spy_wealth, 1e-8)

    step_count = max(len(target_weights) - 1, 0)
    signal_abs = np.abs(signal_clipped)
    step_slice = proposed_steps[1:]
    gate_slice = gate_suppressed[1:]
    turnover_slice = turnovers[1:]
    total_return = wealth - 1.0
    relative_return = wealth / max(spy_wealth, 1e-8) - 1.0
    if len(relative_ratio_path) > 0:
        running_peak = np.maximum.accumulate(relative_ratio_path)
        relative_drawdown = 1.0 - relative_ratio_path / np.maximum(running_peak, 1e-8)
        max_relative_drawdown = float(np.max(relative_drawdown))
    else:
        max_relative_drawdown = 0.0
    return {
        "total_return": float(total_return),
        "relative_return": float(relative_return),
        "worst_daily_relative_return": float(np.min(relative_daily_returns)) if len(relative_daily_returns) > 0 else 0.0,
        "max_relative_drawdown": max_relative_drawdown,
        "relative_daily_returns": relative_daily_returns.tolist(),
        "average_weight": float(np.mean(weights)),
        "average_turnover": float(np.mean(turnovers)),
        "signal_abs_p95": float(np.quantile(signal_abs, 0.95)) if len(signal_abs) > 0 else 0.0,
        "signal_abs_max": float(np.max(signal_abs)) if len(signal_abs) > 0 else 0.0,
        "proposed_step_p95": float(np.quantile(step_slice, 0.95)) if step_count > 0 else 0.0,
        "proposed_step_max": float(np.max(step_slice)) if step_count > 0 else 0.0,
        "proposed_steps_over_band": int(np.sum(step_slice >= cfg.no_trade_band)) if step_count > 0 else 0,
        "executed_step_count": int(np.sum(turnover_slice > 1e-12)) if step_count > 0 else 0,
        "gate_suppressed_step_count": int(np.sum(gate_slice)) if step_count > 0 else 0,
        "gate_suppression_rate": float(np.mean(gate_slice)) if step_count > 0 else 0.0,
    }


def train_medium_capacity_model(
    features: pd.DataFrame,
    spy_returns: pd.Series,
    train_end_index: int,
    validation_start_index: int,
    validation_end_index: int,
    cfg: MediumCapacityConfig,
    seed: int = 7,
) -> tuple[MediumCapacityParams, dict[str, Any]]:
    """Train a medium-capacity two-stage model on the validation window.

    Returns:
        params: Learned MediumCapacityParams
        diagnostics: Dict with training metrics
    """
    # Extract training and validation windows
    train_features = features.iloc[:train_end_index + 1].to_numpy(dtype=float)
    train_returns = spy_returns.iloc[:train_end_index + 1].to_numpy(dtype=float)

    val_features = features.iloc[validation_start_index : validation_end_index + 1].to_numpy(dtype=float)
    val_returns = spy_returns.iloc[validation_start_index : validation_end_index + 1].to_numpy(dtype=float)

    # Standardize training features
    train_features_std, feature_mu, feature_std = _standardize(train_features)

    # Standardize validation features using training statistics
    val_features_std = (val_features - feature_mu) / feature_std

    # Target for stage 1: bounded sign or bounded return-magnitude signal.
    train_target = _build_train_target(train_returns, cfg)
    # Ridge regression for the deviation signal
    # X = [1, features...], solve (X^T X + lambda I) w = X^T y
    X_train = np.c_[np.ones(len(train_features_std)), train_features_std]
    X_val = np.c_[np.ones(len(val_features_std)), val_features_std]

    signal_weights_global = _solve_ridge_weights(X_train, train_target, cfg.ridge_l2)
    signal_weights_stress: np.ndarray | None = None
    signal_weights_normal: np.ndarray | None = None

    # Compute predictions on validation according to the configured signal model family.
    if cfg.signal_model_family == "single_linear":
        val_signal = X_val @ signal_weights_global
    elif cfg.signal_model_family == "regime_two_model":
        if "dd_63" in features.columns:
            dd_idx = int(features.columns.get_loc("dd_63"))
            train_dd = train_features[:, dd_idx]
            val_dd = val_features[:, dd_idx]
            train_stress_mask = train_dd <= cfg.regime_drawdown_threshold
            train_normal_mask = ~train_stress_mask
            min_points = 25

            if int(np.sum(train_stress_mask)) >= min_points:
                signal_weights_stress = _solve_ridge_weights(
                    X_train[train_stress_mask],
                    train_target[train_stress_mask],
                    cfg.ridge_l2,
                )
            else:
                signal_weights_stress = signal_weights_global

            if int(np.sum(train_normal_mask)) >= min_points:
                signal_weights_normal = _solve_ridge_weights(
                    X_train[train_normal_mask],
                    train_target[train_normal_mask],
                    cfg.ridge_l2,
                )
            else:
                signal_weights_normal = signal_weights_global

            val_stress_mask = val_dd <= cfg.regime_drawdown_threshold
            val_signal = np.empty(len(X_val), dtype=float)
            val_signal[val_stress_mask] = X_val[val_stress_mask] @ signal_weights_stress
            val_signal[~val_stress_mask] = X_val[~val_stress_mask] @ signal_weights_normal
        else:
            # Fall back to global model when drawdown state is unavailable.
            signal_weights_stress = signal_weights_global
            signal_weights_normal = signal_weights_global
            val_signal = X_val @ signal_weights_global
    else:
        raise ValueError(f"Unsupported signal_model_family: {cfg.signal_model_family}")

    val_signal_clipped = np.clip(val_signal, -1.0, 1.0)

    # Stage 2: map signal to weight, learning the scale
    # target_weight = initial + signal_scale * signal_clipped
    # We want to maximize returns subject to [min_spy_weight, max_spy_weight] bounds
    # Simple approach: find scale that maximizes validation return with actual weight clipping
    best_scale = 0.0
    best_val_return = float("-inf")
    best_validation_result: dict[str, Any] | None = None

    for scale_candidate in np.linspace(cfg.min_signal_scale, cfg.max_signal_scale, 31):
        validation_result = _simulate_signal_path(
            signal_clipped=val_signal_clipped,
            window_returns=val_returns,
            cfg=cfg,
            signal_scale=float(scale_candidate),
        )
        relative_return = float(validation_result["relative_return"])
        objective_value = relative_return - cfg.scale_turnover_penalty * float(validation_result["average_turnover"])

        if objective_value > best_val_return:
            best_val_return = objective_value
            best_scale = scale_candidate
            best_validation_result = validation_result

    if best_validation_result is None:
        raise RuntimeError("Validation scale search did not produce a result.")

    params = MediumCapacityParams(
        signal_weights=signal_weights_global,
        signal_weights_regime_stress=signal_weights_stress,
        signal_weights_regime_normal=signal_weights_normal,
        signal_scale=float(best_scale),
        feature_mu=feature_mu,
        feature_std=feature_std,
    )

    diagnostics = {
        "n_train_samples": len(train_returns),
        "n_val_samples": len(val_returns),
        "signal_bias": float(signal_weights_global[0]),
        "signal_scale": float(best_scale),
        "val_cumulative_return": float(best_validation_result["relative_return"]),
        "val_objective": float(best_val_return),
        "signal_weights_l2": float(np.linalg.norm(signal_weights_global[1:])),
        "signal_model_family": cfg.signal_model_family,
        "validation_signal_abs_p95": float(best_validation_result["signal_abs_p95"]),
        "validation_signal_abs_max": float(best_validation_result["signal_abs_max"]),
        "validation_proposed_step_p95": float(best_validation_result["proposed_step_p95"]),
        "validation_proposed_step_max": float(best_validation_result["proposed_step_max"]),
        "validation_proposed_steps_over_band": int(best_validation_result["proposed_steps_over_band"]),
        "validation_executed_step_count": int(best_validation_result["executed_step_count"]),
        "validation_gate_suppressed_step_count": int(best_validation_result["gate_suppressed_step_count"]),
        "validation_gate_suppression_rate": float(best_validation_result["gate_suppression_rate"]),
    }

    return params, diagnostics


def simulate_medium_capacity_policy(
    features: pd.DataFrame,
    spy_returns: pd.Series,
    start_index: int,
    end_index: int,
    cfg: MediumCapacityConfig,
    params: MediumCapacityParams,
    feature_mu: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> tuple[float, float, float, float, dict[str, Any]]:
    """Simulate the medium-capacity policy on a window.

    Returns:
        total_return, relative_return, average_weight, average_turnover, diagnostics
    """
    window_features = features.iloc[start_index : end_index + 1].to_numpy(dtype=float)
    window_returns = spy_returns.iloc[start_index : end_index + 1].to_numpy(dtype=float)

    mu = params.feature_mu if feature_mu is None else feature_mu
    std = params.feature_std if feature_std is None else feature_std
    window_features_std = (window_features - mu) / std

    # Compute signal
    X = np.c_[np.ones(len(window_features_std)), window_features_std]
    if cfg.signal_model_family == "regime_two_model":
        stress_w = params.signal_weights_regime_stress
        normal_w = params.signal_weights_regime_normal
        if stress_w is not None and normal_w is not None and "dd_63" in features.columns:
            dd_idx = int(features.columns.get_loc("dd_63"))
            window_dd = window_features[:, dd_idx]
            stress_mask = window_dd <= cfg.regime_drawdown_threshold
            signal = np.empty(len(X), dtype=float)
            signal[stress_mask] = X[stress_mask] @ stress_w
            signal[~stress_mask] = X[~stress_mask] @ normal_w
        else:
            signal = X @ params.signal_weights
    else:
        signal = X @ params.signal_weights
    signal_clipped = np.clip(signal, -1.0, 1.0)

    simulation_result = _simulate_signal_path(
        signal_clipped=signal_clipped,
        window_returns=window_returns,
        cfg=cfg,
        signal_scale=float(params.signal_scale),
    )

    return (
        float(simulation_result["total_return"]),
        float(simulation_result["relative_return"]),
        float(simulation_result["average_weight"]),
        float(simulation_result["average_turnover"]),
        simulation_result,
    )
