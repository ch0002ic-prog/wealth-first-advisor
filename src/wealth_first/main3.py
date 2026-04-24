from __future__ import annotations

import argparse
import json
import hashlib
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from wealth_first.data import load_returns_csv
from wealth_first.data_splits import generate_walk_forward_splits


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Main3Config:
    min_spy_weight: float = 0.80
    max_spy_weight: float = 1.05
    initial_spy_weight: float = 1.0
    action_smoothing: float = 0.5
    no_trade_band: float = 0.02
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0
    borrow_cost_bps: float = 0.0
    coarse_candidates: int = 260
    refine_candidates: int = 220
    refine_top_k: int = 16
    refine_noise_scale: float = 0.20
    refine_restart_period: int = 5
    validation_tail_rows: int = 200
    validation_stress_drawdown_threshold: float = -0.10
    validation_stress_min_rows: int = 40
    turnover_penalty: float = 0.40
    cash_penalty: float = 0.30
    borrow_penalty: float = 0.30
    inactivity_penalty: float = 0.0
    max_tail_cash_weight: float = 0.12
    min_tail_turnover: float = 0.0
    calm_tail_cash_cap: float = 0.10
    calm_tail_min_share_for_cash_cap: float = 0.55
    min_tail_relative_return: float = -0.03
    min_stress_relative_return: float = -0.03
    min_full_relative_return: float = -1.0
    hard_min_tail_turnover: float = 0.0
    min_full_relative_return_offensive: float = -1.0
    offensive_full_relative_shortfall_penalty: float = 0.0
    offensive_drawdown_share_min: float = 0.20
    offensive_drawdown_share_max: float = 0.60
    offensive_high_vol_share_max: float = 0.10
    offensive_tail_cash_cap: float = 0.12
    offensive_full_return_boost: float = 0.0
    offensive_full_cash_penalty: float = 0.0
    active_alpha_bonus: float = 0.0
    active_turnover_target: float = 0.003
    max_tail_turnover: float = 0.015
    robustness_windows: int = 3
    robustness_std_penalty: float = 0.08
    min_robust_window_relative_return: float = -0.08
    action_smoothing_min: float = 0.20
    action_smoothing_max: float = 0.75
    no_trade_band_min: float = 0.00
    no_trade_band_max: float = 0.02
    tactical_scale_max: float = 0.22
    tactical_scale_penalty: float = 0.07
    regime_shift_penalty: float = 0.25
    max_regime_shift: float = 0.50
    max_validation_overfit_gap: float = 0.012
    validation_overfit_gap_penalty: float = 0.45
    require_positive_validation_for_activation: bool = True
    min_validation_relative_return_for_activation: float = 0.001
    min_robust_min_relative_for_deployment: float = -0.0015
    use_lcb_selection: bool = True
    pareto_min_candidates: int = 40
    pareto_min_turnover: float = 0.0025


@dataclass(frozen=True)
class Main3PolicyParams:
    bias: float
    w_ret_1: float
    w_ret_21: float
    w_ret_63: float
    w_ret_5: float
    w_ma_gap_63: float
    w_dd_63: float
    w_vol_21: float
    w_vol_5: float
    w_vol_ratio_5_21: float
    w_ret_accel: float
    tactical_scale: float
    action_smoothing: float
    no_trade_band: float


@dataclass(frozen=True)
class WindowMetrics:
    total_return: float
    relative_total_return: float
    average_cash_weight: float
    average_borrow_weight: float
    average_turnover: float
    rebalance_count: int
    periods: int


@dataclass(frozen=True)
class TrainDiagnostics:
    best_objective: float
    coarse_candidates: int
    refine_candidates: int
    passed_candidates: int
    gate_fail_tail_cash: int
    gate_fail_tail_relative: int
    gate_fail_stress_relative: int
    gate_fail_full_relative: int
    gate_fail_turnover: int
    gate_fail_inactive: int
    gate_fail_robust_window: int
    gate_fail_overfit_gap: int
    gate_fail_regime_shift: int
    rescue_gate_used: bool
    fallback_used: bool
    best_param_near_lower_bound_count: int
    best_param_near_upper_bound_count: int
    tail_calm_share: float
    tail_drawdown_share: float
    tail_high_vol_share: float
    objective_weight_full: float
    objective_weight_tail: float
    objective_weight_stress: float
    robust_window_mean_relative: float
    robust_window_min_relative: float
    robust_window_std_relative: float
    validation_relative_total_return: float
    active_selected: bool
    selected_via_pareto: bool
    stress_rows: int
    tail_rows: int


@dataclass(frozen=True)
class CandidateEval:
    params: Main3PolicyParams
    objective: float
    validation_relative_total_return: float
    tail_relative_total_return: float
    stress_relative_total_return: float
    robust_mean_relative: float
    robust_min_relative: float
    robust_std_relative: float
    lcb_relative: float
    overfit_gap: float
    tail_turnover: float
    turnover_target_gap: float
    regime_shift_score: float


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _load_spy_returns(returns_csv: str | Path, benchmark_column: str, date_column: str | None, start: str | None, end: str | None) -> pd.Series:
    frame = load_returns_csv(returns_csv, date_column=date_column)
    if start is not None:
        frame = frame.loc[frame.index >= pd.Timestamp(start)]
    if end is not None:
        frame = frame.loc[frame.index <= pd.Timestamp(end)]
    if benchmark_column not in frame.columns:
        raise ValueError(f"Benchmark column '{benchmark_column}' not found in CSV.")
    series = frame[benchmark_column].astype(float).rename("SPY")
    if series.empty:
        raise ValueError("SPY return series is empty after filtering.")
    return series


def _build_features(spy_returns: pd.Series) -> pd.DataFrame:
    price = (1.0 + spy_returns).cumprod()
    lagged_price = price.shift(1)
    lagged_ret = spy_returns.shift(1)

    def rolling_ret(window: int) -> pd.Series:
        return ((1.0 + spy_returns).rolling(window, min_periods=1).apply(np.prod, raw=True) - 1.0).shift(1)

    def rolling_vol(window: int) -> pd.Series:
        return spy_returns.rolling(window, min_periods=2).std(ddof=0).shift(1)

    def rolling_dd(window: int) -> pd.Series:
        rolling_peak = lagged_price.rolling(window, min_periods=1).max()
        return lagged_price / rolling_peak - 1.0

    def ma_gap(window: int) -> pd.Series:
        ma = lagged_price.rolling(window, min_periods=1).mean()
        return lagged_price / ma - 1.0

    out = pd.DataFrame(
        {
            "ret_1": lagged_ret,
            "ret_5": rolling_ret(5),
            "ret_21": rolling_ret(21),
            "ret_63": rolling_ret(63),
            "ma_gap_63": ma_gap(63),
            "dd_63": rolling_dd(63),
            "vol_5": rolling_vol(5),
            "vol_21": rolling_vol(21),
        },
        index=spy_returns.index,
    )
    out["vol_ratio_5_21"] = (
        out["vol_5"] / out["vol_21"].replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    out["ret_accel"] = out["ret_21"] - out["ret_63"]
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _normalize_features(features: pd.DataFrame, train_end_index: int) -> pd.DataFrame:
    if train_end_index < 0 or train_end_index >= len(features):
        raise ValueError("train_end_index is outside feature frame.")
    train_slice = features.iloc[: train_end_index + 1]
    means = train_slice.mean(axis=0)
    stds = train_slice.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    return ((features - means) / stds).fillna(0.0)


def _linear_score(window_feat: pd.DataFrame, params: Main3PolicyParams) -> np.ndarray:
    return (
        params.bias
        + params.w_ret_1 * window_feat["ret_1"].to_numpy(dtype=float)
        + params.w_ret_5 * window_feat["ret_5"].to_numpy(dtype=float)
        + params.w_ret_21 * window_feat["ret_21"].to_numpy(dtype=float)
        + params.w_ret_63 * window_feat["ret_63"].to_numpy(dtype=float)
        + params.w_ma_gap_63 * window_feat["ma_gap_63"].to_numpy(dtype=float)
        + params.w_dd_63 * window_feat["dd_63"].to_numpy(dtype=float)
        + params.w_vol_5 * window_feat["vol_5"].to_numpy(dtype=float)
        + params.w_vol_21 * window_feat["vol_21"].to_numpy(dtype=float)
        + params.w_vol_ratio_5_21 * window_feat["vol_ratio_5_21"].to_numpy(dtype=float)
        + params.w_ret_accel * window_feat["ret_accel"].to_numpy(dtype=float)
    )


def _simulate_window(
    features: pd.DataFrame,
    spy_returns: pd.Series,
    start_index: int,
    end_index: int,
    cfg: Main3Config,
    params: Main3PolicyParams,
) -> WindowMetrics:
    if end_index < start_index:
        raise ValueError("Window end index must be >= start index.")

    window_feat = features.iloc[start_index : end_index + 1]
    window_ret = spy_returns.iloc[start_index : end_index + 1].to_numpy(dtype=float)
    score = _linear_score(window_feat, params)

    signed_signal = 2.0 * _sigmoid(score) - 1.0
    target_weights = 1.0 + params.tactical_scale * signed_signal
    target_weights = np.clip(target_weights, cfg.min_spy_weight, cfg.max_spy_weight)

    current_weight = float(np.clip(cfg.initial_spy_weight, cfg.min_spy_weight, cfg.max_spy_weight))
    wealth = 1.0
    spy_wealth = 1.0
    turnovers: list[float] = []
    cash_weights: list[float] = []
    borrow_weights: list[float] = []
    rebalance_count = 0

    for t in range(len(window_ret)):
        smoothed = current_weight + (1.0 - params.action_smoothing) * (float(target_weights[t]) - current_weight)
        next_weight = float(np.clip(smoothed, cfg.min_spy_weight, cfg.max_spy_weight))
        if abs(next_weight - current_weight) < params.no_trade_band:
            next_weight = current_weight

        turnover = abs(next_weight - current_weight)
        cost = turnover * (cfg.transaction_cost_bps + cfg.slippage_bps) / 10_000.0
        cash_weight = max(0.0, 1.0 - next_weight)
        borrow_weight = max(0.0, next_weight - 1.0)
        # Approximate daily financing drag for leveraged notional.
        borrow_drag = borrow_weight * (cfg.borrow_cost_bps / 10_000.0) / 252.0
        net_return = next_weight * float(window_ret[t]) - cost - borrow_drag
        wealth *= max(1.0 + net_return, 1e-8)
        spy_wealth *= 1.0 + float(window_ret[t])

        if turnover > 1e-9:
            rebalance_count += 1
        turnovers.append(turnover)
        cash_weights.append(cash_weight)
        borrow_weights.append(borrow_weight)
        current_weight = next_weight

    total_return = wealth - 1.0
    spy_return = spy_wealth - 1.0
    relative = (1.0 + total_return) / (1.0 + spy_return) - 1.0
    return WindowMetrics(
        total_return=float(total_return),
        relative_total_return=float(relative),
        average_cash_weight=float(np.mean(cash_weights)) if cash_weights else 0.0,
        average_borrow_weight=float(np.mean(borrow_weights)) if borrow_weights else 0.0,
        average_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        rebalance_count=rebalance_count,
        periods=len(window_ret),
    )


def _tail_window(start_index: int, end_index: int, tail_rows: int) -> tuple[int, int]:
    if tail_rows <= 0:
        return start_index, end_index
    total_rows = end_index - start_index + 1
    if total_rows <= tail_rows:
        return start_index, end_index
    return end_index - tail_rows + 1, end_index


def _split_into_windows(start_index: int, end_index: int, windows: int) -> list[tuple[int, int]]:
    total = end_index - start_index + 1
    if total <= 0:
        return []
    if windows <= 1:
        return [(start_index, end_index)]
    windows = int(max(1, min(windows, total)))
    base = total // windows
    rem = total % windows
    out: list[tuple[int, int]] = []
    cursor = start_index
    for i in range(windows):
        span = base + (1 if i < rem else 0)
        s = cursor
        e = cursor + span - 1
        out.append((s, e))
        cursor = e + 1
    return out


def _longest_stress_window(spy_returns: pd.Series, start_index: int, end_index: int, drawdown_threshold: float, min_rows: int) -> tuple[int, int]:
    segment = spy_returns.iloc[start_index : end_index + 1]
    wealth = (1.0 + segment).cumprod().to_numpy(dtype=float)
    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / running_max - 1.0
    stress = drawdown <= drawdown_threshold

    best: tuple[int, int] | None = None
    best_len = 0
    in_seg = False
    seg_start = 0
    for i, flag in enumerate(stress):
        if flag and not in_seg:
            seg_start = i
            in_seg = True
        elif not flag and in_seg:
            seg_len = i - seg_start
            if seg_len >= min_rows and seg_len > best_len:
                best = (seg_start, i - 1)
                best_len = seg_len
            in_seg = False
    if in_seg:
        seg_len = len(stress) - seg_start
        if seg_len >= min_rows and seg_len > best_len:
            best = (seg_start, len(stress) - 1)
            best_len = seg_len

    if best is None:
        return start_index, end_index
    return start_index + best[0], start_index + best[1]


def _window_regime_shares(spy_returns: pd.Series, start_index: int, end_index: int) -> dict[str, float]:
    segment = spy_returns.iloc[start_index : end_index + 1].astype(float)
    if segment.empty:
        return {"calm_bull": 0.0, "drawdown": 0.0, "high_vol": 0.0}

    wealth = (1.0 + segment).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    vol63 = segment.rolling(63, min_periods=2).std(ddof=0) * np.sqrt(252)

    calm = ((vol63 <= 0.14) & (drawdown > -0.05)).mean()
    ddn = (drawdown <= -0.10).mean()
    hv = (vol63 >= 0.30).mean()
    return {
        "calm_bull": float(calm) if np.isfinite(calm) else 0.0,
        "drawdown": float(ddn) if np.isfinite(ddn) else 0.0,
        "high_vol": float(hv) if np.isfinite(hv) else 0.0,
    }


def _clip_params(params: Main3PolicyParams, cfg: Main3Config) -> Main3PolicyParams:
    return Main3PolicyParams(
        bias=float(np.clip(params.bias, -0.50, 0.50)),
        w_ret_1=float(np.clip(params.w_ret_1, -1.00, 1.00)),
        w_ret_5=float(np.clip(params.w_ret_5, -1.00, 1.00)),
        w_ret_21=float(np.clip(params.w_ret_21, -0.80, 1.20)),
        w_ret_63=float(np.clip(params.w_ret_63, -0.80, 1.40)),
        w_ma_gap_63=float(np.clip(params.w_ma_gap_63, -0.80, 1.20)),
        w_dd_63=float(np.clip(params.w_dd_63, 0.00, 1.50)),
        w_vol_5=float(np.clip(params.w_vol_5, -1.20, 0.40)),
        w_vol_21=float(np.clip(params.w_vol_21, -1.20, 0.20)),
        w_vol_ratio_5_21=float(np.clip(params.w_vol_ratio_5_21, -0.80, 0.80)),
        w_ret_accel=float(np.clip(params.w_ret_accel, -1.00, 1.00)),
        tactical_scale=float(np.clip(params.tactical_scale, 0.00, cfg.tactical_scale_max)),
        action_smoothing=float(np.clip(params.action_smoothing, cfg.action_smoothing_min, cfg.action_smoothing_max)),
        no_trade_band=float(np.clip(params.no_trade_band, cfg.no_trade_band_min, cfg.no_trade_band_max)),
    )


def _perturb_policy_params(base: Main3PolicyParams, rng: np.random.Generator, scale: float, cfg: Main3Config) -> Main3PolicyParams:
    noise = float(scale)
    return _clip_params(
        Main3PolicyParams(
            bias=base.bias + rng.normal(0.0, 0.15 * noise),
            w_ret_1=base.w_ret_1 + rng.normal(0.0, 0.25 * noise),
            w_ret_5=base.w_ret_5 + rng.normal(0.0, 0.25 * noise),
            w_ret_21=base.w_ret_21 + rng.normal(0.0, 0.30 * noise),
            w_ret_63=base.w_ret_63 + rng.normal(0.0, 0.30 * noise),
            w_ma_gap_63=base.w_ma_gap_63 + rng.normal(0.0, 0.25 * noise),
            w_dd_63=base.w_dd_63 + rng.normal(0.0, 0.25 * noise),
            w_vol_5=base.w_vol_5 + rng.normal(0.0, 0.25 * noise),
            w_vol_21=base.w_vol_21 + rng.normal(0.0, 0.25 * noise),
            w_vol_ratio_5_21=base.w_vol_ratio_5_21 + rng.normal(0.0, 0.20 * noise),
            w_ret_accel=base.w_ret_accel + rng.normal(0.0, 0.25 * noise),
            tactical_scale=base.tactical_scale + rng.normal(0.0, 0.08 * noise),
            action_smoothing=base.action_smoothing + rng.normal(0.0, 0.08 * noise),
            no_trade_band=base.no_trade_band + rng.normal(0.0, 0.006 * noise),
        ),
        cfg,
    )


def _objective_weights_from_tail_regime(tail_regime: dict[str, float]) -> tuple[float, float, float]:
    # If validation tail is dominated by high-vol turbulence, reduce stress/tail influence
    # to avoid selecting checkpoints that overfit crisis-like behavior for calmer tests.
    if tail_regime.get("high_vol", 0.0) >= 0.20:
        return 0.60, 0.25, 0.15
    # If validation tail is predominantly calm, emphasize tail fit more.
    if tail_regime.get("calm_bull", 0.0) >= 0.60:
        return 0.30, 0.50, 0.20
    return 0.35, 0.40, 0.25


def _regime_shift_score(train_regime: dict[str, float], validation_regime: dict[str, float]) -> float:
    return float(
        abs(train_regime.get("calm_bull", 0.0) - validation_regime.get("calm_bull", 0.0))
        + abs(train_regime.get("drawdown", 0.0) - validation_regime.get("drawdown", 0.0))
        + abs(train_regime.get("high_vol", 0.0) - validation_regime.get("high_vol", 0.0))
    )


def _candidate_objective(
    full_val: WindowMetrics,
    tail_val: WindowMetrics,
    stress_val: WindowMetrics,
    cfg: Main3Config,
    tail_regime: dict[str, float],
    regime_shift_score: float,
    objective_weights: tuple[float, float, float],
    robust_relatives: list[float],
) -> float:
    w_full, w_tail, w_stress = objective_weights
    calm_share = float(tail_regime.get("calm_bull", 0.0))
    drawdown_share = float(tail_regime.get("drawdown", 0.0))
    high_vol_share = float(tail_regime.get("high_vol", 0.0))
    extra_offensive_penalty = 0.10 * calm_share * max(0.0, tail_val.average_cash_weight - cfg.calm_tail_cash_cap)
    full_cash_penalty = 0.05 * max(0.0, 1.0 - drawdown_share) * full_val.average_cash_weight
    in_offensive_regime = (
        cfg.offensive_drawdown_share_min <= drawdown_share <= cfg.offensive_drawdown_share_max
        and high_vol_share <= cfg.offensive_high_vol_share_max
    )
    offensive_full_return_term = cfg.offensive_full_return_boost * full_val.total_return if in_offensive_regime else 0.0
    offensive_full_cash_term = cfg.offensive_full_cash_penalty * full_val.average_cash_weight if in_offensive_regime else 0.0
    offensive_full_shortfall_penalty = (
        cfg.offensive_full_relative_shortfall_penalty
        * max(0.0, cfg.min_full_relative_return_offensive - full_val.relative_total_return)
        if in_offensive_regime
        else 0.0
    )
    active_turnover_scale = min(1.0, tail_val.average_turnover / max(cfg.active_turnover_target, 1e-9))
    active_alpha_term = cfg.active_alpha_bonus * max(0.0, full_val.relative_total_return) * active_turnover_scale
    inactivity_shortfall = max(0.0, cfg.min_tail_turnover - tail_val.average_turnover)
    robust_mean = float(np.mean(robust_relatives)) if robust_relatives else 0.0
    overfit_gap = max(0.0, full_val.relative_total_return - robust_mean)
    robust_std = float(np.std(robust_relatives)) if robust_relatives else 0.0
    return (
        w_full * full_val.relative_total_return
        + w_tail * tail_val.relative_total_return
        + w_stress * stress_val.relative_total_return
        + 0.10 * tail_val.total_return
        + offensive_full_return_term
        + active_alpha_term
        - cfg.turnover_penalty * tail_val.average_turnover
        - cfg.cash_penalty * tail_val.average_cash_weight
        - cfg.borrow_penalty * tail_val.average_borrow_weight
        - cfg.inactivity_penalty * inactivity_shortfall
        - extra_offensive_penalty
        - full_cash_penalty
        - offensive_full_cash_term
        - offensive_full_shortfall_penalty
        - cfg.robustness_std_penalty * robust_std
        - cfg.tactical_scale_penalty * (tail_val.average_turnover / max(cfg.active_turnover_target, 1e-9))
        - cfg.regime_shift_penalty * regime_shift_score
        - cfg.validation_overfit_gap_penalty * overfit_gap
    )


def _sample_policy_params(rng: np.random.Generator, cfg: Main3Config) -> Main3PolicyParams:
    return Main3PolicyParams(
        bias=float(rng.uniform(-0.50, 0.50)),
        w_ret_1=float(rng.uniform(-1.00, 1.00)),
        w_ret_5=float(rng.uniform(-1.00, 1.00)),
        w_ret_21=float(rng.uniform(-0.80, 1.20)),
        w_ret_63=float(rng.uniform(-0.80, 1.40)),
        w_ma_gap_63=float(rng.uniform(-0.80, 1.20)),
        w_dd_63=float(rng.uniform(0.00, 1.50)),
        w_vol_5=float(rng.uniform(-1.20, 0.40)),
        w_vol_21=float(rng.uniform(-1.20, 0.20)),
        w_vol_ratio_5_21=float(rng.uniform(-0.80, 0.80)),
        w_ret_accel=float(rng.uniform(-1.00, 1.00)),
        tactical_scale=float(rng.uniform(0.00, cfg.tactical_scale_max)),
        action_smoothing=float(rng.uniform(cfg.action_smoothing_min, cfg.action_smoothing_max)),
        no_trade_band=float(rng.uniform(cfg.no_trade_band_min, cfg.no_trade_band_max)),
    )


def _pareto_select_candidate(candidates: list[CandidateEval], cfg: Main3Config) -> CandidateEval | None:
    if not candidates:
        return None
    active_pool = [c for c in candidates if c.tail_turnover >= cfg.pareto_min_turnover]
    pool = active_pool if active_pool else candidates
    if len(pool) < cfg.pareto_min_candidates:
        # For small pools, select by LCB if enabled, else by objective
        if cfg.use_lcb_selection:
            return max(pool, key=lambda c: c.lcb_relative)
        else:
            return max(pool, key=lambda c: c.objective)

    front: list[CandidateEval] = []
    for c in pool:
        dominated = False
        for d in pool:
            if d is c:
                continue
            # Domination criteria: prefer robust_min, low std, low lcb-based gaps, low turnover
            better_or_equal = (
                d.robust_min_relative >= c.robust_min_relative
                and d.robust_std_relative <= c.robust_std_relative
                and d.lcb_relative >= c.lcb_relative
                and d.overfit_gap <= c.overfit_gap
                and d.regime_shift_score <= c.regime_shift_score
                and d.params.tactical_scale <= c.params.tactical_scale
                and d.turnover_target_gap <= c.turnover_target_gap
            )
            strictly_better = (
                d.robust_min_relative > c.robust_min_relative
                or d.robust_std_relative < c.robust_std_relative
                or d.lcb_relative > c.lcb_relative
                or d.overfit_gap < c.overfit_gap
                or d.regime_shift_score < c.regime_shift_score
                or d.params.tactical_scale < c.params.tactical_scale
                or d.turnover_target_gap < c.turnover_target_gap
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(c)

    return max(
        front,
        key=lambda c: (
            c.lcb_relative,
            c.robust_min_relative,
            -c.robust_std_relative,
            -c.overfit_gap,
            -c.regime_shift_score,
            -c.params.tactical_scale,
            -c.turnover_target_gap,
            c.validation_relative_total_return,
        ),
    )


def _train_policy(
    features: pd.DataFrame,
    spy_returns: pd.Series,
    train_end_index: int,
    validation_start_index: int,
    validation_end_index: int,
    cfg: Main3Config,
    seed: int,
) -> tuple[Main3PolicyParams, TrainDiagnostics]:
    rng = np.random.default_rng(seed)
    best_params: Main3PolicyParams | None = None
    best_score = float("-inf")
    fallback_params: Main3PolicyParams | None = None
    fallback_score = float("-inf")

    passed_candidates = 0
    gate_fail_tail_cash = 0
    gate_fail_tail_relative = 0
    gate_fail_stress_relative = 0
    gate_fail_full_relative = 0
    gate_fail_turnover = 0
    gate_fail_inactive = 0
    gate_fail_robust_window = 0
    gate_fail_overfit_gap = 0
    gate_fail_regime_shift = 0
    rescue_gate_used = False

    full_start, full_end = validation_start_index, validation_end_index
    tail_start, tail_end = _tail_window(validation_start_index, validation_end_index, cfg.validation_tail_rows)
    stress_start, stress_end = _longest_stress_window(
        spy_returns,
        validation_start_index,
        validation_end_index,
        drawdown_threshold=cfg.validation_stress_drawdown_threshold,
        min_rows=cfg.validation_stress_min_rows,
    )

    tail_regime = _window_regime_shares(spy_returns, tail_start, tail_end)
    robust_windows = _split_into_windows(validation_start_index, validation_end_index, cfg.robustness_windows)
    objective_weights = _objective_weights_from_tail_regime(tail_regime)
    offensive_regime = (
        cfg.offensive_drawdown_share_min <= tail_regime["drawdown"] <= cfg.offensive_drawdown_share_max
        and tail_regime["high_vol"] <= cfg.offensive_high_vol_share_max
    )
    tail_cash_cap = cfg.max_tail_cash_weight
    if tail_regime["calm_bull"] >= cfg.calm_tail_min_share_for_cash_cap:
        tail_cash_cap = min(tail_cash_cap, cfg.calm_tail_cash_cap)
    if offensive_regime:
        tail_cash_cap = min(tail_cash_cap, cfg.offensive_tail_cash_cap)
    top_candidates: list[tuple[float, Main3PolicyParams]] = []
    passed_candidate_evals: list[CandidateEval] = []
    best_robust_mean_relative = 0.0
    best_robust_min_relative = 0.0
    best_robust_std_relative = 0.0
    selected_via_pareto = False

    def evaluate_candidate(
        params: Main3PolicyParams,
        gate_tail_cash_cap: float,
        gate_min_tail_relative: float,
        gate_min_stress_relative: float,
        gate_min_full_relative: float,
        gate_min_robust_window_relative: float,
        gate_max_tail_turnover: float,
        gate_hard_min_tail_turnover: float,
        count_failures: bool = True,
    ) -> None:
        nonlocal best_params, best_score, fallback_params, fallback_score
        nonlocal passed_candidates, gate_fail_tail_cash, gate_fail_tail_relative
        nonlocal gate_fail_stress_relative, gate_fail_full_relative, gate_fail_turnover, gate_fail_inactive
        nonlocal gate_fail_robust_window, gate_fail_overfit_gap, gate_fail_regime_shift
        nonlocal best_robust_mean_relative, best_robust_min_relative, best_robust_std_relative

        full_val = _simulate_window(features, spy_returns, full_start, full_end, cfg, params)
        tail_val = _simulate_window(features, spy_returns, tail_start, tail_end, cfg, params)
        stress_val = _simulate_window(features, spy_returns, stress_start, stress_end, cfg, params)
        robust_relatives = [
            _simulate_window(features, spy_returns, ws, we, cfg, params).relative_total_return
            for ws, we in robust_windows
        ]
        train_scores = _linear_score(features.iloc[: train_end_index + 1], params)
        val_scores = _linear_score(features.iloc[full_start : full_end + 1], params)
        regime_shift_score = float(
            abs(float(np.mean(train_scores)) - float(np.mean(val_scores)))
            + abs(float(np.std(train_scores)) - float(np.std(val_scores)))
        )
        robust_mean = float(np.mean(robust_relatives)) if robust_relatives else 0.0
        overfit_gap = max(0.0, full_val.relative_total_return - robust_mean)
        score = _candidate_objective(
            full_val,
            tail_val,
            stress_val,
            cfg,
            tail_regime=tail_regime,
            regime_shift_score=regime_shift_score,
            objective_weights=objective_weights,
            robust_relatives=robust_relatives,
        )

        if score > fallback_score:
            fallback_score = score
            fallback_params = params

        gate_ok = True
        if tail_val.average_cash_weight > gate_tail_cash_cap:
            if count_failures:
                gate_fail_tail_cash += 1
            gate_ok = False
        if tail_val.relative_total_return < gate_min_tail_relative:
            if count_failures:
                gate_fail_tail_relative += 1
            gate_ok = False
        if stress_val.relative_total_return < gate_min_stress_relative:
            if count_failures:
                gate_fail_stress_relative += 1
            gate_ok = False
        if full_val.relative_total_return < gate_min_full_relative:
            if count_failures:
                gate_fail_full_relative += 1
            gate_ok = False
        if robust_relatives and min(robust_relatives) < gate_min_robust_window_relative:
            if count_failures:
                gate_fail_robust_window += 1
            gate_ok = False
        if overfit_gap > cfg.max_validation_overfit_gap:
            if count_failures:
                gate_fail_overfit_gap += 1
            gate_ok = False
        if regime_shift_score > cfg.max_regime_shift:
            if count_failures:
                gate_fail_regime_shift += 1
            gate_ok = False
        if tail_val.average_turnover > gate_max_tail_turnover:
            if count_failures:
                gate_fail_turnover += 1
            gate_ok = False
        if tail_val.average_turnover < cfg.min_tail_turnover:
            if count_failures:
                gate_fail_inactive += 1
        if tail_val.average_turnover < gate_hard_min_tail_turnover:
            gate_ok = False
        if not gate_ok:
            return

        passed_candidates += 1
        robust_min = float(np.min(robust_relatives)) if robust_relatives else 0.0
        robust_std = float(np.std(robust_relatives)) if robust_relatives else 0.0
        # Compute LCB: lower confidence bound at 95% confidence using t-distribution approximation
        # lcb = mean - 1.96 * std / sqrt(n)
        num_robust_windows = len(robust_relatives) if robust_relatives else 1
        lcb = robust_mean - 1.96 * robust_std / np.sqrt(max(1, num_robust_windows))
        passed_candidate_evals.append(
            CandidateEval(
                params=params,
                objective=float(score),
                validation_relative_total_return=float(full_val.relative_total_return),
                tail_relative_total_return=float(tail_val.relative_total_return),
                stress_relative_total_return=float(stress_val.relative_total_return),
                robust_mean_relative=robust_mean,
                robust_min_relative=robust_min,
                robust_std_relative=robust_std,
                lcb_relative=float(lcb),
                overfit_gap=overfit_gap,
                tail_turnover=float(tail_val.average_turnover),
                turnover_target_gap=float(abs(tail_val.average_turnover - cfg.active_turnover_target)),
                regime_shift_score=regime_shift_score,
            )
        )
        top_candidates.append((score, params))
        top_candidates.sort(key=lambda x: x[0], reverse=True)
        if len(top_candidates) > cfg.refine_top_k:
            top_candidates.pop()

        if score > best_score:
            best_score = score
            best_params = params
            best_robust_mean_relative = robust_mean
            best_robust_min_relative = robust_min
            best_robust_std_relative = robust_std

    for _ in range(cfg.coarse_candidates):
        evaluate_candidate(
            _sample_policy_params(rng, cfg),
            gate_tail_cash_cap=tail_cash_cap,
            gate_min_tail_relative=cfg.min_tail_relative_return,
            gate_min_stress_relative=cfg.min_stress_relative_return,
            gate_min_full_relative=cfg.min_full_relative_return,
            gate_min_robust_window_relative=cfg.min_robust_window_relative_return,
            gate_max_tail_turnover=cfg.max_tail_turnover,
            gate_hard_min_tail_turnover=cfg.hard_min_tail_turnover,
        )

    if top_candidates:
        for _ in range(cfg.refine_candidates):
            use_restart = cfg.refine_restart_period > 0 and (_ % cfg.refine_restart_period == 0)
            if use_restart:
                candidate = _sample_policy_params(rng, cfg)
            else:
                _, base = top_candidates[int(rng.integers(0, len(top_candidates)))]
                candidate = _perturb_policy_params(base, rng, cfg.refine_noise_scale, cfg)
            evaluate_candidate(
                candidate,
                gate_tail_cash_cap=tail_cash_cap,
                gate_min_tail_relative=cfg.min_tail_relative_return,
                gate_min_stress_relative=cfg.min_stress_relative_return,
                gate_min_full_relative=cfg.min_full_relative_return,
                gate_min_robust_window_relative=cfg.min_robust_window_relative_return,
                gate_max_tail_turnover=cfg.max_tail_turnover,
                gate_hard_min_tail_turnover=cfg.hard_min_tail_turnover,
            )

    if best_params is None:
        rescue_gate_used = True
        rescue_tail_cash_cap = min(0.18, tail_cash_cap + 0.02)
        rescue_min_tail_relative = cfg.min_tail_relative_return - 0.02
        rescue_min_stress_relative = cfg.min_stress_relative_return - 0.02
        rescue_min_full_relative = cfg.min_full_relative_return - 0.02
        rescue_min_robust_window_relative = cfg.min_robust_window_relative_return - 0.02
        rescue_max_tail_turnover = cfg.max_tail_turnover + 0.005
        rescue_hard_min_tail_turnover = max(0.0, cfg.hard_min_tail_turnover * 0.5)
        for _ in range(max(80, cfg.refine_candidates // 2)):
            evaluate_candidate(
                _sample_policy_params(rng, cfg),
                gate_tail_cash_cap=rescue_tail_cash_cap,
                gate_min_tail_relative=rescue_min_tail_relative,
                gate_min_stress_relative=rescue_min_stress_relative,
                gate_min_full_relative=rescue_min_full_relative,
                gate_min_robust_window_relative=rescue_min_robust_window_relative,
                gate_max_tail_turnover=rescue_max_tail_turnover,
                gate_hard_min_tail_turnover=rescue_hard_min_tail_turnover,
                count_failures=False,
            )

    fallback_used = False
    if best_params is None:
        fallback_used = True
        if fallback_params is None:
            fallback_params = _sample_policy_params(rng, cfg)
            fallback_score = float("nan")
        best_params = fallback_params
        best_score = fallback_score

    # Prefer stability-selected candidate when candidate pool is sufficiently rich.
    stable_choice = _pareto_select_candidate(passed_candidate_evals, cfg)
    if stable_choice is not None:
        selected_via_pareto = len(passed_candidate_evals) >= cfg.pareto_min_candidates
        best_params = stable_choice.params
        best_score = stable_choice.objective
        best_robust_mean_relative = stable_choice.robust_mean_relative
        best_robust_min_relative = stable_choice.robust_min_relative
        best_robust_std_relative = stable_choice.robust_std_relative

    _PARAM_BOUNDS = [
        (best_params.bias,              -0.50, 0.50),
        (best_params.w_ret_1,           -1.00, 1.00),
        (best_params.w_ret_5,           -1.00, 1.00),
        (best_params.w_ret_21,          -0.80, 1.20),
        (best_params.w_ret_63,          -0.80, 1.40),
        (best_params.w_ma_gap_63,       -0.80, 1.20),
        (best_params.w_dd_63,            0.00, 1.50),
        (best_params.w_vol_5,          -1.20, 0.40),
        (best_params.w_vol_21,          -1.20, 0.20),
        (best_params.w_vol_ratio_5_21, -0.80, 0.80),
        (best_params.w_ret_accel,      -1.00, 1.00),
        (best_params.tactical_scale,     0.00, cfg.tactical_scale_max),
        (best_params.action_smoothing,   cfg.action_smoothing_min, cfg.action_smoothing_max),
        (best_params.no_trade_band,      cfg.no_trade_band_min, cfg.no_trade_band_max),
    ]
    _BOUND_TOL = 0.02  # fraction of range considered "near" a bound
    near_lower = sum(
        1 for v, lo, hi in _PARAM_BOUNDS if v - lo < _BOUND_TOL * (hi - lo)
    )
    near_upper = sum(
        1 for v, lo, hi in _PARAM_BOUNDS if hi - v < _BOUND_TOL * (hi - lo)
    )

    validation_relative_total_return = float(
        _simulate_window(features, spy_returns, full_start, full_end, cfg, best_params).relative_total_return
    )
    
    # Compute robust metrics for best params to check deployment gate
    best_params_robust_relatives = [
        _simulate_window(features, spy_returns, ws, we, cfg, best_params).relative_total_return
        for ws, we in robust_windows
    ]
    best_params_robust_min = float(np.min(best_params_robust_relatives)) if best_params_robust_relatives else 0.0
    
    # Activation gate: require BOTH validation >= threshold AND robust_min >= deployment threshold
    active_selected = bool(
        (not cfg.require_positive_validation_for_activation)
        or (validation_relative_total_return >= cfg.min_validation_relative_return_for_activation
            and best_params_robust_min >= cfg.min_robust_min_relative_for_deployment)
    )

    diagnostics = TrainDiagnostics(
        best_objective=float(best_score),
        coarse_candidates=int(cfg.coarse_candidates),
        refine_candidates=int(cfg.refine_candidates),
        passed_candidates=int(passed_candidates),
        gate_fail_tail_cash=int(gate_fail_tail_cash),
        gate_fail_tail_relative=int(gate_fail_tail_relative),
        gate_fail_stress_relative=int(gate_fail_stress_relative),
        gate_fail_full_relative=int(gate_fail_full_relative),
        gate_fail_turnover=int(gate_fail_turnover),
        gate_fail_inactive=int(gate_fail_inactive),
        gate_fail_robust_window=int(gate_fail_robust_window),
        gate_fail_overfit_gap=int(gate_fail_overfit_gap),
        gate_fail_regime_shift=int(gate_fail_regime_shift),
        rescue_gate_used=bool(rescue_gate_used),
        fallback_used=bool(fallback_used),
        best_param_near_lower_bound_count=int(near_lower),
        best_param_near_upper_bound_count=int(near_upper),
        tail_calm_share=float(tail_regime["calm_bull"]),
        tail_drawdown_share=float(tail_regime["drawdown"]),
        tail_high_vol_share=float(tail_regime["high_vol"]),
        objective_weight_full=float(objective_weights[0]),
        objective_weight_tail=float(objective_weights[1]),
        objective_weight_stress=float(objective_weights[2]),
        robust_window_mean_relative=float(best_robust_mean_relative),
        robust_window_min_relative=float(best_robust_min_relative),
        robust_window_std_relative=float(best_robust_std_relative),
        validation_relative_total_return=validation_relative_total_return,
        active_selected=active_selected,
        selected_via_pareto=selected_via_pareto,
        stress_rows=int(stress_end - stress_start + 1),
        tail_rows=int(tail_end - tail_start + 1),
    )
    return best_params, diagnostics


def _static_hold_return(spy_returns: pd.Series, start_index: int, end_index: int) -> float:
    segment = spy_returns.iloc[start_index : end_index + 1].to_numpy(dtype=float)
    return float(np.prod(1.0 + segment) - 1.0)


def _static_hold_metrics(spy_returns: pd.Series, start_index: int, end_index: int) -> WindowMetrics:
    total_return = _static_hold_return(spy_returns, start_index, end_index)
    return WindowMetrics(
        total_return=total_return,
        relative_total_return=0.0,
        average_cash_weight=0.0,
        average_borrow_weight=0.0,
        average_turnover=0.0,
        rebalance_count=0,
        periods=int(end_index - start_index + 1),
    )


def _build_summary(detail: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {}
    out["overall"] = {
        "rows": int(len(detail)),
        "mean_policy_total_return": float(detail["policy_total_return"].mean()),
        "mean_static_hold_total_return": float(detail["static_hold_total_return"].mean()),
        "mean_policy_relative_total_return": float(detail["policy_relative_total_return"].mean()),
        "mean_policy_cash_weight": float(detail["policy_cash_weight"].mean()),
        "mean_policy_borrow_weight": float(detail["policy_borrow_weight"].mean()),
        "mean_policy_turnover": float(detail["policy_turnover"].mean()),
    }
    by_phase: dict[str, object] = {}
    for phase_name, group in detail.groupby("phase"):
        by_phase[str(phase_name)] = {
            "rows": int(len(group)),
            "mean_policy_total_return": float(group["policy_total_return"].mean()),
            "mean_static_hold_total_return": float(group["static_hold_total_return"].mean()),
            "mean_policy_relative_total_return": float(group["policy_relative_total_return"].mean()),
            "beat_static_rate": float((group["delta_total_return_vs_static_hold"] > 0.0).mean()),
        }
    out["by_phase"] = by_phase

    diag_cols = [
        "train_diag_passed_candidates",
        "train_diag_coarse_candidates",
        "train_diag_refine_candidates",
        "train_diag_gate_fail_tail_cash",
        "train_diag_gate_fail_tail_relative",
        "train_diag_gate_fail_stress_relative",
        "train_diag_gate_fail_full_relative",
        "train_diag_gate_fail_turnover",
        "train_diag_gate_fail_inactive",
        "train_diag_gate_fail_robust_window",
        "train_diag_gate_fail_overfit_gap",
        "train_diag_gate_fail_regime_shift",
        "train_diag_rescue_gate_used",
        "train_diag_near_lower_bound_count",
        "train_diag_near_upper_bound_count",
        "train_diag_tail_calm_share",
        "train_diag_tail_drawdown_share",
        "train_diag_tail_high_vol_share",
        "train_diag_objective_weight_full",
        "train_diag_objective_weight_tail",
        "train_diag_objective_weight_stress",
        "train_diag_robust_window_mean_relative",
        "train_diag_robust_window_min_relative",
        "train_diag_robust_window_std_relative",
        "train_diag_validation_relative_total_return",
        "train_diag_active_selected",
        "train_diag_selected_via_pareto",
        "train_diag_tail_rows",
        "train_diag_stress_rows",
    ]
    available_diag_cols = [c for c in diag_cols if c in detail.columns]
    if available_diag_cols:
        diag = detail[available_diag_cols].copy()
        for c in diag.columns:
            diag[c] = pd.to_numeric(diag[c], errors="coerce")
        out["diagnostics"] = {
            f"mean_{col}": float(diag[col].mean())
            for col in available_diag_cols
            if not np.isnan(diag[col].mean())
        }
        if "train_diag_fallback_used" in detail.columns:
            fallback = detail["train_diag_fallback_used"].astype(bool)
            out["diagnostics"]["fallback_rate"] = float(fallback.mean())

    return out


def _build_run_fingerprint(args: argparse.Namespace, cfg: Main3Config) -> dict[str, object]:
    this_file = Path(__file__).resolve()
    file_sha256 = hashlib.sha256(this_file.read_bytes()).hexdigest()

    git_commit = None
    git_dirty = None
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(PROJECT_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        git_dirty = bool(status.strip())
    except Exception:
        pass

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "main3_file": str(this_file),
        "main3_file_sha256": file_sha256,
        "project_root": str(PROJECT_ROOT),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "cli_args": vars(args),
        "config": cfg.__dict__,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Main3: fresh SPY-only regime-aware policy search.")
    p.add_argument("--returns-csv", default="data/demo_sleeves.csv")
    p.add_argument("--benchmark-column", default="SPY_BENCHMARK")
    p.add_argument("--date-column", default="date")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--splits", nargs="+", choices=["chrono"], default=["chrono"])
    p.add_argument("--walk-forward-folds", type=int, default=3)
    p.add_argument("--validation-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.10)
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    p.add_argument("--coarse-candidates", type=int, default=260)
    p.add_argument("--refine-candidates", type=int, default=220)
    p.add_argument("--refine-top-k", type=int, default=16)
    p.add_argument("--refine-noise-scale", type=float, default=0.20)
    p.add_argument("--refine-restart-period", type=int, default=5)
    p.add_argument("--validation-tail-rows", type=int, default=200)
    p.add_argument("--validation-stress-drawdown-threshold", type=float, default=-0.10)
    p.add_argument("--validation-stress-min-rows", type=int, default=40)
    p.add_argument("--min-spy-weight", type=float, default=0.80)
    p.add_argument("--max-spy-weight", type=float, default=1.05)
    p.add_argument("--initial-spy-weight", type=float, default=1.0)
    p.add_argument("--transaction-cost-bps", type=float, default=5.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--borrow-cost-bps", type=float, default=0.0)
    p.add_argument("--turnover-penalty", type=float, default=0.40)
    p.add_argument("--cash-penalty", type=float, default=0.30)
    p.add_argument("--borrow-penalty", type=float, default=0.30)
    p.add_argument("--inactivity-penalty", type=float, default=0.0)
    p.add_argument("--max-tail-cash-weight", type=float, default=0.12)
    p.add_argument("--min-tail-turnover", type=float, default=0.0)
    p.add_argument("--calm-tail-cash-cap", type=float, default=0.10)
    p.add_argument("--calm-tail-min-share-for-cash-cap", type=float, default=0.55)
    p.add_argument("--min-tail-relative-return", type=float, default=-0.03)
    p.add_argument("--min-stress-relative-return", type=float, default=-0.03)
    p.add_argument("--min-full-relative-return", type=float, default=-1.0)
    p.add_argument("--hard-min-tail-turnover", type=float, default=0.0)
    p.add_argument("--min-full-relative-return-offensive", type=float, default=-1.0)
    p.add_argument("--offensive-full-relative-shortfall-penalty", type=float, default=0.0)
    p.add_argument("--offensive-drawdown-share-min", type=float, default=0.20)
    p.add_argument("--offensive-drawdown-share-max", type=float, default=0.60)
    p.add_argument("--offensive-high-vol-share-max", type=float, default=0.10)
    p.add_argument("--offensive-tail-cash-cap", type=float, default=0.12)
    p.add_argument("--offensive-full-return-boost", type=float, default=0.0)
    p.add_argument("--offensive-full-cash-penalty", type=float, default=0.0)
    p.add_argument("--active-alpha-bonus", type=float, default=0.0)
    p.add_argument("--active-turnover-target", type=float, default=0.003)
    p.add_argument("--max-tail-turnover", type=float, default=0.015)
    p.add_argument("--robustness-windows", type=int, default=3)
    p.add_argument("--robustness-std-penalty", type=float, default=0.08)
    p.add_argument("--min-robust-window-relative-return", type=float, default=-0.08)
    p.add_argument("--action-smoothing-min", type=float, default=0.20)
    p.add_argument("--action-smoothing-max", type=float, default=0.75)
    p.add_argument("--no-trade-band-min", type=float, default=0.00)
    p.add_argument("--no-trade-band-max", type=float, default=0.02)
    p.add_argument("--tactical-scale-max", type=float, default=0.22)
    p.add_argument("--tactical-scale-penalty", type=float, default=0.07)
    p.add_argument("--regime-shift-penalty", type=float, default=0.25)
    p.add_argument("--max-regime-shift", type=float, default=0.50)
    p.add_argument("--max-validation-overfit-gap", type=float, default=0.012)
    p.add_argument("--validation-overfit-gap-penalty", type=float, default=0.45)
    p.add_argument("--require-positive-validation-for-activation", action="store_true")
    p.add_argument("--no-require-positive-validation-for-activation", action="store_true")
    p.add_argument("--min-validation-relative-return-for-activation", type=float, default=0.001)
    p.add_argument("--min-robust-min-relative-for-deployment", type=float, default=-0.0015)
    p.add_argument("--use-lcb-selection", action="store_true", default=True)
    p.add_argument("--no-use-lcb-selection", action="store_false", dest="use_lcb_selection")
    p.add_argument("--pareto-min-candidates", type=int, default=40)
    p.add_argument("--pareto-min-turnover", type=float, default=0.0025)
    p.add_argument("--output-prefix", default="artifacts/main3_spy_fresh")
    return p


def _config_from_args(args: argparse.Namespace) -> Main3Config:
    require_positive_validation_for_activation = True
    if args.no_require_positive_validation_for_activation:
        require_positive_validation_for_activation = False
    if args.require_positive_validation_for_activation:
        require_positive_validation_for_activation = True

    return Main3Config(
        min_spy_weight=args.min_spy_weight,
        max_spy_weight=args.max_spy_weight,
        initial_spy_weight=args.initial_spy_weight,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        borrow_cost_bps=args.borrow_cost_bps,
        coarse_candidates=args.coarse_candidates,
        refine_candidates=args.refine_candidates,
        refine_top_k=args.refine_top_k,
        refine_noise_scale=args.refine_noise_scale,
        refine_restart_period=args.refine_restart_period,
        validation_tail_rows=args.validation_tail_rows,
        validation_stress_drawdown_threshold=args.validation_stress_drawdown_threshold,
        validation_stress_min_rows=args.validation_stress_min_rows,
        turnover_penalty=args.turnover_penalty,
        cash_penalty=args.cash_penalty,
        borrow_penalty=args.borrow_penalty,
        inactivity_penalty=args.inactivity_penalty,
        max_tail_cash_weight=args.max_tail_cash_weight,
        min_tail_turnover=args.min_tail_turnover,
        calm_tail_cash_cap=args.calm_tail_cash_cap,
        calm_tail_min_share_for_cash_cap=args.calm_tail_min_share_for_cash_cap,
        min_tail_relative_return=args.min_tail_relative_return,
        min_stress_relative_return=args.min_stress_relative_return,
        min_full_relative_return=args.min_full_relative_return,
        hard_min_tail_turnover=args.hard_min_tail_turnover,
        min_full_relative_return_offensive=args.min_full_relative_return_offensive,
        offensive_full_relative_shortfall_penalty=args.offensive_full_relative_shortfall_penalty,
        offensive_drawdown_share_min=args.offensive_drawdown_share_min,
        offensive_drawdown_share_max=args.offensive_drawdown_share_max,
        offensive_high_vol_share_max=args.offensive_high_vol_share_max,
        offensive_tail_cash_cap=args.offensive_tail_cash_cap,
        offensive_full_return_boost=args.offensive_full_return_boost,
        offensive_full_cash_penalty=args.offensive_full_cash_penalty,
        active_alpha_bonus=args.active_alpha_bonus,
        active_turnover_target=args.active_turnover_target,
        max_tail_turnover=args.max_tail_turnover,
        robustness_windows=args.robustness_windows,
        robustness_std_penalty=args.robustness_std_penalty,
        min_robust_window_relative_return=args.min_robust_window_relative_return,
        action_smoothing_min=args.action_smoothing_min,
        action_smoothing_max=args.action_smoothing_max,
        no_trade_band_min=args.no_trade_band_min,
        no_trade_band_max=args.no_trade_band_max,
        tactical_scale_max=args.tactical_scale_max,
        tactical_scale_penalty=args.tactical_scale_penalty,
        regime_shift_penalty=args.regime_shift_penalty,
        max_regime_shift=args.max_regime_shift,
        max_validation_overfit_gap=args.max_validation_overfit_gap,
        validation_overfit_gap_penalty=args.validation_overfit_gap_penalty,
        require_positive_validation_for_activation=require_positive_validation_for_activation,
        min_validation_relative_return_for_activation=args.min_validation_relative_return_for_activation,
        min_robust_min_relative_for_deployment=args.min_robust_min_relative_for_deployment,
        use_lcb_selection=args.use_lcb_selection,
        pareto_min_candidates=args.pareto_min_candidates,
        pareto_min_turnover=args.pareto_min_turnover,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = _config_from_args(args)

    spy_returns = _load_spy_returns(
        returns_csv=args.returns_csv,
        benchmark_column=args.benchmark_column,
        date_column=args.date_column,
        start=args.start,
        end=args.end,
    )
    features = _build_features(spy_returns)

    split_families = {
        split_name: generate_walk_forward_splits(
            spy_returns.to_frame(name="SPY"),
            benchmark_returns=spy_returns,
            lookback=20,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            max_splits=args.walk_forward_folds,
        )
        for split_name in args.splits
    }

    rows: list[dict[str, object]] = []
    for split_name, splits in split_families.items():
        for seed in args.seeds:
            for fold_number, split in enumerate(splits, start=1):
                normalized = _normalize_features(features, train_end_index=split.train.end_index - 1)
                params, train_diag = _train_policy(
                    features=normalized,
                    spy_returns=spy_returns,
                    train_end_index=split.train.end_index - 1,
                    validation_start_index=split.validation.start_index,
                    validation_end_index=split.validation.end_index - 1,
                    cfg=cfg,
                    seed=int(seed),
                )

                for phase_name, start_index, end_index, start_label, end_label in [
                    (
                        "validation",
                        split.validation.start_index,
                        split.validation.end_index - 1,
                        split.validation.start_label,
                        split.validation.end_label,
                    ),
                    (
                        "test",
                        split.test.start_index,
                        split.test.end_index - 1,
                        split.test.start_label,
                        split.test.end_label,
                    ),
                ]:
                    if train_diag.active_selected:
                        policy_metrics = _simulate_window(normalized, spy_returns, start_index, end_index, cfg, params)
                    else:
                        policy_metrics = _static_hold_metrics(spy_returns, start_index, end_index)
                    static_hold_total_return = _static_hold_return(spy_returns, start_index, end_index)
                    rows.append(
                        {
                            "split": split_name,
                            "seed": int(seed),
                            "fold": f"fold_{fold_number:02d}",
                            "phase": phase_name,
                            "phase_start": str(start_label),
                            "phase_end": str(end_label),
                            "policy_total_return": policy_metrics.total_return,
                            "static_hold_total_return": static_hold_total_return,
                            "policy_relative_total_return": policy_metrics.relative_total_return,
                            "policy_cash_weight": policy_metrics.average_cash_weight,
                            "policy_borrow_weight": policy_metrics.average_borrow_weight,
                            "policy_turnover": policy_metrics.average_turnover,
                            "policy_rebalance_count": policy_metrics.rebalance_count,
                            "delta_total_return_vs_static_hold": policy_metrics.total_return - static_hold_total_return,
                            "train_best_validation_objective": train_diag.best_objective,
                            "train_diag_passed_candidates": train_diag.passed_candidates,
                            "train_diag_coarse_candidates": train_diag.coarse_candidates,
                            "train_diag_refine_candidates": train_diag.refine_candidates,
                            "train_diag_gate_fail_tail_cash": train_diag.gate_fail_tail_cash,
                            "train_diag_gate_fail_tail_relative": train_diag.gate_fail_tail_relative,
                            "train_diag_gate_fail_stress_relative": train_diag.gate_fail_stress_relative,
                            "train_diag_gate_fail_full_relative": train_diag.gate_fail_full_relative,
                            "train_diag_gate_fail_turnover": train_diag.gate_fail_turnover,
                            "train_diag_gate_fail_inactive": train_diag.gate_fail_inactive,
                            "train_diag_gate_fail_robust_window": train_diag.gate_fail_robust_window,
                            "train_diag_gate_fail_overfit_gap": train_diag.gate_fail_overfit_gap,
                            "train_diag_gate_fail_regime_shift": train_diag.gate_fail_regime_shift,
                            "train_diag_rescue_gate_used": train_diag.rescue_gate_used,
                            "train_diag_fallback_used": train_diag.fallback_used,
                            "train_diag_near_lower_bound_count": train_diag.best_param_near_lower_bound_count,
                            "train_diag_near_upper_bound_count": train_diag.best_param_near_upper_bound_count,
                            "train_diag_tail_calm_share": train_diag.tail_calm_share,
                            "train_diag_tail_drawdown_share": train_diag.tail_drawdown_share,
                            "train_diag_tail_high_vol_share": train_diag.tail_high_vol_share,
                            "train_diag_objective_weight_full": train_diag.objective_weight_full,
                            "train_diag_objective_weight_tail": train_diag.objective_weight_tail,
                            "train_diag_objective_weight_stress": train_diag.objective_weight_stress,
                            "train_diag_robust_window_mean_relative": train_diag.robust_window_mean_relative,
                            "train_diag_robust_window_min_relative": train_diag.robust_window_min_relative,
                            "train_diag_robust_window_std_relative": train_diag.robust_window_std_relative,
                            "train_diag_validation_relative_total_return": train_diag.validation_relative_total_return,
                            "train_diag_active_selected": train_diag.active_selected,
                            "train_diag_selected_via_pareto": train_diag.selected_via_pareto,
                            "train_diag_tail_rows": train_diag.tail_rows,
                            "train_diag_stress_rows": train_diag.stress_rows,
                            "param_bias": params.bias,
                            "param_w_ret_1": params.w_ret_1,
                            "param_w_ret_5": params.w_ret_5,
                            "param_w_ret_21": params.w_ret_21,
                            "param_w_ret_63": params.w_ret_63,
                            "param_w_ma_gap_63": params.w_ma_gap_63,
                            "param_w_dd_63": params.w_dd_63,
                            "param_w_vol_5": params.w_vol_5,
                            "param_w_vol_21": params.w_vol_21,
                            "param_w_vol_ratio_5_21": params.w_vol_ratio_5_21,
                            "param_w_ret_accel": params.w_ret_accel,
                            "param_tactical_scale": params.tactical_scale,
                            "param_action_smoothing": params.action_smoothing,
                            "param_no_trade_band": params.no_trade_band,
                        }
                    )

    detail = pd.DataFrame(rows).sort_values(["split", "seed", "fold", "phase"]).reset_index(drop=True)
    summary = _build_summary(detail)
    summary["run_fingerprint"] = _build_run_fingerprint(args, cfg)

    output_prefix = Path(args.output_prefix)
    if not output_prefix.is_absolute():
        output_prefix = PROJECT_ROOT / output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    detail_path = output_prefix.with_name(f"{output_prefix.name}_detail.csv")
    summary_path = output_prefix.with_name(f"{output_prefix.name}_summary.json")
    detail.round(6).to_csv(detail_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(detail_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())