"""Main6: Multi-scale signal policy with explicit fold-health accounting.

Root causes fixed vs main5
--------------------------
1. **Joint hard activity caps eliminated** — main5's
   ``validation_hard_min_step_rate=0.001`` +
   ``validation_hard_max_suppression_rate=0.9988`` killed folds completely in
   off-slice periods: in 2020-2026, 3/5 folds returned step_rate=0.0 at ALL 31
   scale candidates.  Main6 removes these as defaults; they remain opt-in via
   CLI flags for backward comparison only.

2. **Reduced no_trade_band** — from 0.05 (main5 c_objw05) to 0.02.  The root
   failure mechanism is: after the first executed step, action_smoothing=0.9231
   drives the weight so close to the target that subsequent raw_target_delta
   values are tiny (≈ 0.077 × signal_change × scale).  A 5% deadband requires
   signal changes > 65% of full range to trigger re-execution.  A 2% deadband
   requires only 26%.  This is the most direct fix short of removing smoothing.

3. **Richer multi-scale feature set** (11 features vs 3) — adds cross-regime
   indicators that provide non-zero signal in volatile regimes (2020-2026):
   vol_5, vol_63, vol_ratio_5_21, vol_ratio_21_63, ret_5, ret_21, dd_21,
   ma_ratio_63.  The main5 3-feature significance screen was optimised in-sample
   (2007-2026) and dropped features that still carry cross-regime information.

4. **Explicit fold-death tracking** — dead folds (RuntimeError in scale search)
   are now recorded in fold_diagnostics with fold_status="dead" and
   policy_relative_total_return=NaN.  Summary includes n_dead_folds,
   n_live_folds, dead_fold_fraction.  A new gate check rejects runs where
   dead_fold_fraction > max_dead_fold_fraction (default 0.20 = at most 1/5
   folds dead).

5. **Wider scale search** — default range [-0.30, +0.30] with 51 candidates
   (vs [-0.20, +0.20] / 31 in main5).  This extends the reachable signal
   magnitude without increasing parameter complexity.

Architecture unchanged
----------------------
- Same ridge regression training on horizon-aligned tanh(fwd_21d/std) target.
- Same MediumCapacityConfig / simulate_medium_capacity_policy execution layer.
- Same 5-fold chrono walk-forward splits.
- Same gate_checks schema (compatible with downstream compare.py).
- Output: ``main6_gate{gate}_f{n_folds}_s{seed}_detailed.json`` +
  ``main6_gate{gate}_f{n_folds}_s{seed}_summary.csv``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wealth_first.data import load_returns_csv
from wealth_first.data_splits import generate_walk_forward_splits
from wealth_first.medium_capacity import (
    MediumCapacityConfig,
    MediumCapacityParams,
    _compute_bootstrap_total_return_quantile,  # type: ignore[attr-defined]
    _simulate_signal_path,  # type: ignore[attr-defined]
    simulate_medium_capacity_policy,
)


# ---------------------------------------------------------------------------
# Feature construction  (FIX #3 — multi-scale)
# ---------------------------------------------------------------------------

_FEATURE_NAMES = (
    "vol_5",
    "vol_21",
    "vol_63",
    "vol_ratio_5_21",
    "vol_ratio_21_63",
    "ret_5",
    "ret_21",
    "ret_63",
    "dd_21",
    "dd_63",
    "ma_ratio_63",
)


def build_main6_features(spy_returns: pd.Series) -> pd.DataFrame:
    """Build the 11-factor feature set for main6 (all lagged 1 day).

    Core factors (retained from main5 significance screen):
      vol_21       — 21-day realized volatility (IC_21d ~0.234)
      dd_63        — 63-day drawdown from peak  (IC_21d ~0.267)
      ret_63       — 63-day cumulative return   (IC_21d ~0.230)

    New cross-regime factors:
      vol_5        — 5-day vol; detects acute shocks (COVID, 2008 spike)
      vol_63       — 63-day vol; identifies low-vol vs high-vol regimes
      vol_ratio_5_21  — vol_5 / vol_21 − 1; short vol spike indicator
      vol_ratio_21_63 — vol_21 / vol_63 − 1; medium vol regime transition
      ret_5        — 5-day momentum/reversal signal
      ret_21       — 21-day momentum (dropped by main5 IC screen but adds
                     cross-regime coverage when combined with other factors)
      dd_21        — 21-day drawdown; local stress
      ma_ratio_63  — price / 63-day MA − 1; trend position / mean reversion

    All features are forward-filled from lagged inputs (no lookahead).
    """
    price = (1.0 + spy_returns).cumprod()
    lp = price.shift(1)  # lagged price

    def _vol(n: int) -> pd.Series:
        return spy_returns.rolling(n, min_periods=2).std(ddof=0).shift(1)

    def _ret(n: int) -> pd.Series:
        return (
            (1.0 + spy_returns).rolling(n, min_periods=1).apply(np.prod, raw=True) - 1.0
        ).shift(1)

    def _dd(n: int) -> pd.Series:
        peak = lp.rolling(n, min_periods=1).max()
        return lp / peak - 1.0

    vol_5 = _vol(5)
    vol_21 = _vol(21)
    vol_63 = _vol(63)

    vol_ratio_5_21 = vol_5 / np.maximum(vol_21, 1e-8) - 1.0
    vol_ratio_21_63 = vol_21 / np.maximum(vol_63, 1e-8) - 1.0

    ret_5 = _ret(5)
    ret_21 = _ret(21)
    ret_63 = _ret(63)

    dd_21 = _dd(21)
    dd_63 = _dd(63)

    ma_63 = lp.rolling(63, min_periods=1).mean()
    ma_ratio_63 = lp / np.maximum(ma_63, 1e-8) - 1.0

    out = pd.DataFrame(
        {
            "vol_5": vol_5,
            "vol_21": vol_21,
            "vol_63": vol_63,
            "vol_ratio_5_21": vol_ratio_5_21,
            "vol_ratio_21_63": vol_ratio_21_63,
            "ret_5": ret_5,
            "ret_21": ret_21,
            "ret_63": ret_63,
            "dd_21": dd_21,
            "dd_63": dd_63,
            "ma_ratio_63": ma_ratio_63,
        },
        index=spy_returns.index,
    )
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


# ---------------------------------------------------------------------------
# Horizon-aligned training target  (identical to main5)
# ---------------------------------------------------------------------------


def _build_main6_target(
    spy_returns: pd.Series,
    horizon: int = 21,
    target_temperature: float = 1.0,
) -> np.ndarray:
    """tanh-normalised ``horizon``-day forward cumulative return."""
    if target_temperature <= 0.0:
        raise ValueError("target_temperature must be positive")
    spy_arr = spy_returns.to_numpy(dtype=float)
    n = len(spy_arr)
    fwd = np.zeros(n, dtype=float)
    for t in range(n - horizon):
        fwd[t] = float(np.prod(1.0 + spy_arr[t + 1 : t + 1 + horizon]) - 1.0)
    valid_fwd = fwd[: n - horizon]
    target_std = max(float(np.std(valid_fwd, ddof=0)) * target_temperature, 1e-8)
    return np.tanh(fwd / target_std)


# ---------------------------------------------------------------------------
# Training  (FIX #1 — no hard caps; FIX #5 — wider scale search)
# ---------------------------------------------------------------------------


def _train_main6_model(
    features: pd.DataFrame,
    spy_returns: pd.Series,
    train_end_index: int,
    validation_start_index: int,
    validation_end_index: int,
    cfg: MediumCapacityConfig,
    horizon: int = 21,
    target_temperature: float = 1.0,
    n_scale_candidates: int = 51,
) -> tuple[MediumCapacityParams, dict[str, Any]]:
    """Train main6 ridge model.

    Key difference from main5: hard activity cap filters are still present in
    ``cfg`` (for opt-in backward comparison) but are never set by the default
    Main6 candidate configs.  The scale search range is also wider.

    Returns
    -------
    params:      Learned MediumCapacityParams for simulate_medium_capacity_policy.
    diagnostics: Training and validation metrics dict.
    """
    n_valid = train_end_index + 1 - horizon
    if n_valid < 50:
        raise ValueError(
            f"Only {n_valid} valid training rows after excluding last {horizon} "
            f"rows for forward-target horizon. Need at least 50."
        )

    train_features_raw = features.iloc[:n_valid].to_numpy(dtype=float)

    full_target = _build_main6_target(
        spy_returns,
        horizon=horizon,
        target_temperature=target_temperature,
    )
    train_target = full_target[:n_valid]

    mu = np.mean(train_features_raw, axis=0)
    std = np.std(train_features_raw, axis=0, ddof=0)
    std = np.where(std < 1e-8, 1.0, std)
    train_features_std = (train_features_raw - mu) / std

    X_train = np.c_[np.ones(n_valid), train_features_std]
    XtX = X_train.T @ X_train
    Xty = X_train.T @ train_target
    ridge_weights = np.linalg.solve(XtX + cfg.ridge_l2 * np.eye(XtX.shape[0]), Xty)

    val_features_raw = features.iloc[validation_start_index : validation_end_index + 1].to_numpy(dtype=float)
    val_returns = spy_returns.iloc[validation_start_index : validation_end_index + 1].to_numpy(dtype=float)
    val_features_std = (val_features_raw - mu) / std
    X_val = np.c_[np.ones(len(val_features_std)), val_features_std]
    val_signal = np.clip(X_val @ ridge_weights, -1.0, 1.0)

    best_scale = 0.0
    best_obj = float("-inf")
    best_val_result: dict[str, Any] | None = None
    best_tail_metric = float("nan")

    scale_candidates = np.linspace(cfg.min_signal_scale, cfg.max_signal_scale, n_scale_candidates)
    for scale_cand in scale_candidates:
        val_result = _simulate_signal_path(
            signal_clipped=val_signal,
            window_returns=val_returns,
            cfg=cfg,
            signal_scale=float(scale_cand),
        )
        relative_return = float(val_result["relative_return"])
        if (
            cfg.validation_hard_min_relative_return is not None
            and relative_return < cfg.validation_hard_min_relative_return
        ):
            continue
        obj = relative_return - cfg.scale_turnover_penalty * float(val_result["average_turnover"])
        val_step_den = max(len(val_returns) - 1, 1)
        val_step_rate = float(val_result["executed_step_count"]) / float(val_step_den)
        val_suppression_rate = float(val_result["gate_suppression_rate"])
        # Hard caps: only active when explicitly passed (not set by default configs)
        if (
            cfg.validation_hard_min_step_rate is not None
            and val_step_rate < cfg.validation_hard_min_step_rate
        ):
            continue
        if (
            cfg.validation_hard_max_suppression_rate is not None
            and val_suppression_rate > cfg.validation_hard_max_suppression_rate
        ):
            continue
        if cfg.validation_step_rate_target is not None and cfg.validation_step_rate_penalty > 0.0:
            obj -= cfg.validation_step_rate_penalty * abs(
                val_step_rate - cfg.validation_step_rate_target
            )
        if (
            cfg.validation_suppression_rate_target is not None
            and cfg.validation_suppression_rate_penalty > 0.0
        ):
            obj -= cfg.validation_suppression_rate_penalty * abs(
                val_suppression_rate - cfg.validation_suppression_rate_target
            )
        if cfg.validation_relative_floor_target is not None and cfg.validation_relative_floor_penalty > 0.0:
            obj -= cfg.validation_relative_floor_penalty * max(
                0.0, cfg.validation_relative_floor_target - relative_return
            )
        if cfg.validation_max_relative_drawdown_penalty > 0.0:
            obj -= cfg.validation_max_relative_drawdown_penalty * float(
                val_result["max_relative_drawdown"]
            )
        if cfg.validation_tail_bootstrap_reps > 0:
            tail_metric = _compute_bootstrap_total_return_quantile(
                relative_daily_returns=list(val_result["relative_daily_returns"]),
                reps=cfg.validation_tail_bootstrap_reps,
                block_size=cfg.validation_tail_bootstrap_block_size,
                quantile=cfg.validation_tail_bootstrap_quantile,
                seed=cfg.validation_tail_bootstrap_seed,
            )
            if (
                cfg.validation_tail_bootstrap_floor_target is not None
                and cfg.validation_tail_bootstrap_penalty > 0.0
            ):
                obj -= cfg.validation_tail_bootstrap_penalty * max(
                    0.0,
                    cfg.validation_tail_bootstrap_floor_target - tail_metric,
                )
            if (
                cfg.validation_tail_bootstrap_hard_min is not None
                and tail_metric < cfg.validation_tail_bootstrap_hard_min
            ):
                continue
            if cfg.validation_tail_bootstrap_objective_weight != 0.0:
                obj += cfg.validation_tail_bootstrap_objective_weight * tail_metric
        else:
            tail_metric = float("nan")
        if obj > best_obj:
            best_obj = obj
            best_scale = float(scale_cand)
            best_val_result = val_result
            best_tail_metric = float(tail_metric)

    if best_val_result is None:
        raise RuntimeError("Validation scale search produced no result.")

    params = MediumCapacityParams(
        signal_weights=ridge_weights,
        signal_scale=best_scale,
        feature_mu=mu,
        feature_std=std,
    )
    diagnostics = {
        "n_train_samples": n_valid,
        "n_val_samples": validation_end_index - validation_start_index + 1,
        "signal_bias": float(ridge_weights[0]),
        "signal_scale": float(best_scale),
        "signal_scale_state_slope": 0.0,
        "val_cumulative_return": float(best_val_result["relative_return"]),
        "val_objective": float(best_obj),
        "signal_weights_l2": float(np.linalg.norm(ridge_weights[1:])),
        "signal_model_family": "single_linear",
        "validation_signal_abs_p95": float(best_val_result["signal_abs_p95"]),
        "validation_signal_abs_max": float(best_val_result["signal_abs_max"]),
        "validation_proposed_step_p95": float(best_val_result["proposed_step_p95"]),
        "validation_proposed_step_max": float(best_val_result["proposed_step_max"]),
        "validation_proposed_steps_over_band": int(best_val_result["proposed_steps_over_band"]),
        "validation_executed_step_count": int(best_val_result["executed_step_count"]),
        "validation_gate_suppressed_step_count": int(best_val_result["gate_suppressed_step_count"]),
        "validation_gate_suppression_rate": float(best_val_result["gate_suppression_rate"]),
        "validation_tail_bootstrap_metric": float(best_tail_metric),
    }
    return params, diagnostics


# ---------------------------------------------------------------------------
# Orchestrator config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Main6Config:
    """Configuration for main6 orchestrator."""

    min_validation_threshold: float = 0.0005
    min_robust_min_relative: float = -0.01
    min_active_fraction: float = 0.01
    # FIX #4: gate rejects runs with too many dead folds
    max_dead_fold_fraction: float = 0.20
    n_folds: int = 5
    validation_fraction: float = 0.15
    test_fraction: float = 0.10
    forward_horizon: int = 21
    target_temperature: float = 1.0
    n_scale_candidates: int = 51
    medium_capacity_cfg: MediumCapacityConfig = MediumCapacityConfig()


def _evaluate_gate_checks_main6(
    cfg: Main6Config,
    mean_validation_relative: float,
    robust_min_test_relative: float,
    active_fraction: float,
    dead_fold_fraction: float,
) -> dict[str, Any]:
    """Gate checks with added fold-health dimension."""
    validation_ok = mean_validation_relative >= cfg.min_validation_threshold
    robust_min_ok = robust_min_test_relative >= cfg.min_robust_min_relative
    active_fraction_ok = active_fraction >= cfg.min_active_fraction
    fold_health_ok = dead_fold_fraction <= cfg.max_dead_fold_fraction
    return {
        "validation_threshold": {
            "passed": bool(validation_ok),
            "value": float(mean_validation_relative),
            "threshold": float(cfg.min_validation_threshold),
        },
        "robust_min_threshold": {
            "passed": bool(robust_min_ok),
            "value": float(robust_min_test_relative),
            "threshold": float(cfg.min_robust_min_relative),
        },
        "active_fraction_threshold": {
            "passed": bool(active_fraction_ok),
            "value": float(active_fraction),
            "threshold": float(cfg.min_active_fraction),
        },
        "fold_health_threshold": {
            "passed": bool(fold_health_ok),
            "value": float(dead_fold_fraction),
            "threshold": float(cfg.max_dead_fold_fraction),
            "description": "fraction of folds where scale search produced no result",
        },
        "overall_passed": bool(
            validation_ok and robust_min_ok and active_fraction_ok and fold_health_ok
        ),
    }


def _resolve_validation_threshold(
    gate: str,
    min_validation_threshold: float | None,
    gate_scale: str,
) -> tuple[float, str]:
    if min_validation_threshold is not None:
        return float(min_validation_threshold), "explicit"
    gate_value = float(gate)
    if gate_scale == "bps":
        return gate_value / 10_000.0, "gate_bps"
    if gate_scale == "legacy":
        return gate_value / 1_000.0, "gate_legacy"
    raise ValueError(f"Unsupported gate_scale: {gate_scale}")


def _compute_fingerprint() -> dict[str, str]:
    fingerprint: dict[str, str] = {}
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(Path(__file__).resolve().parents[2]),
            )
            .decode()
            .strip()
        )
        fingerprint["git_commit"] = commit
    except Exception:
        fingerprint["git_commit"] = "unknown"
    for lib in ("numpy", "pandas"):
        try:
            import importlib

            mod = importlib.import_module(lib)
            fingerprint[f"{lib}_version"] = mod.__version__
        except Exception:
            fingerprint[f"{lib}_version"] = "unknown"
    fingerprint["python_version"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    fingerprint["python_version_prefix"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}"
    )
    for label, rel_path in (("main6", "main6.py"), ("medium_capacity", "medium_capacity.py")):
        try:
            target = Path(__file__).parent / rel_path
            with open(target, "rb") as fh:
                fingerprint[f"{label}_sha256"] = hashlib.sha256(fh.read()).hexdigest()
        except Exception:
            fingerprint[f"{label}_sha256"] = "unknown"
    return fingerprint


def _block_bootstrap_relative_return(
    relative_daily_returns: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> float:
    n = int(len(relative_daily_returns))
    if n == 0:
        return 0.0
    if block_size <= 1:
        idx = rng.integers(0, n, size=n)
        sampled = relative_daily_returns[idx]
    else:
        starts = rng.integers(0, n, size=max(1, int(np.ceil(n / block_size))))
        chunks = []
        for start in starts:
            end = min(start + block_size, n)
            chunk = relative_daily_returns[start:end]
            if len(chunk) < block_size:
                remainder = block_size - len(chunk)
                chunk = np.concatenate([chunk, relative_daily_returns[:remainder]])
            chunks.append(chunk)
        sampled = np.concatenate(chunks)[:n]
    return float(np.prod(1.0 + sampled) - 1.0)


def _compute_path_bootstrap_metrics(
    fold_relative_daily_returns: list[list[float]],
    reps: int,
    block_size: int,
    seed: int,
) -> dict[str, float]:
    if reps <= 0 or not fold_relative_daily_returns:
        return {
            "path_bootstrap_mean_test_relative_p05": float("nan"),
            "path_bootstrap_mean_test_relative_p50": float("nan"),
            "path_bootstrap_mean_test_relative_p95": float("nan"),
            "path_bootstrap_robust_min_test_relative_p05": float("nan"),
            "path_bootstrap_robust_min_test_relative_p50": float("nan"),
            "path_bootstrap_robust_min_test_relative_p95": float("nan"),
        }
    rng = np.random.default_rng(seed)
    fold_arrays = [np.asarray(v, dtype=float) for v in fold_relative_daily_returns]
    mean_samples = np.empty(reps, dtype=float)
    robust_min_samples = np.empty(reps, dtype=float)
    for rep in range(reps):
        fold_returns = np.array(
            [
                _block_bootstrap_relative_return(arr, block_size=block_size, rng=rng)
                for arr in fold_arrays
            ],
            dtype=float,
        )
        mean_samples[rep] = float(np.mean(fold_returns))
        robust_min_samples[rep] = float(np.min(fold_returns))
    return {
        "path_bootstrap_mean_test_relative_p05": float(np.quantile(mean_samples, 0.05)),
        "path_bootstrap_mean_test_relative_p50": float(np.quantile(mean_samples, 0.50)),
        "path_bootstrap_mean_test_relative_p95": float(np.quantile(mean_samples, 0.95)),
        "path_bootstrap_robust_min_test_relative_p05": float(np.quantile(robust_min_samples, 0.05)),
        "path_bootstrap_robust_min_test_relative_p50": float(np.quantile(robust_min_samples, 0.50)),
        "path_bootstrap_robust_min_test_relative_p95": float(np.quantile(robust_min_samples, 0.95)),
    }


# ---------------------------------------------------------------------------
# Walk-forward training  (FIX #4 — explicit fold-death tracking)
# ---------------------------------------------------------------------------


def _train_policy(
    spy_returns: pd.Series,
    cfg: Main6Config,
    seed: int = 7,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train main6 model across walk-forward folds.

    Dead folds (RuntimeError from scale search) are recorded explicitly with
    fold_status="dead" rather than silently dropped from the summary DataFrame.
    This prevents the silent compression of failures that masked main5's
    cross-period blocker.
    """
    results: list[dict[str, Any]] = []
    fold_relative_daily_returns: list[list[float]] = []
    features_df = build_main6_features(spy_returns)

    splits = generate_walk_forward_splits(
        spy_returns.to_frame(name="SPY"),
        benchmark_returns=spy_returns,
        lookback=20,
        validation_fraction=cfg.validation_fraction,
        test_fraction=cfg.test_fraction,
        max_splits=cfg.n_folds,
    )

    for fold_idx, split in enumerate(splits, start=1):
        train_end = split.train.end_index - 1
        val_start = split.validation.start_index
        val_end = split.validation.end_index - 1
        test_start = split.test.start_index
        test_end = split.test.end_index - 1

        # Shared index fields — present for both live and dead folds
        fold_base: dict[str, Any] = {
            "split": "chrono",
            "seed": int(seed),
            "fold": f"fold_{fold_idx:02d}",
            "phase": "test",
            "train_end_idx": train_end,
            "val_start_idx": val_start,
            "val_end_idx": val_end,
            "test_start_idx": test_start,
            "test_end_idx": test_end,
            "n_test_samples": test_end - test_start + 1,
        }

        try:
            params, diag = _train_main6_model(
                features=features_df,
                spy_returns=spy_returns,
                train_end_index=train_end,
                validation_start_index=val_start,
                validation_end_index=val_end,
                cfg=cfg.medium_capacity_cfg,
                horizon=cfg.forward_horizon,
                target_temperature=cfg.target_temperature,
                n_scale_candidates=cfg.n_scale_candidates,
            )
            (
                test_total_return,
                test_relative_return,
                test_weight,
                test_turnover,
                test_diag,
            ) = simulate_medium_capacity_policy(
                features=features_df,
                spy_returns=spy_returns,
                start_index=test_start,
                end_index=test_end,
                cfg=cfg.medium_capacity_cfg,
                params=params,
            )
            bh_return = float(
                np.prod(
                    1.0 + spy_returns.iloc[test_start : test_end + 1].to_numpy(dtype=float)
                )
                - 1.0
            )
            active = int(test_turnover > 1e-6)
            fold_relative_daily_returns.append(
                list(test_diag.get("relative_daily_returns", []))
            )
            results.append(
                {
                    **fold_base,
                    "fold_status": "ok",
                    "fold_error": None,
                    "validation_relative_total_return": diag["val_cumulative_return"],
                    "policy_total_return": test_total_return,
                    "policy_relative_total_return": test_relative_return,
                    "static_hold_total_return": bh_return,
                    "delta_total_return_vs_static_hold": test_total_return - bh_return,
                    "mean_spy_weight": test_weight,
                    "mean_turnover": test_turnover,
                    "test_worst_daily_relative_return": test_diag["worst_daily_relative_return"],
                    "test_max_relative_drawdown": test_diag["max_relative_drawdown"],
                    "active": active,
                    "signal_scale": diag["signal_scale"],
                    "signal_scale_state_slope": diag.get("signal_scale_state_slope", 0.0),
                    "signal_bias": diag["signal_bias"],
                    "validation_signal_abs_p95": diag["validation_signal_abs_p95"],
                    "validation_proposed_step_p95": diag["validation_proposed_step_p95"],
                    "validation_proposed_steps_over_band": diag["validation_proposed_steps_over_band"],
                    "validation_gate_suppressed_step_count": diag["validation_gate_suppressed_step_count"],
                    "validation_gate_suppression_rate": diag["validation_gate_suppression_rate"],
                    "test_signal_abs_p95": test_diag["signal_abs_p95"],
                    "test_proposed_step_p95": test_diag["proposed_step_p95"],
                    "test_proposed_steps_over_band": test_diag["proposed_steps_over_band"],
                    "test_executed_step_count": test_diag["executed_step_count"],
                    "test_gate_suppressed_step_count": test_diag["gate_suppressed_step_count"],
                    "test_gate_suppression_rate": test_diag["gate_suppression_rate"],
                }
            )
        except Exception as exc:
            error_msg = str(exc)
            print(
                f"[main6] DEAD fold {fold_idx}: {error_msg}",
                file=sys.stderr,
            )
            # Record dead fold explicitly — NaN for all float metrics
            results.append(
                {
                    **fold_base,
                    "fold_status": "dead",
                    "fold_error": error_msg,
                    "validation_relative_total_return": float("nan"),
                    "policy_total_return": float("nan"),
                    "policy_relative_total_return": float("nan"),
                    "static_hold_total_return": float("nan"),
                    "delta_total_return_vs_static_hold": float("nan"),
                    "mean_spy_weight": float("nan"),
                    "mean_turnover": float("nan"),
                    "test_worst_daily_relative_return": float("nan"),
                    "test_max_relative_drawdown": float("nan"),
                    "active": 0,
                    "signal_scale": float("nan"),
                    "signal_scale_state_slope": float("nan"),
                    "signal_bias": float("nan"),
                    "validation_signal_abs_p95": float("nan"),
                    "validation_proposed_step_p95": float("nan"),
                    "validation_proposed_steps_over_band": 0,
                    "validation_gate_suppressed_step_count": 0,
                    "validation_gate_suppression_rate": float("nan"),
                    "test_signal_abs_p95": float("nan"),
                    "test_proposed_step_p95": float("nan"),
                    "test_proposed_steps_over_band": 0,
                    "test_executed_step_count": 0,
                    "test_gate_suppressed_step_count": 0,
                    "test_gate_suppression_rate": float("nan"),
                }
            )

    summary_df = pd.DataFrame(results)
    metadata: dict[str, Any] = {
        "n_folds": cfg.n_folds,
        "forward_horizon": cfg.forward_horizon,
        "config": asdict(cfg.medium_capacity_cfg),
        "test_fold_relative_daily_returns": fold_relative_daily_returns,
    }
    return summary_df, metadata


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(
    returns_csv: str = "data/demo_sleeves.csv",
    benchmark_column: str = "SPY_BENCHMARK",
    date_column: str = "date",
    gate: str = "001",
    n_folds: int = 5,
    seed: int = 7,
    output_dir: str = "artifacts",
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
    ridge_l2: float = 1.0,
    # FIX #5: wider scale range default
    scale_turnover_penalty: float = 2.9952,
    min_signal_scale: float = -0.30,
    max_signal_scale: float = 0.30,
    n_scale_candidates: int = 51,
    min_spy_weight: float = 0.80,
    max_spy_weight: float = 1.05,
    initial_spy_weight: float = 1.0,
    action_smoothing: float = 0.9231,
    # FIX #2: reduced deadband — 0.005 is the cross-period validated default
    no_trade_band: float = 0.005,
    execution_gate_mode: str = "hard",
    smooth_gate_width_ratio: float = 0.25,
    smooth_gate_floor: float = 0.0,
    execution_gate_tolerance: float = 0.0,
    validation_relative_floor_target: float | None = 0.00015,
    validation_relative_floor_penalty: float = 3.0,
    validation_max_relative_drawdown_penalty: float = 0.0,
    validation_step_rate_target: float | None = None,
    validation_step_rate_penalty: float = 0.0,
    validation_suppression_rate_target: float | None = None,
    validation_suppression_rate_penalty: float = 0.0,
    # FIX #1: hard caps default to None (opt-in only)
    validation_hard_min_step_rate: float | None = None,
    validation_hard_max_suppression_rate: float | None = None,
    validation_hard_min_relative_return: float | None = None,
    validation_tail_bootstrap_reps: int = 320,
    validation_tail_bootstrap_block_size: int = 12,
    validation_tail_bootstrap_quantile: float = 0.05,
    validation_tail_bootstrap_floor_target: float | None = -0.003,
    validation_tail_bootstrap_penalty: float = 2.0,
    validation_tail_bootstrap_hard_min: float | None = -0.007,
    validation_tail_bootstrap_objective_weight: float = 0.5,
    validation_tail_bootstrap_seed: int = 8097,
    min_validation_threshold: float | None = None,
    min_robust_min_relative: float = -0.01,
    min_active_fraction: float = 0.01,
    # FIX #4: fold health gate
    max_dead_fold_fraction: float = 0.20,
    gate_scale: str = "bps",
    fail_on_gate: bool = True,
    forward_horizon: int = 21,
    target_temperature: float = 1.0,
    path_bootstrap_reps: int = 80,
    path_bootstrap_block_size: int = 12,
    path_bootstrap_seed: int = 8097,
) -> int:
    """Run main6 orchestrator."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {returns_csv}...", file=sys.stderr)
    frame = load_returns_csv(returns_csv, date_column=date_column)

    if benchmark_column not in frame.columns:
        print(
            f"Error: Column '{benchmark_column}' not found. Available: {list(frame.columns)}",
            file=sys.stderr,
        )
        return 1

    spy_returns = frame[benchmark_column].astype(float)
    print(f"Loaded {len(spy_returns)} days", file=sys.stderr)

    medium_cfg = MediumCapacityConfig(
        min_spy_weight=min_spy_weight,
        max_spy_weight=max_spy_weight,
        initial_spy_weight=initial_spy_weight,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        ridge_l2=ridge_l2,
        scale_turnover_penalty=scale_turnover_penalty,
        min_signal_scale=min_signal_scale,
        max_signal_scale=max_signal_scale,
        target_mode="tanh_return",
        feature_family="baseline",
        signal_model_family="single_linear",
        execution_gate_mode=execution_gate_mode,
        smooth_gate_width_ratio=smooth_gate_width_ratio,
        smooth_gate_floor=smooth_gate_floor,
        execution_gate_tolerance=execution_gate_tolerance,
        validation_relative_floor_target=validation_relative_floor_target,
        validation_relative_floor_penalty=validation_relative_floor_penalty,
        validation_max_relative_drawdown_penalty=validation_max_relative_drawdown_penalty,
        validation_step_rate_target=validation_step_rate_target,
        validation_step_rate_penalty=validation_step_rate_penalty,
        validation_suppression_rate_target=validation_suppression_rate_target,
        validation_suppression_rate_penalty=validation_suppression_rate_penalty,
        validation_hard_min_step_rate=validation_hard_min_step_rate,
        validation_hard_max_suppression_rate=validation_hard_max_suppression_rate,
        validation_hard_min_relative_return=validation_hard_min_relative_return,
        validation_tail_bootstrap_reps=validation_tail_bootstrap_reps,
        validation_tail_bootstrap_block_size=validation_tail_bootstrap_block_size,
        validation_tail_bootstrap_quantile=validation_tail_bootstrap_quantile,
        validation_tail_bootstrap_floor_target=validation_tail_bootstrap_floor_target,
        validation_tail_bootstrap_penalty=validation_tail_bootstrap_penalty,
        validation_tail_bootstrap_hard_min=validation_tail_bootstrap_hard_min,
        validation_tail_bootstrap_objective_weight=validation_tail_bootstrap_objective_weight,
        validation_tail_bootstrap_seed=validation_tail_bootstrap_seed,
        action_smoothing=action_smoothing,
        no_trade_band=no_trade_band,
    )

    resolved_validation_threshold, threshold_source = _resolve_validation_threshold(
        gate=gate,
        min_validation_threshold=min_validation_threshold,
        gate_scale=gate_scale,
    )

    cfg = Main6Config(
        min_validation_threshold=resolved_validation_threshold,
        min_robust_min_relative=min_robust_min_relative,
        min_active_fraction=min_active_fraction,
        max_dead_fold_fraction=max_dead_fold_fraction,
        n_folds=n_folds,
        forward_horizon=forward_horizon,
        target_temperature=target_temperature,
        n_scale_candidates=n_scale_candidates,
        medium_capacity_cfg=medium_cfg,
    )

    print(
        f"Validation threshold: {resolved_validation_threshold:.6f} "
        f"(source={threshold_source}, gate={gate}, gate_scale={gate_scale})",
        file=sys.stderr,
    )
    print(
        f"Training main6 policy: {n_folds} folds, forward_horizon={forward_horizon}d, "
        f"target_temperature={target_temperature:.3f}, n_scale_candidates={n_scale_candidates}, "
        f"no_trade_band={no_trade_band:.4f}, scale_range=[{min_signal_scale:.3f},{max_signal_scale:.3f}]",
        file=sys.stderr,
    )

    summary_df, metadata = _train_policy(spy_returns, cfg, seed)

    # Aggregate metrics — live folds only for financial metrics
    live_df = summary_df[summary_df["fold_status"] == "ok"] if len(summary_df) > 0 else summary_df
    n_dead_folds = int((summary_df["fold_status"] == "dead").sum()) if len(summary_df) > 0 else 0
    n_live_folds = int(len(live_df))
    dead_fold_fraction = float(n_dead_folds) / max(float(len(summary_df)), 1.0)

    if n_live_folds > 0:
        mean_test_relative = float(live_df["policy_relative_total_return"].mean())
        mean_validation = float(live_df["validation_relative_total_return"].mean())
        beat_hold = float((live_df["policy_relative_total_return"] > 0).sum() / n_live_folds)
        active_fraction = float(live_df["active"].mean())
        mean_turnover = float(live_df["mean_turnover"].mean())
        robust_min = float(live_df["policy_relative_total_return"].min())
        worst_daily_relative = float(live_df["test_worst_daily_relative_return"].min())
        worst_max_relative_drawdown = float(live_df["test_max_relative_drawdown"].max())
        mean_validation_gate_suppression = float(live_df["validation_gate_suppression_rate"].mean())
        mean_test_gate_suppression = float(live_df["test_gate_suppression_rate"].mean())
        mean_test_executed_steps = float(live_df["test_executed_step_count"].mean())
    else:
        mean_test_relative = float("nan")
        mean_validation = float("nan")
        beat_hold = float("nan")
        active_fraction = 0.0
        mean_turnover = 0.0
        robust_min = float("nan")
        worst_daily_relative = float("nan")
        worst_max_relative_drawdown = float("nan")
        mean_validation_gate_suppression = float("nan")
        mean_test_gate_suppression = float("nan")
        mean_test_executed_steps = 0.0

    gate_checks = _evaluate_gate_checks_main6(
        cfg=cfg,
        mean_validation_relative=mean_validation if not np.isnan(mean_validation) else 0.0,
        robust_min_test_relative=robust_min if not np.isnan(robust_min) else 0.0,
        active_fraction=active_fraction,
        dead_fold_fraction=dead_fold_fraction,
    )
    path_bootstrap_metrics = _compute_path_bootstrap_metrics(
        fold_relative_daily_returns=list(metadata.get("test_fold_relative_daily_returns", [])),
        reps=int(path_bootstrap_reps),
        block_size=int(path_bootstrap_block_size),
        seed=int(path_bootstrap_seed),
    )

    fingerprint = _compute_fingerprint()

    # Write summary CSV (includes dead folds — fold_status column distinguishes them)
    output_csv = os.path.join(output_dir, f"main6_gate{gate}_f{n_folds}_s{seed}_summary.csv")
    summary_df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}", file=sys.stderr)

    def _safe_float(v: float) -> float | None:
        if np.isnan(v) or np.isinf(v):
            return None
        return float(v)

    detailed_output: dict[str, Any] = {
        "fingerprint": fingerprint,
        "config": asdict(cfg),
        "summary_metrics": {
            "n_rows": len(summary_df),
            "n_live_folds": n_live_folds,
            "n_dead_folds": n_dead_folds,
            "dead_fold_fraction": dead_fold_fraction,
            "mean_test_relative_total_return": _safe_float(mean_test_relative),
            "mean_validation_relative_total_return": _safe_float(mean_validation),
            "beat_hold_fraction": _safe_float(beat_hold),
            "active_fraction": active_fraction,
            "mean_turnover": mean_turnover,
            "robust_min_test_relative": _safe_float(robust_min),
            "worst_daily_relative_return": _safe_float(worst_daily_relative),
            "worst_max_relative_drawdown": _safe_float(worst_max_relative_drawdown),
            "mean_validation_gate_suppression_rate": _safe_float(mean_validation_gate_suppression),
            "mean_test_gate_suppression_rate": _safe_float(mean_test_gate_suppression),
            "mean_test_executed_step_count": mean_test_executed_steps,
            **path_bootstrap_metrics,
        },
        "fold_diagnostics": json.loads(summary_df.to_json(orient="records")),
        "path_bootstrap": {
            "reps": int(path_bootstrap_reps),
            "block_size": int(path_bootstrap_block_size),
            "seed": int(path_bootstrap_seed),
        },
        "gate_checks": gate_checks,
    }

    output_json = os.path.join(output_dir, f"main6_gate{gate}_f{n_folds}_s{seed}_detailed.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(detailed_output, f, indent=2)
    print(f"Wrote {output_json}", file=sys.stderr)

    print("\n=== Main6 Summary ===", file=sys.stderr)
    print(
        f"Gate: {gate}  Folds: {n_folds}  Seed: {seed}  Horizon: {forward_horizon}d  "
        f"TargetTemp: {target_temperature:.3f}",
        file=sys.stderr,
    )
    print(
        f"Live folds: {n_live_folds}/{n_folds}  Dead folds: {n_dead_folds}  "
        f"DeadFraction: {dead_fold_fraction:.2f}",
        file=sys.stderr,
    )
    if n_live_folds > 0:
        print(
            f"MeanRelRet:  {mean_test_relative:+.4f}  "
            f"RobustMin: {robust_min:+.4f}  "
            f"BeatHold: {beat_hold:.2%}",
            file=sys.stderr,
        )
        print(
            f"MeanVal:     {mean_validation:+.4f}  "
            f"ActiveFrac: {active_fraction:.2%}  "
            f"MeanTurnover: {mean_turnover:.4f}",
            file=sys.stderr,
        )
        print(
            f"TestSuppRate:{mean_test_gate_suppression:.4f}  "
            f"AvgExecSteps: {mean_test_executed_steps:.1f}",
            file=sys.stderr,
        )
    gate_passed = gate_checks["overall_passed"]
    print(
        f"Gate checks: {'PASSED' if gate_passed else 'FAILED'}",
        file=sys.stderr,
    )
    for check_name, check_data in gate_checks.items():
        if check_name == "overall_passed":
            continue
        status = "OK  " if check_data["passed"] else "FAIL"
        print(
            f"  [{status}] {check_name}: {check_data['value']:.6f} "
            f"(threshold={check_data['threshold']:.6f})",
            file=sys.stderr,
        )

    if fail_on_gate and not gate_passed:
        print("Gate failed — returning exit code 1.", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main6: multi-scale ridge policy with explicit fold-health accounting."
    )
    parser.add_argument("--returns-csv", type=str, default="data/demo_sleeves.csv")
    parser.add_argument("--benchmark-column", type=str, default="SPY_BENCHMARK")
    parser.add_argument("--date-column", type=str, default="date")
    parser.add_argument("--gate", type=str, default="001")
    parser.add_argument("--gate-scale", choices=["bps", "legacy"], default="bps")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument("--scale-turnover-penalty", type=float, default=2.9952)
    parser.add_argument("--min-signal-scale", type=float, default=-0.30)
    parser.add_argument("--max-signal-scale", type=float, default=0.30)
    parser.add_argument("--n-scale-candidates", type=int, default=51)
    parser.add_argument("--min-spy-weight", type=float, default=0.80)
    parser.add_argument("--max-spy-weight", type=float, default=1.05)
    parser.add_argument("--initial-spy-weight", type=float, default=1.0)
    parser.add_argument("--action-smoothing", type=float, default=0.9231)
    parser.add_argument("--no-trade-band", type=float, default=0.005)
    parser.add_argument(
        "--execution-gate-mode", choices=["hard", "smooth"], default="hard"
    )
    parser.add_argument("--smooth-gate-width-ratio", type=float, default=0.25)
    parser.add_argument("--smooth-gate-floor", type=float, default=0.0)
    parser.add_argument("--execution-gate-tolerance", type=float, default=0.0)
    parser.add_argument("--validation-relative-floor-target", type=float, default=0.00015)
    parser.add_argument("--validation-relative-floor-penalty", type=float, default=3.0)
    parser.add_argument("--validation-max-relative-drawdown-penalty", type=float, default=0.0)
    parser.add_argument("--validation-step-rate-target", type=float, default=None)
    parser.add_argument("--validation-step-rate-penalty", type=float, default=0.0)
    parser.add_argument("--validation-suppression-rate-target", type=float, default=None)
    parser.add_argument("--validation-suppression-rate-penalty", type=float, default=0.0)
    parser.add_argument("--validation-hard-min-step-rate", type=float, default=None)
    parser.add_argument("--validation-hard-max-suppression-rate", type=float, default=None)
    parser.add_argument("--validation-hard-min-relative-return", type=float, default=None)
    parser.add_argument("--validation-tail-bootstrap-reps", type=int, default=320)
    parser.add_argument("--validation-tail-bootstrap-block-size", type=int, default=12)
    parser.add_argument("--validation-tail-bootstrap-quantile", type=float, default=0.05)
    parser.add_argument("--validation-tail-bootstrap-floor-target", type=float, default=-0.003)
    parser.add_argument("--validation-tail-bootstrap-penalty", type=float, default=2.0)
    parser.add_argument("--validation-tail-bootstrap-hard-min", type=float, default=-0.007)
    parser.add_argument("--validation-tail-bootstrap-objective-weight", type=float, default=0.5)
    parser.add_argument("--validation-tail-bootstrap-seed", type=int, default=8097)
    parser.add_argument("--min-validation-threshold", type=float, default=None)
    parser.add_argument("--min-robust-min-relative", type=float, default=-0.01)
    parser.add_argument("--min-active-fraction", type=float, default=0.01)
    parser.add_argument("--max-dead-fold-fraction", type=float, default=0.20)
    parser.add_argument("--no-fail-on-gate", action="store_true")
    parser.add_argument("--forward-horizon", type=int, default=21)
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument("--path-bootstrap-reps", type=int, default=80)
    parser.add_argument("--path-bootstrap-block-size", type=int, default=12)
    parser.add_argument("--path-bootstrap-seed", type=int, default=8097)

    args = parser.parse_args()
    sys.exit(
        main(
            returns_csv=args.returns_csv,
            benchmark_column=args.benchmark_column,
            date_column=args.date_column,
            gate=args.gate,
            n_folds=args.n_folds,
            seed=args.seed,
            output_dir=args.output_dir,
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            ridge_l2=args.ridge_l2,
            scale_turnover_penalty=args.scale_turnover_penalty,
            min_signal_scale=args.min_signal_scale,
            max_signal_scale=args.max_signal_scale,
            n_scale_candidates=args.n_scale_candidates,
            min_spy_weight=args.min_spy_weight,
            max_spy_weight=args.max_spy_weight,
            initial_spy_weight=args.initial_spy_weight,
            action_smoothing=args.action_smoothing,
            no_trade_band=args.no_trade_band,
            execution_gate_mode=args.execution_gate_mode,
            smooth_gate_width_ratio=args.smooth_gate_width_ratio,
            smooth_gate_floor=args.smooth_gate_floor,
            execution_gate_tolerance=args.execution_gate_tolerance,
            validation_relative_floor_target=args.validation_relative_floor_target,
            validation_relative_floor_penalty=args.validation_relative_floor_penalty,
            validation_max_relative_drawdown_penalty=args.validation_max_relative_drawdown_penalty,
            validation_step_rate_target=args.validation_step_rate_target,
            validation_step_rate_penalty=args.validation_step_rate_penalty,
            validation_suppression_rate_target=args.validation_suppression_rate_target,
            validation_suppression_rate_penalty=args.validation_suppression_rate_penalty,
            validation_hard_min_step_rate=args.validation_hard_min_step_rate,
            validation_hard_max_suppression_rate=args.validation_hard_max_suppression_rate,
            validation_hard_min_relative_return=args.validation_hard_min_relative_return,
            validation_tail_bootstrap_reps=args.validation_tail_bootstrap_reps,
            validation_tail_bootstrap_block_size=args.validation_tail_bootstrap_block_size,
            validation_tail_bootstrap_quantile=args.validation_tail_bootstrap_quantile,
            validation_tail_bootstrap_floor_target=args.validation_tail_bootstrap_floor_target,
            validation_tail_bootstrap_penalty=args.validation_tail_bootstrap_penalty,
            validation_tail_bootstrap_hard_min=args.validation_tail_bootstrap_hard_min,
            validation_tail_bootstrap_objective_weight=args.validation_tail_bootstrap_objective_weight,
            validation_tail_bootstrap_seed=args.validation_tail_bootstrap_seed,
            min_validation_threshold=args.min_validation_threshold,
            min_robust_min_relative=args.min_robust_min_relative,
            min_active_fraction=args.min_active_fraction,
            max_dead_fold_fraction=args.max_dead_fold_fraction,
            gate_scale=args.gate_scale,
            fail_on_gate=not args.no_fail_on_gate,
            forward_horizon=args.forward_horizon,
            target_temperature=args.target_temperature,
            path_bootstrap_reps=args.path_bootstrap_reps,
            path_bootstrap_block_size=args.path_bootstrap_block_size,
            path_bootstrap_seed=args.path_bootstrap_seed,
        )
    )
