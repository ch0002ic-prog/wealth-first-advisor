"""Main4: Medium-capacity two-stage deviation predictor policy.

Fast test harness for the medium-capacity model. Trains across 3 folds with
feature engineering, ridge regression, and bounded weight mapping.
"""
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wealth_first.data import load_returns_csv
from wealth_first.data_splits import generate_walk_forward_splits
from wealth_first.medium_capacity import (
    MediumCapacityConfig,
    _build_simple_features,
    simulate_medium_capacity_policy,
    train_medium_capacity_model,
)


@dataclass(frozen=True)
class Main4Config:
    """Configuration for main4 orchestrator."""

    min_validation_threshold: float = 0.0005
    min_robust_min_relative: float = -0.01
    min_active_fraction: float = 0.01
    n_folds: int = 3
    validation_fraction: float = 0.15
    test_fraction: float = 0.10
    medium_capacity_cfg: MediumCapacityConfig = MediumCapacityConfig()


def _evaluate_gate_checks(
    cfg: Main4Config,
    mean_validation_relative: float,
    robust_min_test_relative: float,
    active_fraction: float,
) -> dict[str, Any]:
    """Evaluate promotion gates for the current run."""
    validation_ok = mean_validation_relative >= cfg.min_validation_threshold
    robust_min_ok = robust_min_test_relative >= cfg.min_robust_min_relative
    active_fraction_ok = active_fraction >= cfg.min_active_fraction
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
        "overall_passed": bool(validation_ok and robust_min_ok and active_fraction_ok),
    }


def _resolve_validation_threshold(
    gate: str,
    min_validation_threshold: float | None,
    gate_scale: str,
) -> tuple[float, str]:
    """Resolve the effective validation threshold and its source.

    Threshold precedence:
      1) Explicit min_validation_threshold override
      2) Gate code converted by gate_scale
    """
    if min_validation_threshold is not None:
        return float(min_validation_threshold), "explicit"

    gate_value = float(gate)
    if gate_scale == "bps":
        return gate_value / 10_000.0, "gate_bps"
    if gate_scale == "legacy":
        return gate_value / 1_000.0, "gate_legacy"

    raise ValueError(f"Unsupported gate_scale: {gate_scale}")


def _compute_fingerprint() -> dict[str, str]:
    """Return git, version, and code fingerprint."""
    fingerprint = {}

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd="/Users/ch0002techvc/Downloads/wealth-first-investing").decode().strip()
        fingerprint["git_commit"] = commit
    except Exception:
        fingerprint["git_commit"] = "unknown"

    try:
        import numpy

        fingerprint["numpy_version"] = numpy.__version__
    except Exception:
        fingerprint["numpy_version"] = "unknown"

    try:
        import pandas

        fingerprint["pandas_version"] = pandas.__version__
    except Exception:
        fingerprint["pandas_version"] = "unknown"

    fingerprint["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    fingerprint["python_version_prefix"] = f"{sys.version_info.major}.{sys.version_info.minor}"

    # Compute main4 code hash
    main4_file = Path(__file__)
    try:
        with open(main4_file, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
            fingerprint["main4_sha256"] = h
    except Exception:
        fingerprint["main4_sha256"] = "unknown"

    # Compute medium_capacity module hash
    try:
        medium_file = Path(__file__).parent / "medium_capacity.py"
        with open(medium_file, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
            fingerprint["medium_capacity_sha256"] = h
    except Exception:
        fingerprint["medium_capacity_sha256"] = "unknown"

    return fingerprint


def _train_policy(
    spy_returns: pd.Series,
    cfg: Main4Config,
    seed: int = 7,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train medium-capacity model across folds.

    Returns:
        summary_df: Results for each fold and phase
        metadata: Execution metadata
    """
    results = []
    features_df = _build_simple_features(spy_returns)
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

        try:
            params, diag = train_medium_capacity_model(
                features=features_df,
                spy_returns=spy_returns,
                train_end_index=train_end,
                validation_start_index=val_start,
                validation_end_index=val_end,
                cfg=cfg.medium_capacity_cfg,
                seed=seed,
            )
            test_total_return, test_relative_return, test_weight, test_turnover, test_diag = simulate_medium_capacity_policy(
                features=features_df,
                spy_returns=spy_returns,
                start_index=test_start,
                end_index=test_end,
                cfg=cfg.medium_capacity_cfg,
                params=params,
            )
            bh_return = float(np.prod(1.0 + spy_returns.iloc[test_start : test_end + 1].to_numpy(dtype=float)) - 1.0)
            active = int(test_turnover > 1e-6)

            results.append(
                {
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
                    "validation_relative_total_return": diag["val_cumulative_return"],
                    "policy_total_return": test_total_return,
                    "policy_relative_total_return": test_relative_return,
                    "static_hold_total_return": bh_return,
                    "delta_total_return_vs_static_hold": test_total_return - bh_return,
                    "mean_spy_weight": test_weight,
                    "mean_turnover": test_turnover,
                    "active": active,
                    "signal_scale": diag["signal_scale"],
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
        except Exception as e:
            print(f"Error in fold {fold_idx}: {e}", file=sys.stderr)

    summary_df = pd.DataFrame(results)
    metadata = {
        "n_folds": cfg.n_folds,
        "config": asdict(cfg.medium_capacity_cfg),
    }

    return summary_df, metadata


def main(
    returns_csv: str = "data/demo_sleeves.csv",
    benchmark_column: str = "SPY_BENCHMARK",
    date_column: str = "date",
    gate: str = "001",
    n_folds: int = 3,
    seed: int = 7,
    output_dir: str = "artifacts",
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
    ridge_l2: float = 1.0,
    scale_turnover_penalty: float = 0.0,
    min_signal_scale: float = -0.75,
    max_signal_scale: float = 0.75,
    target_mode: str = "sign",
    min_spy_weight: float = 0.80,
    max_spy_weight: float = 1.05,
    initial_spy_weight: float = 1.0,
    action_smoothing: float = 0.5,
    no_trade_band: float = 0.02,
    min_validation_threshold: float | None = None,
    min_robust_min_relative: float = -0.01,
    min_active_fraction: float = 0.01,
    gate_scale: str = "bps",
    fail_on_gate: bool = True,
) -> int:
    """Run main4 orchestrator."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {returns_csv}...", file=sys.stderr)
    frame = load_returns_csv(returns_csv, date_column=date_column)

    if benchmark_column not in frame.columns:
        print(f"Error: Column {benchmark_column} not found in CSV.", file=sys.stderr)
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
        target_mode=target_mode,
        action_smoothing=action_smoothing,
        no_trade_band=no_trade_band,
    )

    resolved_validation_threshold, threshold_source = _resolve_validation_threshold(
        gate=gate,
        min_validation_threshold=min_validation_threshold,
        gate_scale=gate_scale,
    )

    cfg = Main4Config(
        min_validation_threshold=resolved_validation_threshold,
        min_robust_min_relative=min_robust_min_relative,
        min_active_fraction=min_active_fraction,
        n_folds=n_folds,
        medium_capacity_cfg=medium_cfg,
    )

    print(
        f"Validation threshold: {resolved_validation_threshold:.6f} "
        f"(source={threshold_source}, gate={gate}, gate_scale={gate_scale})",
        file=sys.stderr,
    )

    print(f"Training medium-capacity policy with {n_folds} folds...", file=sys.stderr)
    summary_df, metadata = _train_policy(spy_returns, cfg, seed)

    # Compute aggregate metrics
    if len(summary_df) > 0:
        mean_test_relative = summary_df["policy_relative_total_return"].mean()
        mean_validation = summary_df["validation_relative_total_return"].mean()
        beat_hold = (summary_df["policy_relative_total_return"] > 0).sum() / len(summary_df)
        active_fraction = summary_df["active"].mean()
        mean_turnover = summary_df["mean_turnover"].mean()
        robust_min = summary_df["policy_relative_total_return"].min()
        mean_validation_gate_suppression = summary_df["validation_gate_suppression_rate"].mean()
        mean_test_gate_suppression = summary_df["test_gate_suppression_rate"].mean()
        mean_test_executed_steps = summary_df["test_executed_step_count"].mean()
    else:
        mean_test_relative = 0.0
        mean_validation = 0.0
        beat_hold = 0.0
        active_fraction = 0.0
        mean_turnover = 0.0
        robust_min = 0.0
        mean_validation_gate_suppression = 0.0
        mean_test_gate_suppression = 0.0
        mean_test_executed_steps = 0.0

    gate_checks = _evaluate_gate_checks(
        cfg=cfg,
        mean_validation_relative=float(mean_validation),
        robust_min_test_relative=float(robust_min),
        active_fraction=float(active_fraction),
    )

    # Collect fingerprint
    fingerprint = _compute_fingerprint()

    # Write summary CSV
    output_csv = os.path.join(output_dir, f"main4_gate{gate}_f{n_folds}_s{seed}_summary.csv")
    summary_df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}", file=sys.stderr)

    # Write detailed JSON with fingerprint and metrics
    detailed_output = {
        "fingerprint": fingerprint,
        "config": asdict(cfg),
        "summary_metrics": {
            "n_rows": len(summary_df),
            "mean_test_relative_total_return": float(mean_test_relative),
            "mean_validation_relative_total_return": float(mean_validation),
            "beat_hold_fraction": float(beat_hold),
            "active_fraction": float(active_fraction),
            "mean_turnover": float(mean_turnover),
            "robust_min_test_relative": float(robust_min),
            "mean_validation_gate_suppression_rate": float(mean_validation_gate_suppression),
            "mean_test_gate_suppression_rate": float(mean_test_gate_suppression),
            "mean_test_executed_step_count": float(mean_test_executed_steps),
        },
        "gate_checks": gate_checks,
    }

    output_json = os.path.join(output_dir, f"main4_gate{gate}_f{n_folds}_s{seed}_detailed.json")
    with open(output_json, "w") as f:
        json.dump(detailed_output, f, indent=2)
    print(f"Wrote {output_json}", file=sys.stderr)

    # Print summary
    print("\n=== Main4 Summary ===", file=sys.stderr)
    print(f"Gate: {gate}", file=sys.stderr)
    print(f"Folds: {n_folds}, Seed: {seed}", file=sys.stderr)
    print(f"Mean test relative return: {mean_test_relative:.6f}", file=sys.stderr)
    print(f"Mean validation relative return: {mean_validation:.6f}", file=sys.stderr)
    print(f"Beat hold: {beat_hold:.1%}", file=sys.stderr)
    print(f"Active fraction: {active_fraction:.1%}, Mean turnover: {mean_turnover:.6f}", file=sys.stderr)
    print(
        f"Validation gate suppression: {mean_validation_gate_suppression:.1%}, "
        f"Test gate suppression: {mean_test_gate_suppression:.1%}, "
        f"Mean executed steps: {mean_test_executed_steps:.2f}",
        file=sys.stderr,
    )
    print(f"Robust min: {robust_min:.6f}", file=sys.stderr)

    overall_passed = bool(gate_checks["overall_passed"])
    print(f"Gate checks passed: {overall_passed}", file=sys.stderr)
    if not overall_passed:
        print(
            "Gate detail: "
            f"validation={gate_checks['validation_threshold']['passed']} "
            f"(value={gate_checks['validation_threshold']['value']:.6f}, "
            f"threshold={gate_checks['validation_threshold']['threshold']:.6f}), "
            f"robust_min={gate_checks['robust_min_threshold']['passed']} "
            f"(value={gate_checks['robust_min_threshold']['value']:.6f}, "
            f"threshold={gate_checks['robust_min_threshold']['threshold']:.6f}), "
            f"active_fraction={gate_checks['active_fraction_threshold']['passed']} "
            f"(value={gate_checks['active_fraction_threshold']['value']:.6f}, "
            f"threshold={gate_checks['active_fraction_threshold']['threshold']:.6f})",
            file=sys.stderr,
        )
        if fail_on_gate:
            print("Run failed gate checks.", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--returns-csv", default="data/demo_sleeves.csv")
    parser.add_argument("--benchmark-column", default="SPY_BENCHMARK")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--gate", type=str, default="001")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument("--scale-turnover-penalty", type=float, default=0.0)
    parser.add_argument("--min-signal-scale", type=float, default=-0.75)
    parser.add_argument("--max-signal-scale", type=float, default=0.75)
    parser.add_argument("--target-mode", choices=["sign", "tanh_return"], default="sign")
    parser.add_argument("--min-spy-weight", type=float, default=0.80)
    parser.add_argument("--max-spy-weight", type=float, default=1.05)
    parser.add_argument("--initial-spy-weight", type=float, default=1.0)
    parser.add_argument("--action-smoothing", type=float, default=0.5)
    parser.add_argument("--no-trade-band", type=float, default=0.02)
    parser.add_argument(
        "--min-validation-threshold",
        type=float,
        default=None,
        help="Optional explicit validation threshold override for gate checks.",
    )
    parser.add_argument(
        "--min-robust-min-relative",
        type=float,
        default=-0.01,
        help="Minimum acceptable worst-fold relative test return.",
    )
    parser.add_argument(
        "--min-active-fraction",
        type=float,
        default=0.01,
        help="Minimum acceptable fraction of active folds.",
    )
    parser.add_argument(
        "--gate-scale",
        type=str,
        choices=["bps", "legacy"],
        default="bps",
        help="How to convert --gate into a validation threshold when no explicit override is provided.",
    )
    parser.add_argument(
        "--no-fail-on-gate",
        action="store_true",
        help="Do not fail process exit code when gate checks fail.",
    )
    args = parser.parse_args()

    exit_code = main(
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
        target_mode=args.target_mode,
        min_spy_weight=args.min_spy_weight,
        max_spy_weight=args.max_spy_weight,
        initial_spy_weight=args.initial_spy_weight,
        action_smoothing=args.action_smoothing,
        no_trade_band=args.no_trade_band,
        min_validation_threshold=args.min_validation_threshold,
        min_robust_min_relative=args.min_robust_min_relative,
        min_active_fraction=args.min_active_fraction,
        gate_scale=args.gate_scale,
        fail_on_gate=not args.no_fail_on_gate,
    )
    sys.exit(exit_code)
