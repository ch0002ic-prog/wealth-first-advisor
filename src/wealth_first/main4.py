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
            test_total_return, test_relative_return, test_weight, test_turnover = simulate_medium_capacity_policy(
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
    action_smoothing: float = 0.5,
    no_trade_band: float = 0.02,
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
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        ridge_l2=ridge_l2,
        action_smoothing=action_smoothing,
        no_trade_band=no_trade_band,
    )

    cfg = Main4Config(
        min_validation_threshold=float(gate) / 1000.0,
        n_folds=n_folds,
        medium_capacity_cfg=medium_cfg,
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
    else:
        mean_test_relative = 0.0
        mean_validation = 0.0
        beat_hold = 0.0
        active_fraction = 0.0
        mean_turnover = 0.0
        robust_min = 0.0

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
        },
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
    print(f"Robust min: {robust_min:.6f}", file=sys.stderr)

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
    parser.add_argument("--action-smoothing", type=float, default=0.5)
    parser.add_argument("--no-trade-band", type=float, default=0.02)
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
        action_smoothing=args.action_smoothing,
        no_trade_band=args.no_trade_band,
    )
    sys.exit(exit_code)
