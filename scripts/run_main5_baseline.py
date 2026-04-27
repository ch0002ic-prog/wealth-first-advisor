#!/usr/bin/env python3
"""run_main5_baseline.py — 16-case robustness baseline for main5.

Runs main5 (horizon-aligned, 3-factor policy) across the same
4-scenario × 2-seed × 2-cost envelope used for the Phase40 current-best
comparison, then summarises results and compares against:

  Ref: main4 v40j_single_linear_ref
        0 breaches, min_slack=+0.000183, mean_test_relative=0.005988

Usage:
    python scripts/run_main5_baseline.py [--reps 300] [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
RUN_ROOT = ARTIFACT_ROOT / "main5_baseline_runs"

FIXED_PATH_THRESHOLD = -0.007

SCENARIOS = [
    {"name": "baseline_b20_s4242", "block_size": 20, "path_seed": 4242},
    {"name": "altblock_b16_s4242", "block_size": 16, "path_seed": 4242},
    {"name": "altseed_b20_s5252", "block_size": 20, "path_seed": 5252},
    {"name": "deepcheck_b12_s8080", "block_size": 12, "path_seed": 8080},
]
SEEDS = [5, 17]
COSTS = [22, 25]

# Execution parameters identical to Phase40 current best (v40j_single_linear_ref)
BASE_PARAMS = {
    "ridge_l2": 0.020815,
    "min_signal_scale": -0.648,
    "max_signal_scale": 0.648,
    "min_spy_weight": 0.809134,
    "max_spy_weight": 1.058647,
}

# Phase40 current-best reference for comparison printout
PHASE40_REF = {
    "variant": "v40j_single_linear_ref (main4)",
    "breach_count": 0,
    "min_case_slack": 0.000183,
    "mean_test_relative": 0.005988,
    "mean_test_gate_suppression_rate": 0.9767,
}


@dataclass(frozen=True)
class Main5Variant:
    name: str
    action_smoothing: float
    no_trade_band: float
    scale_turnover_penalty: float
    validation_relative_floor_target: float | None
    validation_relative_floor_penalty: float
    validation_max_relative_drawdown_penalty: float
    validation_step_rate_target: float | None
    validation_step_rate_penalty: float
    validation_suppression_rate_target: float | None
    validation_suppression_rate_penalty: float
    forward_horizon: int = 21
    min_signal_scale: float | None = None
    max_signal_scale: float | None = None
    execution_gate_mode: str = "hard"
    smooth_gate_width_ratio: float = 0.25
    smooth_gate_floor: float = 0.0


VARIANTS: list[Main5Variant] = [
    # Promoted candidate from full-envelope search:
    # 16/16 pass at reps=80 with mean_test_relative ~0.00606.
    Main5Variant(
        name="m5_h10_ntb050_s020",
        action_smoothing=1.1425,
        no_trade_band=0.050,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.0,
        validation_relative_floor_penalty=1.25,
        validation_max_relative_drawdown_penalty=0.0,
        validation_step_rate_target=None,
        validation_step_rate_penalty=0.0,
        validation_suppression_rate_target=None,
        validation_suppression_rate_penalty=0.0,
        forward_horizon=10,
        min_signal_scale=-0.20,
        max_signal_scale=0.20,
        execution_gate_mode="hard",
    ),
    Main5Variant(
        name="m5_baseline_ref",
        action_smoothing=1.1425,
        no_trade_band=0.021429,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=None,
        validation_relative_floor_penalty=0.0,
        validation_max_relative_drawdown_penalty=0.0,
        validation_step_rate_target=None,
        validation_step_rate_penalty=0.0,
        validation_suppression_rate_target=None,
        validation_suppression_rate_penalty=0.0,
        execution_gate_mode="hard",
    ),
    # ntb sweep — keep main4 turnover penalty, widen band to reduce marginal trades
    Main5Variant(
        name="m5_ntb025",
        action_smoothing=1.1425,
        no_trade_band=0.025,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=None,
        validation_relative_floor_penalty=0.0,
        validation_max_relative_drawdown_penalty=0.0,
        validation_step_rate_target=None,
        validation_step_rate_penalty=0.0,
        validation_suppression_rate_target=None,
        validation_suppression_rate_penalty=0.0,
        execution_gate_mode="hard",
    ),
    Main5Variant(
        name="m5_ntb030",
        action_smoothing=1.1425,
        no_trade_band=0.030,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=None,
        validation_relative_floor_penalty=0.0,
        validation_max_relative_drawdown_penalty=0.0,
        validation_step_rate_target=None,
        validation_step_rate_penalty=0.0,
        validation_suppression_rate_target=None,
        validation_suppression_rate_penalty=0.0,
        execution_gate_mode="hard",
    ),
    # ntb + validation floor: force scale selection to avoid scales that lose to SPY
    Main5Variant(
        name="m5_ntb025_floor",
        action_smoothing=1.1425,
        no_trade_band=0.025,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.0,
        validation_relative_floor_penalty=1.25,
        validation_max_relative_drawdown_penalty=0.0,
        validation_step_rate_target=None,
        validation_step_rate_penalty=0.0,
        validation_suppression_rate_target=None,
        validation_suppression_rate_penalty=0.0,
        execution_gate_mode="hard",
    ),
    # higher turnover penalty with floor — aggressive cost-awareness
    Main5Variant(
        name="m5_stp5_floor",
        action_smoothing=1.1425,
        no_trade_band=0.021429,
        scale_turnover_penalty=5.0,
        validation_relative_floor_target=0.0,
        validation_relative_floor_penalty=1.25,
        validation_max_relative_drawdown_penalty=0.0,
        validation_step_rate_target=None,
        validation_step_rate_penalty=0.0,
        validation_suppression_rate_target=None,
        validation_suppression_rate_penalty=0.0,
        execution_gate_mode="hard",
    ),
    # Tighter scale + wider band + shorter horizon to reduce tail variance
    Main5Variant(
        name="m5_h10_ntb030_s04",
        action_smoothing=1.1425,
        no_trade_band=0.030,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.0,
        validation_relative_floor_penalty=1.25,
        validation_max_relative_drawdown_penalty=0.0,
        validation_step_rate_target=None,
        validation_step_rate_penalty=0.0,
        validation_suppression_rate_target=None,
        validation_suppression_rate_penalty=0.0,
        forward_horizon=10,
        min_signal_scale=-0.40,
        max_signal_scale=0.40,
        execution_gate_mode="hard",
    ),
    # Tighter scale + wider band + longer horizon for smoother signal response
    Main5Variant(
        name="m5_h42_ntb030_s04",
        action_smoothing=1.1425,
        no_trade_band=0.030,
        scale_turnover_penalty=2.9952,
        validation_relative_floor_target=0.0,
        validation_relative_floor_penalty=1.25,
        validation_max_relative_drawdown_penalty=0.0,
        validation_step_rate_target=None,
        validation_step_rate_penalty=0.0,
        validation_suppression_rate_target=None,
        validation_suppression_rate_penalty=0.0,
        forward_horizon=42,
        min_signal_scale=-0.40,
        max_signal_scale=0.40,
        execution_gate_mode="hard",
    ),
]


def run_case(
    variant: Main5Variant,
    scenario: dict[str, Any],
    seed: int,
    cost: float,
    reps: int,
    label: str,
) -> dict[str, Any]:
    """Run one main5 case and return a result record."""
    run_name = (
        f"m5_{label}_{variant.name}_{scenario['name']}"
        f"_r{reps}_c{int(cost)}_s{seed}"
        f"_pbsz{scenario['block_size']}_pbs{scenario['path_seed'] + seed}"
    )
    run_dir = RUN_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    min_signal_scale = (
        variant.min_signal_scale
        if variant.min_signal_scale is not None
        else BASE_PARAMS["min_signal_scale"]
    )
    max_signal_scale = (
        variant.max_signal_scale
        if variant.max_signal_scale is not None
        else BASE_PARAMS["max_signal_scale"]
    )

    cmd = [
        str(PYTHON_BIN),
        "src/wealth_first/main5.py",
        "--gate", "055",
        "--gate-scale", "bps",
        "--n-folds", "5",
        "--seed", str(seed),
        "--output-dir", str(run_dir),
        "--transaction-cost-bps", str(cost),
        "--slippage-bps", str(cost),
        "--ridge-l2", str(BASE_PARAMS["ridge_l2"]),
        "--min-signal-scale", str(min_signal_scale),
        "--max-signal-scale", str(max_signal_scale),
        "--min-spy-weight", str(BASE_PARAMS["min_spy_weight"]),
        "--max-spy-weight", str(BASE_PARAMS["max_spy_weight"]),
        "--initial-spy-weight", "1.0",
        "--action-smoothing", str(variant.action_smoothing),
        "--no-trade-band", str(variant.no_trade_band),
        "--scale-turnover-penalty", str(variant.scale_turnover_penalty),
        "--execution-gate-mode", variant.execution_gate_mode,
        "--smooth-gate-width-ratio", str(variant.smooth_gate_width_ratio),
        "--smooth-gate-floor", str(variant.smooth_gate_floor),
        "--validation-relative-floor-penalty", str(variant.validation_relative_floor_penalty),
        "--validation-max-relative-drawdown-penalty", str(variant.validation_max_relative_drawdown_penalty),
        "--validation-step-rate-penalty", str(variant.validation_step_rate_penalty),
        "--validation-suppression-rate-penalty", str(variant.validation_suppression_rate_penalty),
        "--validation-hard-min-relative-return", "-0.001",
        "--validation-tail-bootstrap-reps", "80",
        "--validation-tail-bootstrap-block-size", str(scenario["block_size"]),
        "--validation-tail-bootstrap-quantile", "0.05",
        "--validation-tail-bootstrap-floor-target", "-0.003",
        "--validation-tail-bootstrap-penalty", "2.0",
        "--validation-tail-bootstrap-hard-min", "-0.007",
        "--validation-tail-bootstrap-objective-weight", "4.0",
        "--validation-tail-bootstrap-seed", str(scenario["path_seed"] + seed),
        "--min-robust-min-relative", "0.0",
        "--min-active-fraction", "0.01",
        "--forward-horizon", str(variant.forward_horizon),
        "--path-bootstrap-reps", str(reps),
        "--path-bootstrap-block-size", str(scenario["block_size"]),
        "--path-bootstrap-seed", str(scenario["path_seed"] + seed),
    ]

    if variant.validation_relative_floor_target is not None:
        cmd.extend(["--validation-relative-floor-target", str(variant.validation_relative_floor_target)])
    if variant.validation_step_rate_target is not None:
        cmd.extend(["--validation-step-rate-target", str(variant.validation_step_rate_target)])
    if variant.validation_suppression_rate_target is not None:
        cmd.extend(["--validation-suppression-rate-target", str(variant.validation_suppression_rate_target)])

    detailed_path = run_dir / f"main5_gate055_f5_s{seed}_detailed.json"
    error_record: dict[str, Any] = {
        "variant": variant.name,
        "scenario": scenario["name"],
        "seed": seed,
        "transaction_cost_bps": cost,
        "path_bootstrap_reps": reps,
        "path_bootstrap_robust_min_p05": None,
        "case_slack": None,
        "breach": True,
        "gate_passed": False,
        "mean_test_relative": None,
        "mean_validation_relative": None,
        "mean_turnover": None,
        "mean_test_executed_step_count": None,
        "robust_min_test_relative": None,
        "mean_test_gate_suppression_rate": None,
    }

    res = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        capture_output=True,
        text=True,
    )

    if res.returncode != 0 and not detailed_path.exists():
        return {**error_record, "error": (res.stderr or "").strip()[:1000]}

    if not detailed_path.exists():
        return {**error_record, "error": "detailed_json_missing_after_nonzero_exit"}

    detail = json.loads(detailed_path.read_text(encoding="utf-8"))
    gate_checks = detail.get("gate_checks", {})
    summary_metrics = detail.get("summary_metrics", {})
    p05 = float(summary_metrics.get("path_bootstrap_robust_min_test_relative_p05", float("nan")))
    gate_passed = bool(gate_checks.get("overall_passed", True))
    # Breach is determined by path bootstrap p05, NOT by gate checks.
    # Gate failures (e.g. robust_min_threshold) are reported as a diagnostic but do
    # not automatically suppress the p05 value; that would hide whether the model's
    # bootstrap distribution clears the path threshold.
    import math
    breach = bool(math.isnan(p05) or p05 < FIXED_PATH_THRESHOLD)
    return {
        "variant": variant.name,
        "scenario": scenario["name"],
        "seed": seed,
        "transaction_cost_bps": cost,
        "path_bootstrap_reps": reps,
        "path_bootstrap_robust_min_p05": p05,
        "case_slack": p05 - FIXED_PATH_THRESHOLD,
        "breach": breach,
        "gate_passed": gate_passed,
        "mean_test_relative": float(summary_metrics.get("mean_test_relative_total_return", 0.0)),
        "mean_validation_relative": float(summary_metrics.get("mean_validation_relative_total_return", 0.0)),
        "mean_turnover": float(summary_metrics.get("mean_turnover", 0.0)),
        "mean_test_executed_step_count": float(summary_metrics.get("mean_test_executed_step_count", 0.0)),
        "robust_min_test_relative": float(summary_metrics.get("robust_min_test_relative", 0.0)),
        "mean_test_gate_suppression_rate": float(summary_metrics.get("mean_test_gate_suppression_rate", 0.0)),
        "error": None,
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant_name, grp in df.groupby("variant"):
        altblock = grp[grp["scenario"] == "altblock_b16_s4242"]
        deepcheck = grp[grp["scenario"] == "deepcheck_b12_s8080"]
        rows.append(
            {
                "variant": variant_name,
                "rows": int(len(grp)),
                "breach_count": int(grp["breach"].sum()),
                "breach_rate": float(grp["breach"].mean()),
                "min_case_slack": float(grp["case_slack"].min()),
                "mean_case_slack": float(grp["case_slack"].mean()),
                "mean_test_relative": float(grp["mean_test_relative"].mean()),
                "mean_validation_relative": float(grp["mean_validation_relative"].mean()),
                "mean_turnover": float(grp["mean_turnover"].mean()),
                "mean_test_executed_step_count": float(grp["mean_test_executed_step_count"].mean()),
                "mean_robust_min_test_relative": float(grp["robust_min_test_relative"].mean()),
                "mean_test_gate_suppression_rate": float(grp["mean_test_gate_suppression_rate"].mean()),
                "altblock_breach_count": int(altblock["breach"].sum()),
                "altblock_min_case_slack": float(altblock["case_slack"].min()),
                "deepcheck_breach_count": int(deepcheck["breach"].sum()),
                "deepcheck_min_case_slack": float(deepcheck["case_slack"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values("breach_count").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Main5 baseline 16-case robustness run")
    parser.add_argument("--reps", type=int, default=300, help="Path-bootstrap reps per case")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument(
        "--label", type=str, default="a", help="Run label suffix for artifact naming"
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="",
        help=(
            "Comma-separated variant names to run (default: all). "
            "Example: --variants m5_h10_ntb050_s020"
        ),
    )
    args = parser.parse_args()

    reps = args.reps
    workers = args.workers
    label = args.label
    selected_variants: list[Main5Variant]
    if args.variants.strip():
        requested = {name.strip() for name in args.variants.split(",") if name.strip()}
        selected_variants = [v for v in VARIANTS if v.name in requested]
        missing = sorted(requested - {v.name for v in selected_variants})
        if missing:
            print(f"Error: unknown variant(s): {', '.join(missing)}", file=sys.stderr)
            print(
                "Available variants: " + ", ".join(v.name for v in VARIANTS),
                file=sys.stderr,
            )
            return 2
    else:
        selected_variants = VARIANTS

    # Build all 16 cases: 1 variant × 4 scenarios × 2 seeds × 2 costs
    cases = [
        (variant, scenario, seed, cost)
        for variant in selected_variants
        for scenario in SCENARIOS
        for seed in SEEDS
        for cost in COSTS
    ]
    total = len(cases)
    print(f"Running {total} cases ({len(selected_variants)} variants × {len(SCENARIOS)} scenarios × "
          f"{len(SEEDS)} seeds × {len(COSTS)} costs), reps={reps}, workers={workers}")

    records: list[dict[str, Any]] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_case, v, sc, s, c, reps, label): (v, sc, s, c)
            for v, sc, s, c in cases
        }
        for fut in as_completed(futures):
            v, sc, s, c = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                rec = {
                    "variant": v.name,
                    "scenario": sc["name"],
                    "seed": s,
                    "transaction_cost_bps": c,
                    "breach": True,
                    "case_slack": None,
                    "error": str(exc)[:500],
                }
            records.append(rec)
            completed += 1
            breach_sym = "BREACH" if rec.get("breach") else "ok   "
            slack_str = f"{rec.get('case_slack', float('nan')):.4f}" if rec.get("case_slack") is not None else "  n/a"
            gate_sym = "" if rec.get("gate_passed", True) else " [gate-fail]"
            print(
                f"[{completed:2d}/{total}] {breach_sym}  slack={slack_str}"
                f"  {v.name}  {sc['name']}  s={s}  c={c}{gate_sym}"
            )

    detail_df = pd.DataFrame(records)
    detail_csv = ARTIFACT_ROOT / f"main5_baseline_{label}_detail.csv"
    detail_df.to_csv(detail_csv, index=False)
    print(f"\nWrote {detail_csv}")

    summary_df = summarize(detail_df)
    summary_csv = ARTIFACT_ROOT / f"main5_baseline_{label}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Wrote {summary_csv}")

    summary_json = ARTIFACT_ROOT / f"main5_baseline_{label}_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "label": label,
                "reps": reps,
                "fixed_path_threshold": FIXED_PATH_THRESHOLD,
                "phase40_ref": PHASE40_REF,
                "base_params": BASE_PARAMS,
                "summary": json.loads(summary_df.to_json(orient="records")),
                "detail": json.loads(detail_df.to_json(orient="records")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {summary_json}")

    # Print comparison table
    print("\n=== Main5 Baseline vs Phase40 Reference ===")
    print(f"{'Variant':<35} {'Breaches':>8} {'MinSlack':>10} {'MeanTest':>10} {'GateSup':>8}")
    print("-" * 75)
    # Phase40 reference row
    print(
        f"{'[REF] ' + PHASE40_REF['variant']:<35}"
        f" {PHASE40_REF['breach_count']:>8}"
        f" {PHASE40_REF['min_case_slack']:>10.4f}"
        f" {PHASE40_REF['mean_test_relative']:>10.4f}"
        f" {PHASE40_REF['mean_test_gate_suppression_rate']:>8.2%}"
    )
    print("-" * 75)
    for _, row in summary_df.iterrows():
        gsr = row.get("mean_test_gate_suppression_rate", float("nan"))
        print(
            f"{str(row['variant']):<35}"
            f" {int(row['breach_count']):>8}"
            f" {float(row['min_case_slack']):>10.4f}"
            f" {float(row['mean_test_relative']):>10.4f}"
            f" {float(gsr):>8.2%}"
        )

    total_breaches = int(detail_df["breach"].sum())
    print(f"\nTotal breaches: {total_breaches}/{total}")
    return 0 if total_breaches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
