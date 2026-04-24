#!/usr/bin/env python3
"""Run a focused main4 promotion sprint around the promoted baseline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
RUN_ROOT = ARTIFACT_ROOT / "main4_promotion_sprint_runs"


@dataclass(frozen=True)
class SprintCase:
    name: str
    gate: str
    transaction_cost_bps: float
    slippage_bps: float
    ridge_l2: float
    action_smoothing: float
    target_mode: str = "sign"
    scale_turnover_penalty: float = 0.0
    min_signal_scale: float = -0.75
    max_signal_scale: float = 0.75
    min_spy_weight: float = 0.80
    max_spy_weight: float = 1.05
    initial_spy_weight: float = 1.0
    no_trade_band: float = 0.02
    seed: int = 5
    n_folds: int = 5
    min_robust_min_relative: float = 0.0
    min_active_fraction: float = 0.01
    gate_scale: str = "bps"


DEPLOYED_PROMOTED_SIGN_CASE = SprintCase(
    name="promoted_sign_12bps_gate55",
    gate="055",
    transaction_cost_bps=12.0,
    slippage_bps=12.0,
    ridge_l2=0.015,
    min_signal_scale=-0.5,
    max_signal_scale=0.5,
    action_smoothing=1.0,
)

LEAD_TANH_CASE = SprintCase(
    name="candidate_tanh_12bps_gate55",
    gate="055",
    transaction_cost_bps=12.0,
    slippage_bps=12.0,
    ridge_l2=0.015,
    target_mode="tanh_return",
    action_smoothing=1.0,
)


QUICK_CASES = [
    DEPLOYED_PROMOTED_SIGN_CASE,
    LEAD_TANH_CASE,
    SprintCase(
        name="promoted_sign_15bps_gate55",
        gate="055",
        transaction_cost_bps=15.0,
        slippage_bps=15.0,
        ridge_l2=0.015,
        min_signal_scale=-0.5,
        max_signal_scale=0.5,
        action_smoothing=1.0,
    ),
    SprintCase(
        name="candidate_tanh_15bps_gate55",
        gate="055",
        transaction_cost_bps=15.0,
        slippage_bps=15.0,
        ridge_l2=0.015,
        target_mode="tanh_return",
        action_smoothing=1.0,
    ),
    SprintCase(
        name="promoted_sign_18bps_gate55",
        gate="055",
        transaction_cost_bps=18.0,
        slippage_bps=18.0,
        ridge_l2=0.015,
        min_signal_scale=-0.5,
        max_signal_scale=0.5,
        action_smoothing=1.0,
    ),
    SprintCase(
        name="candidate_tanh_18bps_gate55",
        gate="055",
        transaction_cost_bps=18.0,
        slippage_bps=18.0,
        ridge_l2=0.015,
        target_mode="tanh_return",
        action_smoothing=1.0,
    ),
]


FULL_EXTRA_CASES = [
    SprintCase(
        name="candidate_tanh_15bps_gate55_seed17",
        gate="055",
        transaction_cost_bps=15.0,
        slippage_bps=15.0,
        ridge_l2=0.015,
        target_mode="tanh_return",
        action_smoothing=1.0,
        seed=17,
    ),
    SprintCase(
        name="promoted_sign_15bps_gate55_seed17",
        gate="055",
        transaction_cost_bps=15.0,
        slippage_bps=15.0,
        ridge_l2=0.015,
        min_signal_scale=-0.5,
        max_signal_scale=0.5,
        action_smoothing=1.0,
        seed=17,
    ),
    SprintCase(
        name="candidate_tanh_12bps_gate60",
        gate="060",
        transaction_cost_bps=12.0,
        slippage_bps=12.0,
        ridge_l2=0.015,
        target_mode="tanh_return",
        action_smoothing=1.0,
    ),
]


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _summary_paths(case: SprintCase, run_dir: Path) -> tuple[Path, Path]:
    stem = f"main4_gate{case.gate}_f{case.n_folds}_s{case.seed}"
    return run_dir / f"{stem}_summary.csv", run_dir / f"{stem}_detailed.json"


def _run_case(case: SprintCase, force: bool) -> dict[str, Any]:
    run_dir = RUN_ROOT / case.name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_csv, detailed_json = _summary_paths(case, run_dir)

    if force or not summary_csv.exists() or not detailed_json.exists():
        cmd = [
            str(PYTHON_BIN),
            "src/wealth_first/main4.py",
            "--gate",
            case.gate,
            "--n-folds",
            str(case.n_folds),
            "--seed",
            str(case.seed),
            "--output-dir",
            str(run_dir),
            "--transaction-cost-bps",
            str(case.transaction_cost_bps),
            "--slippage-bps",
            str(case.slippage_bps),
            "--ridge-l2",
            str(case.ridge_l2),
            "--target-mode",
            case.target_mode,
            "--scale-turnover-penalty",
            str(case.scale_turnover_penalty),
            "--min-signal-scale",
            str(case.min_signal_scale),
            "--max-signal-scale",
            str(case.max_signal_scale),
            "--min-spy-weight",
            str(case.min_spy_weight),
            "--max-spy-weight",
            str(case.max_spy_weight),
            "--initial-spy-weight",
            str(case.initial_spy_weight),
            "--action-smoothing",
            str(case.action_smoothing),
            "--no-trade-band",
            str(case.no_trade_band),
            "--min-robust-min-relative",
            str(case.min_robust_min_relative),
            "--min-active-fraction",
            str(case.min_active_fraction),
            "--gate-scale",
            case.gate_scale,
        ]
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": "src"},
            capture_output=True,
            text=True,
        )
    else:
        result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="cached")

    if not summary_csv.exists() or not detailed_json.exists():
        raise RuntimeError(
            f"Case {case.name} did not produce expected artifacts.\n"
            f"returncode={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    summary_df = pd.read_csv(summary_csv)
    detailed = _read_json(detailed_json)
    gate_checks = detailed.get("gate_checks", {})
    metrics = detailed.get("summary_metrics", {})
    gate_detail = gate_checks.get("validation_threshold", {})
    robust_detail = gate_checks.get("robust_min_threshold", {})
    active_detail = gate_checks.get("active_fraction_threshold", {})

    row = {
        "case": case.name,
        "returncode": int(result.returncode),
        "cached": bool(result.stderr == "cached"),
        **asdict(case),
        "mean_test_relative": float(metrics.get("mean_test_relative_total_return", summary_df["policy_relative_total_return"].mean())),
        "mean_validation_relative": float(metrics.get("mean_validation_relative_total_return", summary_df["validation_relative_total_return"].mean())),
        "robust_min_test_relative": float(metrics.get("robust_min_test_relative", summary_df["policy_relative_total_return"].min())),
        "beat_hold_fraction": float(metrics.get("beat_hold_fraction", (summary_df["policy_relative_total_return"] > 0).mean())),
        "active_fraction": float(metrics.get("active_fraction", summary_df["active"].mean())),
        "mean_turnover": float(metrics.get("mean_turnover", summary_df["mean_turnover"].mean())),
        "mean_validation_gate_suppression_rate": float(metrics.get("mean_validation_gate_suppression_rate", summary_df["validation_gate_suppression_rate"].mean())),
        "mean_test_gate_suppression_rate": float(metrics.get("mean_test_gate_suppression_rate", summary_df["test_gate_suppression_rate"].mean())),
        "mean_test_executed_step_count": float(metrics.get("mean_test_executed_step_count", summary_df["test_executed_step_count"].mean())),
        "overall_gate_passed": bool(gate_checks.get("overall_passed", False)),
        "validation_gate_passed": bool(gate_detail.get("passed", False)),
        "validation_gate_value": float(gate_detail.get("value", 0.0)),
        "validation_gate_threshold": float(gate_detail.get("threshold", 0.0)),
        "robust_gate_passed": bool(robust_detail.get("passed", False)),
        "robust_gate_value": float(robust_detail.get("value", 0.0)),
        "robust_gate_threshold": float(robust_detail.get("threshold", 0.0)),
        "active_gate_passed": bool(active_detail.get("passed", False)),
        "active_gate_value": float(active_detail.get("value", 0.0)),
        "active_gate_threshold": float(active_detail.get("threshold", 0.0)),
        "summary_csv": str(summary_csv.relative_to(PROJECT_ROOT)),
        "detailed_json": str(detailed_json.relative_to(PROJECT_ROOT)),
    }
    return row


def _candidate_margin(row: pd.Series) -> float:
    return min(
        row["validation_gate_value"] - row["validation_gate_threshold"],
        row["robust_gate_value"] - row["robust_gate_threshold"],
        row["active_gate_value"] - row["active_gate_threshold"],
    )


def _build_head_to_head(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Build direct sign-vs-tanh comparisons on matched scenario keys."""
    required_cols = {
        "target_mode",
        "gate",
        "transaction_cost_bps",
        "slippage_bps",
        "seed",
        "mean_test_relative",
        "robust_min_test_relative",
        "overall_gate_passed",
    }
    if not required_cols.issubset(set(table.columns)):
        return []

    sign_rows = table[table["target_mode"] == "sign"]
    tanh_rows = table[table["target_mode"] == "tanh_return"]

    join_keys = ["gate", "transaction_cost_bps", "slippage_bps", "seed"]
    merged = sign_rows.merge(
        tanh_rows,
        on=join_keys,
        how="inner",
        suffixes=("_sign", "_tanh"),
    )
    if merged.empty:
        return []

    merged["delta_test_relative_tanh_minus_sign"] = (
        merged["mean_test_relative_tanh"] - merged["mean_test_relative_sign"]
    )
    merged["delta_robust_min_tanh_minus_sign"] = (
        merged["robust_min_test_relative_tanh"] - merged["robust_min_test_relative_sign"]
    )

    out_cols = [
        "gate",
        "transaction_cost_bps",
        "slippage_bps",
        "seed",
        "case_sign",
        "case_tanh",
        "mean_test_relative_sign",
        "mean_test_relative_tanh",
        "delta_test_relative_tanh_minus_sign",
        "robust_min_test_relative_sign",
        "robust_min_test_relative_tanh",
        "delta_robust_min_tanh_minus_sign",
        "overall_gate_passed_sign",
        "overall_gate_passed_tanh",
    ]
    return merged[out_cols].sort_values(join_keys).to_dict(orient="records")


def _build_summary(table: pd.DataFrame) -> dict[str, Any]:
    promoted_row = table.loc[table["case"] == DEPLOYED_PROMOTED_SIGN_CASE.name].iloc[0]
    ranked = table.sort_values(
        ["overall_gate_passed", "mean_test_relative", "gate_margin_min"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    top = ranked.iloc[0]
    head_to_head = _build_head_to_head(table)
    return {
        "n_cases": int(len(table)),
        "promoted_baseline_case": DEPLOYED_PROMOTED_SIGN_CASE.name,
        "promoted_baseline_mean_test_relative": float(promoted_row["mean_test_relative"]),
        "promoted_baseline_gate_margin_min": float(promoted_row["gate_margin_min"]),
        "best_case": str(top["case"]),
        "best_case_mean_test_relative": float(top["mean_test_relative"]),
        "best_case_gate_margin_min": float(top["gate_margin_min"]),
        "improvement_vs_promoted_baseline": float(top["mean_test_relative"] - promoted_row["mean_test_relative"]),
        "head_to_head_sign_vs_tanh": head_to_head,
        "ranked_cases": ranked[
            [
                "case",
                "overall_gate_passed",
                "mean_test_relative",
                "mean_validation_relative",
                "robust_min_test_relative",
                "gate_margin_min",
                "beat_hold_fraction",
                "active_fraction",
                "mean_turnover",
                "transaction_cost_bps",
                "slippage_bps",
                "ridge_l2",
                "target_mode",
                "action_smoothing",
                "gate",
                "seed",
            ]
        ].to_dict(orient="records"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the main4 promotion sprint around the promoted baseline.")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--force", action="store_true", help="Rerun cases even when isolated artifacts already exist.")
    parser.add_argument("--output-prefix", default="main4_promotion_sprint_quick")
    args = parser.parse_args(argv)

    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"Missing venv Python: {PYTHON_BIN}")

    cases = list(QUICK_CASES)
    if args.mode == "full":
        cases.extend(FULL_EXTRA_CASES)

    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Running main4 promotion sprint ({args.mode})")
    print("=" * 72)

    rows: list[dict[str, Any]] = []
    for case in cases:
        print(
            f"\n{case.name:32s} gate={case.gate} "
            f"cost={case.transaction_cost_bps + case.slippage_bps:.1f}bps "
            f"ridge={case.ridge_l2:.3f} smooth={case.action_smoothing:.2f} "
            f"target={case.target_mode}"
        )
        row = _run_case(case, force=args.force)
        rows.append(row)
        print(
            f"  rel={row['mean_test_relative']:+.6f} "
            f"val={row['mean_validation_relative']:+.6f} "
            f"robust={row['robust_min_test_relative']:+.6f} "
            f"gate={'PASS' if row['overall_gate_passed'] else 'FAIL'}"
        )

    table = pd.DataFrame(rows)
    table["delta_relative_vs_promoted"] = table["mean_test_relative"] - float(
        table.loc[table["case"] == DEPLOYED_PROMOTED_SIGN_CASE.name, "mean_test_relative"].iloc[0]
    )
    table["gate_margin_min"] = table.apply(_candidate_margin, axis=1)
    table = table.sort_values(
        ["overall_gate_passed", "mean_test_relative", "gate_margin_min"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    detail_csv = ARTIFACT_ROOT / f"{args.output_prefix}_detail.csv"
    summary_json = ARTIFACT_ROOT / f"{args.output_prefix}_summary.json"
    table.to_csv(detail_csv, index=False)

    summary = _build_summary(table)
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n" + "=" * 72)
    print("Promotion Sprint Summary")
    print("=" * 72)
    print(f"Wrote {detail_csv.name}")
    print(f"Wrote {summary_json.name}")
    print(
        f"Best case: {summary['best_case']} "
        f"({summary['best_case_mean_test_relative']:+.6f}, "
        f"margin {summary['best_case_gate_margin_min']:+.6f})"
    )
    print(
        f"Improvement vs promoted baseline: "
        f"{summary['improvement_vs_promoted_baseline']:+.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())