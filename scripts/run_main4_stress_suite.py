#!/usr/bin/env python3
"""Run stress sweeps for main4 friction and objective-related knobs."""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"
OUTPUT_DIR = PROJECT_ROOT / "artifacts"


@dataclass(frozen=True)
class StressCase:
    """Single stress test case for main4."""

    name: str
    gate: str
    transaction_cost_bps: float
    slippage_bps: float
    ridge_l2: float
    action_smoothing: float
    no_trade_band: float
    seed: int = 7
    n_folds: int = 3


STRESS_CASES = [
    StressCase(
        name="baseline",
        gate="009",
        transaction_cost_bps=5.0,
        slippage_bps=5.0,
        ridge_l2=1.0,
        action_smoothing=0.5,
        no_trade_band=0.02,
    ),
    StressCase(
        name="low_friction",
        gate="009",
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        ridge_l2=1.0,
        action_smoothing=0.5,
        no_trade_band=0.02,
    ),
    StressCase(
        name="high_friction",
        gate="009",
        transaction_cost_bps=10.0,
        slippage_bps=10.0,
        ridge_l2=1.0,
        action_smoothing=0.5,
        no_trade_band=0.02,
    ),
    StressCase(
        name="very_high_friction",
        gate="009",
        transaction_cost_bps=20.0,
        slippage_bps=20.0,
        ridge_l2=1.0,
        action_smoothing=0.5,
        no_trade_band=0.02,
    ),
    StressCase(
        name="higher_ridge",
        gate="009",
        transaction_cost_bps=5.0,
        slippage_bps=5.0,
        ridge_l2=5.0,
        action_smoothing=0.5,
        no_trade_band=0.02,
    ),
    StressCase(
        name="lower_smoothing",
        gate="009",
        transaction_cost_bps=5.0,
        slippage_bps=5.0,
        ridge_l2=1.0,
        action_smoothing=0.25,
        no_trade_band=0.02,
    ),
    StressCase(
        name="higher_no_trade_band",
        gate="009",
        transaction_cost_bps=5.0,
        slippage_bps=5.0,
        ridge_l2=1.0,
        action_smoothing=0.5,
        no_trade_band=0.04,
    ),
    StressCase(
        name="tighter_gate",
        gate="012",
        transaction_cost_bps=5.0,
        slippage_bps=5.0,
        ridge_l2=1.0,
        action_smoothing=0.5,
        no_trade_band=0.02,
    ),
]


def _read_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _run_case(case: StressCase) -> dict[str, Any]:
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
        str(OUTPUT_DIR),
        "--transaction-cost-bps",
        str(case.transaction_cost_bps),
        "--slippage-bps",
        str(case.slippage_bps),
        "--ridge-l2",
        str(case.ridge_l2),
        "--action-smoothing",
        str(case.action_smoothing),
        "--no-trade-band",
        str(case.no_trade_band),
    ]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Case {case.name} failed with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    summary_csv = OUTPUT_DIR / f"main4_gate{case.gate}_f{case.n_folds}_s{case.seed}_summary.csv"
    detailed_json = OUTPUT_DIR / f"main4_gate{case.gate}_f{case.n_folds}_s{case.seed}_detailed.json"

    summary_df = pd.read_csv(summary_csv)
    detailed = _read_json(detailed_json)

    mean_relative = float(summary_df["policy_relative_total_return"].mean())
    beat_hold = float((summary_df["policy_relative_total_return"] > 0).mean())
    active_fraction = float(summary_df["active"].mean())
    mean_turnover = float(summary_df["mean_turnover"].mean())
    robust_min = float(summary_df["policy_relative_total_return"].min())

    return {
        "case": case.name,
        "gate": case.gate,
        "seed": case.seed,
        "n_folds": case.n_folds,
        "transaction_cost_bps": case.transaction_cost_bps,
        "slippage_bps": case.slippage_bps,
        "ridge_l2": case.ridge_l2,
        "action_smoothing": case.action_smoothing,
        "no_trade_band": case.no_trade_band,
        "mean_test_relative": mean_relative,
        "beat_hold_fraction": beat_hold,
        "active_fraction": active_fraction,
        "mean_turnover": mean_turnover,
        "robust_min_test_relative": robust_min,
        "mean_signal_scale": float(summary_df["signal_scale"].mean()) if "signal_scale" in summary_df.columns else None,
        "mean_signal_bias": float(summary_df["signal_bias"].mean()) if "signal_bias" in summary_df.columns else None,
        "fingerprint": detailed.get("fingerprint", {}),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("Running main4 stress suite")
    print("=" * 70)

    rows: list[dict[str, Any]] = []
    for case in STRESS_CASES:
        print(
            f"\n{case.name:20s} "
            f"gate={case.gate} "
            f"cost={case.transaction_cost_bps + case.slippage_bps:.1f}bps "
            f"ridge={case.ridge_l2:.2f} "
            f"smooth={case.action_smoothing:.2f} "
            f"band={case.no_trade_band:.3f}"
        )
        row = _run_case(case)
        rows.append(row)
        print(
            f"  rel={row['mean_test_relative']:+.6f} "
            f"beat={row['beat_hold_fraction']:.1%} "
            f"active={row['active_fraction']:.1%} "
            f"turnover={row['mean_turnover']:.6f}"
        )

    table = pd.DataFrame(rows)
    baseline_rel = float(table.loc[table["case"] == "baseline", "mean_test_relative"].iloc[0])
    baseline_turn = float(table.loc[table["case"] == "baseline", "mean_turnover"].iloc[0])
    table["delta_relative_vs_baseline"] = table["mean_test_relative"] - baseline_rel
    table["delta_turnover_vs_baseline"] = table["mean_turnover"] - baseline_turn

    table = table.sort_values("mean_test_relative", ascending=False).reset_index(drop=True)

    detail_csv = OUTPUT_DIR / "main4_stress_suite_detail.csv"
    table.to_csv(detail_csv, index=False)

    summary_json = OUTPUT_DIR / "main4_stress_suite_summary.json"
    top = table.iloc[0].to_dict()
    summary = {
        "n_cases": int(len(table)),
        "best_case": top["case"],
        "best_mean_test_relative": float(top["mean_test_relative"]),
        "baseline_mean_test_relative": baseline_rel,
        "improvement_vs_baseline": float(top["mean_test_relative"] - baseline_rel),
        "ranked_cases": table[
            [
                "case",
                "mean_test_relative",
                "delta_relative_vs_baseline",
                "beat_hold_fraction",
                "active_fraction",
                "mean_turnover",
                "delta_turnover_vs_baseline",
            ]
        ].to_dict(orient="records"),
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("Stress Suite Summary")
    print("=" * 70)
    print(f"Wrote {detail_csv.name}")
    print(f"Wrote {summary_json.name}")
    print(f"Best case: {summary['best_case']} ({summary['best_mean_test_relative']:+.6f})")
    print(f"Improvement vs baseline: {summary['improvement_vs_baseline']:+.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
