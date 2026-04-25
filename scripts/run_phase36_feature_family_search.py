#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
RUN_ROOT = ARTIFACT_ROOT / "main4_phase36_runs"

FIXED_PATH_THRESHOLD = -0.007

SCENARIOS = [
    {"name": "baseline_b20_s4242", "block_size": 20, "path_seed": 4242, "group": "curated"},
    {"name": "altblock_b16_s4242", "block_size": 16, "path_seed": 4242, "group": "curated"},
    {"name": "altseed_b20_s5252", "block_size": 20, "path_seed": 5252, "group": "curated"},
    {"name": "deepcheck_b12_s8080", "block_size": 12, "path_seed": 8080, "group": "curated"},
]
SEEDS = [5, 17]
COSTS = [22, 25]


@dataclass(frozen=True)
class Variant:
    name: str
    feature_family: str
    ridge_l2: float
    action_smoothing: float
    scale_turnover_penalty: float
    no_trade_band: float
    min_signal_scale: float
    max_signal_scale: float
    min_spy_weight: float
    max_spy_weight: float


BASE_PARAMS = {
    "ridge_l2": 0.020815,
    "action_smoothing": 1.1425,
    "scale_turnover_penalty": 2.9952,
    "no_trade_band": 0.021429,
    "min_signal_scale": -0.648,
    "max_signal_scale": 0.648,
    "min_spy_weight": 0.809134,
    "max_spy_weight": 1.058647,
}


VARIANTS = [
    Variant(name="v36_baseline_ref", feature_family="baseline", **BASE_PARAMS),
    Variant(name="v36_regime_interactions", feature_family="regime_interactions", **BASE_PARAMS),
    Variant(name="v36_shock_reversal", feature_family="shock_reversal", **BASE_PARAMS),
]


def run_case(variant: Variant, scenario: dict[str, Any], seed: int, cost: float, reps: int, label: str) -> dict[str, Any]:
    run_name = (
        f"phase36_{label}_{variant.name}_{scenario['name']}_r{reps}_c{int(cost)}_s{seed}"
        f"_pbsz{scenario['block_size']}_pbs{scenario['path_seed'] + seed}"
    )
    run_dir = RUN_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PYTHON_BIN),
        "src/wealth_first/main4.py",
        "--gate",
        "055",
        "--gate-scale",
        "bps",
        "--n-folds",
        "5",
        "--seed",
        str(seed),
        "--output-dir",
        str(run_dir),
        "--transaction-cost-bps",
        str(cost),
        "--slippage-bps",
        str(cost),
        "--ridge-l2",
        str(variant.ridge_l2),
        "--target-mode",
        "tanh_return",
        "--feature-family",
        variant.feature_family,
        "--scale-turnover-penalty",
        str(variant.scale_turnover_penalty),
        "--min-signal-scale",
        str(variant.min_signal_scale),
        "--max-signal-scale",
        str(variant.max_signal_scale),
        "--min-spy-weight",
        str(variant.min_spy_weight),
        "--max-spy-weight",
        str(variant.max_spy_weight),
        "--initial-spy-weight",
        "1.0",
        "--action-smoothing",
        str(variant.action_smoothing),
        "--no-trade-band",
        str(variant.no_trade_band),
        "--min-robust-min-relative",
        "0.0",
        "--min-active-fraction",
        "0.01",
        "--path-bootstrap-reps",
        str(reps),
        "--path-bootstrap-block-size",
        str(scenario["block_size"]),
        "--path-bootstrap-seed",
        str(scenario["path_seed"] + seed),
    ]

    detailed_path = run_dir / f"main4_gate055_f5_s{seed}_detailed.json"
    res = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        capture_output=True,
        text=True,
    )
    if res.returncode != 0 and not detailed_path.exists():
        return {
            "variant": variant.name,
            "feature_family": variant.feature_family,
            "scenario": scenario["name"],
            "seed": seed,
            "transaction_cost_bps": cost,
            "path_bootstrap_reps": reps,
            "path_bootstrap_robust_min_p05": None,
            "case_slack": None,
            "breach": True,
            "mean_test_relative": None,
            "mean_validation_relative": None,
            "mean_turnover": None,
            "error": (res.stderr or "").strip()[:800],
        }

    detail = json.loads(detailed_path.read_text(encoding="utf-8"))
    summary_metrics = detail.get("summary_metrics", {})
    p05 = float(summary_metrics.get("path_bootstrap_robust_min_test_relative_p05", float("nan")))
    return {
        "variant": variant.name,
        "feature_family": variant.feature_family,
        "scenario": scenario["name"],
        "seed": seed,
        "transaction_cost_bps": cost,
        "path_bootstrap_reps": reps,
        "path_bootstrap_robust_min_p05": p05,
        "case_slack": p05 - FIXED_PATH_THRESHOLD,
        "breach": bool(p05 < FIXED_PATH_THRESHOLD),
        "mean_test_relative": float(summary_metrics.get("mean_test_relative_total_return", 0.0)),
        "mean_validation_relative": float(summary_metrics.get("mean_validation_relative_total_return", 0.0)),
        "mean_turnover": float(summary_metrics.get("mean_turnover", 0.0)),
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, feature_family), grp in df.groupby(["variant", "feature_family"], dropna=False):
        rows.append(
            {
                "variant": variant,
                "feature_family": feature_family,
                "rows": int(len(grp)),
                "breach_count": int(grp["breach"].sum()),
                "breach_rate": float(grp["breach"].mean()),
                "min_case_slack": float(grp["case_slack"].min()),
                "mean_case_slack": float(grp["case_slack"].mean()),
                "mean_test_relative": float(grp["mean_test_relative"].mean()),
                "mean_validation_relative": float(grp["mean_validation_relative"].mean()),
                "mean_turnover": float(grp["mean_turnover"].mean()),
            }
        )
    rank = pd.DataFrame(rows)
    rank = rank.sort_values(
        ["breach_count", "mean_test_relative", "min_case_slack", "mean_validation_relative"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    return rank


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase36 feature-family mechanism search for main4")
    parser.add_argument("--label", default="feature_family_a")
    parser.add_argument("--reps", type=int, default=300)
    args = parser.parse_args()

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    total = len(VARIANTS) * len(SCENARIOS) * len(SEEDS) * len(COSTS)
    print(f"Phase36 feature-family search: label={args.label} reps={args.reps} cases={total}")

    results = []
    case_index = 0
    for variant in VARIANTS:
        for scenario in SCENARIOS:
            for seed in SEEDS:
                for cost in COSTS:
                    case_index += 1
                    print(
                        f"  search [{case_index}/{total}] {variant.feature_family} / {scenario['name']} / s{seed} / c{int(cost)}",
                        flush=True,
                    )
                    results.append(run_case(variant, scenario, seed, float(cost), args.reps, args.label))

    detail = pd.DataFrame(results)
    rank = summarize(detail)
    best_row = rank.iloc[0].to_dict()
    baseline_row = rank.loc[rank["feature_family"] == "baseline"].iloc[0].to_dict()

    detail_path = ARTIFACT_ROOT / f"main4_phase36_{args.label}_detail.csv"
    rank_path = ARTIFACT_ROOT / f"main4_phase36_{args.label}_rank.csv"
    summary_path = ARTIFACT_ROOT / f"main4_phase36_{args.label}_summary.json"

    detail.to_csv(detail_path, index=False)
    rank.to_csv(rank_path, index=False)

    summary = {
        "phase": "Phase36",
        "title": "Feature-Family Mechanism Search",
        "label": args.label,
        "reps": args.reps,
        "fixed_path_threshold": FIXED_PATH_THRESHOLD,
        "search_hypothesis": "Richer feature families may improve robustness and return without another local threshold microtune.",
        "winner": best_row,
        "baseline_reference": baseline_row,
        "winner_vs_baseline": {
            "delta_mean_test_relative": float(best_row["mean_test_relative"] - baseline_row["mean_test_relative"]),
            "delta_min_case_slack": float(best_row["min_case_slack"] - baseline_row["min_case_slack"]),
            "delta_mean_turnover": float(best_row["mean_turnover"] - baseline_row["mean_turnover"]),
        },
        "all_variants": rank.to_dict("records"),
        "artifacts": {
            "detail": str(detail_path),
            "rank": str(rank_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "winner": best_row["variant"]}))


if __name__ == "__main__":
    main()