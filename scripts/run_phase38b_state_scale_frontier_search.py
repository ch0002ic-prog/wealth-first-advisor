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
RUN_ROOT = ARTIFACT_ROOT / "main4_phase38b_runs"

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
    signal_model_family: str
    regime_drawdown_threshold: float
    regime_drawdown_floor: float
    state_scale_slope_min: float
    state_scale_slope_max: float
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
    "no_trade_band": 0.021429,
    "min_signal_scale": -0.648,
    "max_signal_scale": 0.648,
    "min_spy_weight": 0.809134,
    "max_spy_weight": 1.058647,
}


VARIANTS = [
    Variant(
        name="v38b_single_linear_ref",
        feature_family="baseline",
        signal_model_family="single_linear",
        regime_drawdown_threshold=-0.08,
        regime_drawdown_floor=-0.20,
        state_scale_slope_min=-1.0,
        state_scale_slope_max=0.5,
        scale_turnover_penalty=2.9952,
        **BASE_PARAMS,
    ),
    Variant(
        name="v38b_state_scale_mild_pen24",
        feature_family="baseline",
        signal_model_family="state_scaled_linear",
        regime_drawdown_threshold=-0.08,
        regime_drawdown_floor=-0.20,
        state_scale_slope_min=-0.60,
        state_scale_slope_max=0.15,
        scale_turnover_penalty=2.40,
        **BASE_PARAMS,
    ),
    Variant(
        name="v38b_state_scale_balanced_pen22",
        feature_family="baseline",
        signal_model_family="state_scaled_linear",
        regime_drawdown_threshold=-0.10,
        regime_drawdown_floor=-0.22,
        state_scale_slope_min=-0.45,
        state_scale_slope_max=0.25,
        scale_turnover_penalty=2.20,
        **BASE_PARAMS,
    ),
    Variant(
        name="v38b_state_scale_lowdrag_pen20",
        feature_family="baseline",
        signal_model_family="state_scaled_linear",
        regime_drawdown_threshold=-0.10,
        regime_drawdown_floor=-0.22,
        state_scale_slope_min=-0.30,
        state_scale_slope_max=0.35,
        scale_turnover_penalty=2.00,
        **BASE_PARAMS,
    ),
]


def run_case(variant: Variant, scenario: dict[str, Any], seed: int, cost: float, reps: int, label: str) -> dict[str, Any]:
    run_name = (
        f"phase38b_{label}_{variant.name}_{scenario['name']}_r{reps}_c{int(cost)}_s{seed}"
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
        "--signal-model-family",
        variant.signal_model_family,
        "--regime-drawdown-threshold",
        str(variant.regime_drawdown_threshold),
        "--regime-drawdown-floor",
        str(variant.regime_drawdown_floor),
        "--state-scale-slope-min",
        str(variant.state_scale_slope_min),
        "--state-scale-slope-max",
        str(variant.state_scale_slope_max),
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
            "signal_model_family": variant.signal_model_family,
            "regime_drawdown_threshold": variant.regime_drawdown_threshold,
            "regime_drawdown_floor": variant.regime_drawdown_floor,
            "state_scale_slope_min": variant.state_scale_slope_min,
            "state_scale_slope_max": variant.state_scale_slope_max,
            "scale_turnover_penalty": variant.scale_turnover_penalty,
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
    if int(summary_metrics.get("n_rows", 0)) <= 0:
        return {
            "variant": variant.name,
            "feature_family": variant.feature_family,
            "signal_model_family": variant.signal_model_family,
            "regime_drawdown_threshold": variant.regime_drawdown_threshold,
            "regime_drawdown_floor": variant.regime_drawdown_floor,
            "state_scale_slope_min": variant.state_scale_slope_min,
            "state_scale_slope_max": variant.state_scale_slope_max,
            "scale_turnover_penalty": variant.scale_turnover_penalty,
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
            "error": "main4 returned empty fold set (n_rows=0)",
        }

    p05 = float(summary_metrics.get("path_bootstrap_robust_min_test_relative_p05", float("nan")))
    return {
        "variant": variant.name,
        "feature_family": variant.feature_family,
        "signal_model_family": variant.signal_model_family,
        "regime_drawdown_threshold": variant.regime_drawdown_threshold,
        "regime_drawdown_floor": variant.regime_drawdown_floor,
        "state_scale_slope_min": variant.state_scale_slope_min,
        "state_scale_slope_max": variant.state_scale_slope_max,
        "scale_turnover_penalty": variant.scale_turnover_penalty,
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
    keys = [
        "variant",
        "feature_family",
        "signal_model_family",
        "regime_drawdown_threshold",
        "regime_drawdown_floor",
        "state_scale_slope_min",
        "state_scale_slope_max",
        "scale_turnover_penalty",
    ]
    for key_vals, grp in df.groupby(keys, dropna=False):
        rec = dict(zip(keys, key_vals, strict=False))
        rec.update(
            {
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
        rows.append(rec)
    rank = pd.DataFrame(rows)
    rank = rank.sort_values(
        ["breach_count", "mean_test_relative", "min_case_slack", "mean_validation_relative"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    return rank


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase38b state-scale frontier search for main4")
    parser.add_argument("--label", default="state_scale_frontier_a")
    parser.add_argument("--reps", type=int, default=300)
    args = parser.parse_args()

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    total = len(VARIANTS) * len(SCENARIOS) * len(SEEDS) * len(COSTS)
    print(f"Phase38b state-scale frontier search: label={args.label} reps={args.reps} cases={total}")

    results = []
    case_index = 0
    for variant in VARIANTS:
        for scenario in SCENARIOS:
            for seed in SEEDS:
                for cost in COSTS:
                    case_index += 1
                    print(
                        f"  search [{case_index}/{total}] {variant.name} / {scenario['name']} / s{seed} / c{int(cost)}",
                        flush=True,
                    )
                    results.append(run_case(variant, scenario, seed, float(cost), args.reps, args.label))

    detail = pd.DataFrame(results)
    rank = summarize(detail)
    best_row = rank.iloc[0].to_dict()
    baseline_row = rank.loc[rank["variant"] == "v38b_single_linear_ref"].iloc[0].to_dict()

    detail_path = ARTIFACT_ROOT / f"main4_phase38b_{args.label}_detail.csv"
    rank_path = ARTIFACT_ROOT / f"main4_phase38b_{args.label}_rank.csv"
    summary_path = ARTIFACT_ROOT / f"main4_phase38b_{args.label}_summary.json"

    detail.to_csv(detail_path, index=False)
    rank.to_csv(rank_path, index=False)

    summary = {
        "phase": "Phase38b",
        "title": "State-Scale Frontier Search",
        "label": args.label,
        "reps": args.reps,
        "fixed_path_threshold": FIXED_PATH_THRESHOLD,
        "search_hypothesis": "Milder state-scale slopes and lower turnover penalty can recover return while preserving robustness gains.",
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
