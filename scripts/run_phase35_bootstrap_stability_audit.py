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
RUN_ROOT = ARTIFACT_ROOT / "main4_phase35_runs"

FIXED_PATH_THRESHOLD = -0.007

FULL_CURATED_SCENARIOS = [
    {"name": "baseline_b20_s4242", "block_size": 20, "path_seed": 4242, "group": "curated"},
    {"name": "altseed_b20_s5252", "block_size": 20, "path_seed": 5252, "group": "curated"},
    {"name": "altblock_b16_s4242", "block_size": 16, "path_seed": 4242, "group": "curated"},
    {"name": "altblock_b24_s4242", "block_size": 24, "path_seed": 4242, "group": "curated"},
    {"name": "altboth_b24_s6060", "block_size": 24, "path_seed": 6060, "group": "curated"},
    {"name": "deepcheck_b12_s8080", "block_size": 12, "path_seed": 8080, "group": "curated"},
    {"name": "deepcheck_b28_s9090", "block_size": 28, "path_seed": 9090, "group": "curated"},
    {"name": "stress_b14_s7070", "block_size": 14, "path_seed": 7070, "group": "curated"},
    {"name": "stress_b30_s10010", "block_size": 30, "path_seed": 10010, "group": "curated"},
]
FULL_CURATED_SEEDS = [5, 17, 29]
FULL_CURATED_COSTS = [18, 22, 25]

FULL_RANDOM_SCENARIOS = [
    {"name": "random_train_b10_p7064", "block_size": 10, "path_seed": 7064, "group": "random_train"},
    {"name": "random_train_b10_p9889", "block_size": 10, "path_seed": 9889, "group": "random_train"},
    {"name": "random_train_b11_p6801", "block_size": 11, "path_seed": 6801, "group": "random_train"},
    {"name": "random_train_b12_p8403", "block_size": 12, "path_seed": 8403, "group": "random_train"},
    {"name": "random_train_b13_p14175", "block_size": 13, "path_seed": 14175, "group": "random_train"},
    {"name": "random_train_b13_p14394", "block_size": 13, "path_seed": 14394, "group": "random_train"},
    {"name": "random_train_b16_p10914", "block_size": 16, "path_seed": 10914, "group": "random_train"},
    {"name": "random_train_b16_p9682", "block_size": 16, "path_seed": 9682, "group": "random_train"},
    {"name": "random_holdout_b10_p10572", "block_size": 10, "path_seed": 10572, "group": "random_holdout"},
    {"name": "random_holdout_b11_p12340", "block_size": 11, "path_seed": 12340, "group": "random_holdout"},
    {"name": "random_holdout_b12_p6198", "block_size": 12, "path_seed": 6198, "group": "random_holdout"},
    {"name": "random_holdout_b15_p7889", "block_size": 15, "path_seed": 7889, "group": "random_holdout"},
    {"name": "random_holdout_b15_p8956", "block_size": 15, "path_seed": 8956, "group": "random_holdout"},
    {"name": "random_holdout_b16_p10086", "block_size": 16, "path_seed": 10086, "group": "random_holdout"},
]


@dataclass(frozen=True)
class Variant:
    name: str
    ridge_l2: float
    action_smoothing: float
    scale_turnover_penalty: float
    no_trade_band: float
    min_signal_scale: float
    max_signal_scale: float
    min_spy_weight: float
    max_spy_weight: float


CURRENT_BEST = Variant(
    name="v28d_sig02",
    ridge_l2=0.020815,
    action_smoothing=1.1425,
    scale_turnover_penalty=2.9952,
    no_trade_band=0.021429,
    min_signal_scale=-0.648,
    max_signal_scale=0.648,
    min_spy_weight=0.809134,
    max_spy_weight=1.058647,
)


def run_case(variant: Variant, scenario: dict[str, Any], seed: int, cost: float, reps: int, label: str) -> dict[str, Any]:
    run_name = (
        f"phase35_{label}_{variant.name}_{scenario['name']}_r{reps}_c{int(cost)}_s{seed}"
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
            "reps": reps,
            "group": scenario["group"],
            "scenario": scenario["name"],
            "seed": seed,
            "transaction_cost_bps": cost,
            "path_bootstrap_robust_min_p05": None,
            "case_slack": None,
            "breach": True,
            "mean_test_relative": None,
            "error": (res.stderr or "").strip()[:800],
        }

    det = json.loads(detailed_path.read_text(encoding="utf-8"))
    sm = det.get("summary_metrics", {})
    p05 = float(sm.get("path_bootstrap_robust_min_test_relative_p05", float("nan")))
    return {
        "variant": variant.name,
        "reps": reps,
        "group": scenario["group"],
        "scenario": scenario["name"],
        "seed": seed,
        "transaction_cost_bps": cost,
        "path_bootstrap_robust_min_p05": p05,
        "case_slack": p05 - FIXED_PATH_THRESHOLD,
        "breach": bool(p05 < FIXED_PATH_THRESHOLD),
        "mean_test_relative": float(sm.get("mean_test_relative_total_return", 0.0)),
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for reps, grp in df.groupby("reps"):
        rows.append(
            {
                "reps": int(reps),
                "rows": int(len(grp)),
                "breach_count": int(grp["breach"].sum()),
                "breach_rate": float(grp["breach"].mean()),
                "min_case_slack": float(grp["case_slack"].min()),
                "mean_case_slack": float(grp["case_slack"].mean()),
                "mean_test_relative": float(grp["mean_test_relative"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("reps").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase35 bootstrap stability audit for current-best main4 policy")
    parser.add_argument("--label", default="stability_a")
    parser.add_argument("--reps-grid", default="200,300,500")
    args = parser.parse_args()

    reps_grid = [int(x.strip()) for x in args.reps_grid.split(",") if x.strip()]
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    scenarios = FULL_CURATED_SCENARIOS + FULL_RANDOM_SCENARIOS
    case_rows: list[tuple[dict[str, Any], int, float]] = []
    for scn in scenarios:
        if scn["group"] == "curated":
            for seed in FULL_CURATED_SEEDS:
                for cost in FULL_CURATED_COSTS:
                    case_rows.append((scn, seed, float(cost)))
        else:
            for seed in [5, 17]:
                for cost in [22, 25]:
                    case_rows.append((scn, seed, float(cost)))

    total = len(reps_grid) * len(case_rows)
    print(f"Phase35 bootstrap stability audit: label={args.label} reps_grid={reps_grid} cases={total}")

    results = []
    i = 0
    for reps in reps_grid:
        for scn, seed, cost in case_rows:
            i += 1
            print(f"  audit [{i}/{total}] reps={reps} / {scn['name']} / s{seed} / c{int(cost)}", flush=True)
            results.append(run_case(CURRENT_BEST, scn, seed, cost, reps, args.label))

    detail = pd.DataFrame(results)
    rank = summarize(detail)

    detail_path = ARTIFACT_ROOT / f"main4_phase35_{args.label}_detail.csv"
    rank_path = ARTIFACT_ROOT / f"main4_phase35_{args.label}_rank.csv"
    summary_path = ARTIFACT_ROOT / f"main4_phase35_{args.label}_summary.json"

    detail.to_csv(detail_path, index=False)
    rank.to_csv(rank_path, index=False)

    decision = "stable_enough_for_high_rep_governance" if int(rank.iloc[-1]["breach_count"]) == 0 else "insufficient_resolution_or_structural_instability"
    summary = {
        "phase": "Phase35",
        "title": "Bootstrap Stability Audit",
        "label": args.label,
        "variant": CURRENT_BEST.name,
        "reps_grid": reps_grid,
        "decision": decision,
        "observed": rank.to_dict("records"),
        "artifacts": {
            "detail": str(detail_path),
            "rank": str(rank_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "decision": decision}))


if __name__ == "__main__":
    main()