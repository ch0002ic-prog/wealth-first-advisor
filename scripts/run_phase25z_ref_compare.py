#!/usr/bin/env python3
from __future__ import annotations

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
RUN_ROOT = ARTIFACT_ROOT / "main4_phase25z_refcompare_runs"

FIXED_PATH_THRESHOLD = -0.007
MIN_STEP = 0.02
PATH_BOOTSTRAP_REPS = 600

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
FULL_RANDOM_ROWS = [
    {"seed": 5, "cost": 22},
    {"seed": 5, "cost": 25},
    {"seed": 17, "cost": 22},
    {"seed": 17, "cost": 25},
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


VARIANTS = [
    Variant("v25z_ref", 0.0212, 1.1425, 3.05, 0.02145, -0.650, 0.650),
    Variant("v25z_ridge_hi", 0.0213, 1.1425, 3.05, 0.02145, -0.650, 0.650),
]


def run_case(v: Variant, scenario: dict[str, Any], seed: int, cost: float) -> dict[str, Any]:
    run_name = (
        f"{v.name}_{scenario['name']}_r{PATH_BOOTSTRAP_REPS}_c{int(cost)}_s{seed}_g055"
        f"_pbsz{scenario['block_size']}_pbs{scenario['path_seed'] + seed}"
    )
    run_dir = RUN_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PYTHON_BIN),
        "src/wealth_first/main4.py",
        "--gate", "055",
        "--gate-scale", "bps",
        "--n-folds", "5",
        "--seed", str(seed),
        "--output-dir", str(run_dir),
        "--transaction-cost-bps", str(cost),
        "--slippage-bps", str(cost),
        "--ridge-l2", str(v.ridge_l2),
        "--target-mode", "tanh_return",
        "--scale-turnover-penalty", str(v.scale_turnover_penalty),
        "--min-signal-scale", str(v.min_signal_scale),
        "--max-signal-scale", str(v.max_signal_scale),
        "--min-spy-weight", "0.80",
        "--max-spy-weight", "1.05",
        "--initial-spy-weight", "1.0",
        "--action-smoothing", str(v.action_smoothing),
        "--no-trade-band", str(v.no_trade_band),
        "--min-robust-min-relative", "0.0",
        "--min-active-fraction", "0.01",
        "--path-bootstrap-reps", str(PATH_BOOTSTRAP_REPS),
        "--path-bootstrap-block-size", str(scenario["block_size"]),
        "--path-bootstrap-seed", str(scenario["path_seed"] + seed),
    ]

    detailed_path = run_dir / f"main4_gate055_f5_s{seed}_detailed.json"
    summary_path = run_dir / f"main4_gate055_f5_s{seed}_summary.csv"
    res = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
    )
    if (not detailed_path.exists()) or (not summary_path.exists()):
        raise RuntimeError(f"Case failed {run_name}:\n{res.stderr}\n{res.stdout}")

    detailed = json.loads(detailed_path.read_text())
    fold = pd.read_csv(summary_path)
    m = detailed.get("summary_metrics", {})
    n_test = max(float(fold.get("n_test_samples", pd.Series([1.0])).mean()), 1e-8)
    p = float(m.get("path_bootstrap_robust_min_test_relative_p05", float("nan")))

    return {
        "variant": v.name,
        "group": scenario["group"],
        "scenario": scenario["name"],
        "seed": seed,
        "transaction_cost_bps": cost,
        "path_bootstrap_robust_min_p05": p,
        "case_slack": p - FIXED_PATH_THRESHOLD,
        "breach": bool(p < FIXED_PATH_THRESHOLD),
        "mean_test_relative": float(m.get("mean_test_relative_total_return", 0.0)),
        "robust_min_test_relative": float(m.get("robust_min_test_relative", 0.0)),
        "mean_turnover": float(m.get("mean_turnover", 0.0)),
        "mean_executed_step_rate": float(m.get("mean_test_executed_step_count", 0.0) / n_test),
        "worst_daily_relative_return": float(fold.get("test_worst_daily_relative_return", pd.Series([0.0])).min()),
        "worst_relative_drawdown": float(fold.get("test_max_relative_drawdown", pd.Series([0.0])).max()),
    }


def pass_non_path(df: pd.DataFrame) -> bool:
    return bool(
        float(df["mean_test_relative"].mean()) >= 0.0
        and float(df["robust_min_test_relative"].min()) >= 0.0
        and float(df["mean_turnover"].mean()) <= 0.0015
        and float(df["mean_executed_step_rate"].mean()) >= MIN_STEP
        and float(df["worst_daily_relative_return"].min()) >= -0.010
        and float(df["worst_relative_drawdown"].max()) <= 0.030
    )


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "all_non_path": pass_non_path(df),
        "breach_count": int(df["breach"].sum()),
        "min_case_slack": float(df["case_slack"].min()),
        "mean_case_slack": float(df["case_slack"].mean()),
        "mean_test_relative": float(df["mean_test_relative"].mean()),
        "mean_turnover": float(df["mean_turnover"].mean()),
        "mean_executed_step_rate": float(df["mean_executed_step_rate"].mean()),
    }


def main() -> int:
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"Missing virtualenv python: {PYTHON_BIN}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for v in VARIANTS:
        for sc in FULL_CURATED_SCENARIOS:
            for seed in FULL_CURATED_SEEDS:
                for cost in FULL_CURATED_COSTS:
                    rows.append(run_case(v, sc, seed, cost))
        for sc in FULL_RANDOM_SCENARIOS:
            for row in FULL_RANDOM_ROWS:
                rows.append(run_case(v, sc, row["seed"], row["cost"]))

    detail = pd.DataFrame(rows)

    rank_rows: list[dict[str, Any]] = []
    for variant, g in detail.groupby("variant"):
        rank_rows.append({"variant": variant, **summarize(g)})
    rank = pd.DataFrame(rank_rows).sort_values(
        ["breach_count", "all_non_path", "min_case_slack", "mean_case_slack", "mean_test_relative"],
        ascending=[True, False, False, False, False],
    )

    ref = rank[rank["variant"] == "v25z_ref"].iloc[0].to_dict()
    win = rank[rank["variant"] == "v25z_ridge_hi"].iloc[0].to_dict()
    delta = {
        "delta_min_case_slack": float(win["min_case_slack"] - ref["min_case_slack"]),
        "delta_mean_case_slack": float(win["mean_case_slack"] - ref["mean_case_slack"]),
        "delta_mean_test_relative": float(win["mean_test_relative"] - ref["mean_test_relative"]),
    }

    note = {
        "decision": "PASS" if (int(win["breach_count"]) == 0 and bool(win["all_non_path"])) else "FAIL",
        "reference": ref,
        "candidate": win,
        "delta_vs_reference": delta,
    }

    out_detail = ARTIFACT_ROOT / "main4_phase25z_refcompare_detail.csv"
    out_rank = ARTIFACT_ROOT / "main4_phase25z_refcompare_rank.csv"
    out_note = ARTIFACT_ROOT / "main4_phase25z_refcompare_note.json"

    detail.sort_values(["variant", "group", "scenario", "seed", "transaction_cost_bps"]).to_csv(out_detail, index=False)
    rank.to_csv(out_rank, index=False)
    out_note.write_text(json.dumps(note, indent=2))

    print("WROTE", out_detail)
    print("WROTE", out_rank)
    print("WROTE", out_note)
    print(json.dumps(note, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
