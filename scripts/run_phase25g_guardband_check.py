#!/usr/bin/env python3
from __future__ import annotations

import csv
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
RUN_ROOT = ARTIFACT_ROOT / "main4_phase25g_runs"


@dataclass(frozen=True)
class Variant:
    name: str
    ridge_l2: float
    action_smoothing: float
    scale_turnover_penalty: float
    no_trade_band: float
    min_signal_scale: float
    max_signal_scale: float


WINNER = Variant(
    name="v25e_looser_01",
    ridge_l2=0.021,
    action_smoothing=1.14,
    scale_turnover_penalty=3.0,
    no_trade_band=0.0215,
    min_signal_scale=-0.65,
    max_signal_scale=0.65,
)

SEEDS = [5, 17, 29]
FRICTIONS = [12, 15, 18, 22, 25]
SCENARIOS = [
    {"name": "baseline_b20_s4242", "block_size": 20, "path_seed": 4242},
    {"name": "altseed_b20_s5252", "block_size": 20, "path_seed": 5252},
    {"name": "altblock_b16_s4242", "block_size": 16, "path_seed": 4242},
    {"name": "altblock_b24_s4242", "block_size": 24, "path_seed": 4242},
    {"name": "altboth_b24_s6060", "block_size": 24, "path_seed": 6060},
    {"name": "deepcheck_b12_s8080", "block_size": 12, "path_seed": 8080},
    {"name": "deepcheck_b28_s9090", "block_size": 28, "path_seed": 9090},
]

POLICY = {
    "min_mean_executed_step_rate": 0.02,
    "configured_min_path_bootstrap_robust_min_p05": -0.006,
    "strict_path_bootstrap_gate_max_relaxation": 0.001,
}

GUARDBAND = 0.0002
FIXED_PATH_THRESHOLD = POLICY["configured_min_path_bootstrap_robust_min_p05"] - POLICY["strict_path_bootstrap_gate_max_relaxation"]
REQUIRED_PATH_LEVEL = FIXED_PATH_THRESHOLD + GUARDBAND


def run_case(scenario: dict[str, Any], seed: int, cost: float) -> dict[str, Any]:
    run_name = (
        f"{WINNER.name}_{scenario['name']}_c{int(cost)}_s{seed}_g055"
        f"_pbr300_pbsz{scenario['block_size']}_pbs{scenario['path_seed'] + seed}"
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
        "--ridge-l2", str(WINNER.ridge_l2),
        "--target-mode", "tanh_return",
        "--scale-turnover-penalty", str(WINNER.scale_turnover_penalty),
        "--min-signal-scale", str(WINNER.min_signal_scale),
        "--max-signal-scale", str(WINNER.max_signal_scale),
        "--min-spy-weight", "0.80",
        "--max-spy-weight", "1.05",
        "--initial-spy-weight", "1.0",
        "--action-smoothing", str(WINNER.action_smoothing),
        "--no-trade-band", str(WINNER.no_trade_band),
        "--min-robust-min-relative", "0.0",
        "--min-active-fraction", "0.01",
        "--path-bootstrap-reps", "300",
        "--path-bootstrap-block-size", str(scenario["block_size"]),
        "--path-bootstrap-seed", str(scenario["path_seed"] + seed),
    ]

    detailed_path = run_dir / f"main4_gate055_f5_s{seed}_detailed.json"
    summary_path = run_dir / f"main4_gate055_f5_s{seed}_summary.csv"
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, env={**os.environ, "PYTHONPATH": "src"}, capture_output=True, text=True)
    if (not detailed_path.exists()) or (not summary_path.exists()):
        raise RuntimeError(f"Case failed {run_name}:\n{res.stderr}\n{res.stdout}")

    detailed = json.loads(detailed_path.read_text())
    fold = pd.read_csv(summary_path)
    m = detailed.get("summary_metrics", {})

    return {
        "scenario": scenario["name"],
        "path_bootstrap_block_size": scenario["block_size"],
        "path_bootstrap_seed": scenario["path_seed"],
        "seed": seed,
        "transaction_cost_bps": cost,
        "slippage_bps": cost,
        "mean_test_relative": float(m.get("mean_test_relative_total_return", 0.0)),
        "robust_min_test_relative": float(m.get("robust_min_test_relative", 0.0)),
        "mean_turnover": float(m.get("mean_turnover", 0.0)),
        "mean_executed_step_rate": float(
            m.get("mean_test_executed_step_count", 0.0)
            / max(float(fold.get("n_test_samples", pd.Series([1.0])).mean()), 1e-8)
        ),
        "worst_daily_relative_return": float(fold.get("test_worst_daily_relative_return", pd.Series([0.0])).min()),
        "worst_relative_drawdown": float(fold.get("test_max_relative_drawdown", pd.Series([0.0])).max()),
        "path_bootstrap_robust_min_p05": float(m.get("path_bootstrap_robust_min_test_relative_p05", float("nan"))),
    }


def summarize_scenario(df: pd.DataFrame) -> dict[str, Any]:
    mean_test_relative = float(df["mean_test_relative"].mean())
    worst_robust_min = float(df["robust_min_test_relative"].min())
    mean_turnover = float(df["mean_turnover"].mean())
    mean_executed_step_rate = float(df["mean_executed_step_rate"].mean())
    worst_daily_relative_return = float(df["worst_daily_relative_return"].min())
    worst_relative_drawdown = float(df["worst_relative_drawdown"].max())

    mean_path_p05 = float(df["path_bootstrap_robust_min_p05"].mean())
    min_case_path_p05 = float(df["path_bootstrap_robust_min_p05"].min())

    pass_non_path = (
        mean_test_relative >= 0.0
        and worst_robust_min >= 0.0
        and mean_turnover <= 0.0015
        and mean_executed_step_rate >= POLICY["min_mean_executed_step_rate"]
        and worst_daily_relative_return >= -0.010
        and worst_relative_drawdown <= 0.030
    )

    mean_path_slack_vs_floor = mean_path_p05 - FIXED_PATH_THRESHOLD
    mean_path_slack_vs_guardband = mean_path_p05 - REQUIRED_PATH_LEVEL
    min_case_path_slack_vs_floor = min_case_path_p05 - FIXED_PATH_THRESHOLD
    min_case_path_slack_vs_guardband = min_case_path_p05 - REQUIRED_PATH_LEVEL

    mean_guardband_pass = mean_path_p05 >= REQUIRED_PATH_LEVEL
    min_case_floor_pass = min_case_path_p05 >= FIXED_PATH_THRESHOLD

    strict_guardband_eligible = bool(pass_non_path and mean_guardband_pass)

    return {
        "mean_test_relative": mean_test_relative,
        "worst_robust_min": worst_robust_min,
        "mean_turnover": mean_turnover,
        "mean_executed_step_rate": mean_executed_step_rate,
        "worst_daily_relative_return": worst_daily_relative_return,
        "worst_relative_drawdown": worst_relative_drawdown,
        "mean_path_bootstrap_robust_min_p05": mean_path_p05,
        "min_case_path_bootstrap_robust_min_p05": min_case_path_p05,
        "pass_non_path": pass_non_path,
        "activity_slack": mean_executed_step_rate - POLICY["min_mean_executed_step_rate"],
        "mean_path_slack_vs_floor": mean_path_slack_vs_floor,
        "mean_path_slack_vs_guardband": mean_path_slack_vs_guardband,
        "min_case_path_slack_vs_floor": min_case_path_slack_vs_floor,
        "min_case_path_slack_vs_guardband": min_case_path_slack_vs_guardband,
        "mean_guardband_pass": bool(mean_guardband_pass),
        "min_case_floor_pass": bool(min_case_floor_pass),
        "strict_guardband_eligible": strict_guardband_eligible,
    }


def main() -> int:
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"Missing virtualenv python: {PYTHON_BIN}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for s in SEEDS:
            for c in FRICTIONS:
                rows.append(run_case(scenario, s, c))

    detail = pd.DataFrame(rows)

    summary_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        name = scenario["name"]
        sub = detail[detail["scenario"] == name]
        summary_rows.append(
            {
                "scenario": name,
                "path_bootstrap_block_size": scenario["block_size"],
                "path_bootstrap_seed": scenario["path_seed"],
                **summarize_scenario(sub),
            }
        )

    s_df = pd.DataFrame(summary_rows)

    mean_guardband_all = bool(s_df["strict_guardband_eligible"].all())
    min_case_floor_all = bool(s_df["min_case_floor_pass"].all())

    note = {
        "candidate_variant": WINNER.name,
        "policy": POLICY,
        "fixed_path_threshold": FIXED_PATH_THRESHOLD,
        "guardband_required_margin": GUARDBAND,
        "guardband_required_path_level": REQUIRED_PATH_LEVEL,
        "scenario_count": int(len(SCENARIOS)),
        "strict_guardband_eligible_count": int(s_df["strict_guardband_eligible"].sum()),
        "all_scenarios_strict_guardband_eligible": mean_guardband_all,
        "all_scenarios_min_case_floor_pass": min_case_floor_all,
        "min_activity_slack": float(s_df["activity_slack"].min()),
        "min_mean_path_slack_vs_floor": float(s_df["mean_path_slack_vs_floor"].min()),
        "min_mean_path_slack_vs_guardband": float(s_df["mean_path_slack_vs_guardband"].min()),
        "min_case_path_slack_vs_floor": float(s_df["min_case_path_slack_vs_floor"].min()),
        "min_case_path_slack_vs_guardband": float(s_df["min_case_path_slack_vs_guardband"].min()),
        "decision_mean_guardband": "PASS" if mean_guardband_all else "FAIL",
        "decision_min_case_floor": "PASS" if min_case_floor_all else "FAIL",
        "decision_overall": "PASS" if (mean_guardband_all and min_case_floor_all) else "FAIL",
        # Backward-compatible legacy decision (mean-level criterion used in prior runs).
        "decision": "PASS" if mean_guardband_all else "FAIL",
    }

    out_detail = ARTIFACT_ROOT / "main4_phase25g_guardband_detail.csv"
    out_summary_csv = ARTIFACT_ROOT / "main4_phase25g_guardband_summary.csv"
    out_summary_json = ARTIFACT_ROOT / "main4_phase25g_guardband_summary.json"
    out_note_json = ARTIFACT_ROOT / "main4_phase25g_guardband_note.json"

    detail.sort_values(["scenario", "transaction_cost_bps", "seed"]).to_csv(out_detail, index=False)
    s_df.to_csv(out_summary_csv, index=False)
    out_summary_json.write_text(json.dumps({"rows": summary_rows}, indent=2))
    out_note_json.write_text(json.dumps(note, indent=2))

    print("WROTE", out_detail)
    print("WROTE", out_summary_csv)
    print("WROTE", out_summary_json)
    print("WROTE", out_note_json)
    print(json.dumps(note, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
