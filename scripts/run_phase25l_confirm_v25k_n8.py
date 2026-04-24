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
RUN_ROOT = ARTIFACT_ROOT / "main4_phase25l_runs"

FIXED_PATH_THRESHOLD = -0.007
MIN_STEP = 0.02

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
    {"name": "stress_b14_s7070", "block_size": 14, "path_seed": 7070},
    {"name": "stress_b30_s10010", "block_size": 30, "path_seed": 10010},
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


WINNER = Variant(
    name="v25k_n8",
    ridge_l2=0.0210,
    action_smoothing=1.1425,
    scale_turnover_penalty=3.00,
    no_trade_band=0.0214,
    min_signal_scale=-0.651,
    max_signal_scale=0.651,
)


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

    return {
        "variant": WINNER.name,
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


def summarize(detail: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    srows: list[dict[str, Any]] = []
    for scenario, g in detail.groupby("scenario"):
        mean_test_relative = float(g["mean_test_relative"].mean())
        worst_robust_min = float(g["robust_min_test_relative".strip()].min())
        mean_turnover = float(g["mean_turnover"].mean())
        mean_executed_step_rate = float(g["mean_executed_step_rate"].mean())
        worst_daily = float(g["worst_daily_relative_return"].min())
        worst_dd = float(g["worst_relative_drawdown"].max())
        mean_path = float(g["path_bootstrap_robust_min_p05"].mean())
        min_case = float(g["path_bootstrap_robust_min_p05"].min())

        pass_non_path = bool(
            mean_test_relative >= 0.0
            and worst_robust_min >= 0.0
            and mean_turnover <= 0.0015
            and mean_executed_step_rate >= MIN_STEP
            and worst_daily >= -0.010
            and worst_dd <= 0.030
        )

        srows.append(
            {
                "variant": WINNER.name,
                "scenario": scenario,
                "mean_test_relative": mean_test_relative,
                "worst_robust_min": worst_robust_min,
                "mean_turnover": mean_turnover,
                "mean_executed_step_rate": mean_executed_step_rate,
                "worst_daily_relative_return": worst_daily,
                "worst_relative_drawdown": worst_dd,
                "mean_path": mean_path,
                "min_case_path": min_case,
                "mean_path_slack": mean_path - FIXED_PATH_THRESHOLD,
                "min_case_slack": min_case - FIXED_PATH_THRESHOLD,
                "pass_non_path": pass_non_path,
                "pass_case_floor": bool(min_case >= FIXED_PATH_THRESHOLD),
            }
        )

    summary = pd.DataFrame(srows).sort_values("scenario")
    breaches = detail[detail["path_bootstrap_robust_min_p05"] < FIXED_PATH_THRESHOLD].sort_values(
        "path_bootstrap_robust_min_p05"
    )

    note = {
        "candidate_variant": WINNER.name,
        "fixed_path_threshold": FIXED_PATH_THRESHOLD,
        "min_step": MIN_STEP,
        "scenarios": SCENARIOS,
        "seeds": SEEDS,
        "frictions": FRICTIONS,
        "all_non_path": bool(summary["pass_non_path"].all()),
        "all_case_floor": bool(summary["pass_case_floor"].all()),
        "min_case_slack": float(summary["min_case_slack"].min()),
        "min_mean_path_slack": float(summary["mean_path_slack"].min()),
        "mean_mean_path": float(summary["mean_path"].mean()),
        "mean_test_relative": float(summary["mean_test_relative"].mean()),
        "mean_turnover": float(summary["mean_turnover"].mean()),
        "mean_executed_step_rate": float(summary["mean_executed_step_rate"].mean()),
        "breach_count": int(len(breaches)),
        "worst_case_row": (
            breaches.head(1)[["scenario", "seed", "transaction_cost_bps", "path_bootstrap_robust_min_p05"]]
            .to_dict("records")[0]
            if len(breaches)
            else detail.sort_values("path_bootstrap_robust_min_p05")
            .head(1)[["scenario", "seed", "transaction_cost_bps", "path_bootstrap_robust_min_p05"]]
            .to_dict("records")[0]
        ),
        "decision": "PASS" if bool(summary["pass_non_path"].all() and summary["pass_case_floor"].all()) else "FAIL",
    }

    return summary, note, breaches


def main() -> int:
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"Missing virtualenv python: {PYTHON_BIN}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            for cost in FRICTIONS:
                rows.append(run_case(scenario, seed, cost))

    detail = pd.DataFrame(rows)
    summary, note, breaches = summarize(detail)

    out_detail = ARTIFACT_ROOT / "main4_phase25l_confirm_detail.csv"
    out_summary_csv = ARTIFACT_ROOT / "main4_phase25l_confirm_summary.csv"
    out_summary_json = ARTIFACT_ROOT / "main4_phase25l_confirm_summary.json"
    out_note_json = ARTIFACT_ROOT / "main4_phase25l_confirm_note.json"
    out_breach_csv = ARTIFACT_ROOT / "main4_phase25l_confirm_breaches.csv"

    detail.sort_values(["scenario", "transaction_cost_bps", "seed"]).to_csv(out_detail, index=False)
    summary.to_csv(out_summary_csv, index=False)
    out_summary_json.write_text(json.dumps({"rows": summary.to_dict("records")}, indent=2))
    out_note_json.write_text(json.dumps(note, indent=2))
    breaches.to_csv(out_breach_csv, index=False)

    print("WROTE", out_detail)
    print("WROTE", out_summary_csv)
    print("WROTE", out_summary_json)
    print("WROTE", out_note_json)
    print("WROTE", out_breach_csv)
    print(json.dumps(note, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
