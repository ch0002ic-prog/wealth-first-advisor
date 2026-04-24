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
RUN_ROOT = ARTIFACT_ROOT / "main4_phase25i_runs"

FIXED_PATH_THRESHOLD = -0.007
MIN_STEP = 0.02

SEEDS = [5, 17, 29]
FRICTIONS = [12, 15, 18, 22, 25]

STRESS_SCENARIOS = [
    {"name": "baseline_b20_s4242", "block_size": 20, "path_seed": 4242},
    {"name": "altblock_b16_s4242", "block_size": 16, "path_seed": 4242},
    {"name": "deepcheck_b12_s8080", "block_size": 12, "path_seed": 8080},
]

FULL_SCENARIOS = [
    {"name": "baseline_b20_s4242", "block_size": 20, "path_seed": 4242},
    {"name": "altseed_b20_s5252", "block_size": 20, "path_seed": 5252},
    {"name": "altblock_b16_s4242", "block_size": 16, "path_seed": 4242},
    {"name": "altblock_b24_s4242", "block_size": 24, "path_seed": 4242},
    {"name": "altboth_b24_s6060", "block_size": 24, "path_seed": 6060},
    {"name": "deepcheck_b12_s8080", "block_size": 12, "path_seed": 8080},
    {"name": "deepcheck_b28_s9090", "block_size": 28, "path_seed": 9090},
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
    Variant("v25i_ref", 0.0210, 1.1400, 3.00, 0.0215, -0.650, 0.650),
    Variant("v25i_tight_1", 0.0215, 1.1450, 3.10, 0.0218, -0.645, 0.645),
    Variant("v25i_tight_2", 0.0220, 1.1500, 3.20, 0.0220, -0.640, 0.640),
    Variant("v25i_tight_3", 0.0220, 1.1600, 3.30, 0.0225, -0.635, 0.635),
    Variant("v25i_tight_4", 0.0218, 1.1550, 3.25, 0.0222, -0.638, 0.638),
    Variant("v25i_asym_t1", 0.0215, 1.1500, 3.20, 0.0220, -0.635, 0.640),
    Variant("v25i_asym_t2", 0.0218, 1.1525, 3.25, 0.0222, -0.632, 0.638),
    Variant("v25i_bal_1", 0.0212, 1.1450, 3.10, 0.0216, -0.646, 0.646),
    Variant("v25i_bal_2", 0.0214, 1.1475, 3.15, 0.0217, -0.644, 0.644),
]


def run_case(v: Variant, scenario: dict[str, Any], seed: int, cost: float) -> dict[str, Any]:
    run_name = (
        f"{v.name}_{scenario['name']}_c{int(cost)}_s{seed}_g055"
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
        "variant": v.name,
        "scenario": scenario["name"],
        "path_bootstrap_block_size": scenario["block_size"],
        "path_bootstrap_seed": scenario["path_seed"],
        "seed": seed,
        "transaction_cost_bps": cost,
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


def summarize_variant_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, scenario), g in df.groupby(["variant", "scenario"]):
        mean_test_relative = float(g["mean_test_relative"].mean())
        worst_robust_min = float(g["robust_min_test_relative"].min())
        mean_turnover = float(g["mean_turnover"].mean())
        mean_executed_step_rate = float(g["mean_executed_step_rate"].mean())
        worst_daily_relative_return = float(g["worst_daily_relative_return"].min())
        worst_relative_drawdown = float(g["worst_relative_drawdown"].max())
        mean_path = float(g["path_bootstrap_robust_min_p05"].mean())
        min_case = float(g["path_bootstrap_robust_min_p05"].min())

        pass_non = (
            mean_test_relative >= 0.0
            and worst_robust_min >= 0.0
            and mean_turnover <= 0.0015
            and mean_executed_step_rate >= MIN_STEP
            and worst_daily_relative_return >= -0.010
            and worst_relative_drawdown <= 0.030
        )

        rows.append(
            {
                "variant": variant,
                "scenario": scenario,
                "mean_test_relative": mean_test_relative,
                "worst_robust_min": worst_robust_min,
                "mean_turnover": mean_turnover,
                "mean_executed_step_rate": mean_executed_step_rate,
                "mean_path": mean_path,
                "min_case_path": min_case,
                "mean_path_slack": mean_path - FIXED_PATH_THRESHOLD,
                "min_case_slack": min_case - FIXED_PATH_THRESHOLD,
                "pass_non_path": bool(pass_non),
                "pass_case_floor": bool(min_case >= FIXED_PATH_THRESHOLD),
            }
        )

    return pd.DataFrame(rows)


def rank_variants(scenario_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        scenario_summary.groupby("variant", as_index=False)
        .agg(
            scenarios=("scenario", "count"),
            all_non_path=("pass_non_path", "all"),
            all_case_floor=("pass_case_floor", "all"),
            min_case_slack=("min_case_slack", "min"),
            min_mean_path_slack=("mean_path_slack", "min"),
            mean_mean_path=("mean_path", "mean"),
            mean_test_relative=("mean_test_relative", "mean"),
            mean_turnover=("mean_turnover", "mean"),
            mean_executed_step_rate=("mean_executed_step_rate", "mean"),
        )
        .sort_values(["all_case_floor", "min_case_slack", "mean_mean_path"], ascending=[False, False, False])
    )


def run_grid(variants: list[Variant], scenarios: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for v in variants:
        for sc in scenarios:
            for s in SEEDS:
                for c in FRICTIONS:
                    rows.append(run_case(v, sc, s, c))
    return pd.DataFrame(rows)


def main() -> int:
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"Missing virtualenv python: {PYTHON_BIN}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    stress_detail = run_grid(VARIANTS, STRESS_SCENARIOS)
    stress_summary = summarize_variant_scenarios(stress_detail)
    stress_rank = rank_variants(stress_summary)

    top_variants = stress_rank.head(3)["variant"].tolist()
    top_defs = [v for v in VARIANTS if v.name in top_variants]

    full_detail = run_grid(top_defs, FULL_SCENARIOS)
    full_summary = summarize_variant_scenarios(full_detail)
    full_rank = rank_variants(full_summary)

    note = {
        "fixed_path_threshold": FIXED_PATH_THRESHOLD,
        "stress_scenarios": [x["name"] for x in STRESS_SCENARIOS],
        "full_scenarios": [x["name"] for x in FULL_SCENARIOS],
        "stress_top_variants": top_variants,
        "stress_best": stress_rank.iloc[0].to_dict(),
        "full_best": full_rank.iloc[0].to_dict(),
        "full_any_all_case_floor_pass": bool(full_rank["all_case_floor"].any()),
    }

    (ARTIFACT_ROOT / "main4_phase25i_stress_detail.csv").write_text(stress_detail.to_csv(index=False))
    (ARTIFACT_ROOT / "main4_phase25i_stress_scenario_summary.csv").write_text(stress_summary.to_csv(index=False))
    (ARTIFACT_ROOT / "main4_phase25i_stress_variant_rank.csv").write_text(stress_rank.to_csv(index=False))
    (ARTIFACT_ROOT / "main4_phase25i_full_detail.csv").write_text(full_detail.to_csv(index=False))
    (ARTIFACT_ROOT / "main4_phase25i_full_scenario_summary.csv").write_text(full_summary.to_csv(index=False))
    (ARTIFACT_ROOT / "main4_phase25i_full_variant_rank.csv").write_text(full_rank.to_csv(index=False))
    (ARTIFACT_ROOT / "main4_phase25i_note.json").write_text(json.dumps(note, indent=2))

    print("WROTE", ARTIFACT_ROOT / "main4_phase25i_stress_detail.csv")
    print("WROTE", ARTIFACT_ROOT / "main4_phase25i_stress_scenario_summary.csv")
    print("WROTE", ARTIFACT_ROOT / "main4_phase25i_stress_variant_rank.csv")
    print("WROTE", ARTIFACT_ROOT / "main4_phase25i_full_detail.csv")
    print("WROTE", ARTIFACT_ROOT / "main4_phase25i_full_scenario_summary.csv")
    print("WROTE", ARTIFACT_ROOT / "main4_phase25i_full_variant_rank.csv")
    print("WROTE", ARTIFACT_ROOT / "main4_phase25i_note.json")
    print(json.dumps(note, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
