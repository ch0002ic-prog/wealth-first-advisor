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
RUN_ROOT = ARTIFACT_ROOT / "main4_phase25n_runs"

FIXED_PATH_THRESHOLD = -0.007
MIN_STEP = 0.02

SCREEN_SEEDS = [5, 11, 17, 23, 29]
FULL_SEEDS = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

SCREEN_COSTS = [22, 25]
CORE_COSTS = [18, 22, 25]
SENTINEL_COSTS = [22, 25]

CORE_SCENARIOS = [
    {"name": "deepcheck_b12_s8080", "block_size": 12, "path_seed": 8080},
    {"name": "altblock_b16_s4242", "block_size": 16, "path_seed": 4242},
    {"name": "stress_b14_s7070", "block_size": 14, "path_seed": 7070},
]

SENTINEL_SCENARIOS = [
    {"name": "baseline_b20_s4242", "block_size": 20, "path_seed": 4242},
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


VARIANTS = [
    Variant("v25n_ref_n8", 0.0210, 1.1425, 3.00, 0.0214, -0.651, 0.651),
    Variant("v25n_a", 0.0211, 1.1425, 3.02, 0.0214, -0.651, 0.651),
    Variant("v25n_b", 0.0209, 1.1425, 2.98, 0.0214, -0.651, 0.651),
    Variant("v25n_c", 0.0210, 1.1400, 3.03, 0.02145, -0.650, 0.650),
    Variant("v25n_d", 0.0210, 1.1450, 3.03, 0.02145, -0.650, 0.650),
    Variant("v25n_e", 0.0211, 1.1400, 3.05, 0.0215, -0.649, 0.649),
    Variant("v25n_f", 0.0209, 1.1400, 3.00, 0.02135, -0.652, 0.652),
    Variant("v25n_g", 0.0210, 1.1375, 3.06, 0.0215, -0.649, 0.649),
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


def pass_non_path(df: pd.DataFrame) -> bool:
    return bool(
        float(df["mean_test_relative"].mean()) >= 0.0
        and float(df["robust_min_test_relative"].min()) >= 0.0
        and float(df["mean_turnover"].mean()) <= 0.0015
        and float(df["mean_executed_step_rate"].mean()) >= MIN_STEP
        and float(df["worst_daily_relative_return"].min()) >= -0.010
        and float(df["worst_relative_drawdown"].max()) <= 0.030
    )


def summarize_variant(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "all_non_path": pass_non_path(df),
        "all_case_floor": bool(float(df["path_bootstrap_robust_min_p05"].min()) >= FIXED_PATH_THRESHOLD),
        "min_case_slack": float(df["path_bootstrap_robust_min_p05"].min() - FIXED_PATH_THRESHOLD),
        "mean_case_slack": float(df["path_bootstrap_robust_min_p05"].mean() - FIXED_PATH_THRESHOLD),
        "mean_test_relative": float(df["mean_test_relative"].mean()),
        "mean_turnover": float(df["mean_turnover"].mean()),
        "mean_executed_step_rate": float(df["mean_executed_step_rate"].mean()),
        "breach_count": int((df["path_bootstrap_robust_min_p05"] < FIXED_PATH_THRESHOLD).sum()),
    }


def run_screen() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for v in VARIANTS:
        for sc in CORE_SCENARIOS:
            for seed in SCREEN_SEEDS:
                for cost in SCREEN_COSTS:
                    rows.append(run_case(v, sc, seed, cost))

    detail = pd.DataFrame(rows)
    rank_rows: list[dict[str, Any]] = []
    for variant, g in detail.groupby("variant"):
        r = summarize_variant(g)
        rank_rows.append({"variant": variant, **r})

    rank = pd.DataFrame(rank_rows).sort_values(
        ["all_case_floor", "all_non_path", "min_case_slack", "mean_case_slack", "mean_test_relative"],
        ascending=[False, False, False, False, False],
    )
    return detail, rank


def run_full(top_variants: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    top_defs = [v for v in VARIANTS if v.name in set(top_variants)]

    rows: list[dict[str, Any]] = []
    for v in top_defs:
        for sc in CORE_SCENARIOS:
            for seed in FULL_SEEDS:
                for cost in CORE_COSTS:
                    rows.append(run_case(v, sc, seed, cost))
        for sc in SENTINEL_SCENARIOS:
            for seed in FULL_SEEDS:
                for cost in SENTINEL_COSTS:
                    rows.append(run_case(v, sc, seed, cost))

    detail = pd.DataFrame(rows)

    scenario_rows: list[dict[str, Any]] = []
    for (variant, scenario), g in detail.groupby(["variant", "scenario"]):
        scenario_rows.append(
            {
                "variant": variant,
                "scenario": scenario,
                "rows": int(len(g)),
                "pass_non_path": pass_non_path(g),
                "pass_case_floor": bool(float(g["path_bootstrap_robust_min_p05"].min()) >= FIXED_PATH_THRESHOLD),
                "min_case_path": float(g["path_bootstrap_robust_min_p05"].min()),
                "mean_path": float(g["path_bootstrap_robust_min_p05"].mean()),
                "min_case_slack": float(g["path_bootstrap_robust_min_p05"].min() - FIXED_PATH_THRESHOLD),
                "mean_case_slack": float(g["path_bootstrap_robust_min_p05"].mean() - FIXED_PATH_THRESHOLD),
                "mean_test_relative": float(g["mean_test_relative"].mean()),
                "mean_turnover": float(g["mean_turnover"].mean()),
                "mean_executed_step_rate": float(g["mean_executed_step_rate"].mean()),
            }
        )

    scenario_summary = pd.DataFrame(scenario_rows).sort_values(["variant", "scenario"])

    rank_rows: list[dict[str, Any]] = []
    for variant, g in detail.groupby("variant"):
        r = summarize_variant(g)
        rank_rows.append({"variant": variant, **r})

    rank = pd.DataFrame(rank_rows).sort_values(
        ["all_case_floor", "all_non_path", "min_case_slack", "mean_case_slack", "mean_test_relative"],
        ascending=[False, False, False, False, False],
    )

    return detail, scenario_summary, rank


def main() -> int:
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"Missing virtualenv python: {PYTHON_BIN}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    screen_detail, screen_rank = run_screen()
    top3 = screen_rank.head(3)["variant"].tolist()

    full_detail, full_scenario, full_rank = run_full(top3)

    best = full_rank.iloc[0].to_dict()
    best_name = best["variant"]
    best_rows = full_detail[full_detail["variant"] == best_name].sort_values("path_bootstrap_robust_min_p05")

    note = {
        "fixed_path_threshold": FIXED_PATH_THRESHOLD,
        "min_step": MIN_STEP,
        "screen_seeds": SCREEN_SEEDS,
        "full_seeds": FULL_SEEDS,
        "screen_costs": SCREEN_COSTS,
        "core_costs": CORE_COSTS,
        "sentinel_costs": SENTINEL_COSTS,
        "core_scenarios": CORE_SCENARIOS,
        "sentinel_scenarios": SENTINEL_SCENARIOS,
        "screen_top3": top3,
        "full_best": best,
        "full_any_feasible": bool((full_rank["all_case_floor"] & full_rank["all_non_path"]).any()),
        "worst_row_best": best_rows.head(1)[["scenario", "seed", "transaction_cost_bps", "path_bootstrap_robust_min_p05"]].to_dict("records")[0],
    }

    out_screen_detail = ARTIFACT_ROOT / "main4_phase25n_screen_detail.csv"
    out_screen_rank = ARTIFACT_ROOT / "main4_phase25n_screen_variant_rank.csv"
    out_full_detail = ARTIFACT_ROOT / "main4_phase25n_full_detail.csv"
    out_full_scenario = ARTIFACT_ROOT / "main4_phase25n_full_scenario_summary.csv"
    out_full_rank = ARTIFACT_ROOT / "main4_phase25n_full_variant_rank.csv"
    out_note = ARTIFACT_ROOT / "main4_phase25n_note.json"

    screen_detail.sort_values(["variant", "scenario", "transaction_cost_bps", "seed"]).to_csv(out_screen_detail, index=False)
    screen_rank.to_csv(out_screen_rank, index=False)
    full_detail.sort_values(["variant", "scenario", "transaction_cost_bps", "seed"]).to_csv(out_full_detail, index=False)
    full_scenario.to_csv(out_full_scenario, index=False)
    full_rank.to_csv(out_full_rank, index=False)
    out_note.write_text(json.dumps(note, indent=2))

    print("WROTE", out_screen_detail)
    print("WROTE", out_screen_rank)
    print("WROTE", out_full_detail)
    print("WROTE", out_full_scenario)
    print("WROTE", out_full_rank)
    print("WROTE", out_note)
    print(json.dumps(note, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
