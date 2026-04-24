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
RUN_ROOT = ARTIFACT_ROOT / "main4_phase25k_runs"

FIXED_PATH_THRESHOLD = -0.007
MIN_STEP = 0.02

# Phase25j showed residual failures concentrated in high-cost b12/b16 cells.
CORE_CASES = [
    {"scenario": "deepcheck_b12_s8080", "block_size": 12, "path_seed": 8080, "seed": 5, "cost": 18},
    {"scenario": "deepcheck_b12_s8080", "block_size": 12, "path_seed": 8080, "seed": 5, "cost": 22},
    {"scenario": "deepcheck_b12_s8080", "block_size": 12, "path_seed": 8080, "seed": 5, "cost": 25},
    {"scenario": "deepcheck_b12_s8080", "block_size": 12, "path_seed": 8080, "seed": 17, "cost": 18},
    {"scenario": "deepcheck_b12_s8080", "block_size": 12, "path_seed": 8080, "seed": 17, "cost": 22},
    {"scenario": "deepcheck_b12_s8080", "block_size": 12, "path_seed": 8080, "seed": 17, "cost": 25},
    {"scenario": "altblock_b16_s4242", "block_size": 16, "path_seed": 4242, "seed": 5, "cost": 18},
    {"scenario": "altblock_b16_s4242", "block_size": 16, "path_seed": 4242, "seed": 5, "cost": 22},
    {"scenario": "altblock_b16_s4242", "block_size": 16, "path_seed": 4242, "seed": 5, "cost": 25},
    {"scenario": "altblock_b16_s4242", "block_size": 16, "path_seed": 4242, "seed": 29, "cost": 18},
    {"scenario": "altblock_b16_s4242", "block_size": 16, "path_seed": 4242, "seed": 29, "cost": 22},
    {"scenario": "altblock_b16_s4242", "block_size": 16, "path_seed": 4242, "seed": 29, "cost": 25},
]

# Sentinel rows to guard non-path behavior while optimizing tails.
SENTINEL_CASES = [
    {"scenario": "baseline_b20_s4242", "block_size": 20, "path_seed": 4242, "seed": 17, "cost": 22},
    {"scenario": "baseline_b20_s4242", "block_size": 20, "path_seed": 4242, "seed": 29, "cost": 25},
    {"scenario": "altseed_b20_s5252", "block_size": 20, "path_seed": 5252, "seed": 29, "cost": 22},
    {"scenario": "deepcheck_b28_s9090", "block_size": 28, "path_seed": 9090, "seed": 17, "cost": 22},
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
SEEDS = [5, 17, 29]
FRICTIONS = [12, 15, 18, 22, 25]


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
    Variant("v25k_ref", 0.0209, 1.1450, 3.00, 0.0214, -0.651, 0.651),
    Variant("v25k_n1", 0.0208, 1.1475, 3.03, 0.02145, -0.650, 0.650),
    Variant("v25k_n2", 0.0209, 1.1500, 3.06, 0.0215, -0.649, 0.649),
    Variant("v25k_n3", 0.0210, 1.1475, 3.05, 0.02145, -0.650, 0.650),
    Variant("v25k_n4", 0.0210, 1.1500, 3.08, 0.0215, -0.649, 0.649),
    Variant("v25k_n5", 0.0211, 1.1525, 3.10, 0.02155, -0.648, 0.648),
    Variant("v25k_n6", 0.0208, 1.1450, 3.02, 0.0214, -0.651, 0.651),
    Variant("v25k_n7", 0.0209, 1.1425, 2.98, 0.02135, -0.652, 0.652),
    Variant("v25k_n8", 0.0210, 1.1425, 3.00, 0.0214, -0.651, 0.651),
    Variant("v25k_n9", 0.0211, 1.1450, 3.04, 0.02145, -0.650, 0.650),
    Variant("v25k_n10", 0.0207, 1.1475, 3.00, 0.0214, -0.651, 0.651),
    Variant("v25k_n11", 0.0208, 1.1500, 3.04, 0.02145, -0.650, 0.650),
]


def run_case(v: Variant, scenario: str, block_size: int, path_seed: int, seed: int, cost: float) -> dict[str, Any]:
    run_name = (
        f"{v.name}_{scenario}_c{int(cost)}_s{seed}_g055"
        f"_pbr300_pbsz{block_size}_pbs{path_seed + seed}"
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
        "--path-bootstrap-block-size", str(block_size),
        "--path-bootstrap-seed", str(path_seed + seed),
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
        "scenario": scenario,
        "path_bootstrap_block_size": block_size,
        "path_bootstrap_seed": path_seed,
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


def run_screen() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for v in VARIANTS:
        for c in CORE_CASES + SENTINEL_CASES:
            rows.append(
                run_case(v, c["scenario"], c["block_size"], c["path_seed"], c["seed"], c["cost"])
            )

    detail = pd.DataFrame(rows)

    rank_rows: list[dict[str, Any]] = []
    sentinel_names = {x["scenario"] for x in SENTINEL_CASES}
    for name, g in detail.groupby("variant"):
        worst_path = float(g["path_bootstrap_robust_min_p05"].min())
        mean_path = float(g["path_bootstrap_robust_min_p05"].mean())
        min_case_slack = worst_path - FIXED_PATH_THRESHOLD

        s = g[g["scenario"].isin(sentinel_names)]
        sentinel_non_path_ok = bool(
            float(s["mean_executed_step_rate"].min()) >= MIN_STEP
            and float(s["robust_min_test_relative"].min()) >= 0.0
        )

        rank_rows.append(
            {
                "variant": name,
                "worst_path_p05": worst_path,
                "mean_path_p05": mean_path,
                "min_case_slack": min_case_slack,
                "sentinel_non_path_ok": sentinel_non_path_ok,
                "sentinel_activity_min": float(s["mean_executed_step_rate"].min()),
                "sentinel_robust_min": float(s["robust_min_test_relative"].min()),
            }
        )

    rank = pd.DataFrame(rank_rows).sort_values(
        ["sentinel_non_path_ok", "min_case_slack", "mean_path_p05"],
        ascending=[False, False, False],
    )
    return detail, rank


def run_full(top_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    top_defs = [v for v in VARIANTS if v.name in set(top_names)]

    rows: list[dict[str, Any]] = []
    for v in top_defs:
        for sc in FULL_SCENARIOS:
            for seed in SEEDS:
                for cost in FRICTIONS:
                    rows.append(run_case(v, sc["name"], sc["block_size"], sc["path_seed"], seed, cost))

    detail = pd.DataFrame(rows)

    srows: list[dict[str, Any]] = []
    for (variant, scenario), g in detail.groupby(["variant", "scenario"]):
        mean_test_relative = float(g["mean_test_relative"].mean())
        worst_robust_min = float(g["robust_min_test_relative"].min())
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
                "pass_non_path": pass_non_path,
                "pass_case_floor": bool(min_case >= FIXED_PATH_THRESHOLD),
            }
        )

    summary = pd.DataFrame(srows)
    rank = (
        summary.groupby("variant", as_index=False)
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

    return detail, summary, rank


def main() -> int:
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"Missing virtualenv python: {PYTHON_BIN}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    screen_detail, screen_rank = run_screen()
    top = screen_rank.head(3)["variant"].tolist()
    full_detail, full_summary, full_rank = run_full(top)

    note = {
        "fixed_path_threshold": FIXED_PATH_THRESHOLD,
        "core_case_count": len(CORE_CASES),
        "sentinel_case_count": len(SENTINEL_CASES),
        "screen_top_variants": top,
        "screen_best": screen_rank.iloc[0].to_dict(),
        "full_best": full_rank.iloc[0].to_dict(),
        "full_any_all_case_floor_pass": bool(full_rank["all_case_floor"].any()),
    }

    (ARTIFACT_ROOT / "main4_phase25k_screen_detail.csv").write_text(screen_detail.to_csv(index=False))
    (ARTIFACT_ROOT / "main4_phase25k_screen_rank.csv").write_text(screen_rank.to_csv(index=False))
    (ARTIFACT_ROOT / "main4_phase25k_full_detail.csv").write_text(full_detail.to_csv(index=False))
    (ARTIFACT_ROOT / "main4_phase25k_full_scenario_summary.csv").write_text(full_summary.to_csv(index=False))
    (ARTIFACT_ROOT / "main4_phase25k_full_variant_rank.csv").write_text(full_rank.to_csv(index=False))
    (ARTIFACT_ROOT / "main4_phase25k_note.json").write_text(json.dumps(note, indent=2))

    print("WROTE", ARTIFACT_ROOT / "main4_phase25k_screen_detail.csv")
    print("WROTE", ARTIFACT_ROOT / "main4_phase25k_screen_rank.csv")
    print("WROTE", ARTIFACT_ROOT / "main4_phase25k_full_detail.csv")
    print("WROTE", ARTIFACT_ROOT / "main4_phase25k_full_scenario_summary.csv")
    print("WROTE", ARTIFACT_ROOT / "main4_phase25k_full_variant_rank.csv")
    print("WROTE", ARTIFACT_ROOT / "main4_phase25k_note.json")
    print(json.dumps(note, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
