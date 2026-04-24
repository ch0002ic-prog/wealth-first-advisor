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
RUN_ROOT = ARTIFACT_ROOT / "main4_phase25m_runs"

FIXED_PATH_THRESHOLD = -0.007
MIN_STEP = 0.02

SEEDS = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
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
    Variant("v25k_n8", 0.0210, 1.1425, 3.00, 0.0214, -0.651, 0.651),
    Variant("v25k_ref", 0.0209, 1.1450, 3.00, 0.0214, -0.651, 0.651),
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
        str(v.ridge_l2),
        "--target-mode",
        "tanh_return",
        "--scale-turnover-penalty",
        str(v.scale_turnover_penalty),
        "--min-signal-scale",
        str(v.min_signal_scale),
        "--max-signal-scale",
        str(v.max_signal_scale),
        "--min-spy-weight",
        "0.80",
        "--max-spy-weight",
        "1.05",
        "--initial-spy-weight",
        "1.0",
        "--action-smoothing",
        str(v.action_smoothing),
        "--no-trade-band",
        str(v.no_trade_band),
        "--min-robust-min-relative",
        "0.0",
        "--min-active-fraction",
        "0.01",
        "--path-bootstrap-reps",
        "300",
        "--path-bootstrap-block-size",
        str(scenario["block_size"]),
        "--path-bootstrap-seed",
        str(scenario["path_seed"] + seed),
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
        "worst_daily_relative_return": float(
            fold.get("test_worst_daily_relative_return", pd.Series([0.0])).min()
        ),
        "worst_relative_drawdown": float(
            fold.get("test_max_relative_drawdown", pd.Series([0.0])).max()
        ),
        "path_bootstrap_robust_min_p05": float(
            m.get("path_bootstrap_robust_min_test_relative_p05", float("nan"))
        ),
    }


def pass_non_path(g: pd.DataFrame) -> bool:
    return bool(
        float(g["mean_test_relative"].mean()) >= 0.0
        and float(g["robust_min_test_relative"].min()) >= 0.0
        and float(g["mean_turnover"].mean()) <= 0.0015
        and float(g["mean_executed_step_rate"].mean()) >= MIN_STEP
        and float(g["worst_daily_relative_return"].min()) >= -0.010
        and float(g["worst_relative_drawdown"].max()) <= 0.030
    )


def main() -> int:
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"Missing virtualenv python: {PYTHON_BIN}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for v in VARIANTS:
        for sc in CORE_SCENARIOS:
            for seed in SEEDS:
                for cost in CORE_COSTS:
                    rows.append(run_case(v, sc, seed, cost))
        for sc in SENTINEL_SCENARIOS:
            for seed in SEEDS:
                for cost in SENTINEL_COSTS:
                    rows.append(run_case(v, sc, seed, cost))

    detail = pd.DataFrame(rows)

    scenario_rows: list[dict[str, Any]] = []
    for (variant, scenario), g in detail.groupby(["variant", "scenario"]):
        min_case = float(g["path_bootstrap_robust_min_p05"].min())
        mean_path = float(g["path_bootstrap_robust_min_p05"].mean())
        scenario_rows.append(
            {
                "variant": variant,
                "scenario": scenario,
                "rows": int(len(g)),
                "mean_test_relative": float(g["mean_test_relative"].mean()),
                "worst_robust_min": float(g["robust_min_test_relative"].min()),
                "mean_turnover": float(g["mean_turnover"].mean()),
                "mean_executed_step_rate": float(g["mean_executed_step_rate"].mean()),
                "mean_path": mean_path,
                "min_case_path": min_case,
                "mean_path_slack": mean_path - FIXED_PATH_THRESHOLD,
                "min_case_slack": min_case - FIXED_PATH_THRESHOLD,
                "pass_non_path": pass_non_path(g),
                "pass_case_floor": bool(min_case >= FIXED_PATH_THRESHOLD),
            }
        )

    scenario_summary = pd.DataFrame(scenario_rows).sort_values(["variant", "scenario"])

    variant_rows: list[dict[str, Any]] = []
    for variant, g in scenario_summary.groupby("variant"):
        variant_rows.append(
            {
                "variant": variant,
                "scenarios": int(len(g)),
                "all_non_path": bool(g["pass_non_path"].all()),
                "all_case_floor": bool(g["pass_case_floor"].all()),
                "min_case_slack": float(g["min_case_slack"].min()),
                "min_mean_path_slack": float(g["mean_path_slack"].min()),
                "mean_mean_path": float(g["mean_path"].mean()),
                "mean_test_relative": float(g["mean_test_relative"].mean()),
                "mean_turnover": float(g["mean_turnover"].mean()),
                "mean_executed_step_rate": float(g["mean_executed_step_rate"].mean()),
                "breach_count": int(
                    (
                        detail[detail["variant"] == variant]["path_bootstrap_robust_min_p05"]
                        < FIXED_PATH_THRESHOLD
                    ).sum()
                ),
            }
        )

    variant_rank = pd.DataFrame(variant_rows).sort_values(
        ["all_case_floor", "min_case_slack", "mean_mean_path"],
        ascending=[False, False, False],
    )

    best_variant = variant_rank.iloc[0]["variant"]
    best_rows = detail[detail["variant"] == best_variant].sort_values("path_bootstrap_robust_min_p05")

    note = {
        "fixed_path_threshold": FIXED_PATH_THRESHOLD,
        "min_step": MIN_STEP,
        "seeds": SEEDS,
        "core_costs": CORE_COSTS,
        "sentinel_costs": SENTINEL_COSTS,
        "core_scenarios": CORE_SCENARIOS,
        "sentinel_scenarios": SENTINEL_SCENARIOS,
        "winner": best_variant,
        "variant_rank": variant_rank.to_dict("records"),
        "worst_row_winner": best_rows.head(1)[
            ["scenario", "seed", "transaction_cost_bps", "path_bootstrap_robust_min_p05"]
        ].to_dict("records")[0],
        "decision": (
            "PASS"
            if bool(
                variant_rank.iloc[0]["all_non_path"]
                and variant_rank.iloc[0]["all_case_floor"]
                and int(variant_rank.iloc[0]["breach_count"]) == 0
            )
            else "FAIL"
        ),
    }

    out_detail = ARTIFACT_ROOT / "main4_phase25m_tail_seedexp_detail.csv"
    out_scenario = ARTIFACT_ROOT / "main4_phase25m_tail_seedexp_scenario_summary.csv"
    out_variant = ARTIFACT_ROOT / "main4_phase25m_tail_seedexp_variant_rank.csv"
    out_note = ARTIFACT_ROOT / "main4_phase25m_tail_seedexp_note.json"

    detail.sort_values(["variant", "scenario", "transaction_cost_bps", "seed"]).to_csv(
        out_detail, index=False
    )
    scenario_summary.to_csv(out_scenario, index=False)
    variant_rank.to_csv(out_variant, index=False)
    out_note.write_text(json.dumps(note, indent=2))

    print("WROTE", out_detail)
    print("WROTE", out_scenario)
    print("WROTE", out_variant)
    print("WROTE", out_note)
    print(json.dumps(note, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
