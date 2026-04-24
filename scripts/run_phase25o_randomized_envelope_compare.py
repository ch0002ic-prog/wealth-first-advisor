#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
RUN_ROOT = ARTIFACT_ROOT / "main4_phase25o_runs"

FIXED_PATH_THRESHOLD = -0.007
MIN_STEP = 0.02

RNG_SEED = 2501
RANDOM_SCENARIO_COUNT = 6
SEEDS = [5, 11, 17, 29]
CORE_COSTS = [18, 22, 25]
SENTINEL_COSTS = [22, 25]


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
    Variant("v25o_n8", 0.0210, 1.1425, 3.00, 0.0214, -0.651, 0.651),
    Variant("v25o_nA", 0.0211, 1.1425, 3.02, 0.0214, -0.651, 0.651),
    Variant("v25o_ref", 0.0209, 1.1450, 3.00, 0.0214, -0.651, 0.651),
]

SENTINEL_SCENARIOS = [
    {"name": "baseline_b20_s4242", "block_size": 20, "path_seed": 4242, "group": "sentinel"},
    {"name": "altblock_b24_s4242", "block_size": 24, "path_seed": 4242, "group": "sentinel"},
    {"name": "stress_b30_s10010", "block_size": 30, "path_seed": 10010, "group": "sentinel"},
]


def make_random_scenarios() -> list[dict[str, Any]]:
    rng = random.Random(RNG_SEED)
    seen: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []
    while len(rows) < RANDOM_SCENARIO_COUNT:
        block = rng.randint(10, 18)
        pseed = rng.randint(5000, 12000)
        key = (block, pseed)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "name": f"rand_b{block}_p{pseed}",
                "block_size": block,
                "path_seed": pseed,
                "group": "random_shortblock",
            }
        )
    return rows


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
        "scenario_group": scenario["group"],
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


def main() -> int:
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"Missing virtualenv python: {PYTHON_BIN}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    random_scenarios = make_random_scenarios()
    scenarios = random_scenarios + SENTINEL_SCENARIOS

    rows: list[dict[str, Any]] = []
    for v in VARIANTS:
        for sc in scenarios:
            costs = CORE_COSTS if sc["group"] == "random_shortblock" else SENTINEL_COSTS
            for seed in SEEDS:
                for cost in costs:
                    rows.append(run_case(v, sc, seed, cost))

    detail = pd.DataFrame(rows)

    scenario_rows: list[dict[str, Any]] = []
    for (variant, scenario, group), g in detail.groupby(["variant", "scenario", "scenario_group"]):
        min_case = float(g["path_bootstrap_robust_min_p05"].min())
        scenario_rows.append(
            {
                "variant": variant,
                "scenario": scenario,
                "scenario_group": group,
                "rows": int(len(g)),
                "pass_non_path": pass_non_path(g),
                "pass_case_floor": bool(min_case >= FIXED_PATH_THRESHOLD),
                "min_case_path": min_case,
                "mean_path": float(g["path_bootstrap_robust_min_p05"].mean()),
                "min_case_slack": float(min_case - FIXED_PATH_THRESHOLD),
                "mean_case_slack": float(g["path_bootstrap_robust_min_p05"].mean() - FIXED_PATH_THRESHOLD),
                "mean_test_relative": float(g["mean_test_relative"].mean()),
                "mean_turnover": float(g["mean_turnover"].mean()),
                "mean_executed_step_rate": float(g["mean_executed_step_rate"].mean()),
            }
        )

    scenario_summary = pd.DataFrame(scenario_rows).sort_values(["variant", "scenario"])

    group_rows: list[dict[str, Any]] = []
    for (variant, group), g in detail.groupby(["variant", "scenario_group"]):
        r = summarize_variant(g)
        group_rows.append({"variant": variant, "scenario_group": group, **r})
    group_summary = pd.DataFrame(group_rows).sort_values(["variant", "scenario_group"])

    rank_rows: list[dict[str, Any]] = []
    for variant, g in detail.groupby("variant"):
        r = summarize_variant(g)
        rank_rows.append({"variant": variant, **r})
    variant_rank = pd.DataFrame(rank_rows).sort_values(
        ["all_case_floor", "all_non_path", "min_case_slack", "mean_case_slack", "mean_test_relative"],
        ascending=[False, False, False, False, False],
    )

    winner = variant_rank.iloc[0]["variant"]
    winner_rows = detail[detail["variant"] == winner].sort_values("path_bootstrap_robust_min_p05")

    note = {
        "fixed_path_threshold": FIXED_PATH_THRESHOLD,
        "min_step": MIN_STEP,
        "rng_seed": RNG_SEED,
        "seeds": SEEDS,
        "core_costs": CORE_COSTS,
        "sentinel_costs": SENTINEL_COSTS,
        "random_scenarios": random_scenarios,
        "sentinel_scenarios": SENTINEL_SCENARIOS,
        "winner": winner,
        "variant_rank": variant_rank.to_dict("records"),
        "winner_worst_row": winner_rows.head(1)[
            ["scenario", "scenario_group", "seed", "transaction_cost_bps", "path_bootstrap_robust_min_p05"]
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

    out_detail = ARTIFACT_ROOT / "main4_phase25o_random_env_detail.csv"
    out_scenario = ARTIFACT_ROOT / "main4_phase25o_random_env_scenario_summary.csv"
    out_group = ARTIFACT_ROOT / "main4_phase25o_random_env_group_summary.csv"
    out_variant = ARTIFACT_ROOT / "main4_phase25o_random_env_variant_rank.csv"
    out_note = ARTIFACT_ROOT / "main4_phase25o_random_env_note.json"

    detail.sort_values(["variant", "scenario_group", "scenario", "transaction_cost_bps", "seed"]).to_csv(out_detail, index=False)
    scenario_summary.to_csv(out_scenario, index=False)
    group_summary.to_csv(out_group, index=False)
    variant_rank.to_csv(out_variant, index=False)
    out_note.write_text(json.dumps(note, indent=2))

    print("WROTE", out_detail)
    print("WROTE", out_scenario)
    print("WROTE", out_group)
    print("WROTE", out_variant)
    print("WROTE", out_note)
    print(json.dumps(note, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
