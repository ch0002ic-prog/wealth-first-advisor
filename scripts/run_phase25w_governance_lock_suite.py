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
RUN_ROOT = ARTIFACT_ROOT / "main4_phase25w_runs"

FIXED_PATH_THRESHOLD = -0.007
MIN_STEP = 0.02
REPS_GRID = [600, 900]

CURATED_SCENARIOS = [
    {"name": "baseline_b20_s4242", "block_size": 20, "path_seed": 4242, "group": "curated"},
    {"name": "altseed_b20_s5252", "block_size": 20, "path_seed": 5252, "group": "curated"},
    {"name": "altblock_b16_s4242", "block_size": 16, "path_seed": 4242, "group": "curated"},
    {"name": "altblock_b24_s4242", "block_size": 24, "path_seed": 4242, "group": "curated"},
    {"name": "deepcheck_b12_s8080", "block_size": 12, "path_seed": 8080, "group": "curated"},
    {"name": "deepcheck_b28_s9090", "block_size": 28, "path_seed": 9090, "group": "curated"},
    {"name": "stress_b14_s7070", "block_size": 14, "path_seed": 7070, "group": "curated"},
]
CURATED_SEEDS = [5, 17, 29]
CURATED_COSTS = [18, 22, 25]

RANDOM_TRAIN_SEED = 5601
RANDOM_HOLDOUT_SEED = 6713
RANDOM_SCENARIO_COUNT = 6
RANDOM_TEST_ROWS = [
    {"seed": 5, "cost": 25},
    {"seed": 17, "cost": 25},
    {"seed": 29, "cost": 25},
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


# Candidate from promoted strict line and best candidate from reps governance sweeps.
VARIANTS = [
    Variant("v25w_n8", 0.0210, 1.1425, 3.00, 0.0214, -0.651, 0.651),
    Variant("v25w_pA", 0.0212, 1.1425, 3.05, 0.02145, -0.650, 0.650),
]


def make_random_scenarios(seed: int, envelope: str) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    seen: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []
    while len(rows) < RANDOM_SCENARIO_COUNT:
        block = rng.randint(10, 18)
        pseed = rng.randint(5000, 15000)
        key = (block, pseed)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "name": f"{envelope}_b{block}_p{pseed}",
                "block_size": block,
                "path_seed": pseed,
                "group": envelope,
            }
        )
    return rows


def run_case(v: Variant, scenario: dict[str, Any], seed: int, cost: float, reps: int) -> dict[str, Any]:
    run_name = (
        f"{v.name}_{scenario['name']}_r{reps}_c{int(cost)}_s{seed}_g055"
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
        str(reps),
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

    n_test = max(float(fold.get("n_test_samples", pd.Series([1.0])).mean()), 1e-8)
    p = float(m.get("path_bootstrap_robust_min_test_relative_p05", float("nan")))

    return {
        "variant": v.name,
        "path_bootstrap_reps": reps,
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


def main() -> int:
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"Missing virtualenv python: {PYTHON_BIN}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    random_train = make_random_scenarios(RANDOM_TRAIN_SEED, "random_train")
    random_holdout = make_random_scenarios(RANDOM_HOLDOUT_SEED, "random_holdout")

    rows: list[dict[str, Any]] = []
    for v in VARIANTS:
        for reps in REPS_GRID:
            for sc in CURATED_SCENARIOS:
                for seed in CURATED_SEEDS:
                    for cost in CURATED_COSTS:
                        rows.append(run_case(v, sc, seed, cost, reps))
            for sc in random_train + random_holdout:
                for rc in RANDOM_TEST_ROWS:
                    rows.append(run_case(v, sc, rc["seed"], rc["cost"], reps))

    detail = pd.DataFrame(rows)

    scenario_rows: list[dict[str, Any]] = []
    for (variant, reps, group, scenario), g in detail.groupby(["variant", "path_bootstrap_reps", "group", "scenario"]):
        scenario_rows.append(
            {
                "variant": variant,
                "path_bootstrap_reps": int(reps),
                "group": group,
                "scenario": scenario,
                "rows": int(len(g)),
                "breach_count": int(g["breach"].sum()),
                "breach_rate": float(g["breach"].mean()),
                "min_case_slack": float(g["case_slack"].min()),
                "mean_case_slack": float(g["case_slack"].mean()),
                "pass_non_path": pass_non_path(g),
            }
        )
    scenario_summary = pd.DataFrame(scenario_rows).sort_values(["variant", "path_bootstrap_reps", "group", "scenario"])

    rank_rows: list[dict[str, Any]] = []
    for (variant, reps), g in detail.groupby(["variant", "path_bootstrap_reps"]):
        gc = g[g["group"] == "curated"]
        gt = g[g["group"] == "random_train"]
        gh = g[g["group"] == "random_holdout"]
        rank_rows.append(
            {
                "variant": variant,
                "path_bootstrap_reps": int(reps),
                "rows": int(len(g)),
                "all_non_path": pass_non_path(g),
                "curated_breach_count": int(gc["breach"].sum()),
                "train_breach_count": int(gt["breach"].sum()),
                "holdout_breach_count": int(gh["breach"].sum()),
                "total_breach_count": int(g["breach"].sum()),
                "min_case_slack": float(g["case_slack"].min()),
                "mean_case_slack": float(g["case_slack"].mean()),
                "mean_test_relative": float(g["mean_test_relative"].mean()),
                "mean_turnover": float(g["mean_turnover"].mean()),
                "mean_executed_step_rate": float(g["mean_executed_step_rate"].mean()),
            }
        )

    rank = pd.DataFrame(rank_rows).sort_values(
        [
            "all_non_path",
            "curated_breach_count",
            "train_breach_count",
            "holdout_breach_count",
            "path_bootstrap_reps",
            "min_case_slack",
            "mean_case_slack",
        ],
        ascending=[False, True, True, True, True, False, False],
    )

    safe = rank[
        (rank["all_non_path"])
        & (rank["curated_breach_count"] == 0)
        & (rank["train_breach_count"] == 0)
        & (rank["holdout_breach_count"] == 0)
    ]

    if len(safe) > 0:
        recommended = safe.iloc[0].to_dict()
        decision = "PASS"
    else:
        recommended = None
        decision = "FAIL"

    if recommended is not None:
        best_slice = detail[
            (detail["variant"] == recommended["variant"])
            & (detail["path_bootstrap_reps"] == int(recommended["path_bootstrap_reps"]))
        ]
        worst_row = best_slice.sort_values("case_slack").head(1)[
            ["group", "scenario", "seed", "transaction_cost_bps", "path_bootstrap_robust_min_p05", "case_slack"]
        ].to_dict("records")[0]
    else:
        worst_row = None

    note = {
        "fixed_path_threshold": FIXED_PATH_THRESHOLD,
        "min_step": MIN_STEP,
        "reps_grid": REPS_GRID,
        "curated_scenarios": CURATED_SCENARIOS,
        "random_train_seed": RANDOM_TRAIN_SEED,
        "random_holdout_seed": RANDOM_HOLDOUT_SEED,
        "random_scenario_count": RANDOM_SCENARIO_COUNT,
        "random_test_rows": RANDOM_TEST_ROWS,
        "decision": decision,
        "recommended": recommended,
        "recommended_worst_row": worst_row,
    }

    out_detail = ARTIFACT_ROOT / "main4_phase25w_governance_lock_detail.csv"
    out_scenario = ARTIFACT_ROOT / "main4_phase25w_governance_lock_scenario_summary.csv"
    out_rank = ARTIFACT_ROOT / "main4_phase25w_governance_lock_rank.csv"
    out_note = ARTIFACT_ROOT / "main4_phase25w_governance_lock_note.json"

    detail.sort_values(["variant", "path_bootstrap_reps", "group", "scenario", "seed", "transaction_cost_bps"]).to_csv(
        out_detail, index=False
    )
    scenario_summary.to_csv(out_scenario, index=False)
    rank.to_csv(out_rank, index=False)
    out_note.write_text(json.dumps(note, indent=2))

    print("WROTE", out_detail)
    print("WROTE", out_scenario)
    print("WROTE", out_rank)
    print("WROTE", out_note)
    print(json.dumps(note, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
