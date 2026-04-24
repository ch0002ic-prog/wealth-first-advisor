#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
RUN_ROOT = ARTIFACT_ROOT / "main4_phase25v_runs"

FIXED_PATH_THRESHOLD = -0.007

TRAIN_RANDOM_SEED = 5601
HOLDOUT_RANDOM_SEED = 6713
RANDOM_SCENARIO_COUNT = 10

TEST_ROWS = [
    {"seed": 5, "cost": 22},
    {"seed": 5, "cost": 25},
    {"seed": 17, "cost": 22},
    {"seed": 17, "cost": 25},
    {"seed": 29, "cost": 22},
    {"seed": 29, "cost": 25},
]
REPS_GRID = [600, 900]


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
    Variant("v25v_n8", 0.0210, 1.1425, 3.00, 0.0214, -0.651, 0.651),
    Variant("v25v_pA", 0.0212, 1.1425, 3.05, 0.02145, -0.650, 0.650),
]


def make_random_scenarios(seed: int, count: int, envelope: str) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    seen: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []
    while len(rows) < count:
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
                "envelope": envelope,
            }
        )
    return rows


def run_case(v: Variant, scenario: dict[str, Any], test_seed: int, test_cost: float, reps: int) -> dict[str, Any]:
    run_name = (
        f"{v.name}_{scenario['name']}_r{reps}_c{int(test_cost)}_s{test_seed}"
        f"_g055_pbsz{scenario['block_size']}_pbs{scenario['path_seed'] + test_seed}"
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
        str(test_seed),
        "--output-dir",
        str(run_dir),
        "--transaction-cost-bps",
        str(test_cost),
        "--slippage-bps",
        str(test_cost),
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
        str(scenario["path_seed"] + test_seed),
    ]

    started = time.perf_counter()
    detailed_path = run_dir / f"main4_gate055_f5_s{test_seed}_detailed.json"
    summary_path = run_dir / f"main4_gate055_f5_s{test_seed}_summary.csv"
    res = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started

    if (not detailed_path.exists()) or (not summary_path.exists()):
        raise RuntimeError(f"Case failed {run_name}:\n{res.stderr}\n{res.stdout}")

    detailed = json.loads(detailed_path.read_text())
    m = detailed.get("summary_metrics", {})
    p = float(m.get("path_bootstrap_robust_min_test_relative_p05", float("nan")))

    return {
        "variant": v.name,
        "envelope": scenario["envelope"],
        "scenario": scenario["name"],
        "seed": test_seed,
        "transaction_cost_bps": test_cost,
        "path_bootstrap_reps": reps,
        "runtime_sec": float(elapsed),
        "path_bootstrap_robust_min_p05": p,
        "case_slack": p - FIXED_PATH_THRESHOLD,
        "breach": bool(p < FIXED_PATH_THRESHOLD),
        "mean_test_relative": float(m.get("mean_test_relative_total_return", 0.0)),
    }


def main() -> int:
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"Missing virtualenv python: {PYTHON_BIN}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    scenarios = make_random_scenarios(TRAIN_RANDOM_SEED, RANDOM_SCENARIO_COUNT, "random_train") + make_random_scenarios(
        HOLDOUT_RANDOM_SEED, RANDOM_SCENARIO_COUNT, "random_holdout"
    )

    rows: list[dict[str, Any]] = []
    for v in VARIANTS:
        for rr in REPS_GRID:
            for sc in scenarios:
                for tr in TEST_ROWS:
                    rows.append(run_case(v, sc, tr["seed"], tr["cost"], rr))

    detail = pd.DataFrame(rows)

    env_rows: list[dict[str, Any]] = []
    for (variant, reps, env), g in detail.groupby(["variant", "path_bootstrap_reps", "envelope"]):
        env_rows.append(
            {
                "variant": variant,
                "path_bootstrap_reps": int(reps),
                "envelope": env,
                "rows": int(len(g)),
                "breach_count": int(g["breach"].sum()),
                "breach_rate": float(g["breach"].mean()),
                "min_case_slack": float(g["case_slack"].min()),
                "mean_case_slack": float(g["case_slack"].mean()),
                "mean_test_relative": float(g["mean_test_relative"].mean()),
                "mean_runtime_sec": float(g["runtime_sec"].mean()),
            }
        )
    env_summary = pd.DataFrame(env_rows).sort_values(["path_bootstrap_reps", "variant", "envelope"])

    agg_rows: list[dict[str, Any]] = []
    for (variant, reps), g in detail.groupby(["variant", "path_bootstrap_reps"]):
        gt = g[g["envelope"] == "random_train"]
        gh = g[g["envelope"] == "random_holdout"]
        agg_rows.append(
            {
                "variant": variant,
                "path_bootstrap_reps": int(reps),
                "rows": int(len(g)),
                "breach_count": int(g["breach"].sum()),
                "breach_rate": float(g["breach"].mean()),
                "train_breach_count": int(gt["breach"].sum()),
                "holdout_breach_count": int(gh["breach"].sum()),
                "min_case_slack": float(g["case_slack"].min()),
                "mean_case_slack": float(g["case_slack"].mean()),
                "mean_test_relative": float(g["mean_test_relative"].mean()),
                "mean_runtime_sec": float(g["runtime_sec"].mean()),
                "total_runtime_sec": float(g["runtime_sec"].sum()),
            }
        )
    aggregate = pd.DataFrame(agg_rows).sort_values(
        ["holdout_breach_count", "train_breach_count", "path_bootstrap_reps", "min_case_slack"],
        ascending=[True, True, True, False],
    )

    safe = aggregate[(aggregate["train_breach_count"] == 0) & (aggregate["holdout_breach_count"] == 0)]
    if len(safe) > 0:
        min_safe = safe.sort_values(["path_bootstrap_reps", "total_runtime_sec"]).iloc[0].to_dict()
        decision = "PASS"
    else:
        min_safe = None
        decision = "FAIL"

    note = {
        "fixed_path_threshold": FIXED_PATH_THRESHOLD,
        "train_random_seed": TRAIN_RANDOM_SEED,
        "holdout_random_seed": HOLDOUT_RANDOM_SEED,
        "random_scenario_count_per_envelope": RANDOM_SCENARIO_COUNT,
        "test_rows": TEST_ROWS,
        "reps_grid": REPS_GRID,
        "decision": decision,
        "minimum_safe_reps_row": min_safe,
    }

    out_detail = ARTIFACT_ROOT / "main4_phase25v_reps_confirm_detail.csv"
    out_env = ARTIFACT_ROOT / "main4_phase25v_reps_confirm_env_summary.csv"
    out_agg = ARTIFACT_ROOT / "main4_phase25v_reps_confirm_aggregate.csv"
    out_note = ARTIFACT_ROOT / "main4_phase25v_reps_confirm_note.json"

    detail.sort_values(["variant", "path_bootstrap_reps", "envelope", "scenario", "seed", "transaction_cost_bps"]).to_csv(
        out_detail, index=False
    )
    env_summary.to_csv(out_env, index=False)
    aggregate.to_csv(out_agg, index=False)
    out_note.write_text(json.dumps(note, indent=2))

    print("WROTE", out_detail)
    print("WROTE", out_env)
    print("WROTE", out_agg)
    print("WROTE", out_note)
    print(json.dumps(note, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
