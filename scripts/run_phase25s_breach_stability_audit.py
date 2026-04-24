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
RUN_ROOT = ARTIFACT_ROOT / "main4_phase25s_runs"

FIXED_PATH_THRESHOLD = -0.007

REPS_GRID = [300, 1200]
PATH_SEED_JITTER = [0, 3, 7]
TEST_ROWS = [
    {"seed": 5, "cost": 25},
    {"seed": 17, "cost": 25},
]

SCENARIOS = [
    {"name": "holdout_b12_p8284", "block_size": 12, "path_seed": 8284},
    {"name": "rand_b13_p6575", "block_size": 13, "path_seed": 6575},
    {"name": "pack2603_b10_p8441", "block_size": 10, "path_seed": 8441},
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
    Variant("v25s_n8", 0.0210, 1.1425, 3.00, 0.0214, -0.651, 0.651),
    Variant("v25s_nA", 0.0211, 1.1425, 3.02, 0.0214, -0.651, 0.651),
    Variant("v25s_pA", 0.0212, 1.1425, 3.05, 0.02145, -0.650, 0.650),
]


def run_case(
    v: Variant,
    scenario: dict[str, Any],
    test_seed: int,
    test_cost: float,
    reps: int,
    seed_jitter: int,
) -> dict[str, Any]:
    path_seed = scenario["path_seed"] + seed_jitter
    run_name = (
        f"{v.name}_{scenario['name']}_j{seed_jitter}_r{reps}_"
        f"c{int(test_cost)}_s{test_seed}_g055_pbsz{scenario['block_size']}_pbs{path_seed + test_seed}"
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
        str(path_seed + test_seed),
    ]

    detailed_path = run_dir / f"main4_gate055_f5_s{test_seed}_detailed.json"
    summary_path = run_dir / f"main4_gate055_f5_s{test_seed}_summary.csv"
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
    m = detailed.get("summary_metrics", {})

    p = float(m.get("path_bootstrap_robust_min_test_relative_p05", float("nan")))
    return {
        "variant": v.name,
        "scenario": scenario["name"],
        "block_size": scenario["block_size"],
        "base_path_seed": scenario["path_seed"],
        "seed_jitter": seed_jitter,
        "path_seed": path_seed,
        "seed": test_seed,
        "transaction_cost_bps": test_cost,
        "path_bootstrap_reps": reps,
        "path_bootstrap_robust_min_p05": p,
        "case_slack": p - FIXED_PATH_THRESHOLD,
        "breach": bool(p < FIXED_PATH_THRESHOLD),
        "mean_test_relative": float(m.get("mean_test_relative_total_return", 0.0)),
        "robust_min_test_relative": float(m.get("robust_min_test_relative", 0.0)),
        "mean_turnover": float(m.get("mean_turnover", 0.0)),
        "mean_executed_step_rate": float(m.get("mean_test_executed_step_count", 0.0) / 186.0),
    }


def main() -> int:
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"Missing virtualenv python: {PYTHON_BIN}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for v in VARIANTS:
        for sc in SCENARIOS:
            for jitter in PATH_SEED_JITTER:
                for rr in REPS_GRID:
                    for tr in TEST_ROWS:
                        rows.append(
                            run_case(
                                v,
                                sc,
                                test_seed=tr["seed"],
                                test_cost=tr["cost"],
                                reps=rr,
                                seed_jitter=jitter,
                            )
                        )

    detail = pd.DataFrame(rows)

    summary_rows: list[dict[str, Any]] = []
    for (variant, reps), g in detail.groupby(["variant", "path_bootstrap_reps"]):
        summary_rows.append(
            {
                "variant": variant,
                "path_bootstrap_reps": int(reps),
                "rows": int(len(g)),
                "breach_count": int(g["breach"].sum()),
                "breach_rate": float(g["breach"].mean()),
                "min_case_slack": float(g["case_slack"].min()),
                "mean_case_slack": float(g["case_slack"].mean()),
                "mean_test_relative": float(g["mean_test_relative"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["path_bootstrap_reps", "breach_rate", "min_case_slack", "mean_case_slack"],
        ascending=[True, True, False, False],
    )

    scenario_rows: list[dict[str, Any]] = []
    for (variant, scenario), g in detail.groupby(["variant", "scenario"]):
        scenario_rows.append(
            {
                "variant": variant,
                "scenario": scenario,
                "rows": int(len(g)),
                "breach_count": int(g["breach"].sum()),
                "breach_rate": float(g["breach"].mean()),
                "min_case_slack": float(g["case_slack"].min()),
                "mean_case_slack": float(g["case_slack"].mean()),
            }
        )
    scenario_summary = pd.DataFrame(scenario_rows).sort_values(
        ["breach_rate", "min_case_slack", "mean_case_slack"],
        ascending=[True, False, False],
    )

    best_row = summary.iloc[0].to_dict()
    best_variant = best_row["variant"]
    best_reps = int(best_row["path_bootstrap_reps"])
    worst_of_best = detail[(detail["variant"] == best_variant) & (detail["path_bootstrap_reps"] == best_reps)].sort_values(
        "case_slack"
    ).head(1)

    note = {
        "fixed_path_threshold": FIXED_PATH_THRESHOLD,
        "reps_grid": REPS_GRID,
        "path_seed_jitter": PATH_SEED_JITTER,
        "test_rows": TEST_ROWS,
        "scenarios": SCENARIOS,
        "best_variant_by_breach_rate": best_variant,
        "best_reps": best_reps,
        "best_summary": best_row,
        "best_worst_row": worst_of_best[
            [
                "scenario",
                "seed",
                "transaction_cost_bps",
                "path_bootstrap_reps",
                "path_seed",
                "path_bootstrap_robust_min_p05",
                "case_slack",
                "breach",
            ]
        ].to_dict("records")[0],
        "all_rows_zero_breach": bool((detail["breach"].sum() == 0)),
        "decision": "PASS" if bool((detail["breach"].sum() == 0)) else "FAIL",
    }

    out_detail = ARTIFACT_ROOT / "main4_phase25s_breach_stability_detail.csv"
    out_summary = ARTIFACT_ROOT / "main4_phase25s_breach_stability_summary.csv"
    out_scenario = ARTIFACT_ROOT / "main4_phase25s_breach_stability_scenario_summary.csv"
    out_note = ARTIFACT_ROOT / "main4_phase25s_breach_stability_note.json"

    detail.sort_values(
        ["variant", "scenario", "path_bootstrap_reps", "seed_jitter", "seed", "transaction_cost_bps"]
    ).to_csv(out_detail, index=False)
    summary.to_csv(out_summary, index=False)
    scenario_summary.to_csv(out_scenario, index=False)
    out_note.write_text(json.dumps(note, indent=2))

    print("WROTE", out_detail)
    print("WROTE", out_summary)
    print("WROTE", out_scenario)
    print("WROTE", out_note)
    print(json.dumps(note, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
