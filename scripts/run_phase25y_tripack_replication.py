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
RUN_ROOT = ARTIFACT_ROOT / "main4_phase25y_runs"

FIXED_PATH_THRESHOLD = -0.007
MIN_STEP = 0.02
REPS = 600

CURATED_SCENARIOS = [
    {"name": "baseline_b20_s4242", "block_size": 20, "path_seed": 4242, "group": "curated"},
    {"name": "altseed_b20_s5252", "block_size": 20, "path_seed": 5252, "group": "curated"},
    {"name": "altblock_b16_s4242", "block_size": 16, "path_seed": 4242, "group": "curated"},
    {"name": "altblock_b24_s4242", "block_size": 24, "path_seed": 4242, "group": "curated"},
    {"name": "altboth_b24_s6060", "block_size": 24, "path_seed": 6060, "group": "curated"},
    {"name": "deepcheck_b12_s8080", "block_size": 12, "path_seed": 8080, "group": "curated"},
    {"name": "deepcheck_b28_s9090", "block_size": 28, "path_seed": 9090, "group": "curated"},
    {"name": "stress_b14_s7070", "block_size": 14, "path_seed": 7070, "group": "curated"},
    {"name": "stress_b30_s10010", "block_size": 30, "path_seed": 10010, "group": "curated"},
]
CURATED_SEEDS = [5, 17, 29]
CURATED_COSTS = [18, 22, 25]

RANDOM_SCENARIO_COUNT = 8
RANDOM_TEST_ROWS = [
    {"seed": 5, "cost": 22},
    {"seed": 5, "cost": 25},
    {"seed": 17, "cost": 22},
    {"seed": 17, "cost": 25},
]
PACKS = [
    {"pack": "pack1", "train_seed": 5601, "holdout_seed": 6713},
    {"pack": "pack2", "train_seed": 5629, "holdout_seed": 6749},
    {"pack": "pack3", "train_seed": 5657, "holdout_seed": 6781},
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


# Recommended config from phase25x expanded lock audit.
VARIANT = Variant("v25y_pA", 0.0212, 1.1425, 3.05, 0.02145, -0.650, 0.650)


def make_random_scenarios(seed: int, group: str, pack: str) -> list[dict[str, Any]]:
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
                "name": f"{pack}_{group}_b{block}_p{pseed}",
                "block_size": block,
                "path_seed": pseed,
                "group": group,
                "pack": pack,
            }
        )
    return rows


def run_case(v: Variant, scenario: dict[str, Any], seed: int, cost: float) -> dict[str, Any]:
    run_name = (
        f"{v.name}_{scenario['name']}_r{REPS}_c{int(cost)}_s{seed}_g055"
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
        str(REPS),
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
        "pack": scenario["pack"],
        "group": scenario["group"],
        "scenario": scenario["name"],
        "seed": seed,
        "transaction_cost_bps": cost,
        "path_bootstrap_reps": REPS,
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

    rows: list[dict[str, Any]] = []
    for pack_cfg in PACKS:
        pack = pack_cfg["pack"]
        random_train = make_random_scenarios(pack_cfg["train_seed"], "random_train", pack)
        random_holdout = make_random_scenarios(pack_cfg["holdout_seed"], "random_holdout", pack)

        # Curated scenarios are repeated per pack so each pack is an independent full-suite replication.
        curated = [dict(sc, pack=pack) for sc in CURATED_SCENARIOS]

        for sc in curated:
            for seed in CURATED_SEEDS:
                for cost in CURATED_COSTS:
                    rows.append(run_case(VARIANT, sc, seed, cost))

        for sc in random_train + random_holdout:
            for rc in RANDOM_TEST_ROWS:
                rows.append(run_case(VARIANT, sc, rc["seed"], rc["cost"]))

    detail = pd.DataFrame(rows)

    pack_rows: list[dict[str, Any]] = []
    for pack, g in detail.groupby("pack"):
        gc = g[g["group"] == "curated"]
        gt = g[g["group"] == "random_train"]
        gh = g[g["group"] == "random_holdout"]
        pack_rows.append(
            {
                "pack": pack,
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
    pack_summary = pd.DataFrame(pack_rows).sort_values("pack")

    pass_mask = (
        pack_summary["all_non_path"]
        & (pack_summary["curated_breach_count"] == 0)
        & (pack_summary["train_breach_count"] == 0)
        & (pack_summary["holdout_breach_count"] == 0)
    )
    pass_count = int(pass_mask.sum())
    pack_count = int(len(pack_summary))
    pass_rate = float(pass_count / pack_count) if pack_count else 0.0

    decision = "PASS" if pass_count == pack_count else "FAIL"

    worst_rows = (
        detail.sort_values(["case_slack", "pack"]) 
        .head(5)[["pack", "group", "scenario", "seed", "transaction_cost_bps", "path_bootstrap_robust_min_p05", "case_slack"]]
        .to_dict("records")
    )

    note = {
        "fixed_path_threshold": FIXED_PATH_THRESHOLD,
        "min_step": MIN_STEP,
        "variant": VARIANT.name,
        "path_bootstrap_reps": REPS,
        "packs": PACKS,
        "random_scenario_count": RANDOM_SCENARIO_COUNT,
        "random_test_rows": RANDOM_TEST_ROWS,
        "decision": decision,
        "pack_count": pack_count,
        "pass_count": pass_count,
        "pass_rate": pass_rate,
        "min_pack_min_case_slack": float(pack_summary["min_case_slack"].min()),
        "mean_pack_min_case_slack": float(pack_summary["min_case_slack"].mean()),
        "worst_rows": worst_rows,
    }

    out_detail = ARTIFACT_ROOT / "main4_phase25y_tripack_detail.csv"
    out_pack = ARTIFACT_ROOT / "main4_phase25y_tripack_pack_summary.csv"
    out_note = ARTIFACT_ROOT / "main4_phase25y_tripack_note.json"

    detail.sort_values(["pack", "group", "scenario", "seed", "transaction_cost_bps"]).to_csv(out_detail, index=False)
    pack_summary.to_csv(out_pack, index=False)
    out_note.write_text(json.dumps(note, indent=2))

    print("WROTE", out_detail)
    print("WROTE", out_pack)
    print("WROTE", out_note)
    print("DECISION", decision)
    print("PACK_PASS", f"{pass_count}/{pack_count}")
    print("PASS_RATE", pass_rate)
    print("MIN_PACK_MIN_CASE_SLACK", note["min_pack_min_case_slack"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
