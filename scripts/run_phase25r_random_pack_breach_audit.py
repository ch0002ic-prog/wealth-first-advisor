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
RUN_ROOT = ARTIFACT_ROOT / "main4_phase25r_runs"

FIXED_PATH_THRESHOLD = -0.007
MIN_STEP = 0.02

PACK_SEEDS = [2601, 2602, 2603]
RANDOM_SCENARIOS_PER_PACK = 12

# Focused tail stress rows (historically where breaches appear).
RANDOM_TEST_ROWS = [
    {"seed": 5, "cost": 25},
    {"seed": 17, "cost": 25},
]

# Sentinels for non-path health checks.
SENTINEL_SCENARIOS = [
    {"name": "baseline_b20_s4242", "block_size": 20, "path_seed": 4242, "group": "sentinel"},
    {"name": "altblock_b24_s4242", "block_size": 24, "path_seed": 4242, "group": "sentinel"},
    {"name": "stress_b30_s10010", "block_size": 30, "path_seed": 10010, "group": "sentinel"},
]
SENTINEL_ROWS = [
    {"seed": 17, "cost": 22},
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


VARIANTS = [
    Variant("v25r_n8", 0.0210, 1.1425, 3.00, 0.0214, -0.651, 0.651),
    Variant("v25r_nA", 0.0211, 1.1425, 3.02, 0.0214, -0.651, 0.651),
    Variant("v25r_pA", 0.0212, 1.1425, 3.05, 0.02145, -0.650, 0.650),
]


def make_random_pack(pack_seed: int) -> list[dict[str, Any]]:
    rng = random.Random(pack_seed)
    seen: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []
    while len(rows) < RANDOM_SCENARIOS_PER_PACK:
        block = rng.randint(10, 18)
        pseed = rng.randint(5000, 13000)
        key = (block, pseed)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "name": f"pack{pack_seed}_b{block}_p{pseed}",
                "block_size": block,
                "path_seed": pseed,
                "group": "random_shortblock",
                "pack_seed": pack_seed,
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
        "pack_seed": scenario.get("pack_seed", -1),
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


def main() -> int:
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"Missing virtualenv python: {PYTHON_BIN}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    random_scenarios: list[dict[str, Any]] = []
    for ps in PACK_SEEDS:
        random_scenarios.extend(make_random_pack(ps))

    rows: list[dict[str, Any]] = []
    for v in VARIANTS:
        for sc in random_scenarios:
            for rc in RANDOM_TEST_ROWS:
                rows.append(run_case(v, sc, rc["seed"], rc["cost"]))
        for sc in SENTINEL_SCENARIOS:
            for rc in SENTINEL_ROWS:
                rows.append(run_case(v, sc, rc["seed"], rc["cost"]))

    detail = pd.DataFrame(rows)

    tail = detail[detail["scenario_group"] == "random_shortblock"].copy()
    sent = detail[detail["scenario_group"] == "sentinel"].copy()

    variant_rows: list[dict[str, Any]] = []
    for variant, g in detail.groupby("variant"):
        gt = tail[tail["variant"] == variant]
        gs = sent[sent["variant"] == variant]
        variant_rows.append(
            {
                "variant": variant,
                "tail_rows": int(len(gt)),
                "tail_breach_count": int((gt["path_bootstrap_robust_min_p05"] < FIXED_PATH_THRESHOLD).sum()),
                "tail_breach_rate": float((gt["path_bootstrap_robust_min_p05"] < FIXED_PATH_THRESHOLD).mean()),
                "tail_min_case_slack": float(gt["path_bootstrap_robust_min_p05"].min() - FIXED_PATH_THRESHOLD),
                "tail_mean_case_slack": float(gt["path_bootstrap_robust_min_p05"].mean() - FIXED_PATH_THRESHOLD),
                "tail_mean_test_relative": float(gt["mean_test_relative"].mean()),
                "sentinel_non_path_ok": bool(
                    float(gs["mean_test_relative"].mean()) >= 0.0
                    and float(gs["robust_min_test_relative"].min()) >= 0.0
                    and float(gs["mean_turnover"].mean()) <= 0.0015
                    and float(gs["mean_executed_step_rate"].mean()) >= MIN_STEP
                    and float(gs["worst_daily_relative_return"].min()) >= -0.010
                    and float(gs["worst_relative_drawdown"].max()) <= 0.030
                ),
                "sentinel_mean_turnover": float(gs["mean_turnover"].mean()),
                "sentinel_mean_executed_step_rate": float(gs["mean_executed_step_rate"].mean()),
            }
        )

    variant_rank = pd.DataFrame(variant_rows).sort_values(
        ["tail_breach_rate", "tail_min_case_slack", "tail_mean_case_slack", "tail_mean_test_relative"],
        ascending=[True, False, False, False],
    )

    pack_rows: list[dict[str, Any]] = []
    for (variant, pack_seed), g in tail.groupby(["variant", "pack_seed"]):
        pack_rows.append(
            {
                "variant": variant,
                "pack_seed": int(pack_seed),
                "rows": int(len(g)),
                "breach_count": int((g["path_bootstrap_robust_min_p05"] < FIXED_PATH_THRESHOLD).sum()),
                "breach_rate": float((g["path_bootstrap_robust_min_p05"] < FIXED_PATH_THRESHOLD).mean()),
                "min_case_slack": float(g["path_bootstrap_robust_min_p05"].min() - FIXED_PATH_THRESHOLD),
                "mean_case_slack": float(g["path_bootstrap_robust_min_p05"].mean() - FIXED_PATH_THRESHOLD),
            }
        )
    pack_summary = pd.DataFrame(pack_rows).sort_values(["variant", "pack_seed"])

    best = variant_rank.iloc[0]["variant"]
    worst_best = tail[tail["variant"] == best].sort_values("path_bootstrap_robust_min_p05").head(1)

    note = {
        "fixed_path_threshold": FIXED_PATH_THRESHOLD,
        "rng_pack_seeds": PACK_SEEDS,
        "random_scenarios_per_pack": RANDOM_SCENARIOS_PER_PACK,
        "random_test_rows": RANDOM_TEST_ROWS,
        "sentinel_rows": SENTINEL_ROWS,
        "winner": best,
        "winner_summary": variant_rank.iloc[0].to_dict(),
        "winner_worst_tail_row": worst_best[
            ["scenario", "pack_seed", "seed", "transaction_cost_bps", "path_bootstrap_robust_min_p05"]
        ].to_dict("records")[0],
        "decision": "PASS" if int(variant_rank.iloc[0]["tail_breach_count"]) == 0 else "FAIL",
    }

    out_detail = ARTIFACT_ROOT / "main4_phase25r_random_pack_detail.csv"
    out_rank = ARTIFACT_ROOT / "main4_phase25r_random_pack_variant_rank.csv"
    out_pack = ARTIFACT_ROOT / "main4_phase25r_random_pack_pack_summary.csv"
    out_note = ARTIFACT_ROOT / "main4_phase25r_random_pack_note.json"

    detail.sort_values(["variant", "scenario_group", "pack_seed", "scenario", "seed", "transaction_cost_bps"]).to_csv(
        out_detail, index=False
    )
    variant_rank.to_csv(out_rank, index=False)
    pack_summary.to_csv(out_pack, index=False)
    out_note.write_text(json.dumps(note, indent=2))

    print("WROTE", out_detail)
    print("WROTE", out_rank)
    print("WROTE", out_pack)
    print("WROTE", out_note)
    print(json.dumps(note, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
