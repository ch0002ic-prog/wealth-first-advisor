#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
RUN_ROOT = ARTIFACT_ROOT / "main4_phase25e_runs"


@dataclass(frozen=True)
class Variant:
    name: str
    ridge_l2: float = 0.015
    target_mode: str = "tanh_return"
    action_smoothing: float = 1.0
    scale_turnover_penalty: float = 0.0
    min_signal_scale: float = -0.75
    max_signal_scale: float = 0.75
    min_spy_weight: float = 0.80
    max_spy_weight: float = 1.05
    initial_spy_weight: float = 1.0
    no_trade_band: float = 0.02


VARIANTS = [
    Variant(
        name="v25e_ref_tune_a",
        ridge_l2=0.021,
        action_smoothing=1.15,
        scale_turnover_penalty=3.2,
        no_trade_band=0.022,
        min_signal_scale=-0.64,
        max_signal_scale=0.64,
    ),
    Variant(
        name="v25e_looser_01",
        ridge_l2=0.021,
        action_smoothing=1.14,
        scale_turnover_penalty=3.0,
        no_trade_band=0.0215,
        min_signal_scale=-0.65,
        max_signal_scale=0.65,
    ),
    Variant(
        name="v25e_looser_02",
        ridge_l2=0.021,
        action_smoothing=1.13,
        scale_turnover_penalty=2.8,
        no_trade_band=0.021,
        min_signal_scale=-0.66,
        max_signal_scale=0.66,
    ),
    Variant(
        name="v25e_tighter_01",
        ridge_l2=0.022,
        action_smoothing=1.16,
        scale_turnover_penalty=3.4,
        no_trade_band=0.0225,
        min_signal_scale=-0.63,
        max_signal_scale=0.63,
    ),
    Variant(
        name="v25e_tighter_02",
        ridge_l2=0.022,
        action_smoothing=1.17,
        scale_turnover_penalty=3.6,
        no_trade_band=0.023,
        min_signal_scale=-0.62,
        max_signal_scale=0.62,
    ),
    Variant(
        name="v25e_asym_01",
        ridge_l2=0.021,
        action_smoothing=1.15,
        scale_turnover_penalty=3.2,
        no_trade_band=0.022,
        min_signal_scale=-0.60,
        max_signal_scale=0.64,
    ),
    Variant(
        name="v25e_asym_02",
        ridge_l2=0.022,
        action_smoothing=1.16,
        scale_turnover_penalty=3.3,
        no_trade_band=0.0225,
        min_signal_scale=-0.58,
        max_signal_scale=0.64,
    ),
    Variant(
        name="v25e_ridge_lo",
        ridge_l2=0.020,
        action_smoothing=1.15,
        scale_turnover_penalty=3.2,
        no_trade_band=0.022,
        min_signal_scale=-0.64,
        max_signal_scale=0.64,
    ),
    Variant(
        name="v25e_ridge_hi",
        ridge_l2=0.023,
        action_smoothing=1.15,
        scale_turnover_penalty=3.2,
        no_trade_band=0.022,
        min_signal_scale=-0.64,
        max_signal_scale=0.64,
    ),
]

SEEDS = [5, 17, 29]
FRICTIONS = [12, 15, 18, 22, 25]


def run_case(v: Variant, seed: int, cost: float, path_reps: int, block_size: int, path_seed: int) -> dict[str, Any]:
    run_name = (
        f"{v.name}_c{int(cost)}_s{seed}_g055"
        f"_pbr{int(path_reps)}_pbsz{int(block_size)}_pbs{int(path_seed + seed)}"
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
        "--target-mode", v.target_mode,
        "--scale-turnover-penalty", str(v.scale_turnover_penalty),
        "--min-signal-scale", str(v.min_signal_scale),
        "--max-signal-scale", str(v.max_signal_scale),
        "--min-spy-weight", str(v.min_spy_weight),
        "--max-spy-weight", str(v.max_spy_weight),
        "--initial-spy-weight", str(v.initial_spy_weight),
        "--action-smoothing", str(v.action_smoothing),
        "--no-trade-band", str(v.no_trade_band),
        "--min-robust-min-relative", "0.0",
        "--min-active-fraction", "0.01",
        "--path-bootstrap-reps", str(path_reps),
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
        "seed": seed,
        "transaction_cost_bps": cost,
        "slippage_bps": cost,
        "mean_test_relative": float(m.get("mean_test_relative_total_return", 0.0)),
        "robust_min_test_relative": float(m.get("robust_min_test_relative", 0.0)),
        "mean_turnover": float(m.get("mean_turnover", 0.0)),
        "mean_executed_step_rate": float(m.get("mean_test_executed_step_count", 0.0) / max(float(fold.get("n_test_samples", pd.Series([1.0])).mean()), 1e-8)),
        "worst_daily_relative_return": float(fold.get("test_worst_daily_relative_return", pd.Series([0.0])).min()),
        "worst_relative_drawdown": float(fold.get("test_max_relative_drawdown", pd.Series([0.0])).max()),
        "path_bootstrap_robust_min_p05": float(m.get("path_bootstrap_robust_min_test_relative_p05", float("nan"))),
    }


def wealth_score(row: pd.Series) -> float:
    return float(row["mean_test_relative"]) + 0.5 * float(row["robust_min_test_relative"]) - 0.75 * float(row["mean_turnover"])


def summarize_variant(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("variant", as_index=False).agg(
        n_cases=("variant", "count"),
        mean_score=("wealth_first_score", "mean"),
        mean_test_relative=("mean_test_relative", "mean"),
        worst_robust_min=("robust_min_test_relative", "min"),
        worst_decile=("mean_test_relative", lambda x: float(np.quantile(x, 0.1))),
        mean_turnover=("mean_turnover", "mean"),
        mean_executed_step_rate=("mean_executed_step_rate", "mean"),
        worst_daily_relative_return=("worst_daily_relative_return", "min"),
        worst_relative_drawdown=("worst_relative_drawdown", "max"),
        mean_path_bootstrap_robust_min_p05=("path_bootstrap_robust_min_p05", "mean"),
    )


def apply_policy(profile_summary: pd.DataFrame, min_step: float, configured_path: float, quantile: float, max_relax: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    tbl = profile_summary.copy()
    tbl["pass_tail"] = tbl["worst_decile"] >= 0.0
    tbl["pass_robust"] = tbl["worst_robust_min"] >= 0.0
    tbl["pass_turnover"] = tbl["mean_turnover"] <= 0.0015
    tbl["pass_activity"] = tbl["mean_executed_step_rate"] >= min_step
    tbl["pass_worst_daily"] = tbl["worst_daily_relative_return"] >= -0.010
    tbl["pass_drawdown"] = tbl["worst_relative_drawdown"] <= 0.030
    tbl["pass_non_path"] = (
        tbl["pass_tail"]
        & tbl["pass_robust"]
        & tbl["pass_turnover"]
        & tbl["pass_activity"]
        & tbl["pass_worst_daily"]
        & tbl["pass_drawdown"]
    )

    candidates = tbl[tbl["pass_non_path"]]
    if candidates.empty:
        resolved_before_cap = configured_path
    else:
        resolved_before_cap = float(np.quantile(candidates["mean_path_bootstrap_robust_min_p05"].to_numpy(), quantile))

    resolved_after_cap = resolved_before_cap
    cap_applied = False
    if max_relax >= 0.0:
        floor = configured_path - max_relax
        if resolved_after_cap < floor:
            resolved_after_cap = float(floor)
            cap_applied = True

    tbl["pass_path"] = tbl["mean_path_bootstrap_robust_min_p05"] >= resolved_after_cap
    tbl["strict_eligible"] = tbl["pass_non_path"] & tbl["pass_path"]
    tbl["activity_slack"] = tbl["mean_executed_step_rate"] - min_step
    tbl["path_slack"] = tbl["mean_path_bootstrap_robust_min_p05"] - resolved_after_cap

    eligible = tbl[tbl["strict_eligible"]].sort_values("mean_score", ascending=False)
    best = eligible.iloc[0]["variant"] if not eligible.empty else None

    meta = {
        "min_mean_executed_step_rate": min_step,
        "configured_min_path_bootstrap_robust_min_p05": configured_path,
        "strict_path_bootstrap_gate_quantile": quantile,
        "strict_path_bootstrap_gate_max_relaxation": max_relax,
        "resolved_path_threshold_before_cap": resolved_before_cap,
        "resolved_path_threshold_after_cap": resolved_after_cap,
        "cap_applied": cap_applied,
        "strict_feasible": bool(not eligible.empty),
        "strict_feasible_profile_count": int(eligible.shape[0]),
        "strict_best_eligible_profile": best,
    }
    return tbl, meta


def main() -> int:
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"Missing virtualenv python: {PYTHON_BIN}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for v in VARIANTS:
        for s in SEEDS:
            for c in FRICTIONS:
                rows.append(run_case(v, s, c, path_reps=300, block_size=20, path_seed=4242))

    detail = pd.DataFrame(rows)
    detail["wealth_first_score"] = detail.apply(wealth_score, axis=1)
    profile_summary = summarize_variant(detail)

    policy_specs = [
        {"policy_id": "cap001_step002", "min_step": 0.02, "configured_path": -0.006, "quantile": 0.5, "cap": 0.001},
        {"policy_id": "cap001_step0019", "min_step": 0.019, "configured_path": -0.006, "quantile": 0.5, "cap": 0.001},
        {"policy_id": "cap002_step002", "min_step": 0.02, "configured_path": -0.006, "quantile": 0.5, "cap": 0.002},
    ]

    compact_rows: list[dict[str, Any]] = []
    policy_reports: list[dict[str, Any]] = []
    for spec in policy_specs:
        evaluated, meta = apply_policy(
            profile_summary,
            min_step=float(spec["min_step"]),
            configured_path=float(spec["configured_path"]),
            quantile=float(spec["quantile"]),
            max_relax=float(spec["cap"]),
        )
        meta = {**{"policy_id": spec["policy_id"]}, **meta}
        policy_reports.append(meta)

        for _, r in evaluated.sort_values("mean_score", ascending=False).iterrows():
            compact_rows.append(
                {
                    "policy_id": spec["policy_id"],
                    "variant": r["variant"],
                    "mean_score": float(r["mean_score"]),
                    "mean_test_relative": float(r["mean_test_relative"]),
                    "worst_robust_min": float(r["worst_robust_min"]),
                    "mean_turnover": float(r["mean_turnover"]),
                    "mean_executed_step_rate": float(r["mean_executed_step_rate"]),
                    "mean_path_bootstrap_robust_min_p05": float(r["mean_path_bootstrap_robust_min_p05"]),
                    "activity_slack": float(r["activity_slack"]),
                    "path_slack": float(r["path_slack"]),
                    "pass_non_path": bool(r["pass_non_path"]),
                    "pass_path": bool(r["pass_path"]),
                    "strict_eligible": bool(r["strict_eligible"]),
                }
            )

    out_json = ARTIFACT_ROOT / "main4_phase25e_ultrafine_comparison.json"
    out_csv = ARTIFACT_ROOT / "main4_phase25e_ultrafine_comparison.csv"
    out_detail = ARTIFACT_ROOT / "main4_phase25e_ultrafine_detail.csv"

    detail.sort_values(["variant", "transaction_cost_bps", "seed"]).to_csv(out_detail, index=False)
    out_json.write_text(json.dumps({"variants": [v.name for v in VARIANTS], "policy_reports": policy_reports, "rows": compact_rows}, indent=2))
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(compact_rows[0].keys()))
        writer.writeheader()
        writer.writerows(compact_rows)

    print("WROTE", out_json)
    print("WROTE", out_csv)
    print("WROTE", out_detail)
    for report in policy_reports:
        print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
