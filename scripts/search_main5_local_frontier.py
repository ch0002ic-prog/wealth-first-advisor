#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
RUN_ROOT = ARTIFACT_ROOT / "main5_frontier_runs"
FIXED_PATH_THRESHOLD = -0.007

SCENARIOS = [
    {"name": "baseline_b20_s4242", "block_size": 20, "path_seed": 4242},
    {"name": "altblock_b16_s4242", "block_size": 16, "path_seed": 4242},
    {"name": "altseed_b20_s5252", "block_size": 20, "path_seed": 5252},
    {"name": "deepcheck_b12_s8080", "block_size": 12, "path_seed": 8080},
]

BASE_PARAMS = {
    "ridge_l2": 0.020815,
    "min_spy_weight": 0.809134,
    "max_spy_weight": 1.058647,
    "initial_spy_weight": 1.0,
    "scale_turnover_penalty": 2.9952,
}


@dataclass(frozen=True)
class Candidate:
    name: str
    forward_horizon: int
    no_trade_band: float
    min_signal_scale: float
    max_signal_scale: float
    action_smoothing: float


def build_candidates() -> list[Candidate]:
    cands: list[Candidate] = []
    for ntb in [0.049, 0.050, 0.051]:
        for scale_abs in [0.19, 0.20, 0.21]:
            for smoothing in [1.12, 1.14, 1.16]:
                tag = f"h10_ntb{int(round(ntb * 1000)):03d}_s{int(round(scale_abs * 100)):03d}_sm{int(round(smoothing * 100)):03d}"
                cands.append(
                    Candidate(
                        name=tag,
                        forward_horizon=10,
                        no_trade_band=ntb,
                        min_signal_scale=-scale_abs,
                        max_signal_scale=scale_abs,
                        action_smoothing=smoothing,
                    )
                )
    return cands


def _run_one(
    candidate: Candidate,
    scenario: dict[str, Any],
    seed: int,
    cost: int,
    reps: int,
    label: str,
) -> dict[str, Any]:
    run_name = (
        f"{label}_{candidate.name}_{scenario['name']}"
        f"_r{reps}_c{cost}_s{seed}_pbsz{scenario['block_size']}"
    )
    run_dir = RUN_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PYTHON_BIN),
        "src/wealth_first/main5.py",
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
        str(BASE_PARAMS["ridge_l2"]),
        "--min-signal-scale",
        str(candidate.min_signal_scale),
        "--max-signal-scale",
        str(candidate.max_signal_scale),
        "--min-spy-weight",
        str(BASE_PARAMS["min_spy_weight"]),
        "--max-spy-weight",
        str(BASE_PARAMS["max_spy_weight"]),
        "--initial-spy-weight",
        str(BASE_PARAMS["initial_spy_weight"]),
        "--action-smoothing",
        str(candidate.action_smoothing),
        "--no-trade-band",
        str(candidate.no_trade_band),
        "--scale-turnover-penalty",
        str(BASE_PARAMS["scale_turnover_penalty"]),
        "--validation-relative-floor-target",
        "0.0",
        "--validation-relative-floor-penalty",
        "1.25",
        "--validation-hard-min-relative-return",
        "-0.001",
        "--validation-tail-bootstrap-reps",
        str(reps),
        "--validation-tail-bootstrap-block-size",
        str(scenario["block_size"]),
        "--validation-tail-bootstrap-quantile",
        "0.05",
        "--validation-tail-bootstrap-floor-target",
        "-0.003",
        "--validation-tail-bootstrap-penalty",
        "2.0",
        "--validation-tail-bootstrap-hard-min",
        "-0.007",
        "--validation-tail-bootstrap-objective-weight",
        "4.0",
        "--validation-tail-bootstrap-seed",
        str(scenario["path_seed"] + seed),
        "--min-robust-min-relative",
        "0.0",
        "--min-active-fraction",
        "0.01",
        "--forward-horizon",
        str(candidate.forward_horizon),
        "--path-bootstrap-reps",
        str(reps),
        "--path-bootstrap-block-size",
        str(scenario["block_size"]),
        "--path-bootstrap-seed",
        str(scenario["path_seed"] + seed),
        "--no-fail-on-gate",
    ]

    detailed_path = run_dir / f"main5_gate055_f5_s{seed}_detailed.json"
    res = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        capture_output=True,
        text=True,
    )

    default = {
        "candidate": candidate.name,
        "scenario": scenario["name"],
        "seed": seed,
        "transaction_cost_bps": cost,
        "path_bootstrap_reps": reps,
        "path_bootstrap_robust_min_p05": float("nan"),
        "case_slack": float("nan"),
        "breach": True,
        "mean_test_relative": float("nan"),
        "mean_validation_relative": float("nan"),
        "mean_turnover": float("nan"),
        "mean_test_executed_step_count": float("nan"),
        "mean_test_gate_suppression_rate": float("nan"),
        "error": None,
    }

    if res.returncode != 0 and not detailed_path.exists():
        return {**default, "error": (res.stderr or "").strip()[:800]}

    if not detailed_path.exists():
        return {**default, "error": "detailed_json_missing_after_run"}

    detail = json.loads(detailed_path.read_text(encoding="utf-8"))
    summary = detail.get("summary_metrics", {})
    p05 = float(summary.get("path_bootstrap_robust_min_test_relative_p05", float("nan")))
    breach = pd.isna(p05) or p05 < FIXED_PATH_THRESHOLD

    return {
        "candidate": candidate.name,
        "scenario": scenario["name"],
        "seed": seed,
        "transaction_cost_bps": cost,
        "path_bootstrap_reps": reps,
        "path_bootstrap_robust_min_p05": p05,
        "case_slack": p05 - FIXED_PATH_THRESHOLD if not pd.isna(p05) else float("nan"),
        "breach": bool(breach),
        "mean_test_relative": float(summary.get("mean_test_relative_total_return", 0.0)),
        "mean_validation_relative": float(summary.get("mean_validation_relative_total_return", 0.0)),
        "mean_turnover": float(summary.get("mean_turnover", 0.0)),
        "mean_test_executed_step_count": float(summary.get("mean_test_executed_step_count", 0.0)),
        "mean_test_gate_suppression_rate": float(summary.get("mean_test_gate_suppression_rate", 0.0)),
        "error": None,
    }


def run_batch(
    label: str,
    candidates: list[Candidate],
    seeds: list[int],
    costs: list[int],
    reps: int,
    workers: int,
) -> pd.DataFrame:
    cases = [
        (candidate, scenario, seed, cost)
        for candidate in candidates
        for scenario in SCENARIOS
        for seed in seeds
        for cost in costs
    ]
    total = len(cases)
    print(
        f"Running {total} cases for {label} ({len(candidates)} candidates x "
        f"{len(SCENARIOS)} scenarios x {len(seeds)} seeds x {len(costs)} costs), reps={reps}"
    )

    rows: list[dict[str, Any]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_one, c, sc, s, cost, reps, label): (c, sc, s, cost)
            for c, sc, s, cost in cases
        }
        for fut in as_completed(futures):
            c, sc, s, cost = futures[fut]
            rec = fut.result()
            rows.append(rec)
            done += 1
            status = "BREACH" if rec.get("breach") else "ok"
            slack = rec.get("case_slack")
            slack_txt = f"{slack:.4f}" if isinstance(slack, (int, float)) and not pd.isna(slack) else "n/a"
            print(f"[{done:4d}/{total}] {status:6s} slack={slack_txt:>7s} {c.name} {sc['name']} s={s} c={cost}")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, grp in df.groupby("candidate"):
        rows.append(
            {
                "candidate": candidate,
                "rows": int(len(grp)),
                "breach_count": int(grp["breach"].sum()),
                "min_case_slack": float(grp["case_slack"].min()),
                "mean_case_slack": float(grp["case_slack"].mean()),
                "mean_test_relative": float(grp["mean_test_relative"].mean()),
                "mean_validation_relative": float(grp["mean_validation_relative"].mean()),
                "mean_turnover": float(grp["mean_turnover"].mean()),
                "mean_executed_steps": float(grp["mean_test_executed_step_count"].mean()),
                "mean_gate_suppression": float(grp["mean_test_gate_suppression_rate"].mean()),
                "cost35_min_slack": float(grp.loc[grp["transaction_cost_bps"] == 35, "case_slack"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["breach_count", "min_case_slack", "mean_test_relative"],
        ascending=[True, False, False],
    )


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def keep_top_candidates(summary_df: pd.DataFrame, all_candidates: list[Candidate], topk: int) -> list[Candidate]:
    top_names = list(summary_df["candidate"].head(topk))
    by_name = {c.name: c for c in all_candidates}
    return [by_name[name] for name in top_names if name in by_name]


def main() -> int:
    parser = argparse.ArgumentParser(description="Two-stage constrained local frontier search for main5")
    parser.add_argument("--label", type=str, default="frontier_a")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stage1-reps", type=int, default=30)
    parser.add_argument("--stage2-reps", type=int, default=80)
    parser.add_argument("--stage1-seeds", type=str, default="5,17")
    parser.add_argument("--stage1-costs", type=str, default="22,30,35")
    parser.add_argument("--stage2-seeds", type=str, default="5,17,29")
    parser.add_argument("--stage2-costs", type=str, default="22,25,30,35")
    parser.add_argument("--stage2-topk", type=int, default=5)
    args = parser.parse_args()

    candidates = build_candidates()

    stage1_df = run_batch(
        label=f"{args.label}_stage1",
        candidates=candidates,
        seeds=parse_int_list(args.stage1_seeds),
        costs=parse_int_list(args.stage1_costs),
        reps=args.stage1_reps,
        workers=args.workers,
    )
    stage1_summary = summarize(stage1_df)

    top_candidates = keep_top_candidates(stage1_summary, candidates, args.stage2_topk)
    if not top_candidates:
        print("No candidates survived stage1 ranking.")
        return 2

    stage2_df = run_batch(
        label=f"{args.label}_stage2",
        candidates=top_candidates,
        seeds=parse_int_list(args.stage2_seeds),
        costs=parse_int_list(args.stage2_costs),
        reps=args.stage2_reps,
        workers=args.workers,
    )
    stage2_summary = summarize(stage2_df)

    stage1_detail_path = ARTIFACT_ROOT / f"main5_frontier_{args.label}_stage1_detail.csv"
    stage1_summary_path = ARTIFACT_ROOT / f"main5_frontier_{args.label}_stage1_summary.csv"
    stage2_detail_path = ARTIFACT_ROOT / f"main5_frontier_{args.label}_stage2_detail.csv"
    stage2_summary_path = ARTIFACT_ROOT / f"main5_frontier_{args.label}_stage2_summary.csv"
    overall_json_path = ARTIFACT_ROOT / f"main5_frontier_{args.label}_summary.json"

    stage1_df.to_csv(stage1_detail_path, index=False)
    stage1_summary.to_csv(stage1_summary_path, index=False)
    stage2_df.to_csv(stage2_detail_path, index=False)
    stage2_summary.to_csv(stage2_summary_path, index=False)

    overall_json_path.write_text(
        json.dumps(
            {
                "label": args.label,
                "threshold": FIXED_PATH_THRESHOLD,
                "stage1": {
                    "reps": args.stage1_reps,
                    "seeds": parse_int_list(args.stage1_seeds),
                    "costs": parse_int_list(args.stage1_costs),
                    "summary": json.loads(stage1_summary.to_json(orient="records")),
                },
                "stage2": {
                    "reps": args.stage2_reps,
                    "seeds": parse_int_list(args.stage2_seeds),
                    "costs": parse_int_list(args.stage2_costs),
                    "topk": args.stage2_topk,
                    "candidates": [c.__dict__ for c in top_candidates],
                    "summary": json.loads(stage2_summary.to_json(orient="records")),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nWrote {stage1_detail_path}")
    print(f"Wrote {stage1_summary_path}")
    print(f"Wrote {stage2_detail_path}")
    print(f"Wrote {stage2_summary_path}")
    print(f"Wrote {overall_json_path}")

    print("\n=== Stage2 Frontier Summary ===")
    print(
        f"{'Candidate':<30} {'Breach':>6} {'MinSlack':>9} {'MeanTest':>9} {'Steps':>7} {'Supp':>8} {'C35Min':>8}"
    )
    print("-" * 90)
    for _, row in stage2_summary.iterrows():
        print(
            f"{row['candidate']:<30}"
            f" {int(row['breach_count']):>6d}"
            f" {float(row['min_case_slack']):>9.4f}"
            f" {float(row['mean_test_relative']):>9.4f}"
            f" {float(row['mean_executed_steps']):>7.2f}"
            f" {float(row['mean_gate_suppression']):>8.2%}"
            f" {float(row['cost35_min_slack']):>8.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
