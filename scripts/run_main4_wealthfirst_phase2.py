#!/usr/bin/env python3
"""Phase-2 wealth-first optimization sweep for main4.

This script evaluates candidate policy profiles across friction stress points and
multiple seeds, then ranks profiles with a wealth-first composite objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
RUN_ROOT = ARTIFACT_ROOT / "main4_phase2_runs"


@dataclass(frozen=True)
class Profile:
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


@dataclass(frozen=True)
class Case:
    profile: str
    transaction_cost_bps: float
    slippage_bps: float
    seed: int
    gate: str = "055"
    gate_scale: str = "bps"
    n_folds: int = 5
    min_robust_min_relative: float = 0.0
    min_active_fraction: float = 0.01


PROFILES = {
    "promoted_tanh": Profile(name="promoted_tanh"),
    # Bias toward lower churn with mild smoothing and narrower max scale.
    "turnover_guard": Profile(
        name="turnover_guard",
        action_smoothing=1.12,
        scale_turnover_penalty=5.0,
        max_signal_scale=0.60,
    ),
    # Defensive profile for stress windows.
    "robust_guard": Profile(
        name="robust_guard",
        action_smoothing=1.08,
        scale_turnover_penalty=2.0,
        min_signal_scale=-0.60,
        max_signal_scale=0.60,
        no_trade_band=0.022,
    ),
}


def _summary_paths(case: Case, run_dir: Path) -> tuple[Path, Path]:
    stem = f"main4_gate{case.gate}_f{case.n_folds}_s{case.seed}"
    return run_dir / f"{stem}_summary.csv", run_dir / f"{stem}_detailed.json"


def _run_case(
    profile: Profile,
    case: Case,
    force: bool,
    path_bootstrap_reps: int,
    path_bootstrap_block_size: int,
    path_bootstrap_seed: int,
) -> dict[str, Any]:
    run_name = (
        f"{case.profile}_c{int(case.transaction_cost_bps)}"
        f"_s{case.seed}_g{case.gate}"
    )
    run_dir = RUN_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_csv, detailed_json = _summary_paths(case, run_dir)

    if force or not summary_csv.exists() or not detailed_json.exists():
        cmd = [
            str(PYTHON_BIN),
            "src/wealth_first/main4.py",
            "--gate",
            case.gate,
            "--gate-scale",
            case.gate_scale,
            "--n-folds",
            str(case.n_folds),
            "--seed",
            str(case.seed),
            "--output-dir",
            str(run_dir),
            "--transaction-cost-bps",
            str(case.transaction_cost_bps),
            "--slippage-bps",
            str(case.slippage_bps),
            "--ridge-l2",
            str(profile.ridge_l2),
            "--target-mode",
            profile.target_mode,
            "--scale-turnover-penalty",
            str(profile.scale_turnover_penalty),
            "--min-signal-scale",
            str(profile.min_signal_scale),
            "--max-signal-scale",
            str(profile.max_signal_scale),
            "--min-spy-weight",
            str(profile.min_spy_weight),
            "--max-spy-weight",
            str(profile.max_spy_weight),
            "--initial-spy-weight",
            str(profile.initial_spy_weight),
            "--action-smoothing",
            str(profile.action_smoothing),
            "--no-trade-band",
            str(profile.no_trade_band),
            "--min-robust-min-relative",
            str(case.min_robust_min_relative),
            "--min-active-fraction",
            str(case.min_active_fraction),
            "--path-bootstrap-reps",
            str(path_bootstrap_reps),
            "--path-bootstrap-block-size",
            str(path_bootstrap_block_size),
            "--path-bootstrap-seed",
            str(path_bootstrap_seed + case.seed),
        ]
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": "src"},
            capture_output=True,
            text=True,
        )
    else:
        result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="cached")

    if not summary_csv.exists() or not detailed_json.exists():
        raise RuntimeError(
            f"Case {run_name} did not produce expected artifacts.\n"
            f"returncode={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    with open(detailed_json, encoding="utf-8") as handle:
        detailed = json.load(handle)
    fold_summary = pd.read_csv(summary_csv)

    gate_checks = detailed.get("gate_checks", {})
    metrics = detailed.get("summary_metrics", {})
    gate_detail = gate_checks.get("validation_threshold", {})
    robust_detail = gate_checks.get("robust_min_threshold", {})
    active_detail = gate_checks.get("active_fraction_threshold", {})

    return {
        "run": run_name,
        "cached": bool(result.stderr == "cached"),
        "returncode": int(result.returncode),
        **asdict(profile),
        **asdict(case),
        "mean_test_relative": float(metrics.get("mean_test_relative_total_return", 0.0)),
        "mean_validation_relative": float(metrics.get("mean_validation_relative_total_return", 0.0)),
        "robust_min_test_relative": float(metrics.get("robust_min_test_relative", 0.0)),
        "beat_hold_fraction": float(metrics.get("beat_hold_fraction", 0.0)),
        "active_fraction": float(metrics.get("active_fraction", 0.0)),
        "mean_turnover": float(metrics.get("mean_turnover", 0.0)),
        "mean_test_executed_step_count": float(metrics.get("mean_test_executed_step_count", 0.0)),
        "mean_n_test_samples": float(
            fold_summary.get("n_test_samples", pd.Series([0.0])).mean()
        ),
        "mean_executed_step_rate": float(
            metrics.get("mean_test_executed_step_count", 0.0)
            / max(float(fold_summary.get("n_test_samples", pd.Series([1.0])).mean()), 1e-8)
        ),
        "worst_daily_relative_return": float(
            fold_summary.get("test_worst_daily_relative_return", pd.Series([0.0])).min()
        ),
        "worst_max_relative_drawdown": float(
            fold_summary.get("test_max_relative_drawdown", pd.Series([0.0])).max()
        ),
        "path_bootstrap_mean_test_relative_p05": float(
            metrics.get("path_bootstrap_mean_test_relative_p05", float("nan"))
        ),
        "path_bootstrap_mean_test_relative_p50": float(
            metrics.get("path_bootstrap_mean_test_relative_p50", float("nan"))
        ),
        "path_bootstrap_mean_test_relative_p95": float(
            metrics.get("path_bootstrap_mean_test_relative_p95", float("nan"))
        ),
        "path_bootstrap_robust_min_test_relative_p05": float(
            metrics.get("path_bootstrap_robust_min_test_relative_p05", float("nan"))
        ),
        "path_bootstrap_robust_min_test_relative_p50": float(
            metrics.get("path_bootstrap_robust_min_test_relative_p50", float("nan"))
        ),
        "path_bootstrap_robust_min_test_relative_p95": float(
            metrics.get("path_bootstrap_robust_min_test_relative_p95", float("nan"))
        ),
        "overall_gate_passed": bool(gate_checks.get("overall_passed", False)),
        "validation_gate_passed": bool(gate_detail.get("passed", False)),
        "validation_gate_value": float(gate_detail.get("value", 0.0)),
        "validation_gate_threshold": float(gate_detail.get("threshold", 0.0)),
        "robust_gate_passed": bool(robust_detail.get("passed", False)),
        "robust_gate_value": float(robust_detail.get("value", 0.0)),
        "robust_gate_threshold": float(robust_detail.get("threshold", 0.0)),
        "active_gate_passed": bool(active_detail.get("passed", False)),
        "active_gate_value": float(active_detail.get("value", 0.0)),
        "active_gate_threshold": float(active_detail.get("threshold", 0.0)),
        "summary_csv": str(summary_csv.relative_to(PROJECT_ROOT)),
        "detailed_json": str(detailed_json.relative_to(PROJECT_ROOT)),
    }


def _wealth_first_score(row: pd.Series) -> float:
    # Hard fail profiles should not outrank gate passers.
    gate_penalty = -1.0 if not row["overall_gate_passed"] else 0.0
    # Composite objective: reward return/robustness, penalize excess turnover.
    return (
        gate_penalty
        + float(row["mean_test_relative"])
        + 0.50 * float(row["robust_min_test_relative"])
        - 0.75 * float(row["mean_turnover"])
    )


def _build_cases(mode: str) -> list[Case]:
    if mode == "quick":
        seeds = [5, 17]
        friction_points = [12, 18, 22, 25]
        profile_names = ["promoted_tanh", "turnover_guard"]
    else:
        seeds = [5, 17, 29]
        friction_points = [12, 15, 18, 22, 25]
        profile_names = ["promoted_tanh", "turnover_guard", "robust_guard"]

    cases: list[Case] = []
    for profile in profile_names:
        for seed in seeds:
            for cost in friction_points:
                cases.append(
                    Case(
                        profile=profile,
                        transaction_cost_bps=float(cost),
                        slippage_bps=float(cost),
                        seed=int(seed),
                    )
                )
    return cases


def _build_profile_summary(table: pd.DataFrame) -> list[dict[str, Any]]:
    raise RuntimeError("_build_profile_summary requires gate and bootstrap settings")


def _bootstrap_profile_stats(
    grp: pd.DataFrame,
    reps: int,
    seed: int,
) -> dict[str, float]:
    """Block-bootstrap profile aggregates by friction point for uncertainty bounds."""
    if reps <= 0:
        return {
            "mean_score_bootstrap_p05": float("nan"),
            "mean_score_bootstrap_p50": float("nan"),
            "mean_score_bootstrap_p95": float("nan"),
            "mean_test_relative_bootstrap_p05": float("nan"),
            "mean_test_relative_bootstrap_p50": float("nan"),
            "mean_test_relative_bootstrap_p95": float("nan"),
        }

    rng = np.random.default_rng(seed)
    blocks = [
        block.reset_index(drop=True)
        for _, block in grp.groupby(["transaction_cost_bps", "slippage_bps"], sort=True)
    ]
    if not blocks:
        return {
            "mean_score_bootstrap_p05": float("nan"),
            "mean_score_bootstrap_p50": float("nan"),
            "mean_score_bootstrap_p95": float("nan"),
            "mean_test_relative_bootstrap_p05": float("nan"),
            "mean_test_relative_bootstrap_p50": float("nan"),
            "mean_test_relative_bootstrap_p95": float("nan"),
        }
    n_blocks = len(blocks)

    score_means = np.empty(reps, dtype=float)
    rel_means = np.empty(reps, dtype=float)
    for i in range(reps):
        block_idx = rng.integers(low=0, high=n_blocks, size=n_blocks)
        sample = pd.concat([blocks[j] for j in block_idx], ignore_index=True)
        score_means[i] = float(sample["wealth_first_score"].mean())
        rel_means[i] = float(sample["mean_test_relative"].mean())

    return {
        "mean_score_bootstrap_p05": float(np.quantile(score_means, 0.05)),
        "mean_score_bootstrap_p50": float(np.quantile(score_means, 0.50)),
        "mean_score_bootstrap_p95": float(np.quantile(score_means, 0.95)),
        "mean_test_relative_bootstrap_p05": float(np.quantile(rel_means, 0.05)),
        "mean_test_relative_bootstrap_p50": float(np.quantile(rel_means, 0.50)),
        "mean_test_relative_bootstrap_p95": float(np.quantile(rel_means, 0.95)),
    }


def _stable_profile_seed_offset(profile_name: str) -> int:
    digest = hashlib.sha256(profile_name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _build_profile_summary(
    table: pd.DataFrame,
    tail_worst_decile_threshold: float,
    robust_min_threshold: float,
    max_mean_turnover: float,
    min_worst_daily_relative_return: float,
    max_worst_relative_drawdown: float,
    min_mean_executed_step_rate: float,
    min_path_bootstrap_robust_min_p05: float,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for profile_name, grp in table.groupby("profile", sort=False):
        grp_sorted = grp.sort_values(["transaction_cost_bps", "seed"]).reset_index(drop=True)
        seed_std_rel = (
            grp_sorted.groupby(["transaction_cost_bps", "slippage_bps"], sort=True)["mean_test_relative"]
            .std(ddof=0)
            .fillna(0.0)
        )
        seed_std_robust = (
            grp_sorted.groupby(["transaction_cost_bps", "slippage_bps"], sort=True)["robust_min_test_relative"]
            .std(ddof=0)
            .fillna(0.0)
        )
        worst_decile_test_relative = float(grp_sorted["mean_test_relative"].quantile(0.10))
        worst_robust_min = float(grp_sorted["robust_min_test_relative"].min())
        mean_turnover = float(grp_sorted["mean_turnover"].mean())
        mean_executed_step_rate = float(grp_sorted["mean_executed_step_rate"].mean())
        worst_daily_relative_return = float(grp_sorted["worst_daily_relative_return"].min())
        worst_relative_drawdown = float(grp_sorted["worst_max_relative_drawdown"].max())
        mean_path_bootstrap_robust_min_p05 = float(
            grp_sorted["path_bootstrap_robust_min_test_relative_p05"].mean()
        )
        pass_rate = float(grp_sorted["overall_gate_passed"].mean())
        eligible = bool(
            pass_rate >= 0.999999
            and worst_decile_test_relative >= tail_worst_decile_threshold
            and worst_robust_min >= robust_min_threshold
            and mean_turnover <= max_mean_turnover
            and worst_daily_relative_return >= min_worst_daily_relative_return
            and worst_relative_drawdown <= max_worst_relative_drawdown
            and mean_executed_step_rate >= min_mean_executed_step_rate
            and mean_path_bootstrap_robust_min_p05 >= min_path_bootstrap_robust_min_p05
        )

        bootstrap = _bootstrap_profile_stats(
            grp_sorted,
            reps=bootstrap_reps,
            seed=bootstrap_seed + (_stable_profile_seed_offset(profile_name) % 100_000),
        )

        rows.append(
            {
                "profile": profile_name,
                "n_cases": int(len(grp_sorted)),
                "pass_rate": pass_rate,
                "eligible": eligible,
                "mean_score": float(grp_sorted["wealth_first_score"].mean()),
                "mean_test_relative": float(grp_sorted["mean_test_relative"].mean()),
                "worst_robust_min": worst_robust_min,
                "worst_decile_test_relative": worst_decile_test_relative,
                "mean_turnover": mean_turnover,
                "mean_executed_step_rate": mean_executed_step_rate,
                "worst_daily_relative_return": worst_daily_relative_return,
                "worst_relative_drawdown": worst_relative_drawdown,
                "mean_path_bootstrap_robust_min_p05": mean_path_bootstrap_robust_min_p05,
                "mean_seed_std_test_relative": float(seed_std_rel.mean()),
                "max_seed_std_test_relative": float(seed_std_rel.max()),
                "mean_seed_std_robust_min": float(seed_std_robust.mean()),
                "max_seed_std_robust_min": float(seed_std_robust.max()),
                "mean_path_bootstrap_mean_test_relative_p05": float(
                    grp_sorted["path_bootstrap_mean_test_relative_p05"].mean()
                ),
                "mean_path_bootstrap_mean_test_relative_p50": float(
                    grp_sorted["path_bootstrap_mean_test_relative_p50"].mean()
                ),
                "mean_path_bootstrap_mean_test_relative_p95": float(
                    grp_sorted["path_bootstrap_mean_test_relative_p95"].mean()
                ),
                "mean_path_bootstrap_robust_min_test_relative_p05": float(
                    grp_sorted["path_bootstrap_robust_min_test_relative_p05"].mean()
                ),
                "mean_path_bootstrap_robust_min_test_relative_p50": float(
                    grp_sorted["path_bootstrap_robust_min_test_relative_p50"].mean()
                ),
                "mean_path_bootstrap_robust_min_test_relative_p95": float(
                    grp_sorted["path_bootstrap_robust_min_test_relative_p95"].mean()
                ),
                **bootstrap,
                "min_gate_margin": float(
                    (
                        grp_sorted[[
                            "validation_gate_value",
                            "robust_gate_value",
                            "active_gate_value",
                        ]].values
                        - grp_sorted[[
                            "validation_gate_threshold",
                            "robust_gate_threshold",
                            "active_gate_threshold",
                        ]].values
                    ).min()
                ),
                "stress_rows": grp_sorted[
                    [
                        "transaction_cost_bps",
                        "slippage_bps",
                        "seed",
                        "mean_test_relative",
                        "robust_min_test_relative",
                        "mean_turnover",
                        "mean_executed_step_rate",
                        "worst_daily_relative_return",
                        "worst_max_relative_drawdown",
                        "path_bootstrap_robust_min_test_relative_p05",
                        "overall_gate_passed",
                        "wealth_first_score",
                    ]
                ].to_dict(orient="records"),
            }
        )

    rows.sort(key=lambda r: (r["eligible"], r["pass_rate"], r["mean_score"]), reverse=True)
    return rows


def _gate_slacks(profile_row: dict[str, Any], gates: dict[str, float]) -> dict[str, float]:
    return {
        "tail_worst_decile": float(profile_row["worst_decile_test_relative"] - gates["tail_worst_decile_threshold"]),
        "robust_min": float(profile_row["worst_robust_min"] - gates["robust_min_threshold"]),
        "mean_turnover": float(gates["max_mean_turnover"] - profile_row["mean_turnover"]),
        "mean_executed_step_rate": float(
            profile_row["mean_executed_step_rate"] - gates["min_mean_executed_step_rate"]
        ),
        "worst_daily_relative": float(
            profile_row["worst_daily_relative_return"] - gates["min_worst_daily_relative_return"]
        ),
        "worst_relative_drawdown": float(
            gates["max_worst_relative_drawdown"] - profile_row["worst_relative_drawdown"]
        ),
        "path_bootstrap_robust_min_p05": float(
            profile_row["mean_path_bootstrap_robust_min_p05"]
            - gates["min_path_bootstrap_robust_min_p05"]
        ),
    }


def _build_frontier_report(
    profile_summary: list[dict[str, Any]],
    gates: dict[str, float],
    best_profile: str | None,
) -> dict[str, Any]:
    if not profile_summary:
        return {}

    epsilon = 1e-6
    by_profile = {row["profile"]: row for row in profile_summary}

    profile_gate_limits: dict[str, dict[str, float]] = {}
    for profile_name, row in by_profile.items():
        profile_gate_limits[profile_name] = {
            "max_tail_worst_decile_threshold": float(row["worst_decile_test_relative"]),
            "max_robust_min_threshold": float(row["worst_robust_min"]),
            "min_max_mean_turnover_threshold": float(row["mean_turnover"]),
            "max_min_mean_executed_step_rate_threshold": float(row["mean_executed_step_rate"]),
            "max_min_worst_daily_relative_return_threshold": float(row["worst_daily_relative_return"]),
            "min_max_worst_relative_drawdown_threshold": float(row["worst_relative_drawdown"]),
            "max_min_path_bootstrap_robust_min_p05_threshold": float(row["mean_path_bootstrap_robust_min_p05"]),
        }

    target_profiles = [name for name in [best_profile, "promoted_tanh"] if name in by_profile]
    profile_flip_analysis: dict[str, Any] = {}
    for profile_name in dict.fromkeys(target_profiles):
        row = by_profile[profile_name]
        slacks = _gate_slacks(row, gates)
        binding_gate, binding_slack = min(slacks.items(), key=lambda kv: kv[1])
        profile_flip_analysis[profile_name] = {
            "currently_eligible": bool(row["eligible"]),
            "gate_slacks": slacks,
            "binding_gate": binding_gate,
            "binding_slack": float(binding_slack),
            "just_fail_threshold_examples": {
                "tail_worst_decile_threshold": float(row["worst_decile_test_relative"] + epsilon),
                "robust_min_threshold": float(row["worst_robust_min"] + epsilon),
                "max_mean_turnover": float(max(row["mean_turnover"] - epsilon, 0.0)),
                "min_mean_executed_step_rate": float(row["mean_executed_step_rate"] + epsilon),
                "min_worst_daily_relative_return": float(row["worst_daily_relative_return"] + epsilon),
                "max_worst_relative_drawdown": float(max(row["worst_relative_drawdown"] - epsilon, 0.0)),
                "min_path_bootstrap_robust_min_p05": float(
                    row["mean_path_bootstrap_robust_min_p05"] + epsilon
                ),
            },
        }

    return {
        "profile_gate_limits": profile_gate_limits,
        "profile_flip_analysis": profile_flip_analysis,
    }


def _collect_failed_gates(
    profile_row: dict[str, Any],
    gates: dict[str, float],
) -> list[str]:
    slacks = _gate_slacks(profile_row, gates)
    return [gate for gate, slack in slacks.items() if slack < 0.0]


def _eligible_profiles_for_gates(
    profile_summary: list[dict[str, Any]],
    gates: dict[str, float],
) -> list[str]:
    out: list[str] = []
    for row in profile_summary:
        slacks = _gate_slacks(row, gates)
        if all(value >= 0.0 for value in slacks.values()) and float(row["pass_rate"]) >= 0.999999:
            out.append(str(row["profile"]))
    return out


def _auto_calibrate_gates(
    profile_summary: list[dict[str, Any]],
    gates: dict[str, float],
    relaxation_weights: dict[str, float],
    min_feasible_profiles: int,
) -> dict[str, Any] | None:
    """Find nearest feasible gate set by minimally relaxing current thresholds."""
    if not profile_summary:
        return None

    scale = {
        "tail_worst_decile_threshold": 0.001,
        "robust_min_threshold": 0.001,
        "max_mean_turnover": 0.0001,
        "min_worst_daily_relative_return": 0.001,
        "max_worst_relative_drawdown": 0.001,
        "min_mean_executed_step_rate": 0.001,
        "min_path_bootstrap_robust_min_p05": 0.001,
    }

    best_candidate: dict[str, Any] | None = None
    fallback_best: dict[str, Any] | None = None
    for row in profile_summary:
        relaxed = {
            "tail_worst_decile_threshold": min(
                gates["tail_worst_decile_threshold"],
                float(row["worst_decile_test_relative"]),
            ),
            "robust_min_threshold": min(
                gates["robust_min_threshold"],
                float(row["worst_robust_min"]),
            ),
            "max_mean_turnover": max(
                gates["max_mean_turnover"],
                float(row["mean_turnover"]),
            ),
            "min_worst_daily_relative_return": min(
                gates["min_worst_daily_relative_return"],
                float(row["worst_daily_relative_return"]),
            ),
            "max_worst_relative_drawdown": max(
                gates["max_worst_relative_drawdown"],
                float(row["worst_relative_drawdown"]),
            ),
            "min_mean_executed_step_rate": min(
                gates["min_mean_executed_step_rate"],
                float(row["mean_executed_step_rate"]),
            ),
            "min_path_bootstrap_robust_min_p05": min(
                gates["min_path_bootstrap_robust_min_p05"],
                float(row["mean_path_bootstrap_robust_min_p05"]),
            ),
        }

        relaxation = {
            key: abs(float(relaxed[key] - gates[key])) for key in relaxed
        }
        normalized_relaxation = {
            key: float(relaxation[key] / max(scale[key], 1e-12)) for key in relaxation
        }
        weighted_relaxation_score = float(
            sum(
                float(relaxation_weights.get(key, 1.0)) * float(normalized_relaxation[key])
                for key in normalized_relaxation
            )
        )
        feasible_profiles = _eligible_profiles_for_gates(profile_summary, relaxed)
        feasible_count = len(feasible_profiles)
        meets_feasible_set_requirement = feasible_count >= int(min_feasible_profiles)

        candidate = {
            "target_profile": row["profile"],
            "relaxed_gates": relaxed,
            "relaxation": relaxation,
            "normalized_relaxation": normalized_relaxation,
            "weighted_relaxation_score": weighted_relaxation_score,
            "target_profile_mean_score": float(row["mean_score"]),
            "feasible_profiles": feasible_profiles,
            "feasible_count": int(feasible_count),
            "meets_feasible_set_requirement": bool(meets_feasible_set_requirement),
        }

        if fallback_best is None:
            fallback_best = candidate
        else:
            if candidate["weighted_relaxation_score"] < fallback_best["weighted_relaxation_score"]:
                fallback_best = candidate
            elif (
                candidate["weighted_relaxation_score"] == fallback_best["weighted_relaxation_score"]
                and candidate["target_profile_mean_score"] > fallback_best["target_profile_mean_score"]
            ):
                fallback_best = candidate

        if not meets_feasible_set_requirement:
            continue

        if best_candidate is None:
            best_candidate = candidate
            continue

        if candidate["weighted_relaxation_score"] < best_candidate["weighted_relaxation_score"]:
            best_candidate = candidate
            continue

        if (
            candidate["weighted_relaxation_score"] == best_candidate["weighted_relaxation_score"]
            and candidate["target_profile_mean_score"] > best_candidate["target_profile_mean_score"]
        ):
            best_candidate = candidate

    return best_candidate if best_candidate is not None else fallback_best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run phase-2 wealth-first stress sweep for main4.")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--force", action="store_true", help="Rerun cases even when artifacts exist.")
    parser.add_argument("--output-prefix", default="main4_phase2_wealthfirst_quick")
    parser.add_argument("--tail-worst-decile-threshold", type=float, default=0.0)
    parser.add_argument("--robust-min-threshold", type=float, default=0.0)
    parser.add_argument("--max-mean-turnover", type=float, default=0.0015)
    parser.add_argument("--min-worst-daily-relative-return", type=float, default=-0.010)
    parser.add_argument("--max-worst-relative-drawdown", type=float, default=0.030)
    parser.add_argument("--min-mean-executed-step-rate", type=float, default=0.0)
    parser.add_argument("--min-path-bootstrap-robust-min-p05", type=float, default=-1.0)
    parser.add_argument("--bootstrap-reps", type=int, default=400)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument("--path-bootstrap-reps", type=int, default=0)
    parser.add_argument("--path-bootstrap-block-size", type=int, default=20)
    parser.add_argument("--path-bootstrap-seed", type=int, default=12345)
    parser.add_argument("--auto-calibrate-gates", action="store_true")
    parser.add_argument("--auto-calibrate-apply", action="store_true")
    parser.add_argument("--auto-calibrate-min-feasible-profiles", type=int, default=1)
    parser.add_argument("--auto-calibrate-weight-tail-worst-decile", type=float, default=1.0)
    parser.add_argument("--auto-calibrate-weight-robust-min", type=float, default=1.0)
    parser.add_argument("--auto-calibrate-weight-mean-turnover", type=float, default=1.0)
    parser.add_argument("--auto-calibrate-weight-mean-executed-step-rate", type=float, default=1.0)
    parser.add_argument("--auto-calibrate-weight-worst-daily-relative", type=float, default=1.0)
    parser.add_argument("--auto-calibrate-weight-worst-relative-drawdown", type=float, default=1.0)
    parser.add_argument("--auto-calibrate-weight-path-bootstrap-robust-min-p05", type=float, default=1.0)
    args = parser.parse_args(argv)

    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"Missing venv Python: {PYTHON_BIN}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    cases = _build_cases(args.mode)

    print("=" * 72)
    print(f"Running main4 phase-2 wealth-first sweep ({args.mode})")
    print("=" * 72)

    rows: list[dict[str, Any]] = []
    for case in cases:
        profile = PROFILES[case.profile]
        print(
            f"\n{case.profile:14s} seed={case.seed:2d} "
            f"cost={case.transaction_cost_bps + case.slippage_bps:4.1f}bps"
        )
        row = _run_case(
            profile=profile,
            case=case,
            force=args.force,
            path_bootstrap_reps=int(args.path_bootstrap_reps),
            path_bootstrap_block_size=int(args.path_bootstrap_block_size),
            path_bootstrap_seed=int(args.path_bootstrap_seed),
        )
        rows.append(row)
        print(
            f"  rel={row['mean_test_relative']:+.6f} "
            f"robust={row['robust_min_test_relative']:+.6f} "
            f"turn={row['mean_turnover']:.6f} "
            f"gate={'PASS' if row['overall_gate_passed'] else 'FAIL'}"
        )

    table = pd.DataFrame(rows)
    table["wealth_first_score"] = table.apply(_wealth_first_score, axis=1)

    detail_csv = ARTIFACT_ROOT / f"{args.output_prefix}_detail.csv"
    summary_json = ARTIFACT_ROOT / f"{args.output_prefix}_summary.json"

    table.sort_values(
        ["profile", "transaction_cost_bps", "seed"],
        ascending=[True, True, True],
    ).to_csv(detail_csv, index=False)

    profile_summary = _build_profile_summary(
        table=table,
        tail_worst_decile_threshold=float(args.tail_worst_decile_threshold),
        robust_min_threshold=float(args.robust_min_threshold),
        max_mean_turnover=float(args.max_mean_turnover),
        min_worst_daily_relative_return=float(args.min_worst_daily_relative_return),
        max_worst_relative_drawdown=float(args.max_worst_relative_drawdown),
        min_mean_executed_step_rate=float(args.min_mean_executed_step_rate),
        min_path_bootstrap_robust_min_p05=float(args.min_path_bootstrap_robust_min_p05),
        bootstrap_reps=int(args.bootstrap_reps),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    strict_best_profile = profile_summary[0]["profile"] if profile_summary else None

    warnings: list[str] = []
    if profile_summary:
        max_seed_variability = max(p["max_seed_std_test_relative"] for p in profile_summary)
        if max_seed_variability < 1e-10:
            warnings.append(
                "Seed variability is effectively zero across all profiles in this sweep; "
                "seed may be non-informative for this deterministic model path."
            )
        if not any(p["eligible"] for p in profile_summary):
            warnings.append(
                "No profile met all wealth-first eligibility gates; consider relaxing thresholds "
                "or expanding profile search space."
            )

    strict_gates = {
        "tail_worst_decile_threshold": float(args.tail_worst_decile_threshold),
        "robust_min_threshold": float(args.robust_min_threshold),
        "max_mean_turnover": float(args.max_mean_turnover),
        "min_worst_daily_relative_return": float(args.min_worst_daily_relative_return),
        "max_worst_relative_drawdown": float(args.max_worst_relative_drawdown),
        "min_mean_executed_step_rate": float(args.min_mean_executed_step_rate),
        "min_path_bootstrap_robust_min_p05": float(args.min_path_bootstrap_robust_min_p05),
    }

    strict_eligible_profiles = [row["profile"] for row in profile_summary if row["eligible"]]
    strict_has_eligible = bool(strict_eligible_profiles)
    strict_feasible_profile_count = len(strict_eligible_profiles)

    relaxation_weights = {
        "tail_worst_decile_threshold": float(args.auto_calibrate_weight_tail_worst_decile),
        "robust_min_threshold": float(args.auto_calibrate_weight_robust_min),
        "max_mean_turnover": float(args.auto_calibrate_weight_mean_turnover),
        "min_mean_executed_step_rate": float(args.auto_calibrate_weight_mean_executed_step_rate),
        "min_worst_daily_relative_return": float(args.auto_calibrate_weight_worst_daily_relative),
        "max_worst_relative_drawdown": float(args.auto_calibrate_weight_worst_relative_drawdown),
        "min_path_bootstrap_robust_min_p05": float(args.auto_calibrate_weight_path_bootstrap_robust_min_p05),
    }

    auto_calibration: dict[str, Any] | None = None
    calibrated_summary: list[dict[str, Any]] | None = None
    if args.auto_calibrate_gates and not strict_has_eligible:
        auto_calibration = _auto_calibrate_gates(
            profile_summary,
            strict_gates,
            relaxation_weights=relaxation_weights,
            min_feasible_profiles=max(int(args.auto_calibrate_min_feasible_profiles), 1),
        )
        if auto_calibration is not None and args.auto_calibrate_apply:
            calibrated_summary = _build_profile_summary(
                table=table,
                tail_worst_decile_threshold=float(
                    auto_calibration["relaxed_gates"]["tail_worst_decile_threshold"]
                ),
                robust_min_threshold=float(
                    auto_calibration["relaxed_gates"]["robust_min_threshold"]
                ),
                max_mean_turnover=float(auto_calibration["relaxed_gates"]["max_mean_turnover"]),
                min_worst_daily_relative_return=float(
                    auto_calibration["relaxed_gates"]["min_worst_daily_relative_return"]
                ),
                max_worst_relative_drawdown=float(
                    auto_calibration["relaxed_gates"]["max_worst_relative_drawdown"]
                ),
                min_mean_executed_step_rate=float(
                    auto_calibration["relaxed_gates"]["min_mean_executed_step_rate"]
                ),
                min_path_bootstrap_robust_min_p05=float(
                    auto_calibration["relaxed_gates"]["min_path_bootstrap_robust_min_p05"]
                ),
                bootstrap_reps=int(args.bootstrap_reps),
                bootstrap_seed=int(args.bootstrap_seed),
            )

    selected_gates = (
        auto_calibration["relaxed_gates"]
        if (auto_calibration is not None and calibrated_summary is not None)
        else strict_gates
    )
    selected_eligible_profiles = _eligible_profiles_for_gates(
        calibrated_summary if calibrated_summary is not None else profile_summary,
        selected_gates,
    )

    failed_gate_counts: dict[str, int] = {
        "tail_worst_decile": 0,
        "robust_min": 0,
        "mean_turnover": 0,
        "mean_executed_step_rate": 0,
        "worst_daily_relative": 0,
        "worst_relative_drawdown": 0,
        "path_bootstrap_robust_min_p05": 0,
    }
    for row in profile_summary:
        for gate in _collect_failed_gates(row, strict_gates):
            failed_gate_counts[gate] += 1

    selected_profile_summary = calibrated_summary if (calibrated_summary is not None) else profile_summary
    best_profile = selected_profile_summary[0]["profile"] if selected_profile_summary else None

    summary = {
        "mode": args.mode,
        "n_cases": int(len(table)),
        "best_profile": best_profile,
        "eligibility_gates": strict_gates,
        "bootstrap": {
            "reps": int(args.bootstrap_reps),
            "seed": int(args.bootstrap_seed),
        },
        "path_bootstrap": {
            "reps": int(args.path_bootstrap_reps),
            "block_size": int(args.path_bootstrap_block_size),
            "seed": int(args.path_bootstrap_seed),
        },
        "frontier_report": _build_frontier_report(
            profile_summary=selected_profile_summary,
            gates=selected_gates,
            best_profile=best_profile,
        ),
        "promotion_report": {
            "strict_feasible": strict_has_eligible,
            "strict_feasible_profile_count": int(strict_feasible_profile_count),
            "strict_best_profile_by_score": strict_best_profile,
            "strict_best_eligible_profile": strict_eligible_profiles[0] if strict_eligible_profiles else None,
            "strict_failed_gate_counts": failed_gate_counts,
            "auto_calibration_requested": bool(args.auto_calibrate_gates),
            "auto_calibration_applied": bool(calibrated_summary is not None),
            "auto_calibration_min_feasible_profiles": int(max(args.auto_calibrate_min_feasible_profiles, 1)),
            "auto_calibration_relaxation_weights": relaxation_weights,
            "auto_calibration": auto_calibration,
            "selected_gates": selected_gates,
            "selected_best_profile": best_profile,
            "selected_eligible_profiles": selected_eligible_profiles,
            "selected_feasible_profile_count": int(len(selected_eligible_profiles)),
        },
        "profile_ranking": selected_profile_summary,
        "strict_profile_ranking": profile_summary,
        "calibrated_profile_ranking": calibrated_summary,
        "warnings": warnings,
    }

    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n" + "=" * 72)
    print("Phase-2 Wealth-First Summary")
    print("=" * 72)
    print(f"Wrote {detail_csv.name}")
    print(f"Wrote {summary_json.name}")
    if best_profile is not None:
        top = profile_summary[0]
        print(
            f"Best profile: {top['profile']} "
            f"(pass_rate={top['pass_rate']:.3f}, mean_score={top['mean_score']:+.6f}, "
            f"worst_robust_min={top['worst_robust_min']:+.6f})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
