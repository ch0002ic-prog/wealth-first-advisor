from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wealth_first.ppo_analysis_common import (
    discover_seed_dirs,
    fold_dirs,
    phase_window,
    read_json,
    replay_runtime_overlay_frame,
    requested_splits,
    write_json,
)


def _discover_seed_dirs(artifacts_dir: Path, prefix: str) -> list[Path]:
    return discover_seed_dirs(artifacts_dir, prefix)


def _requested_splits(regime_prefix: str | None, chrono_prefix: str | None) -> list[tuple[str, str]]:
    return requested_splits(regime_prefix, chrono_prefix)


def _replay_runtime_overlay(
    frame: pd.DataFrame,
    *,
    runtime_overlay: str | None,
    trend_symbol: str = "TREND_FOLLOWING",
) -> dict[str, float | int]:
    _, metrics = replay_runtime_overlay_frame(frame, runtime_overlay=runtime_overlay, trend_symbol=trend_symbol)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate saved frozen PPO baseline artifacts.")
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--regime-prefix")
    parser.add_argument("--chrono-prefix")
    parser.add_argument("--runtime-overlay", default="none")
    parser.add_argument("--output-prefix", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts_dir = Path(args.artifacts_dir)
    output_prefix = Path(args.output_prefix)
    rows: list[dict[str, object]] = []

    for split, prefix in _requested_splits(args.regime_prefix, args.chrono_prefix):
        for seed_dir in _discover_seed_dirs(artifacts_dir, prefix):
            seed = int(seed_dir.name.removeprefix(prefix))
            for fold_dir in fold_dirs(seed_dir):
                for phase in ["validation", "test"]:
                    policy_summary = read_json(fold_dir / f"{phase}_policy_summary.json")
                    static_hold_summary = read_json(fold_dir / f"{phase}_static_hold_summary.json")
                    optimizer_summary = read_json(fold_dir / f"{phase}_optimizer_summary.json")
                    rollout_path = fold_dir / f"{phase}_policy_rollout.csv"

                    if rollout_path.exists():
                        rollout = pd.read_csv(rollout_path)
                        _, metrics = replay_runtime_overlay_frame(
                            rollout,
                            runtime_overlay=args.runtime_overlay,
                            allow_overlay=split == "regime",
                        )
                    else:
                        metrics = {
                            "overlay_applied_steps": 0,
                            "overlay_suppressed_steps": 0,
                        }

                    if split != "regime" or args.runtime_overlay == "none" or not rollout_path.exists():
                        metrics["policy_total_return"] = float(policy_summary.get("total_return", 0.0))
                        metrics["policy_relative_total_return"] = float(policy_summary.get("relative_total_return", 0.0))
                        metrics["policy_turnover"] = float(policy_summary.get("total_turnover", 0.0))
                        metrics["policy_cash_weight"] = float(policy_summary.get("average_cash_weight", 0.0))
                        metrics["policy_trend_weight"] = float(policy_summary.get("average_target_weight_TREND_FOLLOWING", 0.0))

                    phase_start, phase_end = phase_window(seed_dir, fold_dir.name, phase)
                    rows.append(
                        {
                            "split": split,
                            "seed": seed,
                            "fold": fold_dir.name,
                            "phase": phase,
                            "phase_start": phase_start,
                            "phase_end": phase_end,
                            "saved_policy_total_return": float(policy_summary.get("total_return", 0.0)),
                            "policy_total_return": float(metrics["policy_total_return"]),
                            "policy_relative_total_return": float(metrics.get("policy_relative_total_return", policy_summary.get("relative_total_return", 0.0))),
                            "delta_total_return_vs_saved_policy": float(metrics["policy_total_return"]) - float(policy_summary.get("total_return", 0.0)),
                            "delta_total_return_vs_static_hold": float(metrics["policy_total_return"]) - float(static_hold_summary.get("total_return", 0.0)),
                            "delta_total_return_vs_optimizer": float(metrics["policy_total_return"]) - float(optimizer_summary.get("total_return", 0.0)),
                            "overlay_applied_steps": int(metrics["overlay_applied_steps"]),
                            "overlay_suppressed_steps": int(metrics["overlay_suppressed_steps"]),
                            "policy_turnover": float(metrics["policy_turnover"]),
                            "policy_cash_weight": float(metrics["policy_cash_weight"]),
                            "policy_trend_weight": float(metrics["policy_trend_weight"]),
                        }
                    )

    detail = pd.DataFrame(rows)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_prefix.with_name(f"{output_prefix.name}_detail.csv"), index=False)

    by_split: dict[str, dict[str, float]] = {}
    for split, split_frame in detail.groupby("split", dropna=False):
        by_split[str(split)] = {
            "mean_delta_total_return_vs_saved_policy": float(split_frame["delta_total_return_vs_saved_policy"].mean()),
            "mean_delta_total_return_vs_static_hold": float(split_frame["delta_total_return_vs_static_hold"].mean()),
            "mean_policy_total_return": float(split_frame["policy_total_return"].mean()),
            "rows": int(len(split_frame)),
        }

    summary = {
        "rows": int(len(detail)),
        "runtime_overlay": {
            "mode": args.runtime_overlay,
            "active": bool(detail["overlay_applied_steps"].sum()) if not detail.empty else False,
        },
        "by_split": by_split,
    }
    write_json(output_prefix.with_name(f"{output_prefix.name}_summary.json"), summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())