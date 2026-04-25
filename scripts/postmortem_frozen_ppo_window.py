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
    write_json,
)


def _discover_seed_dirs(artifacts_dir: Path, prefix: str) -> list[Path]:
    return discover_seed_dirs(artifacts_dir, prefix)


def _replay_runtime_overlay_frame(
    frame: pd.DataFrame,
    *,
    runtime_overlay: str | None,
    trend_symbol: str = "TREND_FOLLOWING",
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    return replay_runtime_overlay_frame(frame, runtime_overlay=runtime_overlay, trend_symbol=trend_symbol)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Postmortem a saved PPO fold window.")
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--split-label", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--runtime-overlay", default="none")
    parser.add_argument("--output-prefix", required=True)
    return parser


def _trade_budget_delta(fold_dir: Path, label: str, policy_total_return: float) -> float | None:
    comparison_path = fold_dir / f"policy_vs_{label}_comparison.json"
    if not comparison_path.exists():
        return None
    payload = read_json(comparison_path)
    saved_difference = float(payload.get("test", {}).get("total_return", {}).get("difference", 0.0))
    saved_policy_total_return = float(read_json(fold_dir / "test_policy_summary.json").get("total_return", 0.0))
    baseline_total_return = saved_policy_total_return - saved_difference
    return float(policy_total_return - baseline_total_return)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts_dir = Path(args.artifacts_dir)
    output_prefix = Path(args.output_prefix)
    rows: list[dict[str, object]] = []

    for seed_dir in _discover_seed_dirs(artifacts_dir, args.prefix):
        seed = int(seed_dir.name.removeprefix(args.prefix))
        for fold_dir in fold_dirs(seed_dir):
            policy_summary = read_json(fold_dir / "test_policy_summary.json")
            static_hold_summary = read_json(fold_dir / "test_static_hold_summary.json")
            optimizer_summary = read_json(fold_dir / "test_optimizer_summary.json")
            rollout = pd.read_csv(fold_dir / "test_policy_rollout.csv")
            allow_overlay = args.split_name == "regime"
            replayed_frame, metrics = replay_runtime_overlay_frame(
                rollout,
                runtime_overlay=args.runtime_overlay,
                allow_overlay=allow_overlay,
            )

            if not allow_overlay or args.runtime_overlay == "none":
                metrics["policy_total_return"] = float(policy_summary.get("total_return", 0.0))
                metrics["policy_relative_total_return"] = float(policy_summary.get("relative_total_return", 0.0))
                metrics["policy_turnover"] = float(policy_summary.get("total_turnover", 0.0))
                metrics["policy_cash_weight"] = float(policy_summary.get("average_cash_weight", 0.0))
                metrics["policy_trend_weight"] = float(policy_summary.get("average_target_weight_TREND_FOLLOWING", 0.0))

            phase_start, phase_end = phase_window(seed_dir, fold_dir.name, "test")
            row = {
                "split": args.split_label,
                "seed": seed,
                "fold": fold_dir.name,
                "phase": "test",
                "phase_start": phase_start,
                "phase_end": phase_end,
                "saved_policy_total_return": float(policy_summary.get("total_return", 0.0)),
                "policy_total_return": float(metrics["policy_total_return"]),
                "delta_total_return_vs_saved_policy": float(metrics["policy_total_return"]) - float(policy_summary.get("total_return", 0.0)),
                "delta_total_return_vs_static_hold": float(metrics["policy_total_return"]) - float(static_hold_summary.get("total_return", 0.0)),
                "delta_total_return_vs_optimizer": float(metrics["policy_total_return"]) - float(optimizer_summary.get("total_return", 0.0)),
                "overlay_applied_steps": int(metrics["overlay_applied_steps"]),
                "overlay_suppressed_steps": int(metrics["overlay_suppressed_steps"]),
                "policy_turnover": float(metrics["policy_turnover"]),
                "policy_cash_weight": float(metrics["policy_cash_weight"]),
                "policy_trend_weight": float(metrics["policy_trend_weight"]),
            }
            for label in ["trade_budget_1", "trade_budget_2", "trade_budget_3"]:
                row[f"delta_total_return_vs_{label}"] = _trade_budget_delta(fold_dir, label, float(metrics["policy_total_return"]))
            rows.append(row)

    detail = pd.DataFrame(rows)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_prefix.with_name(f"{output_prefix.name}_detail.csv"), index=False)
    summary = {
        "rows": int(len(detail)),
        "runtime_overlay": {
            "mode": args.runtime_overlay,
            "active": bool(detail["overlay_applied_steps"].sum()) if not detail.empty else False,
        },
        "event_summary": {
            "rows": int((detail["delta_total_return_vs_static_hold"] < -1e-12).sum()) if not detail.empty else 0,
        },
    }
    write_json(output_prefix.with_name(f"{output_prefix.name}_summary.json"), summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
