#!/usr/bin/env python3
"""Run main4 repro suite for comparison with main3."""
import json
import os
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"


@dataclass(frozen=True)
class Case:
    name: str
    args: list[str]


CASES = [
    Case(
        name="main4_gate009_f3_s10",
        args=[
            "--gate", "009",
            "--n-folds", "3",
        ],
    ),
]

SEEDS = [7, 17, 27, 37, 47, 57, 67, 77, 87, 97]


def _safe_float(value: Any) -> float | None:
    """Convert value to float when possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_or_none(series: pd.Series) -> float | None:
    """Return mean as float or None when empty."""
    if len(series) == 0:
        return None
    return float(series.mean())


def main() -> int:
    """Run main4 repro suite."""
    output_dir = PROJECT_ROOT / "artifacts"
    output_dir.mkdir(exist_ok=True)

    for case in CASES:
        print(f"\n{'=' * 70}")
        print(f"Running {case.name}")
        print(f"{'=' * 70}")

        all_results: list[pd.DataFrame] = []
        seed_diagnostics: list[dict[str, Any]] = []

        for seed in SEEDS:
            cmd = [
                str(PYTHON_BIN),
                "src/wealth_first/main4.py",
                *case.args,
                "--seed", str(seed),
                "--output-dir", str(output_dir),
            ]

            print(f"\nSeed {seed:3d}: ", end="", flush=True)
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                env={**os.environ, "PYTHONPATH": "src"},
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print(f"FAILED")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
                return 1

            # Extract summary metrics from stderr
            for line in result.stderr.split("\n"):
                if "Mean test relative return:" in line:
                    val = float(line.split()[-1])
                    print(f"rel_return={val:+.6f}", end=" ")
                elif "Beat hold:" in line:
                    val = float(line.split()[-1].rstrip("%")) / 100.0
                    print(f"beat={val:.1%}", end=" ")
                elif "Active fraction:" in line:
                    val = float(line.split()[2].rstrip(",%")) / 100.0
                    print(f"active={val:.1%}", end=" ")

            # Load summary CSV
            csv_path = output_dir / f"main4_gate009_f3_s{seed}_summary.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                all_results.append(df)

            # Load detailed JSON for config-level snapshots
            detailed_path = output_dir / f"main4_gate009_f3_s{seed}_detailed.json"
            detailed_json: dict[str, Any] = {}
            if detailed_path.exists():
                with open(detailed_path) as f:
                    detailed_json = json.load(f)

            seed_rows = df[df["seed"] == seed] if csv_path.exists() else pd.DataFrame()
            seed_diag: dict[str, Any] = {
                "seed": int(seed),
                "n_rows": int(len(seed_rows)),
                "mean_relative": _mean_or_none(seed_rows["policy_relative_total_return"]) if "policy_relative_total_return" in seed_rows else None,
                "mean_absolute": _mean_or_none(seed_rows["policy_total_return"]) if "policy_total_return" in seed_rows else None,
                "active_fraction": _mean_or_none(seed_rows["active"]) if "active" in seed_rows else None,
                "mean_turnover": _mean_or_none(seed_rows["mean_turnover"]) if "mean_turnover" in seed_rows else None,
                "mean_spy_weight": _mean_or_none(seed_rows["mean_spy_weight"]) if "mean_spy_weight" in seed_rows else None,
                "mean_signal_scale": _mean_or_none(seed_rows["signal_scale"]) if "signal_scale" in seed_rows else None,
                "mean_signal_bias": _mean_or_none(seed_rows["signal_bias"]) if "signal_bias" in seed_rows else None,
                "action_rate_summary": {
                    "n_active_folds": int(seed_rows["active"].sum()) if "active" in seed_rows else 0,
                    "n_total_folds": int(len(seed_rows)),
                    "inactive_folds": int(len(seed_rows) - seed_rows["active"].sum()) if "active" in seed_rows else 0,
                },
                "parameter_snapshot": {
                    "gate": str(case.args[1]),
                    "n_folds": int(case.args[3]),
                    "signal_scale_values": [float(x) for x in seed_rows["signal_scale"].tolist()] if "signal_scale" in seed_rows else [],
                    "signal_bias_values": [float(x) for x in seed_rows["signal_bias"].tolist()] if "signal_bias" in seed_rows else [],
                    "config_medium_capacity": detailed_json.get("config", {}).get("medium_capacity_cfg", {}),
                },
            }
            seed_diagnostics.append(seed_diag)

            print()

        # Consolidate results
        if all_results:
            consolidated = pd.concat(all_results, ignore_index=True)
            output_csv = output_dir / f"main4_repro_{case.name}_detail.csv"
            consolidated.to_csv(output_csv, index=False)
            print(f"\nWrote {output_csv.name}")

            # Compute summary metrics
            mean_relative = consolidated["policy_relative_total_return"].mean()
            beat_hold = (consolidated["policy_relative_total_return"] > 0).sum() / len(consolidated)
            active_rate = consolidated["active"].mean()
            mean_turnover = consolidated["mean_turnover"].mean()

            summary = {
                "case": case.name,
                "seeds": [int(s) for s in SEEDS],
                "n_rows": len(consolidated),
                "mean_test_relative": float(mean_relative),
                "beat_hold_fraction": float(beat_hold),
                "active_fraction": float(active_rate),
                "mean_turnover": float(mean_turnover),
                "seed_variability": {
                    "metric": "policy_relative_total_return_mean_by_seed",
                    "n_seeds": int(consolidated["seed"].nunique()) if "seed" in consolidated.columns else 0,
                    "mean_of_seed_means": None,
                    "std_of_seed_means": None,
                    "min_seed_mean": None,
                    "max_seed_mean": None,
                },
                "parameter_snapshot": {
                    "mean_signal_scale": _safe_float(consolidated["signal_scale"].mean()) if "signal_scale" in consolidated.columns else None,
                    "std_signal_scale": _safe_float(consolidated["signal_scale"].std(ddof=0)) if "signal_scale" in consolidated.columns else None,
                    "mean_signal_bias": _safe_float(consolidated["signal_bias"].mean()) if "signal_bias" in consolidated.columns else None,
                    "std_signal_bias": _safe_float(consolidated["signal_bias"].std(ddof=0)) if "signal_bias" in consolidated.columns else None,
                },
                "action_rate_summary": {
                    "n_active_rows": int(consolidated["active"].sum()) if "active" in consolidated.columns else 0,
                    "n_total_rows": int(len(consolidated)),
                    "inactive_rows": int(len(consolidated) - consolidated["active"].sum()) if "active" in consolidated.columns else 0,
                },
                "seed_diagnostics": seed_diagnostics,
            }

            if "seed" in consolidated.columns and "policy_relative_total_return" in consolidated.columns:
                by_seed = consolidated.groupby("seed", as_index=False)["policy_relative_total_return"].mean()
                summary["seed_variability"].update(
                    {
                        "mean_of_seed_means": _safe_float(by_seed["policy_relative_total_return"].mean()),
                        "std_of_seed_means": _safe_float(by_seed["policy_relative_total_return"].std(ddof=0)),
                        "min_seed_mean": _safe_float(by_seed["policy_relative_total_return"].min()),
                        "max_seed_mean": _safe_float(by_seed["policy_relative_total_return"].max()),
                    }
                )

            output_json = output_dir / f"main4_repro_{case.name}_summary.json"
            with open(output_json, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Wrote {output_json.name}")

            print(f"\n{'=' * 70}")
            print(f"Summary ({case.name}):")
            print(f"  Rows: {summary['n_rows']}")
            print(f"  Mean test relative:  {summary['mean_test_relative']:+.6f}")
            print(f"  Beat hold rate:      {summary['beat_hold_fraction']:.1%}")
            print(f"  Active rate:         {summary['active_fraction']:.1%}")
            print(f"  Mean turnover:       {summary['mean_turnover']:.6f}")
            print("  Seed variability:")
            std_seed = summary["seed_variability"]["std_of_seed_means"]
            min_seed = summary["seed_variability"]["min_seed_mean"]
            max_seed = summary["seed_variability"]["max_seed_mean"]
            std_text = f"{std_seed:.6f}" if std_seed is not None else "n/a"
            min_text = f"{min_seed:+.6f}" if min_seed is not None else "n/a"
            max_text = f"{max_seed:+.6f}" if max_seed is not None else "n/a"
            print(f"    std(seed means):   {std_text}")
            print(f"    min/max means:     {min_text} / {max_text}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
