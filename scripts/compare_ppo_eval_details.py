from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wealth_first.ppo_analysis_common import write_json


MERGE_KEYS = ["split", "seed", "fold", "phase", "phase_start", "phase_end"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare PPO evaluation detail CSVs.")
    parser.add_argument("--base-detail-csv", required=True)
    parser.add_argument("--candidate-detail-csv", required=True)
    parser.add_argument("--output-prefix", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = pd.read_csv(args.base_detail_csv)
    candidate = pd.read_csv(args.candidate_detail_csv)
    merged = candidate.merge(base, on=MERGE_KEYS, suffixes=("_candidate", "_base"))
    merged["improve_vs_static_hold"] = (
        merged["delta_total_return_vs_static_hold_candidate"]
        - merged["delta_total_return_vs_static_hold_base"]
    )
    merged["improve_vs_optimizer"] = (
        merged["delta_total_return_vs_optimizer_candidate"]
        - merged["delta_total_return_vs_optimizer_base"]
    )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_prefix.with_name(f"{output_prefix.name}_detail.csv"), index=False)

    by_split = merged.groupby("split", dropna=False)[["improve_vs_static_hold", "improve_vs_optimizer"]].mean()
    summary = {
        "rows": int(len(merged)),
        "chrono_mean_improve_vs_static_hold": (
            float(by_split.loc["chrono", "improve_vs_static_hold"]) if "chrono" in by_split.index else None
        ),
        "regime_mean_improve_vs_static_hold": (
            float(by_split.loc["regime", "improve_vs_static_hold"]) if "regime" in by_split.index else None
        ),
        "mean_improve_vs_static_hold_by_split": {
            split: float(values["improve_vs_static_hold"])
            for split, values in by_split.iterrows()
        },
        "mean_improve_vs_optimizer_by_split": {
            split: float(values["improve_vs_optimizer"])
            for split, values in by_split.iterrows()
        },
    }
    write_json(output_prefix.with_name(f"{output_prefix.name}_summary.json"), summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
