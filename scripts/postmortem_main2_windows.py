from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from wealth_first.data import load_returns_csv
from wealth_first.main2 import Main2Config, compute_main2_dynamic_floor_components


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enrich main2 detail CSV rows with recent-path state and summarize weak pockets without manual CSV inspection."
    )
    parser.add_argument("--detail-csv", required=True, help="Main2 detail CSV to analyze.")
    parser.add_argument("--returns-csv", default="data/demo_sleeves.csv", help="Return CSV containing the SPY benchmark series.")
    parser.add_argument("--benchmark-column", default="SPY_BENCHMARK", help="Benchmark column used as the main2 SPY source.")
    parser.add_argument("--date-column", default=None, help="Optional date column for the returns CSV.")
    parser.add_argument("--delta-threshold", type=float, default=-0.08, help="Rows at or below this static-hold delta are tagged as weak.")
    parser.add_argument("--top-n", type=int, default=12, help="Maximum number of weakest rows to retain in the summary payload.")
    parser.add_argument("--output-prefix", default="artifacts/main2_window_postmortem", help="Output path prefix for the enriched CSV and summary JSON.")
    parser.add_argument("--participation-floor-min-spy-weight", type=float, default=None)
    parser.add_argument("--participation-floor-lookback", type=int, default=126)
    parser.add_argument("--participation-floor-return-threshold", type=float, default=0.02)
    parser.add_argument("--participation-floor-ma-gap-threshold", type=float, default=0.02)
    parser.add_argument("--participation-floor-drawdown-threshold", type=float, default=-0.05)
    parser.add_argument("--recovery-floor-min-spy-weight", type=float, default=None)
    parser.add_argument("--recovery-floor-long-lookback", type=int, default=126)
    parser.add_argument("--recovery-floor-drawdown-threshold", type=float, default=-0.10)
    parser.add_argument("--recovery-floor-short-lookback", type=int, default=21)
    parser.add_argument("--recovery-floor-return-threshold", type=float, default=0.01)
    parser.add_argument("--recovery-floor-ma-gap-threshold", type=float, default=0.0)
    parser.add_argument("--early-crack-floor-min-spy-weight", type=float, default=None)
    parser.add_argument("--early-crack-long-lookback", type=int, default=126)
    parser.add_argument("--early-crack-long-return-threshold", type=float, default=0.10)
    parser.add_argument("--early-crack-long-ma-gap-threshold", type=float, default=0.04)
    parser.add_argument("--early-crack-long-drawdown-threshold", type=float, default=-0.03)
    parser.add_argument("--early-crack-trend-ma-gap-threshold", type=float, default=0.02)
    parser.add_argument("--early-crack-short-lookback", type=int, default=21)
    parser.add_argument("--early-crack-short-return-threshold", type=float, default=0.0)
    parser.add_argument("--early-crack-short-ma-gap-threshold", type=float, default=0.0)
    parser.add_argument("--early-crack-short-drawdown-threshold", type=float, default=-0.02)
    parser.add_argument("--constructive-crack-cap-min-spy-weight", type=float, default=None)
    parser.add_argument("--constructive-crack-cap-recent-constructive-lookback", type=int, default=15)
    parser.add_argument("--constructive-crack-cap-long-lookback", type=int, default=126)
    parser.add_argument("--constructive-crack-cap-long-return-threshold", type=float, default=0.10)
    parser.add_argument("--constructive-crack-cap-long-ma-gap-threshold", type=float, default=0.03)
    parser.add_argument("--constructive-crack-cap-long-drawdown-threshold", type=float, default=-0.04)
    parser.add_argument("--constructive-crack-cap-short-lookback", type=int, default=21)
    parser.add_argument("--constructive-crack-cap-short-return-threshold", type=float, default=0.0)
    parser.add_argument("--constructive-crack-cap-short-ma-gap-threshold", type=float, default=0.0)
    parser.add_argument("--constructive-crack-cap-short-drawdown-threshold", type=float, default=-0.015)
    parser.add_argument("--constructive-crack-cap-current-trend-return-cap", type=float, default=0.05)
    parser.add_argument("--constructive-crack-cap-current-trend-ma-gap-cap", type=float, default=0.015)
    parser.add_argument("--constructive-crack-cap-current-long-return-min", type=float, default=0.09)
    parser.add_argument("--constructive-crack-cap-current-long-return-max", type=float, default=0.15)
    return parser


def _config_from_args(args: argparse.Namespace) -> Main2Config:
    return Main2Config(
        participation_floor_min_spy_weight=args.participation_floor_min_spy_weight,
        participation_floor_lookback=args.participation_floor_lookback,
        participation_floor_return_threshold=args.participation_floor_return_threshold,
        participation_floor_ma_gap_threshold=args.participation_floor_ma_gap_threshold,
        participation_floor_drawdown_threshold=args.participation_floor_drawdown_threshold,
        recovery_floor_min_spy_weight=args.recovery_floor_min_spy_weight,
        recovery_floor_long_lookback=args.recovery_floor_long_lookback,
        recovery_floor_drawdown_threshold=args.recovery_floor_drawdown_threshold,
        recovery_floor_short_lookback=args.recovery_floor_short_lookback,
        recovery_floor_return_threshold=args.recovery_floor_return_threshold,
        recovery_floor_ma_gap_threshold=args.recovery_floor_ma_gap_threshold,
        early_crack_floor_min_spy_weight=args.early_crack_floor_min_spy_weight,
        early_crack_long_lookback=args.early_crack_long_lookback,
        early_crack_long_return_threshold=args.early_crack_long_return_threshold,
        early_crack_long_ma_gap_threshold=args.early_crack_long_ma_gap_threshold,
        early_crack_long_drawdown_threshold=args.early_crack_long_drawdown_threshold,
        early_crack_trend_ma_gap_threshold=args.early_crack_trend_ma_gap_threshold,
        early_crack_short_lookback=args.early_crack_short_lookback,
        early_crack_short_return_threshold=args.early_crack_short_return_threshold,
        early_crack_short_ma_gap_threshold=args.early_crack_short_ma_gap_threshold,
        early_crack_short_drawdown_threshold=args.early_crack_short_drawdown_threshold,
        constructive_crack_cap_min_spy_weight=args.constructive_crack_cap_min_spy_weight,
        constructive_crack_cap_recent_constructive_lookback=args.constructive_crack_cap_recent_constructive_lookback,
        constructive_crack_cap_long_lookback=args.constructive_crack_cap_long_lookback,
        constructive_crack_cap_long_return_threshold=args.constructive_crack_cap_long_return_threshold,
        constructive_crack_cap_long_ma_gap_threshold=args.constructive_crack_cap_long_ma_gap_threshold,
        constructive_crack_cap_long_drawdown_threshold=args.constructive_crack_cap_long_drawdown_threshold,
        constructive_crack_cap_short_lookback=args.constructive_crack_cap_short_lookback,
        constructive_crack_cap_short_return_threshold=args.constructive_crack_cap_short_return_threshold,
        constructive_crack_cap_short_ma_gap_threshold=args.constructive_crack_cap_short_ma_gap_threshold,
        constructive_crack_cap_short_drawdown_threshold=args.constructive_crack_cap_short_drawdown_threshold,
        constructive_crack_cap_current_trend_return_cap=args.constructive_crack_cap_current_trend_return_cap,
        constructive_crack_cap_current_trend_ma_gap_cap=args.constructive_crack_cap_current_trend_ma_gap_cap,
        constructive_crack_cap_current_long_return_min=args.constructive_crack_cap_current_long_return_min,
        constructive_crack_cap_current_long_return_max=args.constructive_crack_cap_current_long_return_max,
    )


def _group_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "rows": int(len(frame)),
        "mean_delta_total_return_vs_static_hold": float(frame["delta_total_return_vs_static_hold"].mean()),
        "mean_policy_cash_weight": float(frame["policy_cash_weight"].mean()),
        "mean_policy_turnover": float(frame["policy_turnover"].mean()),
        "mean_ret_21": float(frame["ret_21"].mean()),
        "mean_dd_21": float(frame["dd_21"].mean()),
        "mean_ma_gap_21": float(frame["ma_gap_21"].mean()),
        "mean_ret_63": float(frame["ret_63"].mean()),
        "mean_dd_63": float(frame["dd_63"].mean()),
        "mean_ma_gap_63": float(frame["ma_gap_63"].mean()),
        "mean_participation_ret": float(frame["participation_ret"].mean()),
        "mean_participation_ma_gap": float(frame["participation_ma_gap"].mean()),
        "mean_recovery_long_dd": float(frame["recovery_long_dd"].mean()),
        "mean_recovery_short_ret": float(frame["recovery_short_ret"].mean()),
        "mean_effective_trend_floor_min_spy_weight": float(frame["effective_trend_floor_min_spy_weight"].mean()),
        "mean_early_crack_long_ret": float(frame["early_crack_long_ret"].mean()),
        "mean_early_crack_long_dd": float(frame["early_crack_long_dd"].mean()),
        "mean_early_crack_long_ma_gap": float(frame["early_crack_long_ma_gap"].mean()),
        "mean_early_crack_short_ret": float(frame["early_crack_short_ret"].mean()),
        "mean_early_crack_short_dd": float(frame["early_crack_short_dd"].mean()),
        "mean_early_crack_short_ma_gap": float(frame["early_crack_short_ma_gap"].mean()),
        "early_crack_hit_rate": float(frame["early_crack_active"].mean()),
        "mean_constructive_crack_recent_constructive_days_ago": float(frame["constructive_crack_recent_constructive_days_ago"].mean()),
        "mean_constructive_crack_long_ret": float(frame["constructive_crack_long_ret"].mean()),
        "mean_constructive_crack_long_dd": float(frame["constructive_crack_long_dd"].mean()),
        "mean_constructive_crack_long_ma_gap": float(frame["constructive_crack_long_ma_gap"].mean()),
        "mean_constructive_crack_short_ret": float(frame["constructive_crack_short_ret"].mean()),
        "mean_constructive_crack_short_dd": float(frame["constructive_crack_short_dd"].mean()),
        "mean_constructive_crack_short_ma_gap": float(frame["constructive_crack_short_ma_gap"].mean()),
        "constructive_crack_recent_constructive_hit_rate": float(frame["constructive_crack_recent_constructive_active"].mean()),
        "constructive_crack_cap_hit_rate": float(frame["constructive_crack_cap_active"].mean()),
        "participation_floor_hit_rate": float(frame["participation_floor_active"].mean()),
        "recovery_floor_hit_rate": float(frame["recovery_floor_active"].mean()),
        "trend_floor_hit_rate": float(frame["trend_floor_active"].mean()),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = _config_from_args(args)

    detail_path = Path(args.detail_csv)
    if not detail_path.is_absolute():
        detail_path = PROJECT_ROOT / detail_path
    detail = pd.read_csv(detail_path)

    returns = load_returns_csv(args.returns_csv, date_column=args.date_column)
    if args.benchmark_column not in returns.columns:
        raise ValueError(f"Benchmark column '{args.benchmark_column}' was not found in the returns CSV.")
    spy_returns = returns[args.benchmark_column].astype(float)

    enriched_rows: list[dict[str, object]] = []
    for row in detail.to_dict(orient="records"):
        start_label = pd.Timestamp(row["phase_start"])
        row_index = spy_returns.index.get_loc(start_label)
        if isinstance(row_index, slice):
            row_index = row_index.start
        floor_components = compute_main2_dynamic_floor_components(spy_returns, row_index=int(row_index), config=config)
        enriched_row = dict(row)
        enriched_row.update(floor_components)
        enriched_rows.append(enriched_row)

    enriched = pd.DataFrame(enriched_rows)
    weak_rows = enriched.loc[enriched["delta_total_return_vs_static_hold"] <= args.delta_threshold].copy()
    weakest_rows = enriched.sort_values("delta_total_return_vs_static_hold").head(min(args.top_n, len(enriched)))

    summary = {
        "detail_csv": str(detail_path),
        "delta_threshold": float(args.delta_threshold),
        "overall": _group_summary(enriched),
        "weak_rows": _group_summary(weak_rows) if not weak_rows.empty else {"rows": 0},
        "weak_rows_by_window": {
            f"{split}|{fold}|{phase}": _group_summary(group)
            for (split, fold, phase), group in weak_rows.groupby(["split", "fold", "phase"])
        },
        "weakest_rows": weakest_rows.to_dict(orient="records"),
    }

    output_prefix = Path(args.output_prefix)
    if not output_prefix.is_absolute():
        output_prefix = PROJECT_ROOT / output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    enriched_path = output_prefix.with_name(f"{output_prefix.name}_enriched.csv")
    summary_path = output_prefix.with_name(f"{output_prefix.name}_summary.json")

    enriched.round(6).to_csv(enriched_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(enriched_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())