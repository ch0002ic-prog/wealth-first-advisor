from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from wealth_first.data import load_returns_csv


@dataclass(frozen=True)
class ColumnComparison:
    reference_column: str
    candidate_column: str
    overlapping_rows: int
    mean_difference: float
    mean_abs_error: float
    rmse: float
    max_abs_error: float
    correlation: float
    mismatch_count: int
    mismatch_rate: float


@dataclass(frozen=True)
class ComparisonReport:
    reference_rows: int
    candidate_rows: int
    overlapping_dates: int
    reference_only_dates: int
    candidate_only_dates: int
    reference_only_columns: list[str]
    candidate_only_columns: list[str]
    column_comparisons: list[ColumnComparison]

    def to_frame(self) -> pd.DataFrame:
        if not self.column_comparisons:
            return pd.DataFrame(
                columns=[
                    "reference_column",
                    "candidate_column",
                    "overlapping_rows",
                    "mean_difference",
                    "mean_abs_error",
                    "rmse",
                    "max_abs_error",
                    "correlation",
                    "mismatch_count",
                    "mismatch_rate",
                ]
            )
        return pd.DataFrame(asdict(comparison) for comparison in self.column_comparisons)


def _parse_column_mappings(raw_mappings: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_mapping in raw_mappings or []:
        if "=" not in raw_mapping:
            raise ValueError(f"Invalid column mapping '{raw_mapping}'. Expected format is candidate=reference.")
        candidate_column, reference_column = raw_mapping.split("=", 1)
        candidate_column = candidate_column.strip()
        reference_column = reference_column.strip()
        if not candidate_column or not reference_column:
            raise ValueError(f"Invalid column mapping '{raw_mapping}'. Expected format is candidate=reference.")
        parsed[candidate_column] = reference_column
    return parsed


def _resolve_column_mapping(
    reference_columns: pd.Index,
    candidate_columns: pd.Index,
    candidate_to_reference: dict[str, str] | None = None,
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for candidate_column, reference_column in (candidate_to_reference or {}).items():
        if candidate_column not in candidate_columns:
            raise ValueError(f"Candidate column '{candidate_column}' was not found.")
        if reference_column not in reference_columns:
            raise ValueError(f"Reference column '{reference_column}' was not found.")
        resolved[candidate_column] = reference_column

    for shared_column in sorted(set(reference_columns).intersection(set(candidate_columns))):
        resolved.setdefault(shared_column, shared_column)

    reference_targets = list(resolved.values())
    if len(reference_targets) != len(set(reference_targets)):
        raise ValueError("Each reference column can only be mapped once.")
    return resolved


def compare_return_streams(
    reference_returns: pd.DataFrame,
    candidate_returns: pd.DataFrame,
    candidate_to_reference: dict[str, str] | None = None,
    tolerance: float = 1e-8,
) -> ComparisonReport:
    reference = reference_returns.sort_index().copy()
    candidate = candidate_returns.sort_index().copy()
    mapping = _resolve_column_mapping(reference.columns, candidate.columns, candidate_to_reference=candidate_to_reference)

    overlapping_dates = reference.index.intersection(candidate.index)
    comparisons: list[ColumnComparison] = []
    for candidate_column, reference_column in mapping.items():
        aligned = pd.concat(
            [reference[reference_column], candidate[candidate_column]],
            axis=1,
            join="inner",
        ).dropna(how="any")
        if aligned.empty:
            comparisons.append(
                ColumnComparison(
                    reference_column=reference_column,
                    candidate_column=candidate_column,
                    overlapping_rows=0,
                    mean_difference=np.nan,
                    mean_abs_error=np.nan,
                    rmse=np.nan,
                    max_abs_error=np.nan,
                    correlation=np.nan,
                    mismatch_count=0,
                    mismatch_rate=np.nan,
                )
            )
            continue

        reference_series = aligned.iloc[:, 0]
        candidate_series = aligned.iloc[:, 1]
        differences = candidate_series - reference_series
        absolute_differences = differences.abs()
        mismatch_count = int((absolute_differences > tolerance).sum())
        comparisons.append(
            ColumnComparison(
                reference_column=reference_column,
                candidate_column=candidate_column,
                overlapping_rows=int(len(aligned)),
                mean_difference=float(differences.mean()),
                mean_abs_error=float(absolute_differences.mean()),
                rmse=float(np.sqrt(np.mean(np.square(differences)))),
                max_abs_error=float(absolute_differences.max()),
                correlation=float(reference_series.corr(candidate_series)) if len(aligned) > 1 else np.nan,
                mismatch_count=mismatch_count,
                mismatch_rate=float(mismatch_count / len(aligned)),
            )
        )

    mapped_reference_columns = set(mapping.values())
    mapped_candidate_columns = set(mapping.keys())
    return ComparisonReport(
        reference_rows=int(len(reference)),
        candidate_rows=int(len(candidate)),
        overlapping_dates=int(len(overlapping_dates)),
        reference_only_dates=int(len(reference.index.difference(candidate.index))),
        candidate_only_dates=int(len(candidate.index.difference(reference.index))),
        reference_only_columns=sorted(set(reference.columns) - mapped_reference_columns),
        candidate_only_columns=sorted(set(candidate.columns) - mapped_candidate_columns),
        column_comparisons=comparisons,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare TradingView-exported sleeve returns against Python-generated sleeve returns.")
    parser.add_argument("--reference-csv", required=True, help="CSV file containing the Python-generated reference return streams.")
    parser.add_argument("--candidate-csv", required=True, help="CSV file containing the TradingView or external candidate return streams.")
    parser.add_argument("--reference-date-column", default=None, help="Optional date column name for the reference CSV.")
    parser.add_argument("--candidate-date-column", default=None, help="Optional date column name for the candidate CSV.")
    parser.add_argument(
        "--column-map",
        action="append",
        default=[],
        help="Map external candidate columns to reference columns using candidate=reference. Repeat as needed.",
    )
    parser.add_argument("--tolerance", type=float, default=1e-8, help="Absolute per-row tolerance used for mismatch counting.")
    parser.add_argument("--max-mae", type=float, default=None, help="Optional maximum mean absolute error per compared sleeve.")
    parser.add_argument("--min-correlation", type=float, default=None, help="Optional minimum correlation per compared sleeve.")
    parser.add_argument("--max-mismatch-rate", type=float, default=None, help="Optional maximum mismatch rate per compared sleeve.")
    parser.add_argument("--output-csv", default=None, help="Optional path for saving the comparison metrics table.")
    return parser


def _print_report(report: ComparisonReport) -> None:
    print("Comparison Summary")
    print(f"  reference_rows: {report.reference_rows}")
    print(f"  candidate_rows: {report.candidate_rows}")
    print(f"  overlapping_dates: {report.overlapping_dates}")
    print(f"  reference_only_dates: {report.reference_only_dates}")
    print(f"  candidate_only_dates: {report.candidate_only_dates}")
    print(f"  reference_only_columns: {', '.join(report.reference_only_columns) if report.reference_only_columns else 'none'}")
    print(f"  candidate_only_columns: {', '.join(report.candidate_only_columns) if report.candidate_only_columns else 'none'}")

    frame = report.to_frame()
    if frame.empty:
        print("\nNo columns were matched between the two return sets.")
        return

    rounded = frame.copy()
    for column in ["mean_difference", "mean_abs_error", "rmse", "max_abs_error", "correlation", "mismatch_rate"]:
        rounded[column] = rounded[column].astype(float).round(8)
    print("\nPer-Column Metrics")
    print(rounded.to_string(index=False))


def _validate_thresholds(
    report: ComparisonReport,
    max_mae: float | None,
    min_correlation: float | None,
    max_mismatch_rate: float | None,
) -> list[str]:
    breaches: list[str] = []
    frame = report.to_frame()
    if frame.empty:
        breaches.append("No overlapping columns were available for comparison.")
        return breaches

    if max_mae is not None:
        offending = frame.loc[frame["mean_abs_error"] > max_mae, ["reference_column", "candidate_column", "mean_abs_error"]]
        for _, row in offending.iterrows():
            breaches.append(
                f"Mean absolute error {row['mean_abs_error']:.8f} exceeded max_mae for {row['candidate_column']} -> {row['reference_column']}."
            )

    if min_correlation is not None:
        offending = frame.loc[frame["correlation"] < min_correlation, ["reference_column", "candidate_column", "correlation"]]
        for _, row in offending.iterrows():
            breaches.append(
                f"Correlation {row['correlation']:.8f} fell below min_correlation for {row['candidate_column']} -> {row['reference_column']}."
            )

    if max_mismatch_rate is not None:
        offending = frame.loc[frame["mismatch_rate"] > max_mismatch_rate, ["reference_column", "candidate_column", "mismatch_rate"]]
        for _, row in offending.iterrows():
            breaches.append(
                f"Mismatch rate {row['mismatch_rate']:.8f} exceeded max_mismatch_rate for {row['candidate_column']} -> {row['reference_column']}."
            )

    return breaches


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    reference_returns = load_returns_csv(args.reference_csv, date_column=args.reference_date_column)
    candidate_returns = load_returns_csv(args.candidate_csv, date_column=args.candidate_date_column)
    report = compare_return_streams(
        reference_returns,
        candidate_returns,
        candidate_to_reference=_parse_column_mappings(args.column_map),
        tolerance=args.tolerance,
    )
    _print_report(report)

    if args.output_csv:
        report.to_frame().to_csv(args.output_csv, index=False)
        print(f"\nSaved per-column metrics to {args.output_csv}")

    breaches = _validate_thresholds(
        report=report,
        max_mae=args.max_mae,
        min_correlation=args.min_correlation,
        max_mismatch_rate=args.max_mismatch_rate,
    )
    if breaches:
        print("\nThreshold Breaches")
        for breach in breaches:
            print(f"  {breach}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())