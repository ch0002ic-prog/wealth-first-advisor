#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def candidate_index(candidate: str) -> int:
    # Example: l_s092625478_objw1 -> 478
    token = candidate.split("_")[1]  # s092625478
    return int(token[7:])


def candidate_branch(candidate: str) -> str:
    return candidate.split("_")[0]


def monotonic(values: list[float]) -> bool:
    if len(values) < 2:
        return True
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def summarize_errors(errors: list[float]) -> dict:
    if not errors:
        return {
            "count": 0,
            "mae": None,
            "rmse": None,
            "max_abs_error": None,
        }
    abs_errors = [abs(x) for x in errors]
    sq_errors = [x * x for x in errors]
    return {
        "count": len(errors),
        "mae": sum(abs_errors) / len(abs_errors),
        "rmse": math.sqrt(sum(sq_errors) / len(sq_errors)),
        "max_abs_error": max(abs_errors),
    }


def validation_family(path: Path) -> str:
    name = path.name.lower()
    if "surrogate" in name:
        return "surrogate"
    if "piecewise" in name:
        return "piecewise"
    if "blind_forward" in name:
        return "blind_forward"
    if "reverse_blind" in name:
        return "reverse_blind"
    return "other"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate main5 tolerance ladder consistency and prediction quality."
    )
    parser.add_argument(
        "--ladder-json",
        type=Path,
        default=Path("artifacts/main5_execution_gate_tolerance_continuity_ladder_extended_2026-04-28-v4.json"),
        help="Path to ladder JSON artifact.",
    )
    parser.add_argument(
        "--validation-jsons",
        type=Path,
        nargs="*",
        default=[
            Path("artifacts/main5_execution_gate_tolerance_surrogate_validation_2026-04-28.json"),
            Path("artifacts/main5_execution_gate_tolerance_piecewise_validation_2026-04-28.json"),
            Path("artifacts/main5_execution_gate_tolerance_blind_forward_validation_2026-04-28.json"),
            Path("artifacts/main5_execution_gate_tolerance_reverse_blind_validation_2026-04-28.json"),
        ],
        help="Validation JSON artifacts to aggregate.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("artifacts/main5_execution_gate_tolerance_evaluator_2026-04-28.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("artifacts/main5_execution_gate_tolerance_evaluator_2026-04-28.md"),
        help="Output markdown summary path.",
    )
    parser.add_argument(
        "--robust-summary",
        action="store_true",
        help="Include per-validation-family error summaries and robust aggregate metrics.",
    )
    parser.add_argument(
        "--robust-mae-max",
        type=float,
        default=None,
        help="Optional fail threshold for robust MAE.",
    )
    parser.add_argument(
        "--robust-rmse-max",
        type=float,
        default=None,
        help="Optional fail threshold for robust RMSE.",
    )
    parser.add_argument(
        "--robust-max-abs-error-max",
        type=float,
        default=None,
        help="Optional fail threshold for robust max absolute error.",
    )
    parser.add_argument(
        "--required-families",
        type=str,
        default="",
        help=(
            "Comma-separated validation families that must be present "
            "(e.g., surrogate,piecewise,blind_forward,reverse_blind)."
        ),
    )
    parser.add_argument(
        "--min-validation-rows-per-family",
        type=int,
        default=0,
        help="Minimum validation rows required for each required family.",
    )
    args = parser.parse_args()

    ladder_data = load_json(args.ladder_json)
    ladder = ladder_data.get("ladder", [])
    if not ladder:
        raise ValueError(f"No ladder entries found in {args.ladder_json}")

    # Global monotonicity over provided ordering.
    good_values_global = [float(item["good_from"]) for item in ladder]
    global_monotonic = monotonic(good_values_global)

    # Branch-specific monotonicity by candidate index order.
    branches: dict[str, list[dict]] = {"l": [], "k": []}
    for item in ladder:
        b = candidate_branch(item["candidate"])
        if b in branches:
            branches[b].append(item)

    branch_monotonic = {}
    for b, items in branches.items():
        items_sorted = sorted(items, key=lambda x: candidate_index(x["candidate"]))
        vals = [float(x["good_from"]) for x in items_sorted]
        branch_monotonic[b] = {
            "monotonic_non_decreasing": monotonic(vals),
            "count": len(vals),
        }

    # Interval width quality checks.
    finite_widths = []
    invalid_width_candidates = []
    for item in ladder:
        bad = item.get("bad_through")
        good = float(item["good_from"])
        if bad is None:
            continue
        width = good - float(bad)
        finite_widths.append(width)
        if width <= 0:
            invalid_width_candidates.append(item["candidate"])

    interval_summary = {
        "count_with_finite_bad_through": len(finite_widths),
        "min_width": min(finite_widths) if finite_widths else None,
        "max_width": max(finite_widths) if finite_widths else None,
        "mean_width": (sum(finite_widths) / len(finite_widths)) if finite_widths else None,
        "invalid_width_candidates": invalid_width_candidates,
    }

    # Aggregate prediction errors from validation artifacts.
    prediction_errors = []
    prediction_error_rows = []
    prediction_errors_by_family: dict[str, list[float]] = {}
    validation_rows_by_family: dict[str, int] = {}
    missing_validation_files = []
    for vp in args.validation_jsons:
        if not vp.exists():
            missing_validation_files.append(str(vp))
            continue
        fam = validation_family(vp)
        prediction_errors_by_family.setdefault(fam, [])
        validation_rows_by_family.setdefault(fam, 0)
        data = load_json(vp)
        validation = data.get("validation", {})
        validation_rows_by_family[fam] += len(validation)
        for candidate, rec in validation.items():
            err = None
            if "pred_vs_midpoint_rel_error" in rec:
                err = rec["pred_vs_midpoint_rel_error"]
            elif "relative_error_vs_midpoint" in rec:
                err = rec["relative_error_vs_midpoint"]
            if err is None:
                continue
            err_f = float(err)
            prediction_errors.append(err_f)
            prediction_errors_by_family[fam].append(err_f)
            prediction_error_rows.append(
                {
                    "candidate": candidate,
                    "relative_error": err_f,
                    "source_file": str(vp),
                    "family": fam,
                }
            )

    error_summary = summarize_errors(prediction_errors)
    error_summary_by_family = {
        fam: summarize_errors(vals) for fam, vals in prediction_errors_by_family.items()
    }

    robust_errors = []
    for row in prediction_error_rows:
        # Exclude surrogate (historical coarse model) and reverse-blind from robust aggregate.
        # Surrogate is excluded because k52 (k_s09262552_objw1) has a structural 61.854%
        # extrapolation error caused by non-log-linear manifold shape on the k-branch; the
        # surrogate family must not serve as a promotion signal (see
        # main5_surrogate_outlier_exclusion_policy_2026-04-28.md for full diagnosis).
        # Reverse-blind is excluded because midpoint-relative error is undefined at the zero edge.
        if row["family"] in {"surrogate", "reverse_blind"}:
            continue
        robust_errors.append(row["relative_error"])
    robust_error_summary = summarize_errors(robust_errors)

    robust_thresholds = {
        "mae_max": args.robust_mae_max,
        "rmse_max": args.robust_rmse_max,
        "max_abs_error_max": args.robust_max_abs_error_max,
    }
    robust_threshold_check = {
        "enabled": any(v is not None for v in robust_thresholds.values()),
        "failed": False,
        "violations": [],
    }

    required_families = [
        x.strip() for x in args.required_families.split(",") if x.strip()
    ]
    family_coverage_check = {
        "required_families": required_families,
        "min_validation_rows_per_family": args.min_validation_rows_per_family,
        "failed": False,
        "violations": [],
    }
    for fam in required_families:
        count = validation_rows_by_family.get(fam, 0)
        if count < args.min_validation_rows_per_family:
            family_coverage_check["failed"] = True
            family_coverage_check["violations"].append(
                f"family {fam} validation_rows {count} < {args.min_validation_rows_per_family}"
            )

    if robust_thresholds["mae_max"] is not None and robust_error_summary["mae"] is not None:
        if robust_error_summary["mae"] > robust_thresholds["mae_max"]:
            robust_threshold_check["failed"] = True
            robust_threshold_check["violations"].append(
                f"robust_mae {robust_error_summary['mae']} > {robust_thresholds['mae_max']}"
            )
    if robust_thresholds["rmse_max"] is not None and robust_error_summary["rmse"] is not None:
        if robust_error_summary["rmse"] > robust_thresholds["rmse_max"]:
            robust_threshold_check["failed"] = True
            robust_threshold_check["violations"].append(
                f"robust_rmse {robust_error_summary['rmse']} > {robust_thresholds['rmse_max']}"
            )
    if (
        robust_thresholds["max_abs_error_max"] is not None
        and robust_error_summary["max_abs_error"] is not None
    ):
        if robust_error_summary["max_abs_error"] > robust_thresholds["max_abs_error_max"]:
            robust_threshold_check["failed"] = True
            robust_threshold_check["violations"].append(
                "robust_max_abs_error "
                f"{robust_error_summary['max_abs_error']} > {robust_thresholds['max_abs_error_max']}"
            )

    result = {
        "status": "pass",
        "ladder_file": str(args.ladder_json),
        "entry_count": len(ladder),
        "global_monotonic_non_decreasing": global_monotonic,
        "branch_monotonicity": branch_monotonic,
        "interval_summary": interval_summary,
        "prediction_error_summary": error_summary,
        "prediction_error_summary_by_family": error_summary_by_family,
        "validation_rows_by_family": validation_rows_by_family,
        "prediction_error_summary_robust": robust_error_summary,
        "robust_thresholds": robust_thresholds,
        "robust_threshold_check": robust_threshold_check,
        "family_coverage_check": family_coverage_check,
        "prediction_error_rows": prediction_error_rows,
        "missing_validation_files": missing_validation_files,
    }

    if not global_monotonic:
        result["status"] = "fail"
    if any(not v["monotonic_non_decreasing"] for v in branch_monotonic.values()):
        result["status"] = "fail"
    if interval_summary["invalid_width_candidates"]:
        result["status"] = "fail"
    if robust_threshold_check["failed"]:
        result["status"] = "fail"
    if family_coverage_check["failed"]:
        result["status"] = "fail"

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    md_lines = []
    md_lines.append("# Main5 Tolerance Ladder Evaluator")
    md_lines.append("")
    md_lines.append(f"- status: {result['status']}")
    md_lines.append(f"- ladder_entries: {result['entry_count']}")
    md_lines.append(f"- global_monotonic_non_decreasing: {result['global_monotonic_non_decreasing']}")
    md_lines.append("")
    md_lines.append("## Branch Checks")
    for branch, info in branch_monotonic.items():
        md_lines.append(
            f"- branch {branch}: monotonic_non_decreasing={info['monotonic_non_decreasing']}, count={info['count']}"
        )
    md_lines.append("")
    md_lines.append("## Interval Widths")
    md_lines.append(f"- finite_count: {interval_summary['count_with_finite_bad_through']}")
    md_lines.append(f"- min_width: {interval_summary['min_width']}")
    md_lines.append(f"- max_width: {interval_summary['max_width']}")
    md_lines.append(f"- mean_width: {interval_summary['mean_width']}")
    md_lines.append(f"- invalid_width_candidates: {interval_summary['invalid_width_candidates']}")
    md_lines.append("")
    md_lines.append("## Prediction Error Summary")
    md_lines.append(f"- count: {error_summary['count']}")
    md_lines.append(f"- mae: {error_summary['mae']}")
    md_lines.append(f"- rmse: {error_summary['rmse']}")
    md_lines.append(f"- max_abs_error: {error_summary['max_abs_error']}")
    md_lines.append("")
    if validation_rows_by_family:
        md_lines.append("## Validation Coverage By Family")
        for fam in sorted(validation_rows_by_family):
            md_lines.append(f"- {fam}: validation_rows={validation_rows_by_family[fam]}")
        md_lines.append("")
    if args.robust_summary:
        md_lines.append("## Prediction Error Summary By Family")
        for fam in sorted(error_summary_by_family):
            fam_summary = error_summary_by_family[fam]
            md_lines.append(
                f"- {fam}: count={fam_summary['count']}, "
                f"mae={fam_summary['mae']}, rmse={fam_summary['rmse']}, "
                f"max_abs_error={fam_summary['max_abs_error']}"
            )
        md_lines.append("")
        md_lines.append("## Robust Aggregate Error Summary")
        md_lines.append("- note: excludes surrogate and reverse_blind families")
        md_lines.append(f"- count: {robust_error_summary['count']}")
        md_lines.append(f"- mae: {robust_error_summary['mae']}")
        md_lines.append(f"- rmse: {robust_error_summary['rmse']}")
        md_lines.append(f"- max_abs_error: {robust_error_summary['max_abs_error']}")
        md_lines.append("")
    if robust_threshold_check["enabled"]:
        md_lines.append("## Robust Threshold Check")
        md_lines.append(f"- mae_max: {robust_thresholds['mae_max']}")
        md_lines.append(f"- rmse_max: {robust_thresholds['rmse_max']}")
        md_lines.append(f"- max_abs_error_max: {robust_thresholds['max_abs_error_max']}")
        md_lines.append(f"- failed: {robust_threshold_check['failed']}")
        md_lines.append(f"- violations: {robust_threshold_check['violations']}")
        md_lines.append("")
    if required_families:
        md_lines.append("## Family Coverage Check")
        md_lines.append(f"- required_families: {required_families}")
        md_lines.append(
            f"- min_validation_rows_per_family: {args.min_validation_rows_per_family}"
        )
        md_lines.append(f"- failed: {family_coverage_check['failed']}")
        md_lines.append(f"- violations: {family_coverage_check['violations']}")
        md_lines.append("")
    if missing_validation_files:
        md_lines.append("## Missing Validation Files")
        for p in missing_validation_files:
            md_lines.append(f"- {p}")

    args.out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())