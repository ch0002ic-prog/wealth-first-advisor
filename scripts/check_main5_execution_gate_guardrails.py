#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


KEYS = ["candidate", "scenario", "seed", "transaction_cost_bps"]
METRIC_COLS = [
    "mean_test_relative",
    "mean_validation_relative",
    "mean_turnover",
    "mean_test_executed_step_count",
    "mean_test_gate_suppression_rate",
    "path_bootstrap_robust_min_p05",
    "case_slack",
]


@dataclass(frozen=True)
class DiffResult:
    matched_rows: int
    changed_rows: int
    moved_candidates: list[str]


def strict_diff(base_path: Path, alt_path: Path, tol: float = 1e-15) -> DiffResult:
    base = pd.read_csv(base_path)
    alt = pd.read_csv(alt_path)
    merged = base.merge(alt, on=KEYS, suffixes=("_b", "_t"))

    row_changed = pd.Series(False, index=merged.index)
    moved = set()

    for col in METRIC_COLS:
        delta = (merged[f"{col}_t"] - merged[f"{col}_b"]).abs()
        changed = delta > tol
        row_changed = row_changed | changed

    for candidate, grp in merged.groupby("candidate"):
        cand_changed = False
        for col in METRIC_COLS:
            delta = (grp[f"{col}_t"] - grp[f"{col}_b"]).abs()
            if (delta > tol).any():
                cand_changed = True
                break
        if cand_changed:
            moved.add(candidate)

    return DiffResult(
        matched_rows=int(len(merged)),
        changed_rows=int(row_changed.sum()),
        moved_candidates=sorted(moved),
    )


def load_summary_candidate(summary_path: Path, candidate: str) -> dict:
    """Return the summary row for *candidate* from a batch-runner summary JSON."""
    data = json.loads(summary_path.read_text())
    for row in data["summary"]:
        if row["candidate"] == candidate:
            return row
    raise KeyError(f"{candidate!r} not found in {summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-fast guardrail checks for execution_gate_tolerance acceptance criteria."
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts"),
        help="Root folder containing deep run detail artifacts.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("artifacts/main5_execution_gate_tolerance_guardrail_check.json"),
        help="Output JSON summary path.",
    )
    args = parser.parse_args()

    a = args.artifact_root
    failures: list[str] = []

    # 1) Broad panel must remain invariant at 1e-10.
    zi = strict_diff(
        a / "main5_deep_deep_zi_edge_confirm_detail.csv",
        a / "main5_deep_deep_zi_edge_confirm_tol1e10_detail.csv",
    )
    if zi.changed_rows != 0:
        failures.append(
            "deep_zi_edge_confirm @1e-10 must be fully invariant (changed_rows == 0)."
        )

    # 2) Stress panel frontier checks.
    stress_base = a / "main5_deep_deep_tolstress_expanded_base_detail.csv"
    stress_1e10 = strict_diff(stress_base, a / "main5_deep_deep_tolstress_expanded_tol1e10_detail.csv")
    stress_12e11 = strict_diff(stress_base, a / "main5_deep_deep_tolstress_expanded_tol12e11_detail.csv")
    stress_13e11 = strict_diff(stress_base, a / "main5_deep_deep_tolstress_expanded_tol13e11_detail.csv")
    stress_14e11 = strict_diff(stress_base, a / "main5_deep_deep_tolstress_expanded_tol14e11_detail.csv")
    stress_15e11 = strict_diff(stress_base, a / "main5_deep_deep_tolstress_expanded_tol15e11_detail.csv")

    expected_first = ["l_s092625468_objw1"]
    expected_second = ["l_s092625468_objw1", "l_s092625470_objw1"]

    if stress_1e10.moved_candidates != expected_first:
        failures.append(
            "stress panel @1e-10 moved-candidate set changed; expected only l_s092625468_objw1."
        )
    if stress_12e11.moved_candidates != expected_first:
        failures.append(
            "stress panel @1.2e-10 must match @1e-10 moved-candidate set."
        )
    if stress_13e11.moved_candidates != expected_first:
        failures.append(
            "stress panel @1.3e-10 must match @1e-10 moved-candidate set."
        )
    if stress_14e11.moved_candidates != expected_first:
        failures.append(
            "stress panel @1.4e-10 must match @1e-10 moved-candidate set."
        )
    if stress_15e11.moved_candidates != expected_second:
        failures.append(
            "stress panel @1.5e-10 must include exactly l_s092625468_objw1 and l_s092625470_objw1."
        )

    # 3) Bisection scan frontier checks (2-candidate focused scans).
    bisect_base = a / "main5_deep_deep_tolscan2_expanded_base_detail.csv"
    bisect_145 = strict_diff(bisect_base, a / "main5_deep_deep_tolscan2_expanded_tol145e12_detail.csv")
    bisect_146 = strict_diff(bisect_base, a / "main5_deep_deep_tolscan2_expanded_tol146e12_detail.csv")
    bisect_147 = strict_diff(bisect_base, a / "main5_deep_deep_tolscan2_expanded_tol147e12_detail.csv")
    bisect_148 = strict_diff(bisect_base, a / "main5_deep_deep_tolscan2_expanded_tol148e12_detail.csv")

    if bisect_145.moved_candidates != expected_first:
        failures.append(
            "bisect scan @1.45e-10 must move only l_s092625468_objw1 (onset of _470 > 1.46e-10)."
        )
    if bisect_146.moved_candidates != expected_first:
        failures.append(
            "bisect scan @1.46e-10 must move only l_s092625468_objw1 (onset of _470 > 1.46e-10)."
        )
    if bisect_147.moved_candidates != expected_second:
        failures.append(
            "bisect scan @1.47e-10 must move both l_s092625468_objw1 and l_s092625470_objw1."
        )
    if bisect_148.moved_candidates != expected_second:
        failures.append(
            "bisect scan @1.48e-10 must move both l_s092625468_objw1 and l_s092625470_objw1."
        )

    # 11) l_s092625468_objw1 must be in BAD regime at tol=4e-11 (lower cliff below safe band).
    #     Regime markers: mean_executed_steps > 1.2 and mean_test_relative < 0.015.
    lc_bad = load_summary_candidate(
        a / "main5_deep_deep_z468_lowflip2_tol4e-11_summary.json",
        "l_s092625468_objw1",
    )
    lc_bad_steps_ok = lc_bad["mean_executed_steps"] > 1.2
    lc_bad_test_ok = lc_bad["mean_test_relative"] < 0.015
    if not (lc_bad_steps_ok and lc_bad_test_ok):
        failures.append(
            "l_s092625468_objw1 @4e-11 must be in BAD regime "
            "(mean_executed_steps>1.2, mean_test_relative<0.015); "
            f"got steps={lc_bad['mean_executed_steps']}, test={lc_bad['mean_test_relative']:.8g}."
        )

    # 12) l_s092625468_objw1 must be in GOOD regime at tol=5e-11 (lower bound of safe band).
    #     Regime markers: mean_executed_steps < 1.1 and mean_test_relative > 0.015.
    lc_good = load_summary_candidate(
        a / "main5_deep_deep_z468_lowflip_tol5e-11_summary.json",
        "l_s092625468_objw1",
    )
    lc_good_steps_ok = lc_good["mean_executed_steps"] < 1.1
    lc_good_test_ok = lc_good["mean_test_relative"] > 0.015
    if not (lc_good_steps_ok and lc_good_test_ok):
        failures.append(
            "l_s092625468_objw1 @5e-11 must be in GOOD regime "
            "(mean_executed_steps<1.1, mean_test_relative>0.015); "
            f"got steps={lc_good['mean_executed_steps']}, test={lc_good['mean_test_relative']:.8g}."
        )

    summary = {
        "status": "pass" if not failures else "fail",
        "checks": {
            "deep_zi_1e10": {
                "matched_rows": zi.matched_rows,
                "changed_rows": zi.changed_rows,
                "moved_candidates": zi.moved_candidates,
            },
            "stress_1e10": {
                "matched_rows": stress_1e10.matched_rows,
                "changed_rows": stress_1e10.changed_rows,
                "moved_candidates": stress_1e10.moved_candidates,
            },
            "stress_1p2e10": {
                "matched_rows": stress_12e11.matched_rows,
                "changed_rows": stress_12e11.changed_rows,
                "moved_candidates": stress_12e11.moved_candidates,
            },
            "stress_1p5e10": {
                "matched_rows": stress_15e11.matched_rows,
                "changed_rows": stress_15e11.changed_rows,
                "moved_candidates": stress_15e11.moved_candidates,
            },
            "stress_1p3e10": {
                "matched_rows": stress_13e11.matched_rows,
                "changed_rows": stress_13e11.changed_rows,
                "moved_candidates": stress_13e11.moved_candidates,
            },
            "stress_1p4e10": {
                "matched_rows": stress_14e11.matched_rows,
                "changed_rows": stress_14e11.changed_rows,
                "moved_candidates": stress_14e11.moved_candidates,
            },
            "bisect_tol145e12": {
                "matched_rows": bisect_145.matched_rows,
                "changed_rows": bisect_145.changed_rows,
                "moved_candidates": bisect_145.moved_candidates,
            },
            "bisect_tol146e12": {
                "matched_rows": bisect_146.matched_rows,
                "changed_rows": bisect_146.changed_rows,
                "moved_candidates": bisect_146.moved_candidates,
            },
            "bisect_tol147e12": {
                "matched_rows": bisect_147.matched_rows,
                "changed_rows": bisect_147.changed_rows,
                "moved_candidates": bisect_147.moved_candidates,
            },
            "bisect_tol148e12": {
                "matched_rows": bisect_148.matched_rows,
                "changed_rows": bisect_148.changed_rows,
                "moved_candidates": bisect_148.moved_candidates,
            },
            "lower_cliff_bad_4e11": {
                "candidate": "l_s092625468_objw1",
                "tolerance": "4e-11",
                "mean_executed_steps": lc_bad["mean_executed_steps"],
                "mean_test_relative": lc_bad["mean_test_relative"],
                "regime": "bad" if (lc_bad_steps_ok and lc_bad_test_ok) else "unexpected",
            },
            "lower_cliff_good_5e11": {
                "candidate": "l_s092625468_objw1",
                "tolerance": "5e-11",
                "mean_executed_steps": lc_good["mean_executed_steps"],
                "mean_test_relative": lc_good["mean_test_relative"],
                "regime": "good" if (lc_good_steps_ok and lc_good_test_ok) else "unexpected",
            },
        },
        "failures": failures,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
