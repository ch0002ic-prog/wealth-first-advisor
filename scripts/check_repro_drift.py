"""Reproducibility drift gate.

Compares the current repro suite output against the locked baseline.
Exits 0 on pass, 1 on drift failures.

Usage
-----
  PYTHONPATH=src .venv/bin/python scripts/check_repro_drift.py [--suite-summary PATH] [--baseline PATH]

Defaults
--------
  --suite-summary  artifacts/main3_repro_suite_summary.csv
  --baseline       artifacts/main3_repro_baseline_locked.json

Typical CI invocation
---------------------
  PYTHONPATH=src .venv/bin/python scripts/run_main3_repro_suite.py && \\
  PYTHONPATH=src .venv/bin/python scripts/check_repro_drift.py

Fingerprint check
-----------------
  When --check-fingerprint is passed (default: on) the script also reads each
  per-case summary JSON and verifies that main3_file_sha256, numpy_version,
  pandas_version, and python_version_prefix all match the locked baseline
  fingerprint.  A fingerprint mismatch is reported as a WARNING but does NOT
  cause a non-zero exit unless --strict-fingerprint is also passed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_summary(path: Path) -> dict[str, dict]:
    """Return {case_name: metrics_dict} from the suite CSV."""
    import csv

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {row["case"]: {k: float(v) if k != "case" else v for k, v in row.items()} for row in reader}


def _load_baseline(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_case_summary_json(case: str, suite_summary_path: Path) -> Path:
    """Resolve per-case summary JSON path from case naming conventions."""
    artifacts_dir = suite_summary_path.parent
    direct = artifacts_dir / f"main3_repro_{case}_summary.json"
    if direct.exists():
        return direct

    # Case names are of form '<prefix>_<rest>', but artifact names drop some tags
    # for the A locked case: A_locked_f3_s10 -> main3_repro_A_f3_s10_summary.json.
    parts = case.split("_")
    if len(parts) >= 3:
        compact = artifacts_dir / f"main3_repro_{parts[0]}_{'_'.join(parts[-2:])}_summary.json"
        if compact.exists():
            return compact

    return direct


def _check_fingerprints(baseline_fp: dict, case_names: list[str], suite_summary_path: Path) -> list[str]:
    """Inspect per-case summary JSONs for fingerprint consistency.  Returns list of warnings."""
    warnings: list[str] = []
    for case in case_names:
        json_path = _resolve_case_summary_json(case, suite_summary_path)
        if not json_path.exists():
            warnings.append(f"[FP] case={case}: summary JSON not found at {json_path}")
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            fp = data.get("run_fingerprint", {})
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"[FP] case={case}: failed to load {json_path}: {exc}")
            continue

        checks = {
            "main3_file_sha256": (fp.get("main3_file_sha256"), baseline_fp.get("main3_file_sha256")),
            "numpy_version": (fp.get("numpy_version"), baseline_fp.get("numpy_version")),
            "pandas_version": (fp.get("pandas_version"), baseline_fp.get("pandas_version")),
        }
        # python_version_prefix: compare only the prefix (major.minor.patch) to avoid build-string diffs
        prefix_expect = baseline_fp.get("python_version_prefix")
        if prefix_expect:
            pv = fp.get("python_version", "")
            checks["python_version_prefix"] = (
                pv[:len(prefix_expect)] if pv else None,
                prefix_expect,
            )

        for field, (got, want) in checks.items():
            if got != want:
                warnings.append(f"[FP] case={case} field={field}: locked={want!r} current={got!r}")
    return warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check repro suite for metric drift vs locked baseline.")
    parser.add_argument(
        "--suite-summary",
        default=str(PROJECT_ROOT / "artifacts" / "main3_repro_suite_summary.csv"),
        help="Path to the current suite summary CSV (default: artifacts/main3_repro_suite_summary.csv)",
    )
    parser.add_argument(
        "--baseline",
        default=str(PROJECT_ROOT / "artifacts" / "main3_repro_baseline_locked.json"),
        help="Path to the locked baseline JSON (default: artifacts/main3_repro_baseline_locked.json)",
    )
    parser.add_argument(
        "--no-fingerprint",
        action="store_true",
        default=False,
        help="Skip fingerprint comparison.",
    )
    parser.add_argument(
        "--strict-fingerprint",
        action="store_true",
        default=False,
        help="Treat fingerprint mismatches as failures (exit 1).",
    )
    args = parser.parse_args(argv)

    suite_path = Path(args.suite_summary)
    baseline_path = Path(args.baseline)

    if not suite_path.exists():
        print(f"ERROR: suite summary not found: {suite_path}", file=sys.stderr)
        return 1
    if not baseline_path.exists():
        print(f"ERROR: baseline not found: {baseline_path}", file=sys.stderr)
        return 1

    baseline = _load_baseline(baseline_path)
    current = _load_summary(suite_path)
    tolerances: dict[str, float] = baseline["tolerances"]
    locked_cases: dict[str, dict] = baseline["cases"]
    metrics_to_check = [m for m in tolerances if not m.startswith("_")]

    failures: list[str] = []
    fp_warnings: list[str] = []

    print("=" * 72)
    print("REPRO DRIFT CHECK")
    print(f"  suite: {suite_path}")
    print(f"  baseline: {baseline_path}")
    print("=" * 72)

    # Per-case metric drift
    for case, locked in locked_cases.items():
        cur = current.get(case)
        if cur is None:
            failures.append(f"case={case}: missing from current suite summary")
            continue

        # Row count must match exactly
        locked_rows = int(locked["rows"])
        cur_rows = int(cur["rows"])
        if cur_rows != locked_rows:
            failures.append(f"case={case} rows: locked={locked_rows} current={cur_rows} (must match exactly)")

        print(f"\n  {case}")
        print(f"  {'metric':<30} {'locked':>12} {'current':>12} {'delta':>12} {'tol':>10} {'status':>8}")
        print("  " + "-" * 88)
        for metric in metrics_to_check:
            locked_val = locked.get(metric)
            cur_val = cur.get(metric)
            if locked_val is None or cur_val is None:
                print(f"  {metric:<30} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>10} {'SKIP':>8}")
                continue
            delta = abs(cur_val - locked_val)
            tol = tolerances[metric]
            status = "PASS" if delta <= tol else "FAIL"
            if status == "FAIL":
                failures.append(
                    f"case={case} metric={metric}: locked={locked_val:.6f} current={cur_val:.6f} "
                    f"delta={delta:.6f} > tol={tol:.6f}"
                )
            print(f"  {metric:<30} {locked_val:>12.6f} {cur_val:>12.6f} {cur_val - locked_val:>+12.6f} {tol:>10.6f} {status:>8}")

    # Fingerprint check
    if not args.no_fingerprint:
        fp_warnings = _check_fingerprints(baseline["fingerprint"], list(locked_cases.keys()), suite_path)

    print()
    if fp_warnings:
        print("FINGERPRINT WARNINGS:")
        for w in fp_warnings:
            print(f"  {w}")
    else:
        print("Fingerprint: OK (all cases match locked sha256/versions)")

    print()
    if failures:
        print("DRIFT FAILURES:")
        for f in failures:
            print(f"  FAIL: {f}")
        print(f"\nResult: FAIL ({len(failures)} drift failure(s))")
        return 1

    if fp_warnings and args.strict_fingerprint:
        print(f"\nResult: FAIL (strict-fingerprint mode, {len(fp_warnings)} fingerprint mismatch(es))")
        return 1

    print("Result: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
