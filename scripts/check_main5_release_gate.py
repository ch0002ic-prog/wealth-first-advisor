#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Composite release gate for main5 locked-tolerance promotion artifacts."
    )
    parser.add_argument(
        "--guardrail-json",
        type=Path,
        default=Path("artifacts/main5_execution_gate_tolerance_guardrail_check.json"),
    )
    parser.add_argument(
        "--evaluator-json",
        type=Path,
        default=Path("artifacts/main5_execution_gate_tolerance_evaluator_deeper_2026-04-28.json"),
    )
    parser.add_argument(
        "--promotion-json",
        type=Path,
        default=Path("artifacts/main5_execution_gate_promotion_confirmation_seed6_2026-04-28.json"),
    )
    parser.add_argument(
        "--activity-json",
        type=Path,
        default=Path("artifacts/main5_execution_gate_activity_threshold_sensitivity_2026-04-28.json"),
    )
    parser.add_argument(
        "--expected-winner",
        type=str,
        default="l_s092625468_objw1",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("artifacts/main5_execution_gate_release_gate_2026-04-28.json"),
    )
    args = parser.parse_args()

    failures: list[str] = []
    checks: dict[str, object] = {}

    for p in [args.guardrail_json, args.evaluator_json, args.promotion_json, args.activity_json]:
        if not p.exists():
            failures.append(f"missing_required_artifact: {p}")

    if failures:
        result = {
            "status": "fail",
            "checks": checks,
            "failures": failures,
        }
        args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 1

    guardrail = load_json(args.guardrail_json)
    evaluator = load_json(args.evaluator_json)
    promotion = load_json(args.promotion_json)
    activity = load_json(args.activity_json)

    checks["guardrail_status"] = guardrail.get("status")
    if guardrail.get("status") != "pass":
        failures.append("guardrail_status_not_pass")

    checks["evaluator_status"] = evaluator.get("status")
    if evaluator.get("status") != "pass":
        failures.append("evaluator_status_not_pass")

    fam_cov = evaluator.get("family_coverage_check", {})
    checks["family_coverage_check_failed"] = bool(fam_cov.get("failed", True))
    if fam_cov.get("failed", True):
        failures.append("family_coverage_check_failed")

    robust = evaluator.get("robust_threshold_check", {})
    checks["robust_threshold_check_failed"] = bool(robust.get("failed", True))
    if robust.get("failed", True):
        failures.append("robust_threshold_check_failed")

    winner = promotion.get("winner")
    checks["promotion_winner"] = winner
    if winner != args.expected_winner:
        failures.append(
            f"promotion_winner_mismatch expected={args.expected_winner} got={winner}"
        )

    tie_delta = (
        promotion.get("comparisons", {})
        .get("l468_vs_j6254", {})
        .get("delta_mean_test_relative")
    )
    checks["l468_vs_j6254_delta_mean_test_relative"] = tie_delta
    if tie_delta is None or float(tie_delta) <= 0.0:
        failures.append("non_positive_l468_vs_j6254_delta")

    b1 = activity.get("summary", {}).get("boundary1_09988", [])
    top_b1 = b1[0]["candidate"] if b1 else None
    checks["boundary1_top_candidate"] = top_b1
    if top_b1 != args.expected_winner:
        failures.append(
            f"boundary1_top_candidate_mismatch expected={args.expected_winner} got={top_b1}"
        )

    # Two-threshold activity cap policy check.
    # Promotion uses boundary1-aligned threshold (max_supp=0.9988); strict threshold
    # (max_supp=0.997) is reserved for stress diagnostics only and MUST NOT gate promotion.
    # Here we document whether the strict threshold would invert the ranking — if it does,
    # this is expected and correct (strict inverts toward lower-return 1.5-step candidates).
    strict = activity.get("summary", {}).get("strict_0997", [])
    top_strict = strict[0]["candidate"] if strict else None
    strict_inverts = top_strict != args.expected_winner if top_strict is not None else None
    checks["activity_cap_policy"] = {
        "promotion_threshold": "boundary1_max_supp_09988",
        "stress_diagnostic_threshold": "strict_max_supp_0997",
        "strict_0997_top_candidate": top_strict,
        "strict_cap_inverts_ranking": strict_inverts,
        "policy": (
            "strict threshold (0.997) is for stress diagnostics only; "
            "boundary1 (0.9988) is the binding promotion criterion"
        ),
    }

    result = {
        "status": "pass" if not failures else "fail",
        "checks": checks,
        "failures": failures,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
