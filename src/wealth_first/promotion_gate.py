from __future__ import annotations

from typing import Any


def build_promotion_gate(
    *,
    finalists: list[str],
    best_row: dict[str, Any],
    ref_row: dict[str, Any],
    improvement_vs_ref: dict[str, float],
    materiality_eps: float = 1e-5,
) -> dict[str, Any]:
    """Return a promotion decision payload for gate checks."""

    reference_variant = str(ref_row.get("variant", "v25z_ref"))
    best_variant = str(best_row.get("variant", reference_variant))

    checks = {
        "baseline_in_finalists": reference_variant in finalists,
        "full_ref_present": bool(ref_row),
        "best_is_feasible": int(best_row.get("breach_count", 1)) == 0 and bool(best_row.get("all_non_path", False)),
        "materiality_min_case_slack": float(improvement_vs_ref.get("delta_min_case_slack", 0.0)) >= materiality_eps,
        "materiality_mean_case_slack": float(improvement_vs_ref.get("delta_mean_case_slack", 0.0)) >= materiality_eps,
        "materiality_mean_test_relative": float(improvement_vs_ref.get("delta_mean_test_relative", 0.0)) >= materiality_eps,
    }

    promote = (
        checks["baseline_in_finalists"]
        and checks["full_ref_present"]
        and checks["best_is_feasible"]
        and best_variant != reference_variant
        and checks["materiality_min_case_slack"]
        and checks["materiality_mean_case_slack"]
        and checks["materiality_mean_test_relative"]
    )

    return {
        "status": "PROMOTE_NEW_VARIANT" if promote else "KEEP_REFERENCE",
        "recommended_variant": best_variant if promote else reference_variant,
        "best_variant": best_variant,
        "reference_variant": reference_variant,
        "materiality_eps": materiality_eps,
        "checks": checks,
        "improvement_vs_ref": improvement_vs_ref,
    }
