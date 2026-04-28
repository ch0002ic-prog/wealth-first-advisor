# Main5 Surrogate Outlier Exclusion Policy (2026-04-28)

## Summary

The surrogate family validation produces a structural outlier for candidate `k_s09262552_objw1`
(k52) with a relative prediction error of **+61.854%**. This document records the diagnosis,
exclusion rationale, and constraints on using the surrogate family as a promotion signal.

## Outlier Diagnosis

| candidate | predicted | bad_through | good_from | midpoint | rel_error |
|---|---:|---:|---:|---:|---:|
| l_s092625474_objw1 | 3.4903e-10 | 3.477e-10 | 3.478e-10 | 3.4775e-10 | +0.368% |
| k_s09262552_objw1 | 4.297e-09 | 2.65e-09 | 2.66e-09 | 2.655e-09 | **+61.854%** |

**Root cause:** The surrogate model for k52 uses log-linear extrapolation across the (k48, k52)
interval. The local manifold for the k branch is not globally log-linear over that interval.
The predicted threshold overestimates the true transition midpoint by ~1.64 × 10⁻⁹, which is
large relative to the midpoint (~2.655 × 10⁻⁹). This is a structural limitation of coarse
log-linear extrapolation and is **not** indicative of a model inference failure.

l474 produces near-exact predictions (0.368% error), confirming the surrogate method works for
smooth regions; k52 is an exception because the k-branch manifold changes slope more sharply.

## Exclusion Policy

1. **Robust aggregate exclusion**: The surrogate family (both l474 and k52 rows) is excluded
   from the `prediction_error_summary_robust` aggregate used for threshold gate enforcement.
   Only piecewise and blind_forward families contribute to robust MAE/RMSE/max_abs.

2. **Surrogate is not primary promotion signal**: Surrogate predictions MUST NOT be used as
   primary evidence for or against candidate promotion. Piecewise and blind_forward families
   are the authoritative evidence sources.

3. **k52 cannot be safely extrapolated**: Any tolerance decision for k52 MUST be based on
   direct probe measurement, not surrogate extrapolation.

4. **Exclusion is implicit in evaluator code**: The `evaluate_main5_tolerance_ladder.py`
   script excludes the `surrogate` family from the robust aggregate via the family filter
   `{"surrogate", "reverse_blind"}`. This exclusion was in place before this artifact was
   created.

## Impact Assessment

- Robust metrics (piecewise + blind_forward only): MAE=0.00150, RMSE=0.00164, max_abs=0.00264
- All three metrics pass the 0.005 threshold enforced by the release gate.
- Including surrogate in the aggregate raises MAE to 0.1047 and max_abs to 0.6185, which would
  fail the 0.005 threshold — confirming that surrogate exclusion is load-bearing for gate pass.

## Resolution Status

- Structural diagnosis: complete
- Exclusion from robust aggregate: implemented in evaluator code
- Future work: if surrogate predictions are needed for k-branch candidates beyond k52, direct
  deep probe measurements should be preferred over extrapolation.
