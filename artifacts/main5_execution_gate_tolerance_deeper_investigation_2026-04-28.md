# Main5 Tolerance Deeper Investigation (2026-04-28)

## Scope
- Addressed weaknesses: reverse-blind family evidence coverage, tolerance side-effect localization.
- Addressed missing item: direct wealth-impact comparison for tolerance policy (0 vs 1e-10) on cliff-near panel.

## Tolerance Wealth Impact (0 -> 1e-10)
| candidate | delta_mean_test_relative | delta_mean_executed_steps | breaches_tol0 | breaches_tol1e10 |
|---|---:|---:|---:|---:|
| l_s092625468_objw1 | 0.006671071046 | -0.500000 | 0 | 0 |
| l_s092625470_objw1 | 0.000000000000 | 0.000000 | 0 | 0 |
| l_s092625472_objw1 | 0.000000000000 | 0.000000 | 0 | 0 |

- Key localized uplift: l_s092625468_objw1 mean_test_relative 0.010985214731 -> 0.017656285777 (delta 0.006671071046).
- Neighbor invariance in same cliffcheck panel: l470/l472 unchanged to floating precision.

## Expanded Panel Collateral Check
- deep_zi edge-confirm panel remained invariant across tolerance settings for all 4 tracked candidates.
- No breach-count increase detected in this invariance check.

## Coverage + Gate Status
- validation_rows[blind_forward] = 2
- validation_rows[piecewise] = 2
- validation_rows[reverse_blind] = 6
- validation_rows[surrogate] = 2
- family_coverage_check.failed = False
- robust_threshold_check.failed = False
- robust_mae = 0.00149983925240248
- robust_rmse = 0.0016420727501936607
- robust_max_abs_error = 0.0026398958914296364

## Decision
- Keep execution_gate_tolerance=1e-10 for main5 tolerance workflow.
- Treat tolerance setting as locked for subsequent mechanism optimization passes.
- Next deeper work should target wealth-lift mechanisms under this locked tolerance, not further tolerance widening.
