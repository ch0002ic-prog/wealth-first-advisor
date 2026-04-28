# Main5 Reverse-Blind Validation (V2, Expanded Coverage)

| candidate | predicted_good_from | observed_good_from_nonnegative | abs_error_to_nonnegative_good_from | tested_nonnegative_range | status |
|---|---:|---:|---:|---|---|
| k_s09262542_objw1 | -2.3675e-09 | 0 | 2.3675e-09 | [0.0, 1e-09] | always_good_in_tested_nonnegative_range |
| k_s09262544_objw1 | -1.362e-09 | 0 | 1.362e-09 | [0.0, 2e-10] | always_good_in_tested_range |
| k_s09262546_objw1 | -3.565e-10 | 0 | 3.565e-10 | [0.0, 2e-10] | always_good_in_tested_range |
| l_s092625462_objw1 | -2.47e-10 | 0 | 2.47e-10 | [0.0, 1e-09] | always_good_in_tested_nonnegative_range |
| l_s092625464_objw1 | -1.48e-10 | 0 | 1.48e-10 | [0.0, 2e-10] | always_good_in_tested_range |
| l_s092625466_objw1 | -4.9e-11 | 0 | 4.9e-11 | [0.0, 2e-10] | always_good_in_tested_range |

Note: these rows are zero-edge cases (always-good in tested nonnegative range), so midpoint-relative error is not defined.
Coverage objective: ensure reverse-blind family contributes validated rows to family-coverage checks.
