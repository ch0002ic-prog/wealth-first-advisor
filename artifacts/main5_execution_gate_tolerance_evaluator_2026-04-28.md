# Main5 Tolerance Ladder Evaluator

- status: pass
- ladder_entries: 13
- global_monotonic_non_decreasing: True

## Branch Checks
- branch l: monotonic_non_decreasing=True, count=7
- branch k: monotonic_non_decreasing=True, count=6

## Interval Widths
- finite_count: 11
- min_width: 1.0000000000008237e-14
- max_width: 1.999999999999993e-11
- mean_width: 5.055454545454554e-12
- invalid_width_candidates: []

## Prediction Error Summary
- count: 6
- mae: 0.10470353627836704
- rmse: 0.252526358775423
- max_abs_error: 0.6185410554844598

## Prediction Error Summary By Family
- blind_forward: count=2, mae=0.0010651443804504633, rmse=0.0010737077503276955, max_abs_error=0.0012004801920768018
- piecewise: count=2, mae=0.0019345341243544967, rmse=0.002059115708439762, max_abs_error=0.0026398958914296364
- reverse_blind: count=0, mae=None, rmse=None, max_abs_error=None
- surrogate: count=2, mae=0.31111093033029613, rmse=0.43738231882791867, max_abs_error=0.6185410554844598

## Robust Aggregate Error Summary
- note: excludes surrogate and reverse_blind families
- count: 4
- mae: 0.00149983925240248
- rmse: 0.0016420727501936607
- max_abs_error: 0.0026398958914296364

