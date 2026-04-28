# Main5 Robust Threshold Sweep (2026-04-28)

Evaluator command used:

`python scripts/evaluate_main5_tolerance_ladder.py --robust-summary --robust-mae-max T --robust-rmse-max T --robust-max-abs-error-max T`

## Sweep Results

| Threshold T | Status |
|---|---|
| 0.0010 | fail |
| 0.0015 | fail |
| 0.0020 | fail |
| 0.0025 | fail |
| 0.00264 | pass |
| 0.00270 | pass |
| 0.0030 | pass |

## Frontier Interpretation

- Observed pass/fail frontier is near `T ~= 0.00264`.
- Binding robust metric is `max_abs_error = 0.0026398958914296364`.
- Current robust metrics:
  - `mae = 0.00149983925240248`
  - `rmse = 0.0016420727501936607`
  - `max_abs_error = 0.0026398958914296364`

## Practical Gate Recommendation

- Strict gate: `T = 0.00264` (minimal passing threshold).
- Safer operational gate: `T = 0.00270` (small margin against numeric jitter).
