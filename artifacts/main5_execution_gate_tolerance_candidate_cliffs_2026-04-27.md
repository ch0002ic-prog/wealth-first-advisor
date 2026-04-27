# Main5 Execution-Gate Tolerance Candidate Cliffs (2026-04-27)

## Scope
Focused continuation after the lower-cliff confirmation for `l_s092625468_objw1`:
- Existing known flip: BAD at 4e-11, GOOD at 5e-11.
- New targets: `l_s092625470_objw1` and `k_s09262548_objw1`.
- Panel: expanded, seeds 5/17, costs 25/35, reps 30.

## Findings

### `l_s092625470_objw1`
- Coarse bracket: BAD through 1.469e-10, GOOD at 1.470e-10.
- Ultra-fine bracket: BAD through 1.46960e-10, GOOD at 1.46970e-10.
- Estimated transition: [1.46960e-10, 1.46970e-10].
- Regime metrics: BAD = (mean_test_relative 0.0109852147, steps 1.5), GOOD = (mean_test_relative 0.0176562858, steps 1.0).

### `k_s09262548_objw1`
- Coarse bracket: BAD through 6.4e-10, GOOD at 6.5e-10.
- Ultra-fine bracket: BAD through 6.48e-10, GOOD at 6.49e-10.
- Estimated transition: [6.48e-10, 6.49e-10].
- Regime metrics: BAD = (mean_test_relative 0.0109852147, steps 1.5), GOOD = (mean_test_relative 0.0176562858, steps 1.0).

## Interpretation
Tolerance cliffs are strongly candidate-specific, even inside the same micro-family near the smoothing edge. A single global tolerance setting can simultaneously be:
- safely post-cliff for one candidate,
- still pre-cliff for another.

This supports maintaining candidate-aware guardrails rather than relying only on a global tolerance policy.
