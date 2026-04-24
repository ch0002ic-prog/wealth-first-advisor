# Root Cause Analysis: Main4 Wealth-First Phase-2.1 (2026-04-24)

## Scope
This memo summarizes deeper investigation findings after:
- full phase-2.1 baseline run
- full phase-2.1 tight-gate run
- new frontier diagnostics in the phase-2 runner

## Executive Findings
1. The policy winner is threshold-sensitive.
- Baseline gates choose `promoted_tanh`.
- Tight tail gates choose `turnover_guard`.

2. Seed-based uncertainty is currently non-informative.
- All seed-dependent aggregates are identical in this experiment grid.
- Reported seed std values are exactly zero.

3. Action activity is sparse across all profiles.
- Gate suppression is very high (~96-98% of steps).
- Executed rebalance rate proxy is low (~1.9%-4.2% of test days).

4. Tight-gate failure for `promoted_tanh` is tail-driven, not return-driven.
- Under tight gates, `promoted_tanh` fails on worst daily relative return and worst relative drawdown.
- It still has strong return and robust-min margins.

## Evidence

### A) Full baseline vs full tight-gate outcomes
- Baseline full summary: `artifacts/main4_phase21_wealthfirst_full_summary.json`
- Tight-gate full summary: `artifacts/main4_phase21_wealthfirst_full_tightgates_summary.json`

Observed profile ranking behavior:
- Baseline: `promoted_tanh` best and eligible.
- Tight gates (`min_worst_daily_relative_return=-0.0035`, `max_worst_relative_drawdown=0.0115`):
  - `turnover_guard` best eligible
  - `promoted_tanh` ineligible

### B) Seed invariance is exact in this setup
For every `(profile, transaction_cost_bps)`, each metric has one unique value across seeds:
- `mean_test_relative`
- `robust_min_test_relative`
- `mean_turnover`
- `worst_daily_relative_return`
- `worst_max_relative_drawdown`

### C) Tight-gate slacks for promoted_tanh
Computed against full tight-gate thresholds:
- tail slack: `+0.004783`
- robust slack: `+0.002264`
- turnover slack: `+0.000371`
- worst-day slack: `-0.000376` (fails)
- drawdown slack: `-0.000658` (fails)

### D) Action sparsity / gate suppression
Aggregated from per-run fold summaries in `artifacts/main4_phase2_runs`:
- `promoted_tanh`: mean suppression ~0.958, execution proxy ~0.0416
- `turnover_guard`: mean suppression ~0.981, execution proxy ~0.0188
- `robust_guard`: mean suppression ~0.980, execution proxy ~0.0200

Interpretation: these policies are near-static most days and trade on a small subset of steps.

## Root Causes (Likely)

### 1) Seed has no effect in training/evaluation path
Likely root cause:
- `seed` is passed through interfaces but not used in model fitting logic.
- Split generation is deterministic anchored walk-forward.
- Ridge solve + deterministic scale grid search gives deterministic outputs.

Code pointers:
- `src/wealth_first/medium_capacity.py` (`train_medium_capacity_model`): seed arg present, no RNG usage.
- `src/wealth_first/data_splits.py` (`generate_walk_forward_splits`): deterministic split construction.

### 2) Policy is constrained toward low activity
Likely root cause:
- deadband/no-trade gate suppresses most candidate steps
- bounded weights and smoothing further damp transitions

Result:
- Lower turnover and friction resilience, but limited adaptability and reduced uncertainty expression.

### 3) Current uncertainty bands are scenario-level, not path-level uncertainty
Current bootstrap design:
- block bootstrap over friction scenarios in phase summary
- does not resample temporal return paths or estimation noise

Implication:
- good for scenario aggregation stability
- insufficient for confidence about path-level tail behavior under stochastic variation

## Strengths
1. Clear and reproducible winner under baseline wealth-first gates.
2. New tail metrics and frontier report improve transparency of gate sensitivity.
3. Tight-gate stress test successfully surfaced an interpretable regime where a lower-turnover profile dominates.

## Weaknesses
1. Zero seed variability can create false confidence if interpreted as robust stochastic stability.
2. High suppression implies limited active decision frequency.
3. Tail gates currently rely on coarse fold-level summaries rather than richer path diagnostics (for example distribution of drawdown durations, CVaR-like path tails).

## Missing Items
1. True stochastic uncertainty mechanism (path bootstrap or injected estimation noise).
2. Time-series block bootstrap in training/evaluation loop, not only scenario aggregation bootstrap.
3. Additional tail diagnostics:
- downside semivariance
- rolling drawdown duration metrics
- tail cluster severity (multiple consecutive bad days)
4. Regime-segmented attribution (high vol vs low vol windows).

## Next Steps (Prioritized)
1. Implement path-level bootstrap mode in `main4` evaluation.
- Goal: create non-zero uncertainty intervals with temporal dependence preserved.

2. Add path-tail diagnostics into per-fold outputs.
- Include CVaR-like daily relative tail and drawdown duration stats.

3. Introduce an activity floor gate in phase-2 runner.
- Example: minimum executed-step-rate to avoid over-selecting near-static policies.

4. Run a 2D frontier sweep around tight tail gates.
- Sweep `min_worst_daily_relative_return` and `max_worst_relative_drawdown` to map profile switching boundary.

5. Re-rank with multi-objective reporting.
- Keep current wealth-first score but also emit Pareto front across:
  - mean return
  - robust-min
  - turnover
  - worst-day
  - drawdown

## Current Recommendation
- Keep `promoted_tanh` as baseline promoted profile for current gate set.
- For stricter tail-risk mandates, `turnover_guard` is a credible fallback.
- Do not interpret zero seed variability as model robustness until path-level stochastic evaluation is added.
