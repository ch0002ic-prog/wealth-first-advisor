# Main6 Gap-Closure Deep Investigation (2026-04-29)

## Objective
Address the previously identified missing items:
1. Add a third out-of-slice stress window.
2. Run seed10 quantile/block-size sensitivity around promoted `tailr320`.

## New Panels

### A) Third-window top-3 stress (seed10)
- Label: `canary_2026_tail_top3_seed10_w2010_2014`
- Window: 2010-01-01 to 2014-12-31
- Candidates: baseline, `tailr240`, `tailr320`
- Coverage: `3 x 4 scenarios x 10 seeds x 3 costs = 360` cases.
- Breaches: `0/360`.

### B) Tailr320 sensitivity grid (seed10)
- Labels:
  - `canary_2026_tailr320_sens_seed10_w2020_2026`
  - `canary_2026_tailr320_sens_seed10_w2015_2019`
- Grid: quantile `{0.04,0.05,0.06}` x tail block-size `{12,16,20}` plus baseline and plain `tailr320`.
- Coverage per window: `11 x 4 x 10 x 3 = 1320` cases.
- Breaches: `0/1320` in both windows.

## New Rank Artifacts
- Two-window sensitivity rank:
  - `canary_2026_tailr320_sens_seed10_cross_window_rank.csv`
- Three-window top-3 rank:
  - `canary_2026_tail_top3_seed10_three_window_rank.csv`

## Key Findings

### 1) Sensitivity around tailr320 (2020-2026 + 2015-2019)
- Multiple grid points now pass strict two-window criteria (`strict_pass=True`).
- Top variant by combined delta score:
  - `objw05_ntb005_floor_t15_p3_tailr320_q06_b12`
- Strong cluster of near-equivalent winners:
  - `q05_b20`, plain `tailr320`, `q05_b16`, `q05_b12`.
- Failure edge cases:
  - `q04_b12` fails due to slight negative 2020 mean delta.
  - `q06_b20` fails due to slight negative 2015 mean delta and weaker robust delta.

### 2) Third-window top-3 confirmation (2010-2014)
- All candidates remain zero-breach.
- But three-window strict dominance (`strict3_pass`) is `False` for all top-3 candidates:
  - `tailr320` and `tailr240` have slightly negative 2010-2014 mean deltas vs baseline.
- Therefore, 3-window strict criterion is not yet met despite strong 2-window strict evidence.

## Interpretation vs Prior Weaknesses/Missing Items
- Missing item #1 (third window): **Closed** (executed and analyzed).
- Missing item #2 (seed10 sensitivity map): **Closed** (full q x block grid executed).
- Residual weakness remains:
  - Improvement is still modest and can reverse slightly in an additional era (2010-2014), depending on strictness definition.

## Updated Decision Guidance
- Keep current promoted `tailr320` default as operational champion under 2-window strict policy.
- Do not claim universal 3-window strict dominance yet.
- Use `q06_b12` as best current sensitivity challenger for follow-up testing.

## Recommended Next Steps
1. Expand third-window investigation from top-3 to include top sensitivity challengers (`q06_b12`, `q05_b20`, `q05_b16`).
2. Add fourth historical window (e.g. 2008-2012) for crisis-regime validation.
3. Define a practical governance rule:
   - hard requirement: zero breaches, no dead-fold regressions
   - soft requirement: non-negative mean delta in at least 2/3 windows with non-negative robust deltas in all windows.
4. If governance prefers strict all-window mean positivity, continue search along low-impact axes only:
   - tiny quantile shifts around 0.05-0.06
   - block-size around 12-16
   - keep floor/penalty and turnover fixed (already shown sensitive when moved).
