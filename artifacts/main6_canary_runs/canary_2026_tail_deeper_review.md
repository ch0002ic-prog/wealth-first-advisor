# Main6 Deeper Tail Investigation Review (2026-04-28)

## Scope
- Promote from relaxed tail improvements to strict two-window evidence.
- Compare promoted baseline candidate (`objw05_ntb005_floor_t15_p3`) against higher tail-bootstrap fidelity variants.
- Probe whether floor-penalty retuning on top of `tailr240` can improve strict dominance.

## Panels Executed

### 1) Top-3 high-seed stability panel (10 seeds)
- Labels:
  - `canary_2026_tail_top3_seed10_w2020_2026`
  - `canary_2026_tail_top3_seed10_w2015_2019`
- Candidates:
  - `objw05_ntb005_floor_t15_p3` (baseline)
  - `objw05_ntb005_floor_t15_p3_tailr240`
  - `objw05_ntb005_floor_t15_p3_tailr320`
- Coverage: `3 candidates x 4 scenarios x 10 seeds x 3 costs = 360` cases/window.

### 2) Tailr240 floor-penalty micro-sweep (2 seeds)
- Labels:
  - `canary_2026_tailr240_floorpen_w2020_2026`
  - `canary_2026_tailr240_floorpen_w2015_2019`
- Candidates:
  - baseline
  - `tailr240` with floor-penalty p in `{2.0, 2.5, 3.0, 3.5, 4.0}`
- Coverage: `6 candidates x 4 scenarios x 2 seeds x 3 costs = 144` cases/window.

## Summary Results

### Top-3 high-seed panel
- Breaches: `0/360` in both windows for all three candidates.
- Cross-window strict ranking (`canary_2026_tail_top3_seed10_cross_window_rank.csv`):
  - `tailr320`: `strict_pass=True`, best combined score.
  - `tailr240`: `strict_pass=True`, second best.
  - baseline: `strict_pass=False`.
- Main reason `tailr320` leads:
  - Slight positive mean deltas in both windows.
  - Material improvement in 2015 robust tail metric (min robust p05 less negative).

### Floor-penalty micro-sweep
- Breaches: `0/144` in both windows for all variants.
- Cross-window ranking (`canary_2026_tailr240_floorpen_cross_window_rank.csv`):
  - Best variant remains `tailr240` at floor penalty `3.0` (relaxed pass only in this sweep).
  - `p<=2.5`: hurts 2020 mean materially.
  - `p>=3.5`: hurts 2015 mean and robust tail.
- Conclusion: floor-penalty retuning around `tailr240` does not beat the top-3 seed10 winner.

## Promotion Decision
- Promote tail-bootstrap reps from `240` to `320` while keeping the rest of promoted settings unchanged.
- Implemented in runtime defaults:
  - `src/wealth_first/main6.py`
    - `main(... validation_tail_bootstrap_reps=320 ...)`
    - CLI default `--validation-tail-bootstrap-reps=320`

## Strengths
- First strict two-window pass under broader seed stress (10-seed panel).
- Zero breach profile maintained during deeper sweeps.
- Tail-risk robustness improved without changing dead-fold health profile.

## Weaknesses
- Absolute mean delta improvements are small in magnitude; mostly consistency gains.
- Strict success currently demonstrated most strongly on top-3 panel, while micro-sweeps still show narrow trade-offs.

## Missing Items / Residual Risk
- No explicit third out-of-slice stress window yet (for example, 2008-2012 or 2010-2014).
- Seed breadth widened for top-3 only; broader candidate families are still at 2 seeds.
- Sensitivity to block size / quantile under seed10 not fully mapped for the new promoted reps=320 default.

## Next Steps (Wealth-First Optimization)
1. Add a third historical stress window and rerun top-3 seed10 panel to confirm transfer.
2. Run seed10 for a targeted quantile/block-size grid around `tailr320`:
   - quantile in `{0.04, 0.05, 0.06}`
   - block size in `{12, 16, 20}`
3. If stable, lock a champion/challenger policy:
   - champion: `tailr320`
   - challenger: `tailr240`
4. Add rolling-period drift diagnostics for robust-min p05 and active-fraction to detect future regime drift before promotion updates.
