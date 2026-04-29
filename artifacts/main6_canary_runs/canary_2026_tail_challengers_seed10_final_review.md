# Tail Bootstrap Challengers — Unified 3-Window Final Review
**Date:** 2025  
**Harness:** `scripts/investigate_main6_canary.py`  
**Seeds:** 7, 17, 27, 37, 47, 57, 67, 77, 87, 97 (10 seeds × 4 scenarios × 3 costs = 120 cases per candidate per window)  
**Candidates:** 6 (baseline + tailr240 + tailr320 + q06_b12 + q05_b20 + q05_b16)

---

## Windows Evaluated

| Window | Rows | Breaches | Status |
|--------|------|----------|--------|
| 2020–2026 | 1581 | 0/720 | ✅ valid |
| 2015–2019 | 1258 | 0/720 | ✅ valid |
| 2010–2014 | 1258 | 0/720 | ✅ valid |
| 2008–2012 | 1259 | 720/720 | ❌ **universal gate failure** — regime break (see below) |

### 2008–2012 Crisis Window: Uninformative for Ranking

All 720 cases for every candidate failed with `gate_fail:validation_threshold`. All candidates produced **identical** mean_test_relative=−0.000007 and min_robust_min_p05=−0.000393. This is a regime break: the crisis-era data violates the validation threshold gate uniformly, making this window structurally incompatible with current gate settings for cross-candidate selection. Finding documented; window excluded from ranking.

---

## 3-Window Ranking Results

Strict-3 criterion: zero breaches in all 3 valid windows **AND** positive mean delta vs baseline in all 3 **AND** non-negative robust delta vs baseline in all 3.

| Candidate | strict3_pass | relaxed3_pass | Σ mean delta | Σ robust delta |
|-----------|:---:|:---:|---:|---:|
| **tailr320_q05_b20** | ✅ **TRUE** | ✅ TRUE | +0.000015 | +0.000647 |
| tailr320_q06_b12 | ❌ | ❌ | +0.000011 | +0.000636 |
| tailr320 | ❌ | ❌ | +0.000009 | +0.000647 |
| tailr320_q05_b16 | ❌ | ❌ | +0.000007 | +0.000647 |
| tailr240 | ❌ | ❌ | +0.000006 | +0.000589 |
| baseline | — | — | 0 | 0 |

---

## Per-Window Deltas vs Baseline

### Mean test-relative delta

| Candidate | w2020–2026 | w2015–2019 | w2010–2014 | ALL ≥ 0? |
|-----------|---:|---:|---:|:---:|
| **q05_b20** | +0.000003 | +0.000010 | **+0.000001** | ✅ |
| tailr320 | +0.000003 | +0.000010 | −0.000004 | ❌ |
| q05_b16 | +0.000003 | +0.000009 | −0.000005 | ❌ |
| q06_b12 | +0.000003 | +0.000011 | −0.000002 | ❌ |
| tailr240 | +0.000003 | +0.000008 | −0.000005 | ❌ |

### Robust p05 delta vs baseline

| Candidate | w2020–2026 | w2015–2019 | w2010–2014 | ALL ≥ 0? |
|-----------|---:|---:|---:|:---:|
| **q05_b20** | 0 | +0.000636 | +0.000011 | ✅ |
| tailr320 | 0 | +0.000636 | +0.000011 | ✅ |
| q05_b16 | 0 | +0.000636 | +0.000011 | ✅ |
| q06_b12 | 0 | +0.000636 | 0 | ✅ |
| tailr240 | 0 | +0.000589 | 0 | ✅ |

---

## Key Findings

### 1. `q05_b20` is the sole strict-3 pass candidate
The only knob combination that simultaneously achieves positive mean delta vs baseline in **all three** historical windows is `validation_tail_bootstrap_reps=320, validation_tail_quantile=0.05, validation_tail_bootstrap_block_size=20`. Its w2010–2014 mean delta over baseline is tiny (+0.000001, i.e. 0.000090 vs 0.000089) but strictly positive, and its robust delta in that era is also positive (+0.000011). No other candidate achieves strict-3.

### 2. The critical differentiator is w2010–2014 mean
All tailr320 variants improve on baseline robust delta in w2015–2019 (the largest gain: +0.000636) and are neutral-to-positive in w2020–2026. The only window where candidates diverge on mean is w2010–2014: `q05_b20` ekes out +0.000001 while `tailr320`, `q05_b16`, and `tailr240` are slightly negative (−0.000004, −0.000005, −0.000005).

### 3. Robust tail improvements are consistent and material
The 2015–2019 robust p05 gain of +0.000636 is large relative to the baseline value of −0.002829 → −0.002193 for all tailr320 variants. This represents a ≈22% improvement in the worst-tail outcome for that era.

### 4. 2010–2014 mean margin is below statistical noise threshold
The q05_b20 w2010–2014 mean advantage (+0.000001 over baseline) is a single unit in the 6th decimal place. Over 120 cases it is directionally consistent, but the caution flag is warranted: this is a thin reed on which to hang promotion.

### 5. tailr320 vs baseline is not the full story
The baseline itself uses the promoted 320-rep default after earlier sessions. The explicit `tailr320` candidate differs from baseline on the tail quantile/block-size parameters, which is why they produce non-identical results despite both using 320 reps.

---

## Governance Decision

**Recommendation: Conditional promotion of `q05_b20`**

The case for promotion:
- Sole strict-3 pass candidate in a 10-seed × 4-scenario × 3-cost panel
- No breaches across 1,800 valid evaluation cases (0/360 × 3 windows × 5 candidates)
- Material robust-tail improvement in 2015–2019 (+22% on worst-tail)
- Zero degradation in w2020–2026 mean relative to other challengers (tied at +0.000003 over baseline)

The case for caution:
- w2010–2014 mean dominance margin (+0.000001) is effectively noise
- q05 quantile and b20 block size were selected from a grid search — overfitting risk if the grid was exhaustive
- 2008–2012 crisis regime is not gateable with current settings; no downside robustness evidence in a full bear market

**Suggested next step before promotion:** Run a full-seed retrain (seeds 1–30 or similar) comparing `tailr320` vs `q05_b20` in w2010–2014 only to confirm whether the mean-delta sign is structurally positive or noise-driven. If q05_b20 holds positive mean delta in a wider seed panel, promote it as the new champion with the promoted defaults:
- `validation_tail_bootstrap_reps = 320`
- `validation_tail_quantile = 0.05` (if different from current default)
- `validation_tail_bootstrap_block_size = 20` (if different from scenario default)

---

## Artifacts

| File | Description |
|------|-------------|
| `canary_2026_tail_challengers_seed10_w2020_2026/` | Per-case detail + summary, 2020–2026 |
| `canary_2026_tail_challengers_seed10_w2015_2019/` | Per-case detail + summary, 2015–2019 |
| `canary_2026_tail_challengers_seed10_w2010_2014/` | Per-case detail + summary, 2010–2014 |
| `canary_2026_tail_challengers_seed10_w2008_2012/` | Per-case detail + summary, 2008–2012 (all breached) |
| `canary_2026_tail_challengers_seed10_3window_rank.csv` | Wide-format 3-window ranking table |
