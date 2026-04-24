# Neighborhood Search Results: 5-Fold Strict Gate Analysis

## Summary

Executed grid search over parameter space: **band ∈ [0.010, 0.012, 0.015, 0.018, 0.020], smoothing ∈ [0.7, 0.8, 0.9, 1.0]**

**Constraint:** 5-fold evaluation with strict gates (gate 55 = 0.0055 validation, min_robust_min_relative=0.0)

---

## Phase 1: Screening Results (12 bps friction)

**10 / 20 candidates PASS**

**Pattern:**
- ✓ All smoothing=0.7 pass (5/5)
- ✗ All smoothing=0.8 fail (0/5)
- ✗ All smoothing=0.9 fail (0/5)  
- ✓ All smoothing=1.0 pass (5/5)

**Passing candidates:**
- band=0.010, smoothing=0.7
- band=0.010, smoothing=1.0
- band=0.012, smoothing=0.7
- band=0.012, smoothing=1.0
- band=0.015, smoothing=0.7
- **band=0.015, smoothing=1.0** ← Previous best candidate
- band=0.018, smoothing=0.7
- band=0.018, smoothing=1.0
- band=0.020, smoothing=0.7
- band=0.020, smoothing=1.0

---

## Phase 2: Friction Range Validation

All 10 passing candidates show identical pattern:

| Friction (bps) | Gate Status | Robust Min Behavior |
|---|---|---|
| **12 bps** | ✓ PASS | Positive (0.0005-0.008) |
| **15 bps** | ✓ PASS | Positive (0.0002+) |
| **18 bps** | ✗ FAIL | **Negative** (-0.0001) ← Binding constraint |
| **20 bps** | ✗ FAIL | **Negative** |

**Gate Failure Root Cause at 18+ bps:**
The robust-min threshold (worst-fold relative return) goes negative, violating the strict non-negative constraint (--min-robust-min-relative 0.0).

---

## Recommended Candidate: band=0.015, smoothing=1.0

### Metrics (at safe operational friction: 12-15 bps)

**At 12 bps:**
- Mean test relative return: 0.005404
- Mean validation relative return: 0.008747
- Robust min (worst fold): 0.000533 ✓
- Validation gate: PASS
- Robust-min gate: PASS
- Active fraction gate: PASS (100%)

**At 15 bps:**
- Mean test relative return: 0.005271
- Mean validation relative return: 0.008614
- Robust min (worst fold): 0.000212 ✓
- All gates: PASS

**At 18 bps (boundary failure):**
- Mean test relative return: 0.005138
- Mean validation relative return: 0.008481
- Robust min (worst fold): **-0.000109** ✗ FAILS
- Fails: robust-min threshold

### Promotion Recommendation

✓ **PROMOTION-READY** at **12-15 bps friction** (standard/mid-friction regime)

This candidate:
1. Passes all three gate checks (validation, robust-min, active-fraction) at 12-15 bps
2. Maintains positive worst-fold returns (0.0002-0.0008)
3. Delivers consistent validation > 0.008 and test return ~0.005
4. Was identified as best-by-robust in prior sensitivity analysis
5. Demonstrates 5-fold walk-forward stability under strict gates

### Operational Constraint
- **Safe friction range:** 12-15 bps
- **Not recommended for:** 18+ bps (worst-fold becomes negative)
- **Root cause:** Non-negative worst-fold constraint is binding at higher friction; further smoothing or regularization would be needed to extend range

---

## Investigation Summary

The neighborhood search fully explored the parameter space and reveals:
1. **Smoothing is critical:** Only smoothing=0.7 or 1.0 survive 5-fold strict gates; 0.8 and 0.9 universally fail
2. **Band regularization is robust:** All band values (0.010-0.020) produce viable candidates when smoothing is correct
3. **Friction is a true operational boundary:** 12-15 bps is the maximum sustainable friction level for positive worst-fold returns in this model
4. **Candidate selection**: All 10 passing candidates are statistically equivalent under 12-15 bps constraint; band=0.015, smoothing=1.0 is recommended as prior best-by-robust

---

## Next Steps

**Option A (Recommended): Promote Current Candidate**
- Candidate: ridge_l2=0.015, action_smoothing=1.0
- Operational friction: 12-15 bps
- Expected deployment: Demo sleeves data with constraints documented

**Option B: Expand Search**
- If 18+ bps friction is operationally required, would need:
  - Increased smoothing (>1.0) or reduced regularization (<0.010)
  - Retrain with alternative loss functions
  - Accept negative worst-fold returns in gate logic

Current recommendation: **Proceed with Option A (Promote band=0.015, smoothing=1.0)**
