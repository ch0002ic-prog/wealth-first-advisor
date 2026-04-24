# Promoted Main4 Policy Configuration (2026-04-24)

## Executive Summary

A production-ready ridge-regularization and action-smoothing policy configuration has been validated and promoted for deployment. The configuration passes strict 5-fold walk-forward gate evaluations with non-negative worst-fold returns across the safe friction range.

---

## Policy Configuration

### Parameters

| Parameter | Value | Description |
|---|---|---|
| **ridge_l2** | 0.015 | L2 regularization strength for linear weights |
| **action_smoothing** | 1.0 | Exponential smoothing factor for action magnitude |
| **transaction_cost_bps** | 12 | One-way transaction cost (half-spread) in basis points |
| **slippage_bps** | 12 | Slippage allowance in basis points |
| **gate (validation)** | 55 | Validation threshold code (= 0.0055 in bps scale) |
| **gate_scale** | bps | Interpretation: gate code as basis points (÷10,000) |
| **min_robust_min_relative** | 0.0 | Strict: worst-fold returns must be ≥ 0.0 |
| **n_folds** | 5 | Walk-forward evaluation folds |
| **seed** | 5 | Deterministic random seed |

### Validated Performance

| Metric | Value | Status |
|---|---|---|
| Mean test relative return | 0.005404 | ✓ Target achieved |
| Mean validation return | 0.008747 | ✓ Above threshold (0.0055) |
| Robust min (worst fold) | 0.000533 | ✓ Positive |
| Beat hold (win rate) | 100.0% | ✓ Consistent |
| Active fraction | 100.0% | ✓ Full engagement |
| Mean turnover | 0.000475 | ✓ Low friction impact |
| Gate checks | PASS | ✓ All three gates |

---

## Deployment Methods

### Method 1: Environment Variable Configuration (Recommended for Docker)

Set environment variables before starting the bridge:

```bash
export WEALTH_FIRST_MAIN4_RIDGE_L2=0.015
export WEALTH_FIRST_MAIN4_ACTION_SMOOTHING=1.0
export WEALTH_FIRST_MAIN4_TRANSACTION_COST_BPS=12
export WEALTH_FIRST_MAIN4_SLIPPAGE_BPS=12
export WEALTH_FIRST_MAIN4_GATE=55
export WEALTH_FIRST_MAIN4_GATE_SCALE=bps
export WEALTH_FIRST_MAIN4_MIN_ROBUST_MIN_RELATIVE=0.0
export WEALTH_FIRST_MAIN4_N_FOLDS=5
export WEALTH_FIRST_MAIN4_SEED=5
export WEALTH_FIRST_MAIN4_PROMOTION_DATE=2026-04-24
export WEALTH_FIRST_MAIN4_PROMOTION_STATUS=stable
export WEALTH_FIRST_MAIN4_PROMOTION_LABEL="Ridge & Smoothing Policy (5-Fold Validated)"
export WEALTH_FIRST_MAIN4_PROMOTION_FRICTION_RANGE="12-15 bps"

python -m wealth_first.tradingview_bridge serve-env
```

### Method 2: Configuration File (Recommended for Dev/Test)

Source the config file and then start the bridge:

```bash
source config/promoted_main4.env
python -m wealth_first.tradingview_bridge serve-env
```

### Method 3: Start Script (Recommended for Production)

The updated `scripts/start_bridge_runtime.sh` automatically sources the promoted config:

```bash
export WEALTH_FIRST_PROMOTED_CONFIG=/app/config/promoted_main4.env
./scripts/start_bridge_runtime.sh
```

### Method 4: Docker Compose

Add environment variables to docker-compose.yml:

```yaml
services:
  wealth-first-bridge:
    environment:
      WEALTH_FIRST_MAIN4_RIDGE_L2: "0.015"
      WEALTH_FIRST_MAIN4_ACTION_SMOOTHING: "1.0"
      WEALTH_FIRST_MAIN4_TRANSACTION_COST_BPS: "12"
      WEALTH_FIRST_MAIN4_SLIPPAGE_BPS: "12"
      WEALTH_FIRST_MAIN4_GATE: "55"
      WEALTH_FIRST_MAIN4_GATE_SCALE: "bps"
      WEALTH_FIRST_MAIN4_MIN_ROBUST_MIN_RELATIVE: "0.0"
      WEALTH_FIRST_MAIN4_N_FOLDS: "5"
      WEALTH_FIRST_MAIN4_SEED: "5"
      WEALTH_FIRST_MAIN4_PROMOTION_DATE: "2026-04-24"
      WEALTH_FIRST_MAIN4_PROMOTION_STATUS: "stable"
      WEALTH_FIRST_MAIN4_PROMOTION_LABEL: "Ridge & Smoothing Policy (5-Fold Validated)"
      WEALTH_FIRST_MAIN4_PROMOTION_FRICTION_RANGE: "12-15 bps"
```

---

## Operational Constraints

### Safe Friction Range: 12-15 bps

The policy maintains non-negative worst-fold returns within this friction range.

**Performance Boundary:**
- **12 bps:** Robust min = 0.000533 ✓ PASS
- **15 bps:** Robust min = 0.000212 ✓ PASS
- **18 bps:** Robust min = -0.000109 ✗ FAIL (violates gate)

### Critical Operational Limits

⚠️ **Do not deploy at friction > 15 bps**

At higher friction levels, the worst-fold relative return becomes negative, violating the non-negative robust-min gate constraint. To extend the safe range would require:
- Increasing action_smoothing (>1.0)
- Decreasing ridge_l2 (<0.010)
- Retraining with alternative loss functions

---

## API Integration

The promoted configuration is exposed via the `/api/config/main4-promoted` endpoint:

```bash
curl http://localhost:8000/api/config/main4-promoted
```

Response:
```json
{
  "promoted": true,
  "policy": {
    "ridge_l2": 0.015,
    "action_smoothing": 1.0,
    "transaction_cost_bps": 12,
    "slippage_bps": 12,
    "gate": "55",
    "gate_scale": "bps",
    "min_robust_min_relative": 0.0,
    "n_folds": 5,
    "seed": 5
  },
  "promotion_metadata": {
    "date": "2026-04-24",
    "status": "stable",
    "label": "Ridge & Smoothing Policy (5-Fold Validated)",
    "friction_range": "12-15 bps"
  },
  "deployment_cli_command": "python -m wealth_first.main4 --ridge-l2 0.015 --action-smoothing 1.0 --transaction-cost-bps 12 --slippage-bps 12 --gate 55 --gate-scale bps --min-robust-min-relative 0.0 --n-folds 5 --seed 5"
}
```

---

## Testing & Validation

### 1. Verify Configuration is Loaded

Start the bridge and check the API:

```bash
curl http://localhost:8000/api/config/main4-promoted | jq .
```

Expected: Returns configuration with `"promoted": true`

### 2. Verify Gate Enforcement

Run the policy directly:

```bash
python -m wealth_first.main4 \
  --ridge-l2 0.015 \
  --action-smoothing 1.0 \
  --transaction-cost-bps 12 \
  --slippage-bps 12 \
  --gate 55 \
  --gate-scale bps \
  --min-robust-min-relative 0.0 \
  --n-folds 5 \
  --seed 5
```

Expected output includes:
```
Gate checks passed: True
Robust min: 0.000533
```

### 3. Test Friction Boundary (Optional)

Verify failure at 18 bps:

```bash
python -m wealth_first.main4 \
  --ridge-l2 0.015 \
  --action-smoothing 1.0 \
  --transaction-cost-bps 18 \
  --slippage-bps 18 \
  --gate 55 \
  --gate-scale bps \
  --min-robust-min-relative 0.0 \
  --n-folds 5 \
  --seed 5
```

Expected: Gate check fails (robust_min < 0.0)

---

## Monitoring & Alerts

### Key Metrics to Track

1. **Validation gate:** Should stay ≥ 0.0055
2. **Robust min:** Should stay > 0.0 (ideally > 0.0001)
3. **Active fraction:** Should stay ≥ 1.0%
4. **Turnover:** Monitor for degradation (baseline: 0.000475)

### Alert Thresholds

- ⚠️ **WARNING** if validation < 0.005
- 🔴 **CRITICAL** if robust_min < 0.0
- 🔴 **CRITICAL** if active_fraction < 0.01

---

## Troubleshooting

### Issue: API returns 404 for `/api/config/main4-promoted`

**Cause:** Environment variables not set or loaded

**Solution:**
```bash
# Verify variables are set
env | grep WEALTH_FIRST_MAIN4

# If empty, source the config:
source config/promoted_main4.env
python -m wealth_first.tradingview_bridge serve-env
```

### Issue: Gate checks fail in deployment

**Cause:** Friction > 15 bps or other gate parameters misconfigured

**Solution:**
1. Verify transaction_cost_bps + slippage_bps ≤ 15
2. Check that --gate and --gate-scale match config
3. Verify --min-robust-min-relative is 0.0

### Issue: Worst-fold returns become negative

**Cause:** Friction exceeded safe range (> 15 bps)

**Solution:**
- Reduce friction to 12-15 bps range
- Or implement alternative policy with higher smoothing

---

## Implementation Files

| File | Purpose |
|---|---|
| `config/promoted_main4.env` | Configuration constants (environment variables) |
| `scripts/start_bridge_runtime.sh` | Updated to source promoted config |
| `src/wealth_first/tradingview_bridge.py` | Extended with main4 config fields and API endpoint |
| `PROMOTION_ANALYSIS_2026-04-24.md` | Detailed analysis and neighborhood search results |

---

## Rollback Plan

If the promoted configuration does not perform as expected in production:

1. **Immediate:** Unset environment variables or source the previous config
2. **Verify:** Run gate evaluation with previous parameters
3. **Deploy:** Restart bridge with alternative configuration

Example rollback:
```bash
# Unset promoted config variables
unset WEALTH_FIRST_MAIN4_RIDGE_L2
unset WEALTH_FIRST_MAIN4_ACTION_SMOOTHING
# ... etc for all variables

# Start bridge without promoted config
python -m wealth_first.tradingview_bridge serve-env
```

---

## Support & Documentation

- **Validation Report:** `artifacts/main4_neighborhood_search.json`
- **Analysis Details:** `PROMOTION_ANALYSIS_2026-04-24.md`
- **Gate Logic:** `src/wealth_first/main4.py` → `_evaluate_gate_checks()`
- **Policy Training:** `src/wealth_first/medium_capacity.py`

---

**Promoted:** 2026-04-24  
**Status:** Stable ✓  
**Validated Folds:** 5  
**Gate:** PASS ✓  
**Friction Range:** 12-15 bps
