# Promoted Main4 Policy Configuration (2026-04-24)

## Executive Summary

The promoted main4 deployment profile is now the target-aware branch (`target_mode=tanh_return`) validated under strict 5-fold gates.

Source of truth artifacts:
- `artifacts/main4_promotion_headtohead_full_detail.csv`
- `artifacts/main4_promotion_headtohead_full_summary.json`

Head-to-head outcome versus deployed sign-target baseline:
- `tanh_return` wins on all matched 12/15/18 bps scenarios.
- `tanh_return` passes strict robust-min gate in all matched scenarios.
- matched sign-target rows fail robust-min.

## Promoted Parameters

- `ridge_l2=0.015`
- `target_mode=tanh_return`
- `scale_turnover_penalty=0.0`
- `min_signal_scale=-0.75`
- `max_signal_scale=0.75`
- `min_spy_weight=0.80`
- `max_spy_weight=1.05`
- `initial_spy_weight=1.0`
- `action_smoothing=1.0`
- `transaction_cost_bps=12`
- `slippage_bps=12`
- `gate=55`
- `gate_scale=bps`
- `min_robust_min_relative=0.0`
- `n_folds=5`
- `seed=5`

## Validated Performance Snapshot

- 12 bps: mean test relative `+0.006693`, robust-min `+0.003661` (PASS)
- 15 bps: mean test relative `+0.006268`, robust-min `+0.003088` (PASS)
- 18 bps: mean test relative `+0.005505`, robust-min `+0.003043` (PASS)

## Deployment Methods

### Method 1: Source promoted env file

```bash
source config/promoted_main4.env
python -m wealth_first.tradingview_bridge serve-env
```

### Method 2: Runtime startup script

```bash
export WEALTH_FIRST_PROMOTED_CONFIG=/app/config/promoted_main4.env
./scripts/start_bridge_runtime.sh
```

### Method 3: Direct main4 validation command

```bash
python -m wealth_first.main4 \
  --ridge-l2 0.015 \
  --target-mode tanh_return \
  --scale-turnover-penalty 0.0 \
  --min-signal-scale -0.75 \
  --max-signal-scale 0.75 \
  --min-spy-weight 0.80 \
  --max-spy-weight 1.05 \
  --initial-spy-weight 1.0 \
  --action-smoothing 1.0 \
  --transaction-cost-bps 12 \
  --slippage-bps 12 \
  --gate 55 \
  --gate-scale bps \
  --min-robust-min-relative 0.0 \
  --n-folds 5 \
  --seed 5
```

## API Integration

`GET /api/config/main4-promoted` returns the promoted policy and deployment command assembled from runtime env.

Quick check:

```bash
curl -s http://localhost:8000/api/config/main4-promoted | jq .
```

## Operational Notes

- Promotion label in env: `Target-Aware Main4 Policy (tanh_return, 5-Fold Validated)`
- Promotion friction range in env: `12-18 bps`
- If env vars are missing from the container, verify `config/promoted_main4.env` is present and sourced by `scripts/start_bridge_runtime.sh`.
