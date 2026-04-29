# Main6 Optimization Matrix (2026-04-28)

## Scope
This matrix is designed to improve relative return while preserving the current safety profile from the promoted `objw05_ntb005_floor` defaults.

## Baseline Safety Anchor
- Candidate: `objw05_ntb005_floor`
- Gate condition: `breach == False` for every case
- Health condition: `dead_fold_fraction <= 0.0` target (hard ceiling 0.20)
- Windows:
  - `2020-01-01` to `2026-04-17`
  - `2015-01-01` to `2019-12-31`

## Phase 1: Existing Candidate Frontier (No Code Changes)
Use only pre-defined candidates in `scripts/investigate_main6_canary.py`.

### Candidate Set A (ntb + floor behavior)
- `baseline_ntb005`
- `objw05_ntb005`
- `objw05_ntb005_floor`
- `objw05_ntb005_floor_hardmin`

### Candidate Set B (ntb010 control family)
- `objw05_ntb010_floor`
- `objw05_ntb010_floor30`
- `objw05_ntb010_floor35`
- `objw05_ntb010_floor_hardmin`

## Commands
Run each set on both windows.

### Set A on 2020-2026
```bash
.venv/bin/python scripts/investigate_main6_canary.py \
  --label canary_2026_opt_setA_2020_2026 \
  --start-date 2020-01-01 \
  --end-date 2026-04-17 \
  --candidates baseline_ntb005,objw05_ntb005,objw05_ntb005_floor,objw05_ntb005_floor_hardmin
```

### Set A on 2015-2019
```bash
.venv/bin/python scripts/investigate_main6_canary.py \
  --label canary_2026_opt_setA_2015_2019 \
  --start-date 2015-01-01 \
  --end-date 2019-12-31 \
  --candidates baseline_ntb005,objw05_ntb005,objw05_ntb005_floor,objw05_ntb005_floor_hardmin
```

### Set B on 2020-2026
```bash
.venv/bin/python scripts/investigate_main6_canary.py \
  --label canary_2026_opt_setB_2020_2026 \
  --start-date 2020-01-01 \
  --end-date 2026-04-17 \
  --candidates objw05_ntb010_floor,objw05_ntb010_floor30,objw05_ntb010_floor35,objw05_ntb010_floor_hardmin
```

### Set B on 2015-2019
```bash
.venv/bin/python scripts/investigate_main6_canary.py \
  --label canary_2026_opt_setB_2015_2019 \
  --start-date 2015-01-01 \
  --end-date 2019-12-31 \
  --candidates objw05_ntb010_floor,objw05_ntb010_floor30,objw05_ntb010_floor35,objw05_ntb010_floor_hardmin
```

## Promotion Rule
Promote only if all conditions pass:
1. Zero breaches across both windows.
2. `dead_fold_fraction` does not exceed the incumbent on either window.
3. Mean `mean_test_relative` improves on incumbent in at least one window and does not materially degrade in the other.
4. `robust_min_p05` is not worse than incumbent by more than 5e-4 on either window.

## Missing Item After Phase 1
If no candidate dominates, add a Phase 2 with new candidate definitions around:
- `validation_relative_floor_target` near `0.0002`
- `validation_relative_floor_penalty` near `4.0`
- `scale_turnover_penalty` near `2.9952`
- `no_trade_band` near `0.005`

Then re-run the same 2-window canary protocol.
