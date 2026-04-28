# Main5 Execution-Gate Monitoring Protocol (2026-04-28)

## Scope
- Promoted leader: l_s092625468_objw1
- Fallback: j_s0926254_objw1
- Locked tolerance: 1e-10

## Cadence
- Weekly: run quick drift check.
- Monthly: run expanded panel check.
- Pre-promotion (any new candidate): run full release gate.

## Weekly Drift Check
- Run leader vs fallback only on seeds 5/17/29 and costs 25/35.
- Compare:
  - mean_test_relative delta (l468 - j6254)
  - min_case_slack delta
  - breach counts
- Alert conditions:
  - delta_mean_test_relative <= 0
  - any breach on either candidate
  - min_case_slack drops below 0.007 for either candidate

## Monthly Expanded Check
- Run leader vs fallback on expanded scenarios with seeds 5/17/29/37/47/57 and costs 22/25/30/35.
- Compare same metrics as weekly plus:
  - mean_turnover
  - mean_gate_suppression
- Alert conditions:
  - weekly alert conditions persist for 2 consecutive weeks
  - monthly delta_mean_test_relative <= 0

## Release Gate Command
- Composite gate script:
  - scripts/check_main5_release_gate.py
- Pass criteria:
  - guardrail status pass
  - evaluator status pass
  - family coverage check pass
  - robust threshold check pass
  - promotion winner equals l_s092625468_objw1
  - boundary1 activity-threshold top candidate equals l_s092625468_objw1

## Tie Handling Policy
- If l468 and j6254 are near-tied, keep l468 unless l468-vs-j6254 mean_test_relative delta becomes non-positive.
- If non-positive, suspend promotion updates and run deeper tie-break panel before changing leader.
