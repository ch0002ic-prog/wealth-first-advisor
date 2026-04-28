# Main5 Promotion Confirmation (Seed6, 2026-04-28)

## Run Scope
- Locked tolerance: 1e-10
- Expanded scenarios, seeds 5/17/29/37/47/57, costs 22/25/30/35, reps=40
- Candidates: l_s092625468_objw1, j_s0926254_objw1, q_s09231_h10_ntb050

## Core Result
- Winner: l_s092625468_objw1 (wealth-first objective)
- l468 vs j6254 delta mean_test_relative = 0.000000001310126
- l468 vs j6254 delta min_case_slack = 0.000000000037369

## Baseline Comparison
- l468 vs q delta mean_test_relative = 0.000868831402
- l468 vs q delta min_case_slack = 0.000543173351

## Candidate Table
| candidate | rows | breaches | min_case_slack | mean_test_relative | mean_turnover | mean_executed_steps | mean_gate_suppression |
|---|---:|---:|---:|---:|---:|---:|---:|
| l_s092625468_objw1 | 192 | 0 | 0.007543173351 | 0.017674081238 | 0.000100325565 | 1.000000 | 0.997844827586 |
| j_s0926254_objw1 | 192 | 0 | 0.007543173313 | 0.017674079928 | 0.000100325558 | 1.000000 | 0.997844827586 |
| q_s09231_h10_ntb050 | 192 | 0 | 0.007000000000 | 0.016805249836 | 0.000095014159 | 0.947917 | 0.997957076149 |

## Decision
- Promote/retain l_s092625468_objw1 as economic leader under locked tolerance.
- Keep j_s0926254_objw1 as near-tie fallback candidate.
- Use boundary1-aligned activity threshold (max_supp=0.9988) for promotion ranking; strict 0.997 remains stress-diagnostic only.
