# execution_gate_tolerance deeper stress report

## Panel

- Label family: deep_tolstress_expanded_*
- Candidates: 14 near-boundary names mined by minimum |case_slack|
- Scenarios: expanded pack (8 scenarios)
- Seeds: 5, 17
- Costs: 25, 35
- Cases per run: 448
- Matching keys: candidate, scenario, seed, transaction_cost_bps
- Delta threshold: abs(delta) > 1e-15

## Frontier summary

| tolerance | matched_rows | changed_rows | moved_candidates | mean_delta_test | mean_delta_steps | moved_list |
|---|---:|---:|---:|---:|---:|---|
| 1e-10 | 448 | 32 | 1 | 0.000476505 | -0.035714 | l_s092625468_objw1 |
| 1.2e-10 | 448 | 32 | 1 | 0.000476505 | -0.035714 | l_s092625468_objw1 |
| 1.3e-10 | 448 | 32 | 1 | 0.000476505 | -0.035714 | l_s092625468_objw1 |
| 1.4e-10 | 448 | 32 | 1 | 0.000476505 | -0.035714 | l_s092625468_objw1 |
| 1.5e-10 | 448 | 64 | 2 | 0.000953010 | -0.071429 | l_s092625468_objw1, l_s092625470_objw1 |

## Bisection scan (2-candidate focused, 64 cases per run)

Label family: deep_tolscan2_expanded_*  
Candidates: l_s092625468_objw1, l_s092625470_objw1  
Scenarios: expanded pack (8 scenarios) · Seeds: 5, 17 · Costs: 25, 35

| tolerance | changed_rows | moved_candidates | moved_list |
|---|---:|---:|---|
| base (0.0) | 0 | 0 | — |
| 1.45e-10 | 32 | 1 | l_s092625468_objw1 |
| 1.46e-10 | 32 | 1 | l_s092625468_objw1 |
| 1.47e-10 | 64 | 2 | l_s092625468_objw1, l_s092625470_objw1 |
| 1.48e-10 | 64 | 2 | l_s092625468_objw1, l_s092625470_objw1 |
| 1.49e-10 | 64 | 2 | l_s092625468_objw1, l_s092625470_objw1 |
| 1.5e-10  | 64 | 2 | l_s092625468_objw1, l_s092625470_objw1 |

## Candidate first-flip tolerance (bisection-refined)

- l_s092625468_objw1: ≤ 1e-10 (already flipped at lowest stress panel point)
- l_s092625470_objw1: **(1.46e-10, 1.47e-10]** — onset window resolved by bisection