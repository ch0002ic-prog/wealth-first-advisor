# main5 execution gate tolerance frontier (ZL/ZN)

## ZL micro canary: baseline-relative movement

| tolerance | changed_rows | moved_candidates | mean_delta_test | mean_delta_steps | candidates |
|---|---:|---:|---:|---:|---|
| 1e-10 | 16 | 1 | 0.000625413 | -0.046875 | l_s092625468_objw1 |
| 1.2e-10 | 16 | 1 | 0.000625413 | -0.046875 | l_s092625468_objw1 |
| 1.5e-10 | 32 | 2 | 0.001250826 | -0.093750 | l_s092625468_objw1, l_s092625470_objw1 |
| 1.8e-10 | 32 | 2 | 0.001250826 | -0.093750 | l_s092625468_objw1, l_s092625470_objw1 |
| 2e-10 | 32 | 2 | 0.001250826 | -0.093750 | l_s092625468_objw1, l_s092625470_objw1 |
| 5e-10 | 80 | 5 | 0.003127065 | -0.234375 | l_s092625468_objw1, l_s092625470_objw1, l_s092625472_objw1, l_s092625474_objw1, l_s092625476_objw1 |
| 1e-9 | 112 | 7 | 0.004377890 | -0.328125 | k_s09262548_objw1, l_s092625468_objw1, l_s092625470_objw1, l_s092625472_objw1, l_s092625474_objw1, l_s092625476_objw1, l_s092625478_objw1 |

## ZL first-flip thresholds by candidate

- k_s09262548_objw1: 1e-9
- l_s092625468_objw1: 1e-10
- l_s092625470_objw1: 1.5e-10
- l_s092625472_objw1: 5e-10
- l_s092625474_objw1: 5e-10
- l_s092625476_objw1: 5e-10
- l_s092625478_objw1: 1e-9

## ZN lower cliff stability

- tolerance 1e-10: matched_rows=96, aggregate_changed_counts=0
- tolerance 1e-9: matched_rows=96, aggregate_changed_counts=0

## Recommendation

- Keep execution_gate_tolerance at 1e-10 as the max default for locality-safe fixes.
- 1.2e-10 is observationally identical to 1e-10 in this panel, but 1.5e-10 already introduces a second moved candidate.