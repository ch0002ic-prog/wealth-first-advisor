# Main5 Mechanism Edge-Band Deeper Investigation (2026-04-28)

## Setup
- Locked tolerance: execution_gate_tolerance = 1e-10
- Panel: expanded scenarios, seeds 5/17, costs 25/35, reps=30
- Candidate strip: q baseline + h/j/l edge neighbors around 0.923625x

## Decision Views
- Economic ranking winner: l_s092625468_objw1
- Activity-first ranking winner: j_s0926258_objw1

## Key Comparisons
- l468 vs q: delta mean_test_relative = 0.000524800995
- l468 vs q: delta min_case_slack = 0.000059729458
- l468 vs q: delta mean_executed_steps = 0.031250
- l468 vs j6254: delta mean_test_relative = 0.000000001309

## Top 5 Economic Candidates (zero-breach set)
| candidate | mean_test_relative | min_case_slack | activity_violations | mean_executed_steps | mean_gate_suppression |
|---|---:|---:|---:|---:|---:|
| l_s092625468_objw1 | 0.017656285777 | 0.007059729458 | 32 | 1.00 | 0.997845 |
| j_s0926254_objw1 | 0.017656284468 | 0.007059729457 | 32 | 1.00 | 0.997845 |
| j_s0926252_objw1 | 0.017656280619 | 0.007059729451 | 32 | 1.00 | 0.997845 |
| j_s0926250_objw1 | 0.017656276769 | 0.007059729445 | 32 | 1.00 | 0.997845 |
| j_s0926248_objw1 | 0.017656272920 | 0.007059729440 | 32 | 1.00 | 0.997845 |

## Interpretation
- Under economic objective (wealth-first), l_s092625468_objw1 remains the strongest point in this strip.
- j_s0926254_objw1 and nearby h/j neighbors are effectively tied but slightly below l468.
- j6256/j6258 move to a higher-activity (1.5-step) regime with materially lower mean_test_relative.
- No new post-l468 winner was discovered in this deeper local strip.
