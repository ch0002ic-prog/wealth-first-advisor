# Main5 Activity Threshold Sensitivity (2026-04-28)

- Source: main5_deep_edgeband_tol1e10_deeper_detail.csv

## Ranking Under strict_0997 (min_steps=1.0, max_supp=0.997)
| candidate | breaches | activity_violations | mean_test_relative | min_case_slack | mean_executed_steps | mean_gate_suppression |
|---|---:|---:|---:|---:|---:|---:|
| j_s0926258_objw1 | 0 | 0 | 0.010985216485 | 0.006828024117 | 1.50 | 0.996767 |
| j_s0926256_objw1 | 0 | 0 | 0.010985215428 | 0.006828024490 | 1.50 | 0.996767 |
| l_s092625468_objw1 | 0 | 32 | 0.017656285777 | 0.007059729458 | 1.00 | 0.997845 |
| j_s0926254_objw1 | 0 | 32 | 0.017656284468 | 0.007059729457 | 1.00 | 0.997845 |
| j_s0926252_objw1 | 0 | 32 | 0.017656280619 | 0.007059729451 | 1.00 | 0.997845 |
| j_s0926250_objw1 | 0 | 32 | 0.017656276769 | 0.007059729445 | 1.00 | 0.997845 |
| j_s0926248_objw1 | 0 | 32 | 0.017656272920 | 0.007059729440 | 1.00 | 0.997845 |
| j_s0926246_objw1 | 0 | 32 | 0.017656269071 | 0.007059729434 | 1.00 | 0.997845 |

## Ranking Under boundary1_09988 (min_steps=1.0, max_supp=0.9988)
| candidate | breaches | activity_violations | mean_test_relative | min_case_slack | mean_executed_steps | mean_gate_suppression |
|---|---:|---:|---:|---:|---:|---:|
| l_s092625468_objw1 | 0 | 0 | 0.017656285777 | 0.007059729458 | 1.00 | 0.997845 |
| j_s0926254_objw1 | 0 | 0 | 0.017656284468 | 0.007059729457 | 1.00 | 0.997845 |
| j_s0926252_objw1 | 0 | 0 | 0.017656280619 | 0.007059729451 | 1.00 | 0.997845 |
| j_s0926250_objw1 | 0 | 0 | 0.017656276769 | 0.007059729445 | 1.00 | 0.997845 |
| j_s0926248_objw1 | 0 | 0 | 0.017656272920 | 0.007059729440 | 1.00 | 0.997845 |
| j_s0926246_objw1 | 0 | 0 | 0.017656269071 | 0.007059729434 | 1.00 | 0.997845 |
| j_s0926244_objw1 | 0 | 0 | 0.017656265221 | 0.007059729429 | 1.00 | 0.997845 |
| j_s0926242_objw1 | 0 | 0 | 0.017656261372 | 0.007059729423 | 1.00 | 0.997845 |

## Interpretation
- strict_0997 favors 1.5-step candidates but with materially lower mean_test_relative.
- boundary1_09988 restores wealth-first ordering around l468/j6254 and removes strict-threshold false penalties.
- Use boundary1-aligned activity gate for promotion decisions; reserve strict gate for stress diagnostics.
