# Optimization Research Audit

## Key Findings
- main3 base6 test mean relative return: -0.036315
- main3 rich15 test mean relative return: -0.035442
- main3 rich15 test uplift vs base6: 0.000873
- main4 best stress case: low_friction
- main4 best mean relative return: 0.001570
- main4 uplift vs baseline: 0.000014

## Recommended Priorities
- Do not spend the next cycle on more linear feature expansion alone; rich15 only marginally improves main3 and remains negative on test.
- Prioritize the medium-capacity path around main4: extend regime features, tighten action gating, and keep the evaluation harness friction-aware.
- Use 'low_friction' as the current promoted main4 stress configuration and rerun deeper validation before adding model complexity.
- If you revisit DRL later, constrain it to a bounded-delta overlay that must beat main4 under the same walk-forward and friction tests.
- Add an experiment harness that ranks candidates by full-sample metrics, decisive-only metrics, decision rate, and turnover so improvements are not hidden by abstention or inactivity.

## Generated Artifacts
- structural summary: /Users/ch0002techvc/Downloads/wealth-first-investing/artifacts/optimization_research_structural_summary.json
- main4 stress summary: /Users/ch0002techvc/Downloads/wealth-first-investing/artifacts/main4_stress_suite_summary.json
- audit summary: /Users/ch0002techvc/Downloads/wealth-first-investing/artifacts/optimization_research_audit_summary.json