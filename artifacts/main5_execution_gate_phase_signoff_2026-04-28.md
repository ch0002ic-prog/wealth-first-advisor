# Main5 Execution-Gate Phase Signoff (2026-04-28)

## Final Status
- Phase status: ready_for_next_wealth_first_mechanism_phase
- Locked tolerance: 1e-10
- Promoted economic leader: l_s092625468_objw1
- Fallback near-tie: j_s0926254_objw1

## Evidence Summary
- Tolerance lock artifact: main5_execution_gate_tolerance_deeper_investigation_2026-04-28.md
- Guardrail pass artifact: main5_execution_gate_tolerance_guardrail_check.json
- Evaluator+coverage pass artifact: main5_execution_gate_tolerance_evaluator_deeper_2026-04-28.md
- Promotion confirmation artifact: main5_execution_gate_promotion_confirmation_seed6_2026-04-28.md
- Activity-threshold policy artifact: main5_execution_gate_activity_threshold_sensitivity_2026-04-28.md

## Strengths
- Tolerance policy is now evidence-locked with localized benefit and no observed broad collateral drift on checked panels.
- Guardrail suite passes and explicitly captures both lower-cliff bad/good regimes and tolerance drift envelope behavior.
- Validation family coverage now includes reverse-blind rows, closing the prior coverage gap for release decisions.
- Economic winner is stable across a larger six-seed expanded panel under locked tolerance.

## Weaknesses
- Economic delta between l468 and j6254 is very small; ranking is robust but near-tied, so tie-handling policy must be explicit.
- Surrogate family still has large outlier error and should not be used as primary promotion signal.
- Strict activity threshold (max_supp=0.997) can invert rankings toward lower-return 1.5-step candidates.

## Missing Items
- A final automated release gate command that combines guardrail pass, evaluator pass, coverage pass, and promotion winner consistency in one executable check.
- A post-promotion monitoring protocol (e.g., weekly drift check) for l468 vs j6254 to catch future reroute/regime shifts early.

## Next Steps (Wealth-First Optimization)
- Keep l468 as promoted economic leader and j6254 as fallback in decision docs and runtime config metadata.
- Use boundary1-aligned activity threshold (max_supp=0.9988) for promotion ranking; keep strict 0.997 for stress diagnostics only.
- Add one script-level composite release gate and wire it into CI for future main5 promotions.
- Begin next wealth-first mechanism search with tolerance locked at 1e-10 and compare only against l468 baseline + j6254 fallback.

## Key Numeric Anchors
- l468 vs j6254 mean_test_relative delta: 0.000000001310126
- l468 vs q mean_test_relative delta: 0.000868831402
- robust_mae/rmse/max_abs_error: 0.00149983925240248 / 0.0016420727501936607 / 0.0026398958914296364
- promotion activity ranking threshold: min_steps=1.0, max_supp=0.9988
- stress diagnostic threshold: min_steps=1.0, max_supp=0.997
