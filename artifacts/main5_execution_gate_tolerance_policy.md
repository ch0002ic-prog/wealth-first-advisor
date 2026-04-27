# main5 execution_gate_tolerance policy

## Decision

- Approved default: execution_gate_tolerance = 1e-10.
- Non-default investigational: 1.2e-10 to 1.4e-10 (observationally identical in current stress panel; first second-candidate onset is above 1.46e-10).
- Not approved for default: 1.47e-10 and larger (second candidate `l_s092625470_objw1` confirmed flipped at 1.47e-10; onset window bisection-resolved to **(1.46e-10, 1.47e-10]**).

## First-flip onset (bisection-resolved)

| Candidate | First-flip window |
|---|---|
| l_s092625468_objw1 | ≤ 1e-10 (already present at lowest stress panel point) |
| l_s092625470_objw1 | **(1.46e-10, 1.47e-10]** — confirmed by 64-case focused bisection scan |

## Evidence Baseline

- Broad panel invariance at 1e-10: deep_zi_edge_confirm baseline vs tol1e10 must remain unchanged.
- Expanded stress frontier: second moved candidate must not appear before 1.47e-10.
- Bisection scan frontier: `_468` only at 1.45e-10 and 1.46e-10; both candidates at 1.47e-10 and above.

## Required Guardrail Command

Run:

```bash
.venv/bin/python scripts/check_main5_execution_gate_guardrails.py
```

Pass criteria:

- deep_zi_1e10: changed_rows == 0.
- stress_1e10 moved set == [l_s092625468_objw1].
- stress_1p2e10 moved set == [l_s092625468_objw1].
- stress_1p3e10 moved set == [l_s092625468_objw1].
- stress_1p4e10 moved set == [l_s092625468_objw1].
- stress_1p5e10 moved set == [l_s092625468_objw1, l_s092625470_objw1].
- bisect_tol145e12 moved set == [l_s092625468_objw1].
- bisect_tol146e12 moved set == [l_s092625468_objw1].
- bisect_tol147e12 moved set == [l_s092625468_objw1, l_s092625470_objw1].
- bisect_tol148e12 moved set == [l_s092625468_objw1, l_s092625470_objw1].

If any criterion fails, treat as regression and block promotion until re-investigated.

## Change Control

- Any proposal to raise default tolerance above 1e-10 requires:
  - refreshed expanded stress sweep,
  - refreshed broad panel invariance check,
  - updated frontier and first-flip report artifacts,
  - policy update with explicit approval note.
