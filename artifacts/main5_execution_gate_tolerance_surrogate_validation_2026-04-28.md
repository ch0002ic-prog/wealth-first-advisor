# Main5 Surrogate Validation (2026-04-28)

Validated surrogate predictions against new deep probes for l474 and k52.

| candidate | predicted | bad_through | good_from | midpoint | rel_error |
|---|---:|---:|---:|---:|---:|
| l_s092625474_objw1 | 3.4903e-10 | 3.477e-10 | 3.478e-10 | 3.4775e-10 | +0.368% |
| k_s09262552_objw1 | 4.297226502e-09 | 2.65e-09 | 2.66e-09 | 2.655e-09 | +61.854% |

Notes:
- l474 linear extrapolation is nearly exact (0.368% midpoint error).
- k52 log-linear extrapolation overestimates threshold materially; local manifold is not globally log-linear across k48->k52.
