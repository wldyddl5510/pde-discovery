# Independent NLS certificate verification

Passed: **True**. Verified 35,910 exact-rational dual witnesses, covering every one of 17,955 pairs in each equation. No LP or MILP was rerun.

The verifier independently reconstructs weights with rational Gaussian elimination. It checks exact physical H stationarity, nonnegative unit mass, the actual dual vector's l1 norm ≤ 1, and an exact dual objective strictly above the stored physical lambda.

| Equation | Verified pair exclusions | Minimum bound / lambda at .01 | Saved feasible three-term E2 |
| ---: | ---: | ---: | ---: |
| 0 | 17,955 | 1.4882314 | 2.74141327e-05 |
| 1 | 17,955 | 1.58630107 | 2.75936053e-05 |

Every support of size ≤2 is contained in a checked pair. The saved three-term coefficient vectors were checked against all physical constraints with exact rational arithmetic. Thus their minimum cardinality is three for the stated saved score and lambda.

This proof does not establish residual tie-break optimality. Main experiment flags and summaries remain unchanged. It concerns the saved binary64 score and finite lambda candidates, not continuum integral exactness.
