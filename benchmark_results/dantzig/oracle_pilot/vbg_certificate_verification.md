# Independent VBG certificate verification

Passed: **True**. Verified 123,410 exact dual witnesses covering every four-column support, without rerunning any LP.

The verifier independently solves each witness by fraction-free Gaussian elimination and checks exact physical stationarity, nonnegative mass one, dual l1 norm ≤1, and a strictly larger lower bound than the stored target lambda.

Smallest exact dual bound / target lambda: **9.64588110322**.

| Saved lambda / ‖b‖∞ | Relative E2 | Minimum dual bound / lambda | Minimum cardinality5 proved |
| ---: | ---: | ---: | :--- |
| 1e-05 | 1.28615113167e-10 | 0.96458811 | False |
| 3.16227766e-06 | 2.52567906941e-10 | 3.05029543 | True |
| 1e-06 | 2.52567906941e-10 | 9.6458811 | True |
| 5.62341325e-06 | 2.52567906941e-10 | 1.71530718 | True |

Every smaller support is contained in a checked four-column support. Exact feasibility of a nonzero five-term vector therefore establishes minimum cardinality five at each certified saved lambda.

The lambda ratio1e-5 candidate is outside this certificate's bound and is excluded from the augmented reporting pool. The proof does not establish residual tie-break optimality or continuum-score accuracy. Main flags remain unchanged.
