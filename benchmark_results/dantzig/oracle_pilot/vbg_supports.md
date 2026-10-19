# VBG exact-pilot four-column diagnostic

Scanned 123,410/123,410 supports in 288.6s with one HiGHS thread. Exact physical dual witnesses: 123,410; exact exclusions at λ/‖b‖∞=1e-06: 123,410.

Smallest numerical four-column residual / ‖b‖∞: 9.64588110351e-06. Smallest exact dual bound / ‖b‖∞: 9.64588110322e-06.

Every witness uses at most five signed rows of the original physical H,b. Integer cofactors produce nonnegative exact rational weights summing to one and annihilating all four columns. The NPZ stores signed row IDs for independent reconstruction. Every five-term primal candidate is also checked exactly.

| Saved λ / ‖b‖∞ | Relative E₂ | Exact dual lower bound / λ | Independently minimum cardinality 5 |
| ---: | ---: | ---: | :--- |
| 1e-05 | 1.28615113e-10 | 0.96458811 | False |
| 3.16227766e-06 | 2.52567907e-10 | 3.05029543 | True |
| 1e-06 | 2.52567907e-10 | 9.6458811 | True |
| 5.62341325e-06 | 2.52567907e-10 | 1.71530718 | True |

An independently cardinality-optimal saved five-term vector has coefficients:

- 1 = 1.0000000002703
- u_xx = 0.0099999999996875
- u^2 = 1.9999999994828
- (u^2)_x = -0.50000000000153
- u^3 = -0.99999999975882

Exact physical residual 7.11028649717e-26 ≤ λ=2.14289008652e-17. Relative coefficient error is 2.52567906941e-10.

The scan uses no true coefficient or support information. True coefficients are used only after the scan to evaluate saved candidates. Primary study files and Gurobi flags remain unchanged. This proves cardinality only for the saved floating score, not residual tie-break optimality or a continuous-λ oracle.
