# NLS exact-pilot two-column diagnostic

Scanned 17,955/17,955 column pairs for each of 2 equations (35,910 LPs), using one HiGHS thread; runtime 141.2 s.

The LP fits every support without true coefficients. Column and response scaling only changes coordinates; residuals are checked again in the saved physical score. True coefficients enter only the final error comparison.

| Equation | Best two-column residual / ‖b‖∞ | Smallest exact dual bound / ‖b‖∞ | Exact exclusions at λ / ‖b‖∞ = .01 |
| :--- | ---: | ---: | ---: |
| u_t = 0.5 v_xx + u^2 v + v^3 | 0.014882314 | 0.014882314 | 17,955/17,955 |
| v_t = -0.5 u_xx - u v^2 - u^3 | 0.0158630107 | 0.0158630107 | 17,955/17,955 |

For each row witness, exact rational nonnegative weights sum to one and annihilate both physical columns. Their signed response sum is a rigorous lower bound for the saved binary64 score. The NPZ stores the signed row IDs so every witness can be reconstructed from the hashed score cache.

- Equation 0: a saved feasible three-term candidate has relative E₂ 2.74141327e-05. All supports of size ≤2 are independently excluded for its λ (dual lower bound / λ = 1.48823). Thus its cardinality is 3 for this floating score. The primal residual is also checked with exact rational arithmetic: 0.00455382276 ≤ λ=25.7407399.
  Coefficients: v_xx = 0.500012582253, u^2 v = 1.00003691583, v^3 = 1.00001303308.
- Equation 1: a saved feasible three-term candidate has relative E₂ 2.75936053e-05. All supports of size ≤2 are independently excluded for its λ (dual lower bound / λ = 1.5863). Thus its cardinality is 3 for this floating score. The primal residual is also checked with exact rational arithmetic: 0.0041090045 ≤ λ=25.67399.
  Coefficients: u_xx = -0.499984553614, u^3 = -0.999980500574, u v^2 = -0.999966919078.

This is an independent optimization diagnostic. The main experiment and its Gurobi certificate flags remain unchanged. Residual tie-break optimality is not established here; the result applies to the saved finite-grid candidates and score, rather than a continuous-λ oracle or a theorem for the population score.
