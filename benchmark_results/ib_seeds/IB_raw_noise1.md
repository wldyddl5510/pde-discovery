## IB case study: wsindy, best-subset on single seeds (noise ratio 1)

Trials of `run.py` (same observations as the tables). Objective = ||b - G w||^2 of the method's own regression (G, b from its loss) at the returned coefficients, in the scaled weak system; for LASSO and EIV, which minimize other objectives, only this least-squares part. `wsindy` is raw WSINDy with MSTLS; `best-subset` solves the r = 1 MIO with Gurobi. E2 and E_inf as in the tables. Truth: u_t = -0.5 (u^2)_x.

| Trial | Method | Objective | Selected PDE | E2 | E_inf | Gurobi optimal |
| ---: | :--- | ---: | :--- | ---: | ---: | :---: |
| 0 | wsindy | 0.8892 | u_t = -0.4949 (u^2)_x | 0.01021 | 0.01021 | — |
| 0 | best-subset | 0.7465 | u_t = -494.6 u_x | 989.2 | 1 | yes |
| 5 | wsindy | 0.8327 | u_t = -0.477 (u^2)_x | 0.04603 | 0.04603 | — |
| 5 | best-subset | 0.8216 | u_t = -0.0001943 (u^3)_x | 1 | 1 | yes |
| 7 | wsindy | 0.8407 | u_t = -0.5081 (u^2)_x | 0.01623 | 0.01623 | — |
| 7 | best-subset | 0.7284 | u_t = -512.9 u_x | 1026 | 1 | yes |
