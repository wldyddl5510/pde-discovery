## IB case study: wsindy, lasso, best-subset, filtered, filtered-lasso, filtered-best-subset, debiased, debiased-lasso, debiased-best-subset, eiv on single seeds (noise ratio 1)

Trials of `run.py` (same observations as the tables). Objective = ||b - G w||^2 of the method's own regression (G, b from its loss) at the returned coefficients, in the scaled weak system; for LASSO and EIV, which minimize other objectives, only this least-squares part. `wsindy` is raw WSINDy with MSTLS; `best-subset` solves the r = 1 MIO with Gurobi. E2 and E_inf as in the tables. Truth: u_t = -0.5 (u^2)_x.

| Trial | Method | Objective | Selected PDE | E2 | E_inf | Gurobi optimal |
| ---: | :--- | ---: | :--- | ---: | ---: | :---: |
| 0 | wsindy | 0.8892 | u_t = -0.4949 (u^2)_x | 0.01021 | 0.01021 | — |
| 0 | lasso | 0.8392 | u_t = -331.4 u_x -0.1281 (u^2)_x | 662.8 | 0.7438 | — |
| 0 | best-subset | 0.7465 | u_t = -494.6 u_x | 989.2 | 1 | yes |
| 0 | filtered | 0.6246 | u_t = +3.797 1 -0.0544 u -0.6144 (u^2)_x -0.09039 (u^4)_xxx -0.004108 (u^6)_xxxxx +0.1859 (u^6)_xxxxxx | 7.61 | 0.2287 | — |
| 0 | filtered-lasso | 0.6106 | u_t = -403.2 u_x -0.0741 (u^2)_x | 806.4 | 0.8518 | — |
| 0 | filtered-best-subset | 0.7824 | u_t = -495.9 u_x | 991.7 | 1 | yes |
| 0 | debiased | 2.6219 | u_t = -0.4705 (u^2)_x | 0.05907 | 0.05907 | — |
| 0 | debiased-lasso | 2.0316 | u_t = -407.6 u_x -0.04778 (u^2)_x | 815.2 | 0.9044 | — |
| 0 | debiased-best-subset | 1.9742 | u_t = -481.8 u_x | 963.7 | 1 | yes |
| 0 | eiv | 1.0381 | 43 nonzero terms; largest: -0.318 (u^2)_x -124.9 u_x +1.131e-05 u_xxx | 249.8 | 0.364 | — |
| 1 | wsindy | 0.8439 | u_t = -0.4906 (u^2)_x | 0.01881 | 0.01881 | — |
| 1 | lasso | 0.9199 | u_t = -0.2322 (u^2)_x -9.125e-05 (u^3)_x | 0.5356 | 0.5356 | — |
| 1 | best-subset | 0.8439 | u_t = -0.4906 (u^2)_x | 0.01881 | 0.01881 | yes |
| 1 | filtered | 0.7037 | u_t = +11.52 1 -0.07598 u -0.6082 (u^2)_x -0.08846 (u^4)_xxx -0.004118 (u^6)_xxxxx +0.1761 (u^6)_xxxxxx | 23.04 | 0.2163 | — |
| 1 | filtered-lasso | 0.6498 | u_t = -427.7 u_x -0.03451 (u^2)_x | 855.5 | 0.931 | — |
| 1 | filtered-best-subset | 0.5817 | u_t = -484.5 u_x | 968.9 | 1 | yes |
| 1 | debiased | 2.2196 | u_t = -0.4692 (u^2)_x | 0.06167 | 0.06167 | — |
| 1 | debiased-lasso | 1.7296 | u_t = -344 u_x -0.128 (u^2)_x | 687.9 | 0.7439 | — |
| 1 | debiased-best-subset | 1.7792 | u_t = -495.3 u_x | 990.6 | 1 | yes |
| 1 | eiv | 1.0814 | 43 nonzero terms; largest: -401 u_x -0.03315 (u^2)_x +9.494e+06 u_xxx | 1.899e+07 | 0.9337 | — |
| 2 | wsindy | 1.1664 | u_t = -0.4938 (u^2)_x | 0.01232 | 0.01232 | — |
| 2 | lasso | 1.1889 | u_t = -0.478 (u^2)_x | 0.04392 | 0.04392 | — |
| 2 | best-subset | 1.1664 | u_t = -0.4938 (u^2)_x | 0.01232 | 0.01232 | yes |
| 2 | filtered | 1.6553 | u_t = +9.235 1 +0.02674 u -0.3705 (u^2)_x +0.2215 (u^3)_xx +0.03346 (u^5)_xxxx +1.244 (u^6)_xxxxxx | 18.64 | 0.2589 | — |
| 2 | filtered-lasso | 1.7586 | u_t = -451.3 u_x -0.0359 (u^2)_x | 902.5 | 0.9282 | — |
| 2 | filtered-best-subset | 1.8095 | u_t = -492.9 u_x | 985.8 | 1 | yes |
| 2 | debiased | 2.8708 | u_t = -0.4809 (u^2)_x | 0.03814 | 0.03814 | — |
| 2 | debiased-lasso | 2.6179 | u_t = -269.6 u_x -0.208 (u^2)_x | 539.2 | 0.584 | — |
| 2 | debiased-best-subset | 2.7713 | u_t = -500.1 u_x | 1000 | 1 | yes |
| 2 | eiv | 1.1742 | 43 nonzero terms; largest: -0.4629 (u^2)_x -3.909e-12 (u^5)_x +9.591e-19 (u^3)_x | 0.1717 | 0.07415 | — |
