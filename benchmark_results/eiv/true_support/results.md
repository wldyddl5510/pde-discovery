# EIV with the true support supplied

Paired diagnostic of the saved `benchmark_results/runs/eiv` results. Only IB, KdV and KS.

## Protocol

- Same observations/seeds, true noise variance, Hermite correction, weak equations, unit-norm columns, Gurobi conic solver and tolerance (1e-8) as the saved EIV benchmark.
- Keep each saved tau exactly; mu = 0. Forbid all columns outside the true support. Allowed true terms may still receive zero coefficients; their signs and values are not supplied.
- **Fixed lambda:** reuse the lambda selected in the full dictionary. **Oracle lambda:** repeat the same 13-value grid (1e-3 to 1e3) and select minimum physical coefficient E2.
- Original-solver failures are retained in the path and fixed-lambda diagnostics. The oracle column selects only converged, constraint-validated candidates; raw_oracle separately records the original unchecked argmin rule. Fixed-lambda averages include returned finite coefficients even when convergence was not certified; see the audit table.
- **OLS control:** unpenalized least squares on the same corrected matrix and true support. Its infinity-residual constraint is checked, not imposed.
- No coefficient thresholding or refitting after the conic solve. Support diagnostics use the original relative 1e-3 cutoff in normalized coordinates.
- E2 is relative L2 error over all coefficients. Forbidden coefficients are exactly zero. Exact support is not a useful standalone score here because the support was supplied.
- Noise level 0 repeats identical clean observations 100 times; those repeats are not independent data.

## Mean E2

| Task | Noise | Trials | Full EIV | True support, fixed lambda | True support, oracle lambda | True-support OLS |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| IB | 0 | 100 | 6.51529 | 0.000196043 | 0.000196035 | 4.30846e-05 |
| IB | 0.1 | 100 | 7.15125e+10 | 0.00606226 | 0.00606226 | 0.00237355 |
| IB | 0.2 | 100 | 2.02694e+09 | 0.0117743 | 0.0117743 | 0.00408295 |
| IB | 0.5 | 100 | 1.50187e+07 | 0.0381423 | 0.0381423 | 0.0106589 |
| IB | 0.75 | 100 | 1.22638e+11 | 0.0649556 | 0.0649556 | 0.0182255 |
| IB | 1 | 100 | 3.73013e+11 | 0.0848365 | 0.0848365 | 0.0217288 |
| IB | 2 | 100 | 4.46225e+12 | 0.240088 | 0.240088 | 0.0816916 |
| KdV | 0 | 100 | 9.15543e-06 | 1.23541e-06 | 1.74526e-06 | 2.84321e-07 |
| KdV | 0.1 | 100 | 5.60309 | 0.032907 | 0.0316287 | 0.00398894 |
| KdV | 0.2 | 100 | 12.7341 | 0.064225 | 0.0609816 | 0.00700317 |
| KdV | 0.5 | 100 | 49.305 | 0.153117 | 0.140676 | 0.0191497 |
| KdV | 0.75 | 100 | 98.046 | 0.236963 | 0.211418 | 0.0290362 |
| KdV | 1 | 100 | 79.7522 | 0.314927 | 0.286456 | 0.0377484 |
| KdV | 2 | 100 | 103.099 | 0.677725 | 0.615925 | 0.118143 |
| KS | 0 | 100 | 7.54185e-05 | 8.6886e-06 | 8.68648e-06 | 5.54625e-07 |
| KS | 0.1 | 100 | 0.434798 | 0.0476079 | 0.0474149 | 0.00406559 |
| KS | 0.2 | 100 | 0.728973 | 0.0873426 | 0.0866547 | 0.0104377 |
| KS | 0.5 | 100 | 0.841825 | 0.271649 | 0.264646 | 0.0670009 |
| KS | 0.75 | 100 | 0.912234 | 0.471935 | 0.459717 | 0.156629 |
| KS | 1 | 100 | 0.966407 | 0.721781 | 0.702803 | 0.272792 |
| KS | 2 | 100 | 0.999988 | 0.99955 | 0.999307 | 0.751704 |

## Median E2

Large false physical coefficients can dominate the full-dictionary mean; paired medians are also shown.

| Task | Noise | Full EIV | True support, fixed lambda | True support, oracle lambda | True-support OLS |
| :--- | ---: | ---: | ---: | ---: | ---: |
| IB | 0 | 6.51529 | 0.000196043 | 0.000196035 | 4.30846e-05 |
| IB | 0.1 | 101.481 | 0.00331428 | 0.00331428 | 0.00219142 |
| IB | 0.2 | 244.028 | 0.00398293 | 0.00398292 | 0.00342493 |
| IB | 0.5 | 271.745 | 0.0282741 | 0.0282741 | 0.00879773 |
| IB | 0.75 | 384.396 | 0.0584428 | 0.0584428 | 0.0152297 |
| IB | 1 | 352.87 | 0.0729033 | 0.0729033 | 0.0180717 |
| IB | 2 | 721.07 | 0.241026 | 0.241026 | 0.0692967 |
| KdV | 0 | 9.15543e-06 | 1.23541e-06 | 1.74526e-06 | 2.84321e-07 |
| KdV | 0.1 | 0.978211 | 0.0341884 | 0.0341884 | 0.0032587 |
| KdV | 0.2 | 2.99243 | 0.0567922 | 0.0538271 | 0.00587159 |
| KdV | 0.5 | 5.17389 | 0.145517 | 0.136714 | 0.0143916 |
| KdV | 0.75 | 4.63081 | 0.220279 | 0.185302 | 0.0240972 |
| KdV | 1 | 1.14487 | 0.298406 | 0.27458 | 0.026666 |
| KdV | 2 | 0.909976 | 0.731704 | 0.650505 | 0.104439 |
| KS | 0 | 7.54185e-05 | 8.6886e-06 | 8.68648e-06 | 5.54625e-07 |
| KS | 0.1 | 0.438573 | 0.0474913 | 0.0474913 | 0.00323557 |
| KS | 0.2 | 0.789654 | 0.0755352 | 0.074581 | 0.00938375 |
| KS | 0.5 | 0.839993 | 0.251459 | 0.245188 | 0.0675746 |
| KS | 0.75 | 0.915588 | 0.49734 | 0.478716 | 0.151947 |
| KS | 1 | 0.96848 | 0.757643 | 0.726643 | 0.269418 |
| KS | 2 | 1 | 1 | 1 | 0.752952 |

## Coefficients and feasibility

The coefficient ratios are beta_hat / beta_true, averaged per term: 1 = correct, 0 = dropped, values between 0 and 1 = attenuated magnitude.

### IB

Term order: (u^2)_x.

True coefficients: -0.5.

| Noise | Oracle EIV mean coefficient ratios | OLS mean coefficient ratios | EIV trials dropping a true term | OLS feasible | EIV objective below truth |
| ---: | :--- | :--- | ---: | ---: | ---: |
| 0 | 0.9998 | 1 | 0/100 | 100% | 100/100 |
| 0.1 | 0.9939 | 1 | 0/100 | 47% | 60/100 |
| 0.2 | 0.9882 | 1 | 0/100 | 56% | 55/100 |
| 0.5 | 0.9619 | 1 | 0/100 | 58% | 61/100 |
| 0.75 | 0.935 | 0.9936 | 0/100 | 61% | 69/100 |
| 1 | 0.9152 | 0.9909 | 0/100 | 62% | 71/100 |
| 2 | 0.7599 | 0.9388 | 0/100 | 71% | 86/100 |
### KdV

Term order: u_xxx, (u^2)_x.

True coefficients: -1.0, -0.5.

| Noise | Oracle EIV mean coefficient ratios | OLS mean coefficient ratios | EIV trials dropping a true term | OLS feasible | EIV objective below truth |
| ---: | :--- | :--- | ---: | ---: | ---: |
| 0 | 1, 1 | 1, 1 | 0/100 | 100% | 0/100 |
| 0.1 | 0.9655, 0.9967 | 0.9993, 0.9999 | 0/100 | 52% | 100/100 |
| 0.2 | 0.9332, 0.9938 | 0.9992, 1 | 0/100 | 58% | 100/100 |
| 0.5 | 0.844, 0.9868 | 0.9952, 1 | 0/100 | 51% | 100/100 |
| 0.75 | 0.7696, 0.9765 | 1.001, 0.9997 | 0/100 | 56% | 100/100 |
| 1 | 0.6821, 0.969 | 0.9902, 0.9992 | 1/100 | 45% | 100/100 |
| 2 | 0.3174, 0.9266 | 0.9147, 0.9915 | 21/100 | 61% | 100/100 |
### KS

Term order: u_xx, u_xxxx, (u^2)_x.

True coefficients: -1.0, -1.0, -0.5.

| Noise | Oracle EIV mean coefficient ratios | OLS mean coefficient ratios | EIV trials dropping a true term | OLS feasible | EIV objective below truth |
| ---: | :--- | :--- | ---: | ---: | ---: |
| 0 | 1, 1, 1 | 1, 1, 1 | 0/100 | 100% | 0/100 |
| 0.1 | 0.9604, 0.9428, 0.9842 | 0.997, 0.9968, 0.9973 | 0/100 | 65% | 100/100 |
| 0.2 | 0.9249, 0.899, 0.965 | 0.9901, 0.99, 0.9899 | 0/100 | 64% | 100/100 |
| 0.5 | 0.762, 0.6952, 0.8844 | 0.934, 0.9314, 0.9372 | 0/100 | 67% | 100/100 |
| 0.75 | 0.5707, 0.4873, 0.7465 | 0.8448, 0.8409, 0.8487 | 2/100 | 79% | 100/100 |
| 1 | 0.3332, 0.2238, 0.5724 | 0.7301, 0.7225, 0.7358 | 15/100 | 90% | 100/100 |
| 2 | 6.15e-15, 0.001639, -1.863e-14 | 0.2549, 0.2382, 0.2632 | 100/100 | 100% | 100/100 |

## Audit

- 2100 paired trials; 27300 conic solves; 3590 failed path solves.
- Maximum relative tau reconstruction discrepancy: 2.84e-14.
- Maximum relative full-fit residual reconstruction discrepancy: 6.41e-15.
- Every reconstructed true residual matches saved tau, and every saved full-fit E2 was independently recomputed.

| Task | Noise | Failed fixed-lambda solves | Failed unchecked oracle choices | Failed path solves |
| :--- | ---: | ---: | ---: | ---: |
| IB | 0 | 0/100 | 0/100 | 400/1300 |
| IB | 0.1 | 6/100 | 39/100 | 447/1300 |
| IB | 0.2 | 5/100 | 40/100 | 394/1300 |
| IB | 0.5 | 7/100 | 34/100 | 327/1300 |
| IB | 0.75 | 7/100 | 25/100 | 295/1300 |
| IB | 1 | 6/100 | 19/100 | 300/1300 |
| IB | 2 | 6/100 | 11/100 | 221/1300 |
| KdV | 0 | 100/100 | 100/100 | 1200/1300 |
| KdV | 0.1 | 0/100 | 0/100 | 2/1300 |
| KdV | 0.2 | 0/100 | 0/100 | 2/1300 |
| KdV | 0.5 | 0/100 | 0/100 | 0/1300 |
| KdV | 0.75 | 0/100 | 0/100 | 0/1300 |
| KdV | 1 | 0/100 | 0/100 | 1/1300 |
| KdV | 2 | 0/100 | 0/100 | 0/1300 |
| KS | 0 | 0/100 | 0/100 | 0/1300 |
| KS | 0.1 | 0/100 | 0/100 | 1/1300 |
| KS | 0.2 | 0/100 | 0/100 | 0/1300 |
| KS | 0.5 | 0/100 | 0/100 | 0/1300 |
| KS | 0.75 | 0/100 | 0/100 | 0/1300 |
| KS | 1 | 0/100 | 0/100 | 0/1300 |
| KS | 2 | 0/100 | 0/100 | 0/1300 |

## Reproduce

```sh
python benchmark_results/eiv/true_support/run_study.py --workers 4
```

Completed trials are skipped. Use --report-only to regenerate this report. Raw paired coefficients and every lambda candidate are in trials.jsonl; aggregate values are in summary.json.
