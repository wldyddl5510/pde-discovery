# Oracle discrete Dantzig: small-seed sanity check

Completed 104/104 paired realizations. This is a preliminary coefficient-recovery experiment, not a tuning rule or a statistical power study.

## Protocol

- Same local trajectories, dictionaries, weak test functions, observation noise definitions and deterministic paired seeds as the repository.
- PDEs: IB, KdV, KS, NLS, RD, NS, HKS, VBG; noise ratios: [0.0, 0.2, 0.5, 0.75, 1.0]; 3 independent observation-noise seeds per nonzero noise level. Noise-free data are evaluated once.
- A 50:50 split pilot is estimated from training time slices only. Both Dantzig methods use the same pilot, regression time slices and physical-unit quadrature.
- Debiased score: H=(X'X+dX'X+X'dX)/K, b=(X'Y+dX'Y+X'dY)/K. The product dX'dX and dX'dY terms are excluded.
- Pilot-score control: H=X'X/K, b=X'Y/K on the same pilot and split. The contrast isolates score correction within this setup.
- Each Dantzig fit minimizes nonzero count subject to the physical score infinity-residual bound. No coefficient thresholding or OLS refit is used to improve its returned estimate.
- Truth enters only oracle tolerance selection and evaluation, never support selection, tie-breaking, pilot estimation or coefficient fitting.
- Coupled PDEs use a separate oracle tolerance for each equation. This is a particularly favorable oracle comparison; baselines use the repository's native common-threshold MSTLS selection.
- Initial tolerance ratios are 13 logarithmically spaced values from 1e-06 to 1, lambda=ratio*||b||inf, with 1 local refinement round(s). See each saved path for the actual grid and any lower-bound extension.
- Per mixed-integer primary solve time limit: 5 s; secondary phase gets max(.2,min(2,T/2)) seconds; relative MIP gap target: 0. The main oracle requires finite constraint-validated candidates with both cardinality and secondary optimality certified.
- The main lambda oracle chooses the smallest physical coefficient E2 among certified candidates. A separate incumbent diagnostic also admits validated time-limited candidates. Neither is guaranteed to find the globally best tolerance; the grid is finite and solver limits can leave parts unresolved.
- The l0 objective may admit multiple supports or coefficients. The solver uses a truth-independent secondary score-residual criterion; its exact tie-break and certification metadata are saved in every path.
- Solver optimality certifies the cardinality problem and secondary score-residual criterion at a fixed tolerance, not recovery of the true PDE. In particular main E2=1 can arise because only the analytic zero model was certified; this alone does not establish estimator failure.
- E2 is the relative Frobenius norm over all physical coefficients. Exact support requires no false positives or false negatives across every equation. FP/FN are counts across all equations.
- Runtime columns exclude shared pilot construction; Dantzig runtime also excludes score construction. The raw per-trial records store these costs separately.

## Mean coefficient E2

| PDE | Noise | Seeds | WSINDy/MSTLS | Split debiased MSTLS | Pilot-score oracle | Debiased-score oracle |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| IB | 0 | 1 | 4.3085e-05 | 0.68637 | 1 | 1 |
| IB | 0.2 | 3 | 0.0031013 | 9.7436 | 1 | 1 |
| IB | 0.5 | 3 | 0.006025 | 10.434 | 1 | 0.70795 |
| IB | 0.75 | 3 | 0.026762 | 0.077269 | 0.74652 | 1 |
| IB | 1 | 3 | 0.013778 | 0.052959 | 0.72681 | 0.71723 |
| KdV | 0 | 1 | 2.8432e-07 | 0.0012064 | 0.1411 | 0.020479 |
| KdV | 0.2 | 3 | 0.0032684 | 0.94106 | 1 | 1 |
| KdV | 0.5 | 3 | 0.60368 | 0.63248 | 1 | 1 |
| KdV | 0.75 | 3 | 0.036674 | 0.75973 | 1 | 1 |
| KdV | 1 | 3 | 0.036246 | 94.893 | 1 | 1 |
| KS | 0 | 1 | 5.5463e-07 | 0.0017033 | 0.89336 | 0.02162 |
| KS | 0.2 | 3 | 0.005046 | 2.3262 | 0.9369 | 1 |
| KS | 0.5 | 3 | 0.07331 | 1.2655 | 1 | 1 |
| KS | 0.75 | 3 | 0.11747 | 1.5541 | 0.96863 | 0.99276 |
| KS | 1 | 3 | 0.26986 | 2.096 | 1 | 1 |
| NLS | 0 | 1 | 5.6049e-08 | 0.00013368 | 1 | 1 |
| NLS | 0.2 | 3 | 0.021164 | 0.017581 | 1 | 1 |
| NLS | 0.5 | 3 | 0.11111 | 1.0903 | 1 | 1 |
| NLS | 0.75 | 3 | 0.40039 | 1.9988 | 1 | 1 |
| NLS | 1 | 3 | 0.86042 | 2.1775 | 1 | 1 |
| RD | 0 | 1 | 1.791e-10 | 0.00025116 | 1 | 1 |
| RD | 0.2 | 3 | 0.02754 | 0.02422 | 1 | 1 |
| RD | 0.5 | 3 | 0.091999 | 0.11256 | 1 | 1 |
| RD | 0.75 | 3 | 0.23303 | 0.31654 | 1 | 1 |
| RD | 1 | 3 | 0.47221 | 0.68492 | 0.94569 | 1 |
| NS | 0 | 1 | 0.0007895 | 0.00059926 | 0.0034849 | 0.0024922 |
| NS | 0.2 | 3 | 0.0016068 | 0.001396 | 0.0057327 | 0.39901 |
| NS | 0.5 | 3 | 0.069473 | 0.085035 | 0.28192 | 1 |
| NS | 0.75 | 3 | 0.059091 | 0.10639 | 0.774 | 1 |
| NS | 1 | 3 | 0.04592 | 0.11413 | 0.77557 | 1 |
| HKS | 0 | 1 | 5.11e-07 | 0.0041954 | 1 | 1 |
| HKS | 0.2 | 3 | 0.068871 | 8.816 | 1 | 1 |
| HKS | 0.5 | 3 | 0.1007 | 12.722 | 1 | 1 |
| HKS | 0.75 | 3 | 0.22762 | 4.1029 | 1 | 1 |
| HKS | 1 | 3 | 0.48469 | 18.394 | 1 | 1 |
| VBG | 0 | 1 | 1.1719e-12 | 1.4906e-05 | 1 | 1 |
| VBG | 0.2 | 3 | 0.03436 | 0.072217 | 1 | 0.99327 |
| VBG | 0.5 | 3 | 0.38767 | 0.516 | 1 | 1 |
| VBG | 0.75 | 3 | 0.66666 | 1.2862 | 1 | 1 |
| VBG | 1 | 3 | 0.83437 | 0.95586 | 1 | 0.99336 |

## Support recovery

Each cell shows exact-support rate, then mean (FP, FN).

| PDE | Noise | WSINDy/MSTLS | Split debiased MSTLS | Pilot-score oracle | Debiased-score oracle |
| :--- | ---: | ---: | ---: | ---: | ---: |
| IB | 0 | 100%; (0, 0) | 0%; (4, 0) | 0%; (0, 1) | 0%; (0, 1) |
| IB | 0.2 | 100%; (0, 0) | 0%; (4, 0) | 0%; (0, 1) | 0%; (0, 1) |
| IB | 0.5 | 100%; (0, 0) | 33%; (2.3, 0) | 0%; (0, 1) | 33%; (0, 0.67) |
| IB | 0.75 | 100%; (0, 0) | 33%; (1, 0) | 33%; (0, 0.67) | 0%; (0.67, 1) |
| IB | 1 | 100%; (0, 0) | 100%; (0, 0) | 33%; (0, 0.67) | 33%; (0, 0.67) |
| KdV | 0 | 100%; (0, 0) | 100%; (0, 0) | 100%; (0, 0) | 100%; (0, 0) |
| KdV | 0.2 | 100%; (0, 0) | 0%; (0, 1) | 0%; (0, 2) | 0%; (0, 2) |
| KdV | 0.5 | 33%; (0, 0.67) | 67%; (0, 0.33) | 0%; (0.33, 2) | 0%; (0.33, 2) |
| KdV | 0.75 | 100%; (0, 0) | 33%; (0, 0.67) | 0%; (0, 2) | 0%; (0.67, 2) |
| KdV | 1 | 100%; (0, 0) | 33%; (1, 0.33) | 0%; (0.33, 2) | 0%; (0, 2) |
| KS | 0 | 100%; (0, 0) | 100%; (0, 0) | 0%; (1, 2) | 100%; (0, 0) |
| KS | 0.2 | 100%; (0, 0) | 0%; (11, 1) | 0%; (1, 2) | 0%; (0, 3) |
| KS | 0.5 | 100%; (0, 0) | 0%; (1.7, 3) | 0%; (0, 3) | 0%; (0, 3) |
| KS | 0.75 | 100%; (0, 0) | 0%; (3.3, 2.7) | 0%; (0.33, 2.7) | 0%; (0.33, 2.7) |
| KS | 1 | 100%; (0, 0) | 0%; (4.3, 2.7) | 0%; (0, 3) | 0%; (0, 3) |
| NLS | 0 | 100%; (0, 0) | 100%; (0, 0) | 0%; (0, 6) | 0%; (0, 6) |
| NLS | 0.2 | 100%; (0, 0) | 100%; (0, 0) | 0%; (1.7, 6) | 0%; (1, 6) |
| NLS | 0.5 | 100%; (0, 0) | 0%; (10, 0.33) | 0%; (0, 6) | 0%; (0, 6) |
| NLS | 0.75 | 0%; (4.3, 0) | 0%; (20, 0) | 0%; (0, 6) | 0%; (0, 6) |
| NLS | 1 | 0%; (8, 0.33) | 0%; (19, 0.33) | 0%; (0, 6) | 0%; (0, 6) |
| RD | 0 | 100%; (0, 0) | 100%; (0, 0) | 0%; (0, 14) | 0%; (0, 14) |
| RD | 0.2 | 100%; (0, 0) | 100%; (0, 0) | 0%; (0, 14) | 0%; (0, 14) |
| RD | 0.5 | 0%; (2, 0) | 100%; (0, 0) | 0%; (0, 14) | 0%; (0, 14) |
| RD | 0.75 | 0%; (3.7, 0) | 0%; (1.3, 0) | 0%; (0, 14) | 0%; (0, 14) |
| RD | 1 | 0%; (7.3, 0) | 0%; (10, 1) | 0%; (58, 12) | 0%; (0, 14) |
| NS | 0 | 100%; (0, 0) | 100%; (0, 0) | 100%; (0, 0) | 100%; (0, 0) |
| NS | 0.2 | 100%; (0, 0) | 100%; (0, 0) | 100%; (0, 0) | 0%; (14, 1.3) |
| NS | 0.5 | 0%; (1, 2) | 0%; (1.7, 2) | 0%; (42, 0) | 0%; (0, 4) |
| NS | 0.75 | 0%; (0.67, 2) | 0%; (1.7, 2) | 0%; (0, 3) | 0%; (0, 4) |
| NS | 1 | 0%; (0, 2) | 0%; (1.7, 2) | 0%; (0, 3) | 0%; (0, 4) |
| HKS | 0 | 100%; (0, 0) | 100%; (0, 0) | 0%; (0, 4) | 0%; (0, 4) |
| HKS | 0.2 | 33%; (0.33, 0.67) | 0%; (11, 2) | 0%; (0, 4) | 0%; (0, 4) |
| HKS | 0.5 | 0%; (0, 1) | 0%; (20, 2) | 0%; (0, 4) | 0%; (0, 4) |
| HKS | 0.75 | 0%; (2, 1) | 0%; (3, 2.7) | 0%; (0, 4) | 0%; (0, 4) |
| HKS | 1 | 0%; (0, 1.3) | 0%; (25, 2.3) | 0%; (0, 4) | 0%; (0, 4) |
| VBG | 0 | 100%; (0, 0) | 100%; (0, 0) | 0%; (0, 5) | 0%; (0, 5) |
| VBG | 0.2 | 0%; (0, 1) | 0%; (0, 1) | 0%; (0, 5) | 0%; (0.33, 4.7) |
| VBG | 0.5 | 0%; (1.3, 1.3) | 0%; (1, 1.3) | 0%; (0, 5) | 0%; (0, 5) |
| VBG | 0.75 | 0%; (1.3, 2) | 0%; (2.7, 2) | 0%; (0, 5) | 0%; (0, 5) |
| VBG | 1 | 0%; (1.7, 2) | 0%; (3, 1.3) | 0%; (0, 5) | 0%; (0.33, 4.7) |

## Solver audit

Counts are equation fits. Candidate certification refers to validated candidates with proven minimal cardinality; time-limited incumbents are eligible only for the separately reported incumbent diagnostic.

| PDE | Noise | Score | Valid candidates / total | Certified / valid | Certified selections / equations | Lower-bound selections | Upper-bound selections |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| IB | 0 | plugin | 14/14 | 14/14 | 1/1 | 0 | 1 |
| IB | 0 | debiased | 14/14 | 14/14 | 1/1 | 0 | 1 |
| IB | 0.2 | plugin | 42/42 | 42/42 | 3/3 | 0 | 3 |
| IB | 0.2 | debiased | 42/42 | 42/42 | 3/3 | 0 | 3 |
| IB | 0.5 | plugin | 42/42 | 42/42 | 3/3 | 0 | 3 |
| IB | 0.5 | debiased | 43/43 | 43/43 | 3/3 | 0 | 2 |
| IB | 0.75 | plugin | 43/43 | 43/43 | 3/3 | 0 | 2 |
| IB | 0.75 | debiased | 44/44 | 43/44 | 3/3 | 0 | 2 |
| IB | 1 | plugin | 43/43 | 43/43 | 3/3 | 0 | 2 |
| IB | 1 | debiased | 43/43 | 43/43 | 3/3 | 0 | 2 |
| KdV | 0 | plugin | 15/16 | 9/15 | 1/1 | 0 | 0 |
| KdV | 0 | debiased | 15/16 | 10/15 | 1/1 | 0 | 0 |
| KdV | 0.2 | plugin | 42/42 | 21/42 | 3/3 | 0 | 3 |
| KdV | 0.2 | debiased | 44/45 | 15/44 | 3/3 | 0 | 3 |
| KdV | 0.5 | plugin | 43/43 | 22/43 | 3/3 | 0 | 2 |
| KdV | 0.5 | debiased | 43/43 | 19/43 | 3/3 | 0 | 2 |
| KdV | 0.75 | plugin | 42/42 | 20/42 | 3/3 | 0 | 3 |
| KdV | 0.75 | debiased | 43/43 | 17/43 | 3/3 | 0 | 2 |
| KdV | 1 | plugin | 43/43 | 22/43 | 3/3 | 0 | 2 |
| KdV | 1 | debiased | 44/44 | 16/44 | 3/3 | 0 | 3 |
| KS | 0 | plugin | 15/15 | 3/15 | 1/1 | 0 | 0 |
| KS | 0 | debiased | 15/15 | 7/15 | 1/1 | 0 | 0 |
| KS | 0.2 | plugin | 45/45 | 21/45 | 3/3 | 0 | 0 |
| KS | 0.2 | debiased | 43/43 | 30/43 | 3/3 | 0 | 3 |
| KS | 0.5 | plugin | 44/44 | 20/44 | 3/3 | 0 | 3 |
| KS | 0.5 | debiased | 42/42 | 28/42 | 3/3 | 0 | 3 |
| KS | 0.75 | plugin | 43/43 | 26/43 | 3/3 | 0 | 2 |
| KS | 0.75 | debiased | 43/43 | 28/43 | 3/3 | 0 | 2 |
| KS | 1 | plugin | 45/45 | 20/45 | 3/3 | 0 | 3 |
| KS | 1 | debiased | 42/42 | 26/42 | 3/3 | 0 | 3 |
| NLS | 0 | plugin | 30/31 | 4/30 | 2/2 | 0 | 2 |
| NLS | 0 | debiased | 30/31 | 4/30 | 2/2 | 0 | 2 |
| NLS | 0.2 | plugin | 90/90 | 12/90 | 6/6 | 0 | 1 |
| NLS | 0.2 | debiased | 90/90 | 15/90 | 6/6 | 0 | 3 |
| NLS | 0.5 | plugin | 88/88 | 8/88 | 6/6 | 0 | 6 |
| NLS | 0.5 | debiased | 86/86 | 10/86 | 6/6 | 0 | 6 |
| NLS | 0.75 | plugin | 86/87 | 11/86 | 6/6 | 0 | 6 |
| NLS | 0.75 | debiased | 86/86 | 10/86 | 6/6 | 0 | 6 |
| NLS | 1 | plugin | 87/87 | 10/87 | 6/6 | 0 | 6 |
| NLS | 1 | debiased | 88/88 | 11/88 | 6/6 | 0 | 6 |
| RD | 0 | plugin | 30/30 | 2/30 | 2/2 | 0 | 2 |
| RD | 0 | debiased | 30/30 | 2/30 | 2/2 | 0 | 2 |
| RD | 0.2 | plugin | 90/90 | 6/90 | 6/6 | 0 | 6 |
| RD | 0.2 | debiased | 90/90 | 6/90 | 6/6 | 0 | 6 |
| RD | 0.5 | plugin | 90/90 | 6/90 | 6/6 | 0 | 6 |
| RD | 0.5 | debiased | 89/89 | 7/89 | 6/6 | 0 | 6 |
| RD | 0.75 | plugin | 90/90 | 6/90 | 6/6 | 0 | 6 |
| RD | 0.75 | debiased | 90/90 | 6/90 | 6/6 | 0 | 6 |
| RD | 1 | plugin | 91/91 | 8/91 | 6/6 | 0 | 5 |
| RD | 1 | debiased | 89/89 | 6/89 | 6/6 | 0 | 6 |
| NS | 0 | plugin | 15/15 | 7/15 | 1/1 | 0 | 0 |
| NS | 0 | debiased | 15/15 | 7/15 | 1/1 | 0 | 0 |
| NS | 0.2 | plugin | 45/45 | 19/45 | 3/3 | 0 | 0 |
| NS | 0.2 | debiased | 45/45 | 12/45 | 3/3 | 1 | 1 |
| NS | 0.5 | plugin | 44/45 | 13/44 | 3/3 | 2 | 0 |
| NS | 0.5 | debiased | 45/45 | 18/45 | 3/3 | 0 | 3 |
| NS | 0.75 | plugin | 45/45 | 14/45 | 3/3 | 0 | 0 |
| NS | 0.75 | debiased | 45/45 | 20/45 | 3/3 | 0 | 3 |
| NS | 1 | plugin | 45/45 | 15/45 | 3/3 | 0 | 0 |
| NS | 1 | debiased | 44/45 | 19/44 | 3/3 | 0 | 3 |
| HKS | 0 | plugin | 15/15 | 2/15 | 1/1 | 0 | 1 |
| HKS | 0 | debiased | 15/15 | 2/15 | 1/1 | 0 | 1 |
| HKS | 0.2 | plugin | 44/44 | 8/44 | 3/3 | 0 | 3 |
| HKS | 0.2 | debiased | 42/42 | 18/42 | 3/3 | 0 | 3 |
| HKS | 0.5 | plugin | 44/44 | 8/44 | 3/3 | 0 | 3 |
| HKS | 0.5 | debiased | 42/42 | 16/42 | 3/3 | 0 | 3 |
| HKS | 0.75 | plugin | 42/42 | 10/42 | 3/3 | 0 | 3 |
| HKS | 0.75 | debiased | 43/43 | 16/43 | 3/3 | 0 | 3 |
| HKS | 1 | plugin | 42/42 | 9/42 | 3/3 | 0 | 3 |
| HKS | 1 | debiased | 43/43 | 17/43 | 3/3 | 0 | 3 |
| VBG | 0 | plugin | 15/15 | 8/15 | 1/1 | 0 | 1 |
| VBG | 0 | debiased | 15/16 | 8/15 | 1/1 | 0 | 1 |
| VBG | 0.2 | plugin | 45/46 | 23/45 | 3/3 | 0 | 3 |
| VBG | 0.2 | debiased | 44/46 | 23/44 | 3/3 | 0 | 2 |
| VBG | 0.5 | plugin | 45/46 | 22/45 | 3/3 | 0 | 3 |
| VBG | 0.5 | debiased | 42/42 | 23/42 | 3/3 | 0 | 3 |
| VBG | 0.75 | plugin | 44/46 | 20/44 | 3/3 | 0 | 3 |
| VBG | 0.75 | debiased | 45/47 | 15/45 | 3/3 | 0 | 3 |
| VBG | 1 | plugin | 45/47 | 17/45 | 3/3 | 0 | 3 |
| VBG | 1 | debiased | 45/46 | 15/45 | 3/3 | 0 | 2 |

## Time-limited incumbent diagnostic

This additional oracle admits every constraint-validated candidate, including unresolved cardinality or tie-break solves. It can look better than the exact estimator by retaining more terms than the minimum-cardinality solution. Use it to diagnose optimization limitations, not as the estimator's main performance.

| PDE | Noise | Pilot certified E2 | Pilot incumbent E2 | Debiased certified E2 | Debiased incumbent E2 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| IB | 0 | 1 | 1 | 1 | 1 |
| IB | 0.2 | 1 | 1 | 1 | 1 |
| IB | 0.5 | 1 | 1 | 0.70795 | 0.70795 |
| IB | 0.75 | 0.74652 | 0.74652 | 1 | 1 |
| IB | 1 | 0.72681 | 0.72681 | 0.71723 | 0.71723 |
| KdV | 0 | 0.1411 | 0.016216 | 0.020479 | 0.001831 |
| KdV | 0.2 | 1 | 1 | 1 | 0.99134 |
| KdV | 0.5 | 1 | 1 | 1 | 1 |
| KdV | 0.75 | 1 | 1 | 1 | 0.99104 |
| KdV | 1 | 1 | 1 | 1 | 0.98384 |
| KS | 0 | 0.89336 | 0.065604 | 0.02162 | 0.02162 |
| KS | 0.2 | 0.9369 | 0.77467 | 1 | 0.96648 |
| KS | 0.5 | 1 | 0.98275 | 1 | 1 |
| KS | 0.75 | 0.96863 | 0.96863 | 0.99276 | 0.99276 |
| KS | 1 | 1 | 0.98674 | 1 | 1 |
| NLS | 0 | 1 | 0.0077507 | 1 | 0.011018 |
| NLS | 0.2 | 1 | 0.32599 | 1 | 0.87883 |
| NLS | 0.5 | 1 | 0.89261 | 1 | 0.94376 |
| NLS | 0.75 | 1 | 0.94085 | 1 | 0.93279 |
| NLS | 1 | 1 | 0.90523 | 1 | 0.89212 |
| RD | 0 | 1 | 0.015247 | 1 | 0.0011878 |
| RD | 0.2 | 1 | 0.069538 | 1 | 0.38489 |
| RD | 0.5 | 1 | 0.1509 | 1 | 0.55941 |
| RD | 0.75 | 1 | 0.16223 | 1 | 0.48511 |
| RD | 1 | 0.94569 | 0.31307 | 1 | 0.66561 |
| NS | 0 | 0.0034849 | 0.0034849 | 0.0024922 | 0.0019082 |
| NS | 0.2 | 0.0057327 | 0.0057327 | 0.39901 | 0.0028899 |
| NS | 0.5 | 0.28192 | 0.020033 | 1 | 0.037398 |
| NS | 0.75 | 0.774 | 0.045555 | 1 | 0.053292 |
| NS | 1 | 0.77557 | 0.051299 | 1 | 0.043878 |
| HKS | 0 | 1 | 0.26307 | 1 | 0.15267 |
| HKS | 0.2 | 1 | 0.97695 | 1 | 1 |
| HKS | 0.5 | 1 | 0.96539 | 1 | 1 |
| HKS | 0.75 | 1 | 1 | 1 | 0.98673 |
| HKS | 1 | 1 | 1 | 1 | 0.98251 |
| VBG | 0 | 1 | 0.082984 | 1 | 0.0052332 |
| VBG | 0.2 | 1 | 0.70114 | 0.99327 | 0.91774 |
| VBG | 0.5 | 1 | 0.90051 | 1 | 1 |
| VBG | 0.75 | 1 | 0.4887 | 1 | 0.92546 |
| VBG | 1 | 1 | 0.52709 | 0.99336 | 0.79242 |

## Paired descriptive comparison

Among 96 nonzero-noise realizations, debiased-score oracle E2 is strictly smaller than pilot-score oracle E2 in 5, and smaller than split debiased MSTLS in 43. Ties count in neither direction. This count weights PDE/noise settings equally per seed and is not a significance test.

## Reproduce

```sh
python benchmark_results/dantzig/sanity/run_study.py --workers 2

# Regenerate figures and tables from saved trials only:
python benchmark_results/dantzig/sanity/run_study.py --report-only
```

The manifest stores source/data SHA256 hashes and package versions. trials.jsonl stores every oracle candidate, coefficients, physical-score truth residuals, solver statuses and seeds. summary.csv/json contain grouped statistics; results.png plots mean E2. Results are exploratory: three noise draws on a single trajectory per PDE cannot establish robustness.
