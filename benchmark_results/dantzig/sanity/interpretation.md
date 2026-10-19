# Interpretation checks for the oracle sanity study

104 completed realizations, including 96 with nonzero observation noise.

The analytic zero model is always on the candidate path and has relative coefficient E2=1. Consequently every main oracle result has E2<=1 by construction. A win against a baseline whose E2 exceeds1 can consist entirely of selecting the zero model; the table distinguishes these from nonzero model wins.

Three oracle levels are kept separate: (a) main lexicographic certification requires both minimum cardinality and the secondary residual criterion proved optimal; (b) cardinality certification requires a validated minimum-cardinality solution but permits an unresolved secondary criterion; (c) the incumbent diagnostic requires feasibility only. Level(b) is still a certified minimizer of Equation(9). Its returned coefficients use the same truth-independent secondary solver, but the residual tie-break may not have reached its optimum. Truth only selects the tolerance.

Certificate coverage measures how much of the finite tolerance path was actually resolved. Poor coefficient recovery with low coverage cannot by itself establish failure of the mathematical estimator.

## Native baselines, nonzero-noise realizations

| Method | N | Mean E2 | Median E2 | Exact support | All-zero selections |
| :--- | ---: | ---: | ---: | ---: | ---: |
| wsindy | 96 | 0.19982 | 0.070848 | 47 | 0 |
| debiased_mstls | 96 | 5.5741 | 0.94042 | 21 | 0 |

## Dantzig oracle levels, nonzero-noise realizations

| Oracle | Score | N | Mean E2 | Median E2 | All-zero selections | Exact support | Wins vs native WSINDy | Wins using zero | Wins using nonzero |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lexicographic | plugin_dantzig | 96 | 0.91131 | 1 | 72 | 5 | 3 | 3 | 0 |
| lexicographic | debiased_dantzig | 96 | 0.96261 | 1 | 83 | 2 | 3 | 2 | 1 |
| cardinality | plugin_dantzig | 96 | 0.84972 | 1 | 56 | 5 | 9 | 3 | 6 |
| cardinality | debiased_dantzig | 96 | 0.85709 | 1 | 65 | 2 | 10 | 2 | 8 |
| incumbent | plugin_dantzig | 96 | 0.6759 | 0.94675 | 34 | 5 | 17 | 0 | 17 |
| incumbent | debiased_dantzig | 96 | 0.77611 | 0.97133 | 39 | 2 | 10 | 0 | 10 |

## Per-setting mean E2 by oracle level

Each error cell lists lexicographic / cardinality / incumbent. Support columns use the cardinality-certified oracle.

| PDE | Noise | Pilot E2 (three levels) | Debiased E2 (three levels) | Pilot exact / N; mean(FP,FN) | Debiased exact / N; mean(FP,FN) |
| :--- | ---: | :--- | :--- | :--- | :--- |
| IB | 0 | 1 / 1 / 1 | 1 / 1 / 1 | 0/1; (0,1) | 0/1; (0,1) |
| IB | 0.2 | 1 / 1 / 1 | 1 / 1 / 1 | 0/3; (0,1) | 0/3; (0,1) |
| IB | 0.5 | 1 / 1 / 1 | 0.70795 / 0.70795 / 0.70795 | 0/3; (0,1) | 1/3; (0,0.667) |
| IB | 0.75 | 0.74652 / 0.74652 / 0.74652 | 1 / 1 / 1 | 1/3; (0,0.667) | 0/3; (0.667,1) |
| IB | 1 | 0.72681 / 0.72681 / 0.72681 | 0.71723 / 0.71723 / 0.71723 | 1/3; (0,0.667) | 1/3; (0,0.667) |
| KdV | 0 | 0.1411 / 0.016216 / 0.016216 | 0.020479 / 0.001831 / 0.001831 | 0/1; (1,0) | 0/1; (1,0) |
| KdV | 0.2 | 1 / 1 / 1 | 1 / 0.99669 / 0.99134 | 0/3; (0,2) | 0/3; (2,1.67) |
| KdV | 0.5 | 1 / 1 / 1 | 1 / 1 / 1 | 0/3; (0.333,2) | 0/3; (0.333,2) |
| KdV | 0.75 | 1 / 1 / 1 | 1 / 1 / 0.99104 | 0/3; (0,2) | 0/3; (0.667,2) |
| KdV | 1 | 1 / 1 / 1 | 1 / 1 / 0.98384 | 0/3; (0.333,2) | 0/3; (0,2) |
| KS | 0 | 0.89336 / 0.69216 / 0.065604 | 0.02162 / 0.02162 / 0.02162 | 0/1; (1,1) | 1/1; (0,0) |
| KS | 0.2 | 0.9369 / 0.77467 / 0.77467 | 1 / 1 / 0.96648 | 0/3; (1,1.67) | 0/3; (0,3) |
| KS | 0.5 | 1 / 1 / 0.98275 | 1 / 1 / 1 | 0/3; (0,3) | 0/3; (0,3) |
| KS | 0.75 | 0.96863 / 0.96863 / 0.96863 | 0.99276 / 0.99276 / 0.99276 | 0/3; (0.333,2.67) | 0/3; (0.333,2.67) |
| KS | 1 | 1 / 0.99781 / 0.98674 | 1 / 1 / 1 | 0/3; (1,2.67) | 0/3; (0,3) |
| NLS | 0 | 1 / 1 / 0.0077507 | 1 / 1 / 0.011018 | 0/1; (0,6) | 0/1; (0,6) |
| NLS | 0.2 | 1 / 1 / 0.32599 | 1 / 1 / 0.87883 | 0/3; (1.67,6) | 0/3; (1,6) |
| NLS | 0.5 | 1 / 0.97304 / 0.89261 | 1 / 1 / 0.94376 | 0/3; (0.667,5.33) | 0/3; (0,6) |
| NLS | 0.75 | 1 / 0.97783 / 0.94085 | 1 / 0.96749 / 0.93279 | 0/3; (0.667,5.33) | 0/3; (0.333,5.67) |
| NLS | 1 | 1 / 0.98877 / 0.90523 | 1 / 0.97397 / 0.89212 | 0/3; (0.333,5.67) | 0/3; (0.333,5.67) |
| RD | 0 | 1 / 1 / 0.015247 | 1 / 1 / 0.0011878 | 0/1; (0,14) | 0/1; (0,14) |
| RD | 0.2 | 1 / 1 / 0.069538 | 1 / 1 / 0.38489 | 0/3; (0,14) | 0/3; (0,14) |
| RD | 0.5 | 1 / 1 / 0.1509 | 1 / 1 / 0.55941 | 0/3; (0,14) | 0/3; (0,14) |
| RD | 0.75 | 1 / 0.94406 / 0.16223 | 1 / 1 / 0.48511 | 0/3; (53.7,11.7) | 0/3; (0,14) |
| RD | 1 | 0.94569 / 0.8383 / 0.31307 | 1 / 1 / 0.66561 | 0/3; (168,7) | 0/3; (0,14) |
| NS | 0 | 0.0034849 / 0.0034849 / 0.0034849 | 0.0024922 / 0.0019082 / 0.0019082 | 1/1; (0,0) | 0/1; (4,0) |
| NS | 0.2 | 0.0057327 / 0.0057327 / 0.0057327 | 0.39901 / 0.015151 / 0.0028899 | 3/3; (0,0) | 0/3; (1,0.333) |
| NS | 0.5 | 0.28192 / 0.020033 / 0.020033 | 1 / 0.051953 / 0.037398 | 0/3; (0,1) | 0/3; (1,1) |
| NS | 0.75 | 0.774 / 0.045555 / 0.045555 | 1 / 0.053292 / 0.053292 | 0/3; (1,1) | 0/3; (1,1) |
| NS | 1 | 0.77557 / 0.28727 / 0.051299 | 1 / 0.045279 / 0.043878 | 0/3; (0.333,1.67) | 0/3; (1,1) |
| HKS | 0 | 1 / 0.63793 / 0.26307 | 1 / 0.15267 / 0.15267 | 0/1; (1,2) | 0/1; (0,1) |
| HKS | 0.2 | 1 / 0.97695 / 0.97695 | 1 / 1 / 1 | 0/3; (0.667,3.33) | 0/3; (0,4) |
| HKS | 0.5 | 1 / 0.97859 / 0.96539 | 1 / 1 / 1 | 0/3; (0.667,3.33) | 0/3; (0,4) |
| HKS | 0.75 | 1 / 1 / 1 | 1 / 0.98673 / 0.98673 | 0/3; (0,4) | 0/3; (0.333,3.67) |
| HKS | 1 | 1 / 1 / 1 | 1 / 0.98251 / 0.98251 | 0/3; (0,4) | 0/3; (0.333,3.67) |
| VBG | 0 | 1 / 0.95582 / 0.082984 | 1 / 1 / 0.0052332 | 0/1; (0,1) | 0/1; (0,5) |
| VBG | 0.2 | 1 / 0.98417 / 0.70114 | 0.99327 / 0.96854 / 0.91774 | 0/3; (0.667,4.33) | 0/3; (1,4) |
| VBG | 0.5 | 1 / 0.99456 / 0.90051 | 1 / 1 / 1 | 0/3; (1,4.67) | 0/3; (0,5) |
| VBG | 0.75 | 1 / 1 / 0.4887 | 1 / 1 / 0.92546 | 0/3; (0,5) | 0/3; (0,5) |
| VBG | 1 | 1 / 0.96165 / 0.52709 | 0.99336 / 0.96746 / 0.79242 | 0/3; (1,4.67) | 0/3; (1,4) |

## Finite-path certificate coverage (all realizations)

| Score | Valid / total candidates | Lexicographic certified / total | Cardinality certified / total | Lexicographic nonanalytic / nonanalytic | Main all-zero / realizations |
| :--- | ---: | ---: | ---: | ---: | ---: |
| plugin_dantzig | 1908/1918 | 636/1918 | 860/1918 | 506/1788 | 77/104 |
| debiased_dantzig | 1898/1908 | 687/1908 | 873/1908 | 557/1778 | 88/104 |

The source of these counts is trials.jsonl; interpretation.json contains status histograms and per-PDE details, interpretation.csv contains every PDE/noise/oracle summary, and cardinality_oracle.jsonl stores the additional oracle's selected candidates and coefficients. Paired wins are descriptive counts, not significance tests. The native MSTLS comparison uses its existing tuning rule, while Dantzig receives per-equation oracle tolerance selection.
