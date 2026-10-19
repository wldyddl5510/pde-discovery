# Oracle-pilot Dantzig sanity check

Completed 104/104 paired realizations; 320 unique score paths in the planned full scope.

Two clean-state oracle pilots are evaluated with the same noisy observations, split masks, physical quadrature and dictionaries as the original sanity study. The rate oracle contracts the original pilot error direction until its squared error energy is at most n_reg^(-1/2); the exact oracle is clean u* at the regression nodes. Both use the clean state deliberately and are diagnostic oracles.

n_reg counts regression spacetime nodes, without counting vector components as separate nodes. Energy sums squared vector error with Delta z times SplitSystem time weights; the unweighted Delta z sum is also saved and bounded. Thus energy/n_reg^(-1/4) <= n_reg^(-1/4) tends to zero. This deterministic construction satisfies the probability requirement with probability one; a single grid is not an empirical asymptotic-rate validation.

The same finite lambda grid, truth-independent residual tie-break, time limits and no-refit rule are used. True beta only chooses lambda and evaluates coefficients. Exact plugin scores are identical within each PDE and are solved once then reused with score hashes checked. Original baseline and original-pilot results are copied byte-for-byte numerically from the completed study.

Oracle levels: lexicographic requires certified cardinality and residual tie-break; cardinality requires a certified Equation (9) minimizer; incumbent permits any physically feasible candidate, including unresolved cardinality. The last is only an optimization diagnostic.

## Aggregate coefficient recovery

| Scope | Oracle level | Method | N | Mean E2 | Median E2 | Exact support | All zero |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| all | lexicographic | wsindy | 104 | 0.184454 | 0.0692201 | 55 | 0 |
| all | lexicographic | debiased_mstls | 104 | 5.15196 | 0.672147 | 28 | 0 |
| all | lexicographic | plugin_dantzig | 104 | 0.899262 | 1 | 7 | 77 |
| all | lexicographic | debiased_dantzig | 104 | 0.937071 | 1 | 5 | 88 |
| all | lexicographic | rate_plugin_dantzig | 104 | 0.501055 | 0.503247 | 29 | 52 |
| all | lexicographic | rate_debiased_dantzig | 104 | 0.798028 | 1 | 10 | 61 |
| all | lexicographic | exact_plugin_dantzig | 104 | 0.250564 | 0.000519432 | 52 | 26 |
| all | lexicographic | exact_debiased_dantzig | 104 | 0.773128 | 1 | 11 | 57 |
| all | cardinality | plugin_dantzig | 104 | 0.83537 | 1 | 6 | 59 |
| all | cardinality | debiased_dantzig | 104 | 0.831337 | 1 | 3 | 69 |
| all | cardinality | rate_plugin_dantzig | 104 | 0.492947 | 0.467682 | 25 | 38 |
| all | cardinality | rate_debiased_dantzig | 104 | 0.648448 | 0.935418 | 14 | 42 |
| all | cardinality | exact_plugin_dantzig | 104 | 0.250564 | 0.000519432 | 52 | 26 |
| all | cardinality | exact_debiased_dantzig | 104 | 0.656418 | 0.934324 | 13 | 39 |
| all | incumbent | plugin_dantzig | 104 | 0.637896 | 0.931932 | 7 | 35 |
| all | incumbent | debiased_dantzig | 104 | 0.727903 | 0.954372 | 3 | 40 |
| all | incumbent | rate_plugin_dantzig | 104 | 0.0950205 | 0.00173245 | 31 | 5 |
| all | incumbent | rate_debiased_dantzig | 104 | 0.440604 | 0.521198 | 9 | 5 |
| all | incumbent | exact_plugin_dantzig | 104 | 0.00056702 | 1.71646e-05 | 52 | 0 |
| all | incumbent | exact_debiased_dantzig | 104 | 0.389335 | 0.204434 | 12 | 5 |
| nonzero_noise | lexicographic | wsindy | 96 | 0.199816 | 0.0708482 | 47 | 0 |
| nonzero_noise | lexicographic | debiased_mstls | 96 | 5.57406 | 0.940421 | 21 | 0 |
| nonzero_noise | lexicographic | plugin_dantzig | 96 | 0.911306 | 1 | 5 | 72 |
| nonzero_noise | lexicographic | debiased_dantzig | 96 | 0.962612 | 1 | 2 | 83 |
| nonzero_noise | lexicographic | rate_plugin_dantzig | 96 | 0.501086 | 0.503247 | 27 | 48 |
| nonzero_noise | lexicographic | rate_debiased_dantzig | 96 | 0.833233 | 1 | 7 | 58 |
| nonzero_noise | lexicographic | exact_plugin_dantzig | 96 | 0.250564 | 0.000519432 | 48 | 24 |
| nonzero_noise | lexicographic | exact_debiased_dantzig | 96 | 0.816675 | 1 | 7 | 55 |
| nonzero_noise | cardinality | plugin_dantzig | 96 | 0.849717 | 1 | 5 | 56 |
| nonzero_noise | cardinality | debiased_dantzig | 96 | 0.857094 | 1 | 2 | 65 |
| nonzero_noise | cardinality | rate_plugin_dantzig | 96 | 0.492829 | 0.467682 | 23 | 35 |
| nonzero_noise | cardinality | rate_debiased_dantzig | 96 | 0.681604 | 0.943502 | 10 | 40 |
| nonzero_noise | cardinality | exact_plugin_dantzig | 96 | 0.250564 | 0.000519432 | 48 | 24 |
| nonzero_noise | cardinality | exact_debiased_dantzig | 96 | 0.690239 | 0.941844 | 9 | 37 |
| nonzero_noise | incumbent | plugin_dantzig | 96 | 0.675905 | 0.946754 | 5 | 34 |
| nonzero_noise | incumbent | debiased_dantzig | 96 | 0.776109 | 0.971333 | 2 | 39 |
| nonzero_noise | incumbent | rate_plugin_dantzig | 96 | 0.10191 | 0.00231758 | 28 | 5 |
| nonzero_noise | incumbent | rate_debiased_dantzig | 96 | 0.47712 | 0.554611 | 7 | 5 |
| nonzero_noise | incumbent | exact_plugin_dantzig | 96 | 0.00056702 | 1.71646e-05 | 48 | 0 |
| nonzero_noise | incumbent | exact_debiased_dantzig | 96 | 0.421732 | 0.349919 | 8 | 5 |
| noiseless | lexicographic | wsindy | 8 | 0.000104248 | 3.97659e-07 | 8 | 0 |
| noiseless | lexicographic | debiased_mstls | 8 | 0.0868094 | 0.000902807 | 7 | 0 |
| noiseless | lexicographic | plugin_dantzig | 8 | 0.754742 | 1 | 2 | 5 |
| noiseless | lexicographic | debiased_dantzig | 8 | 0.630574 | 1 | 3 | 5 |
| noiseless | lexicographic | rate_plugin_dantzig | 8 | 0.500681 | 0.50173 | 2 | 4 |
| noiseless | lexicographic | rate_debiased_dantzig | 8 | 0.375572 | 0.00219186 | 3 | 3 |
| noiseless | lexicographic | exact_plugin_dantzig | 8 | 0.250564 | 0.000519432 | 4 | 2 |
| noiseless | lexicographic | exact_debiased_dantzig | 8 | 0.250564 | 0.000519432 | 4 | 2 |
| noiseless | cardinality | plugin_dantzig | 8 | 0.663201 | 0.823987 | 1 | 3 |
| noiseless | cardinality | debiased_dantzig | 8 | 0.522254 | 0.576336 | 1 | 4 |
| noiseless | cardinality | rate_plugin_dantzig | 8 | 0.494365 | 0.476488 | 2 | 3 |
| noiseless | cardinality | rate_debiased_dantzig | 8 | 0.250579 | 0.000547081 | 4 | 2 |
| noiseless | cardinality | exact_plugin_dantzig | 8 | 0.250564 | 0.000519432 | 4 | 2 |
| noiseless | cardinality | exact_debiased_dantzig | 8 | 0.250564 | 0.000519432 | 4 | 2 |
| noiseless | incumbent | plugin_dantzig | 8 | 0.181794 | 0.0409099 | 2 | 1 |
| noiseless | incumbent | debiased_dantzig | 8 | 0.149434 | 0.00812558 | 1 | 1 |
| noiseless | incumbent | rate_plugin_dantzig | 8 | 0.0123445 | 0.00133237 | 3 | 0 |
| noiseless | incumbent | rate_debiased_dantzig | 8 | 0.00240746 | 0.000547081 | 2 | 0 |
| noiseless | incumbent | exact_plugin_dantzig | 8 | 0.00056702 | 1.71646e-05 | 4 | 0 |
| noiseless | incumbent | exact_debiased_dantzig | 8 | 0.000565778 | 1.21943e-05 | 4 | 0 |

## Per-setting cardinality-certified mean E2

| PDE | Noise | Native MSTLS | Original debiased Dantzig | Rate plugin | Rate debiased | Exact plugin | Exact debiased |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IB | 0 | 4.30846e-05 | 1 | 0.00346062 | 0.00346158 | 0.00346158 | 0.00346158 |
| IB | 0.2 | 0.0031013 | 1 | 0.0034606 | 0.669438 | 0.00346158 | 0.669438 |
| IB | 0.5 | 0.006025 | 0.707953 | 0.00346064 | 0.339249 | 0.00346158 | 0.339249 |
| IB | 0.75 | 0.0267621 | 1 | 0.00346058 | 0.671426 | 0.00346158 | 0.671426 |
| IB | 1 | 0.0137783 | 0.717233 | 0.00346057 | 0.670702 | 0.00346158 | 0.670702 |
| KdV | 0 | 2.84321e-07 | 0.00183095 | 0.000839551 | 0.000172026 | 0.000117088 | 0.000117088 |
| KdV | 0.2 | 0.0032684 | 0.996688 | 0.0041171 | 0.417637 | 0.000117088 | 0.417983 |
| KdV | 0.5 | 0.603678 | 1 | 0.00383066 | 0.844116 | 0.000117088 | 0.841033 |
| KdV | 0.75 | 0.0366741 | 1 | 0.00300538 | 0.348192 | 0.000117088 | 0.610015 |
| KdV | 1 | 0.036246 | 1 | 0.0047854 | 0.640558 | 0.000117088 | 0.343982 |
| KS | 0 | 5.54625e-07 | 0.0216197 | 0.00035308 | 1.41338e-05 | 9.96404e-06 | 9.96404e-06 |
| KS | 0.2 | 0.00504596 | 1 | 0.00023078 | 0.0210812 | 9.96404e-06 | 0.0211139 |
| KS | 0.5 | 0.0733097 | 1 | 0.000229262 | 0.40804 | 9.96404e-06 | 0.409174 |
| KS | 0.75 | 0.117466 | 0.992756 | 0.000235086 | 0.0850532 | 9.96404e-06 | 0.315363 |
| KS | 1 | 0.269861 | 1 | 0.000227658 | 0.871955 | 9.96404e-06 | 0.946846 |
| NLS | 0 | 5.60489e-08 | 1 | 1 | 1 | 1 | 1 |
| NLS | 0.2 | 0.0211635 | 1 | 1 | 1 | 1 | 1 |
| NLS | 0.5 | 0.111107 | 1 | 1 | 1 | 1 | 1 |
| NLS | 0.75 | 0.400391 | 0.96749 | 1 | 1 | 1 | 1 |
| NLS | 1 | 0.86042 | 0.97397 | 1 | 1 | 1 | 1 |
| RD | 0 | 1.79098e-10 | 1 | 1 | 2.26526e-06 | 1.49915e-07 | 1.49915e-07 |
| RD | 0.2 | 0.0275403 | 1 | 1 | 1 | 1.49915e-07 | 1 |
| RD | 0.5 | 0.0919989 | 1 | 1 | 1 | 1.49915e-07 | 1 |
| RD | 0.75 | 0.233027 | 1 | 1 | 1 | 1.49915e-07 | 1 |
| RD | 1 | 0.472211 | 1 | 1 | 1 | 1.49915e-07 | 1 |
| NS | 0 | 0.000789496 | 0.00190822 | 0.000753911 | 0.000922136 | 0.000921776 | 0.000921776 |
| NS | 0.2 | 0.00160679 | 0.0151515 | 0.000767563 | 0.00257441 | 0.000921776 | 0.0025752 |
| NS | 0.5 | 0.0694726 | 0.0519529 | 0.000810163 | 0.008355 | 0.000921776 | 0.0276361 |
| NS | 0.75 | 0.0590912 | 0.0532917 | 0.000902011 | 0.0487434 | 0.000921776 | 0.0487519 |
| NS | 1 | 0.0459204 | 0.0452787 | 0.000905588 | 0.0655267 | 0.000921776 | 0.0655269 |
| HKS | 0 | 5.10996e-07 | 0.152671 | 0.949516 | 5.75515e-05 | 1.23973e-06 | 1.23973e-06 |
| HKS | 0.2 | 0.0688713 | 1 | 0.935864 | 0.888592 | 1.23973e-06 | 0.888445 |
| HKS | 0.5 | 0.100703 | 1 | 0.935837 | 0.955837 | 1.23973e-06 | 0.955889 |
| HKS | 0.75 | 0.227619 | 0.986731 | 0.935833 | 1 | 1.23973e-06 | 1 |
| HKS | 1 | 0.484694 | 0.982514 | 0.935836 | 0.954384 | 1.23973e-06 | 0.955317 |
| VBG | 0 | 1.17192e-12 | 1 | 1 | 1 | 1 | 1 |
| VBG | 0.2 | 0.0343603 | 0.96854 | 1 | 0.985199 | 1 | 0.968501 |
| VBG | 0.5 | 0.387671 | 1 | 1 | 0.94818 | 1 | 0.953969 |
| VBG | 0.75 | 0.666656 | 1 | 1 | 1 | 1 | 1 |
| VBG | 1 | 0.834371 | 0.967458 | 0.993265 | 0.966485 | 1 | 0.964714 |

Full values, all three oracle levels, support counts and per-setting comparisons are in summary.json/csv. Every prepared JSON saves n, Delta z, energy, contraction alpha, target cap, pilot and observation hashes. Solver paths and certificates are saved in solved/*.json; trials.jsonl joins them with explicit reuse provenance.
