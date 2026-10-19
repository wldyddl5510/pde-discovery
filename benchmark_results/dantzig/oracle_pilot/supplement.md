# 독립 증명을 추가한 minimum-cardinality oracle

Completed **104/104** paired realizations. This is a separate reporting supplement; raw Gurobi flags and main result files are unchanged.

The independently verified NLS and VBG proofs certify 22 existing exact-plugin candidates as minimum-cardinality solutions of Equation (9). The report adds them to the Gurobi-cardinality-certified candidate pool, then applies the same per-equation minimum coefficient-error lambda oracle. It verifies exact coefficient/lambda matches and identical physical H,b before transferring certificates to reused paths.

**The independent proof certifies bare Equation (9) sparsity optimality. Global optimality of the added residual tie-break is not established.** Certificates are not transferred to exact-debiased or rate scores.

## Aggregate comparison

| Scope | Method | N | Raw Gurobi mean E2 | + independent proof mean E2 | Raw median | + proof median | Raw exact support | + proof exact support | Raw zero | + proof zero |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | wsindy | 104 | 0.1844536 | 0.1844536 | 0.06922005 | 0.06922005 | 55 | 55 | 0 | 0 |
| all | debiased_mstls | 104 | 5.151964 | 5.151964 | 0.6721468 | 0.6721468 | 28 | 28 | 0 | 0 |
| all | plugin_dantzig | 104 | 0.8353696 | 0.8353696 | 1 | 1 | 6 | 6 | 59 | 59 |
| all | debiased_dantzig | 104 | 0.831337 | 0.831337 | 1 | 1 | 3 | 3 | 69 | 69 |
| all | rate_plugin_dantzig | 104 | 0.4929471 | 0.4929471 | 0.4676824 | 0.4676824 | 25 | 25 | 38 | 38 |
| all | rate_debiased_dantzig | 104 | 0.6484481 | 0.6484481 | 0.9354182 | 0.9354182 | 14 | 14 | 42 | 42 |
| all | exact_plugin_dantzig | 104 | 0.250564 | 0.0005674128 | 0.000519432 | 1.873403e-05 | 52 | 78 | 26 | 0 |
| all | exact_debiased_dantzig | 104 | 0.6564179 | 0.6564179 | 0.9343235 | 0.9343235 | 13 | 13 | 39 | 39 |
| nonzero_noise | wsindy | 96 | 0.199816 | 0.199816 | 0.0708482 | 0.0708482 | 47 | 47 | 0 | 0 |
| nonzero_noise | debiased_mstls | 96 | 5.57406 | 5.57406 | 0.940421 | 0.940421 | 21 | 21 | 0 | 0 |
| nonzero_noise | plugin_dantzig | 96 | 0.849717 | 0.849717 | 1 | 1 | 5 | 5 | 56 | 56 |
| nonzero_noise | debiased_dantzig | 96 | 0.857094 | 0.857094 | 1 | 1 | 2 | 2 | 65 | 65 |
| nonzero_noise | rate_plugin_dantzig | 96 | 0.4928289 | 0.4928289 | 0.4676824 | 0.4676824 | 23 | 23 | 35 | 35 |
| nonzero_noise | rate_debiased_dantzig | 96 | 0.6816039 | 0.6816039 | 0.9435021 | 0.9435021 | 10 | 10 | 40 | 40 |
| nonzero_noise | exact_plugin_dantzig | 96 | 0.250564 | 0.0005674128 | 0.000519432 | 1.873403e-05 | 48 | 72 | 24 | 0 |
| nonzero_noise | exact_debiased_dantzig | 96 | 0.690239 | 0.690239 | 0.9418445 | 0.9418445 | 9 | 9 | 37 | 37 |
| noiseless | wsindy | 8 | 0.0001042484 | 0.0001042484 | 3.976586e-07 | 3.976586e-07 | 8 | 8 | 0 | 0 |
| noiseless | debiased_mstls | 8 | 0.08680937 | 0.08680937 | 0.0009028075 | 0.0009028075 | 7 | 7 | 0 | 0 |
| noiseless | plugin_dantzig | 8 | 0.6632011 | 0.6632011 | 0.8239872 | 0.8239872 | 1 | 1 | 3 | 3 |
| noiseless | debiased_dantzig | 8 | 0.5222538 | 0.5222538 | 0.5763357 | 0.5763357 | 1 | 1 | 4 | 4 |
| noiseless | rate_plugin_dantzig | 8 | 0.4943654 | 0.4943654 | 0.4764883 | 0.4764883 | 2 | 2 | 3 | 3 |
| noiseless | rate_debiased_dantzig | 8 | 0.2505787 | 0.2505787 | 0.0005470807 | 0.0005470807 | 4 | 4 | 2 | 2 |
| noiseless | exact_plugin_dantzig | 8 | 0.250564 | 0.0005674128 | 0.000519432 | 1.873403e-05 | 4 | 6 | 2 | 0 |
| noiseless | exact_debiased_dantzig | 8 | 0.250564 | 0.250564 | 0.000519432 | 0.000519432 | 4 | 4 | 2 | 2 |

## Exact-plugin results by PDE

All other methods retain their original cardinality-oracle metrics.

| PDE | Scope | N | Raw Gurobi mean E2 | + independent proof mean E2 | Raw exact support | + proof exact support |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| IB | all | 13 | 0.00346158039 | 0.00346158039 | 13 | 13 |
| IB | nonzero_noise | 12 | 0.00346158039 | 0.00346158039 | 12 | 12 |
| IB | noiseless | 1 | 0.00346158039 | 0.00346158039 | 1 | 1 |
| KdV | all | 13 | 0.000117088177 | 0.000117088177 | 13 | 13 |
| KdV | nonzero_noise | 12 | 0.000117088177 | 0.000117088177 | 12 | 12 |
| KdV | noiseless | 1 | 0.000117088177 | 0.000117088177 | 1 | 1 |
| KS | all | 13 | 9.9640405e-06 | 9.9640405e-06 | 0 | 0 |
| KS | nonzero_noise | 12 | 9.9640405e-06 | 9.9640405e-06 | 0 | 0 |
| KS | noiseless | 1 | 9.9640405e-06 | 9.9640405e-06 | 0 | 0 |
| NLS | all | 13 | 1 | 2.75040154e-05 | 0 | 13 |
| NLS | nonzero_noise | 12 | 1 | 2.75040154e-05 | 0 | 12 |
| NLS | noiseless | 1 | 1 | 2.75040154e-05 | 0 | 1 |
| RD | all | 13 | 1.49915433e-07 | 1.49915433e-07 | 13 | 13 |
| RD | nonzero_noise | 12 | 1.49915433e-07 | 1.49915433e-07 | 12 | 12 |
| RD | noiseless | 1 | 1.49915433e-07 | 1.49915433e-07 | 1 | 1 |
| NS | all | 13 | 0.000921775885 | 0.000921775885 | 0 | 0 |
| NS | nonzero_noise | 12 | 0.000921775885 | 0.000921775885 | 0 | 0 |
| NS | noiseless | 1 | 0.000921775885 | 0.000921775885 | 0 | 0 |
| HKS | all | 13 | 1.23972918e-06 | 1.23972918e-06 | 13 | 13 |
| HKS | nonzero_noise | 12 | 1.23972918e-06 | 1.23972918e-06 | 12 | 12 |
| HKS | noiseless | 1 | 1.23972918e-06 | 1.23972918e-06 | 1 | 1 |
| VBG | all | 13 | 1 | 2.52567907e-10 | 0 | 13 |
| VBG | nonzero_noise | 12 | 1 | 2.52567907e-10 | 0 | 12 |
| VBG | noiseless | 1 | 1 | 2.52567907e-10 | 0 | 1 |

## Independently proved NLS example

| Equation | lambda / ‖b‖∞ | Feasible three-term E2 | Minimum two-column dual bound / lambda | Original Gurobi cardinality flag |
| ---: | ---: | ---: | ---: | :--- |
| 0 | 0.01 | 2.74141327e-05 | 1.4882314 | False |
| 1 | 0.01 | 2.75936053e-05 | 1.58630107 | False |

Every support of size ≤2 is contained in an independently excluded pair; the three-term primal is exactly feasible for the saved binary64 score. These candidates therefore are Equation (9) minimizers despite their original unresolved Gurobi flags. The better performance demonstrates that restricting interpretation to the original solver-certified zero models would miss valid accurate minimizers.

## Independently proved VBG examples

| lambda / ‖b‖∞ | Feasible five-term E2 | Minimum four-column dual bound / lambda | Original Gurobi cardinality flag |
| ---: | ---: | ---: | :--- |
| 3.16227766e-06 | 2.52567906941e-10 | 3.05029543 | False |
| 1e-06 | 2.52567906941e-10 | 9.6458811 | False |
| 5.62341325e-06 | 2.52567906941e-10 | 1.71530718 | False |

All 123,410 four-column supports are exactly excluded at each certified VBG lambda; an exactly feasible five-term vector proves minimum cardinality five. The VBG lambda ratio1e-5 candidate is not certified by these bounds and is excluded from the expanded pool.

The finite lambda grid, remaining unproved supports, fixed trajectories, and oracle access to the clean state remain practical limits of this sanity check.

```sh
/opt/homebrew/anaconda3/envs/pde_discovery/bin/python benchmark_results/dantzig/oracle_pilot/supplement.py --require-complete
```
