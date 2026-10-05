# Scalar Section 6 EIV sanity study

Status: complete; 25/25 attempted fits. Trials per PDE/noise level: 1.

Empirical plug-in sanity study of the scalar Section 6 conic estimator. The variance is estimated from each observed array, including at zero injected noise. The Gaussian polynomial correction and derived tolerances use that estimate. No clean state, injected variance, true coefficient or support selects the fit. The known-variance conditional calculations do not establish 1-delta coverage for this data-dependent plug-in procedure. The specified g2 uses the operator 2-norm. Physical state and coordinates, the full polynomial library, and the benchmark's weak tests are retained; there is no silent state, column or response rescaling. Coefficient error uses the returned coefficients without thresholding or refitting. Support alone uses the solver's declared support tolerance.

λ = 1, δ = 0.05; solver tolerance = 1e-08; support tolerance = 1e-07.

When τ ≥ ‖y‖∞, (w,t)=(0,0) is feasible and has objective zero. Because ‖w‖₁ + λt is nonnegative, the zero solution is globally optimal. A zero estimate in this regime is a consequence of these tolerances, not a solver failure.

`Exact support` uses the returned support mask. E2 is ‖w−w*‖₂/‖w*‖₂ from the unmodified conic estimate. `Truth feasible` tests the residual bound at t=‖w*‖₂ after fitting and is diagnostic only. Failed fits remain in the trial records; summary errors average successful fits only. Nonfinite values are saved as JSON null.

| PDE | Noise | Trial | Status | σ̂² | μ | τ | ‖y‖∞ | τ/‖y‖∞ | Zero optimal | Exact support (tol=1e-07) | E2 | Truth feasible |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|---:|---|
| IB | 0 | 0 | AnalyticZero | 132.7 | 1.961e+26 | 9.83e+10 | 2.918e+05 | 3.368e+05 | yes | no | 1 | yes |
| IB | 0.2 | 0 | AnalyticZero | 1.969e+04 | 4.874e+27 | 2.551e+11 | 2.987e+05 | 8.538e+05 | yes | no | 1 | yes |
| IB | 0.5 | 0 | AnalyticZero | 1.228e+05 | 8.049e+28 | 5.037e+11 | 3.127e+05 | 1.611e+06 | yes | no | 1 | yes |
| IB | 0.75 | 0 | AnalyticZero | 2.747e+05 | 5.197e+29 | 8.166e+11 | 3.412e+05 | 2.393e+06 | yes | no | 1 | yes |
| IB | 1 | 0 | AnalyticZero | 4.897e+05 | 2.069e+30 | 9.319e+11 | 3.242e+05 | 2.875e+06 | yes | no | 1 | yes |
| KdV | 0 | 0 | AnalyticZero | 1.359e-06 | 3.296e+26 | 4.46e+07 | 207.1 | 2.154e+05 | yes | no | 1 | yes |
| KdV | 0.2 | 0 | AnalyticZero | 3011 | 1.753e+27 | 1.05e+08 | 207.4 | 5.061e+05 | yes | no | 1 | yes |
| KdV | 0.5 | 0 | AnalyticZero | 1.874e+04 | 1.064e+28 | 2.386e+08 | 208.5 | 1.144e+06 | yes | no | 1 | yes |
| KdV | 0.75 | 0 | AnalyticZero | 4.25e+04 | 2.097e+28 | 3.079e+08 | 209.4 | 1.471e+06 | yes | no | 1 | yes |
| KdV | 1 | 0 | AnalyticZero | 7.599e+04 | 3.26e+28 | 4.046e+08 | 212.7 | 1.902e+06 | yes | no | 1 | yes |
| KS | 0 | 0 | AnalyticZero | 1.625e-07 | 1.05e+08 | 1.452e+04 | 6.971 | 2082 | yes | no | 1 | yes |
| KS | 0.2 | 0 | AnalyticZero | 0.05015 | 4.403e+08 | 2.056e+04 | 7.039 | 2921 | yes | no | 1 | yes |
| KS | 0.5 | 0 | AnalyticZero | 0.3103 | 2.177e+09 | 3.925e+04 | 7.406 | 5300 | yes | no | 1 | yes |
| KS | 0.75 | 0 | AnalyticZero | 0.7019 | 7.966e+09 | 5.226e+04 | 7.233 | 7225 | yes | no | 1 | yes |
| KS | 1 | 0 | AnalyticZero | 1.245 | 1.266e+10 | 6.708e+04 | 7.866 | 8528 | yes | no | 1 | yes |
| HKS | 0 | 0 | AnalyticZero | 1.405e-10 | 35.57 | 0.0001609 | 6.267e-08 | 2567 | yes | no | 1 | yes |
| HKS | 0.2 | 0 | AnalyticZero | 0.03771 | 104.2 | 0.0001958 | 6.327e-08 | 3094 | yes | no | 1 | yes |
| HKS | 0.5 | 0 | AnalyticZero | 0.2336 | 719.5 | 0.0004249 | 6.708e-08 | 6335 | yes | no | 1 | yes |
| HKS | 0.75 | 0 | AnalyticZero | 0.5405 | 2586 | 0.0005589 | 6.146e-08 | 9093 | yes | no | 1 | yes |
| HKS | 1 | 0 | AnalyticZero | 0.9552 | 7662 | 0.0007133 | 6.196e-08 | 1.151e+04 | yes | no | 1 | yes |
| VBG | 0 | 0 | AnalyticZero | 3.644e-11 | 5622 | 6.977e-07 | 3.081e-09 | 226.5 | yes | no | 1 | yes |
| VBG | 0.2 | 0 | AnalyticZero | 0.03664 | 8.222e+04 | 1.996e-06 | 3.073e-09 | 649.4 | yes | no | 1 | yes |
| VBG | 0.5 | 0 | AnalyticZero | 0.2288 | 1.15e+06 | 4.84e-06 | 3.137e-09 | 1543 | yes | no | 1 | yes |
| VBG | 0.75 | 0 | AnalyticZero | 0.5146 | 4.512e+06 | 6.521e-06 | 3.06e-09 | 2131 | yes | no | 1 | yes |
| VBG | 1 | 0 | AnalyticZero | 0.9143 | 1.901e+07 | 9.5e-06 | 3.105e-09 | 3060 | yes | no | 1 | yes |

Reproducibility: [manifest](results.manifest.json), [summary](results.summary.json), per-PDE JSONL in `results_trials/`, and frozen numerical sources in `results_source/`. The manifest records source and dataset SHA-256 hashes. Original and consistency datasets retain their existing noise conventions and seeds.
