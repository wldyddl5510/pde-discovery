# Scalar Section 6 EIV sanity study

Run from the repository root:

```sh
python benchmark_results/eiv/run_sanity.py
```

The default is **25 fits**: IB, KdV, KS, HKS and VBG; noise ratios 0, 0.2,
0.5, 0.75 and 1; one trial per setting. It uses only existing verified data
(`download=False`). The original benchmark noise instances are reproduced
with `experiments.trial_seed(0, name, noise_ratio, trial)` and
`simulation_generation.observation_instance`.

The scalar estimator always receives `sigma2=None`: variance is estimated
from the observed field even at zero injected noise. It derives μ and τ
without coefficient truth, clean data or the injected noise standard deviation.
The full library, physical coordinates, original weak supports and strides,
polynomial degrees for original datasets, and bump tests for HKS/VBG are used.
Solver equilibration is disabled and failures are recorded without silently
rescaling the problem.

This is an empirical plug-in sanity study. The known-variance conditional
calculations do **not** supply a 1−δ coverage guarantee when variance and other
bounds are estimated from the same observed data. The requested definition of
g₂ uses the operator 2-norm.

Coefficient errors use the returned conic coefficients verbatim. Support
scoring uses the separately returned support mask at the declared tolerance;
there is no thresholded coefficient error or least-squares refit. Truth
residuals and feasibility at t=‖w*‖₂ are recorded only after fitting.

If τ ≥ ‖y‖∞, the zero coefficient vector with t=0 is feasible and attains
objective zero. Since ‖w‖₁+λt is nonnegative for λ>0, this is a global optimum.
An all-zero fit in that regime is therefore explained by the tolerances.

Outputs are written under `benchmark_results/eiv/scalar_sanity/`:

- `results.md`: per-trial status, estimated variance, μ, τ, response norm,
  τ/‖y‖∞, analytic zero-optimality flag, support recovery and E2.
- `results_trials/*.jsonl`: coefficients, support, complete tolerance and
  solver diagnostics, post-fit truth diagnostics, hashes and failure records.
- `results.summary.json` and `results.status.json`: counts and aggregate errors
  over successful solves.
- `results.manifest.json` and `results_source/`: complete protocol, software
  versions, data/source hashes and frozen numerical sources.

The runner does not edit previous benchmark reports or baseline artifacts.
An existing run requires `--resume` and an identical protocol and source hash.
Use `--report-only` to regenerate its report from saved trial records. Use a
fresh output directory when changing settings or code, for example:

```sh
python benchmark_results/eiv/run_sanity.py --benchmarks IB KS --noise-ratios 0 0.5 \
  --lambda 1 --delta 0.05 --trials 1 \
  --output-dir benchmark_results/eiv/custom_sanity
```

The defaults deliberately do not run the 100-trial benchmark suite. Coupled
NLS/RD and the multicomponent NS library are outside this scalar runner.
