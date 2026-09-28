# Monte Carlo PDE coefficient benchmarks

Every method result below uses independent, uniformly sampled space-time
observations and Monte Carlo weak integrals. Reference grids are used only to
construct a numerical PDE solution or set a noise scale; no grid-based
estimator result is included. Errors are full-library
`||beta_hat-beta_true||_2^2`. These are small numerical sanity checks, not
established accuracy rankings. The earlier 2D porous-medium experiment had
only grid measurements, so it is absent here.

## 3D anisotropic porous medium

PDE: `u_t = 0.3 d_xx(u^2) - 0.2 d_xy(u^2) + 0.1 d_xz(u^2) + 0.7 d_yy(u^2) - 0.16 d_yz(u^2) + d_zz(u^2)`.
The exact mass-one weak solution is sampled uniformly on
`[-5,5]^3 × [0.5,2.5]`. The library has maximum total spatial derivative
order `S=5` (55 multi-indices), powers `u,...,u^5` (`J=5`), and 275
coefficients. The zero-vector squared error is `1.6556`.

There are `n=524,288` iid observations: 262,144 pilot and 262,144 evaluation
points. All methods use the same evaluation points and `K=4096=8^4` translated
tensor-product tests. Each test uses `b_p(r)=(1-r^2)^p` on `|r|<1`, zero
outside. Exponents are `(16,16,16,28)`, physical support half-widths are
`(80/31,80/31,80/31,8/15)`, and center strides on the reference lattice are
`(2,2,2,1)`. The pilot is a box-kernel moving average with bandwidths
`(0.8,0.8,0.8,0.35)`. Raw WSINDy integrates `U^j`; debiased WSINDy uses the
independent pilot to form the corrected powers. Both use identical Monte Carlo
weights and target `Y`.

Noise ratios `0` and `1` use seed `0`, with standard deviation equal to the
ratio times the clean solution's RMS on a `32×32×32×16` reference grid. The
LASSO objective is `||Y-X beta||_2^2/2 + lambda||beta||_1`. Among 25
log-spaced penalties from `1e-4` to `0.2`, `lambda=0.2` had the smallest
coefficient error against known truth for both methods and noise levels. This
is oracle selection for one seed, not a practical tuning rule. MSTLS searches
`logspace(-4,0,50)` thresholds. Runtime is the median of three fits after a
warm-up, including weak-system construction and regression; debiased calls
also include pilot fitting. Data generation and penalty search are excluded.

| Noise ratio | Method | LASSO `lambda` | Squared error | Runtime (s) |
| ---: | --- | ---: | ---: | ---: |
| 0 | Sampled WSINDy (OLS) | — | 1.898895e+12 | 45.732935 |
| 0 | Sampled WSINDy (LASSO) | 0.2 | 1.655600 | 45.698647 |
| 0 | Sampled WSINDy (MSTLS) | — | 1.664912 | 47.217092 |
| 0 | Sampled debiased WSINDy (OLS) | — | 8.871171e+11 | 54.615432 |
| 0 | Sampled debiased WSINDy (LASSO) | 0.2 | 1.655600 | 53.463993 |
| 0 | Sampled debiased WSINDy (MSTLS) | — | 71.946856 | 57.602692 |
| 1 | Sampled WSINDy (OLS) | — | 3.798131e+10 | 47.373257 |
| 1 | Sampled WSINDy (LASSO) | 0.2 | 1.655600 | 47.337400 |
| 1 | Sampled WSINDy (MSTLS) | — | 1.165900 | 49.389236 |
| 1 | Sampled debiased WSINDy (OLS) | — | 1.079440e+12 | 56.455327 |
| 1 | Sampled debiased WSINDy (LASSO) | 0.2 | 1.655600 | 56.432819 |
| 1 | Sampled debiased WSINDy (MSTLS) | — | 62.563155 | 58.195397 |

All four LASSO fits are exactly zero. At noise `0` and `1`, raw MSTLS selected
one spurious term and one true term, respectively. Debiased MSTLS selected
`1/6` and `2/6` true terms, plus `10` and `3` false positives. The correction
did not improve coefficient selection on this seed. Monte Carlo integration
error is substantial even without observation noise, so these measurements do
not establish PDE recovery.

## Nonlinear viscous Burgers

PDE: `u_t = 0.01 u_xx - 0.5 d_x(u^2) - u^3 + 2u^2 + 1`.
This adapted instance uses periodic `x∈[-1,1)`, `t∈[0,1.5]`, and
`u(x,0)=0.5+0.7 sin(pi*x)+0.25 sin(2*pi*x+0.3)`. A centered-difference BDF
solution provides the numerical truth; it is not an estimator result. The
[Messenger–Bortz paper](https://arxiv.org/pdf/2211.16000) does not specify
these initial and boundary data.

At each noise level, `n=524,288` iid observations are split equally into
independent pilot and evaluation samples. The `K=600` translated tests use the
paper bump `exp(9/(r^2-1))` on `|r|<1`, with physical half-widths
`(0.25,0.1875)` in `(x,t)` and support-volume ratio `1/16`. The library uses
spatial orders `0..6` and powers `u^0..u^6`, giving 43 nonzero columns after
omitting derivatives of the constant. Every fit uses MSTLS with
`logspace(-4,0,100)` thresholds. The plug-in and debiased methods use a box
moving-average pilot; the bandwidths below adapt the paper's grid filter
rule to iid points, with a floor of 64 expected training neighbors per
interior box.

Noise standard deviation is `sigma/sigma_c × sigma_c`, where
`sigma_c=sqrt(0.01/3)=0.057735`. Each level has three independent seeds from
`0`. Entries are mean squared coefficient error, exact-support recovery
count, mean relative weak-system residual at the true coefficients, and
median complete fit runtime. Data generation and bandwidth selection are
excluded from runtime. The pilot MSE uses evaluation points and known clean
truth; it is a diagnostic, not a coefficient benchmark.

| `sigma/sigma_c` | Pilot bandwidths `(x,t)` | Pilot MSE |
| ---: | --- | ---: |
| 0 | `(0.015625,0.011719)` | 4.363e-05 |
| 1 | `(0.015625,0.011719)` | 9.838e-05 |
| 2 | `(0.03125,0.023438)` | 0.000326 |

| `sigma/sigma_c` | Method | Squared error | Exact support | True-system residual | Runtime (s) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | Sampled WSINDy (MSTLS) | 17.0592 | 0/3 | 0.262 | 2.292 |
| 0 | Sampled plug-in WSINDy (MSTLS) | 16.7601 | 0/3 | 0.263 | 4.136 |
| 0 | Sampled debiased WSINDy (MSTLS) | 17.0596 | 0/3 | 0.262 | 4.153 |
| 1 | Sampled WSINDy (MSTLS) | 21.72 | 0/3 | 0.262 | 2.380 |
| 1 | Sampled plug-in WSINDy (MSTLS) | 21.2095 | 0/3 | 0.2639 | 4.252 |
| 1 | Sampled debiased WSINDy (MSTLS) | 16.8183 | 0/3 | 0.2619 | 4.255 |
| 2 | Sampled WSINDy (MSTLS) | 14.4543 | 0/3 | 0.2631 | 2.405 |
| 2 | Sampled plug-in WSINDy (MSTLS) | 14.2042 | 0/3 | 0.2693 | 7.235 |
| 2 | Sampled debiased WSINDy (MSTLS) | 9.75798 | 0/3 | 0.2622 | 7.306 |

No method recovered the exact five-term support in these three seeds. In a
separate noise-free iid diagnostic, increasing the sample count reduced the
relative weak-system residual:

| Total iid `n` | Evaluation points | `||Y-X beta_true|| / ||Y||` |
| ---: | ---: | ---: |
| 131,072 | 65,536 | 0.4885 |
| 524,288 | 262,144 | 0.2894 |
| 2,097,152 | 1,048,576 | 0.1435 |

The shrinking residual supports Monte Carlo integration error as a major
limit. The pilot cannot remove error in the shared target `Y`. This adapted,
three-seed check is not a reproduction of the paper's 200-realization figure.

Reproduce the coefficient comparison with:

```sh
OPENBLAS_NUM_THREADS=1 python experiments.py --instance nonlinear_viscous_burgers --methods sampled-wsindy-mstls sampled-plug-in-wsindy sampled-debiased-wsindy --noise-multipliers 0 1 2 --replicates 3 --n-observations 524288 --append
```

## 2D linear advection–diffusion: J=1 control

PDE: `u_t = -0.3 u_x + 0.2 u_y + 0.08 u_xx + 0.04 u_xy + 0.05 u_yy`
on the periodic square `[0,2π)^2`, observed over `t∈[0,0.5]`. The exact
solution sums 31 Fourier modes with `kx=0..4`, `ky=-3..3`, omitting zero and
conjugate duplicates. Each has amplitude `0.18/(1+0.15*(kx^2+ky^2))`, a
uniform phase from fixed seed `42`, advection velocity `(0.3,-0.2)`, and
decay `exp(-t*(0.08*kx^2+0.04*kx*ky+0.05*ky^2))`.

There are `n=131,072` iid observations, split into 65,536 pilot and 65,536
evaluation points. A `64×64×32` reference lattice defines the noise RMS and
test centers; no estimator uses its solution values. The library has maximum
total spatial derivative order `S=5` (20 multi-indices) and `J=1` (`u` only),
giving 20 coefficients, five nonzero. All methods use the same evaluation
points and `K=512` translated tests. The test bump is
`b_p(r)=(1-r^2)^p` on `|r|<1`, zero outside. Reference-lattice half-widths
are `(8,8,5)`, center strides `(6,6,3)`, exponents `(8,8,4)`, and physical
half-widths `(0.785398,0.785398,0.080645)` in `(x,y,t)`. The pilot box
bandwidths are `(0.2,0.2,0.05)`.

Noise ratios are `0`, `0.1`, and `1`; standard deviation is ratio × clean
reference RMS. Iid sampling and observation noise use seed `10000`. MSTLS
searches `logspace(-4,0,50)` thresholds. Runtime is one complete
system-construction-and-fit call, excluding data generation; timings are
illustrative. The zero coefficient vector has squared error `0.1405`.

| Noise ratio | Method | Squared error | True-system residual | Nonzero | Exact support | Runtime (s) |
| ---: | --- | ---: | ---: | ---: | --- | ---: |
| 0 | Sampled WSINDy (OLS) | 0.10888 | 0.9848 | 20 | — | 0.210 |
| 0 | Sampled WSINDy (MSTLS) | 0.126618 | 0.9848 | 4 | no | 0.210 |
| 0 | Sampled debiased WSINDy (OLS) | 0.10888 | 0.9848 | 20 | — | 0.670 |
| 0 | Sampled debiased WSINDy (MSTLS) | 0.126618 | 0.9848 | 4 | no | 0.693 |
| 0.1 | Sampled WSINDy (OLS) | 0.102475 | 0.9845 | 20 | — | 0.212 |
| 0.1 | Sampled WSINDy (MSTLS) | 0.126686 | 0.9845 | 4 | no | 0.221 |
| 0.1 | Sampled debiased WSINDy (OLS) | 0.102475 | 0.9845 | 20 | — | 0.676 |
| 0.1 | Sampled debiased WSINDy (MSTLS) | 0.126686 | 0.9845 | 4 | no | 0.667 |
| 1 | Sampled WSINDy (OLS) | 0.028805 | 0.9935 | 20 | — | 0.213 |
| 1 | Sampled WSINDy (MSTLS) | 0.0298393 | 0.9935 | 19 | no | 0.212 |
| 1 | Sampled debiased WSINDy (OLS) | 0.028805 | 0.9935 | 20 | — | 0.678 |
| 1 | Sampled debiased WSINDy (MSTLS) | 0.0298393 | 0.9935 | 19 | no | 0.674 |

The raw and debiased iid weak systems differ by at most `7.11e-15` across
these noise levels. For `J=1`, the corrected power is exactly
`u_hat-(u_hat-U)=U`; equal coefficients are the expected negative control,
not evidence of a debiasing gain. The true-system residual is large even
without observation noise, so coefficient errors here are dominated by Monte
Carlo integration. A smaller error at a noisier level is one random
realization, not a noise-robustness result.

Reproduce with:

```sh
OPENBLAS_NUM_THREADS=1 python experiments.py --instance linear_advection_diffusion --nx 64 --ny 64 --nt 32 --n-observations 131072 --seed 0 --noise-ratios 0 0.1 1 --methods sampled-wsindy-ols sampled-wsindy-mstls sampled-debiased-wsindy-ols sampled-debiased-wsindy-mstls --append
```
