# pde-discovery

Weak SINDy for PDEs ([Messenger & Bortz, arXiv:2007.02848v3](https://arxiv.org/abs/2007.02848)) with
pluggable dictionaries, state estimators and regressions. Dictionaries, optimizers and smoothers can
be taken directly from [PySINDy](https://pysindy.readthedocs.io) (pinned to 2.1.0); MIO and conic programs use gurobipy (license required); the convolutional
weak form, the authors' scaling and MSTLS are implemented here.

```
wsindy/
  dictionary.py     Block(functions=<PySINDy library>, alphas) -> terms D^alpha f(U)
  testfunctions.py  Tests(m, p, s, shape); unset fields are chosen from data
  system.py         WeakSystem(pde, U, grid, X=..., Y=...); SplitSystem: regression times of a split;
                    StrongSystem: pointwise SINDy (PySINDy derivatives)
  regression.py     MSTLS, LassoPath, BestSubset (MIO, gurobipy) with loss "ls" | "debiased";
                    FromAPI(any PySINDy/sklearn optimizer)
  eiv.py            EIV conic estimator (Section 6, scalar polynomial dictionaries), solved with Gurobi
  estimators.py     uhat: MovingAverage, SplitPilot, FromPySINDy, FromCallable
pdes.py             benchmark PDEs (data, dictionary, truth, test functions), noise, seeds, metrics
run.py              trials -> benchmark_results/runs/<method>/<PDE>.jsonl
report.py           saved trials -> results.md, results.summary.csv, results.png
```

## Usage

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests

python run.py KS NLS RD --method wsindy filtered lasso --trials 100 --workers 4
python run.py IB --method debiased --split .3 --out benchmark_results/split-0.3   # uhat from 30% of the times
python report.py benchmark_results -o results.md --plot results.png
```

`run.py` resumes: completed `(noise ratio, trial)` records are skipped. Noise seeds depend only on
(PDE, noise ratio, trial), so every method sees identical observations.

## Composing a method

A method is a state estimator `uhat`, the state used on each side of the weak system, and a regression.

```python
import pysindy as ps
from pdes import PDES, load, observe, trial_seed
from wsindy import WeakSystem, SplitSystem, estimate, fit, MSTLS, LassoPath, BestSubset, SplitPilot

pde = PDES["KS"]
clean, grid = load("KS")                                   # U: (*space, time, n_components)
U, _ = observe(pde, clean, 0.5, trial_seed("KS", 0.5, 0))

uhat, info = estimate(U, grid, pde, "movavg")              # ours: "movavg", MovingAverage(...)
uhat, info = estimate(U, grid, pde, ps.SmoothedFiniteDifference())        # PySINDy smoother (time axis)
uhat, info = estimate(U, grid, pde, lambda U, grid: my_smoother(U))      # any function

system = WeakSystem(pde, U, grid, X=uhat, Y=U)             # choose the state for X and for Y
W = fit(system, "mstls").coef                              # (n_equations, n_terms), physical units
W = fit(system, LassoPath()).coef
W = fit(system, BestSubset(r=3)).coef                     # MIO best subset: <= r = 3 nonzeros (tuple: per equation)
W = fit(system, BestSubset()).coef                        # r = 1, 2, ... per equation until the Eq. 4.4 loss rises
W = fit(system, ps.SR3(reg_weight_lam=0.1, regularizer="L1")).coef            # any PySINDy/sklearn optimizer

uhat, info = estimate(U, grid, pde, SplitPilot(ratio=.5))  # from half of the time points only; NaN there
system = SplitSystem(pde, U, grid, X=uhat, Y=U)            # integrals over the other half only
W = fit(system, MSTLS(loss="debiased")).coef               # linearize monomials at uhat
```

New regressions implement `regress(system) -> Fit`; new losses implement `matrices(system)` and
`solve(X, y)` and plug into `MSTLS(loss=...)` / `LassoPath(loss=...)`. New estimators are callables
`(U, grid, pde) -> (uhat, info)`. Register named combinations in `run.METHODS`:

| method | uhat | X | Y | regression |
| --- | --- | --- | --- | --- |
| `wsindy` | — | U | U | MSTLS |
| `filtered` | moving average | uhat | uhat | MSTLS |
| `debiased` | `SplitPilot`: box average over a fraction `--split` (default .5) of the times | uhat | U, other times only (`SplitSystem`) | MSTLS, debiased loss |
| `lasso`, `filtered-lasso`, `debiased-lasso` | as above | | | LASSO path |
| `eiv` | — | U | U | EIV |
| `sindy` | — | U | U | MSTLS on the pointwise (strong-form) system `StrongSystem` |
| `best-subset`, `filtered-best-subset`, `debiased-best-subset` | as above | | | `BestSubset` with sparsity r per PDE (`TRUE_R` in `run.py`: true r per equation) |

A dictionary is a tuple of blocks; any PySINDy feature library works, polynomial or not:

```python
from wsindy import Block, Tests, spatial
Block(ps.PolynomialLibrary(degree=4) + ps.FourierLibrary(n_frequencies=2), spatial(order=4, dim=2))
Tests(m=(23, 22), p=(10, 10), s=(5, 6))   # fixed (paper Table 3)
Tests()                                     # m: spectral changepoint (findcorners.m), p: Algorithm 4.1, s: N/50 or N/25
```

Monomial features (from `PolynomialLibrary`) get the state scaling and the debiased linearization;
other features are evaluated on the unscaled state, as the authors' code does for trigonometric terms.

### Data splitting in the debiased methods

Every debiased method splits the time points. `SplitPilot(ratio)` reserves a fraction `ratio` of them,
spread evenly (ratio .5: the odd times), and only these determine uhat, the noise level, the averaging
window and the state scaling. The corrected weak integrals and the regression use the remaining times
only. `SplitSystem` reads the split from the estimate, which is NaN at its training times, so the two
cannot disagree. No observation enters both uhat and the regression; the groups are never swapped or
pooled. Fixed `Tests.m` is required, since data-dependent supports would look at the regression times.

`SplitSystem` keeps the full-grid supports, derivatives and query centers, and integrates the regression
times with trapezoid weights on their own grid (`2*dt` each for ratio .5). For unequally spaced training
times (e.g. ratio .3) the noise level uses sixth divided differences, and the time window is long enough
to reach a training time from every regression time.

Conditioning on the training times, the corrected weak matrices are affine in the regression
observations, so the residual is Gaussian under independent Gaussian noise. The design still contains
that noise (`grad f(uhat) eps`), like the linear columns of any weak form: conditionally this is an
errors-in-variables model, not a fixed design. The affine property holds for monomial features; other
PySINDy features are not linearized. Coefficient-error and support-recovery guarantees still need
approximation-error, conditioning and sparse-regression analysis.

Earlier debiased runs (`benchmark_results/debiasing/time2_foldwise`, and `debiased*` in
`benchmark_results/debiasing/single_split`) averaged each time parity over the other but pooled both
parities in every integral; `report.py` lists them as `crossfit-debiased*`. The `split-debiased*` runs in
`single_split` equal the current `debiased*` with `--split .5`.

## Benchmarks and protocol

- Original suite: IB, KdV, KS, NLS, RD, NS with the authors' `U_exact` arrays
  ([Zenodo](https://zenodo.org/records/20787783), MD5-checked), Table 3 supports/degrees/strides,
  50 thresholds `logspace(-4, 0, 50)`, physical-unit MSTLS bounds and state scale
  `(||U||/||U^P||)^(1/(P-1))` as in the authors' MATLAB code. RD uses the archived 181-term
  library (degree 5, derivative order 4).
- Consistency suite ([arXiv:2211.16000v1](https://arxiv.org/abs/2211.16000)): HKS and VBG are
  declared resimulations (Fourier ETDRK4; ICs, periodic BCs, grids and strides chosen here), bump
  test functions, no rescaling, 100 absolute thresholds.
- Noise: Gaussian, std = ratio × component RMS (original) or pooled centered std (consistency);
  default schedule 5 ratios × 100 trials (the paper used 41 × 200).
- Metrics: TPR = TP/(TP+FN+FP), E_inf over true terms (Eq. 5.4), relative E2 (Eq. 5.5).

## Known differences from the authors' code and paper

Checked against [dm973/WSINDy_PDE](https://github.com/dm973/WSINDy_PDE) with a line-by-line NumPy
port: weak matrices, scales and MSTLS agree to machine precision once query points are aligned.

- Query centers are grid indices `m, m+s, ...`; the MATLAB FFT indexing centers them at `m+1, ...`
  (same count). The shift does not change recovery statistics, but last digits of errors differ.
- When thresholding would delete every term, the previous fit is kept (MathBioCU/WSINDy_PDE rule);
  dm973 returns an empty model. No difference was observed over 310 checked trials.
- The paper's Table 5 E_inf values equal `||w - w*||_inf / ||w*||_inf`, not Eq. 5.4; the two differ
  for RD and NS (e.g. NS noise-free: 1.1e-3 vs 5.4e-3).
- RD strides follow Table 3 `(13, 13, 12)`, giving 4860 rows; Table 4's 11638 rows correspond to the
  stated N/25 rule `(10, 10, 8)`.
- Original MATLAB noise draws are not reproduced.

Saved runs live in `benchmark_results/`; `results.md` is regenerated from them by `report.py`.
Earlier code versions are in the git history.
