# Implementation corrections

The PDE residual, noise Jacobian, covariance, and likelihood derivatives pass independent numerical checks. The audit also found a real L1 convergence bug.

| Issue | Correction |
|---|---|
| Tiny L1 penalties could accept the initial coefficients | Penalty-aware KKT checks; active-set SVD LASSO for WENDy; corrected objective scaling for MLE-L1 |
| Nearly singular covariance from overlapping tests | Orthonormal SVD basis, applied consistently to Ŷ, X̂, and every PDE operator |
| First-order optimization for PDE MLE | Trust-region Newton-CG with exact Hessian-vector products; L1 retains constrained L-BFGS-B |
| Unbounded fit runtime | 200-second limit; unfinished fits are marked failed and retained |

At 20% noise, seed 0, evaluated at the true coefficients:

| PDE | Previous cond(C) | Corrected cond(C) | Retained equations |
|---|---:|---:|---:|
| Burgers | 7.95e11 | 91.3 | 52 / 784 |
| KdV | 2.90e11 | 315 | 200 / 1443 |

**20 tests pass**, including an independent convolution Jacobian, GLS update, projected operators, likelihood gradient/Hessian, exact LASSO solutions, objective-scale invariance, tiny-λ stationarity on the paper's Burgers data, and timeout handling.

The SVD basis targets 95% of the singular-value sum, with condition limit 1e4 and at most 200 rows. It retains 95.1% for Burgers and 85.2% for KdV (the cap binds). This follows the conditioning approach in [WENDy.jl](https://github.com/nrummel/WENDy.jl/blob/edaaab7392adf8b307a4a607c4210a910db41076/src/wendyTestFunctions.jl); our separable PDE basis uses the complete spectrum.

The residual and covariance remain

$$R(w)=\widehat Y-\widehat Xw,\qquad L(w)=A_0-\sum_{s,j}w_{sj}A_s\operatorname{diag}(f'_j(U)),\qquad C(w)=\sigma^2L(w)L(w)^T+\varepsilon I.$$

Preprocessing scales are held fixed in these derivatives. The ridge is 1e-10 times the mean diagonal of C(0) before regularization. Data, windows, candidate terms, noise seeds, and α values are unchanged; the test basis, λ_ref, L1 initialization (zero), and solvers change. Thus the rerun measures the combined corrections, not an isolated solver effect.

These remain **sparse PDE extensions**, not reproductions of the ODE experiments. They retain WSINDy polynomial windows, known σ, an IRLS coefficient-step stopping rule, and HT/L1 selection without refitting. Original WENDy's multiscale bump functions, identity shrinkage, and Shapiro–Wilk stopping safeguard are not reproduced. At σ=0, both WENDy variants use least squares/LASSO. Solver convergence does not guarantee support recovery; false high-derivative terms can have very large coefficients in physical units.

[Corrected results](paper_method_corrected/summary.md) · [Previous results](paper_method_comparison/summary.md)

References: [WENDy Algorithm 2](https://arxiv.org/html/2302.13271v3#S2.SS2) · [WENDy MATLAB code](https://github.com/MathBioCU/WENDy/tree/65c52a893b1f693cae3cd17e21854665ea814b9d) · [MLE algorithm and time limit](https://arxiv.org/html/2502.08881v3#S3) · [MLE experiment code](https://github.com/nrummel/rummelWENDy2025/tree/1983c5c965b7802af5c270ecf2565e8fca3dce6e).
