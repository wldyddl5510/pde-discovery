"""Quick 1D/5D linear PDE comparison with exact Fourier solutions and weak tests."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy import sparse

import wsindy
import wendy
import wendy_mle
from experiment import comparison_tables, save_csv, support_metrics


def model(dimension):
    """u_t = c*u + b.grad(u) + sum_{i<=j} a_ij D_ij(u)."""
    eye = np.eye(dimension, dtype=int)
    alpha = [np.zeros(dimension, dtype=int), *eye]
    alpha += [eye[i]+eye[j] for i in range(dimension) for j in range(i, dimension)]
    target = np.full(len(alpha), .001)
    strong = np.array([0, 1, dimension+1])
    target[strong] = [-1., -1., 1.]
    return np.array(alpha), target, strong


def modes(dimension):
    if dimension == 1:
        return np.arange(1, 4)[:, None]
    eye = np.eye(dimension, dtype=int)
    vectors = [*eye, *(2*eye)]
    vectors += [eye[i]+sign*eye[j] for i in range(dimension) for j in range(i+1, dimension)
                for sign in (-1, 1)]
    vectors += [np.ones(dimension, dtype=int), np.arange(dimension) % 2*2-1]
    return np.array(vectors)


def fourier_operators(wavevectors, alpha):
    """Real cosine/sine coefficient operators, paired by wavevector."""
    symbols = np.prod((1j*wavevectors[None, :, :])**alpha[:, None, :], axis=2)
    operators = [sparse.block_diag([[[z.real, z.imag], [-z.imag, z.real]] for z in row], format="csr")
                 for row in symbols]
    return symbols, operators


def data(dimension, nt=129, seed=0):
    # Same total spatial observations in 1D and 5D: 32768 versus 8^5.
    nx = 32768 if dimension == 1 else 8
    x = np.linspace(-np.pi, np.pi, nx, endpoint=False)
    points = np.stack(np.meshgrid(*([x]*dimension), indexing="ij"), axis=-1).reshape(-1, dimension)
    t = np.linspace(0, .5, nt)
    alpha, target, strong = model(dimension)
    wavevectors = modes(dimension)
    start = perf_counter()
    phase = points@wavevectors.T
    basis = np.stack((np.sqrt(2)*np.cos(phase), np.sqrt(2)*np.sin(phase)), axis=-1).reshape(len(points), -1)
    symbols, operators = fourier_operators(wavevectors, alpha)
    weights = np.full(nt, t[1]-t[0])
    weights[[0, -1]] *= .5
    wt = weights*wsindy._bump(t, .25, .25)
    dwt = -weights*wsindy._bump(t, .25, .25, derivative=1)
    A = (sparse.kron(dwt[None, :], sparse.eye(basis.shape[1]), format="csr"),)
    A += tuple(sparse.kron(wt[None, :], operator, format="csr") for operator in operators)
    assembly_seconds = perf_counter()-start
    # Every spatial direction and mixed derivative participates in the exact PDE.
    rates = target@symbols
    phase0 = np.random.default_rng(1).uniform(-np.pi, np.pi, len(wavevectors))
    complex_modes = .3/np.sqrt(len(wavevectors))*np.exp(t[:, None]*rates+1j*phase0)
    dominant_modes = .3/np.sqrt(len(wavevectors))*np.exp(t[:, None]*(target[strong]@symbols[strong])+1j*phase0)
    clean = np.stack((complex_modes.real, -complex_modes.imag), axis=-1).reshape(nt, -1)
    rms = np.sqrt(np.mean(np.sum(clean**2, axis=1)))
    physical_noise = np.random.default_rng(seed).normal(size=(len(points), nt))
    start = perf_counter()
    projected_noise = (basis.T@physical_noise/len(points)).T
    projection_seconds = perf_counter()-start
    # Orthonormal spatial projection: independent coefficient noise has variance sigma^2/Nx.
    gram_error = np.max(np.abs(basis.T@basis/len(points)-np.eye(basis.shape[1])))
    full_alpha = ((0,)*dimension+(1,),)+tuple(tuple(a)+(0,) for a in alpha)
    system = wsindy.WeakSystem(clean, A, wsindy.polynomial_dictionary(1)[1:], full_alpha)
    validation = dict(spatial_orthogonality_error=float(gram_error),
                      relative_weak_term_effect=float(np.linalg.norm(complex_modes-dominant_modes)/np.linalg.norm(complex_modes)),
                      spatial_wavevector_rank=int(np.linalg.matrix_rank(wavevectors)),
                      design_rank=int(np.linalg.matrix_rank(system.X_hat)),
                      relative_clean_weak_residual=float(np.linalg.norm(system.R(target))/np.linalg.norm(system.Y_hat)),
                      relative_noiseless_dense_error=float(np.linalg.norm(wsindy.least_squares(system.X_hat, system.Y_hat)-target)/np.linalg.norm(target)))
    settings = dict(dimension=dimension, spatial_grid=[nx]*dimension, time_samples=nt, time_interval=[0, .5],
                    spatial_observations=len(points), raw_observations=len(points)*nt,
                    spatial_domain="[-pi, pi)^d; periodic", alpha=alpha.tolist(), target=target.tolist(),
                    strong_indices=strong.tolist(), wavevectors=wavevectors.tolist(), initial_phase=phase0.tolist(),
                    initial_mode_amplitude=float(.3*np.sqrt(2/len(wavevectors))), clean_rms=float(rms),
                    weak_equations=system.K, validation=validation,
                    assembly_seconds=assembly_seconds, projection_seconds=projection_seconds)
    return clean, projected_noise, A, full_alpha, target, strong, rms, settings


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), environment=dict(python=platform.python_version(), numpy=np.__version__,
                  scipy=scipy.__version__, platform=platform.platform(),
                  blas={key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}),
                  solver_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ("dimension_experiment.py", "experiment.py", "wsindy.py", "wendy.py", "wendy_mle.py")},
                  cases={}, noise="iid physical-space Gaussian; projected sigma = physical sigma/sqrt(spatial observations)")
    trials = []
    methods = [("WSINDy", wsindy, {})]
    for label, module in (("WENDy", wendy), ("WENDy-MLE", wendy_mle)):
        methods += [(label+" HT", module, dict(sparsity=3)), (f"{label} L1={args.l1:g}", module, dict(l1=args.l1))]
    for dimension in args.dimensions:
        clean, noise, A, alpha, target, strong, rms, settings = data(dimension, seed=args.seed)
        config["cases"][str(dimension)] = settings
        (out/"config.json").write_text(json.dumps(config, indent=2))
        for level in args.noise:
            sigma = level*rms
            start = perf_counter()
            system = wsindy.WeakSystem(clean+sigma*noise, A, wsindy.polynomial_dictionary(1)[1:], alpha)
            common_seconds = settings["assembly_seconds"]+settings["projection_seconds"]+perf_counter()-start
            for name, module, selection in methods:
                start = perf_counter()
                try:
                    result = module.fit(system) if module is wsindy else module.fit(
                        system, sigma=sigma/np.sqrt(settings["spatial_observations"]),
                        maxiter=args.maxiter, time_limit=args.time_limit, **selection)
                except (np.linalg.LinAlgError, FloatingPointError) as exc:
                    result = wsindy.FitResult(np.full_like(target, np.nan), [], perf_counter()-start, 0, False, str(exc))
                dominant = wendy.hard_threshold(result.w, 3)
                strong_target = np.zeros_like(target)
                strong_target[strong] = target[strong]
                error = np.linalg.norm(result.w-target)/np.linalg.norm(target)
                strong_error = np.linalg.norm((result.w-target)[strong])/np.linalg.norm(target[strong])
                row = dict(problem=f"{dimension}D", grid="×".join(map(str, settings["spatial_grid"])),
                           noise=level, sigma=float(sigma), seed=args.seed, method=name,
                           coefficient_error=float(error) if np.isfinite(error) else float("inf"),
                           strong_error=float(strong_error) if np.isfinite(strong_error) else float("inf"),
                           **support_metrics(dominant if np.isfinite(result.w).all() else result.w, strong_target),
                           optimizer_success=result.success, status=result.message, iterations=result.iterations,
                           fit_seconds=result.seconds, total_seconds=common_seconds+result.seconds,
                           w=json.dumps(result.w.tolist()), selected_terms=json.dumps(np.flatnonzero(abs(result.w)>1e-12).tolist()),
                           selected_strong=json.dumps(np.flatnonzero(abs(dominant)>1e-12).tolist()))
                trials.append(row)
                save_csv(out/"trials.csv", [row], append=True)
                print(f"{dimension}D noise={level:g} {name}: hits={row['tp']}/3, E2={error:.3g}, "
                      f"{row['total_seconds']:.2f}s, success={result.success}", flush=True)
    lines = ["# Spatial dimension: 1D and 5D", "",
             "Linear PDE: u_t = c u + Σ_i b_i ∂_i u + Σ_{i≤j} a_ij ∂_i∂_j u, on periodic [−π,π)^d.",
             "c=−1, b₁=−1, a₁₁=1; every other coefficient is +0.001. There are 3 coefficients in 1D and 21 in 5D.",
             "The exact Fourier solution varies in every spatial direction. Both cases use 32,768 spatial samples × 129 times and one paired noise seed.",
             "Both cases use the same linear operator family; the earlier nonlinear 1D results are a separate experiment.",
             "The 1D library is fully active: WSINDy's sparsity criterion selects only one term. Dimension and candidate count change together here.",
             f"W = WENDy; MLE = WENDy-MLE; HT keeps three terms; :{args.l1:g} uses λ={args.l1:g}λ_ref.",
             f"Budget: {args.maxiter} iterations / {args.time_limit:g} seconds per fit.", ""]
    lines += comparison_tables(trials)
    lines += ["Top-3 hits counts dominant coordinates, not a success probability. Strong/Full E₂ are relative coefficient errors, not percentages.",
              "Noise = σ/RMS(clean u), before spatial projection. Time includes Fourier tests, projection, weak assembly, and fitting; excludes data generation.",
              "Optimizer counts fits meeting their stopping criterion; failures remain included. At zero noise, MLE uses the WENDy least-squares/LASSO fallback.",
              "Projection averages physical noise: coefficient noise has standard deviation σ/√32768. Fixed total observations control this averaging across dimensions.",
              "These timings measure the Fourier weak implementation; dense-grid PDE simulation is not timed.",
              "Settings and numerical checks: [config.json](config.json). Raw fits: [trials.csv](trials.csv).", ""]
    (out/"summary.md").write_text("\n".join(lines))
    print(f"Saved {len(trials)} fits to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", nargs="+", type=int, choices=[1, 5], default=[1, 5])
    parser.add_argument("--noise", nargs="+", type=float, default=[0, .01, .05, .1, .2, .5, 1])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--l1", type=float, default=1e-4)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--time-limit", type=float, default=5.)
    parser.add_argument("--output", default="results/spatial_dimension")
    args = parser.parse_args()
    if not np.isfinite(args.noise).all() or min(args.noise) < 0 or args.seed < 0 or args.maxiter < 1 or not np.isfinite([args.l1, args.time_limit]).all() or min(args.l1, args.time_limit) <= 0:
        parser.error("Require finite noise>=0, seed>=0, maxiter>=1 and positive finite L1/time limit")
    run(args)
