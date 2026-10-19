"""Identification trials, one JSONL record per (PDE, method, noise ratio, trial).

    python run.py KS NLS --method wsindy filtered --trials 100 --workers 4
    python run.py IB --method debiased debiased-lasso --split .3 --out benchmark_results/split-0.3

Records go to <out>/<method>/<PDE>.jsonl; completed trials are skipped on rerun.
A method is a uhat estimator, the state for each side of the weak system, and a
regression; add new combinations to METHODS. Debiased methods estimate uhat from a
fraction --split of the time points and fit the debiased loss on the other times only.
"""
from __future__ import annotations
import os
for _variable in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")   # many small least-squares fits per process
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import json
from pathlib import Path
import subprocess
from time import perf_counter
import numpy as np
from pdes import PDES, load, metrics, observe, trial_seed
from wsindy import (EIV, BestSubset, Dantzig, LassoPath, MSTLS, MovingAverage, SplitPilot, SplitSystem, StrongSystem, WeakSystem,
                    estimate, fit)


@dataclass(frozen=True)
class Method:
    estimator: object = None   # uhat: name, our estimator, PySINDy smoother or f(U, grid)
    X: str = "U"               # state for the dictionary side: "U" or "uhat"
    Y: str = "U"               # state for the time-derivative side
    regression: object = "mstls"
    system: type = WeakSystem  # or StrongSystem (pointwise SINDy / PDE-FIND)


@dataclass
class OraclePilot(SplitPilot):
    """Sanity check: uhat = u* + a (SplitPilot - u*) at the regression times (NaN at the training times), with
    the largest a <= 1 such that dz * sum_i |uhat(z_i) - u*(z_i)|^2 <= n^-rate over the n regression nodes,
    dz = |Omega|/n. rate=None gives uhat = u*."""
    rate: float | None = None

    def __call__(self, U, grid, pde):
        pilot, info = super().__call__(U, grid, pde)
        clean = load(pde.name)[0]
        error = pilot-clean   # NaN at the training times
        n = int(np.count_nonzero(~np.isnan(error[..., 0])))
        energy = float(np.prod([g[-1]-g[0] for g in grid])*np.nanmean(np.sum(error**2, axis=-1)))
        a = 0. if self.rate is None or not energy else min(1., np.sqrt(n**-self.rate/energy))
        return clean+a*error, dict(info, oracle_rate=self.rate, oracle_scale=a, oracle_energy=a*a*energy,
                                   regression_nodes=n)


# Best-subset sparsity r per PDE and equation: the true r, as in the PDE experiments of arXiv:2206.00176.
TRUE_R = {name: tuple(len(equation) for equation in pde.truth) for name, pde in PDES.items()}

METHODS = {
    "sindy": Method(system=StrongSystem),
    "wsindy": Method(),
    "filtered": Method(MovingAverage(), "uhat", "uhat"),
    "debiased": Method(SplitPilot(), "uhat", "U", MSTLS(loss="debiased"), SplitSystem),
    "lasso": Method(regression="lasso"),
    "filtered-lasso": Method(MovingAverage(), "uhat", "uhat", "lasso"),
    "debiased-lasso": Method(SplitPilot(), "uhat", "U", LassoPath(loss="debiased"), SplitSystem),
    "eiv": Method(regression=EIV(oracle=True, normalize=True)),   # sanity check: oracle tau, mu; lam grid
    "eiv-rownorm": Method(regression=EIV(oracle=True, normalize=True, row_normalize=True)),
    "best-subset": Method(regression=BestSubset(r=TRUE_R)),
    "filtered-best-subset": Method(MovingAverage(), "uhat", "uhat", BestSubset(r=TRUE_R)),
    "debiased-best-subset": Method(SplitPilot(), "uhat", "U", BestSubset(r=TRUE_R, loss="debiased"), SplitSystem),
    # Sanity check: the debiased methods with the true state as the pilot (oracle-only information).
    "oracle-debiased": Method(OraclePilot(), "uhat", "U", MSTLS(loss="debiased"), SplitSystem),
    "oracle-debiased-lasso": Method(OraclePilot(), "uhat", "U", LassoPath(loss="debiased"), SplitSystem),
    "oracle-debiased-best-subset": Method(OraclePilot(), "uhat", "U", BestSubset(r=TRUE_R, loss="debiased"), SplitSystem),
    # Discrete Dantzig selector (Eqs. 9-10) with the oracle lambda; OraclePilot(rate=.3) has
    # dz * sum |uhat - u*|^2 <= n^-0.3 = o(n^-1/4).
    "dantzig": Method(SplitPilot(), "uhat", "U", Dantzig(), SplitSystem),
    "oracle-dantzig": Method(OraclePilot(rate=.3), "uhat", "U", Dantzig(), SplitSystem),
    "oracle-plugin-dantzig": Method(OraclePilot(rate=.3), "uhat", "uhat", Dantzig(loss="ls"), SplitSystem),
    "wsindy-dantzig": Method(regression=Dantzig(loss="ls")),
}


def with_split(method, ratio):
    """The method with a split estimator that uses a fraction `ratio` of the time points."""
    if not isinstance(method.estimator, SplitPilot):
        return method
    return replace(method, estimator=replace(method.estimator, ratio=ratio))


def run_trial(name, method_name, ratio, trial, method=None):
    pde, method = PDES[name], method or METHODS[method_name]
    clean, grid = load(name)
    seed = trial_seed(name, ratio, trial)
    U, noise_std = observe(pde, clean, ratio, seed)
    start = perf_counter()
    uhat, info = estimate(U, grid, pde, method.estimator) if "uhat" in (method.X, method.Y) else (U, {})
    states = dict(U=U, uhat=uhat)
    system = method.system(pde, U, grid, X=states[method.X], Y=states[method.Y])
    info = dict(info, **getattr(system, "sampling_info", {}))
    system.clean, system.noise_std = clean, noise_std   # oracle-only information (sanity checks)
    result = fit(system, method.regression)
    runtime = perf_counter()-start
    truth = pde.truth_matrix(system.terms)
    support = result.info.pop("support", None)
    support = None if support is None else np.asarray(support)
    return dict(name=name, noise_ratio=ratio, trial=trial, seed=seed, method=method_name, noise_std=noise_std,
                **metrics(result.coef, truth, support), runtime=runtime,
                equations=[metrics(result.coef[e], truth[e], None if support is None else support[e])
                           for e in range(len(truth))],
                coefficients=result.coef.tolist(), estimator=info, **result.info)


def _chunk(name, method_name, method, ratio, trials):
    return [run_trial(name, method_name, ratio, trial, method) for trial in trials]


def _config(method, args):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=Path(__file__).parent).stdout.strip()
    return dict(method=repr(method), noise_ratios=args.noise, trials=args.trials, commit=commit,
                pdes={name: dict(terms=[t.label for t in PDES[name].dictionary_terms()[1]],
                                 lhs=list(PDES[name].lhs), tests=repr(PDES[name].tests)) for name in args.pdes})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdes", nargs="+", choices=tuple(PDES))
    parser.add_argument("--method", nargs="+", choices=tuple(METHODS), default=["wsindy"])
    parser.add_argument("--noise", nargs="+", type=float, default=[0., .2, .5, .75, 1.])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--split", type=float, default=.5, help="debiased methods: fraction of time points for uhat")
    parser.add_argument("--out", type=Path, default=Path("benchmark_results/runs"))
    args = parser.parse_args()
    tasks = []
    for method_name in args.method:
        method = with_split(METHODS[method_name], args.split)
        directory = args.out/method_name
        directory.mkdir(parents=True, exist_ok=True)
        config = directory/"config.json"
        if config.exists() and json.loads(config.read_text())["method"] != repr(method):
            raise SystemExit(f"{directory} holds a different method definition; choose another --out.")
        config.write_text(json.dumps(_config(method, args), indent=2)+"\n")
        for name in args.pdes:
            path, done = directory/f"{name}.jsonl", set()
            if path.exists():
                with path.open() as source:
                    done = {(r["noise_ratio"], r["trial"]) for r in map(json.loads, source)}
            for ratio in args.noise:
                todo = [t for t in range(args.trials) if (ratio, t) not in done]
                tasks.extend((name, method_name, method, ratio, todo[i:i+10]) for i in range(0, len(todo), 10))
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(_chunk, *task): task for task in tasks}
        for count, future in enumerate(as_completed(futures), 1):
            name, method_name, _, ratio, trials = futures[future]
            with (args.out/method_name/f"{name}.jsonl").open("a") as sink:
                sink.writelines(json.dumps(record, allow_nan=False)+"\n" for record in future.result())
            print(f"[{count}/{len(tasks)}] {method_name} {name} noise {ratio:g} trials {trials[0]}-{trials[-1]}", flush=True)


if __name__ == "__main__":
    main()
