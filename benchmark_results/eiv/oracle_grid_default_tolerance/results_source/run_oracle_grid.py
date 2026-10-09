"""Small, explicitly oracle-selected Section 6 grid; writes only a separate study.

Truth is used to rank converged candidates by coefficient E2. This is a
diagnostic of the estimator, not a deployable tuning rule or benchmark score.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_sanity as base
ROOT, SOURCE = base.ROOT, base.SOURCE
sys.path.insert(0, str(SOURCE))
import numpy as np
from scipy import linalg
from eiv_study import conic_eiv, fit_eiv
from experiments import trial_seed
from simulation_generation import BENCHMARKS, load_clean_benchmark, observation_instance

MU_RATIOS = np.r_[0., 10.**np.arange(-8, 3)].tolist()
TAU_RATIOS = np.r_[0., 10.**np.arange(-6, 0), .2, .4, .6, .8, .95].tolist()
VALIDATION_TOLERANCE = 1e-6
SOLVER_TOLERANCE = 1e-8
SUPPORT_TOLERANCE = 1e-7


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmarks', nargs='+', choices=base.SCALAR_BENCHMARKS,
                   default=list(base.SCALAR_BENCHMARKS))
    p.add_argument('--noise-ratios', nargs='+', type=float, default=list(base.NOISE_RATIOS))
    p.add_argument('--trials', type=int, default=1)
    p.add_argument('--jobs', type=int, default=3)
    p.add_argument('--solver-tolerance', type=float, default=SOLVER_TOLERANCE)
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'benchmark_results/eiv/oracle_grid_default_tolerance')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args(argv)
    if args.trials < 1 or args.jobs < 1:
        p.error('Trials and jobs must be positive.')
    if not np.isfinite(args.solver_tolerance) or args.solver_tolerance <= 0:
        p.error('Solver tolerance must be finite and positive.')
    if any(not np.isfinite(x) or not 0 <= x <= 1 for x in args.noise_ratios):
        p.error('Noise ratios must be finite and in [0, 1].')
    if len(set(args.benchmarks)) != len(args.benchmarks) or len(set(args.noise_ratios)) != len(args.noise_ratios):
        p.error('Benchmarks and noise ratios must be unique.')
    args.data_dir, args.output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    study_root = ROOT / 'benchmark_results/eiv'
    if (not args.output_dir.is_relative_to(study_root) or args.output_dir == study_root
            or args.output_dir.is_relative_to(study_root / 'scalar_sanity')):
        p.error('Use a separate output folder under benchmark_results/eiv, outside scalar_sanity.')
    return args


def candidate_validation(result, y_inf, mu, tau):
    """Independently check physical feasibility and returned objective.

    The residual tolerance is relative to y, with NO unit-size floor. This
    matters for the bump-test systems, whose responses are below 1e-8.
    """
    d = result.diagnostics
    reasons = []
    if not result.success:
        reasons.append('solver_not_successful')
    beta = np.asarray(result.coefficients)
    values = [*beta, result.t, result.objective, d.get('objective_dual', np.nan),
              d.get('residual_violation', np.nan), d.get('cone_violation', np.nan)]
    if not np.all(np.isfinite(values)):
        return dict(eligible=False, rejection_reasons=reasons + ['nonfinite_output'])
    eps = VALIDATION_TOLERANCE
    residual_limit = eps * y_inf
    cone_limit = eps * max(1., linalg.norm(beta), abs(result.t))
    dual = d['objective_dual']
    objective_limit = eps * max(1., abs(result.objective), abs(dual))
    if d['residual_violation'] > residual_limit:
        reasons.append('physical_residual_violation')
    if d['cone_violation'] > cone_limit:
        reasons.append('cone_violation')
    gap = abs(result.objective - dual)
    if gap > objective_limit:
        reasons.append('returned_objective_dual_gap')
    upper = max(0., y_inf - tau) / mu if mu > 0 else None
    if upper is not None and result.objective > upper + eps * max(1., upper):
        reasons.append('worse_than_feasible_zero_beta')
    return dict(eligible=not reasons, rejection_reasons=reasons,
                residual_violation_over_y_inf=d['residual_violation'] / y_inf,
                residual_limit=residual_limit, cone_limit=cone_limit,
                returned_objective_dual_gap=gap, objective_gap_limit=objective_limit,
                feasible_zero_beta_objective=upper)


def select_best(candidates):
    """Truth E2 selects only independently eligible finite candidates."""
    good = [r for r in candidates if r.get('eligible') and
            r.get('e2') is not None and np.isfinite(r['e2'])]
    # Candidate id is a deterministic tie-break; support and objective do not tune.
    return min(good, key=lambda r: (r['e2'], r['candidate_id'])) if good else None


def neighborhood(value, coarse):
    """Seven points between neighboring coarse values, retaining zero if present."""
    a = np.asarray(coarse)
    i = int(np.argmin(abs(a-value)))
    low, high = a[max(0, i-1)], a[min(len(a)-1, i+1)]
    if low == 0:
        points = np.r_[0., np.geomspace(high / 1000., high, 6)]
    else:
        points = np.geomspace(low, high, 7)
    return np.unique(points).tolist()


def run_case(args, manifest_id, name, ratio, trial):
    start = perf_counter()
    key = f'{name}_noise{ratio:g}_trial{trial}'
    candidate_file = args.output_dir / 'candidates' / f'{key}.jsonl'
    seed = trial_seed(0, name, ratio, trial)
    record = dict(name=name, noise_ratio=ratio, trial=trial, seed=seed,
                  protocol_id=manifest_id, state='failed', best=None,
                  candidate_file=str(candidate_file.relative_to(args.output_dir)))
    try:
        clean = load_clean_benchmark(name, data_dir=args.data_dir, download=False)
        spec = clean.benchmark
        data = observation_instance(clean, ratio, seed)
        terms = spec.library()
        reference, system, calibration = fit_eiv(
            data.u_observed[0], data.spatial_grid, data.time,
            library_terms=terms, half_widths=spec.half_widths, strides=spec.strides,
            test_degrees=spec.degrees if spec.test_function == 'polynomial' else None,
            test_function=spec.test_function, lhs_time_order=spec.lhs_time_order,
            delta=.05, lam=1., sigma2=None, solver_tolerance=args.solver_tolerance,
            support_tolerance=SUPPORT_TOLERANCE, equilibrate=False)
        y = system.b[:, 0]
        y_inf = float(linalg.norm(y, np.inf))
        if y_inf == 0:
            raise ValueError('A positive response norm is required for the relative grid.')
        truth = spec.truth(terms)[:, 0]
        record.update(
            observation_sha256=[base.array_digest(v) for v in data.u_observed],
            G_sha256=base.array_digest(system.G), b_sha256=base.array_digest(system.b),
            G_shape=system.G.shape, y_inf=y_inf, X_max=float(np.max(abs(system.G))),
            coefficient_scales=system.coefficient_scales, state_scales=system.state_scales,
            coordinate_scales=system.coordinate_scales, sigma2=calibration['sigma2'],
            calibration=calibration, truth_coefficients=truth,
            term_labels=[t.label(spec.component_names) for t in terms],
            truth_residual_inf=float(linalg.norm(y-system.G@truth, np.inf)),
            formula_reference=dict(status=reference.status, mu=calibration['mu'],
                tau=calibration['tau'], **base.score(reference.coefficients, reference.support, truth)))
        candidates, seen = [], set()
        # A whole incomplete case is rerun on resume; truncate its incomplete trace.
        with candidate_file.open('w') as stream:
            def evaluate(mu_ratio, tau_ratio, phase):
                pair = (float(mu_ratio), float(tau_ratio))
                if pair in seen:
                    return
                seen.add(pair)
                started = perf_counter()
                mu, tau = mu_ratio*y_inf, tau_ratio*y_inf
                item = dict(candidate_id=len(candidates), phase=phase, mu=float(mu), tau=float(tau),
                            mu_over_y_inf=float(mu_ratio), tau_over_y_inf=float(tau_ratio),
                            eligible=False, status='exception')
                try:
                    fit = conic_eiv(system.G, y, mu=mu, tau=tau, lam=1.,
                                    solver_tolerance=args.solver_tolerance,
                                    support_tolerance=SUPPORT_TOLERANCE, equilibrate=False)
                    validation = candidate_validation(fit, y_inf, mu, tau)
                    item.update(status=fit.status, solver_success=fit.success,
                        coefficients=fit.coefficients, support=fit.support, t=fit.t,
                        objective=fit.objective, diagnostics=fit.diagnostics, **validation)
                    if validation['eligible']:
                        item.update(base.score(fit.coefficients, fit.support, truth))
                except Exception as error:
                    item.update(error_type=type(error).__name__, error=str(error),
                                rejection_reasons=['exception'])
                item['runtime'] = perf_counter()-started
                item = base.serializable(item)
                candidates.append(item)
                stream.write(json.dumps(item, allow_nan=False, sort_keys=True)+'\n')
                stream.flush()

            for mu_ratio in MU_RATIOS:
                for tau_ratio in TAU_RATIOS:
                    evaluate(mu_ratio, tau_ratio, 'coarse')
            coarse_best = select_best(candidates)
            if coarse_best is not None:
                for mu_ratio in neighborhood(coarse_best['mu_over_y_inf'], MU_RATIOS):
                    for tau_ratio in neighborhood(coarse_best['tau_over_y_inf'], TAU_RATIOS):
                        evaluate(mu_ratio, tau_ratio, 'refine')
        best = select_best(candidates)
        record.update(state='complete', candidate_count=len(candidates),
            coarse_best=coarse_best, best=best, eligible_count=sum(r['eligible'] for r in candidates),
            status_counts=dict(Counter(r['status'] for r in candidates)),
            rejection_counts=dict(Counter(reason for r in candidates for reason in r.get('rejection_reasons', []))),
            exact_support_candidates=sum(r.get('exact_support', False) for r in candidates),
            refined=coarse_best is not None,
            best_on_upper_grid_boundary=bool(best and (best['mu_over_y_inf'] == max(MU_RATIOS)
                                                       or best['tau_over_y_inf'] == max(TAU_RATIOS))))
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error))
    record['runtime'] = perf_counter()-start
    record = base.serializable(record)
    base.write_json(args.output_dir / 'cases' / f'{key}.json', record)
    return record


def prepare(args):
    # Reuse the verified dataset/source manifest builder, without its fit protocol.
    for key, value in dict(lam=1., delta=.05,
                           support_tolerance=SUPPORT_TOLERANCE).items():
        setattr(args, key, value)
    manifest, sources = base.protocol(args)
    sources['run_oracle_grid.py'] = Path(__file__)
    manifest.update(schema_version=2, method='Section 6 oracle tau/mu grid, lambda=1',
        description=__doc__, truth_use='Oracle minimum coefficient E2 among eligible candidates; diagnostic only',
        mu_over_y_inf=MU_RATIOS, tau_over_y_inf=TAU_RATIOS,
        grid_units='mu=q_mu*||y||inf; tau=q_tau*||y||inf; physical beta reference unit is one',
        refinement='One 7x7 neighborhood around best eligible coarse point, bounded by adjacent coarse points; duplicates skipped. No refinement if none eligible.',
        validation_tolerance=VALIDATION_TOLERANCE,
        validation='Solver success, residual violation <= 1e-6*||y||inf without a unit floor; cone, returned-objective/dual gap and feasible-zero-beta objective checks.',
        reference='Formula tolerances at delta=.05, using estimated variance and observed M/M1; not ranked as grid candidates.',
        source_sha256={name: base.sha256(path) for name, path in sources.items()})
    manifest.pop('protocol_id')
    manifest = base.serializable(manifest)
    manifest['protocol_id'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    path = args.output_dir / 'results.manifest.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['protocol_id'] != manifest['protocol_id'] or not args.resume:
            raise ValueError('Existing study requires --resume and identical source/protocol; otherwise use a fresh directory.')
        return saved
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Refusing a nonempty output directory without a matching manifest.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ('cases', 'candidates', 'results_source'):
        (args.output_dir / name).mkdir()
    for name, source in sources.items():
        shutil.copyfile(source, args.output_dir / 'results_source' / name)
    # Hash all already-existing benchmark data/reports before this study runs.
    protected = [ROOT / 'results.md', *sorted((ROOT / 'benchmark_results').rglob('*'))]
    hashes = {str(p.relative_to(ROOT)): base.sha256(p) for p in protected
              if p.is_file() and not p.is_relative_to(args.output_dir)
              and p.suffix != '.pyc' and p.name not in ('README.md', 'run_oracle_grid.py')}
    base.write_json(args.output_dir / 'protected_artifacts.json', hashes)
    manifest['created_at'] = datetime.now(timezone.utc).isoformat()
    base.write_json(path, manifest)
    return manifest


EQUATIONS = {
    'IB': r'$u_t=-\frac12\partial_x(u^2)$.',
    'KdV': r'$u_t=-\frac12\partial_x(u^2)-u_{xxx}$.',
    'KS': r'$u_t=-\frac12\partial_x(u^2)-u_{xx}-u_{xxxx}$.',
    'HKS': r'$u_t=u_{xxxx}+0.75u_{xxxxxx}-0.5\partial_x(u^2)+0.1\partial_x^3(u^2)$.',
    'VBG': r'$u_t=0.01u_{xx}-0.5\partial_x(u^2)-u^3+2u^2+1$.',
}


def refresh(folder, manifest, records, state):
    order = {name: i for i, name in enumerate(manifest['benchmark_order'])}
    records = sorted(records, key=lambda r: (order[r['name']], r['noise_ratio'], r['trial']))
    best = [r['best'] for r in records if r['best'] is not None]
    requested = len(manifest['benchmark_order'])*len(manifest['noise_ratios'])*manifest['trials']
    summary = dict(state=state, completed=len(records), requested=requested,
        cases_with_eligible_candidates=len(best), exact_support_selected=sum(r['exact_support'] for r in best),
        candidate_count=sum(r.get('candidate_count', 0) for r in records),
        eligible_count=sum(r.get('eligible_count', 0) for r in records),
        status_counts=dict(sum((Counter(r.get('status_counts', {})) for r in records), Counter())),
        updated_at=datetime.now(timezone.utc).isoformat())
    base.write_json(folder / 'results.summary.json', summary)
    base.write_json(folder / 'results.status.json', summary)
    lines = ['# Section 6 conic estimator: oracle μ/τ sanity grid', '',
        f'Status: **{state}**, {len(records)}/{requested} cases; '
        f'{manifest["trials"]} trial per PDE/noise setting, λ=1.', '',
        '**Oracle diagnostic:** μ and τ minimize coefficient E2 against the known true equation. '
        'These are best values among the searched candidates, not globally optimal hyperparameters '
        'or a data-driven tuning procedure. Do not compare these scores as a fair tuned benchmark.', '',
        'Variance is estimated from each observed field (also at zero injected noise). The same '
        'noise seeds, full polynomial libraries, weak kernels and Gaussian correction as the formula '
        'sanity study are used. State, coordinates, columns and response retain physical units; '
        'rescale=False and solver equilibration=False. No coefficient thresholding or refit.', '',
        '## Search and numerical checks', '',
        '- Coarse μ/‖y‖∞: 0 and 10^q for integer q = −8,…,2 (12 values).',
        '- Coarse τ/‖y‖∞: 0, 10^q for integer q = −6,…,−1, 0.2, 0.4, 0.6, 0.8, 0.95 (12 values).',
        '- Each case has 144 coarse solves, then up to 49 points near the best eligible coarse point '
        '(duplicates skipped). No refinement if the coarse search has no eligible candidate.',
        '- Grid ratios only construct physical μ and τ; the solver arrays and objective are not normalized. '
        'The μ reference assumes a coefficient of one in the existing physical units.',
        f'- Clarabel tolerance {manifest["solver_tolerance"]:g}, maximum 200 iterations. Eligibility requires solver success plus '
        'physical residual violation ≤ 1e-6‖y‖∞ (no unit-size floor), cone feasibility, returned '
        'objective/dual gap consistency and the feasible-zero-coefficient objective bound.',
        '- Other validation limits use 1e-6 times max(1, relevant objective or coefficient norm). '
        'Raw solver status and each rejection reason remain in the candidate JSONL.',
        '- E2 = ‖β−β*‖₂/‖β*‖₂ uses unmodified coefficients. Exact support uses |β|>1e-7 only '
        'for its separate support mask. E2 alone selects parameters; ties use candidate order.',
        '- Formula μδ,τδ at δ=0.05 are reference values only. τ≥‖y‖∞ makes β=t=0 globally optimal. '
        'For μ>0 even τ<‖y‖∞ permits β=0 at t=(‖y‖∞−τ)/μ; a nonzero solution is not guaranteed.',
        '- A positive μ is always mathematically feasible; any PrimalInfeasible status there is '
        'a numerical failure. At μ=0 an infeasibility status may reflect genuine infeasibility.',
        '- Physical-unit L1/L2 penalties depend on dictionary units: large high-degree columns can '
        'fit a response with tiny coefficients. An eligible fit with E2≈1 and empty support can '
        'contain tiny nonzero coefficients; it is different from a failed solve (shown as —).',
        '- One trial per setting is a sanity check, not an estimate of recovery probability. '
        'HKS and VBG use the repository’s declared resimulations of the consistency-paper equations.', '',
        f'Total candidates: {summary["candidate_count"]}; eligible: {summary["eligible_count"]}. '
        f'Cases with an eligible fit: {len(best)}/{len(records)}; selected exact support: '
        f'{summary["exact_support_selected"]}/{len(best)} eligible cases.', '']
    for name in manifest['benchmark_order']:
        lines += [f'## {name}', '', 'True equation: '+EQUATIONS[name], '',
            '| Noise | Trial | μ selected | τ selected | E2 oracle | Exact support | Eligible / tried | Formula E2 |',
            '|---:|---:|---:|---:|---:|---|---:|---:|']
        group = [r for r in records if r['name'] == name]
        for r in group:
            b = r['best'] or {}
            vals = [r['noise_ratio'], r['trial'], b.get('mu'), b.get('tau'), b.get('e2'),
                    b.get('exact_support'), f'{r.get("eligible_count", 0)} / {r.get("candidate_count", 0)}',
                    r.get('formula_reference', {}).get('e2')]
            lines.append('| '+' | '.join(base.fmt(v) for v in vals)+' |')
        lines += ['', '**Per-case diagnostics**', '',
            '| Noise | ‖y‖∞ | max absolute X entry | σ̂² | μδ reference | τδ reference | Selected τ/‖y‖∞ |',
            '|---:|---:|---:|---:|---:|---:|---:|']
        for r in group:
            ref = r.get('formula_reference', {})
            vals = [r['noise_ratio'], r.get('y_inf'), r.get('X_max'), r.get('sigma2'),
                    ref.get('mu'), ref.get('tau'), (r['best'] or {}).get('tau_over_y_inf')]
            lines.append('| '+' | '.join(base.fmt(v) for v in vals)+' |')
        lines.append('')
        for r in group:
            statuses = ', '.join(f'{s}: {n}' for s, n in sorted(r.get('status_counts', {}).items()))
            note = ' No eligible solution; E2 is not reported.' if r['best'] is None else ''
            if r.get('best_on_upper_grid_boundary'):
                note += ' Selected point is at an upper grid boundary; the range may limit the result.'
            if r.get('error'):
                note += f' Case error: {r["error"]}'
            lines.append(f'- Noise {r["noise_ratio"]:g}, trial {r["trial"]}: {statuses}.{note}')
        lines.append('')
    lines += ['## Reproducibility', '',
        '[Protocol and hashes](results.manifest.json); [summary](results.summary.json); '
        '[preservation checks](verification.json). `cases/` contains selected coefficients, truth, '
        'correction parameters and reference results. `candidates/` retains every solve and failure. '
        '`results_source/` contains the frozen sources. To rerun frozen sources from the repo root, '
        f'run `python {folder.relative_to(ROOT)}/results_source/run_oracle_grid.py '
        '--output-dir benchmark_results/eiv/oracle_grid_rerun`.', '',
        'The repository root `results.md` and previous sanity/benchmark outputs are preserved.', '']
    (folder / 'results.md').write_text('\n'.join(lines))


def main(argv=None):
    args = arguments(argv)
    if args.report_only:
        manifest = json.loads((args.output_dir / 'results.manifest.json').read_text())
    else:
        manifest = prepare(args)
    records = [json.loads(p.read_text()) for p in sorted((args.output_dir / 'cases').glob('*.json'))]
    keys = {(r['name'], r['noise_ratio'], r['trial']) for r in records}
    expected = {(n, q, i) for n in manifest['benchmark_order'] for q in manifest['noise_ratios']
                for i in range(manifest['trials'])}
    if len(keys) != len(records) or not keys <= expected or any(r['protocol_id'] != manifest['protocol_id'] for r in records):
        raise ValueError('Case records do not match this protocol.')
    if args.report_only:
        refresh(args.output_dir, manifest, records, 'complete' if keys == expected else 'partial')
        return 0
    refresh(args.output_dir, manifest, records, 'running')
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_case, args, manifest['protocol_id'], n, q, i)
                   for n in args.benchmarks for q in args.noise_ratios for i in range(args.trials)
                   if (n, q, i) not in keys]
        for future in as_completed(futures):
            r = future.result()
            records.append(r)
            refresh(args.output_dir, manifest, records, 'running')
            b = r['best'] or {}
            print(f'{r["name"]} noise={r["noise_ratio"]:g} trial={r["trial"]}: '
                  f'eligible={r.get("eligible_count", 0)}/{r.get("candidate_count", 0)}, '
                  f'E2={base.fmt(b.get("e2"))}, mu={base.fmt(b.get("mu"))}, tau={base.fmt(b.get("tau"))}', flush=True)
    protected = json.loads((args.output_dir / 'protected_artifacts.json').read_text())
    changed = [p for p, digest in protected.items() if not (ROOT / p).exists() or base.sha256(ROOT / p) != digest]
    verification = dict(protected_artifacts=len(protected), changed=changed,
        main_results_unchanged='results.md' in protected and 'results.md' not in changed,
        source_snapshots_verified=all(base.sha256(args.output_dir / 'results_source' / p) == h
                                      for p, h in manifest['source_sha256'].items()),
        cases=len(records), expected_cases=len(expected),
        all_cases_finished=all(r['state'] == 'complete' for r in records) and len(records) == len(expected),
        all_physical_scales_one=all(all(np.all(np.asarray(r.get(field, [np.nan])) == 1.)
            for field in ('coefficient_scales', 'state_scales', 'coordinate_scales')) for r in records))
    base.write_json(args.output_dir / 'verification.json', verification)
    refresh(args.output_dir, manifest, records, 'complete' if verification['all_cases_finished'] else 'completed_with_case_errors')
    if changed:
        raise RuntimeError(f'Protected files changed: {changed}')
    return 0 if verification['all_cases_finished'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
