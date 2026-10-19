"""Discrete Dantzig paths for physical, possibly indefinite debiased scores.

The optimization never sees the true coefficients. Truth is optional and is used
only to rank the returned, constraint-validated solutions along the lambda path.
SOS1 constraints avoid coefficient bounds or a data-dependent big-M assumption.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

from .regression import gurobi_env


def validate_score(H, b, beta, lam, *, relative_tolerance=2e-8):
    """Validate in physical units, with tolerance relative to the score scale."""
    beta = np.asarray(beta)
    residual = float(np.max(np.abs(b-H@beta)))
    scale = max(float(np.max(np.abs(b))), np.finfo(float).tiny)
    violation = max(0., residual-lam) if np.isfinite(residual) else np.inf
    slack = min(relative_tolerance*scale, .01*lam)+50*np.finfo(float).eps*scale
    return dict(residual_inf=residual, constraint_violation=violation,
                relative_constraint_violation=violation/scale,
                validation_slack=slack,
                validated=bool(np.all(np.isfinite(beta)) and np.isfinite(residual) and np.isfinite(lam)
                               and np.isfinite(scale) and violation <= slack))


def solve_one(H, b, lam, *, time_limit=5., mip_gap=0., start=None):
    """Minimize cardinality, then score infinity residual at that cardinality.

    Both phases have bounded running time. A valid incumbent is retained on a
    timeout and is explicitly distinguished from a certified lexicographic solve.
    A variable substitution and common score rescaling improve numerics without
    changing the physical infinity-norm constraint or the cardinality objective.
    """
    import gurobipy as gp

    H, b = np.asarray(H, dtype=float), np.asarray(b, dtype=float).reshape(-1)
    if H.ndim != 2 or H.shape[0] != len(b) or not np.all(np.isfinite(H)) or not np.all(np.isfinite(b)):
        raise ValueError("A finite score matrix and matching vector are required")
    if not np.isfinite(lam) or lam < 0 or not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("lambda must be nonnegative and time_limit positive")
    p = H.shape[1]
    score_scale = max(float(np.max(np.abs(b))), np.finfo(float).tiny)
    if lam >= np.max(np.abs(b)):
        beta = np.zeros(p)
        return dict(coefficients=beta.tolist(), support=[False]*p, lambda_=float(lam),
                    status="AnalyticZero", primary_status="AnalyticZero", secondary_status="AnalyticZero",
                    optimal=True, cardinality_optimal=True, secondary_optimal=True,
                    cardinality=0, cardinality_bound=0., mip_gap=0., runtime=0.,
                    **validate_score(H, b, beta, lam))
    column_scale = np.max(np.abs(H), axis=0)/score_scale
    column_scale = np.where(column_scale > 0, column_scale, 1.)
    A, y, tolerance = H/column_scale[None, :]/score_scale, b/score_scale, lam/score_scale
    with gp.Model(env=gurobi_env()) as model:
        model.Params.Threads = 1
        model.Params.Seed = 0
        model.Params.TimeLimit = time_limit
        model.Params.MIPGap = mip_gap
        model.Params.FeasibilityTol = 1e-9
        model.Params.IntFeasTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        model.Params.NumericFocus = 2
        model.Params.PreSOS1BigM = 0
        theta = model.addMVar(p, lb=-gp.GRB.INFINITY, name="theta")
        zero = model.addMVar(p, vtype=gp.GRB.BINARY, name="zero")
        radius = model.addVar(lb=0., ub=tolerance, name="score_radius")
        for w, z in zip(theta.tolist(), zero.tolist()):
            model.addSOS(gp.GRB.SOS_TYPE1, [w, z])
        model.addConstr(A@theta-y <= radius)
        model.addConstr(y-A@theta <= radius)
        model.setObjective(p-zero.sum(), gp.GRB.MINIMIZE)
        proposal = linalg.lstsq(A, y, lapack_driver="gelsd")[0] if start is None else np.asarray(start)*column_scale
        theta.Start = proposal
        zero.Start = (proposal == 0).astype(float)
        radius.Start = min(tolerance, float(np.max(np.abs(A@proposal-y))))
        model.optimize()
        primary_status, primary_time = int(model.Status), float(model.Runtime)
        if not model.SolCount:
            return dict(coefficients=None, support=None, lambda_=float(lam),
                        status=f"NoIncumbent:{primary_status}", primary_status=primary_status,
                        optimal=False, cardinality_optimal=False, secondary_optimal=False,
                        validated=False, runtime=primary_time, mip_gap=None)
        first_theta, first_zero = theta.X.copy(), zero.X.copy()
        count = int(np.count_nonzero(first_zero < .5))
        lower = float(model.ObjBound)
        primary_optimal = bool(np.isfinite(lower) and np.ceil(lower-1e-7) >= count)
        model.addConstr(zero.sum() == p-count, name="fixed_cardinality")
        theta.Start, zero.Start = first_theta, first_zero
        model.setObjective(radius, gp.GRB.MINIMIZE)
        model.Params.TimeLimit = max(.2, min(2., time_limit/2))
        model.optimize()
        secondary_status, secondary_time = int(model.Status), float(model.Runtime)
        if model.SolCount:
            values, support = theta.X.copy(), zero.X < .5
        else:
            values, support = first_theta, first_zero < .5
        beta = np.where(support, values/column_scale, 0.)
        second_optimal = bool(secondary_status == gp.GRB.OPTIMAL and model.MIPGap <= 1e-8)
        result = dict(coefficients=beta.tolist(), support=support.tolist(), lambda_=float(lam),
                      status="Optimal" if primary_optimal and second_optimal else "ValidIncumbent",
                      primary_status=primary_status, secondary_status=secondary_status,
                      optimal=primary_optimal and second_optimal, cardinality_optimal=primary_optimal,
                      secondary_optimal=second_optimal, cardinality=int(np.count_nonzero(beta)),
                      cardinality_bound=lower, mip_gap=max(0., (count-lower)/max(1., count)),
                      runtime=primary_time+secondary_time, **validate_score(H, b, beta, lam))
        # A final numerical failure must not discard a valid primary incumbent.
        if not result["validated"]:
            fallback = np.where(first_zero < .5, first_theta/column_scale, 0.)
            check = validate_score(H, b, fallback, lam)
            if check["validated"]:
                result.update(coefficients=fallback.tolist(), support=(first_zero < .5).tolist(),
                              status="PrimaryFallback", secondary_optimal=False, optimal=False, **check)
        return result


def solve_path(H, b, truth=None, *, ratios=None, time_limit=5., mip_gap=0., refinements=1):
    """Oracle-select a finite lambda path; refine neighbors of the best grid point.

    Ratios multiply ||b||_inf. The zero estimator is an exact analytic endpoint.
    Refinement uses truth only for choosing additional lambda values, never inside
    an optimization problem. Failed or physically invalid candidates cannot win.
    """
    H, b = np.asarray(H), np.asarray(b).reshape(-1)
    ratios = np.logspace(-6, 0, 13) if ratios is None else np.asarray(ratios, dtype=float)
    if ratios.ndim != 1 or not len(ratios) or np.any(ratios <= 0) or np.any(ratios > 1):
        raise ValueError("Ratios must lie in (0, 1]")
    if truth is None and (len(ratios) > 1 or refinements):
        raise ValueError("Selecting a path requires truth; fixed-lambda solve_one does not")
    truth = None if truth is None else np.asarray(truth).reshape(-1)
    scale = float(np.max(np.abs(b)))
    records, seen = [], set()

    def add(ratio):
        key = float(ratio)
        if key in seen:
            return
        seen.add(key)
        starts = [r for r in records if r.get("validated") and r.get("coefficients") is not None]
        start = min(starts, key=lambda r: abs(np.log(r["lambda_ratio"]/key)))["coefficients"] if starts else None
        result = solve_one(H, b, scale*key, time_limit=time_limit, mip_gap=mip_gap, start=start)
        result["lambda"] = result.pop("lambda_")
        result["lambda_ratio"] = key
        if truth is not None and result["coefficients"] is not None:
            result["e2"] = float(np.linalg.norm(np.asarray(result["coefficients"])-truth)/np.linalg.norm(truth))
        records.append(result)

    for ratio in sorted(set(ratios.tolist()) | {1.}, reverse=True):
        add(ratio)
    for _ in range(refinements):
        eligible = [r for r in records if r["validated"] and np.isfinite(r.get("e2", 0.))]
        winner = min(eligible, key=lambda r: (r.get("e2", 0.), r["lambda_ratio"]))
        ordered = sorted(seen)
        index = ordered.index(winner["lambda_ratio"])
        neighbors = []
        if index:
            neighbors.append(np.sqrt(ordered[index-1]*ordered[index]))
        elif ordered[index] > 1e-12:
            neighbors.extend([ordered[index]/10, ordered[index]/100])
        if index+1 < len(ordered):
            neighbors.append(np.sqrt(ordered[index]*ordered[index+1]))
        for ratio in neighbors:
            add(ratio)
    valid_indices = [i for i, r in enumerate(records) if r["validated"] and np.isfinite(r.get("e2", 0.))]
    incumbent_index = min(valid_indices, key=lambda i: (records[i].get("e2", 0.), records[i]["lambda_ratio"]))
    # A timed-out, denser incumbent can beat the l0 optimum in coefficient E2.
    # Such a result is useful diagnostically, but is not the estimator of Eq. 9.
    selected_index = min((i for i in valid_indices if records[i]["optimal"]),
                         key=lambda i: (records[i].get("e2", 0.), records[i]["lambda_ratio"]))
    selected = records[selected_index]
    certified = [r for r in records if r["validated"] and r["optimal"]]
    certified_best = min(certified, key=lambda r: r.get("e2", 0.)) if certified else None
    return dict(coefficients=selected["coefficients"], support=selected["support"], selected=selected,
                selected_index=selected_index, path=records, score_scale=scale,
                truth_feasibility_ratio=float(np.max(np.abs(b-H@truth))/scale) if truth is not None and scale else None,
                certified_only_selected=certified_best,
                incumbent_oracle_selected=records[incumbent_index],
                selected_from="certified_lexicographic_candidates",
                boundary_best=bool(selected["lambda_ratio"] in (min(seen), max(seen))),
                runtime=float(sum(r["runtime"] for r in records)))
