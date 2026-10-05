"""Check all PDE/noise/preprocessing combinations before a full LASSO run."""
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from methods import build_wsindy_system, lasso
from experiments import moving_average_observations, trial_seed
from simulation_generation import BENCHMARKS, load_clean_benchmark, observation_instance


def main():
    checks = []
    for name, spec in BENCHMARKS.items():
        clean = load_clean_benchmark(name, data_dir=ROOT/"data", download=False)
        path = ROOT/("benchmark_results/consistency/raw/results_source/methods.py"
            if spec.suite == "consistency" else "benchmark_results/authors/results_source/methods.py")
        module_name = "reference_"+name
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        reference = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = reference
        module_spec.loader.exec_module(reference)
        for method in ("wsindy", "filtered-wsindy"):
            for ratio in (0., .2, .5, .75, 1.):
                observed = observation_instance(clean, ratio, trial_seed(0, name, ratio, 0)).u_observed
                if method == "filtered-wsindy":
                    observed, _ = moving_average_observations(observed, spec.half_widths, spec.max_degree,
                        support_rule="volume" if spec.suite == "consistency" else "per_axis")
                settings = dict(library_terms=spec.library(), lhs_components=spec.lhs_components,
                    lhs_time_order=spec.lhs_time_order, half_widths=spec.half_widths, strides=spec.strides,
                    test_degrees=spec.degrees, rescale=spec.suite != "consistency", state_scale_rule="authors")
                if spec.suite == "consistency":
                    settings["test_function"] = spec.test_function
                system = build_wsindy_system(observed, clean.spatial_grid, clean.time, **settings)
                if ratio == .2:
                    settings["library_terms"] = tuple(reference.LibraryTerm(t.powers, t.derivative,
                        t.kind, t.component, t.frequency) for t in spec.library())
                    old = reference.build_wsindy_system(observed, clean.spatial_grid, clean.time, **settings)
                    for field in ("G", "b", "coefficient_scales", "state_scales", "coordinate_scales"):
                        np.testing.assert_array_equal(getattr(system, field), getattr(old, field))
                start = perf_counter()
                result = lasso(system.G, system.b, coefficient_scales=system.coefficient_scales)
                checks.append(dict(name=name, method=method, noise_ratio=ratio,
                    weak_matches_saved_source=True if ratio == .2 else None,
                    kkt_max=float(result.kkt_errors.max()), solver_seconds=perf_counter()-start,
                    solver_messages=len(result.solver_warnings)))
                (ROOT/"benchmark_results/lasso/preflight.json").write_text(json.dumps(checks, indent=2)+"\n")
                print(checks[-1], flush=True)
    print("All 80 polynomial preflight combinations passed.", flush=True)


if __name__ == "__main__":
    main()
