"""Original WSINDy paper datasets and Eq. 5.1 observation instances.

Use authors' U_exact arrays without interpolation, subsampling, or replacing
initial conditions. Metadata follows arXiv:2007.02848v3 with the author's
archived RD library (degree 5, derivative order 4; 181 candidates).
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import md5
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import numpy as np
from scipy.io import loadmat
from scipy import fft
from methods import LibraryTerm, polynomial_library

DATA_URL = "https://zenodo.org/records/20787783/files/"
DEFAULT_DATA_DIR = Path(__file__).parent / "data"
# Preserve saved noise instances when an experiment is removed. SG's former
# index remains reserved; SG is not an active benchmark.
BENCHMARK_SEED_ORDER = ("IB", "KdV", "KS", "NLS", "SG", "RD", "NS", "HKS", "VBG")


@dataclass(frozen=True)
class Benchmark:
    name: str
    filename: str
    checksum: str
    shape: tuple[int, ...]
    component_names: tuple[str, ...]
    lhs_components: tuple[int, ...]
    lhs_time_order: int
    half_widths: tuple[int, ...]
    strides: tuple[int, ...]
    degrees: tuple[int, ...]
    max_degree: int
    max_derivative: int
    true_coefficients: tuple[dict[LibraryTerm, float], ...]
    archive_component_order: tuple[int, ...] | None = None
    suite: str = "original"
    noise_rule: str = "rms"
    test_function: str = "polynomial"

    def library(self):
        n, spatial = len(self.component_names), len(self.shape)-1
        if self.name != "NS":
            return polynomial_library(n, spatial, self.max_degree, self.max_derivative)
        # Table 3: degree <= 2 at order zero; degree <= 3 with omega power > 0
        # at positive spatial derivative orders. Velocity fields are observed
        # auxiliary components, not additional evolution equations.
        zero = polynomial_library(3, 2, 2, 0)
        differentiated = polynomial_library(3, 2, 3, 2)
        return zero + tuple(t for t in differentiated if any(t.derivative) and t.powers[0] > 0)

    def truth(self, terms=None):
        terms = self.library() if terms is None else tuple(terms)
        missing = set().union(*self.true_coefficients)-set(terms)
        if missing:
            raise ValueError(f"Library omits true terms: {missing}")
        return np.array([[equation.get(t, 0.0) for equation in self.true_coefficients] for t in terms])


def _term(powers, spatial, kind="poly"):
    return LibraryTerm(tuple(powers), tuple(spatial)+(0,), kind)


BENCHMARKS = {
    "IB": Benchmark("IB", "burgers.mat", "3ccddfa4fe4173140f27e9a404b252f4",
        (256, 256), ("u",), (0,), 1, (60, 60), (5, 5), (7, 7), 6, 6,
        ({_term((2,), (1,)): -0.5},)),
    "KdV": Benchmark("KdV", "KdV.mat", "aa526a0d61c026d266b6c1127f9350b2",
        (400, 601), ("u",), (0,), 1, (45, 80), (8, 12), (8, 7), 6, 6,
        ({_term((2,), (1,)): -0.5, _term((1,), (3,)): -1.0},)),
    "KS": Benchmark("KS", "KS.mat", "01208398e459323f4db7401083998e0a",
        (256, 301), ("u",), (0,), 1, (23, 22), (5, 6), (10, 10), 6, 6,
        ({_term((2,), (1,)): -0.5, _term((1,), (2,)): -1.0, _term((1,), (4,)): -1.0},)),
    "NLS": Benchmark("NLS", "NLS.mat", "fbc31379b2d7a6921a6a97dc56346c79",
        (256, 251), ("u", "v"), (0, 1), 1, (19, 25), (5, 5), (11, 10), 6, 6,
        ({_term((0, 1), (2,)): 0.5, _term((2, 1), (0,)): 1.0, _term((0, 3), (0,)): 1.0},
         {_term((1, 0), (2,)): -0.5, _term((1, 2), (0,)): -1.0, _term((3, 0), (0,)): -1.0})),
    "RD": Benchmark("RD", "rxn_diff.mat", "8c8f0d13e2ab6c9ae3ee13f3a5fb9747",
        (256, 256, 201), ("u", "v"), (0, 1), 1, (13, 13, 14), (13, 13, 12), (13, 13, 12), 5, 4,
        ({_term((1, 0), (2, 0)): 0.1, _term((1, 0), (0, 2)): 0.1,
          _term((1, 2), (0, 0)): -1.0, _term((3, 0), (0, 0)): -1.0,
          _term((0, 3), (0, 0)): 1.0, _term((2, 1), (0, 0)): 1.0, _term((1, 0), (0, 0)): 1.0},
         {_term((0, 1), (2, 0)): 0.1, _term((0, 1), (0, 2)): 0.1,
          _term((1, 2), (0, 0)): -1.0, _term((3, 0), (0, 0)): -1.0,
          _term((0, 3), (0, 0)): -1.0, _term((2, 1), (0, 0)): -1.0, _term((0, 1), (0, 0)): 1.0})),
    "NS": Benchmark("NS", "Nav_Stokes.mat", "bf334620b2f0cbee7c00b65103acae06",
        (324, 149, 201), ("omega", "u", "v"), (0,), 1, (31, 31, 14), (12, 12, 8), (9, 9, 12), 3, 2,
        ({_term((1, 1, 0), (1, 0)): -1.0, _term((1, 0, 1), (0, 1)): -1.0,
          _term((1, 0, 0), (2, 0)): 0.01, _term((1, 0, 0), (0, 2)): 0.01},),
        archive_component_order=(2, 0, 1)),
}

# Active experiments use only polynomial state dictionaries. Seed indices are
# controlled separately by BENCHMARK_SEED_ORDER rather than registry order.
ORIGINAL_BENCHMARKS = tuple(BENCHMARKS)
CONSISTENCY_BENCHMARKS = ("HKS", "VBG")
BENCHMARKS.update({
    "HKS": Benchmark("HKS", "hyper_ks.npz", "", (256, 257), ("u",), (0,), 1,
        (26, 26), (7, 7), (0, 0), 8, 8,
        ({_term((1,), (4,)): 1., _term((1,), (6,)): .75,
          _term((2,), (1,)): -.5, _term((2,), (3,)): .1},),
        suite="consistency", noise_rule="centered_std", test_function="bump"),
    "VBG": Benchmark("VBG", "viscous_burgers_growth.npz", "", (512, 451), ("u",), (0,), 1,
        (64, 56), (12, 11), (0, 0), 6, 6,
        ({_term((1,), (2,)): .01, _term((2,), (1,)): -.5,
          _term((3,), (0,)): -1., _term((2,), (0,)): 2., _term((0,), (0,)): 1.},),
        suite="consistency", noise_rule="centered_std", test_function="bump"),
})


def generated_config(name):
    """Declared resimulations: the paper does not specify these initial data.

    Equations, domains and fine-grid sizes follow Sections 5.3.2/5.4.2.
    ICs, periodic BCs, integrator steps and fixed observation grids are ours.
    """
    if name not in CONSISTENCY_BENCHMARKS:
        raise ValueError("Only consistency PDEs have generated data.")
    hyper = name == "HKS"
    return dict(version=1, name=name, source="https://arxiv.org/html/2211.16000v1",
        provenance="Declared resimulation, not the authors' original trajectory",
        initial_condition="cos(x/16)*(1+sin(x/16))" if hyper else "2*sin(pi*x)",
        boundary_condition="periodic", x_interval=[0., float(32*np.pi)] if hyper else [-1., 1.],
        time_interval=[0., 82.] if hyper else [0., 1.5],
        fine_shape=[1024, 1025] if hyper else [2048, 1801],
        observation_shape=list(BENCHMARKS[name].shape), subsample=[4, 4],
        solver="Fourier spectral ETDRK4; nonlinearities evaluated on a 2x padded grid",
        base_steps_per_saved_interval=8 if hyper else 4,
        retained_steps_per_saved_interval=16 if hyper else 8,
        temporal_refinement_relative_tolerance=2e-5,
        noise_rule="Gaussian std = noise ratio * clean centered sample std (ddof=1)")


def _spectral_nonlinearity(v, wave_numbers, name):
    """Exactly pad/truncate all resolved quadratic and cubic products."""
    size = len(v)
    padded_size = 2*size
    padded = np.zeros(padded_size, dtype=complex)
    # The Nyquist mode is set to zero by the integrator; copy both remaining halves.
    padded[:size//2] = 2*v[:size//2]
    padded[-(size//2-1):] = 2*v[-(size//2-1):]
    values = fft.ifft(padded).real
    square = fft.fft(values*values)
    quadratic = np.concatenate((square[:size//2], [0j], square[-(size//2-1):]))/2
    if name == "HKS":
        result = (-.5j*wave_numbers + .1*(1j*wave_numbers)**3)*quadratic
    elif name == "VBG":
        cubic = fft.fft(values**3)
        cubic = np.concatenate((cubic[:size//2], [0j], cubic[-(size//2-1):]))/2
        result = (-.5j*wave_numbers+2)*quadratic-cubic
        result[0] += size  # Fourier transform of the constant +1
    else:
        raise ValueError("Unknown generated PDE.")
    result[size//2] = 0
    return result


def _etdrk4_solution(name, spatial_points, saved_timepoints, steps_per_interval):
    """Fourier ETDRK4 with contour quadrature for stable exponential coefficients."""
    config = generated_config(name)
    left, right = config["x_interval"]
    final_time = config["time_interval"][1]
    x = np.linspace(left, right, spatial_points, endpoint=False)
    time = np.linspace(0, final_time, saved_timepoints)
    dt = final_time/((saved_timepoints-1)*steps_per_interval)
    waves = 2*np.pi*fft.fftfreq(spatial_points, d=(right-left)/spatial_points)
    initial = np.cos(x/16)*(1+np.sin(x/16)) if name == "HKS" else 2*np.sin(np.pi*x)
    linear = waves**4-.75*waves**6 if name == "HKS" else -.01*waves**2
    E, E2 = np.exp(dt*linear), np.exp(dt*linear/2)
    roots = np.exp(1j*np.pi*(np.arange(1, 65)-.5)/64)
    LR = dt*linear[:, None]+roots
    Q = dt*np.mean((np.exp(LR/2)-1)/LR, axis=1).real
    f1 = dt*np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1).real
    f2 = dt*np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1).real
    f3 = dt*np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1).real
    v = fft.fft(initial)
    v[spatial_points//2] = 0
    solution = np.empty((spatial_points, saved_timepoints))
    solution[:, 0] = fft.ifft(v).real
    for point in range(1, saved_timepoints):
        for _ in range(steps_per_interval):
            Nv = _spectral_nonlinearity(v, waves, name)
            a = E2*v+Q*Nv
            Na = _spectral_nonlinearity(a, waves, name)
            b = E2*v+Q*Na
            Nb = _spectral_nonlinearity(b, waves, name)
            c = E2*a+Q*(2*Nb-Nv)
            Nc = _spectral_nonlinearity(c, waves, name)
            v = E*v+f1*Nv+2*f2*(Na+Nb)+f3*Nc
            v[spatial_points//2] = 0
        solution[:, point] = fft.ifft(v).real
        if not np.all(np.isfinite(solution[:, point])):
            raise ValueError(f"Nonfinite {name} numerical trajectory at t={time[point]}.")
    return x, time, solution


def _array_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _generated_dataset_path(spec, data_dir, generate):
    path = Path(data_dir)/spec.filename
    config = generated_config(spec.name)
    if not path.exists():
        if not generate:
            raise FileNotFoundError(f"Missing {path}; generate using --benchmarks {spec.name} --download-only.")
        nx, nt = config["fine_shape"]
        x, time, coarse = _etdrk4_solution(spec.name, nx, nt, config["base_steps_per_saved_interval"])
        _, _, refined = _etdrk4_solution(spec.name, nx, nt, config["retained_steps_per_saved_interval"])
        relative = float(np.linalg.norm(refined-coarse)/np.linalg.norm(refined))
        maximum = float(np.max(np.abs(refined-coarse)))
        if relative > config["temporal_refinement_relative_tolerance"]:
            raise ValueError(f"{spec.name} timestep refinement failed: relative difference {relative:g}.")
        sx, st = config["subsample"]
        x, time, data = x[::sx], time[::st], refined[::sx, ::st]
        metadata = dict(config=config, temporal_refinement_relative_difference=relative,
                        temporal_refinement_max_difference=maximum,
                        arrays_sha256=_array_digest(x, time, data))
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".npz.part")
        with partial.open("wb") as target:
            np.savez_compressed(target, x=x, time=time, u=data, metadata=json.dumps(metadata, sort_keys=True))
        partial.replace(path)
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata"]))
        if metadata["config"] != config or metadata["arrays_sha256"] != _array_digest(source["x"], source["time"], source["u"]):
            raise ValueError(f"Generated data/config mismatch: {path}.")
    return path


@dataclass
class SimulationData:
    benchmark: Benchmark
    spatial_grid: tuple[np.ndarray, ...]
    time: np.ndarray
    u_true: tuple[np.ndarray, ...]
    u_observed: tuple[np.ndarray, ...]
    noise_std: tuple[float, ...]


def file_checksum(path):
    digest = md5()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_path(spec, data_dir, download):
    path = Path(data_dir)/spec.filename
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Missing {path}; run experiments.py --download-only.")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix+".part")
        try:
            with urlopen(DATA_URL+spec.filename+"?download=1", timeout=120) as source, partial.open("wb") as target:
                for chunk in iter(lambda: source.read(1024*1024), b""):
                    target.write(chunk)
            if file_checksum(partial) != spec.checksum:
                raise ValueError(f"Checksum mismatch for downloaded {spec.filename}.")
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    if file_checksum(path) != spec.checksum:
        raise ValueError(f"Checksum mismatch for {path}; cached data were changed.")
    return path


def load_clean_benchmark(name, *, data_dir=DEFAULT_DATA_DIR, download=True):
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown paper benchmark: {name}")
    spec = BENCHMARKS[name]
    if spec.suite == "consistency":
        with np.load(_generated_dataset_path(spec, data_dir, download), allow_pickle=False) as archive:
            x, time, values = archive["x"], archive["time"], archive["u"]
        if values.shape != spec.shape:
            raise ValueError(f"Unexpected generated shape for {name}.")
        return SimulationData(spec, (x,), time, (values,), (values,), (0.,))
    archive = loadmat(_dataset_path(spec, data_dir, download), squeeze_me=True,
                      variable_names=("U_exact", "xs"))
    clean = archive["U_exact"]
    components = tuple(clean.ravel()) if clean.dtype == object else (clean,)
    components = tuple(np.asarray(v, dtype=float) for v in components)
    if spec.archive_component_order is not None:
        if len(components) != len(spec.archive_component_order):
            raise ValueError(f"Unexpected component count in {spec.filename}.")
        components = tuple(components[i] for i in spec.archive_component_order)
    axes = tuple(np.asarray(g, dtype=float).reshape(-1) for g in archive["xs"].ravel())
    if len(components) != len(spec.component_names) or any(v.shape != spec.shape for v in components):
        raise ValueError(f"Unexpected component shapes in {spec.filename}.")
    if len(axes) != len(spec.shape):
        raise ValueError(f"Unexpected grid dimensions in {spec.filename}.")
    for axis, size in zip(axes, spec.shape):
        steps = np.diff(axis)
        if len(axis) != size or not np.all(np.isfinite(axis)) or not len(steps) or steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=1e-7):
            raise ValueError(f"Unexpected observation grid in {spec.filename}.")
    if any(not np.all(np.isfinite(v)) for v in components):
        raise ValueError(f"Nonfinite observations in {spec.filename}.")
    return SimulationData(spec, axes[:-1], axes[-1], components, components, (0.0,)*len(components))


def add_gaussian_noise(u_true, noise_ratio=0.0, seed=None, *, scale_rule="rms"):
    """Eq. 5.1; each component receives independent noise scaled by its RMS."""
    if not np.isfinite(noise_ratio) or noise_ratio < 0:
        raise ValueError("noise_ratio must be finite and nonnegative.")
    scalar = isinstance(u_true, np.ndarray)
    clean = (u_true,) if scalar else tuple(u_true)
    if scale_rule not in ("rms", "centered_std"):
        raise ValueError("scale_rule must be rms or centered_std.")
    common_std = float(np.std(np.concatenate([np.asarray(v).ravel() for v in clean]), ddof=1)) if scale_rule == "centered_std" else None
    rng = np.random.default_rng(seed)
    noisy, deviations = [], []
    for values in clean:
        values = np.asarray(values, dtype=float)
        if not values.size or not np.all(np.isfinite(values)):
            raise ValueError("Clean observations must be nonempty and finite.")
        sigma = float(noise_ratio*(common_std if scale_rule == "centered_std" else np.sqrt(np.mean(values**2))))
        noisy.append(values.copy() if sigma == 0 else values+rng.normal(0, sigma, values.shape))
        deviations.append(sigma)
    return (noisy[0], deviations[0]) if scalar else (tuple(noisy), tuple(deviations))


def observation_instance(clean, noise_ratio=0.0, seed=None):
    observed, deviations = add_gaussian_noise(clean.u_true, noise_ratio, seed,
                                             scale_rule=clean.benchmark.noise_rule)
    return SimulationData(clean.benchmark, clean.spatial_grid, clean.time,
                          clean.u_true, observed, deviations)


def load_paper_wsindy_benchmark(name, *, noise_ratio=0.0, seed=0, data_dir=DEFAULT_DATA_DIR, download=True):
    return observation_instance(load_clean_benchmark(name, data_dir=data_dir, download=download), noise_ratio, seed)
