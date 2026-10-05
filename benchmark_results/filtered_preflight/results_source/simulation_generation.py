"""Original WSINDy paper datasets and Eq. 5.1 observation instances.

Use authors' U_exact arrays without interpolation, subsampling, or replacing
initial conditions. Metadata follows arXiv:2007.02848v3 with the author's
archived RD library (degree 5, derivative order 4; 181 candidates).
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from urllib.request import urlopen
import numpy as np
from scipy.io import loadmat
from methods import LibraryTerm, polynomial_library

DATA_URL = "https://zenodo.org/records/20787783/files/"
DEFAULT_DATA_DIR = Path(__file__).parent / "data"


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

    def library(self):
        n, spatial = len(self.component_names), len(self.shape)-1
        if self.name != "NS":
            return polynomial_library(n, spatial, self.max_degree, self.max_derivative,
                trigonometric_frequencies=(1, 2) if self.name == "SG" else ())
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
    "SG": Benchmark("SG", "Sine_Gordon.mat", "a10b069735312122f0f1046d0fa52cdc",
        (129, 403, 205), ("u",), (0,), 2, (40, 40, 25), (5, 5, 8), (8, 8, 10), 4, 4,
        ({_term((1,), (2, 0)): 1.0, _term((1,), (0, 2)): 1.0,
          _term((0,), (0, 0), "sin"): -1.0},)),
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


def add_gaussian_noise(u_true, noise_ratio=0.0, seed=None):
    """Eq. 5.1; each component receives independent noise scaled by its RMS."""
    if not np.isfinite(noise_ratio) or noise_ratio < 0:
        raise ValueError("noise_ratio must be finite and nonnegative.")
    scalar = isinstance(u_true, np.ndarray)
    clean = (u_true,) if scalar else tuple(u_true)
    rng = np.random.default_rng(seed)
    noisy, deviations = [], []
    for values in clean:
        values = np.asarray(values, dtype=float)
        if not values.size or not np.all(np.isfinite(values)):
            raise ValueError("Clean observations must be nonempty and finite.")
        sigma = float(noise_ratio*np.sqrt(np.mean(values**2)))
        noisy.append(values.copy() if sigma == 0 else values+rng.normal(0, sigma, values.shape))
        deviations.append(sigma)
    return (noisy[0], deviations[0]) if scalar else (tuple(noisy), tuple(deviations))


def observation_instance(clean, noise_ratio=0.0, seed=None):
    observed, deviations = add_gaussian_noise(clean.u_true, noise_ratio, seed)
    return SimulationData(clean.benchmark, clean.spatial_grid, clean.time,
                          clean.u_true, observed, deviations)


def load_paper_wsindy_benchmark(name, *, noise_ratio=0.0, seed=0, data_dir=DEFAULT_DATA_DIR, download=True):
    return observation_instance(load_clean_benchmark(name, data_dir=data_dir, download=download), noise_ratio, seed)
