"""Benchmark PDEs: data, dictionaries, test functions, truth and observation noise.

Original suite (arXiv:2007.02848v3, Table 3; RD uses the archived degree-5/order-4 library):
the authors' U_exact arrays. Consistency suite (arXiv:2211.16000v1): declared resimulations.
Arrays follow PySINDy: U has shape (*space, time, n_components).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import pysindy as ps
from scipy import fft
from scipy.io import loadmat
from wsindy.dictionary import Block, expand, lhs_term, spatial
from wsindy.testfunctions import Tests

DATA_DIR = Path(__file__).parent/"data"
DATA_URL = "https://zenodo.org/records/20787783/files/"
# Seed index per PDE; SG's slot is kept so that saved noise draws stay reproducible.
SEED_ORDER = ("IB", "KdV", "KS", "NLS", "SG", "RD", "NS", "HKS", "VBG")


@dataclass(frozen=True, eq=False)
class Profile:
    rescale: bool               # authors' state/coordinate scaling
    thresholds: np.ndarray      # MSTLS threshold path
    rule: str                   # "relative" bounds or "absolute" thresholds
    allow_empty: bool
    noise: str                  # noise std = ratio * component RMS ("rms") or pooled centered std ("std")
    filter_rule: str            # moving-average window cap: "per_axis" or "volume"


AUTHORS = Profile(True, np.logspace(-4, 0, 50), "relative", False, "rms", "per_axis")
CONSISTENCY = Profile(False, np.logspace(-4, 0, 100), "absolute", True, "std", "volume")


@dataclass(frozen=True, eq=False)
class PDE:
    name: str
    file: str
    md5: str
    components: tuple[str, ...]
    dictionary: tuple[Block, ...]
    lhs: tuple[str, ...]
    truth: tuple[dict[str, float], ...]   # one {term label: coefficient} per LHS
    tests: Tests
    profile: Profile = AUTHORS
    periodic: bool = True                 # spatial boundary for pilots
    archive_order: tuple[int, ...] | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def dictionary_terms(self):
        if "terms" not in self._cache:
            self._cache["terms"] = expand(self.dictionary, self.components)
        return self._cache["terms"]

    def lhs_terms(self):
        return tuple(lhs_term(label, self.components, len(self.dictionary[0].alphas[0])) for label in self.lhs)

    @property
    def degree(self):
        return max(sum(t.powers) for t in self.dictionary_terms()[1] if t.powers is not None)

    def truth_matrix(self, terms):
        labels = [t.label for t in terms]
        missing = {k for equation in self.truth for k in equation} - set(labels)
        if missing:
            raise ValueError(f"Dictionary omits true terms: {missing}")
        return np.array([[equation.get(label, 0.) for label in labels] for equation in self.truth])

    def equations(self):
        lines = []
        for lhs, equation in zip(self.lhs, self.truth):
            rhs = ""
            for k, c in equation.items():
                term = k if abs(c) == 1 else f"{abs(c):g} {k}"
                rhs += (" - " if c < 0 else " + ")+term if rhs else ("-" if c < 0 else "")+term
            lines.append(f"{lhs} = {rhs}")
        return lines


def _poly(degree, order, dim=1, **kwargs):
    return (Block(ps.PolynomialLibrary(degree=degree, **kwargs), spatial(order, dim)),)


PDES = {pde.name: pde for pde in (
    PDE("IB", "burgers.mat", "3ccddfa4fe4173140f27e9a404b252f4", ("u",), _poly(6, 6), ("u_t",),
        ({"(u^2)_x": -.5},), Tests(m=(60, 60), p=(7, 7), s=(5, 5))),
    PDE("KdV", "KdV.mat", "aa526a0d61c026d266b6c1127f9350b2", ("u",), _poly(6, 6), ("u_t",),
        ({"(u^2)_x": -.5, "u_xxx": -1.},), Tests(m=(45, 80), p=(8, 7), s=(8, 12))),
    PDE("KS", "KS.mat", "01208398e459323f4db7401083998e0a", ("u",), _poly(6, 6), ("u_t",),
        ({"(u^2)_x": -.5, "u_xx": -1., "u_xxxx": -1.},), Tests(m=(23, 22), p=(10, 10), s=(5, 6))),
    PDE("NLS", "NLS.mat", "fbc31379b2d7a6921a6a97dc56346c79", ("u", "v"), _poly(6, 6), ("u_t", "v_t"),
        ({"v_xx": .5, "u^2 v": 1., "v^3": 1.}, {"u_xx": -.5, "u v^2": -1., "u^3": -1.}),
        Tests(m=(19, 25), p=(11, 10), s=(5, 5))),
    PDE("RD", "rxn_diff.mat", "8c8f0d13e2ab6c9ae3ee13f3a5fb9747", ("u", "v"), _poly(5, 4, dim=2), ("u_t", "v_t"),
        ({"u_xx": .1, "u_yy": .1, "u v^2": -1., "u^3": -1., "v^3": 1., "u^2 v": 1., "u": 1.},
         {"v_xx": .1, "v_yy": .1, "u v^2": -1., "u^3": -1., "v^3": -1., "u^2 v": -1., "v": 1.}),
        Tests(m=(13, 13, 14), p=(13, 13, 12), s=(13, 13, 12))),
    PDE("NS", "Nav_Stokes.mat", "bf334620b2f0cbee7c00b65103acae06", ("omega", "u", "v"),
        # degree <= 2 without derivatives; degree <= 3 with omega when differentiated (Table 3)
        (Block(ps.PolynomialLibrary(degree=2), spatial(0, 2)),
         Block(ps.PolynomialLibrary(degree=3, include_bias=False), spatial(2, 2)[1:], keep=lambda f: "omega" in f)),
        ("omega_t",), ({"(omega u)_x": -1., "(omega v)_y": -1., "omega_xx": .01, "omega_yy": .01},),
        Tests(m=(31, 31, 14), p=(9, 9, 12), s=(12, 12, 8)), periodic=False, archive_order=(2, 0, 1)),
    PDE("HKS", "hyper_ks.npz", "", ("u",), _poly(8, 8), ("u_t",),
        ({"u_xxxx": 1., "u_xxxxxx": .75, "(u^2)_x": -.5, "(u^2)_xxx": .1},),
        Tests(m=(26, 26), s=(7, 7), shape="bump"), CONSISTENCY),
    PDE("VBG", "viscous_burgers_growth.npz", "", ("u",), _poly(6, 6), ("u_t",),
        ({"u_xx": .01, "(u^2)_x": -.5, "u^3": -1., "u^2": 2., "1": 1.},),
        Tests(m=(64, 56), s=(12, 11), shape="bump"), CONSISTENCY),
)}


def trial_seed(name, noise_ratio, trial, root=0):
    """Stable across PDE order, noise subsets and reruns."""
    bits = int(np.float64(noise_ratio).view(np.uint64))
    sequence = np.random.SeedSequence([root, SEED_ORDER.index(name), bits & 0xffffffff, bits >> 32, trial])
    return sequence.generate_state(4).tolist()


def observe(pde, U, noise_ratio, seed):
    """Independent Gaussian noise per component (Eq. 5.1); returns (U_noisy, per-component std)."""
    rng = np.random.default_rng(seed)
    pooled = float(np.std(np.concatenate([U[..., c].ravel() for c in range(U.shape[-1])]), ddof=1))
    noisy, stds = [], []
    for c in range(U.shape[-1]):
        v = U[..., c]
        std = float(noise_ratio*(pooled if pde.profile.noise == "std" else np.sqrt(np.mean(v**2))))
        noisy.append(v.copy() if std == 0 else v+rng.normal(0, std, v.shape))
        stds.append(std)
    return np.stack(noisy, axis=-1), stds


def metrics(coefficients, truth, support=None):
    """TPR = TP/(TP+FN+FP) (Eq. 5.3), E_inf over true terms (5.4), relative E2 (5.5)."""
    selected = np.asarray(support, dtype=bool) if support is not None else coefficients != 0
    actual = truth != 0
    tp, fn, fp = (int(np.count_nonzero(a)) for a in (selected & actual, ~selected & actual, selected & ~actual))
    return dict(tpr=tp/(tp+fn+fp), e2=float(np.linalg.norm(coefficients-truth)/np.linalg.norm(truth)),
                e_inf=float(np.max(np.abs((coefficients-truth)[actual]/truth[actual]))),
                tp=tp, fn=fn, fp=fp, exact_support=fn == 0 and fp == 0)


@lru_cache(maxsize=2)
def load(name, data_dir=DATA_DIR, download=True):
    """Clean (U, grid) for a PDE, downloading or generating the data file when missing."""
    pde, path = PDES[name], Path(data_dir)/PDES[name].file
    if pde.profile is CONSISTENCY:
        with np.load(_generated(pde, path, download), allow_pickle=False) as archive:
            return archive["u"][..., None], (archive["x"], archive["time"])
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Missing {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(DATA_URL+pde.file+"?download=1", timeout=120) as source:
            content = source.read()
        if hashlib.md5(content).hexdigest() != pde.md5:
            raise ValueError(f"Checksum mismatch for downloaded {pde.file}")
        path.write_bytes(content)
    if hashlib.md5(path.read_bytes()).hexdigest() != pde.md5:
        raise ValueError(f"Checksum mismatch: {path}")
    archive = loadmat(path, squeeze_me=True, variable_names=("U_exact", "xs"))
    clean = archive["U_exact"]
    components = [np.asarray(v, dtype=float) for v in (clean.ravel() if clean.dtype == object else [clean])]
    if pde.archive_order:
        components = [components[i] for i in pde.archive_order]
    grid = tuple(np.asarray(g, dtype=float).reshape(-1) for g in archive["xs"].ravel())
    return np.stack(components, axis=-1), grid


# ---- consistency-suite resimulation (Fourier ETDRK4; ICs, BCs, grids and steps are declared here) ----

def generated_config(name):
    hyper = name == "HKS"
    return dict(version=1, name=name, source="https://arxiv.org/html/2211.16000v1",
        provenance="Declared resimulation, not the authors' original trajectory",
        initial_condition="cos(x/16)*(1+sin(x/16))" if hyper else "2*sin(pi*x)",
        boundary_condition="periodic", x_interval=[0., float(32*np.pi)] if hyper else [-1., 1.],
        time_interval=[0., 82.] if hyper else [0., 1.5],
        fine_shape=[1024, 1025] if hyper else [2048, 1801],
        observation_shape=[256, 257] if hyper else [512, 451], subsample=[4, 4],
        solver="Fourier spectral ETDRK4; nonlinearities evaluated on a 2x padded grid",
        base_steps_per_saved_interval=8 if hyper else 4,
        retained_steps_per_saved_interval=16 if hyper else 8,
        temporal_refinement_relative_tolerance=2e-5,
        noise_rule="Gaussian std = noise ratio * clean centered sample std (ddof=1)")


def _nonlinearity(v, waves, name):
    """Exact padding/truncation of the resolved quadratic and cubic products."""
    size = len(v)
    padded = np.zeros(2*size, dtype=complex)
    padded[:size//2], padded[-(size//2-1):] = 2*v[:size//2], 2*v[-(size//2-1):]
    values = fft.ifft(padded).real
    def truncate(spectrum):
        return np.concatenate((spectrum[:size//2], [0j], spectrum[-(size//2-1):]))/2
    quadratic = truncate(fft.fft(values*values))
    if name == "HKS":
        result = (-.5j*waves+.1*(1j*waves)**3)*quadratic
    else:
        result = (-.5j*waves+2)*quadratic-truncate(fft.fft(values**3))
        result[0] += size   # transform of the constant +1
    result[size//2] = 0
    return result


def _etdrk4(name, points, saved, steps):
    config = generated_config(name)
    (left, right), final = config["x_interval"], config["time_interval"][1]
    x = np.linspace(left, right, points, endpoint=False)
    dt = final/((saved-1)*steps)
    waves = 2*np.pi*fft.fftfreq(points, d=(right-left)/points)
    linear = waves**4-.75*waves**6 if name == "HKS" else -.01*waves**2
    E, E2 = np.exp(dt*linear), np.exp(dt*linear/2)
    LR = dt*linear[:, None]+np.exp(1j*np.pi*(np.arange(1, 65)-.5)/64)
    Q = dt*np.mean((np.exp(LR/2)-1)/LR, axis=1).real
    f1 = dt*np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1).real
    f2 = dt*np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1).real
    f3 = dt*np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1).real
    v = fft.fft(np.cos(x/16)*(1+np.sin(x/16)) if name == "HKS" else 2*np.sin(np.pi*x))
    v[points//2] = 0
    solution = np.empty((points, saved))
    solution[:, 0] = fft.ifft(v).real
    for k in range(1, saved):
        for _ in range(steps):
            Nv = _nonlinearity(v, waves, name)
            a = E2*v+Q*Nv
            Na = _nonlinearity(a, waves, name)
            b = E2*v+Q*Na
            Nb = _nonlinearity(b, waves, name)
            c = E2*a+Q*(2*Nb-Nv)
            v = E*v+f1*Nv+2*f2*(Na+Nb)+f3*_nonlinearity(c, waves, name)
            v[points//2] = 0
        solution[:, k] = fft.ifft(v).real
    return x, np.linspace(0, final, saved), solution


def _digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _generated(pde, path, generate):
    config = generated_config(pde.name)
    if not path.exists():
        if not generate:
            raise FileNotFoundError(f"Missing {path}")
        points, saved = config["fine_shape"]
        _, _, coarse = _etdrk4(pde.name, points, saved, config["base_steps_per_saved_interval"])
        x, time, fine = _etdrk4(pde.name, points, saved, config["retained_steps_per_saved_interval"])
        relative = float(np.linalg.norm(fine-coarse)/np.linalg.norm(fine))
        if relative > config["temporal_refinement_relative_tolerance"]:
            raise ValueError(f"{pde.name} timestep refinement failed: {relative:g}")
        sx, st = config["subsample"]
        x, time, u = x[::sx], time[::st], fine[::sx, ::st]
        metadata = dict(config=config, temporal_refinement_relative_difference=relative,
                        temporal_refinement_max_difference=float(np.max(np.abs(fine-coarse))),
                        arrays_sha256=_digest(x, time, u))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, x=x, time=time, u=u, metadata=json.dumps(metadata, sort_keys=True))
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata["config"] != config or metadata["arrays_sha256"] != _digest(archive["x"], archive["time"], archive["u"]):
            raise ValueError(f"Generated data/config mismatch: {path}")
    return path
