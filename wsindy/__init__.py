"""Weak SINDy for PDEs with pluggable dictionaries, state estimators and regressions."""
from .dictionary import Block, Term, spatial
from .testfunctions import Tests
from .system import SplitSystem, StrongSystem, WeakSystem
from .regression import BestSubset, Debiased, Fit, FromAPI, LassoPath, LeastSquares, MSTLS, fit
from .eiv import EIV
from .dantzig import Dantzig
from .estimators import FromCallable, FromPySINDy, Identity, MovingAverage, SplitPilot, estimate, noise_std
