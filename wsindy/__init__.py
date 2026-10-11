"""Weak SINDy for PDEs with pluggable dictionaries, state estimators and regressions."""
from .dictionary import Block, Term, spatial
from .testfunctions import Tests
from .system import StrongSystem, WeakSystem
from .regression import BestSubset, Debiased, Fit, FromAPI, LassoPath, LeastSquares, MSTLS, fit
from .eiv import EIV
from .estimators import CrossFitPilot, FromCallable, FromPySINDy, Identity, MovingAverage, estimate, noise_std
