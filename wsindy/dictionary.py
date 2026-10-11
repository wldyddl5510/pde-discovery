"""Dictionary terms D^alpha f(U), with f taken from any PySINDy feature library."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pysindy as ps
from sklearn.base import clone


@dataclass(frozen=True)
class Term:
    feature: str                            # library feature name, e.g. "u^2 v" or "sin(1 u)"
    alpha: tuple[int, ...]                  # derivative orders: spatial axes, then time
    powers: tuple[int, ...] | None = None   # monomial exponents; None for other features
    source: tuple[int, int] | None = None   # (block, column) of the feature; None on the LHS

    @property
    def label(self):
        letters = "xyz"[:len(self.alpha)-1]+"t"
        suffix = "".join(a*q for a, q in zip(letters, self.alpha))
        if not suffix:
            return self.feature
        single = self.powers is not None and sum(self.powers) == 1
        return f"{self.feature}_{suffix}" if single else f"({self.feature})_{suffix}"


@dataclass(frozen=True)
class Block:
    """{D^alpha f : f in functions, alpha in alphas}; derivatives of constants are dropped."""
    functions: ps.feature_library.base.BaseFeatureLibrary
    alphas: tuple[tuple[int, ...], ...]
    keep: Callable[[str], bool] | None = None   # filter on feature names


def spatial(order, dim):
    """Zero derivative and pure spatial derivatives 1..order on each of `dim` axes."""
    zero = (0,)*(dim+1)
    return (zero,)+tuple(tuple(q if d == a else 0 for d in range(dim+1))
                         for a in range(dim) for q in range(1, order+1))


def _powers(library):
    if isinstance(library, ps.PolynomialLibrary):
        return [tuple(int(p) for p in row) for row in library.powers_]
    if isinstance(library, ps.ConcatLibrary):
        return [p for sub in library.libraries for p in _powers(sub)]
    return [None]*library.n_output_features_


def expand(blocks, names):
    """Fitted copies of the block libraries and the ordered dictionary terms."""
    libraries, terms = [], []
    for b, block in enumerate(blocks):
        library = clone(block.functions).fit(np.zeros((1, len(names))))
        libraries.append(library)
        for c, (feature, powers) in enumerate(zip(library.get_feature_names(list(names)), _powers(library))):
            if block.keep and not block.keep(feature):
                continue
            constant = powers is not None and not any(powers)
            terms.extend(Term(feature, alpha, powers, (b, c)) for alpha in block.alphas
                         if not (constant and any(alpha)))
    labels = [t.label for t in terms]
    if len(set(labels)) != len(labels):
        raise ValueError("Dictionary terms must be unique.")
    return libraries, tuple(terms)


def lhs_term(label, names, dims):
    """'u_t' or 'u_tt' -> D_t^k of one observed component, on `dims` space-time axes."""
    name, letters = label.rsplit("_", 1)
    powers = tuple(int(n == name) for n in names)
    return Term(name, (0,)*(dims-1)+(letters.count("t"),), powers)
