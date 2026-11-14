"""Universes for pair classification (the [P2]/[P3] setting): units on both sides of a pair, with optional
observed pairs (known interactions) that give each unit its depth.

symmetric   one unit set and unordered pairs (protein-protein interaction)
bipartite   two unit sets, ids prefixed 'a:' and 'b:' (drug-target interaction)
"""
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from ..core import Universe


def from_unit_pairs(units_a: Optional[Sequence[str]], units_b: Optional[Sequence[str]] = None,
                    pairs: Iterable[Tuple[str, str]] = (), symmetric: bool = True, name: str = '',
                    version: str = '') -> Tuple[Universe, Dict]:
    """The declared unit lists fix the universe; observed pairs with an end outside it are dropped and counted.
    Without unit lists, the universe is every unit that occurs in an observed pair."""
    pairs = list(pairs)
    info = {'observed_pairs_given': len(pairs)}
    if symmetric:
        units = sorted(set(units_a)) if units_a is not None else sorted({x for p in pairs for x in p})
        idx = {u: i for i, u in enumerate(units)}
        keep = {(min(a, b), max(a, b)) for a, b in pairs if a in idx and b in idx and a != b}
        info['observed_pairs_outside_universe'] = sum(1 for a, b in pairs if a not in idx or b not in idx)
        info['self_pairs_dropped'] = sum(1 for a, b in pairs if a == b and a in idx)
        edges = np.array(sorted((idx[a], idx[b]) for a, b in keep), dtype=np.int64).reshape(-1, 2)
        uni = Universe(name=name, version=version, units=np.array(units, dtype=object), edges=edges)
    else:
        A = sorted(set(units_a)) if units_a is not None else sorted({a for a, _ in pairs})
        B = sorted(set(units_b)) if units_b is not None else sorted({b for _, b in pairs})
        units = ['a:' + x for x in A] + ['b:' + x for x in B]
        ia = {x: i for i, x in enumerate(A)}
        ib = {x: len(A) + i for i, x in enumerate(B)}
        keep = {(a, b) for a, b in pairs if a in ia and b in ib}
        info['observed_pairs_outside_universe'] = sum(1 for a, b in pairs if a not in ia or b not in ib)
        edges = np.array(sorted((ia[a], ib[b]) for a, b in keep), dtype=np.int64).reshape(-1, 2)
        part = np.array([0] * len(A) + [1] * len(B), dtype=np.int8)
        uni = Universe(name=name, version=version, units=np.array(units, dtype=object), edges=edges, part=part)
    info['observed_pairs_kept'] = int(len(uni.edges))
    return uni, info
