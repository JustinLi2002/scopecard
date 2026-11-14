"""Rule diagnosis: which simple depth rule, if any, reproduces the release's unit set.

The rule is never an input. Families, each with one threshold T on source depth:
  threshold         units with depth >= T                                (single pass)
  both_ends         ends of relations whose two ends both have depth >= T (YAGO3-10's rule)
  seeds_neighbours  units with depth >= T plus everything they relate to
  kcore             the maximal sub-relation in which every unit has >= T relations (iterated)
search_depth_threshold applies the threshold family to a depth supplied by the caller.
The search reports, per family, the threshold with the highest Jaccard against the release
and a verdict: reproduced (Jaccard 1), approximately reproduced (>= 0.95), not reproduced.
Release units that do not map onto the universe can be selected by no rule over it; pass
their number as `unmapped_release_units` so that they count against the rule (YAGO3-10 has
2,044 such entities).
"""
from typing import Dict, Iterable, List, Optional

import numpy as np

from .core import Universe, counts

FAMILIES = ('threshold', 'both_ends', 'seeds_neighbours', 'kcore')


def rule_units(universe: Universe, family: str, T: int, edge_mask: Optional[np.ndarray] = None) -> np.ndarray:
    e = universe.edges if edge_mask is None else universe.edges[edge_mask]
    n = universe.n
    deg = counts(e.ravel(), n)
    sel = np.zeros(n, bool)
    if family == 'threshold':
        return (deg >= T) & (deg > 0)
    if family == 'both_ends':
        keep = (deg[e[:, 0]] >= T) & (deg[e[:, 1]] >= T)
        sel[e[keep].ravel()] = True
        return sel
    if family == 'seeds_neighbours':
        seed = deg >= T
        keep = seed[e[:, 0]] | seed[e[:, 1]]
        sel[e[keep].ravel()] = True
        return sel
    if family == 'kcore':
        keep = np.ones(len(e), bool)
        while True:
            d = counts(e[keep].ravel(), n)
            nxt = keep & (d[e[:, 0]] >= T) & (d[e[:, 1]] >= T)
            if nxt.sum() == keep.sum():
                break
            keep = nxt
        sel[e[keep].ravel()] = True
        return sel
    raise ValueError(f'unknown rule family {family}')


def jaccard(a: np.ndarray, b: np.ndarray, extra_b: int = 0) -> float:
    """Jaccard of two unit masks; `extra_b` units of b lie outside the mask's universe."""
    u = (a | b).sum() + extra_b
    return float((a & b).sum() / u) if u else 1.0


def verdict(j: float) -> str:
    if j == 1.0:
        return 'reproduced'
    if j >= 0.95:
        return 'approximately reproduced'
    return 'not reproduced'


def search_depth_threshold(depth: np.ndarray, unit_mask: np.ndarray, thresholds: Iterable[int] = range(1, 41),
                           stated_T: Optional[int] = None, unmapped_release_units: int = 0) -> Dict:
    """The threshold family on a depth the caller supplies, for query-level units whose depth is not a
    count of universe relations among units (e.g. a paper's references to papers outside the universe).
    Same output shape as search_rules."""
    thresholds = list(thresholds)
    extra = int(unmapped_release_units)
    best = None
    for T in thresholds:
        s = depth >= T
        j = jaccard(s, unit_mask, extra)
        if best is None or j > best['jaccard']:
            best = {'family': 'threshold', 'T': int(T), 'jaccard': j, 'selected': int(s.sum()),
                    'release': int(unit_mask.sum()) + extra, 'overlap': int((s & unit_mask).sum())}
    best['verdict'] = verdict(best['jaccard'])
    res = {'best': best, 'families': [best], 'thresholds_searched': [thresholds[0], thresholds[-1]],
           'unmapped_release_units': extra}
    if stated_T is not None:
        s = depth >= stated_T
        j = jaccard(s, unit_mask, extra)
        res['stated'] = {'family': 'threshold', 'T': int(stated_T), 'jaccard': j, 'selected': int(s.sum()),
                         'overlap': int((s & unit_mask).sum()), 'verdict': verdict(j)}
    return res


def search_rules(universe: Universe, unit_mask: np.ndarray, families: Iterable[str] = FAMILIES,
                 thresholds: Iterable[int] = range(1, 41), edge_mask: Optional[np.ndarray] = None,
                 stated: Optional[Dict] = None, unmapped_release_units: int = 0) -> Dict:
    """Best threshold per family; `stated` = {'family': ..., 'T': ...} is evaluated as well."""
    thresholds = list(thresholds)
    extra = int(unmapped_release_units)
    out: List[Dict] = []
    for fam in families:
        best = None
        for T in thresholds:
            s = rule_units(universe, fam, T, edge_mask)
            j = jaccard(s, unit_mask, extra)
            if best is None or j > best['jaccard']:
                best = {'family': fam, 'T': int(T), 'jaccard': j, 'selected': int(s.sum()),
                        'release': int(unit_mask.sum()) + extra, 'overlap': int((s & unit_mask).sum())}
        best['verdict'] = verdict(best['jaccard'])
        out.append(best)
    out.sort(key=lambda r: -r['jaccard'])
    res = {'best': out[0], 'families': out, 'thresholds_searched': [thresholds[0], thresholds[-1]],
           'unmapped_release_units': extra}
    if stated:
        s = rule_units(universe, stated['family'], int(stated['T']), edge_mask)
        j = jaccard(s, unit_mask, extra)
        res['stated'] = dict(stated, jaccard=j, selected=int(s.sum()), overlap=int((s & unit_mask).sum()), verdict=verdict(j))
    return res
