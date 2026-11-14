"""What a release can sample, measured against the declared universe.

unit_exclusion   share of universe units absent from the release (entity-level W)
pair_exclusion   share of the universe's unobserved pairs outside the release's retained
                 rectangle (bipartite) or retained unit set (unipartite); [P2]'s convention
query_shares     per-query excluded shares W_q of filtered link prediction on the target, and their mean
user_shares      the same for all-item recommendation, one query per test user
depth_profile    source depth of included vs excluded units
derived_scope    the breadth/size trade-off: restrict the universe to depth >= d and report W;
                 d_min is the smallest d with W <= 0.5 ([P2]'s derived scope)
"""
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .core import Universe


def _as_mask(universe: Universe, among: Optional[np.ndarray]) -> np.ndarray:
    return np.ones(universe.n, bool) if among is None else np.asarray(among, bool)


def unit_exclusion(universe: Universe, unit_mask: np.ndarray, among: Optional[np.ndarray] = None,
                   by_strata: bool = True) -> Dict:
    """W = share of units in `among` (default: all universe units) that the release does not contain."""
    among = _as_mask(universe, among)
    inc = unit_mask & among
    n, k = int(among.sum()), int(inc.sum())
    out = {'units': n, 'included': k, 'excluded': n - k, 'W': (n - k) / n if n else float('nan')}
    if universe.part is not None:
        out['by_part'] = {}
        for p in (0, 1):
            m = among & (universe.part == p)
            out['by_part'][str(p)] = {'units': int(m.sum()), 'included': int((m & unit_mask).sum()),
                                      'W': 1 - (m & unit_mask).sum() / m.sum() if m.sum() else float('nan')}
    if by_strata:
        out['by_strata'] = {}
        for sname, labels in universe.strata.items():
            d = {}
            for lab in sorted(set(labels[among].tolist())):
                m = among & (labels == lab)
                d[str(lab)] = {'units': int(m.sum()), 'included': int((m & unit_mask).sum()),
                               'W': float(1 - (m & unit_mask).sum() / m.sum())}
            out['by_strata'][sname] = d
    return out


def pair_exclusion(universe: Universe, unit_mask: np.ndarray, release_pairs: Optional[int] = None) -> Dict:
    """Candidates are the universe's unobserved pairs (pairs of distinct units for a unipartite universe).
    Bipartite: W = 1 - (|A_r||B_r| - P_r) / (|A||B| - P); unipartite: W = 1 - (C(n_r, 2) - P_r) / (C(n, 2) - P),
    where P counts the universe's observed pairs and P_r those among the retained units. W is undefined (NaN,
    with a note) when the universe has no unobserved pair.

    `release_pairs` adds the 2026-10 pilot table's convention, which takes P_r from the release's own positive
    count instead. Its two counts come from different candidate sets, so it is not a share: W_pairs (the
    headline) always uses the universe's positives, and the pilot value is kept beside it as
    W_pairs_release_convention, undefined outside [0, 1]."""
    P = len(universe.edges)
    inside = unit_mask[universe.edges[:, 0]] & unit_mask[universe.edges[:, 1]]
    Pr = int(inside.sum())
    if universe.part is None:
        n, nr = universe.n, int(unit_mask.sum())
        rect_all, rect_rel = n * (n - 1) // 2, nr * (nr - 1) // 2
        out = {'units': n, 'units_retained': nr}
    else:
        a, b = universe.part == 0, universe.part == 1
        A, B = int(a.sum()), int(b.sum())
        Ar, Br = int((a & unit_mask).sum()), int((b & unit_mask).sum())
        rect_all, rect_rel = A * B, Ar * Br
        out = {'part0_units': A, 'part1_units': B, 'part0_retained': Ar, 'part1_retained': Br}
    cand_all = rect_all - P
    share = lambda num: 1 - num / cand_all if cand_all > 0 else float('nan')
    out.update({'observed_pairs': P, 'observed_pairs_retained': Pr, 'candidate_pairs': cand_all,
                'positives_convention': 'universe', 'W_pairs': share(rect_rel - Pr)})
    if cand_all <= 0:
        out['note'] = 'The universe has no unobserved pair, so no candidate can be excluded: W_pairs is undefined.'
    if release_pairs is not None:
        w = share(rect_rel - int(release_pairs))
        out.update({'observed_pairs_release': int(release_pairs),
                    'W_pairs_release_convention': w if 0 <= w <= 1 else float('nan'),
                    'release_convention_note': 'Pilot-table convention: release positives inside the retained units, '
                                               'universe positives in the denominator. The two counts come from different '
                                               'candidate sets, so this is not a share and is kept only for comparison '
                                               'with the 2026-10 pilot table; W_pairs is the share.'})
    return out


def _share_summary(w: np.ndarray, protocol: str, **extra) -> Dict:
    """Mean and spread of per-query excluded shares, with the queries that need care counted, not dropped:
    without any candidate (every unit is a known answer; left out of the mean), without an excluded candidate
    (W_q = 0, which adds nothing to the target difference), and without a kept one (W_q = 1: the benchmark
    difference is undefined there, so no break-even can be exact)."""
    empty = ~np.isfinite(w)
    w = w[~empty]
    out = {'queries': int(len(w)), 'W_bar': float(w.mean()) if len(w) else float('nan'),
           'W_q': ({'min': float(w.min()), 'median': float(np.median(w)), 'max': float(w.max())} if len(w) else {}),
           'queries_without_candidates': int(empty.sum()), 'queries_without_excluded': int((w == 0).sum()),
           'queries_without_kept': int((w == 1).sum()), 'protocol': protocol}
    out.update(extra)
    return out


def _known_counts(keys: np.ndarray, answers: np.ndarray, query_keys: np.ndarray, among: np.ndarray,
                  outside: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """For each query key, the number of distinct known answers among the candidates' units (`among`) and among
    the excluded ones (`outside`). keys/answers list every known (key, answer) pair, positives included."""
    pairs = np.unique(keys * n + answers)
    pk, pa = pairs // n, pairs % n
    uq = np.unique(query_keys)
    slot = np.searchsorted(uq, pk)
    hit = (slot < len(uq)) & (uq[np.minimum(slot, len(uq) - 1)] == pk)
    kV = np.bincount(slot[hit], weights=among[pa[hit]], minlength=len(uq))
    kU = np.bincount(slot[hit], weights=outside[pa[hit]], minlength=len(uq))
    at = np.searchsorted(uq, query_keys)
    return kV[at], kU[at]


def query_shares(universe: Universe, unit_mask: np.ndarray, test: Sequence[Tuple[int, str, int]],
                 known: Sequence[Tuple[int, str, int]] = (), among: Optional[np.ndarray] = None, **extra) -> Dict:
    """Per-query excluded shares of filtered link prediction on the target (Section 2 of the paper). Each test
    triple (h, r, t), as unit indexes and a relation label, poses a tail query (h, r, ?) and a head query (?, r, t).
    A query's candidates are the units in `among` (default: all) minus its known answers, which are the universe's
    triples of relation r and the `known` triples (the release's), the positive included. W_q is the share of
    those candidates outside the release."""
    among = _as_mask(universe, among)
    outside = among & ~np.asarray(unit_mask, bool)
    nV, nU, n = int(among.sum()), int(outside.sum()), universe.n
    types = universe.edge_types.tolist() if universe.edge_types is not None else [''] * len(universe.edges)
    tix: Dict[str, int] = {}
    code = lambda rs: np.fromiter((tix.setdefault(r, len(tix)) for r in rs), dtype=np.int64, count=len(rs))
    ut = code(types)
    test, known = list(test), list(known)
    th, tr, tt = (np.array([x[0] for x in test], np.int64), code([x[1] for x in test]),
                  np.array([x[2] for x in test], np.int64))
    kh, kr, kt = (np.array([x[0] for x in known], np.int64), code([x[1] for x in known]),
                  np.array([x[2] for x in known], np.int64))
    R = max(len(tix), 1)
    H = np.concatenate([universe.edges[:, 0], kh, th])           # the test triples are known answers too
    T = np.concatenate([universe.edges[:, 1], kt, tt])
    Rr = np.concatenate([ut, kr, tr])
    sel_t = np.isin(H * R + Rr, th * R + tr)                     # answers of the tail queries (h, r, ?)
    kVt, kUt = _known_counts((H * R + Rr)[sel_t], T[sel_t], th * R + tr, among, outside, n)
    sel_h = np.isin(Rr * n + T, tr * n + tt)                     # answers of the head queries (?, r, t)
    kVh, kUh = _known_counts((Rr * n + T)[sel_h], H[sel_h], tr * n + tt, among, outside, n)
    with np.errstate(invalid='ignore', divide='ignore'):
        w = np.concatenate([(nU - kUt) / (nV - kVt), (nU - kUh) / (nV - kVh)])
    return _share_summary(w, 'filtered ranking over the universe: a query\'s candidates are the universe\'s units '
                             'minus its known answers (the universe\'s triples of its relation and the release\'s)',
                          test_triples_used=len(test), **extra)


def user_shares(universe: Universe, unit_mask: np.ndarray, test_users: Sequence[int],
                known_pairs: Sequence[Tuple[int, int]] = (), **extra) -> Dict:
    """Per-query excluded shares of all-item ranking on the target (the NGCF / LightGCN protocol). Each test user
    is a query; its candidates are the universe's items minus every item it is known to have interacted with
    (the universe's pairs and the release's, `known_pairs` as (user, item) unit indexes). W_q is the share of
    those items outside the release."""
    items = universe.part == 1
    outside = items & ~np.asarray(unit_mask, bool)
    n = universe.n
    e = universe.edges
    first_is_user = universe.part[e[:, 0]] == 0
    users = np.where(first_is_user, e[:, 0], e[:, 1])
    tu = np.unique(np.asarray(test_users, np.int64))
    sel = np.isin(users, tu)
    ku = users[sel]
    ki = np.where(first_is_user, e[:, 1], e[:, 0])[sel]
    kp = np.asarray(known_pairs, np.int64).reshape(-1, 2)
    kV, kU = _known_counts(np.concatenate([ku, kp[:, 0]]), np.concatenate([ki, kp[:, 1]]), tu,
                           items, outside, n)
    with np.errstate(invalid='ignore', divide='ignore'):
        w = (int(outside.sum()) - kU) / (int(items.sum()) - kV)
    return _share_summary(w, 'all-item ranking over the universe: a test user\'s candidates are the universe\'s '
                             'items minus every item the user is known to have interacted with (universe and release)',
                          **extra)


def depth_profile(universe: Universe, unit_mask: np.ndarray, depth: np.ndarray,
                  among: Optional[np.ndarray] = None, bins: Iterable[int] = (1, 2, 3, 5, 10, 20, 50, 100)) -> Dict:
    """Depth of included vs excluded units, and the excluded share per depth bin."""
    among = _as_mask(universe, among)
    inc, exc = depth[unit_mask & among], depth[~unit_mask & among]

    def q(x):
        return {p: float(np.quantile(x, p / 100)) for p in (10, 25, 50, 75, 90)} if len(x) else {}
    edges = list(bins) + [np.inf]
    rows = []
    lo = 0
    for hi in edges:
        m = among & (depth >= lo) & (depth < hi)
        if m.sum():
            rows.append({'depth_from': int(lo), 'depth_to': None if hi == np.inf else int(hi) - 1,
                         'units': int(m.sum()), 'excluded_share': float(1 - (m & unit_mask).sum() / m.sum())})
        lo = hi
    return {'included_depth_quantiles': q(inc), 'excluded_depth_quantiles': q(exc), 'by_depth': rows}


def derived_scope(universe: Universe, unit_mask: np.ndarray, depth: np.ndarray, among: Optional[np.ndarray] = None,
                  thresholds: Optional[Iterable[int]] = None, pairs: bool = False) -> Dict:
    """For each threshold d, the universe restricted to depth >= d and its W (unit- or pair-level).
    d_min is the smallest d with W <= 0.5, or None."""
    among = _as_mask(universe, among)
    if thresholds is None:
        top = int(np.quantile(depth[among], 0.99)) if among.any() else 1
        thresholds = range(0, max(top, 1) + 1)
    curve, d_min = [], None
    for d in thresholds:
        restrict = among & (depth >= d)
        if not restrict.any():
            break
        if pairs:
            e0, e1 = universe.edges[:, 0], universe.edges[:, 1]
            keep = unit_mask & restrict
            P = int((restrict[e0] & restrict[e1]).sum())
            Pr = int((keep[e0] & keep[e1]).sum())
            if universe.part is None:
                n, nr = int(restrict.sum()), int(keep.sum())
                cand_all, cand_rel = n * (n - 1) // 2 - P, nr * (nr - 1) // 2 - Pr
            else:
                a, b = restrict & (universe.part == 0), restrict & (universe.part == 1)
                cand_all = int(a.sum()) * int(b.sum()) - P
                cand_rel = int((a & unit_mask).sum()) * int((b & unit_mask).sum()) - Pr
            if cand_all <= 0:            # no candidate pairs left at this depth: the curve ends
                break
            w = 1 - cand_rel / cand_all
        else:
            w = 1 - (unit_mask & restrict).sum() / restrict.sum()
        curve.append({'d': int(d), 'units': int(restrict.sum()), 'included': int((unit_mask & restrict).sum()), 'W': float(w)})
        if d_min is None and w <= 0.5:
            d_min = int(d)
    return {'d_min': d_min, 'curve': curve, 'level': 'pairs' if pairs else 'units'}
