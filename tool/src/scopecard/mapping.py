"""Mapping benchmark ids onto universe units, with a report of what did not map cleanly.

Two routes:
  map_units    direct: a release id maps to every universe unit whose key equals it.
  map_triples  through relations: each release triple (h, r, t) is matched to universe
               triples with the same relation and keyed ends. This resolves ids that are
               ambiguous under the key (WN18's bare WordNet offsets are shared across parts
               of speech) and exposes ids the release used for more than one universe unit.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import Release, Universe

Key = Optional[Callable[[str], str]]


@dataclass
class MappingReport:
    route: str
    n_release_ids: int = 0
    n_mapped_ids: int = 0
    unmapped_ids: List[str] = field(default_factory=list)
    ambiguous_ids: Dict[str, List[str]] = field(default_factory=dict)   # direct route: id -> candidates
    merged_ids: Dict[str, List[str]] = field(default_factory=dict)      # triple route: id used for >1 unit
    n_release_triples: int = 0
    n_matched_triples: int = 0
    unmatched_triples: List[Tuple[str, str, str]] = field(default_factory=list)
    n_ambiguous_triples: int = 0

    def to_dict(self, examples: int = 5) -> dict:
        return {
            'route': self.route,
            'release_ids': self.n_release_ids,
            'mapped_ids': self.n_mapped_ids,
            'unmapped_ids': len(self.unmapped_ids),
            'unmapped_examples': sorted(self.unmapped_ids)[:examples],
            'ambiguous_ids': len(self.ambiguous_ids),
            'merged_ids': len(self.merged_ids),
            'merged_examples': {k: v for k, v in sorted(self.merged_ids.items())[:examples]},
            'release_triples': self.n_release_triples,
            'matched_triples': self.n_matched_triples,
            'unmatched_triples': len(self.unmatched_triples),
            'unmatched_examples': [list(t) for t in self.unmatched_triples[:examples]],
            'ambiguous_triples': self.n_ambiguous_triples,
        }


def _reverse(universe: Universe, key: Key) -> Dict[str, List[int]]:
    rev = defaultdict(list)
    for i, u in enumerate(universe.units.tolist()):
        rev[key(u) if key else u].append(i)
    return rev


def map_units(release_ids: Sequence[str], universe: Universe, key: Key = None,
              ambiguous: str = 'all', name: str = 'release') -> Tuple[Release, MappingReport]:
    """Map ids directly. `ambiguous` = 'all' keeps every candidate unit, 'none' drops the id."""
    if key is None:          # ids are unique universe ids: reuse the universe's index (no second large dict)
        lookup = lambda r: [universe.index[r]] if r in universe.index else []
    else:
        rev = _reverse(universe, key)
        lookup = lambda r: rev.get(r, [])
    rep = MappingReport(route='direct')
    mask = np.zeros(universe.n, bool)
    ids = sorted(set(release_ids))
    rep.n_release_ids = len(ids)
    for r in ids:
        cand = lookup(r)
        if not cand:
            rep.unmapped_ids.append(r)
            continue
        if len(cand) > 1:
            rep.ambiguous_ids[r] = [universe.units[c] for c in cand]
            if ambiguous == 'none':
                continue
        mask[cand] = True
        rep.n_mapped_ids += 1
    return Release(name=name, unit_mask=mask), rep


def match_triples_exact(splits: Dict[str, Sequence[Tuple[str, str, str]]], universe: Universe,
                        report: Optional[MappingReport] = None) -> Tuple[MappingReport, Dict[str, int]]:
    """Count release triples that are universe triples, when release ids equal universe unit ids
    (no key). Vectorised, for universes with millions of triples (YAGO). Fills the triple fields
    of `report` and returns it with matched counts per split."""
    if universe.edge_types is None:
        raise ValueError('match_triples_exact needs a universe with edge types')
    rep = report or MappingReport(route='direct')
    types = sorted(set(universe.edge_types.tolist()))
    tix = {r: i for i, r in enumerate(types)}
    n, R = universe.n, len(types)
    ukey = (universe.edges[:, 0] * R + np.array([tix[r] for r in universe.edge_types.tolist()], dtype=np.int64)) * n \
        + universe.edges[:, 1]
    ukey = np.unique(ukey)
    by_split = {}
    for s, trips in splits.items():
        h = np.array([universe.index.get(x, -1) for x, _, _ in trips], dtype=np.int64)
        r = np.array([tix.get(x, -1) for _, x, _ in trips], dtype=np.int64)
        t = np.array([universe.index.get(x, -1) for _, _, x in trips], dtype=np.int64)
        ok = (h >= 0) & (r >= 0) & (t >= 0)
        hit = np.zeros(len(trips), bool)
        hit[ok] = np.isin((h[ok] * R + r[ok]) * n + t[ok], ukey)
        by_split[s] = int(hit.sum())
        rep.n_release_triples += len(trips)
        rep.n_matched_triples += int(hit.sum())
        rep.unmatched_triples += [trips[i] for i in np.flatnonzero(~hit)]
    return rep, by_split


def map_triples(splits: Dict[str, Sequence[Tuple[str, str, str]]], universe: Universe, key: Key = None,
                name: str = 'release') -> Tuple[Release, MappingReport]:
    """Map release triples (per split) onto universe triples with the same relation and keyed ends."""
    if universe.edge_types is None:
        raise ValueError('map_triples needs a universe with edge types')
    units = universe.units.tolist()
    k = (lambda x: key(x)) if key else (lambda x: x)
    lookup = defaultdict(list)
    for j, ((h, t), r) in enumerate(zip(universe.edges.tolist(), universe.edge_types.tolist())):
        lookup[(k(units[h]), r, k(units[t]))].append(j)
    rep = MappingReport(route='triples')
    used_for = defaultdict(set)
    edges, types, split_idx = [], [], {}
    seen_ids = set()
    for s, trips in splits.items():
        rows = []
        for h, r, t in trips:
            rep.n_release_triples += 1
            seen_ids.update((h, t))
            m = lookup.get((h, r, t), [])
            if not m:
                rep.unmatched_triples.append((h, r, t))
                continue
            rep.n_matched_triples += 1
            if len(m) > 1:
                rep.n_ambiguous_triples += 1
            for j in m:
                a, b = universe.edges[j]
                used_for[h].add(int(a))
                used_for[t].add(int(b))
                rows.append(len(edges))
                edges.append((int(a), int(b)))
                types.append(r)
        split_idx[s] = np.asarray(rows, dtype=np.int64)
    rep.n_release_ids = len(seen_ids)
    rep.n_mapped_ids = len(used_for)
    rep.unmapped_ids = sorted(seen_ids - set(used_for))
    rep.merged_ids = {r: sorted(units[u] for u in us) for r, us in used_for.items() if len(us) > 1}
    mask = np.zeros(universe.n, bool)
    if edges:
        e = np.asarray(edges, dtype=np.int64)
        mask[e.ravel()] = True
    else:
        e = np.zeros((0, 2), dtype=np.int64)
    rel = Release(name=name, unit_mask=mask, edges=e, edge_types=np.asarray(types, dtype=object), splits=split_idx)
    return rel, rep


# A release whose ids mostly do not occur in the universe is not a release of that universe (wrong files, wrong
# audit type or wrong universe). Below MAP_REFUSE the audit stops; below MAP_WARN the card carries a warning.
MAP_REFUSE, MAP_WARN = 0.5, 0.95


def check_mapping(rep: Dict, universe_name: str, what: str = 'release ids') -> List[str]:
    """Raise InputError when fewer than MAP_REFUSE of the release ids map onto the universe; return a warning
    when fewer than MAP_WARN do. `rep` is MappingReport.to_dict()."""
    from .io import InputError
    n, m = rep['release_ids'], rep['mapped_ids']
    share = m / n if n else 0.0
    if share < MAP_REFUSE:
        raise InputError(f'Only {m:,} of the {n:,} {what} ({share:.1%}) occur in {universe_name}, so these files do not look '
                         f'like a release of this universe. Check the audit type, the files and the universe (the audit '
                         f'stops when fewer than {MAP_REFUSE:.0%} of the ids match).')
    if share < MAP_WARN:
        return [f'Only {m:,} of the {n:,} {what} ({share:.1%}) occur in {universe_name}. The release has units this '
                f'universe does not contain, so it may not be the universe the benchmark was drawn from.']
    return []
