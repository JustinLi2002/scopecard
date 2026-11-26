"""Data model: a declared universe of units with the source relations that give each unit a depth,
and a release expressed as the universe units it contains."""
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


def counts(x, n: int) -> np.ndarray:
    """np.bincount with minlength n for index arrays of any integer type. Unit indexes are stored as int64;
    bincount needs the platform's intp, which is int32 in the browser (Pyodide, wasm32)."""
    return np.bincount(np.asarray(x).astype(np.intp, copy=False), minlength=n)


@dataclass
class Universe:
    """Units of a declared target population.

    units       unit ids, unique (strings)
    edges       (m, 2) int array of unit indexes, one row per distinct source relation instance
                (a triple, an interaction); a unit's depth is the number of rows it occurs in
    edge_types  (m,) relation labels, or None
    part        (n,) 0/1 for bipartite universes (e.g. users / items), None if unipartite
    strata      unit attributes used to break results down (e.g. part of speech)
    """
    name: str
    units: np.ndarray
    edges: np.ndarray
    edge_types: Optional[np.ndarray] = None
    part: Optional[np.ndarray] = None
    strata: Dict[str, np.ndarray] = field(default_factory=dict)
    version: str = ''

    def __post_init__(self):
        self.units = np.asarray(self.units, dtype=object)
        self.edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        self.index = {u: i for i, u in enumerate(self.units.tolist())}
        if len(self.index) != len(self.units):
            raise ValueError('universe unit ids must be unique')

    @property
    def n(self) -> int:
        return len(self.units)

    def edge_mask(self, types=None) -> np.ndarray:
        """Boolean mask of edges whose type is in `types` (all edges if None)."""
        if types is None or self.edge_types is None:
            return np.ones(len(self.edges), bool)
        return np.isin(self.edge_types, np.asarray(sorted(types), dtype=object))

    def depth(self, edge_mask: Optional[np.ndarray] = None) -> np.ndarray:
        e = self.edges if edge_mask is None else self.edges[edge_mask]
        return counts(e.ravel(), self.n)

    def fingerprint(self) -> str:
        """SHA-256 of the universe's content: its unit ids and its typed relations, independent of unit order
        and of the files it was read from (WordNet's dict folder and the browser's table give the same value)."""
        order = np.argsort(self.units, kind='stable')
        rank = np.empty(self.n, dtype=np.int64)
        rank[order] = np.arange(self.n, dtype=np.int64)
        h = hashlib.sha256()
        h.update(('\n'.join(self.units[order].tolist()) + '\n').encode('utf-8'))
        if self.edge_types is None:
            T, tid = 1, np.zeros(len(self.edges), dtype=np.int64)
        else:
            types = sorted(set(self.edge_types.tolist()))
            h.update(('\t'.join(types) + '\n').encode('utf-8'))
            tix = {t: i for i, t in enumerate(types)}
            T, tid = len(types), np.array([tix[t] for t in self.edge_types.tolist()], dtype=np.int64)
        if len(self.edges):
            keys = np.sort((rank[self.edges[:, 0]] * T + tid) * self.n + rank[self.edges[:, 1]])
            h.update(keys.astype('<i8').tobytes())
        return 'sha256:' + h.hexdigest()


@dataclass
class Release:
    """A benchmark release mapped onto a universe.

    unit_mask   (n,) bool: universe units that occur in the release
    edges       (k, 2) universe unit indexes of the release's positive relations, or None
    edge_types  (k,) labels, or None
    splits      split name -> index array into `edges` (e.g. train / valid / test)
    """
    name: str
    unit_mask: np.ndarray
    edges: Optional[np.ndarray] = None
    edge_types: Optional[np.ndarray] = None
    splits: Dict[str, np.ndarray] = field(default_factory=dict)
    files_sha256: Dict[str, str] = field(default_factory=dict)
