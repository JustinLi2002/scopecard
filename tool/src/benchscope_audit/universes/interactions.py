"""Bipartite universes built from raw interaction records (users x items)."""
from array import array
from typing import Iterable, Iterator, Optional, Tuple

import numpy as np

from ..core import Universe
from ..io import InputLog, basename, open_maybe_gz


def from_pair_stream(pairs: Iterator[Tuple[str, str]], name: str, version: str = '') -> Universe:
    """Like from_pairs, for tens of millions of interactions: ids are encoded while streaming and
    duplicates removed on integer keys, so no set of string tuples is built. Units are users in
    order of first appearance ('u:' prefix), then items ('i:')."""
    uid, iid = {}, {}
    uc, ic = array('q'), array('q')
    for u, i in pairs:
        uc.append(uid.setdefault(u, len(uid)))
        ic.append(iid.setdefault(i, len(iid)))
    users, items = list(uid), list(iid)
    del uid, iid
    nU, nI = len(users), len(items)
    keys = np.unique(np.frombuffer(uc, dtype=np.int64) * nI + np.frombuffer(ic, dtype=np.int64))
    del uc, ic
    edges = np.stack([keys // nI, keys % nI + nU], axis=1)
    del keys
    units = np.empty(nU + nI, dtype=object)
    units[:nU] = ['u:' + u for u in users]
    del users
    units[nU:] = ['i:' + i for i in items]
    del items
    part = np.zeros(nU + nI, dtype=np.int8)
    part[nU:] = 1
    return Universe(name=name, version=version, units=units, edges=edges, part=part)


def load_ratings_csv(path: str, name: str = '', log: Optional[InputLog] = None) -> Universe:
    """Amazon 2014 ratings-only CSV (user,item,rating,timestamp; e.g. ratings_Books.csv): one interaction
    per distinct (user, item). Lines with fewer than three comma-separated fields are skipped and counted."""
    st = {'lines': 0, 'skipped': 0, 'first': None}

    def rows():
        with open_maybe_gz(path) as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                st['lines'] += 1
                p = line.split(',', 2)
                if len(p) == 3 and p[0] and p[1]:
                    yield p[0], p[1]
                else:
                    st['skipped'] += 1
                    st['first'] = st['first'] or (n, line.strip())
    fn = basename(path)
    uni = from_pair_stream(rows(), name=name or fn, version=fn)
    (log or InputLog()).record(path, st['lines'], st['skipped'], 'user,item,rating,timestamp', st['first'])
    return uni


def from_pairs(pairs: Iterable[Tuple[str, str]], name: str, version: str = '') -> Universe:
    """Distinct (user, item) pairs -> bipartite universe. Unit ids are prefixed 'u:' and 'i:'."""
    pairs = sorted(set(pairs))
    users = sorted({u for u, _ in pairs})
    items = sorted({i for _, i in pairs})
    units = ['u:' + u for u in users] + ['i:' + i for i in items]
    ui = {u: k for k, u in enumerate(users)}
    ii = {i: len(users) + k for k, i in enumerate(items)}
    edges = np.array([(ui[u], ii[i]) for u, i in pairs], dtype=np.int64)
    part = np.array([0] * len(users) + [1] * len(items), dtype=np.int8)
    return Universe(name=name, version=version, units=np.array(units, dtype=object), edges=edges, part=part)


def load_snap_checkins(path: str, name: str = 'SNAP check-ins', log: Optional[InputLog] = None) -> Universe:
    """SNAP loc-*_totalCheckins.txt(.gz): user, time, lat, lon, location id; one interaction per
    distinct (user, location). Lines without five fields are skipped and counted."""
    pairs = set()
    lines = skipped = 0
    first = None
    with open_maybe_gz(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            lines += 1
            tok = line.split()
            if len(tok) == 5:
                pairs.add((tok[0], tok[4]))
            else:
                skipped += 1
                first = first or (n, line.strip())
    (log or InputLog()).record(path, lines, skipped, '5 whitespace-separated fields: user, time, latitude, '
                                                     'longitude, location', first)
    return from_pairs(pairs, name=name, version=basename(path))
