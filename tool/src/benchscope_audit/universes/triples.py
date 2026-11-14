"""Universes of entities related by typed triples: a generic TSV loader and YAGO 3's Turtle themes."""
from typing import Iterable, Optional, Set, Tuple

import numpy as np

from ..core import Universe
from ..io import InputLog, basename, open_maybe_gz


def from_triples(triples: Iterable[Tuple[str, str, str]], name: str, version: str = '',
                 relations: Optional[Set[str]] = None) -> Universe:
    """Distinct (head, relation, tail) triples -> universe of the entities that carry at least one.
    `relations` restricts the triples (and so the entities) to those relation types."""
    trip = sorted({x for x in triples if relations is None or x[1] in relations})
    units = sorted({e for h, _, t in trip for e in (h, t)})
    idx = {u: i for i, u in enumerate(units)}
    edges = np.array([(idx[h], idx[t]) for h, _, t in trip], dtype=np.int64).reshape(-1, 2)
    types = np.array([r for _, r, _ in trip], dtype=object)
    return Universe(name=name, version=version, units=np.array(units, dtype=object), edges=edges, edge_types=types)


def load_triples_tsv(path: str, name: str = '', relations: Optional[Set[str]] = None,
                     log: Optional[InputLog] = None) -> Universe:
    """Tab-separated head, relation, tail per line (.gz allowed). Other lines are skipped and counted."""
    st = {'lines': 0, 'skipped': 0, 'first': None}

    def rows():
        with open_maybe_gz(path) as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                st['lines'] += 1
                tok = line.rstrip('\r\n').split('\t')
                if len(tok) == 3 and tok[0] and tok[2]:
                    yield tok[0], tok[1], tok[2]
                else:
                    st['skipped'] += 1
                    st['first'] = st['first'] or (n, line.strip())
    uni = from_triples(rows(), name=name or basename(path), version='', relations=relations)
    (log or InputLog()).record(path, st['lines'], st['skipped'], '3 tab-separated fields: head, relation, tail',
                               st['first'])
    return uni


def load_yago3_facts(path: str, relations: Optional[Set[str]] = None) -> Universe:
    """YAGO 3 Turtle theme (e.g. yagoFacts.ttl of 3.0.2): '<s>\\t<p>\\t<o>\\t.' lines; only facts whose
    object is an entity (in angle brackets) are kept. Brackets are stripped, so ids match the
    ConvE release (YAGO3-10)."""
    def rows():
        with open_maybe_gz(path) as f:
            for line in f:
                if not line.startswith('<'):
                    continue
                p = line.rstrip('\n').split('\t')
                if len(p) < 3 or not p[2].startswith('<'):
                    continue
                yield p[0][1:-1], p[1][1:-1], p[2][1:-1]
    fn = path.replace('\\', '/').rsplit('/', 1)[-1]
    version = '3'
    with open_maybe_gz(path) as f:                 # header: '# of the YAGO knowledge base, version 3.0.2.'
        for _, line in zip(range(20), f):
            if line.startswith('#') and 'version' in line:
                version = line.split('version', 1)[1].strip().rstrip('.')
                break
    return from_triples(rows(), name='YAGO ' + version, version=fn, relations=relations)
