"""WordNet 3.0 as a universe of synsets related by WN18-style relation types.

Synset identity is (POS, offset); satellite adjectives share data.adj's offsets and are keyed
'a'. Lexical pointers are lifted to synsets and duplicates collapsed, as WN18 does.
identity='bare_offset' reproduces WN18's own keying, which merges synsets of different POS
that share an offset (use it only to reproduce the benchmark's convention).
"""
import gzip
import os
import re
from typing import Iterable

import numpy as np

from ..core import Universe

# WN18 relation name -> WordNet 3.0 pointer symbol; triple read as (source, relation, target)
WN18_RELATIONS = {
    '_hypernym': '@', '_hyponym': '~', '_instance_hypernym': '@i', '_instance_hyponym': '~i',
    '_member_holonym': '#m', '_member_meronym': '%m', '_part_of': '#p', '_has_part': '%p',
    '_derivationally_related_form': '+', '_synset_domain_topic_of': ';c', '_member_of_domain_topic': '-c',
    '_synset_domain_region_of': ';r', '_member_of_domain_region': '-r', '_synset_domain_usage_of': ';u',
    '_member_of_domain_usage': '-u', '_also_see': '^', '_verb_group': '$', '_similar_to': '&',
}
WN18RR_RELATIONS = ('_hypernym', '_instance_hypernym', '_member_meronym', '_has_part', '_derivationally_related_form',
                    '_synset_domain_topic_of', '_member_of_domain_region', '_member_of_domain_usage', '_also_see',
                    '_verb_group', '_similar_to')


def load_wordnet30(dict_dir: str, relations: Iterable[str] = tuple(WN18_RELATIONS), identity: str = 'pos_offset',
                   include_isolated: bool = True) -> Universe:
    """Universe of WordNet 3.0 synsets. `include_isolated` keeps synsets with no triple of the
    chosen relations (they then have depth 0)."""
    sym2rel = {WN18_RELATIONS[r]: r for r in relations}
    synsets, triples = {}, set()
    for posfile in ('noun', 'verb', 'adj', 'adv'):
        with open(os.path.join(dict_dir, 'data.' + posfile), encoding='latin-1') as f:
            for line in f:
                if line.startswith('  '):
                    continue
                tok = line.split(' | ')[0].split()
                off, sstype = tok[0], tok[2]
                pos = 'a' if sstype == 's' else sstype
                sid = pos + off if identity == 'pos_offset' else off
                synsets.setdefault(sid, set()).add(pos)
                i = 4 + 2 * int(tok[3], 16)
                for _ in range(int(tok[i])):
                    sym, toff, tpos = tok[i + 1], tok[i + 2], tok[i + 3]
                    i += 4
                    if sym in sym2rel:
                        tid = ('a' if tpos == 's' else tpos) + toff if identity == 'pos_offset' else toff
                        triples.add((sid, sym2rel[sym], tid))
    return _build(synsets, triples, relations, identity, include_isolated)


def _build(synsets, triples, relations, identity, include_isolated) -> Universe:
    connected = {x for h, _, t in triples for x in (h, t)}
    units = sorted(synsets if include_isolated else connected)
    idx = {u: i for i, u in enumerate(units)}
    trip = sorted(triples)
    edges = np.array([(idx[h], idx[t]) for h, _, t in trip], dtype=np.int64)
    types = np.array([r for _, r, _ in trip], dtype=object)
    pos = np.array(['/'.join(sorted(synsets[u])) for u in units], dtype=object)
    return Universe(name='WordNet 3.0', version=f'{identity}, {len(set(relations))} relations', units=np.array(units, dtype=object),
                    edges=edges, edge_types=types, strata={'pos': pos})


TABLE_HEADER = 'scopecard WordNet 3.0 table: one synset id (POS + offset) per line, then head/relation/tail of the 18 WN18 relations'


def export_wordnet30_table(dict_dir: str, out_path: str) -> int:
    """Write the compact WordNet 3.0 table that the browser page ships instead of the dict files. WordNet's
    licence notice (read from data.adv) is copied into the header, as the licence requires on all copies.
    The gzip stream has no timestamp, so rebuilds are byte-identical. Returns the number of lines written."""
    uni = load_wordnet30(dict_dir, tuple(WN18_RELATIONS), identity='pos_offset', include_isolated=True)
    notice = []
    with open(os.path.join(dict_dir, 'data.adv'), encoding='latin-1') as f:
        for line in f:
            m = re.match(r'^\s+\d+ (.*?)\s*$', line)
            if not line.startswith('  ') or not m:
                break
            notice.append(m.group(1))
    units = uni.units.tolist()
    rows = ['# ' + TABLE_HEADER] + ['# ' + x if x else '#' for x in notice]
    rows += units
    rows += [f'{units[h]}\t{r}\t{units[t]}' for (h, t), r in zip(uni.edges.tolist(), uni.edge_types.tolist())]
    with open(out_path, 'wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as gz:
        gz.write(('\n'.join(rows) + '\n').encode('utf-8'))
    return len(rows)


def load_wordnet30_table(path: str, relations: Iterable[str] = tuple(WN18_RELATIONS), identity: str = 'pos_offset',
                         include_isolated: bool = True) -> Universe:
    """The universe of load_wordnet30, built from the table of export_wordnet30_table (identical result;
    the bare-offset identity is derived by dropping the POS letter)."""
    rels = set(relations)
    synsets, triples = {}, set()
    with gzip.open(path, 'rt', encoding='utf-8') if path.endswith('.gz') else open(path, encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                continue
            p = line.rstrip('\n').split('\t')
            if len(p) == 1:
                synsets.setdefault(p[0] if identity == 'pos_offset' else p[0][1:], set()).add(p[0][0])
            elif p[1] in rels:
                triples.add((p[0], p[1], p[2]) if identity == 'pos_offset' else (p[0][1:], p[1], p[2][1:]))
    return _build(synsets, triples, relations, identity, include_isolated)


def load_wordnet30_source(source: str, relations: Iterable[str] = tuple(WN18_RELATIONS), identity: str = 'pos_offset',
                          include_isolated: bool = True) -> Universe:
    """A WordNet-3.0/dict folder or a table file from export_wordnet30_table."""
    if os.path.isdir(source):
        return load_wordnet30(source, relations, identity, include_isolated)
    return load_wordnet30_table(source, relations, identity, include_isolated)
