"""Shared candidate-cell counting for scorers that are not LibKGE models (T0 popularity,
NBFNet). It reproduces score_wn18rr.py exactly: same filters, same cells, same tie rule
(torch.isclose, LibKGE's atol/rtol), same output files. check_cells_module.py verifies the
equality on LibKGE checkpoints.

A scorer is called as scorer(side, h, rels, t) with index tensors h, t and relation names,
and returns a (B, N) tensor of candidate scores (tails for side 'tail', heads for 'head')
over the entity order given by `names` (synset ids for E2 / deploy, bare offsets for E1).
"""
import gzip
import hashlib
import json
import os

import numpy as np
import torch

BINS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 10 ** 9)]
ATOL, RTOL = 1e-5, 1e-4          # LibKGE entity_ranking.tie_handling defaults


def read_tsv(path):
    with open(path, encoding='utf-8') as f:
        head = f.readline().rstrip('\n').split('\t')
        return [dict(zip(head, line.rstrip('\n').split('\t'))) for line in f]


def deg_bin(d):
    for k, (lo, hi) in enumerate(BINS):
        if lo <= d <= hi:
            return k
    raise ValueError(d)


class Target:
    def __init__(self, target_dir):
        self.E = read_tsv(os.path.join(target_dir, 'entities.tsv'))
        self.T = read_tsv(os.path.join(target_dir, 'triples.tsv'))
        self.ent = {e['entity_id']: e for e in self.E}

    def queries(self, role, key=lambda x: x):
        return [(t['triple_id'], key(t['head']), t['relation'], key(t['tail'])) for t in self.T if t['role'] == role]

    def all_triples(self, key=lambda x: x):
        return [(key(t['head']), t['relation'], key(t['tail'])) for t in self.T]

    def train_triples(self):
        return [(t['head'], t['relation'], t['tail']) for t in self.T if t['role'] in ('bench_train', 'extra')]


def e2_cells(target, names, device):
    """base cell per entity (region * 14 + bin * 2) and POS-compatibility per (relation, side)."""
    slot = {}
    for h, r, t in target.train_triples():
        slot.setdefault((r, 'head'), set()).add(h[0])
        slot.setdefault((r, 'tail'), set()).add(t[0])
    base = np.zeros(len(names), dtype=np.int64)
    pos = []
    for i, n in enumerate(names):
        e = target.ent[n]
        d = int(e['train_degree'])
        region = 2 if e['region'] == 'U' else (0 if d > 0 else 1)
        base[i] = region * 14 + deg_bin(d) * 2
        pos.append(e['pos'])
    pos_ok = {k: torch.tensor([p in v for p in pos], dtype=torch.long, device=device) for k, v in slot.items()}
    return torch.tensor(base, device=device), pos_ok


def build_filters(triples, idx):
    tails, heads = {}, {}
    for h, r, t in triples:
        if h in idx and t in idx:
            tails.setdefault((idx[h], r), set()).add(idx[t])
            heads.setdefault((r, idx[t]), set()).add(idx[h])
    return tails, heads


def run(scorer, queries, names, filters, cells, out, meta, batch=64, device='cuda'):
    """queries: list of (triple_id, h_name, r_name, t_name, mapped). cells: None (one cell) or (base, pos_ok)."""
    os.makedirs(out, exist_ok=True)
    idx = {n: i for i, n in enumerate(names)}
    N = len(names)
    tails, heads = filters
    n_cells = 1 if cells is None else 42
    rows_q, rows_c = [], []
    with torch.no_grad():
        for side in ('tail', 'head'):
            for b0 in range(0, len(queries), batch):
                bq = queries[b0:b0 + batch]
                h = torch.tensor([idx[q[1]] for q in bq], device=device)
                o = torch.tensor([idx[q[3]] for q in bq], device=device)
                rels = [q[2] for q in bq]
                scores = scorer(side, h, rels, o)
                true = o if side == 'tail' else h
                pscore = scores.gather(1, true.view(-1, 1)).view(-1)
                valid = torch.ones_like(scores, dtype=torch.bool)
                for k, q in enumerate(bq):
                    known = tails.get((idx[q[1]], q[2]), set()) if side == 'tail' else heads.get((q[2], idx[q[3]]), set())
                    valid[k, list(known | {int(true[k])})] = False
                close = torch.isclose(scores, pscore.view(-1, 1), rtol=RTOL, atol=ATOL)
                above = (scores > pscore.view(-1, 1)) & ~close
                if cells is None:
                    cid = torch.zeros_like(scores, dtype=torch.long)
                else:
                    base, pos_ok = cells
                    pok = torch.stack([pos_ok.get((r, side), torch.zeros(N, dtype=torch.long, device=device)) for r in rels])
                    cid = base.view(1, -1) + pok
                res = []
                for m in (valid, valid & above, valid & close):
                    cnt = torch.zeros(len(bq), n_cells, dtype=torch.long, device=device)
                    cnt.scatter_add_(1, cid, m.long())
                    res.append(cnt.cpu().numpy())
                cn, ca, ct = res
                for k, q in enumerate(bq):
                    qid = f'{q[0]}:{side}'
                    rank = int(ca[k].sum()) + (int(ct[k].sum()) + 1) // 2 + 1
                    rows_q.append(f'{qid}\t{q[0]}\t{side}\t{q[2]}\t{names[int(true[k])]}\t'
                                  f'{float(pscore[k]):.17g}\t{int(q[4])}\t{rank}\n')
                    for c in np.flatnonzero(cn[k]):
                        rows_c.append(f'{qid}\t{c}\t{cn[k, c]}\t{ca[k, c]}\t{ct[k, c]}\n')
    with gzip.open(os.path.join(out, 'queries.tsv.gz'), 'wt', encoding='utf-8') as f:
        f.write('query_id\ttriple_id\tside\trelation\tpositive\tpos_score\tmapped\trank_all\n')
        f.writelines(rows_q)
    with gzip.open(os.path.join(out, 'cells.tsv.gz'), 'wt', encoding='utf-8') as f:
        f.write('query_id\tcell\tn\tabove\tties\n')
        f.writelines(rows_c)
    ranks = np.array([int(r.rstrip('\n').split('\t')[-1]) for r in rows_q])
    meta = dict(meta, n_queries=len(rows_q), n_entities=N, atol=ATOL, rtol=RTOL,
                mrr_all_candidates=float(np.mean(1.0 / ranks)), hits10_all_candidates=float(np.mean(ranks <= 10)))
    with open(os.path.join(out, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    return meta


def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()
