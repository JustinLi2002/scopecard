"""Per-query scorer for the WN18RR measured-c experiment (MEASURED_C_DESIGN.md Sect. 4, 5, 9).

For every query (head and tail side of each test triple) it records, per candidate
cell, how many filtered candidates there are (n), how many score strictly above the
positive (above), and how many tie with it (ties). Ties use LibKGE's rule
(torch.isclose with the run's atol/rtol), so ranks reproduce LibKGE's
rounded_mean_rank exactly: rank = above + (ties + 1) // 2 + 1, the +1 inside being
the positive itself.

Cells (E2): region x training-degree bin x POS-compatible flag.
  region   0 = O with >= 1 E2 training triple, 1 = O without, 2 = U
  deg bin  0 = no training triple, 1, 2, 3, 4-5, 6-10, 11+  (codes 0..6)
  pos_ok   1 if the candidate's POS occurs in that relation slot in E2 training
  cell = region * 14 + bin * 2 + pos_ok
E1 has one cell (0): every candidate is a benchmark entity.

Filtering: --filter target removes every triple of the declared target (all roles);
--filter splits uses only the run's own train/valid/test (LibKGE's convention), for
cross-checking against `kge test`.

usage (in ~/bs4/env):
  python score_wn18rr.py --checkpoint CKPT --mode {E1,E2} --target DIR --queries {bench_test,bench_valid,u_test}
                         --filter {target,splits} [--all-wn18rr-test] --out DIR
"""
import argparse
import gzip
import hashlib
import json
import os

import numpy as np
import torch
from kge.model import KgeModel
from kge.util.io import load_checkpoint

BINS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 10 ** 9)]


def read_tsv(path):
    with open(path, encoding='utf-8') as f:
        head = f.readline().rstrip('\n').split('\t')
        return [dict(zip(head, line.rstrip('\n').split('\t'))) for line in f]


def deg_bin(d):
    for k, (lo, hi) in enumerate(BINS):
        if lo <= d <= hi:
            return k
    raise ValueError(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--mode', choices=['E1', 'E2'], required=True)
    ap.add_argument('--target', required=True)
    ap.add_argument('--queries', default='bench_test')
    ap.add_argument('--filter', choices=['target', 'splits'], default='target')
    ap.add_argument('--all-wn18rr-test', action='store_true',
                    help='E1 only: score all 3,134 original WN18RR test triples (cross-check with kge test)')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt = load_checkpoint(a.checkpoint, dev)
    model = KgeModel.create_from(ckpt)
    model.eval()
    ds = model.dataset
    cfg = model.config
    atol = float(cfg.get('entity_ranking.tie_handling.atol'))
    rtol = float(cfg.get('entity_ranking.tie_handling.rtol'))
    ent_names = list(ds.entity_ids())
    rel_names = list(ds.relation_ids())
    e_idx = {n: i for i, n in enumerate(ent_names)}
    r_idx = {n: i for i, n in enumerate(rel_names)}
    N = len(ent_names)

    E = read_tsv(os.path.join(a.target, 'entities.tsv'))
    T = read_tsv(os.path.join(a.target, 'triples.tsv'))
    key = (lambda sid: sid) if a.mode == 'E2' else (lambda sid: sid[1:])   # E1 uses bare offsets

    # queries: (triple_id, h, r, t, mapped flag)
    if a.mode == 'E1' and a.all_wn18rr_test:
        test = ds.split('test').tolist()
        mapped = {(key(t['head']), t['relation'], key(t['tail'])): t['triple_id'] for t in T if t['role'] == 'bench_test'}
        Q = []
        for s, p, o in test:
            trip = (ent_names[s], rel_names[p], ent_names[o])
            Q.append((mapped.get(trip, 'unmapped:' + '|'.join(trip)), s, p, o, trip in mapped))
    else:
        Q = [(t['triple_id'], e_idx[key(t['head'])], r_idx[t['relation']], e_idx[key(t['tail'])], True)
             for t in T if t['role'] == a.queries]

    # filter sets
    if a.filter == 'target':
        trip = [(key(t['head']), t['relation'], key(t['tail'])) for t in T]
        trip = [(e_idx[h], r_idx[r], e_idx[t]) for h, r, t in trip if h in e_idx and t in e_idx]
    else:
        trip = [tuple(x) for s in ('train', 'valid', 'test') for x in ds.split(s).tolist()]
    tails, heads = {}, {}
    for h, r, t in trip:
        tails.setdefault((h, r), set()).add(t)
        heads.setdefault((r, t), set()).add(h)

    # candidate cells
    cell = np.zeros(N, dtype=np.int64)
    if a.mode == 'E2':
        ent = {e['entity_id']: e for e in E}
        slot = {}
        for t in T:
            if t['role'] in ('bench_train', 'extra'):
                slot.setdefault((t['relation'], 'head'), set()).add(t['head'][0])
                slot.setdefault((t['relation'], 'tail'), set()).add(t['tail'][0])
        base = np.zeros(N, dtype=np.int64)
        pos = np.empty(N, dtype=object)
        for i, n in enumerate(ent_names):
            e = ent[n]
            d = int(e['train_degree'])
            region = 2 if e['region'] == 'U' else (0 if d > 0 else 1)
            base[i] = region * 14 + deg_bin(d) * 2
            pos[i] = e['pos']
        pos_ok = {k: torch.tensor([p in v for p in pos], dtype=torch.long, device=dev) for k, v in slot.items()}
    n_cells = 42 if a.mode == 'E2' else 1
    base_t = torch.tensor(cell if a.mode == 'E1' else base, device=dev)

    all_ent = torch.arange(N, device=dev)   # LibKGE passes the entity range explicitly
    rows_q, rows_c = [], []
    with torch.no_grad():
        for side in ('tail', 'head'):
            for b0 in range(0, len(Q), a.batch):
                batch = Q[b0:b0 + a.batch]
                s = torch.tensor([q[1] for q in batch], device=dev)
                p = torch.tensor([q[2] for q in batch], device=dev)
                o = torch.tensor([q[3] for q in batch], device=dev)
                # score_sp_po is the call LibKGE's evaluator uses; score_sp / score_po can differ
                # in the last float bits and move a near-tie across the tolerance boundary
                both = model.score_sp_po(s, p, o, all_ent)
                scores = both[:, :N] if side == 'tail' else both[:, N:]
                true = o if side == 'tail' else s
                pscore = scores.gather(1, true.view(-1, 1)).view(-1)
                valid = torch.ones_like(scores, dtype=torch.bool)
                for k, q in enumerate(batch):
                    known = tails.get((q[1], q[2]), set()) if side == 'tail' else heads.get((q[2], q[3]), set())
                    idx = list(known | {int(true[k])})
                    valid[k, idx] = False
                close = torch.isclose(scores, pscore.view(-1, 1), rtol=rtol, atol=atol)
                above = (scores > pscore.view(-1, 1)) & ~close
                if a.mode == 'E2':
                    rel_list = [rel_names[q[2]] for q in batch]
                    pok = torch.stack([pos_ok.get((r, side), torch.zeros(N, dtype=torch.long, device=dev)) for r in rel_list])
                    cid = base_t.view(1, -1) + pok
                else:
                    cid = base_t.view(1, -1).expand_as(scores)
                for name, m in (('n', valid), ('above', valid & above), ('ties', valid & close)):
                    cnt = torch.zeros(len(batch), n_cells, dtype=torch.long, device=dev)
                    cnt.scatter_add_(1, cid, m.long())
                    if name == 'n':
                        cn = cnt
                    elif name == 'above':
                        ca = cnt
                    else:
                        ct = cnt
                cn, ca, ct = cn.cpu().numpy(), ca.cpu().numpy(), ct.cpu().numpy()
                tot_above = ca.sum(1)
                tot_ties = ct.sum(1)
                for k, q in enumerate(batch):
                    qid = f'{q[0]}:{side}'
                    rank = int(tot_above[k]) + (int(tot_ties[k]) + 1) // 2 + 1
                    rows_q.append(f'{qid}\t{q[0]}\t{side}\t{rel_names[q[2]]}\t{ent_names[int(true[k])]}\t'
                                  f'{float(pscore[k]):.17g}\t{int(q[4])}\t{rank}\n')
                    for c in np.flatnonzero(cn[k]):
                        rows_c.append(f'{qid}\t{c}\t{cn[k, c]}\t{ca[k, c]}\t{ct[k, c]}\n')

    with gzip.open(os.path.join(a.out, 'queries.tsv.gz'), 'wt', encoding='utf-8') as f:
        f.write('query_id\ttriple_id\tside\trelation\tpositive\tpos_score\tmapped\trank_all\n')
        f.writelines(rows_q)
    with gzip.open(os.path.join(a.out, 'cells.tsv.gz'), 'wt', encoding='utf-8') as f:
        f.write('query_id\tcell\tn\tabove\tties\n')
        f.writelines(rows_c)
    ranks = np.array([int(r.rstrip('\n').split('\t')[-1]) for r in rows_q])
    meta = {'checkpoint': a.checkpoint, 'checkpoint_sha256': hashlib.sha256(open(a.checkpoint, 'rb').read()).hexdigest(),
            'mode': a.mode, 'queries': a.queries if not a.all_wn18rr_test else 'all_wn18rr_test', 'filter': a.filter,
            'n_queries': len(rows_q), 'n_entities': N, 'atol': atol, 'rtol': rtol,
            'mrr_all_candidates': float(np.mean(1.0 / ranks)), 'hits10_all_candidates': float(np.mean(ranks <= 10))}
    with open(os.path.join(a.out, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
