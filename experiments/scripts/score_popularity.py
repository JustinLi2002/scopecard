"""T0 of the ladder (MEASURED_C_DESIGN.md Sect. 6): relation-slot popularity.
score(candidate e | relation r, tail side) = number of training triples (., r, e);
head side: number of training triples (e, r, .). It ignores the query entity, so it is
the analogue of [P3]'s drug-only baseline. Deterministic: one run, labelled seed 0.

E1: WN18RR train as training data, benchmark filter, bare-offset entities.
E2: E2 training triples (bench_train + extra), target filter, synset entities, 42 cells;
    scored on bench_test and u_test.

usage (in ~/bs4/env): python score_popularity.py <target dir> <WN18RR dir> <runs dir>
"""
import collections
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wn18rr_cells as wc  # noqa: E402


def read_raw(path):
    with open(path, encoding='utf-8') as f:
        return [tuple(line.rstrip('\n').split('\t')) for line in f if line.strip()]


def popularity_scorer(train, names, dev):
    idx = {n: i for i, n in enumerate(names)}
    N = len(names)
    cnt = collections.defaultdict(lambda: torch.zeros(N, dtype=torch.float64))
    for h, r, t in train:
        cnt[(r, 'tail')][idx[t]] += 1
        cnt[(r, 'head')][idx[h]] += 1
    vec = {k: v.to(dev) for k, v in cnt.items()}
    zero = torch.zeros(N, dtype=torch.float64, device=dev)

    def scorer(side, h, rels, t):
        return torch.stack([vec.get((r, side), zero) for r in rels])
    return scorer


def main(target_dir, rr_dir, runs):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    target = wc.Target(target_dir)
    # E1
    rr = {s: read_raw(os.path.join(rr_dir, f'{s}.txt')) for s in ('train', 'valid', 'test')}
    names1 = sorted({e for s in rr.values() for h, _, t in s for e in (h, t)})
    idx1 = {n: i for i, n in enumerate(names1)}
    key = lambda x: x[1:]
    Q1 = [(tid, h, r, t, True) for tid, h, r, t in target.queries('bench_test', key)]
    m = wc.run(popularity_scorer(rr['train'], names1, dev), Q1, names1,
               wc.build_filters([x for s in rr.values() for x in s], idx1), None,
               os.path.join(runs, 'E1', 'popularity_s0', 'score_bench_test'),
               {'model': 'popularity', 'mode': 'E1', 'filter': 'splits'}, batch=256, device=dev)
    print('E1', m['n_queries'], round(m['mrr_all_candidates'], 6))
    # E2
    names2 = [e['entity_id'] for e in target.E]
    idx2 = {n: i for i, n in enumerate(names2)}
    scorer2 = popularity_scorer(target.train_triples(), names2, dev)
    filters2 = wc.build_filters(target.all_triples(), idx2)
    cells2 = wc.e2_cells(target, names2, dev)
    for role in ('bench_test', 'u_test'):
        Q2 = [(tid, h, r, t, True) for tid, h, r, t in target.queries(role)]
        m = wc.run(scorer2, Q2, names2, filters2, cells2, os.path.join(runs, 'E2', 'popularity_s0', 'score_' + role),
                   {'model': 'popularity', 'mode': 'E2', 'queries': role, 'filter': 'target'}, batch=128, device=dev)
        print('E2', role, m['n_queries'], round(m['mrr_all_candidates'], 6))


if __name__ == '__main__':
    main(*sys.argv[1:4])
