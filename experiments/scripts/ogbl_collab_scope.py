"""Profile ogbl-collab's node set, evaluation edges and sampled negatives.

Hu et al. 2020 (OGB): "a subset of the collaboration network between authors
indexed by MAG"; training = collaborations until 2017, validation 2018, test
2019; each positive is ranked against 100,000 randomly sampled negative pairs
(Hits@50). The subset rule is not stated. This script reads the release only.

usage: python -I ogbl_collab_scope.py <collab dir (has raw/)> <npz dir> <out dir>
"""
import gzip
import os
import sys

import numpy as np


def main(collab_dir, npz_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out = []

    def say(*a):
        print(*a)
        out.append(' '.join(str(x) for x in a))

    with gzip.open(os.path.join(collab_dir, 'raw', 'num-node-list.csv.gz'), 'rt') as f:
        n = int(f.read().split()[0])
    sp = {s: np.load(os.path.join(npz_dir, f'{s}.npz')) for s in ('train', 'valid', 'test')}
    tr = sp['train']['edge']
    deg = np.bincount(tr.ravel(), minlength=n)
    say('nodes', n, '| training edges (multi-graph by year)', len(tr), '| nodes with no training edge', int((deg == 0).sum()))
    say('training years', int(sp['train']['year'].min()), '-', int(sp['train']['year'].max()),
        '| valid years', sorted(set(sp['valid']['year'].tolist())), '| test years', sorted(set(sp['test']['year'].tolist())))
    pairs_tr = {(min(a, b), max(a, b)) for a, b in tr.tolist()}
    say('distinct training pairs', len(pairs_tr))
    for s in ('valid', 'test'):
        e = sp[s]['edge']
        unseen = int(((deg[e[:, 0]] == 0) | (deg[e[:, 1]] == 0)).sum())
        rep = sum((min(a, b), max(a, b)) in pairs_tr for a, b in e.tolist())
        say(f'{s}: positives {len(e)} | with an endpoint absent from training {unseen} | repeat of a training pair {rep} ({rep / len(e):.3f})')
        neg = sp[s]['edge_neg']
        nd = deg[neg.ravel()]
        say(f'  negatives {len(neg)} | endpoints with no training edge {int((nd == 0).sum())} of {nd.size} '
            f'| distinct endpoints {len(np.unique(neg))} | negatives that are training pairs '
            f'{sum((min(a, b), max(a, b)) in pairs_tr for a, b in neg.tolist())}')
        # uniform-node sampling check: share of negative endpoints among the lowest-degree half of nodes
        order = np.argsort(deg, kind='stable')
        low = np.zeros(n, bool)
        low[order[: n // 2]] = True
        say(f'  share of negative endpoints in the lower-degree half of nodes {low[neg.ravel()].mean():.3f} (uniform node sampling -> 0.5)')
    q = np.quantile(deg, [0, .1, .25, .5, .75, .9])
    say('training degree quantiles (0,10,25,50,75,90%)', q.tolist(), '| nodes with degree 1:', int((deg == 1).sum()))
    with open(os.path.join(out_dir, 'ogbl_collab_scope.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:4])
