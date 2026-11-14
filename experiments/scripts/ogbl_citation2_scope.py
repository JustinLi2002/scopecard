"""Profile ogbl-citation2's node set, evaluation sources and sampled negatives.

Hu et al. 2020 (OGB, Sec. 5.4): "a subset of papers extracted from MAG"; source
papers are those published in 2019; for each, two references are dropped (one
validation, one test) and ranked against 1,000 negatives "randomly-sampled from
all the previous papers that are not referenced by the source paper" (MRR).
The subset rule is not stated. This script reads the release only.

usage: python -E -P ogbl_citation2_scope.py <citation-v2 dir (has raw/)> <npz dir> <out dir>
"""
import gzip
import os
import sys

import numpy as np


def read_csv_int(path, cols):
    with gzip.open(path, 'rt') as f:
        a = np.loadtxt(f, delimiter=',', dtype=np.int64, ndmin=2)
    return a if cols > 1 else a[:, 0]


def main(cdir, npz_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out = []

    def say(*a):
        print(*a)
        out.append(' '.join(str(x) for x in a))

    year = read_csv_int(os.path.join(cdir, 'raw', 'node_year.csv.gz'), 1)
    n = len(year)
    sp = {s: np.load(os.path.join(npz_dir, f'{s}.npz')) for s in ('train', 'valid', 'test')}
    src, dst = sp['train']['source_node'], sp['train']['target_node']
    outdeg = np.bincount(src, minlength=n)
    indeg = np.bincount(dst, minlength=n)
    deg = outdeg + indeg
    say('nodes', n, '| training edges', len(src), '| nodes with no training edge', int((deg == 0).sum()))
    say('years', int(year.min()), '-', int(year.max()), '| 2019 papers', int((year == 2019).sum()))
    for s in ('valid', 'test'):
        S, T, N = sp[s]['source_node'], sp[s]['target_node'], sp[s]['target_node_neg']
        say(f'{s}: sources {len(S)} (distinct {len(np.unique(S))}) | source years {sorted(set(year[S].tolist()))} '
            f'| negatives per source {N.shape[1]}')
        say(f'  source training out-degree: min {int(outdeg[S].min())}, median {int(np.median(outdeg[S]))}')
        say(f'  target papers with no training edge: {int((deg[T] == 0).sum())}')
        later = (year[N] > year[S][:, None]).sum()
        same = (year[N] == year[S][:, None]).sum()
        say(f'  negatives published after the source: {int(later)}, same year: {int(same)}, of {N.size}')
        say(f'  negatives with no training edge: {int((deg[N.ravel()] == 0).sum())}')
        order = np.argsort(deg, kind='stable')
        low = np.zeros(n, bool)
        low[order[: n // 2]] = True
        say(f'  share of negatives in the lower-degree half of all nodes {low[N.ravel()].mean():.3f}')
    y19 = np.flatnonzero(year == 2019)
    srcs = np.unique(np.concatenate([sp['valid']['source_node'], sp['test']['source_node']]))
    not_src = np.setdiff1d(y19, srcs)
    # a 2019 paper can be a source only if it keeps references after two are dropped
    say('2019 papers that are not sources:', len(not_src), 'of', len(y19), f'({len(not_src) / len(y19):.3f})',
        '| their training out-degree: 0:', int((outdeg[not_src] == 0).sum()), ', 1:', int((outdeg[not_src] == 1).sum()),
        ', >= 2:', int((outdeg[not_src] >= 2).sum()))
    say('sources: total references in the release (train out-degree + 2): min', int(outdeg[srcs].min()) + 2)
    # a 2019 non-source keeps all its references in train, so its out-degree is its total
    tot_non = outdeg[not_src]
    say('2019 non-sources by total references (0..6):', {k: int((tot_non == k).sum()) for k in range(7)},
        '| non-sources with >= 6 references:', int((tot_non >= 6).sum()),
        '| 2019 papers with >= 6 references that are sources:', len(srcs), 'of', len(srcs) + int((tot_non >= 6).sum()))
    say('=> source rule found in the release: a 2019 paper is a source exactly when it has >= 6 in-subset references; '
        f'excluded share of 2019 papers {len(not_src) / len(y19):.4f}')
    with open(os.path.join(out_dir, 'ogbl_citation2_scope.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:4])
