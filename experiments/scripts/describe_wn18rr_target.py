"""Describe composition differences between O and U in the WN18RR target:
POS mix, E2 training degree, and which candidates are POS-compatible with each
relation slot (the type-restricted sensitivity analysis). Descriptive only.

usage: python -I describe_wn18rr_target.py <target dir>
"""
import collections
import os
import sys


def read_tsv(path):
    with open(path, encoding='utf-8') as f:
        head = f.readline().rstrip('\n').split('\t')
        return [dict(zip(head, line.rstrip('\n').split('\t'))) for line in f]


def main(tdir):
    E = read_tsv(os.path.join(tdir, 'entities.tsv'))
    T = read_tsv(os.path.join(tdir, 'triples.tsv'))
    out = []

    def say(*a):
        print(*a)
        out.append(' '.join(str(x) for x in a))

    reg = {e['entity_id']: e['region'] for e in E}
    pos = {e['entity_id']: e['pos'] for e in E}
    for r in ('O', 'U'):
        c = collections.Counter(e['pos'] for e in E if e['region'] == r)
        n = sum(c.values())
        say(f'{r}: {n} synsets, POS shares', {p: round(c[p] / n, 4) for p in 'nvar'})
        d = sorted(int(e['train_degree']) for e in E if e['region'] == r)
        say(f'   E2 training degree: median {d[len(d) // 2]}, 90th pct {d[int(.9 * len(d))]}, share with degree 1: {sum(x == 1 for x in d) / n:.4f}')
    train = [t for t in T if t['role'] in ('bench_train', 'extra')]
    slot = collections.defaultdict(collections.Counter)
    for t in train:
        slot[(t['relation'], 'head')][pos[t['head']]] += 1
        slot[(t['relation'], 'tail')][pos[t['tail']]] += 1
    nO = collections.Counter(e['pos'] for e in E if e['region'] == 'O')
    nU = collections.Counter(e['pos'] for e in E if e['region'] == 'U')
    bt = [t for t in T if t['role'] == 'bench_test']
    qw = collections.Counter()
    for t in bt:
        for side in ('head', 'tail'):
            allowed = set(slot[(t['relation'], side)])
            o = sum(nO[p] for p in allowed)
            u = sum(nU[p] for p in allowed)
            qw[side] += u / (o + u)
    say('mean per-query U share of candidates, all POS: ', round(sum(nU.values()) / (sum(nU.values()) + sum(nO.values())), 4))
    say('mean per-query U share, restricted to POS seen in the relation slot in E2 training (before filtering):',
        {s: round(qw[s] / len(bt), 4) for s in ('head', 'tail')})
    for (r, s), c in sorted(slot.items()):
        say(f'   {r} {s}: POS seen {dict(c)}')
    with open(os.path.join(tdir, 'composition.txt'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
