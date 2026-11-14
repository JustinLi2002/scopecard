"""Export the E2 (declared target) split as LibKGE raw files.

train.txt = bench_train + extra, valid.txt = bench_valid, test.txt = bench_test,
plus u_valid.txt / u_test.txt for the query-side estimand (not read by LibKGE).
Rows keep the target's (POS, offset) synset ids, so LibKGE's entity index can be
checked against entities.tsv after preprocessing.

usage: python -I export_libkge_e2.py <target dir> <out dir>
"""
import collections
import os
import sys


def main(tdir, out):
    os.makedirs(out, exist_ok=True)
    files = collections.defaultdict(list)
    dest = {'bench_train': 'train', 'extra': 'train', 'bench_valid': 'valid', 'bench_test': 'test',
            'u_valid': 'u_valid', 'u_test': 'u_test'}
    with open(os.path.join(tdir, 'triples.tsv'), encoding='utf-8') as f:
        next(f)
        for line in f:
            _, h, r, t, role = line.rstrip('\n').split('\t')
            files[dest[role]].append(f'{h}\t{r}\t{t}\n')
    for name, rows in files.items():
        with open(os.path.join(out, f'{name}.txt'), 'w', encoding='utf-8', newline='\n') as f:
            f.writelines(rows)
        print(name, len(rows))


if __name__ == '__main__':
    main(*sys.argv[1:3])
