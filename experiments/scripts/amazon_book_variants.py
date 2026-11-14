"""Search for the unstated filter behind NGCF's Amazon-Book: 10-core of the 2014
Books ratings under rating thresholds and time windows, compared with the release.

usage: python -E -P amazon_book_variants.py <ratings_Books.csv> <release dir> <out file>
"""
import os
import sys
import time

import numpy as np
import pandas as pd


def release_sets(d):
    def mapping(fn):
        m = pd.read_csv(os.path.join(d, fn), sep=' ', dtype=str)
        return dict(zip(m['remap_id'].astype(int), m['org_id']))
    um, im = mapping('user_list.txt'), mapping('item_list.txt')
    pairs = set()
    for s in ('train', 'test'):
        with open(os.path.join(d, f'{s}.txt'), encoding='utf-8') as f:
            for line in f:
                tok = line.split()
                if len(tok) > 1:
                    pairs.update((um[int(tok[0])], im[int(i)]) for i in tok[1:])
    return pairs


def kcore(u, i, k):
    keep = np.ones(len(u), bool)
    while True:
        cu = np.bincount(u[keep], minlength=u.max() + 1)
        ci = np.bincount(i[keep], minlength=i.max() + 1)
        nk = keep & (cu[u] >= k) & (ci[i] >= k)
        if nk.sum() == keep.sum():
            return keep
        keep = nk


def main(path, rel_dir, out_path):
    t0 = time.time()
    df = pd.read_csv(path, header=None, names=['user', 'item', 'rating', 'ts'],
                     dtype={'user': str, 'item': str, 'rating': float, 'ts': np.int64})
    df['year'] = pd.to_datetime(df['ts'], unit='s').dt.year
    P = release_sets(rel_dir)
    RU = {u for u, _ in P}
    RI = {i for _, i in P}
    rel_ratings = df[[(a, b) in P for a, b in zip(df['user'], df['item'])]]
    lines = [f'raw rows {len(df)}; release pairs {len(P)}; release pairs found {len(rel_ratings)}',
             f'rating distribution inside the release: {rel_ratings["rating"].value_counts().sort_index().to_dict()}',
             f'year range inside the release: {int(rel_ratings["year"].min())}-{int(rel_ratings["year"].max())}; '
             f'raw: {int(df["year"].min())}-{int(df["year"].max())}']
    uc, ic = df['user'].astype('category'), df['item'].astype('category')
    u, i = uc.cat.codes.to_numpy(), ic.cat.codes.to_numpy()
    variants = {'all ratings': np.ones(len(df), bool)}
    for r in (3, 4, 5):
        variants[f'rating >= {r}'] = df['rating'].to_numpy() >= r
    for y in (2010, 2012, 2013):
        variants[f'year >= {y}'] = df['year'].to_numpy() >= y
    for name, m in variants.items():
        keep = m & kcore(np.where(m, u, 0), np.where(m, i, 0), 10) if False else None
        uu, ii = u[m], i[m]
        k = kcore(uu, ii, 10)
        cu = set(uc.cat.categories[uu[k]])
        ci = set(ic.cat.categories[ii[k]])
        np_pairs = int(k.sum())
        ju = len(cu & RU) / len(cu | RU)
        ji = len(ci & RI) / len(ci | RI)
        lines.append(f'{name}: 10-core users {len(cu)}, items {len(ci)}, pairs {np_pairs} | Jaccard users {ju:.4f}, items {ji:.4f}')
        print(lines[-1], f'({time.time() - t0:.0f}s)')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines[:3]))


def reviews_variant(reviews_gz, rel_dir, out_path):
    """10-core of the reviews that carry text. The 5-core review file suffices: the
    10-core of the full review set lies inside its 5-core, so both give the same 10-core."""
    import gzip
    import re
    pat = re.compile(r'"reviewerID": "([^"]+)".*?"asin": "([^"]+)"')
    users, items = [], []
    with gzip.open(reviews_gz, 'rt', encoding='utf-8') as f:
        for line in f:
            m = pat.search(line)
            users.append(m.group(1))
            items.append(m.group(2))
    df = pd.DataFrame({'user': users, 'item': items}).drop_duplicates()
    P = release_sets(rel_dir)
    RU = {u for u, _ in P}
    RI = {i for _, i in P}
    uc, ic = df['user'].astype('category'), df['item'].astype('category')
    u, i = uc.cat.codes.to_numpy(), ic.cat.codes.to_numpy()
    k = kcore(u, i, 10)
    cu, ci = set(uc.cat.categories[u[k]]), set(ic.cat.categories[i[k]])
    cp = set(zip(uc.cat.categories[u[k]], ic.cat.categories[i[k]]))
    line = (f'reviews with text (5-core file, {len(df)} distinct pairs): 10-core users {len(cu)}, items {len(ci)}, pairs {len(cp)} | '
            f'equal to release: users {cu == RU}, items {ci == RI}, pairs {cp == P} | Jaccard users {len(cu & RU) / len(cu | RU):.4f}, '
            f'items {len(ci & RI) / len(ci | RI):.4f}, pairs {len(cp & P) / len(cp | P):.4f}')
    print(line)
    with open(out_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


if __name__ == '__main__':
    if sys.argv[1] == '--reviews':
        reviews_variant(*sys.argv[2:5])
    else:
        main(*sys.argv[1:4])
