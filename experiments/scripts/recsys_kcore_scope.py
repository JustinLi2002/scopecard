"""Check a LightGCN/NGCF recommendation release against its raw source and compute W.

NGCF (Wang et al. 2019, Sect. 4.1.1), reused unchanged by LightGCN: "we use the
10-core setting, i.e., retaining users and items with at least ten
interactions" (Gowalla, Yelp2018, Amazon-Book); Yelp2018 additionally drops
cold-start items from the test set (NGCF footnote 2).

usage: python -E -P recsys_kcore_scope.py <name> <release dir> <raw kind: gowalla|amazon2014|none> <raw file or -> <out dir>
"""
import collections
import gzip
import os
import sys


def read_release(d):
    """Return {split: [(user_org, item_org)]}, using user_list/item_list to map remap ids back."""
    def mapping(fn):
        m = {}
        with open(os.path.join(d, fn), encoding='utf-8') as f:
            next(f)
            for line in f:
                org, rid = line.split()
                m[int(rid)] = org
        return m
    um, im = mapping('user_list.txt'), mapping('item_list.txt')
    out = {}
    for s in ('train', 'test'):
        rows = []
        with open(os.path.join(d, f'{s}.txt'), encoding='utf-8') as f:
            for line in f:
                tok = line.split()
                if len(tok) < 2:
                    continue
                u = um[int(tok[0])]
                rows += [(u, im[int(i)]) for i in tok[1:]]
        out[s] = rows
    return out, um, im


def read_raw(kind, path):
    pairs = set()
    if kind == 'gowalla':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                tok = line.split()
                if len(tok) == 5:
                    pairs.add((tok[0], tok[4]))
    elif kind == 'amazon2014':
        with open(path, encoding='utf-8') as f:
            for line in f:
                tok = line.rstrip('\n').split(',')
                pairs.add((tok[0], tok[1]))
    return pairs


def kcore(pairs, k):
    cur = set(pairs)
    while True:
        du = collections.Counter(u for u, _ in cur)
        di = collections.Counter(i for _, i in cur)
        nxt = {(u, i) for u, i in cur if du[u] >= k and di[i] >= k}
        if nxt == cur:
            return cur
        cur = nxt


def main(name, rel_dir, raw_kind, raw_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out = []

    def say(*a):
        print(*a)
        out.append(' '.join(str(x) for x in a))

    rel, um, im = read_release(rel_dir)
    P = set(rel['train']) | set(rel['test'])
    U = {u for u, _ in P}
    I = {i for _, i in P}
    say(f'[{name}] release: users {len(U)} (list {len(um)}), items {len(I)} (list {len(im)}), interactions {len(P)} '
        f'(train {len(rel["train"])}, test {len(rel["test"])}, train/test overlap {len(set(rel["train"]) & set(rel["test"]))})')
    du = collections.Counter(u for u, _ in P)
    di = collections.Counter(i for _, i in P)
    say(f'  10-core inside the release: users with < 10 {sum(c < 10 for c in du.values())}, items with < 10 {sum(c < 10 for c in di.values())}, '
        f'min user degree {min(du.values())}, min item degree {min(di.values())}')
    tr_items = {i for _, i in rel['train']}
    tr_users = {u for u, _ in rel['train']}
    say(f'  test interactions whose item is absent from train {sum(i not in tr_items for _, i in rel["test"])}, '
        f'whose user is absent from train {sum(u not in tr_users for u, _ in rel["test"])}')

    if raw_kind != 'none':
        raw = read_raw(raw_kind, raw_path)
        RU = {u for u, _ in raw}
        RI = {i for _, i in raw}
        say(f'  raw source: users {len(RU)}, items {len(RI)}, distinct user-item pairs {len(raw)}')
        say(f'  release users in raw {len(U & RU)}, items in raw {len(I & RI)}, interactions in raw {len(P & raw)} of {len(P)}')
        for k in (10,):
            core = kcore(raw, k)
            CU = {u for u, _ in core}
            CI = {i for _, i in core}
            say(f'  raw {k}-core: users {len(CU)}, items {len(CI)}, pairs {len(core)} | equal to release: users {CU == U}, '
                f'items {CI == I}, pairs {core == P} | Jaccard users {len(CU & U) / len(CU | U):.4f}, items {len(CI & I) / len(CI | I):.4f}, '
                f'pairs {len(core & P) / len(core | P):.4f}')
            say(f'  release inside the raw {k}-core: users {len(U & CU)} of {len(U)}, items {len(I & CI)} of {len(I)}, '
                f'pairs {len(P & core)} of {len(P)} | raw {k}-core users not in release {len(CU - U)}, items {len(CI - I)}')
        # W: excluded share of the raw source's unobserved user-item pairs ([P2]'s target convention)
        cand_all = len(RU) * len(RI) - len(raw)
        cand_rel = len(U) * len(I) - len(P)
        say(f'  W (unobserved user-item pairs, universe = raw source users x items): {1 - cand_rel / cand_all:.6f}')
        say(f'  retained users {len(U) / len(RU):.4f}, items {len(I) / len(RI):.4f}, interactions {len(P) / len(raw):.4f}')

    with open(os.path.join(out_dir, f'{name}_scope.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:6])
