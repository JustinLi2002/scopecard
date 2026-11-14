"""Verify that wn18rr_cells.run reproduces score_wn18rr.py byte for byte on LibKGE checkpoints:
an E1 checkpoint (one cell, benchmark filter) and an E2 checkpoint (42 cells, target filter).
The reference outputs must have been produced by score_wn18rr.py on the same checkpoints.

usage (in ~/bs4/env): python check_cells_module.py <E1 ckpt> <E1 ref dir> <E2 ckpt> <E2 ref dir> <target dir> <tmp dir>
"""
import gzip
import os
import sys

import torch
from kge.model import KgeModel
from kge.util.io import load_checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wn18rr_cells as wc  # noqa: E402


def libkge_scorer(model, N, dev):
    all_ent = torch.arange(N, device=dev)
    rel_idx = {n: i for i, n in enumerate(model.dataset.relation_ids())}

    def scorer(side, h, rels, t):
        p = torch.tensor([rel_idx[r] for r in rels], device=dev)
        both = model.score_sp_po(h, p, t, all_ent)
        return both[:, :N] if side == 'tail' else both[:, N:]
    return scorer


def read_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return sorted(f.readlines())


def main(e1_ckpt, e1_ref, e2_ckpt, e2_ref, target_dir, tmp):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    target = wc.Target(target_dir)
    for mode, ckpt, ref in (('E1', e1_ckpt, e1_ref), ('E2', e2_ckpt, e2_ref)):
        model = KgeModel.create_from(load_checkpoint(ckpt, dev))
        model.eval()
        names = list(model.dataset.entity_ids())
        idx = {n: i for i, n in enumerate(names)}
        key = (lambda x: x) if mode == 'E2' else (lambda x: x[1:])
        Q = [(tid, h, r, t, True) for tid, h, r, t in target.queries('bench_test', key)]
        if mode == 'E2':
            filters = wc.build_filters(target.all_triples(key), idx)
            cells = wc.e2_cells(target, names, dev)
        else:
            rel_names = list(model.dataset.relation_ids())
            trip = [(names[s], rel_names[p], names[o]) for sp in ('train', 'valid', 'test') for s, p, o in model.dataset.split(sp).tolist()]
            filters = wc.build_filters(trip, idx)
            cells = None
        out = os.path.join(tmp, mode)
        wc.run(libkge_scorer(model, len(names), dev), Q, names, filters, cells, out, {'check': mode}, batch=256, device=dev)
        for fn in ('queries.tsv.gz', 'cells.tsv.gz'):
            a, b = read_gz(os.path.join(out, fn)), read_gz(os.path.join(ref, fn))
            same = a == b
            print(f'{mode} {fn}: {"IDENTICAL" if same else "DIFFERENT"} ({len(a)} vs {len(b)} lines)')
            if not same:
                diff = [x for x in set(a) ^ set(b)][:4]
                print('  e.g.', diff)


if __name__ == '__main__':
    main(*sys.argv[1:7])
