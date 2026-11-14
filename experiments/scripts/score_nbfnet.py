"""NBFNet scorer for the WN18RR experiment (MEASURED_C_DESIGN.md Sect. 6 and Sect. 13, item 1).

--mode E1      the benchmark graph exactly as NBFNet trained on it (util.build_dataset), the
               mapped bench_test queries, benchmark filter, one cell.
--mode deploy  the same E1-trained model reads the E2 training graph (bench_train + extra,
               every target synset) at inference, scores bench_test against all 112,199
               synsets with the target filter, 42 cells as in E2. NBFNet has no entity
               embeddings, so no retraining is needed: this is the deployment test.
Counting, ties and output files come from wn18rr_cells.run, identical to score_wn18rr.py.
--selfcheck (E1) also ranks with NBFNet's own tasks.compute_ranking on the same predictions.

usage (in env_nbf, from the NBFNet-PyG folder):
  python score_nbfnet.py --config CFG --checkpoint CKPT --mode {E1,deploy} --target DIR
                         --raw WN18RR_RAW_DIR --queries bench_test --out DIR [--selfcheck]
"""
import argparse
import itertools
import os
import sys

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import WordNet18RR

sys.path.insert(0, os.getcwd())                       # NBFNet-PyG root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbfnet import tasks, util  # noqa: E402
import wn18rr_cells as wc  # noqa: E402

EDGE2ID = WordNet18RR.edge2id
NREL = len(EDGE2ID)


def pyg_node_order(raw):
    """Node indices exactly as torch_geometric's WordNet18RR.process assigns them."""
    node2id = {}
    for split in ('train', 'valid', 'test'):
        with open(os.path.join(raw, f'{split}.txt')) as f:
            data = f.read().split()
        for i in itertools.chain(data[::3], data[2::3]):
            if i not in node2id:
                node2id[i] = len(node2id)
    names = [None] * len(node2id)
    for n, i in node2id.items():
        names[i] = n
    return names


def graph_from_triples(triples, idx):
    """Message-passing graph as util.build_dataset builds it: sorted edges plus flipped edges."""
    src = torch.tensor([idx[h] for h, _, _ in triples])
    dst = torch.tensor([idx[t] for _, _, t in triples])
    typ = torch.tensor([EDGE2ID[r] for _, r, _ in triples])
    n = len(idx)
    perm = (n * src + dst).argsort()
    ei = torch.stack([src[perm], dst[perm]])
    et = typ[perm]
    return Data(edge_index=torch.cat([ei, ei.flip(0)], dim=-1), edge_type=torch.cat([et, et + NREL]), num_nodes=n)


def make_scorer(model, data, n, dev):
    allidx = torch.arange(n, device=dev)

    def scorer(side, h, rels, t):
        r = torch.tensor([EDGE2ID[x] for x in rels], device=dev).view(-1, 1).expand(-1, n)
        if side == 'tail':
            batch = torch.stack([h.view(-1, 1).expand(-1, n), allidx.view(1, -1).expand(len(h), -1), r], dim=-1)
        else:
            batch = torch.stack([allidx.view(1, -1).expand(len(t), -1), t.view(-1, 1).expand(-1, n), r], dim=-1)
        return model(data, batch)
    return scorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--mode', choices=['E1', 'deploy'], required=True)
    ap.add_argument('--target', required=True)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--queries', default='bench_test')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--out', required=True)
    ap.add_argument('--selfcheck', action='store_true')
    ap.add_argument('--limit', type=int, default=0, help='score only the first K test triples (testing)')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = util.load_config(a.config, context={'gpus': [0] if dev == 'cuda' else None})
    cfg.model.num_relation = 2 * NREL
    model = util.build_model(cfg)
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'] if 'model' in state else state)
    model = model.to(dev).eval()
    target = wc.Target(a.target)

    if a.mode == 'E1':
        names = pyg_node_order(a.raw)
        idx = {x: i for i, x in enumerate(names)}
        dataset = util.build_dataset(cfg)
        data = dataset[2].to(dev)                      # the training graph NBFNet evaluates on
        raw = {s: [tuple(l.split()) for l in open(os.path.join(a.raw, f'{s}.txt')) if l.strip()] for s in ('train', 'valid', 'test')}
        # the mapping must reproduce PyG's test targets exactly
        mine = {(idx[h], EDGE2ID[r], idx[t]) for h, r, t in raw['test']}
        theirs = set(zip(data.target_edge_index[0].tolist(), data.target_edge_type.tolist(), data.target_edge_index[1].tolist()))
        assert mine == theirs, 'node / relation mapping does not reproduce PyG WordNet18RR'
        key = lambda x: x[1:]
        Q = [(tid, h, r, t, True) for tid, h, r, t in target.queries(a.queries, key)]
        filters = wc.build_filters([x for s in raw.values() for x in s], idx)
        cells = None
    else:
        names = [e['entity_id'] for e in target.E]
        idx = {x: i for i, x in enumerate(names)}
        data = graph_from_triples(target.train_triples(), idx).to(dev)
        Q = [(tid, h, r, t, True) for tid, h, r, t in target.queries(a.queries)]
        filters = wc.build_filters(target.all_triples(), idx)
        cells = wc.e2_cells(target, names, dev)

    if a.limit:
        Q = Q[:a.limit]
    scorer = make_scorer(model, data, len(names), dev)
    meta = wc.run(scorer, Q, names, filters, cells, a.out,
                  {'model': 'nbfnet', 'mode': a.mode, 'queries': a.queries, 'checkpoint': a.checkpoint,
                   'checkpoint_sha256': wc.sha256(a.checkpoint), 'graph_edges': int(data.edge_index.shape[1])},
                  batch=a.batch, device=dev)
    print(meta)

    if a.selfcheck and a.mode == 'E1':
        # NBFNet's own filtered ranking on the same predictions (pessimistic ties) for comparison
        test = torch.tensor([[idx[h], idx[t], EDGE2ID[r]] for _, h, r, t, _ in Q], device=dev)
        filt = dataset
        fdata = Data(edge_index=torch.cat([filt[0].target_edge_index, filt[1].target_edge_index, filt[2].target_edge_index], dim=1),
                     edge_type=torch.cat([filt[0].target_edge_type, filt[1].target_edge_type, filt[2].target_edge_type]),
                     num_nodes=data.num_nodes).to(dev)
        rr = []
        with torch.no_grad():
            for b0 in range(0, len(test), a.batch):
                batch = test[b0:b0 + a.batch]
                t_batch, h_batch = tasks.all_negative(data, batch)
                t_pred, h_pred = model(data, t_batch), model(data, h_batch)
                t_mask, h_mask = tasks.strict_negative_mask(fdata, batch)
                rr += [1.0 / tasks.compute_ranking(t_pred, batch[:, 1], t_mask).float(),
                       1.0 / tasks.compute_ranking(h_pred, batch[:, 0], h_mask).float()]
        print('selfcheck: NBFNet compute_ranking MRR', float(torch.cat(rr).mean()), '| ours', meta['mrr_all_candidates'])


if __name__ == '__main__':
    main()
