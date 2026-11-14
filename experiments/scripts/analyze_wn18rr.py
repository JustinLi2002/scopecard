"""Prespecified analysis of the WN18RR measured-c experiment (MEASURED_C_DESIGN.md
Sections 4, 5, 7, 8 and 13). Reads the per-query scorer outputs and writes keyed records.

Per query and candidate set S (counts n, above, ties from cells.tsv.gz):
  AUC_q(S) = 1 - (above + ties / 2) / n          (ties count one half)
  rank_q(S) = above + (ties + 1) // 2 + 1        (LibKGE rounded_mean_rank)
Regions (cell = region * 14 + bin * 2 + pos_ok): 0 = O trained, 1 = O untrained, 2 = U.
Estimands (E2 bench_test):
  primary  O = regions 0, 1; U = region 2; C = O + U; W_q = n_U / n_C
  S1       only POS-compatible candidates (pos_ok = 1)
  S2       AUC_q(U) reweighted over training-degree bins to O's bin distribution
  S4       O = region 0 only
Pairs (A, B): delta_O = mean(AUC^A_q(O) - AUC^B_q(O)), delta_U likewise,
  delta_target = mean((1 - W_q) d_O,q + W_q d_U,q) exactly, c = delta_O - delta_U,
  c* = |delta_O| / mean(W_q), approx_err = delta_target - (delta_O - mean(W) c).
Uncertainty: 2,000 bootstrap replicates over test triples (both queries of a triple
move together); sensitivity clustering by the triple's head synset. RNG streams are
keyed by record, so reruns reproduce every interval.

Deployment test (design Sect. 13 item 1), a separate estimand: NBFNet trained on WN18RR
(E1) reads the E2 training graph at inference and scores bench_test over the same 42
cells and target filter as E2. Design 'deploy' gets the per-model quantities of every
E2 estimand; with one model there are no pairs (no T1 comparison exists). The
graph-side effect (deploy-O minus E1-O, same checkpoint) is reported like the
training-side effect; it changes the inference graph, the filter and the O keying.

S3 (Sect. 5, benchmark identity): the same E2 design on the target keyed by bare offset
(results/wn18rr_target_s3, built by build_wn18rr_target_s3.py), trained and scored separately. Its runs
sit in E2_S3/; its records carry design E2 and estimand S3, with the primary region sets and pairs. Its
triple ids differ from the primary target's (they hash bare offsets), so S3 is compared with the primary
estimand at the level of model and pair statistics, not query by query. Pass --target-s3 to analyse it.
The S3 scorer's POS flag is not meaningful (ids carry no POS), so S3 is not crossed with S1.

Runs dir layout: <design>/<model>_s<seed>/score_<queries>/ for design E1, E2, E2_S3, deploy.
NBFNet goes to E1/nbfnet_s<seed>/ (score_bench_test = the job's score_E1_bench_test,
train.log = run.py's log) and deploy/nbfnet_s<seed>/score_bench_test.

usage: python -E -P analyze_wn18rr.py <runs dir> <target dir> <out dir> [--reps 2000]
"""
import argparse
import collections
import gzip
import hashlib
import itertools
import json
import os
import re
import sys

import numpy as np

N_CELLS = 42
MODELS = ['popularity', 'transe', 'conve', 'distmult', 'rescal', 'complex', 'nbfnet']   # ladder order
PUBLISHED = {'transe': 0.228, 'conve': 0.442, 'distmult': 0.452, 'rescal': 0.467, 'complex': 0.475}


def read_tsv(path):
    with open(path, encoding='utf-8') as f:
        head = f.readline().rstrip('\n').split('\t')
        return [dict(zip(head, line.rstrip('\n').split('\t'))) for line in f]


def load_score(d):
    """Return (query_ids, triple_ids, sides, rank_all, counts[q, cell, (n, above, ties)])."""
    with gzip.open(os.path.join(d, 'queries.tsv.gz'), 'rt', encoding='utf-8') as f:
        next(f)
        rows = [line.rstrip('\n').split('\t') for line in f]
    rows.sort(key=lambda r: r[0])
    qid = [r[0] for r in rows]
    idx = {q: i for i, q in enumerate(qid)}
    counts = np.zeros((len(qid), N_CELLS, 3), dtype=np.int64)
    with gzip.open(os.path.join(d, 'cells.tsv.gz'), 'rt', encoding='utf-8') as f:
        next(f)
        for line in f:
            q, c, n, a, t = line.rstrip('\n').split('\t')
            counts[idx[q], int(c)] = (int(n), int(a), int(t))
    return (np.array(qid), np.array([r[1] for r in rows]), np.array([r[2] for r in rows]),
            np.array([int(r[7]) for r in rows]), counts)


def cell_mask(region=None, pos_ok=None, bins=None):
    m = np.zeros(N_CELLS, bool)
    for c in range(N_CELLS):
        r, b, p = c // 14, (c % 14) // 2, c % 2
        if (region is None or r in region) and (pos_ok is None or p == pos_ok) and (bins is None or b in bins):
            m[c] = True
    return m


def agg(counts, mask):
    s = counts[:, mask, :].sum(1)
    return s[:, 0], s[:, 1], s[:, 2]


def auc(n, a, t):
    with np.errstate(invalid='ignore', divide='ignore'):
        return 1.0 - (a + 0.5 * t) / n


def rank(a, t):
    return a + (t + 1) // 2 + 1


def region_sets(counts, estimand):
    """Return (O, U) as (n, above, ties) tuples for an estimand."""
    if estimand == 'S1':
        return agg(counts, cell_mask({0, 1}, pos_ok=1)), agg(counts, cell_mask({2}, pos_ok=1))
    if estimand == 'S4':
        return agg(counts, cell_mask({0})), agg(counts, cell_mask({2}))
    return agg(counts, cell_mask({0, 1})), agg(counts, cell_mask({2}))


def s2_auc_u(counts):
    """AUC_q(U) reweighted to O's training-degree bin distribution (bins with U candidates)."""
    out = np.zeros(len(counts))
    wsum = np.zeros(len(counts))
    nO_b = {b: agg(counts, cell_mask({0, 1}, bins={b}))[0] for b in range(7)}
    for b in range(1, 7):
        n, a, t = agg(counts, cell_mask({2}, bins={b}))
        ok = n > 0
        w = np.where(ok, nO_b[b], 0).astype(float)
        out += np.where(ok, w * auc(np.maximum(n, 1), a, t), 0.0)
        wsum += w
    with np.errstate(invalid='ignore', divide='ignore'):
        return out / wsum


def per_query(counts, estimand):
    """Per-query values for one run under an estimand."""
    (nO, aO, tO), (nU, aU, tU) = region_sets(counts, estimand)
    nC, aC, tC = nO + nU, aO + aU, tO + tU
    v = {'W': nU / nC, 'auc_O': auc(nO, aO, tO), 'auc_C': auc(nC, aC, tC)}
    v['auc_U'] = s2_auc_u(counts) if estimand == 'S2' else auc(nU, aU, tU)
    if estimand == 'S2':   # target score with the reweighted U term, same W_q
        v['auc_C'] = (1 - v['W']) * v['auc_O'] + v['W'] * v['auc_U']
    rO, rC = rank(aO, tO), rank(aC, tC)
    v.update({'rr_O': 1.0 / rO, 'rr_C': 1.0 / rC, 'h1_O': (rO <= 1) * 1.0, 'h1_C': (rC <= 1) * 1.0,
              'h10_O': (rO <= 10) * 1.0, 'h10_C': (rC <= 10) * 1.0,
              'intrusion': (aU >= 1) * 1.0, 'top10_drop': ((rO <= 10) & (rC > 10)) * 1.0,
              'n_O': nO.astype(float), 'n_U': nU.astype(float)})
    return v


class Boot:
    """Cluster bootstrap of means of per-query values; keyed RNG per record."""

    def __init__(self, cluster_of_query, reps):
        self.cl, inv = np.unique(cluster_of_query, return_inverse=True)
        self.inv = inv
        self.reps = reps
        self.nq_c = np.bincount(inv, minlength=len(self.cl)).astype(float)

    def draws(self, key):
        seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        k = len(self.cl)
        return rng.multinomial(k, np.full(k, 1.0 / k), size=self.reps).astype(float)

    def means(self, K, values):
        """values: dict name -> per-query array; returns dict name -> (point, replicate means)."""
        out = {}
        denom = K @ self.nq_c
        for name, v in values.items():
            ok = ~np.isnan(v)
            s = np.bincount(self.inv[ok], weights=v[ok], minlength=len(self.cl))
            nc = np.bincount(self.inv[ok], minlength=len(self.cl)).astype(float)
            point = v[ok].mean()
            reps = (K @ s) / (K @ nc) if not ok.all() else (K @ s) / denom
            out[name] = (point, reps)
        return out


def ci(reps):
    return np.percentile(reps, 2.5), np.percentile(reps, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs')
    ap.add_argument('target')
    ap.add_argument('out')
    ap.add_argument('--reps', type=int, default=2000)
    ap.add_argument('--target-s3', default=None, help='S3 target dir (needed when E2_S3 runs exist)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    records = []
    log = []

    def say(*x):
        print(*x)
        log.append(' '.join(str(y) for y in x))

    def rec(design, estimand, subject, seed, quantity, value, lo=np.nan, hi=np.nan, n=0, cluster='triple'):
        records.append((design, estimand, subject, str(seed), quantity, value, lo, hi, n, cluster))

    # ---------- load
    runs = collections.defaultdict(dict)        # (design, queries) -> {(model, seed): data}
    for design in ('E1', 'E2', 'E2_S3', 'deploy'):
        dd = os.path.join(a.runs, design)
        if not os.path.isdir(dd):
            continue
        for name in sorted(os.listdir(dd)):
            m = re.fullmatch(r'(\w+)_s(\d+)', name)
            if not m:
                continue
            for q in ('bench_test', 'u_test'):
                sd = os.path.join(dd, name, 'score_' + q)
                if os.path.isdir(sd):
                    runs[(design, q)][(m.group(1), int(m.group(2)))] = load_score(sd)
    for key, rr in runs.items():
        say(f'loaded {key}: {len(rr)} runs, models {sorted({m for m, _ in rr})}')

    T = read_tsv(os.path.join(a.target, 'triples.tsv'))
    E = {e['entity_id']: e['region'] for e in read_tsv(os.path.join(a.target, 'entities.tsv'))}
    head_of = {t['triple_id']: t['head'] for t in T}
    tail_of = {t['triple_id']: t['tail'] for t in T}
    if runs.get(('E2_S3', 'bench_test')):
        assert a.target_s3, 'E2_S3 runs exist: pass --target-s3'
        T3 = read_tsv(os.path.join(a.target_s3, 'triples.tsv'))
        head_of.update({t['triple_id']: t['head'] for t in T3})   # bare-offset ids: no collision with primary ids
        tail_of.update({t['triple_id']: t['tail'] for t in T3})

    # ---------- self-checks
    for (design, q), rr in runs.items():
        ref = None
        for (m, s), (qid, tid, side, rank_all, counts) in rr.items():
            nC, aC, tC = agg(counts, np.ones(N_CELLS, bool))
            assert (rank(aC, tC) == rank_all).all(), f'rank mismatch {design} {m} {s}'
            if ref is None:
                ref = (qid, counts[:, :, 0])
            else:
                assert (ref[0] == qid).all(), 'query sets differ'
                assert (ref[1] == counts[:, :, 0]).all(), f'candidate counts differ across runs ({design} {m} {s})'
            if design in ('E2', 'E2_S3', 'deploy'):
                v = per_query(counts, 'primary')
                mix = (1 - v['W']) * v['auc_O'] + v['W'] * v['auc_U']
                assert np.allclose(mix, v['auc_C'], atol=1e-12), 'AUC decomposition failed'
        say(f'self-checks passed for {design} {q}: ranks reproduce rank_all, identical candidate counts across runs'
            + (', AUC_C = (1-W)AUC_O + W AUC_U per query' if design in ('E2', 'E2_S3', 'deploy') else ''))

    # deployment test: same queries and candidate cells as E2 (same target filter), and the
    # checkpoint scored is the E1 checkpoint of the same seed
    rrd = runs.get(('deploy', 'bench_test'), {})
    if rrd:
        e2 = runs.get(('E2', 'bench_test'), {})
        if e2:
            ref = next(iter(e2.values()))
            for (m, s), (qid, _, _, _, counts) in rrd.items():
                assert (qid == ref[0]).all(), f'deploy {m} s{s}: query set differs from E2'
                assert (counts[:, :, 0] == ref[4][:, :, 0]).all(), f'deploy {m} s{s}: candidate counts differ from E2'
        for (m, s) in rrd:
            meta = {}
            for design, sub in (('E1', 'score_bench_test'), ('deploy', 'score_bench_test')):
                p = os.path.join(a.runs, design, f'{m}_s{s}', sub, 'meta.json')
                meta[design] = json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None
            assert meta['deploy'] and meta['deploy'].get('mode') == 'deploy', f'deploy {m} s{s}: meta.json mode'
            if meta['E1']:
                assert meta['E1']['checkpoint_sha256'] == meta['deploy']['checkpoint_sha256'], \
                    f'deploy {m} s{s}: checkpoint differs from E1'
        say('deploy checks passed: ' + ('query sets and candidate counts equal E2; ' if e2 else 'no E2 runs to compare; ')
            + 'checkpoint hashes equal E1 where E1 scores exist')

    # external check: our O-only MRR against LibKGE's own test MRR for the same checkpoint
    worst = {}
    for (design, q), rr in runs.items():
        if q != 'bench_test' or design == 'deploy':
            continue
        logname = 'test_O.log' if design in ('E2', 'E2_S3') else 'test.log'
        for (m, s), (_, _, _, _, counts) in rr.items():
            p = os.path.join(a.runs, design, f'{m}_s{s}', logname)
            if not os.path.exists(p):
                continue
            hits = re.findall(r'  mean_reciprocal_rank_filtered: ([0-9.]+)', open(p, encoding='utf-8', errors='replace').read())
            if not hits:
                continue
            n, a_, t = (agg(counts, cell_mask({0, 1})) if design in ('E2', 'E2_S3') else agg(counts, np.ones(N_CELLS, bool)))
            ours = float(np.mean(1.0 / rank(a_, t)))
            d = abs(ours - float(hits[-1]))
            worst[design] = max(worst.get(design, 0), d)
            tol = 1e-5 if design in ('E2', 'E2_S3') else 2e-3    # E1 LibKGE value includes the one unmapped test triple
            assert d < tol, f'{design} {m} s{s}: ours {ours} vs LibKGE {hits[-1]}'
    say('LibKGE agreement, max |our O-only MRR - LibKGE test MRR|:', {k: f'{v:.2e}' for k, v in worst.items()})

    # external check for NBFNet E1: our MRR against run.py's own test MRR (the last 'mrr:' after
    # 'Evaluate on test'). run.py includes the one unmapped test triple and breaks ties pessimistically.
    worst_nbf = None
    for (m, s), (_, _, _, _, counts) in runs.get(('E1', 'bench_test'), {}).items():
        p = os.path.join(a.runs, 'E1', f'{m}_s{s}', 'train.log')
        if m != 'nbfnet' or not os.path.exists(p):
            continue
        txt = open(p, encoding='utf-8', errors='replace').read()
        hits = re.findall(r'\bmrr: ([0-9.eE+-]+)', txt.rsplit('Evaluate on test', 1)[-1]) if 'Evaluate on test' in txt else []
        assert hits, f'E1 nbfnet s{s}: no test MRR in train.log'
        n, a_, t = agg(counts, np.ones(N_CELLS, bool))
        d = abs(float(np.mean(1.0 / rank(a_, t))) - float(hits[0]))
        worst_nbf = max(worst_nbf or 0.0, d)
        assert d < 2e-3, f'E1 nbfnet s{s}: ours vs run.py {hits[0]}'
    if worst_nbf is not None:
        say(f'NBFNet agreement, max |our E1 MRR - run.py test MRR|: {worst_nbf:.2e}')

    # ---------- E2, deploy and S3 bench_test: per-model magnitudes and pairs, every estimand
    def triple_clusters(tid, mode):
        return np.array([head_of[t] for t in tid]) if mode == 'head' else tid

    ALL = ('primary', 'S1', 'S2', 'S4', 'posthoc_noleak')
    for design, q, label, estimands in (('E2', 'bench_test', 'E2', ALL), ('deploy', 'bench_test', 'deploy', ALL),
                                        ('E2_S3', 'bench_test', 'E2', ('S3',))):
        rr = runs.get((design, q), {})
        if not rr:
            continue
        any_run = next(iter(rr.values()))
        qid, tid = any_run[0], any_run[1]
        models = [m for m in MODELS if any(k[0] == m for k in rr)]
        seeds = {m: sorted(s for mm, s in rr if mm == m) for m in models}
        if design == 'E2_S3':
            prim = runs.get(('E2', q), {})
            assert not prim or len(next(iter(prim.values()))[0]) == len(qid), 'S3 and primary query counts differ'
        noleak = None
        if 'posthoc_noleak' in estimands:
            # POST HOC (design Sect. 13 item 3): the primary estimand on queries whose test triple's
            # reverse is not an E2 training triple (no one-edge answer through a symmetric relation)
            rel_of0 = {t['triple_id']: t['relation'] for t in T}
            train0 = {(t['head'], t['relation'], t['tail']) for t in T if t['role'] in ('bench_train', 'extra')}
            noleak = np.array([(tail_of[x], rel_of0[x], head_of[x]) not in train0 for x in tid])
            say(f'{design} {q}: non-leaking queries {int(noleak.sum())} of {len(noleak)}')
        for estimand in estimands:
            base = 'primary' if estimand in ('posthoc_noleak', 'S3') else estimand
            sel = noleak if estimand == 'posthoc_noleak' else np.ones(len(tid), bool)
            vals = {(m, s): {k: v[sel] for k, v in per_query(rr[(m, s)][4], base).items()} for m in models for s in seeds[m]}
            # seed-mean per model: average per-query values over seeds
            for m in models:
                vals[(m, 'mean')] = {k: np.mean([vals[(m, s)][k] for s in seeds[m]], axis=0) for k in vals[(m, seeds[m][0])]}
            for mode in (['triple', 'head'] if estimand in ('primary', 'S3') else ['triple']):
                boot = Boot(triple_clusters(tid[sel], mode), a.reps)
                for m in models:
                    for s in seeds[m] + ['mean']:
                        v = vals[(m, s)]
                        K = boot.draws(f'{design}|{q}|{estimand}|{m}|{s}|{mode}')
                        qs = {k: v[k] for k in ('auc_O', 'auc_U', 'auc_C', 'rr_O', 'rr_C', 'h1_O', 'h1_C', 'h10_O', 'h10_C',
                                                'intrusion', 'top10_drop', 'W')}
                        qs['auc_U_minus_O'] = v['auc_U'] - v['auc_O']
                        qs['mrr_C_minus_O'] = v['rr_C'] - v['rr_O']
                        res = boot.means(K, qs)
                        for name, (pt, reps) in res.items():
                            lo, hi = ci(reps)
                            rec(label, estimand, m, s, name, pt, lo, hi, int(sel.sum()), mode)
                for A, B in itertools.combinations(models, 2):
                    common = sorted(set(seeds[A]) & set(seeds[B]))
                    for s in common + ['mean']:
                        va, vb = vals[(A, s)], vals[(B, s)]
                        W = va['W']
                        dO = va['auc_O'] - vb['auc_O']
                        dU = va['auc_U'] - vb['auc_U']
                        qs = {'delta_O': dO, 'delta_U': dU, 'delta_target': (1 - W) * dO + W * dU, 'W': W,
                              'delta_U_minus_O': dU - dO, 'delta_mrr_O': va['rr_O'] - vb['rr_O'],
                              'delta_mrr_C': va['rr_C'] - vb['rr_C']}
                        K = boot.draws(f'{design}|{q}|{estimand}|{A}-{B}|{s}|{mode}')
                        res = boot.means(K, qs)
                        for name, (pt, reps) in res.items():
                            lo, hi = ci(reps)
                            rec(label, estimand, f'{A}-{B}', s, name, pt, lo, hi, int(sel.sum()), mode)
                        pO, rO = res['delta_O']
                        pU, rU = res['delta_U']
                        pW, rW = res['W']
                        pT, rT = res['delta_target']
                        c_rep, cstar_rep = rO - rU, np.abs(rO) / rW
                        lo, hi = ci(c_rep)
                        rec(label, estimand, f'{A}-{B}', s, 'c', pO - pU, lo, hi, int(sel.sum()), mode)
                        lo, hi = ci(cstar_rep)
                        rec(label, estimand, f'{A}-{B}', s, 'c_star', abs(pO) / pW, lo, hi, int(sel.sum()), mode)
                        rec(label, estimand, f'{A}-{B}', s, 'approx_err', pT - (pO - pW * (pO - pU)), np.nan, np.nan, int(sel.sum()), mode)

    # ---------- E1 bench_test: benchmark metrics over O, training-side effect E2-O minus E1-O,
    # and for NBFNet the graph-side effect deploy-O minus E1-O
    rr1, rr2 = runs.get(('E1', 'bench_test'), {}), runs.get(('E2', 'bench_test'), {})
    rrd = runs.get(('deploy', 'bench_test'), {})
    label = {'train': ('E1', 'training_side'), 'graph': ('deploy', 'graph_side')}
    if rr1:
        any_run = next(iter(rr1.values()))
        tid1 = any_run[1]
        boot = Boot(tid1, a.reps)
        models = [m for m in MODELS if any(k[0] == m for k in rr1)]
        for m in models:
            for s in sorted(s for mm, s in rr1 if mm == m):
                n, a_, t = agg(rr1[(m, s)][4], np.ones(N_CELLS, bool))
                r = rank(a_, t)
                qs = {'auc_O': auc(n, a_, t), 'rr_O': 1.0 / r, 'h10_O': (r <= 10) * 1.0}
                if (m, s) in rr2:
                    assert (rr2[(m, s)][1] == tid1).all()
                    v2 = per_query(rr2[(m, s)][4], 'primary')
                    qs['train_effect_auc_O'] = v2['auc_O'] - qs['auc_O']
                    qs['train_effect_mrr_O'] = v2['rr_O'] - qs['rr_O']
                if (m, s) in rrd:
                    assert (rrd[(m, s)][1] == tid1).all()
                    vd = per_query(rrd[(m, s)][4], 'primary')
                    qs['graph_effect_auc_O'] = vd['auc_O'] - qs['auc_O']
                    qs['graph_effect_mrr_O'] = vd['rr_O'] - qs['rr_O']
                K = boot.draws(f'E1|bench_test|{m}|{s}')
                for name, (pt, reps) in boot.means(K, qs).items():
                    lo, hi = ci(reps)
                    rec(*label.get(name.split('_')[0], ('E1', 'benchmark')), m, s, name, pt, lo, hi, len(tid1))

    # ---------- query side: u_test, E2 models, by stratum (query-entity region -> answer region)
    rru = runs.get(('E2', 'u_test'), {})
    if rru:
        any_run = next(iter(rru.values()))
        qid, tid, side = any_run[0], any_run[1], any_run[2]
        strat = []
        for t, sd in zip(tid, side):
            qe, ans = (head_of[t], tail_of[t]) if sd == 'tail' else (tail_of[t], head_of[t])
            strat.append(f'{E[qe]}->{E[ans]}')
        strat = np.array(strat)
        say('u_test query strata:', dict(collections.Counter(strat)))
        for (m, s), (_, _, _, _, counts) in sorted(rru.items()):
            n, a_, t = agg(counts, np.ones(N_CELLS, bool))
            r = rank(a_, t)
            for st in ['all'] + sorted(set(strat)):
                sel = np.ones(len(qid), bool) if st == 'all' else strat == st
                boot = Boot(tid[sel], a.reps)
                K = boot.draws(f'E2|u_test|{m}|{s}|{st}')
                qs = {'auc_C': auc(n, a_, t)[sel], 'rr_C': (1.0 / r)[sel], 'h10_C': (r <= 10)[sel] * 1.0}
                for name, (pt, reps) in boot.means(K, qs).items():
                    lo, hi = ci(reps)
                    rec('E2', f'query_side:{st}', m, s, name, pt, lo, hi, int(sel.sum()))

    # ---------- query side, POST HOC (design Sect. 13 item 3): relation group and reverse-edge leakage.
    # A query leaks when the reverse triple (t, r, h) of its test triple is an E2 training triple:
    # for the symmetric WordNet relations the answer is then one training edge away (for deploy, one
# edge of the graph NBFNet reads at inference).
    SYMMETRIC = {'_derivationally_related_form', '_also_see', '_verb_group', '_similar_to'}
    rel_of = {t['triple_id']: t['relation'] for t in T}
    train_set = {(t['head'], t['relation'], t['tail']) for t in T if t['role'] in ('bench_train', 'extra')}
    for design, q in (('E2', 'bench_test'), ('E2', 'u_test'), ('deploy', 'bench_test')):
        rr = runs.get((design, q), {})
        if not rr:
            continue
        any_run = next(iter(rr.values()))
        qid, tid = any_run[0], any_run[1]
        leak = np.array([(tail_of[x], rel_of[x], head_of[x]) in train_set for x in tid])
        sym = np.array([rel_of[x] in SYMMETRIC for x in tid])
        groups = {'all': np.ones(len(qid), bool), 'leak': leak, 'no_leak': ~leak,
                  'symmetric': sym, 'non_symmetric': ~sym, 'no_leak_non_symmetric': ~leak & ~sym}
        if q == 'u_test':
            regions = np.array([f"{E[head_of[x] if sd == 'tail' else tail_of[x]]}->{E[tail_of[x] if sd == 'tail' else head_of[x]]}"
                                for x, sd in zip(tid, any_run[2])])
            for st in sorted(set(regions)):
                groups[f'no_leak_non_symmetric&{st}'] = ~leak & ~sym & (regions == st)
        say(f'{design} {q}: leaking queries {int(leak.sum())} of {len(leak)}, symmetric-relation queries {int(sym.sum())}')
        for (m, s), (_, _, _, _, counts) in sorted(rr.items()):
            n, a_, t = agg(counts, np.ones(N_CELLS, bool))
            r = rank(a_, t)
            for gname, sel in groups.items():
                if sel.sum() == 0:
                    continue
                boot = Boot(tid[sel], a.reps)
                K = boot.draws(f'{design}|{q}|posthoc|{m}|{s}|{gname}')
                qs = {'auc_C': auc(n, a_, t)[sel], 'rr_C': (1.0 / r)[sel], 'h10_C': (r <= 10)[sel] * 1.0}
                for name, (pt, reps) in boot.means(K, qs).items():
                    lo, hi = ci(reps)
                    rec(design, f'posthoc_{q}:{gname}', m, s, name, pt, lo, hi, int(sel.sum()))

    # ---------- write
    with open(os.path.join(a.out, 'records.tsv'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('design\testimand\tsubject\tseed\tquantity\tvalue\tlo\thi\tn_queries\tcluster\n')
        for r in records:
            f.write('\t'.join([r[0], r[1], r[2], r[3], r[4], f'{r[5]:.10g}', f'{r[6]:.10g}', f'{r[7]:.10g}', str(r[8]), r[9]]) + '\n')
    with open(os.path.join(a.out, 'analysis.log'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(log) + '\n')
    say(f'wrote {len(records)} records to {a.out}/records.tsv')


if __name__ == '__main__':
    main()
