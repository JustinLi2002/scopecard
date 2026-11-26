"""Order of systems on all questions and on the kept questions of original MMLU, against random subsets matched on subject,
estimated difficulty and answer key (experiments/MMLU_RECONSTRUCTION_DESIGN.md, Section 13, with Revision 1).

Exploratory extension. Reads the correctness matrix written by mmlu_reconstruction_test.py, the families of Section 10
(families_panel_size.tsv), the records of Section 11 (records_pairs.tsv, to check that R0 is reproduced) and the answer keys of
cais/mmlu, which are read in memory; nothing in that file is executed. No download.

Usage (from experiments/):
  python -E -P -X utf8 scripts/mmlu_difficulty_reference_test.py results/mmlu_reconstruction \
      results/mmlu_reconstruction/raw/cais_mmlu_all_test.parquet
Options for tests on synthetic data only: --draws=N (subsets per fold where the design has 2,000, and the R0 subsets, which must
equal the number used for records_pairs.tsv), --small=N (where the design has 500), --synthetic (the halves are not compared with
those of the matching check).
"""
import csv
import hashlib
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq

ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
FLAGS = dict((a[2:].split('=', 1) + [''])[:2] for a in sys.argv[1:] if a.startswith('--'))
OUT, MMLU = pathlib.Path(ARGS[0]), pathlib.Path(ARGS[1])
DRAWS = int(FLAGS.get('draws') or 2000)
SMALL = int(FLAGS.get('small') or 500)
SYNTHETIC = 'synthetic' in FLAGS
MIN_KEPT, FLOOR_ALL, CHUNK = 10, 0.40, 250
CUTS = ('0.28', '0.30', '0.35')
STRATA, N_TIE_KEYS, N_SPLITS, MAX_DIFF = 8, 10, 5, 6
LINE = {'llama2-7b': 'Llama-2-7b', 'llama2-13b': 'Llama-2-13b', 'mistral-7b': 'Mistral-7B', 'gemma-7b': 'gemma-7b', 'yi-6b': 'Yi-6B'}
LINEAGE_LINES = ('Llama-2-7b', 'Llama-2-13b', 'Mistral-7B', 'gemma-7b')
# Lines of half 1 in the matching check (design, Section 13); the script must find the same split.
HALF_1 = {'cerebras', 'command-r', 'llama', 'mpt', 'own:EleutherAI__gpt-j-6b', 'own:facebook__xglm-7.5B', 'own:gpt2-xl', 'phi',
          'pythia', 'redpajama', 'stablelm', 'yi'}
RECORDS, SUMMARY = [], []


def rng_for(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big'))


def rec(analysis, group, quantity, value):
    if isinstance(value, (float, np.floating)):
        value = f'{float(value):.6g}'
    RECORDS.append((analysis, group, quantity, str(value)))


def say(line=''):
    print(line, flush=True)
    SUMMARY.append(line)


def num(x):
    text = f'{float(x):.1f}'
    return text[:-2] if text.endswith('.0') else text


# ---------------------------------------------------------------------------------------------------------------
# Data and scores (as in Section 11)
# ---------------------------------------------------------------------------------------------------------------
data = np.load(OUT / 'correct_matrix.npz', allow_pickle=False)
Y = data['Y'].astype(np.float64)
names = [str(x) for x in data['systems']]
sets = np.array([str(x) for x in data['sets']])
tags = np.array([str(x) for x in data['tags']])
kept = data['kept'].astype(bool)
subject = data['subject']
subjects = sorted(set(subject.tolist()))
cols = [np.flatnonzero(subject == s) for s in subjects]
S, (N, Q) = len(subjects), Y.shape
n_all = np.array([len(c) for c in cols])
n_kept = np.array([int(kept[c].sum()) for c in cols])
used = [s for s in range(S) if n_kept[s] >= MIN_KEPT]
if any(n_all[s] - n_kept[s] < 1 for s in used):
    raise SystemExit('a used subject has no removed question (design, Section 11)')
omega = n_kept[used] / n_kept[used].sum()
K_TOTAL = int(n_kept[used].sum())
s_v = sum(w * Y[:, cols[s]].mean(axis=1) for w, s in zip(omega, used))
s_o = sum(w * Y[:, cols[s][kept[cols[s]]]].mean(axis=1) for w, s in zip(omega, used))
s_u = sum(w * Y[:, cols[s][~kept[cols[s]]]].mean(axis=1) for w, s in zip(omega, used))
used_q = np.concatenate([cols[s] for s in used])
kept_sel = np.zeros(Q, dtype=bool)
kept_sel[used_q] = kept[used_q]
wq = np.zeros(Q)
for w, s in zip(omega, used):
    wq[cols[s]] = w / (n_all[s] - n_kept[s])
if max(np.abs(Y[:, kept_sel].sum(axis=1) / K_TOTAL - s_o).max(), np.abs(Y @ (wq * ~kept_sel) - s_u).max()) > 1e-12:
    raise SystemExit('internal check failed: the linear forms of the scores do not reproduce Section 11')

table = pq.read_table(MMLU, columns=['subject', 'answer'])
if not np.array_equal(np.array(table.column('subject').to_pylist()), np.array(subject.tolist())):
    raise SystemExit('the question file does not line up with the matrix')
key = np.array(table.column('answer').to_pylist())

family = {}
with (OUT / 'families_panel_size.tsv').open(encoding='utf-8') as fh:
    for row in csv.DictReader(fh, delimiter='\t'):
        family[row['system']] = row['family']
line = np.array([family[n] for n in names])
idx_f = np.flatnonzero(sets == 'filter')
clean = np.flatnonzero(sets == 'clean')
lineage = np.flatnonzero(sets == 'lineage')
rows_a = np.flatnonzero((sets == 'filter') | (sets == 'clean'))
filter_lines = set(line[idx_f])
eligible = set(int(j) for j in clean if line[j] not in filter_lines)
capable = set(j for j in eligible if s_v[j] >= FLOOR_ALL)
lines = sorted(set(line[clean]))
size = {L: int(np.sum(line[clean] == L)) for L in lines}
floor_n = {L: int(np.sum((line[clean] == L) & (s_v[clean] < FLOOR_ALL))) for L in lines}
elig_n = {L: sum(1 for j in eligible if line[j] == L) for L in lines}
cap_n = {L: sum(1 for j in capable if line[j] == L) for L in lines}
lineage_line = {int(j): LINE[tags[j]] for j in lineage}


# ---------------------------------------------------------------------------------------------------------------
# Splits, estimators, difficulty, cells
# ---------------------------------------------------------------------------------------------------------------
def primary_split():
    """The split of the matching check (scripts/mmlu_difficulty_check.py), unchanged."""
    gen = rng_for('mmlu_pairs/difficulty/split')
    order = sorted(lines, key=lambda L: (-size[L], gen.random()))
    half, tot = {}, {1: [0, 0, 0], 2: [0, 0, 0]}
    for L in order:
        costs = {}
        for h in (1, 2):
            t = {k: list(v) for k, v in tot.items()}
            t[h] = [t[h][0] + size[L], t[h][1] + floor_n[L], t[h][2] + elig_n[L]]
            costs[h] = sum(abs(t[1][i] - t[2][i]) for i in range(3))
        best = [h for h in (1, 2) if costs[h] == min(costs.values())]
        h = best[0] if len(best) == 1 else (1 if gen.random() < 0.5 else 2)
        half[L] = h
        tot[h] = [tot[h][0] + size[L], tot[h][1] + floor_n[L], tot[h][2] + elig_n[L]]
    return half


def further_splits(primary):
    """Random assignments of lines, accepted when the halves differ by at most six systems and six estimators and the split is new."""
    gen = rng_for('mmlu_pairs/difficulty/split/sensitivity')
    seen = {frozenset(L for L in lines if primary[L] == h) for h in (1, 2)}
    out, tries = [], 0
    while len(out) < N_SPLITS and tries < 100000:
        tries += 1
        assign = {L: (1 if gen.random() < 0.5 else 2) for L in lines}
        h1 = frozenset(L for L in lines if assign[L] == 1)
        n1, e1 = sum(size[L] for L in h1), sum(cap_n[L] for L in h1)
        n2, e2 = len(clean) - n1, len(capable) - e1
        if abs(n1 - n2) <= MAX_DIFF and abs(e1 - e2) <= MAX_DIFF and min(e1, e2) > 0 and h1 not in seen:
            seen.update({h1, frozenset(lines) - h1})
            out.append(assign)
    return out, tries


def halves_of(assign):
    return {h: np.array([j for j in clean if assign[line[j]] == h]) for h in (1, 2)}


def estimators(members, kind):
    if kind == 'capable':
        return np.array([j for j in members if j in capable])
    if kind == 'eligible':
        return np.array([j for j in members if j in eligible])
    return np.asarray(members)


def difficulty(est, weighting):
    if weighting == 'plain':
        return Y[est].mean(axis=0)
    return np.mean([Y[est[line[est] == L]].mean(axis=0) for L in sorted(set(line[est]))], axis=0)


def make_cells(d, ties, m, by_key):
    """As cells_of in the matching check: per used subject, order by difficulty (keyed ties), cut into m near-equal strata."""
    cells = []
    for s in used:
        c = cols[s]
        ordered = c[np.lexsort((ties[c], d[c]))]
        for cell in np.array_split(ordered, min(m, len(c))):
            if by_key:
                cells.extend(cell[key[cell] == v] for v in sorted(set(key[cell].tolist())))
            else:
                cells.append(cell)
    return cells


def tie_dependence(d, ties, m):
    """Distinct difficulty values, and the share of questions in a group of tied values that a stratum boundary cuts."""
    dependent = 0
    for s in used:
        c = cols[s]
        dv = d[c[np.lexsort((ties[c], d[c]))]]
        starts = np.cumsum([len(p) for p in np.array_split(np.arange(len(c)), min(m, len(c)))])[:-1]
        cut = [dv[i] for i in starts if dv[i - 1] == dv[i]]
        if cut:
            dependent += int(np.isin(dv, cut).sum())
    return len(np.unique(d[used_q])), dependent / len(used_q)


def matching(cells, d, scored_clean):
    """Matching diagnostics of the check: exact expectations, no draw."""
    n_c = np.array([len(c) for c in cells])
    k_c = np.array([int(kept_sel[c].sum()) for c in cells])
    exp_ref = sum(k * Y[:, c].mean(axis=1) for k, c in zip(k_c, cells) if k) / K_TOTAL
    d_ref = sum(k * d[c].mean() for k, c in zip(k_c, cells) if k) / K_TOTAL
    groups = (scored_clean, scored_clean[s_v[scored_clean] < FLOOR_ALL], scored_clean[s_v[scored_clean] >= FLOOR_ALL])
    return {'cells': len(cells), 'all_kept': int(np.sum((k_c == n_c) & (n_c > 0))), 'smallest': int(n_c[n_c > 0].min()),
            'gap': 100 * (float(np.mean(d[kept_sel])) - d_ref), 'overlap': float(np.sum(k_c ** 2 / np.maximum(n_c, 1)) / K_TOTAL),
            'level': [100 * float(np.mean(exp_ref[g] - s_o[g])) if len(g) else float('nan') for g in groups],
            'key': [100 * sum(k * np.mean(key[c] == v) for k, c in zip(k_c, cells) if k) / K_TOTAL for v in range(4)]}


def draw_matched(cells, n_draws, gen_key):
    """In every cell as many questions as the kept set has there, uniformly without replacement."""
    gen = rng_for(gen_key)
    sel = np.zeros((n_draws, Q), dtype=bool)
    rows = np.arange(n_draws)[:, None]
    for c in cells:
        k = int(kept_sel[c].sum())
        if k == 0:
            continue
        if k == len(c):
            sel[:, c] = True
            continue
        sel[rows, c[gen.random((n_draws, len(c))).argsort(axis=1)[:, :k]]] = True
    return sel


def composition_ok(sel, cells):
    if sel[:, ~np.isin(np.arange(Q), used_q)].any():
        return False
    if any(not np.all(sel[:, cols[s]].sum(axis=1) == n_kept[s]) for s in used):
        return False
    return all(np.all(sel[:, c].sum(axis=1) == kept_sel[c].sum()) for c in cells)


def draw_r0(n_draws):
    """The random subsets of Section 11, same generator key and order of draws."""
    r = rng_for('mmlu_pairs/reference')
    sel = np.zeros((n_draws, Q), dtype=bool)
    rows = np.arange(n_draws)[:, None]
    for s in used:
        c = cols[s]
        sel[rows, c[r.random((n_draws, len(c))).argsort(axis=1)[:, :n_kept[s]]]] = True
    return sel


# ---------------------------------------------------------------------------------------------------------------
# Pairs, groups and the reference values
# ---------------------------------------------------------------------------------------------------------------
GROUPS = (['cc', 'fc', 'ff', 'lc', 'llf', 'A', 'K']
          + [f'{t}_{status}_{cut}' for cut in CUTS for t in ('cc', 'fc') for status in ('above', 'below')]
          + [f'lc_{L}' for L in LINEAGE_LINES])
GI = {g: i for i, g in enumerate(GROUPS)}
READ = ('cc', 'fc', 'lc', 'llf')
LABEL = {'cc': 'clean–clean, same half', 'fc': 'filter–clean', 'ff': 'filter–filter (described only)', 'lc': 'lineage–clean',
         'llf': 'lineage–lineage and lineage–filter', 'A': 'population A, all evaluated pairs',
         'K': 'population K, all evaluated pairs'}


def group_mask(g, kind, a, b):
    if g in ('cc', 'fc', 'ff', 'lc', 'llf'):
        return kind == g
    if g == 'A':
        return np.isin(kind, ['cc', 'fc', 'ff'])
    if g == 'K':
        return np.ones(len(kind), dtype=bool)
    if g.startswith('lc_'):
        other = np.where(sets[a] == 'lineage', a, b)
        return (kind == 'lc') & np.array([lineage_line.get(int(j), '') == g[3:] for j in other])
    t, status, cut = g.split('_')
    above = (s_o[a] >= float(cut)) & (s_o[b] >= float(cut))
    return (kind == t) & (above if status == 'above' else ~above)


def fold_setup(H):
    """Fold f: difficulty from half f; scored: the filter models, the clean-pool systems of the other half and the lineage pool.
    Pairs scored in one fold weigh 1, pairs scored in both folds 1/2 in each."""
    folds = {}
    for f in (1, 2):
        scored = np.concatenate([idx_f, H[3 - f], lineage])
        ia, ib = np.triu_indices(len(scored), 1)
        a, b = scored[ia], scored[ib]
        sa, sb = sets[a], sets[b]

        def both(x, y):
            return ((sa == x) & (sb == y)) | ((sa == y) & (sb == x))
        kind = np.full(len(a), '', dtype='<U3')
        kind[both('clean', 'clean')] = 'cc'
        kind[both('filter', 'clean')] = 'fc'
        kind[both('filter', 'filter')] = 'ff'
        kind[both('lineage', 'clean')] = 'lc'
        kind[both('lineage', 'lineage') | both('lineage', 'filter')] = 'llf'
        if np.any(kind == ''):
            raise SystemExit('a pair of unknown type')
        weight = np.where(np.isin(kind, ['ff', 'llf']), 0.5, 1.0)
        G = np.array([weight * group_mask(g, kind, a, b) for g in GROUPS])
        folds[f] = {'scored': scored, 'ia': ia, 'ib': ib, 'a': a, 'b': b, 'kind': kind, 'G': G, 'd_all': s_v[a] - s_v[b],
                    'mask_A': np.isin(kind, ['cc', 'fc', 'ff']), 'mask_K': np.ones(len(a), dtype=bool)}
    return folds


def tau_b(x, y):
    sx, sy = np.sign(x), np.sign(y)
    return float(np.sum(sx * sy) / np.sqrt(float(np.sum(sx != 0)) * float(np.sum(sy != 0))))


def kept_values(folds):
    v_all, v_comp, n_pairs = np.zeros(len(GROUPS)), np.zeros(len(GROUPS)), np.zeros(len(GROUPS))
    tau = {'A': 0.0, 'K': 0.0}
    for f in (1, 2):
        fd = folds[f]
        d_o, d_u = s_o[fd['a']] - s_o[fd['b']], s_u[fd['a']] - s_u[fd['b']]
        v_all += fd['G'] @ (d_o * fd['d_all'] < 0).astype(np.float64)
        v_comp += fd['G'] @ (d_o * d_u < 0).astype(np.float64)
        n_pairs += fd['G'].sum(axis=1)
        for k in ('A', 'K'):
            m = fd['mask_' + k]
            tau[k] += 0.5 * tau_b(d_o[m], fd['d_all'][m])
    return {'all': v_all, 'comp': v_comp, 'tau': tau, 'pairs': n_pairs}


def evaluate(sel, folds, n_draws):
    """Reference values: sign changes between subset and all questions and between subset and complement for every group, and
    tau_b (mean over the folds) for A and K. Subset i of fold 1 and subset i of fold 2 give reference value i."""
    v_all, v_comp = np.zeros((len(GROUPS), n_draws)), np.zeros((len(GROUPS), n_draws))
    tau = {'A': np.zeros(n_draws), 'K': np.zeros(n_draws)}
    for f in (1, 2):
        fd = folds[f]
        Ys = Y[fd['scored']]
        Yw = Ys * wq
        s_all = np.sign(fd['d_all'])
        for start in range(0, n_draws, CHUNK):
            stop = min(start + CHUNK, n_draws)
            M = sel[f][start:stop].astype(np.float64)
            sub = Ys @ M.T / K_TOTAL
            comp = Yw @ (1.0 - M).T
            d_sub = sub[fd['ia']] - sub[fd['ib']]
            d_comp = comp[fd['ia']] - comp[fd['ib']]
            v_all[:, start:stop] += fd['G'] @ (d_sub * fd['d_all'][:, None] < 0).astype(np.float64)
            v_comp[:, start:stop] += fd['G'] @ (d_sub * d_comp < 0).astype(np.float64)
            sg = np.sign(d_sub)
            for k in ('A', 'K'):
                m = fd['mask_' + k]
                top = (sg[m] * s_all[m][:, None]).sum(axis=0)
                tau[k][start:stop] += 0.5 * top / np.sqrt((sg[m] != 0).sum(axis=0) * float(np.sum(s_all[m] != 0)))
    return {'all': v_all, 'comp': v_comp, 'tau': tau}


def position(obs, ref):
    lo, hi = np.percentile(ref, [2.5, 97.5])
    pos = 'inside' if lo <= obs <= hi else ('above' if obs > hi else 'below')
    return {'pos': pos, 'lo': float(lo), 'hi': float(hi), 'median': float(np.median(ref)), 'pct': float(np.mean(ref <= obs))}


def reading(p0, p1, p2):
    """Readings fixed in the design (Section 13, revised), for the sign changes between subset and all questions."""
    if p2 == 'above':
        return ('a residual difference remains after matching on subject, estimated difficulty stratum and answer key '
                '(not attributed to the panel)')
    if p0 == 'above' and p1 == 'inside' and p2 == 'inside':
        return 'no excess detected beyond random subsets of the same subject and estimated-difficulty composition'
    if p1 == 'above' and p2 == 'inside':
        return 'an excess beyond the difficulty match, not detected once the answer-key mix is also matched'
    return f'{p0} R0, {p1} R1, {p2} R2 (described as it is)'


def ref_text(p, digits=0):
    if digits:
        return f'{p["lo"]:.{digits}f} to {p["hi"]:.{digits}f} ({p["median"]:.{digits}f})'
    return f'{num(p["lo"])} to {num(p["hi"])} ({num(p["median"])})'


def build(H, est_kind, weighting, tie_key_of, m, by_key, n_draws, gen_tag):
    sel, diag = {}, {}
    for f in (1, 2):
        est = estimators(H[f], est_kind)
        d = difficulty(est, weighting)
        ties = rng_for(tie_key_of(f)).random(Q)
        cells = make_cells(d, ties, m, by_key)
        sel[f] = draw_matched(cells, n_draws, f'{gen_tag}/fold{f}')
        distinct, dependent = tie_dependence(d, ties, m)
        diag[f] = dict(matching(cells, d, H[3 - f]), estimators=est, distinct=distinct, tie_dependent=dependent,
                       composition=composition_ok(sel[f], cells))
    return sel, diag


def primary_ties(f):
    return f'mmlu_pairs/difficulty/ties/{f}'


# ---------------------------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------------------------
say('# Difficulty-matched reference for the order of Section 11: results')
say()
say(f'Specification: `experiments/MMLU_RECONSTRUCTION_DESIGN.md`, Section 13 with Revision 1 (2026-11-16; exploratory; estimators and '
    f'strata chosen from a matching check that used the scored halves\' levels, then fixed before any order statistic on matched '
    f'subsets was computed). {N} systems; {len(used)} subjects with {len(used_q)} questions, {K_TOTAL} kept. Scores standardised '
    f'to the kept questions\' subject mix, as in Section 11. A sign change: the gap of a pair has opposite signs on the subset and on '
    f'all questions (or on the subset and its complement). References: R0, random subsets with the kept set\'s number of questions '
    f'per subject (the {DRAWS} subsets of Section 11); R1, also matched on {STRATA} difficulty strata per subject; R2, the same strata '
    f'crossed with the answer key. Central 95% of the reference, its median in brackets.' + (' **Synthetic data.**' if SYNTHETIC else ''))
say()

half = primary_split()
H = halves_of(half)
folds = fold_setup(H)
obs = kept_values(folds)

say('## 0. Checks')
say()
h1_lines = {L for L in lines if half[L] == 1}
if SYNTHETIC:
    say(f'- Halves: synthetic run, not compared with the matching check (half 1: {len(H[1])} systems, half 2: {len(H[2])}).')
else:
    same = h1_lines == HALF_1
    say(f'- Halves equal those of the matching check: **{"yes" if same else "NO"}**.')
    rec('check', 'all', 'halves_equal_matching_check', int(same))
    if not same:
        raise SystemExit('the split differs from the matching check')
sel0 = draw_r0(DRAWS)
if not all(np.all(sel0[:, cols[s]].sum(axis=1) == n_kept[s]) for s in used):
    raise SystemExit('R0 subsets do not have the kept set\'s composition')
iu = np.triu_indices(len(rows_a), 1)
pa_, pb_ = rows_a[iu[0]], rows_a[iu[1]]
r0_tau, r0_all, r0_comp = np.zeros(DRAWS), np.zeros(DRAWS), np.zeros(DRAWS)
Ya, d_all_a = Y[rows_a], s_v[pa_] - s_v[pb_]
pos_in_a = {int(j): i for i, j in enumerate(rows_a)}
ia_a, ib_a = np.array([pos_in_a[int(j)] for j in pa_]), np.array([pos_in_a[int(j)] for j in pb_])
for start in range(0, DRAWS, CHUNK):
    stop = min(start + CHUNK, DRAWS)
    M = sel0[start:stop].astype(np.float64)
    sub, comp = Ya @ M.T / K_TOTAL, (Ya * wq) @ (1.0 - M).T
    d_sub, d_comp = sub[ia_a] - sub[ib_a], comp[ia_a] - comp[ib_a]
    sg = np.sign(d_sub)
    r0_tau[start:stop] = (sg * np.sign(d_all_a)[:, None]).sum(axis=0) / np.sqrt((sg != 0).sum(axis=0) * float(np.sum(d_all_a != 0)))
    r0_all[start:stop] = (d_sub * d_all_a[:, None] < 0).sum(axis=0)
    r0_comp[start:stop] = (d_sub * d_comp < 0).sum(axis=0)
d_o_a, d_u_a = s_o[pa_] - s_o[pb_], s_u[pa_] - s_u[pb_]
obs_a = {'tau_b': tau_b(d_o_a, d_all_a), 'opposite_kept_all': float(np.sum(d_o_a * d_all_a < 0)),
         'opposite_kept_removed': float(np.sum(d_o_a * d_u_a < 0))}
earlier = {}
with (OUT / 'records_pairs.tsv').open(encoding='utf-8') as fh:
    for row in csv.DictReader(fh, delimiter='\t'):
        if row['analysis'] == 'A' and row['subject'] == 'reference':
            earlier[row['quantity']] = row['value']
mismatch = []
for slug, ref in (('tau_b', r0_tau), ('opposite_kept_all', r0_all), ('opposite_kept_removed', r0_comp)):
    lo, hi = np.percentile(ref, [2.5, 97.5])
    for q, v in ((slug, obs_a[slug]), (slug + '_reference_lo', lo), (slug + '_reference_hi', hi),
                 (slug + '_percentile', float(np.mean(ref <= obs_a[slug])))):
        mine = f'{float(v):.6g}'
        rec('check_R0_population_A', 'all', q, mine)
        if earlier.get(q) != mine:
            mismatch.append(f'{q}: {mine} against {earlier.get(q)}')
say(f'- R0 on all {len(pa_)} pairs of population A, with the {DRAWS} subsets of Section 11: sign changes between subset and all '
    f'questions {num(np.percentile(r0_all, 2.5))} to {num(np.percentile(r0_all, 97.5))}, between subset and complement '
    f'{num(np.percentile(r0_comp, 2.5))} to {num(np.percentile(r0_comp, 97.5))}, tau_b {np.percentile(r0_tau, 2.5):.4f} to '
    f'{np.percentile(r0_tau, 97.5):.4f}; kept set {num(obs_a["opposite_kept_all"])}, {num(obs_a["opposite_kept_removed"])} and '
    f'{obs_a["tau_b"]:.4f}. Agreement with `records_pairs.tsv` of Section 11 (every value to six digits): '
    f'**{"yes" if not mismatch else "NO: " + "; ".join(mismatch)}**.')
rec('check', 'all', 'R0_reproduces_section_11', int(not mismatch))
if mismatch:
    raise SystemExit('R0 does not reproduce Section 11')

res0 = evaluate({1: sel0, 2: sel0}, folds, DRAWS)
sel1, diag1 = build(H, 'capable', 'plain', primary_ties, STRATA, False, DRAWS, 'mmlu_pairs/difficulty/primary/R1')
res1 = evaluate(sel1, folds, DRAWS)
del sel1
sel2, diag2 = build(H, 'capable', 'plain', primary_ties, STRATA, True, DRAWS, 'mmlu_pairs/difficulty/primary/R2')
res2 = evaluate(sel2, folds, DRAWS)
del sel2
comp_ok = all(diag[f]['composition'] for diag in (diag1, diag2) for f in (1, 2))
say(f'- Every matched subset has, in every cell, the kept set\'s number of questions: **{"yes" if comp_ok else "NO"}**.')
rec('check', 'all', 'matched_composition', int(comp_ok))
if not comp_ok:
    raise SystemExit('a matched subset has the wrong composition')
say()

say('## 1. Setup')
say()
for h in (1, 2):
    say(f'- Half {h}: {len(H[h])} clean-pool systems; lines ' + '; '.join(f'{L} ({size[L]})' for L in lines if half[L] == h) + '.')
for f, diag in ((1, diag1), (2, diag2)):
    est = diag[f]['estimators']
    by_line = {}
    for j in est:
        by_line[line[j]] = by_line.get(line[j], 0) + 1
    say(f'- Fold {f}: difficulty from {len(est)} systems of half {f} (' + ', '.join(f'{L} {n}' for L, n in sorted(by_line.items()))
        + f'); scored: the {len(idx_f)} filter models, the {len(H[3 - f])} clean-pool systems of half {3 - f} and the {len(lineage)} '
        f'lineage systems.')
say()
say('| Fold | Reference | Cells | Cells all kept | Smallest cell | Kept minus matched difficulty, points | Expected overlap | Level gap, '
    'scored clean, points: all / below 40% / at least 40% | Key shares A/B/C/D, % | Distinct difficulty values | Questions whose '
    'stratum depends on the tie-break |')
say('|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|')
for ref_name, diag in (('R1', diag1), ('R2', diag2)):
    for f in (1, 2):
        g = diag[f]
        say(f'| {f} | {ref_name} | {g["cells"]} | {g["all_kept"]} | {g["smallest"]} | {g["gap"]:+.2f} | {g["overlap"]:.3f} | '
            + ' / '.join(f'{x:+.2f}' for x in g['level']) + ' | ' + ' / '.join(f'{x:.1f}' for x in g['key'])
            + f' | {g["distinct"]} | {100 * g["tie_dependent"]:.1f}% |')
        for q in ('cells', 'all_kept', 'smallest', 'gap', 'overlap', 'distinct', 'tie_dependent'):
            rec(f'matching_{ref_name}_fold{f}', 'all', q, g[q])
        for q, v in zip(('level_all', 'level_below_40', 'level_at_least_40'), g['level']):
            rec(f'matching_{ref_name}_fold{f}', 'all', q, v)
say()
say('Level gap: the expected score of the scored clean-pool systems on a matched subset minus their score on the kept questions; '
    'positive means the matched subsets are easier for them. This residual is unresolved matching bias; no direction of its effect on '
    'the counts is assumed.')
say()

say('## 2. Primary result')
say()
say('Sign changes between subset and all questions. Pairs scored in both folds count with the mean of their two indicators, so a '
    'count can end in .5.')
say()
say('| Pair type | Pairs | Kept set | R0 | R1 | R2 | Position against R0 / R1 / R2 | Share of the excess over the R0 median reached by the '
    'R1 / R2 median | Reading fixed in the design |')
say('|---|---:|---:|---|---|---|---|---|---|')
PRIMARY = {}
for g in ['cc', 'fc', 'lc', 'llf', 'ff', 'A', 'K']:
    i = GI[g]
    o = obs['all'][i]
    p = {name: position(o, res['all'][i]) for name, res in (('R0', res0), ('R1', res1), ('R2', res2))}
    excess = o - p['R0']['median']
    shares = [f'{100 * (p[n]["median"] - p["R0"]["median"]) / excess:.0f}%' if p['R0']['pos'] == 'above' else 'n/a'
              for n in ('R1', 'R2')]
    for n, share in zip(('R1', 'R2'), shares):
        rec('primary_all', g, f'share_of_excess_{n}', share)
    text = reading(p['R0']['pos'], p['R1']['pos'], p['R2']['pos']) if g in READ else 'described only'
    PRIMARY[g] = (o, p, text)
    say(f'| {LABEL[g]} | {num(obs["pairs"][i])} | {num(o)} | {ref_text(p["R0"])} | {ref_text(p["R1"])} | {ref_text(p["R2"])} | '
        f'{p["R0"]["pos"]} / {p["R1"]["pos"]} / {p["R2"]["pos"]} | {shares[0]} / {shares[1]} | {text} |')
    rec('primary_all', g, 'pairs', obs['pairs'][i])
    rec('primary_all', g, 'kept', o)
    for name in ('R0', 'R1', 'R2'):
        for q in ('lo', 'hi', 'median', 'pct', 'pos'):
            rec('primary_all', g, f'{name}_{q}', p[name][q])
    rec('primary_all', g, 'reading', text)
say()
say('Sign changes between subset and complement (for the kept set: kept and removed questions).')
say()
say('| Pair type | Kept set | R0 | R1 | R2 | Position against R0 / R1 / R2 |')
say('|---|---:|---|---|---|---|')
for g in ['cc', 'fc', 'lc', 'llf', 'ff', 'A', 'K']:
    i = GI[g]
    o = obs['comp'][i]
    p = {name: position(o, res['comp'][i]) for name, res in (('R0', res0), ('R1', res1), ('R2', res2))}
    say(f'| {LABEL[g]} | {num(o)} | {ref_text(p["R0"])} | {ref_text(p["R1"])} | {ref_text(p["R2"])} | '
        f'{p["R0"]["pos"]} / {p["R1"]["pos"]} / {p["R2"]["pos"]} |')
    rec('primary_complement', g, 'kept', o)
    for name in ('R0', 'R1', 'R2'):
        for q in ('lo', 'hi', 'median', 'pos'):
            rec('primary_complement', g, f'{name}_{q}', p[name][q])
say()
say('Kendall tau_b between the scores on the subset and on all questions, per fold over the systems scored there, mean of the two folds.')
say()
say('| Population | Kept set | R0 | R1 | R2 | Position against R0 / R1 / R2 |')
say('|---|---:|---|---|---|---|')
for k in ('A', 'K'):
    o = obs['tau'][k]
    p = {name: position(o, res['tau'][k]) for name, res in (('R0', res0), ('R1', res1), ('R2', res2))}
    say(f'| {k} | {o:.4f} | {ref_text(p["R0"], 4)} | {ref_text(p["R1"], 4)} | {ref_text(p["R2"], 4)} | '
        f'{p["R0"]["pos"]} / {p["R1"]["pos"]} / {p["R2"]["pos"]} |')
    rec('primary_tau', k, 'kept', o)
    for name in ('R0', 'R1', 'R2'):
        for q in ('lo', 'hi', 'median', 'pos'):
            rec('primary_tau', k, f'{name}_{q}', p[name][q])
say()

say('## 3. Descriptions')
say()
say('By floor status (a(kept) at or above the cut for both systems, or below it for at least one), sign changes between subset and '
    'all questions.')
say()
say('| Pairs | Cut | Status | Pairs | Kept set | R0 | R1 | R2 | Position against R0 / R1 / R2 |')
say('|---|---|---|---:|---:|---|---|---|---|')
for t in ('cc', 'fc'):
    for cut in CUTS:
        for status in ('above', 'below'):
            g = f'{t}_{status}_{cut}'
            i = GI[g]
            o = obs['all'][i]
            p = {name: position(o, res['all'][i]) for name, res in (('R0', res0), ('R1', res1), ('R2', res2))}
            say(f'| {LABEL[t]} | {cut} | {"both above" if status == "above" else "at least one below"} | {num(obs["pairs"][i])} | '
                f'{num(o)} | {ref_text(p["R0"])} | {ref_text(p["R1"])} | {ref_text(p["R2"])} | '
                f'{p["R0"]["pos"]} / {p["R1"]["pos"]} / {p["R2"]["pos"]} |')
            rec('floor', g, 'pairs', obs['pairs'][i])
            rec('floor', g, 'kept', o)
            for name in ('R0', 'R1', 'R2'):
                for q in ('lo', 'hi', 'median', 'pos'):
                    rec('floor', g, f'{name}_{q}', p[name][q])
say()
say('Lineage–clean pairs by line of the lineage system, sign changes between subset and all questions.')
say()
say('| Line | Pairs | Kept set | R0 | R1 | R2 | Position against R0 / R1 / R2 |')
say('|---|---:|---:|---|---|---|---|')
for L in LINEAGE_LINES:
    i = GI[f'lc_{L}']
    o = obs['all'][i]
    p = {name: position(o, res['all'][i]) for name, res in (('R0', res0), ('R1', res1), ('R2', res2))}
    say(f'| {L} | {num(obs["pairs"][i])} | {num(o)} | {ref_text(p["R0"])} | {ref_text(p["R1"])} | {ref_text(p["R2"])} | '
        f'{p["R0"]["pos"]} / {p["R1"]["pos"]} / {p["R2"]["pos"]} |')
    rec('by_line', L, 'pairs', obs['pairs'][i])
    rec('by_line', L, 'kept', o)
    for name in ('R0', 'R1', 'R2'):
        for q in ('lo', 'hi', 'median', 'pos'):
            rec('by_line', L, f'{name}_{q}', p[name][q])
say()

# ---------------------------------------------------------------------------------------------------------------
# Sensitivity (reported, not read)
# ---------------------------------------------------------------------------------------------------------------
say('## 4. Sensitivity (reported, not read)')
say()
say('Sign changes between subset and all questions for the four pair types that are read. Each row is one reference; the kept '
    'values are those of the primary split unless the row is a further split.')
say()
say('| Variant | Reference | Subsets per fold | Level gap, scored clean at least 40%, folds 1 / 2 | ' + ' | '.join(LABEL[g] for g in READ) + ' |')
say('|---|---|---:|---|' + '---|' * len(READ))


def sensitivity_row(label, ref_name, n_draws, diag, res, kept_obs, tag):
    cells = []
    for g in READ:
        i = GI[g]
        p = position(kept_obs['all'][i], res['all'][i])
        cells.append(f'{num(kept_obs["all"][i])} against {ref_text(p)}: {p["pos"]}')
        for q in ('lo', 'hi', 'median', 'pos'):
            rec(f'sensitivity/{tag}', g, q, p[q])
        rec(f'sensitivity/{tag}', g, 'kept', kept_obs['all'][i])
    levels = ' / '.join(f'{diag[f]["level"][2]:+.2f}' for f in (1, 2))
    say(f'| {label} | {ref_name} | {n_draws} | {levels} | ' + ' | '.join(cells) + ' |')


VARIANTS = [('R1 at 4 strata', 'R1', 'capable', 'plain', 4, False), ('R1 at 16 strata', 'R1', 'capable', 'plain', 16, False),
            ('R2 at 4 strata', 'R2', 'capable', 'plain', 4, True)]
for est_kind, est_label in (('eligible', 'all eligible estimators'), ('whole', 'whole half as estimators')):
    for ref_name, by_key in (('R1', False), ('R2', True)):
        VARIANTS.append((est_label, ref_name, est_kind, 'plain', STRATA, by_key))
for ref_name, by_key in (('R1', False), ('R2', True)):
    VARIANTS.append(('equal weight per line', ref_name, 'capable', 'line', STRATA, by_key))
for label, ref_name, est_kind, weighting, m, by_key in VARIANTS:
    tag = f'{label}/{ref_name}'.replace(' ', '_')
    sel, diag = build(H, est_kind, weighting, primary_ties, m, by_key, DRAWS, f'mmlu_pairs/difficulty/{tag}')
    if not all(diag[f]['composition'] for f in (1, 2)):
        raise SystemExit(f'composition check failed in {tag}')
    res = evaluate(sel, folds, DRAWS)
    del sel
    sensitivity_row(label, ref_name, DRAWS, diag, res, obs, tag)
    for f in (1, 2):
        rec(f'sensitivity/{tag}', f'fold{f}', 'distinct_values', diag[f]['distinct'])
        rec(f'sensitivity/{tag}', f'fold{f}', 'tie_dependent', diag[f]['tie_dependent'])
        rec(f'sensitivity/{tag}', f'fold{f}', 'level_at_least_40', diag[f]['level'][2])
say()

# Tie-break keys
TIE = {name: [] for name in ('R1', 'R2')}
tie_diag = []
for t in range(1, N_TIE_KEYS + 1):
    for ref_name, by_key in (('R1', False), ('R2', True)):
        sel, diag = build(H, 'capable', 'plain', lambda f, t=t: f'mmlu_pairs/difficulty/ties/{f}/sensitivity/{t}', STRATA, by_key,
                          SMALL, f'mmlu_pairs/difficulty/tie{t}/{ref_name}')
        if not all(diag[f]['composition'] for f in (1, 2)):
            raise SystemExit(f'composition check failed for tie key {t}')
        res = evaluate(sel, folds, SMALL)
        del sel
        TIE[ref_name].append({g: position(obs['all'][GI[g]], res['all'][GI[g]]) for g in READ})
        tie_diag.append((t, ref_name, [diag[f]['tie_dependent'] for f in (1, 2)]))
say(f'Ten further tie-break keys ({SMALL} subsets per fold each), against the primary key ({DRAWS} subsets per fold). Share of questions '
    f'whose stratum depends on the tie-break, folds 1 and 2: R1 ' + ', '.join(f'{100 * x:.1f}%' for x in (diag1[1]['tie_dependent'],
                                                                                                            diag1[2]['tie_dependent']))
    + ' with the primary key.')
say()
say('| Pair type | Reference | Kept set | Primary key | Lower bound over the ten keys | Upper bound over the ten keys | Median over the '
    'ten keys | Keys giving the primary\'s position |')
say('|---|---|---:|---|---|---|---|---:|')
for g in READ:
    for ref_name, res in (('R1', res1), ('R2', res2)):
        prim = position(obs['all'][GI[g]], res['all'][GI[g]])
        rows = TIE[ref_name]
        los, his, meds = [r[g]['lo'] for r in rows], [r[g]['hi'] for r in rows], [r[g]['median'] for r in rows]
        same = sum(r[g]['pos'] == prim['pos'] for r in rows)
        say(f'| {LABEL[g]} | {ref_name} | {num(obs["all"][GI[g]])} | {ref_text(prim)}: {prim["pos"]} | {num(min(los))} to {num(max(los))} | '
            f'{num(min(his))} to {num(max(his))} | {num(min(meds))} to {num(max(meds))} | {same} of {len(rows)} |')
        for q, v in (('lo_min', min(los)), ('lo_max', max(los)), ('hi_min', min(his)), ('hi_max', max(his)), ('median_min', min(meds)),
                     ('median_max', max(meds)), ('keys_same_position', same)):
            rec(f'tie_keys/{ref_name}', g, q, v)
say()

# Further splits
splits, tries = further_splits(half)
say(f'Five further splits of the clean pool by line ({len(splits)} accepted after {tries} random assignments; {SMALL} subsets per fold '
    f'for R1 and R2, the {DRAWS} subsets of R0). The kept values change with the split, since the pairs scored change.')
say()
say('| Split | Systems in halves 1 / 2 | Estimators 1 / 2 | Pair type | Kept set | R0 | R1 | R2 | Position against R0 / R1 / R2 | Reading |')
say('|---|---|---|---|---:|---|---|---|---|---|')
with (OUT / 'halves.tsv').open('w', encoding='utf-8', newline='\n') as fh:
    fh.write('split\tline\thalf\tsystems\tbelow_40\teligible\testimators\n')
    for L in lines:
        fh.write(f'primary\t{L}\t{half[L]}\t{size[L]}\t{floor_n[L]}\t{elig_n[L]}\t{cap_n[L]}\n')
    for s_id, assign in enumerate(splits, 1):
        for L in lines:
            fh.write(f'{s_id}\t{L}\t{assign[L]}\t{size[L]}\t{floor_n[L]}\t{elig_n[L]}\t{cap_n[L]}\n')
for s_id, assign in enumerate(splits, 1):
    Hs = halves_of(assign)
    fs = fold_setup(Hs)
    obs_s = kept_values(fs)
    r0s = evaluate({1: sel0, 2: sel0}, fs, DRAWS)
    out = {}
    for ref_name, by_key in (('R1', False), ('R2', True)):
        sel, diag = build(Hs, 'capable', 'plain', lambda f, s_id=s_id: f'mmlu_pairs/difficulty/ties/{f}/split/{s_id}', STRATA, by_key,
                          SMALL, f'mmlu_pairs/difficulty/split{s_id}/{ref_name}')
        if not all(diag[f]['composition'] for f in (1, 2)):
            raise SystemExit(f'composition check failed for split {s_id}')
        out[ref_name] = evaluate(sel, fs, SMALL)
        del sel
    n_est = [len(estimators(Hs[h], 'capable')) for h in (1, 2)]
    for g in READ:
        i = GI[g]
        o = obs_s['all'][i]
        p = {'R0': position(o, r0s['all'][i]), 'R1': position(o, out['R1']['all'][i]), 'R2': position(o, out['R2']['all'][i])}
        say(f'| {s_id} | {len(Hs[1])} / {len(Hs[2])} | {n_est[0]} / {n_est[1]} | {LABEL[g]} | {num(o)} | {ref_text(p["R0"])} | '
            f'{ref_text(p["R1"])} | {ref_text(p["R2"])} | {p["R0"]["pos"]} / {p["R1"]["pos"]} / {p["R2"]["pos"]} | '
            f'{reading(p["R0"]["pos"], p["R1"]["pos"], p["R2"]["pos"])} |')
        rec(f'split{s_id}', g, 'kept', o)
        rec(f'split{s_id}', g, 'pairs', obs_s['pairs'][i])
        for name in ('R0', 'R1', 'R2'):
            for q in ('lo', 'hi', 'median', 'pos'):
                rec(f'split{s_id}', g, f'{name}_{q}', p[name][q])
say()

with (OUT / 'records_difficulty_reference.tsv').open('w', encoding='utf-8', newline='\n') as fh:
    fh.write('analysis\tgroup\tquantity\tvalue\n')
    for row in RECORDS:
        fh.write('\t'.join(row) + '\n')
(OUT / 'summary_difficulty_reference.md').write_text('\n'.join(SUMMARY) + '\n', encoding='utf-8')
print(f'wrote {len(RECORDS)} records')
