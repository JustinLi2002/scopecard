"""Post hoc descriptions of the pairwise run (experiments/MMLU_RECONSTRUCTION_DESIGN.md, Section 11, "Post hoc descriptions").

Recomputes the bootstrap, the labels and the reference subsets of mmlu_pairs_test.py with the same generator keys, checks
them against records_pairs.tsv, and then only describes: the floor of the kept questions, sign changes by floor status,
and the pairs that are reversals under simultaneous intervals. The parquet file is read in memory; nothing is executed.

Usage (from experiments/):
  python -E -P -X utf8 scripts/mmlu_pairs_posthoc.py results/mmlu_reconstruction results/mmlu_reconstruction/raw/cais_mmlu_all_test.parquet
"""
import collections
import hashlib
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq

OUT, MMLU = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
B = K = 2000
MIN_KEPT, CHUNK, CUTS = 10, 1500, (0.28, 0.30, 0.35)
SUMMARY = []
LINE = {'llama2-7b': 'Llama-2-7b', 'llama2-13b': 'Llama-2-13b', 'mistral-7b': 'Mistral-7B', 'gemma-7b': 'gemma-7b', 'yi-6b': 'Yi-6B'}


def rng_for(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big'))


def say(line=''):
    print(line, flush=True)
    SUMMARY.append(line)


# ---------------------------------------------------------------------------------------------------------------
# Data, scores, bootstrap and reference subsets: the same statements and keys as in mmlu_pairs_test.py
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
omega = n_kept[used] / n_kept[used].sum()
n_used_kept = int(n_kept[used].sum())

s_v = sum(w * Y[:, cols[s]].mean(axis=1) for w, s in zip(omega, used))
s_o = sum(w * Y[:, cols[s][kept[cols[s]]]].mean(axis=1) for w, s in zip(omega, used))
s_u = sum(w * Y[:, cols[s][~kept[cols[s]]]].mean(axis=1) for w, s in zip(omega, used))

r = rng_for('mmlu_pairs/bootstrap')
num_v, num_o, num_u = np.zeros((B, N)), np.zeros((B, N)), np.zeros((B, N))
den, den_u = np.zeros(B), np.zeros(B)
for s in range(S):
    c = cols[s]
    Ws = r.multinomial(len(c), np.full(len(c), 1.0 / len(c)), size=B).astype(np.float64)
    k_s = kept[c].astype(np.float64)
    all_s = Ws @ Y[:, c].T
    kept_den = Ws @ k_s
    kept_num = (Ws * k_s) @ Y[:, c].T
    if s in used:
        rem_den = len(c) - kept_den
        ok = rem_den > 0
        num_v += kept_den[:, None] * all_s / len(c)
        num_o += kept_num
        den += kept_den
        num_u += np.where(ok, kept_den, 0.0)[:, None] * (all_s - kept_num) / np.where(ok, rem_den, 1.0)[:, None]
        den_u += np.where(ok, kept_den, 0.0)
b_v, b_o, b_u = (num_v / den[:, None]).T, (num_o / den[:, None]).T, (num_u / den_u[:, None]).T
EST, BOOT = {'O': s_o, 'U': s_u, 'V': s_v}, {'O': b_o, 'U': b_u, 'V': b_v}

r = rng_for('mmlu_pairs/reference')
ref_sub, ref_comp = np.zeros((N, K)), np.zeros((N, K))
for w, s in zip(omega, used):
    c = cols[s]
    chosen = r.random((K, len(c))).argsort(axis=1)[:, :n_kept[s]]
    M = np.zeros((K, len(c)))
    np.put_along_axis(M, chosen, 1.0, axis=1)
    corr = Y[:, c] @ M.T
    ref_sub += corr
    ref_comp += w * (Y[:, c].sum(axis=1)[:, None] - corr) / (len(c) - n_kept[s])
ref_sub /= n_used_kept


def pair_statistics(rows):
    ia, ib = np.triu_indices(len(rows), 1)
    ia, ib = rows[ia], rows[ib]
    first = EST['O'][ia] >= EST['O'][ib]
    a, b = np.where(first, ia, ib), np.where(first, ib, ia)
    out = {'a': a, 'b': b}
    for name in 'OUV':
        out[name] = EST[name][a] - EST[name][b]
        for part in ('lo_', 'hi_', 'se_'):
            out[part + name] = np.empty(len(a))
    max_ov, max_ou = np.zeros(B), np.zeros(B)
    for start in range(0, len(a), CHUNK):
        sl = slice(start, start + CHUNK)
        t = {}
        for name in 'OUV':
            d = BOOT[name][a[sl]] - BOOT[name][b[sl]]
            out['lo_' + name][sl], out['hi_' + name][sl] = np.percentile(d, [2.5, 97.5], axis=1)
            se = d.std(axis=1, ddof=1)
            out['se_' + name][sl] = se
            with np.errstate(divide='ignore', invalid='ignore'):
                t[name] = np.where(se[:, None] > 0, np.abs(d - out[name][sl][:, None]) / se[:, None], 0.0).max(axis=0)
        max_ov = np.maximum(max_ov, np.maximum(t['O'], t['V']))
        max_ou = np.maximum(max_ou, np.maximum(t['O'], t['U']))
    out['q_ov'], out['q_ou'] = float(np.percentile(max_ov, 95)), float(np.percentile(max_ou, 95))
    return out


def reversal(p, simultaneous):
    def resolved(name):
        if simultaneous:
            lo = np.where(p['se_' + name] > 0, p[name] - p['q_ov'] * p['se_' + name], -np.inf)
            hi = np.where(p['se_' + name] > 0, p[name] + p['q_ov'] * p['se_' + name], np.inf)
        else:
            lo, hi = p['lo_' + name], p['hi_' + name]
        return (lo > 0) | (hi < 0)
    return resolved('O') & resolved('V') & (p['O'] * p['V'] < 0)


def order_stats(sub, whole, iu):
    d_sub = (sub[:, None] - sub[None, :])[iu]
    d_all = (whole[:, None] - whole[None, :])[iu]
    s_sub, s_all = np.sign(d_sub), np.sign(d_all)
    denom = np.sqrt(float(np.sum(s_sub != 0)) * float(np.sum(s_all != 0)))
    return (float(np.sum(s_sub * s_all) / denom) if denom > 0 else float('nan')), int(np.sum(d_sub * d_all < 0))


POP = {'A': np.flatnonzero((sets == 'filter') | (sets == 'clean')), 'K': np.flatnonzero(sets != 'random'), 'B': np.arange(N)}
PAIRS = {key: pair_statistics(rows) for key, rows in POP.items()}
REV_U = {key: reversal(PAIRS[key], False) for key in POP}
REV_S = {key: reversal(PAIRS[key], True) for key in POP}

records = {}
with (OUT / 'records_pairs.tsv').open(encoding='utf-8') as fh:
    next(fh)
    for text_line in fh:
        analysis, subj, quantity, value = text_line.rstrip('\n').split('\t')
        records[(analysis, subj, quantity)] = value
problems = []
for key in POP:
    p = PAIRS[key]
    got = {'reversal_simultaneous': int(REV_S[key].sum()), 'reversal_unadjusted': int(REV_U[key].sum()),
           'opposite_sign_kept_all': int(np.sum(p['O'] * p['V'] < 0))}
    for quantity, value in got.items():
        if int(records[(key, 'all', quantity)]) != value:
            problems.append(f'{key} {quantity}: {value} here, {records[(key, "all", quantity)]} in the run')
    if abs(float(records[(key, 'all', 'q_kept_all')]) - p['q_ov']) > 1e-4:
        problems.append(f'{key} q: {p["q_ov"]} here, {records[(key, "all", "q_kept_all")]} in the run')
if problems:
    raise SystemExit('the recomputation does not reproduce the run: ' + '; '.join(problems))

family = {}
with (OUT / 'families_panel_size.tsv').open(encoding='utf-8') as fh:
    next(fh)
    for text_line in fh:
        cells = text_line.rstrip('\n').split('\t')
        family[cells[0]] = cells[3]


def group_of(j):
    if sets[j] == 'lineage':
        return 'lineage, ' + LINE[tags[j]]
    if sets[j] == 'filter':
        return 'filter model'
    fam = family.get(names[j], '')
    return sets[j] + (', sibling' if tags[j] == 'sibling' else '') + (f', family {fam}' if fam and not fam.startswith('own:') else '')


say('# Pairwise order on original MMLU: post hoc descriptions')
say()
say('Rules: `experiments/MMLU_RECONSTRUCTION_DESIGN.md`, Section 11, "Post hoc descriptions" (written 2026-11-16, after the run and before '
    'these were computed). The labels are recomputed with the generator keys of the run and agree with `records_pairs.tsv` (reversals '
    'and opposite-sign pairs of A, K and B; critical values). Nothing here is a test fixed in advance. 1 point = 0.01 of accuracy.')
say()

# ---------------------------------------------------------------------------------------------------------------
# 1. The floor
# ---------------------------------------------------------------------------------------------------------------
say('## 1. The floor of the kept questions')
say()
table = pq.read_table(MMLU, columns=['subject', 'answer'])
subj_col = np.array(table.column('subject').to_pylist())
answers = np.array(table.column('answer').to_pylist())
if len(subj_col) != Q or not np.array_equal(subj_col, np.array(subject.tolist())):
    say('- The question file does not line up with the matrix; answer letters are not reported.')
else:
    letters = sorted(set(answers.tolist()))
    def shares(mask):
        return ', '.join(f'{"ABCD"[int(v)] if str(v).isdigit() and int(v) < 4 else v} {100 * np.mean(answers[mask] == v):.1f}%' for v in letters)
    in_used = np.isin(np.arange(Q), np.concatenate([cols[s] for s in used]))
    say(f'- Share of each answer letter, which is what a system that always gives that letter scores: all questions of the used '
        f'subjects {shares(in_used)}; kept questions {shares(kept)}; removed questions {shares(in_used & ~kept)}.')
clean = np.flatnonzero(sets == 'clean')
order = clean[np.argsort(s_v[clean])]
say()
say('| Clean-pool systems by a(all) | Systems | a(kept), lowest to highest | a(removed), lowest to highest |')
say('|---|---:|---|---|')
for lo, hi in ((0.0, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 1.01)):
    sel = order[(s_v[order] >= lo) & (s_v[order] < hi)]
    label = f'below {100 * hi:.0f}%' if lo == 0 else (f'{100 * lo:.0f}% and above' if hi > 1 else f'{100 * lo:.0f}% to {100 * hi:.0f}%')
    say(f'| {label} | {len(sel)} | {100 * s_o[sel].min():.1f}% to {100 * s_o[sel].max():.1f}% | {100 * s_u[sel].min():.1f}% to '
        f'{100 * s_u[sel].max():.1f}% |')
say()
hi_o = np.percentile(b_o, 97.5, axis=1)
low = np.flatnonzero(hi_o < 0.25)
say(f'- Systems whose bootstrap 95% interval of a(kept) lies below 25%: {len(low)} of {N}. By group: '
    + ('; '.join(f'{g} {n}' for g, n in sorted(collections.Counter(group_of(j) for j in low).items(), key=lambda kv: -kv[1])) or 'none') + '.')
if len(low):
    say('- The ten lowest of them: ' + '; '.join(f'{names[j]} ({group_of(j)}) {100 * s_o[j]:.1f}% on kept, {100 * s_v[j]:.1f}% on all'
                                                 for j in low[np.argsort(s_o[low])][:10]) + '.')
say(f'- The lowest a(kept) in the clean pool is {100 * s_o[clean].min():.1f}% ({names[clean[np.argmin(s_o[clean])]]}).')
say()

# ---------------------------------------------------------------------------------------------------------------
# 2. Sign changes by floor status
# ---------------------------------------------------------------------------------------------------------------
say('## 2. Sign changes by floor status')
say()
say('A system is above the floor at a cut c if its a(kept) is at least c. "Reference" is the central 95% of the same statistic over '
    'the 2,000 random subsets of the run, for the same systems. Reversals use the labels of the run (families of the whole population).')
say()
say('| Population | Cut | Systems above | Pairs of two systems above | Opposite signs, kept and all | Reference | tau_b | Reference | '
    'Reversals among them: unadjusted / simultaneous | Pairs with a system below | Opposite signs among those | Reversals among those: '
    'unadjusted / simultaneous |')
say('|---|---:|---:|---:|---:|---|---:|---|---|---:|---:|---|')
for key, rows in POP.items():
    p = PAIRS[key]
    opposite = p['O'] * p['V'] < 0
    for cut in CUTS:
        above = rows[s_o[rows] >= cut]
        both = np.isin(p['a'], above) & np.isin(p['b'], above)
        iu = np.triu_indices(len(above), 1)
        tau, opp = order_stats(s_o[above], s_v[above], iu)
        ref = np.array([order_stats(ref_sub[above, k], s_v[above], iu) for k in range(K)])
        t_lo, t_hi = np.percentile(ref[:, 0], [2.5, 97.5])
        o_lo, o_hi = np.percentile(ref[:, 1], [2.5, 97.5])
        assert opp == int(np.sum(both & opposite))
        say(f'| {key} | {100 * cut:.0f}% | {len(above)} | {int(both.sum())} | {opp} | {o_lo:.0f} to {o_hi:.0f} | {tau:.4f} | '
            f'{t_lo:.4f} to {t_hi:.4f} | {int(np.sum(both & REV_U[key]))} / {int(np.sum(both & REV_S[key]))} | {int(np.sum(~both))} | '
            f'{int(np.sum(~both & opposite))} | {int(np.sum(~both & REV_U[key]))} / {int(np.sum(~both & REV_S[key]))} |')
say()

# ---------------------------------------------------------------------------------------------------------------
# 3. The established reversals
# ---------------------------------------------------------------------------------------------------------------
say('## 3. The reversals established under simultaneous intervals')
say()
with (OUT / 'pairs_reversals_simultaneous.tsv').open('w', encoding='utf-8', newline='\n') as fh:
    fh.write('population\tleads_on_kept\tgroup\ta_all\ta_kept\tleads_on_all\tgroup\ta_all\ta_kept\tgap_kept\tgap_kept_lo\tgap_kept_hi\t'
             'gap_all\tgap_all_lo\tgap_all_hi\tgap_removed\n')
    for key in ('K', 'B'):
        p = PAIRS[key]
        chosen = np.flatnonzero(REV_S[key])
        for i in chosen[np.argsort(p['V'][chosen])]:
            a, b = p['a'][i], p['b'][i]
            q = p['q_ov']
            fh.write('\t'.join([key, names[a], group_of(a), f'{s_v[a]:.6f}', f'{s_o[a]:.6f}', names[b], group_of(b), f'{s_v[b]:.6f}',
                                f'{s_o[b]:.6f}', f'{p["O"][i]:.6f}', f'{p["O"][i] - q * p["se_O"][i]:.6f}',
                                f'{p["O"][i] + q * p["se_O"][i]:.6f}', f'{p["V"][i]:.6f}', f'{p["V"][i] - q * p["se_V"][i]:.6f}',
                                f'{p["V"][i] + q * p["se_V"][i]:.6f}', f'{p["U"][i]:.6f}']) + '\n')
        leaders, fallers = sorted(set(p['a'][chosen].tolist())), sorted(set(p['b'][chosen].tolist()))
        say(f'### Population {key}: {len(chosen)} reversals')
        say(f'- In every one the first system leads on the kept questions and the second leads on all questions. Gap on the kept '
            f'questions from {100 * p["O"][chosen].min():+.2f} to {100 * p["O"][chosen].max():+.2f} points; on all questions from '
            f'{100 * p["V"][chosen].min():+.2f} to {100 * p["V"][chosen].max():+.2f}.')
        say(f'- Systems that fall behind on the kept questions: {len(fallers)}. By group: '
            + '; '.join(f'{g} {n}' for g, n in sorted(collections.Counter(group_of(j) for j in fallers).items(), key=lambda kv: -kv[1]))
            + f'. Their a(all) is {100 * s_v[fallers].min():.1f}% to {100 * s_v[fallers].max():.1f}% and their a(kept) '
            f'{100 * s_o[fallers].min():.1f}% to {100 * s_o[fallers].max():.1f}%.')
        say('  - ' + '; '.join(f'{names[j]} ({group_of(j)}): {int(np.sum(p["b"][chosen] == j))} pairs' for j in fallers))
        say(f'- Systems that lead on the kept questions: {len(leaders)}. By group: '
            + '; '.join(f'{g} {n}' for g, n in sorted(collections.Counter(group_of(j) for j in leaders).items(), key=lambda kv: -kv[1]))
            + f'. Their a(all) is {100 * s_v[leaders].min():.1f}% to {100 * s_v[leaders].max():.1f}% and their a(kept) '
            f'{100 * s_o[leaders].min():.1f}% to {100 * s_o[leaders].max():.1f}%.')
        say('  - ' + '; '.join(f'{names[j]} ({group_of(j)}): {int(np.sum(p["a"][chosen] == j))} pairs' for j in leaders))
        say()
(OUT / 'summary_pairs_posthoc.md').write_text('\n'.join(SUMMARY) + '\n', encoding='utf-8')
print('done')
