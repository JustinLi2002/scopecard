"""Matching check for a difficulty-matched random-subset reference on original MMLU
(experiments/MMLU_RECONSTRUCTION_DESIGN.md, Section 13). Reads no order outcome: no pairwise statistic is computed.

It splits the clean pool into two halves by model line, estimates each question's difficulty from systems of one half, cuts every used
subject into difficulty strata (optionally crossed with the answer key), and reports for each choice: how far the kept questions'
difficulty is from that of a matched random subset, how much a matched subset overlaps the kept set, how far the expected score of the
other half's clean-pool systems on a matched subset is from their score on the kept set, and the expected share of each answer key.
Expected values are exact (no draws): a matched subset holds k_c questions drawn at random from each cell c.

Usage (from experiments/):
  python -E -P -X utf8 scripts/mmlu_difficulty_check.py results/mmlu_reconstruction results/mmlu_reconstruction/raw/cais_mmlu_all_test.parquet
The parquet file is read in memory for the answer keys; nothing in it is executed.
"""
import csv
import hashlib
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq

OUT, MMLU = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
MIN_KEPT, FLOOR_ALL = 10, 0.40
STRATA = (2, 4, 8, 16)


def rng_for(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big'))


data = np.load(OUT / 'correct_matrix.npz', allow_pickle=False)
Y = data['Y'].astype(np.float64)
names = [str(x) for x in data['systems']]
sets = np.array([str(x) for x in data['sets']])
kept = data['kept'].astype(bool)
subject = data['subject']
subjects = sorted(set(subject.tolist()))
cols = [np.flatnonzero(subject == s) for s in subjects]
n_all = np.array([len(c) for c in cols])
n_kept = np.array([int(kept[c].sum()) for c in cols])
used = [s for s in range(len(subjects)) if n_kept[s] >= MIN_KEPT]
omega = n_kept[used] / n_kept[used].sum()
K_total = int(n_kept[used].sum())
s_v = sum(w * Y[:, cols[s]].mean(axis=1) for w, s in zip(omega, used))
s_o = sum(w * Y[:, cols[s][kept[cols[s]]]].mean(axis=1) for w, s in zip(omega, used))
kept_used = np.concatenate([cols[s][kept[cols[s]]] for s in used])

table = pq.read_table(MMLU, columns=['subject', 'answer'])
if not np.array_equal(np.array(table.column('subject').to_pylist()), np.array(subject.tolist())):
    raise SystemExit('the question file does not line up with the matrix')
key = np.array(table.column('answer').to_pylist())

family = {}
with (OUT / 'families_panel_size.tsv').open(encoding='utf-8') as fh:
    for row in csv.DictReader(fh, delimiter='\t'):
        family[row['system']] = row['family']
line = np.array([family[n] for n in names])
filter_lines = set(line[sets == 'filter'])
clean = np.flatnonzero(sets == 'clean')
eligible = set(j for j in clean if line[j] not in filter_lines)

# ---------------------------------------------------------------------------------------------------------------
# Split of the clean pool into two halves by line
# ---------------------------------------------------------------------------------------------------------------
lines = sorted(set(line[clean]))
members = {L: [j for j in clean if line[j] == L] for L in lines}
size = {L: len(members[L]) for L in lines}
floor = {L: sum(s_v[j] < FLOOR_ALL for j in members[L]) for L in lines}
elig = {L: sum(j in eligible for j in members[L]) for L in lines}
gen = rng_for('mmlu_pairs/difficulty/split')
order = sorted(lines, key=lambda L: (-size[L], gen.random()))
half, tot = {}, {1: [0, 0, 0], 2: [0, 0, 0]}
for L in order:
    costs = {}
    for h in (1, 2):
        t = {k: list(v) for k, v in tot.items()}
        t[h] = [t[h][0] + size[L], t[h][1] + floor[L], t[h][2] + elig[L]]
        costs[h] = sum(abs(t[1][i] - t[2][i]) for i in range(3))
    best = [h for h in (1, 2) if costs[h] == min(costs.values())]
    h = best[0] if len(best) == 1 else (1 if gen.random() < 0.5 else 2)
    half[L] = h
    tot[h] = [tot[h][0] + size[L], tot[h][1] + floor[L], tot[h][2] + elig[L]]
H = {h: np.array([j for j in clean if half[line[j]] == h]) for h in (1, 2)}

print('# Matching check for the difficulty-matched reference (reads no order outcome)')
print()
print(f'{len(used)} used subjects, {int(n_all[used].sum())} questions, {K_total} kept. Clean pool {len(clean)} systems in {len(lines)} lines; '
      f'lines holding a filter model: {", ".join(sorted(filter_lines))}; eligible as difficulty estimators: {len(eligible)}.')
print()
for h in (1, 2):
    print(f'Half {h}: {len(H[h])} systems ({sum(j in eligible for j in H[h])} eligible, of which '
          f'{sum(j in eligible and s_v[j] >= FLOOR_ALL for j in H[h])} with a(all) >= {FLOOR_ALL:.0%}), '
          f'{sum(s_v[j] < FLOOR_ALL for j in H[h])} below {FLOOR_ALL:.0%} on all questions, a(all) mean {s_v[H[h]].mean():.3f}, '
          f'quartiles {", ".join(f"{q:.3f}" for q in np.percentile(s_v[H[h]], [25, 50, 75]))}.')
    print('  lines: ' + '; '.join(f'{L} ({size[L]})' for L in lines if half[L] == h))
print()


def cells_of(h, est, m, by_key):
    d = Y[est].mean(axis=0)
    ties = rng_for(f'mmlu_pairs/difficulty/ties/{h}').random(Y.shape[1])
    out = []
    for s in used:
        c = cols[s]
        ordered = c[np.lexsort((ties[c], d[c]))]
        for cell in np.array_split(ordered, min(m, len(c))):
            if by_key:
                out.extend(cell[key[cell] == v] for v in sorted(set(key[cell].tolist())))
            else:
                out.append(cell)
    return out, d


def matching(h, est, m, by_key):
    cells, d = cells_of(h, est, m, by_key)
    scored = H[3 - h]
    groups = (scored, scored[s_v[scored] < FLOOR_ALL], scored[s_v[scored] >= FLOOR_ALL])
    n_c = np.array([len(c) for c in cells])
    k_c = np.array([int(kept[c].sum()) for c in cells])
    d_ref = sum(k * d[c].mean() for k, c in zip(k_c, cells) if k) / K_total
    exp_ref = sum(k * Y[:, c].mean(axis=1) for k, c in zip(k_c, cells) if k) / K_total
    key_ref = {v: sum(k * np.mean(key[c] == v) for k, c in zip(k_c, cells) if k) / K_total for v in range(4)}
    return {'cells': len(cells), 'all_kept': int(np.sum((k_c == n_c) & (n_c > 0))), 'smallest': int(n_c[n_c > 0].min()),
            'gap': 100 * (float(np.mean(d[kept_used])) - d_ref), 'overlap': float(np.sum(k_c ** 2 / np.maximum(n_c, 1)) / K_total),
            'level': [100 * float(np.mean(exp_ref[g] - s_o[g])) for g in groups], 'key': key_ref}


ESTIMATORS = {
    'eligible': lambda h: np.array([j for j in H[h] if j in eligible]),
    'eligible, a(all) >= 40%': lambda h: np.array([j for j in H[h] if j in eligible and s_v[j] >= FLOOR_ALL]),
    'whole half (lines of filter models included)': lambda h: H[h],
}
print(f'Answer keys of the kept questions: ' + ', '.join(f'{"ABCD"[v]} {100 * np.mean(key[kept_used] == v):.1f}%' for v in range(4))
      + '; of all questions of the used subjects: '
      + ', '.join(f'{"ABCD"[v]} {100 * np.mean(key[np.concatenate([cols[s] for s in used])] == v):.1f}%' for v in range(4)) + '.')
print()
for by_key in (False, True):
    for est_name, est_of in ESTIMATORS.items():
        tag = 'difficulty strata crossed with the answer key' if by_key else 'difficulty strata'
        print(f'## {tag}; estimators: {est_name} (half 1: {len(est_of(1))}, half 2: {len(est_of(2))} systems)')
        print()
        print('| Fold (difficulty from half) | Strata per subject | Cells | Cells all kept | Smallest cell | Kept minus matched difficulty, '
              'points | Expected overlap with the kept set | Level gap, other half clean, points: all / a(all) < 40% / a(all) >= 40% | '
              'Expected key shares A/B/C/D, % |')
        print('|---|---:|---:|---:|---:|---:|---:|---|---|')
        for h in (1, 2):
            for m in (1,) + STRATA:
                r = matching(h, est_of(h), m, by_key)
                label = 'subject only' if m == 1 else str(m)
                print(f'| {h} | {label} | {r["cells"]} | {r["all_kept"]} | {r["smallest"]} | {r["gap"]:+.2f} | {r["overlap"]:.3f} | '
                      f'{r["level"][0]:+.2f} / {r["level"][1]:+.2f} / {r["level"][2]:+.2f} | '
                      + ' / '.join(f'{100 * r["key"][v]:.1f}' for v in range(4)) + ' |')
        print()
print('Kept minus matched difficulty: mean difficulty (share right among the estimators) of the kept questions minus that of a matched '
      'subset; negative means the kept set is harder. Level gap: mean over the other half\'s clean-pool systems of the expected score on '
      'a matched subset minus the score on the kept questions; positive means the matched subset is easier for them.')
