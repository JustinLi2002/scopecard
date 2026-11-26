"""Precision check made before the pairwise design was fixed (experiments/MMLU_RECONSTRUCTION_DESIGN.md, Section 11).

Prints bootstrap standard errors of pairwise gaps, simultaneous critical values, and counts of clean-pool systems near
each filter model on all questions. It prints no pairwise sign, no kept-set score and no count of sign changes.

Usage (from experiments/):  python -E -P -X utf8 scripts/mmlu_pairs_precision_check.py results/mmlu_reconstruction/correct_matrix.npz
"""
import hashlib
import sys

import numpy as np

B, MIN_KEPT = 2000, 10
data = np.load(sys.argv[1], allow_pickle=False)
Y = data['Y'].astype(np.float64)
sets = np.array([str(x) for x in data['sets']])
names = [str(x) for x in data['systems']]
kept = data['kept'].astype(bool)
subject = data['subject']
subjects = sorted(set(subject.tolist()))
cols = [np.flatnonzero(subject == s) for s in subjects]
N, Q = Y.shape
n_kept = np.array([int(kept[c].sum()) for c in cols])
used = [s for s in range(len(subjects)) if n_kept[s] >= MIN_KEPT]
print(f'{N} systems, {Q} questions, {int(kept.sum())} kept; subjects used {len(used)} of {len(subjects)}')

rng = np.random.default_rng(int.from_bytes(hashlib.sha256(b'mmlu_pairs/precision_check').digest()[:8], 'big'))
num_all, num_kept, den = np.zeros((B, N)), np.zeros((B, N)), np.zeros(B)
for s in used:
    c = cols[s]
    Ws = rng.multinomial(len(c), np.full(len(c), 1.0 / len(c)), size=B).astype(np.float64)
    k = kept[c].astype(np.float64)
    all_s = Ws @ Y[:, c].T
    kept_den = Ws @ k
    num_all += kept_den[:, None] * all_s / len(c)
    num_kept += (Ws * k) @ Y[:, c].T
    den += kept_den
b_v, b_o = (num_all / den[:, None]).T, (num_kept / den[:, None]).T     # systems x replicates
omega = n_kept[used] / n_kept[used].sum()
s_v = sum(w * Y[:, cols[s]].mean(axis=1) for w, s in zip(omega, used))
s_o = sum(w * Y[:, cols[s][kept[cols[s]]]].mean(axis=1) for w, s in zip(omega, used))


def check(label, rows):
    ia, ib = np.triu_indices(len(rows), 1)
    a, b = rows[ia], rows[ib]
    se = {}
    max_t = np.zeros(B)
    for start in range(0, len(a), 1500):
        sl = slice(start, start + 1500)
        for name, est, boot in (('V', s_v, b_v), ('O', s_o, b_o)):
            d = boot[a[sl]] - boot[b[sl]]
            sd = d.std(axis=1, ddof=1)
            se.setdefault(name, []).append(sd)
            centre = (est[a[sl]] - est[b[sl]])[:, None]
            with np.errstate(divide='ignore', invalid='ignore'):
                t = np.where(sd[:, None] > 0, np.abs(d - centre) / sd[:, None], 0.0)
            max_t = np.maximum(max_t, t.max(axis=0))
    se_v, se_o = np.concatenate(se['V']), np.concatenate(se['O'])
    q = float(np.percentile(max_t, 95))
    swing = q * (se_v + se_o)

    def qs(x):
        return ' / '.join(f'{100 * v:.2f}' for v in np.percentile(x, [5, 50, 95]))

    print(f'{label}: {len(rows)} systems, {len(a)} pairs; simultaneous q = {q:.3f}')
    print(f'   SE of the gap on all questions, points (5th / median / 95th): {qs(se_v)}')
    print(f'   SE of the gap on kept questions, points: {qs(se_o)}')
    print(f'   margin needed on all questions q*SE: {qs(q * se_v)}; on kept: {qs(q * se_o)}')
    print(f'   smallest change of gap that can be an established reversal: {qs(swing)}')
    print(f'   same with unadjusted 1.96: {qs(1.96 * (se_v + se_o))}')


A = np.flatnonzero((sets == 'filter') | (sets == 'clean'))
AL = np.flatnonzero((sets == 'filter') | (sets == 'clean') | (sets == 'lineage'))
check('Population A (filter + clean)', A)
check('Known provenance (filter + clean + lineage)', AL)
check('All systems', np.arange(N))

clean = np.flatnonzero(sets == 'clean')
print('Clean-pool systems near each filter model on all questions only (standardised a(all)):')
for f in np.flatnonzero(sets == 'filter'):
    diff = 100 * (s_v[clean] - s_v[f])
    print(f'   {names[f]}: below by 0-3 points {int(np.sum((diff < 0) & (diff >= -3)))}, by 3-6 {int(np.sum((diff < -3) & (diff >= -6)))}, '
          f'by 6-9 {int(np.sum((diff < -6) & (diff >= -9)))}; above by 0-3 {int(np.sum((diff > 0) & (diff <= 3)))}, '
          f'by 3-6 {int(np.sum((diff > 3) & (diff <= 6)))}, by 6-9 {int(np.sum((diff > 6) & (diff <= 9)))}')
lin = np.flatnonzero(sets == 'lineage')
below = [int(np.sum((s_v[clean] < s_v[j]) & (s_v[clean] >= s_v[j] - 0.06))) for j in lin]
above = [int(np.sum((s_v[clean] > s_v[j]) & (s_v[clean] <= s_v[j] + 0.06))) for j in lin]
print(f'Lineage systems ({len(lin)}): clean-pool systems within 6 points below, median {np.median(below):.0f} '
      f'(min {min(below)}, max {max(below)}); within 6 points above, median {np.median(above):.0f} (min {min(above)}, max {max(above)})')
