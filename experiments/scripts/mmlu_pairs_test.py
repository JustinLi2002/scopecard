"""Order of systems on all questions and on the kept questions of original MMLU
(experiments/MMLU_RECONSTRUCTION_DESIGN.md, Section 11; decision language, intervals and reference of SUBSET_C_DESIGN.md,
Sections 5 to 7).

Exploratory extension. Reads the correctness matrix written by mmlu_reconstruction_test.py. No download.

Usage (from experiments/):  python -E -P -X utf8 scripts/mmlu_pairs_test.py results/mmlu_reconstruction
A second argument sets the number of bootstrap replicates and random subsets; it is for tests on synthetic data only.
"""
import hashlib
import itertools
import pathlib
import sys

import numpy as np
from scipy import stats

OUT = pathlib.Path(sys.argv[1])
B = K = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
WINDOW, MIN_CONTROLS, MIN_KEPT, MAX_SUBSETS, CHUNK, LISTED = 0.03, 5, 10, 200000, 1500, 25
RECORDS, SUMMARY = [], []
LINE = {'llama2-7b': 'Llama-2-7b', 'llama2-13b': 'Llama-2-13b', 'mistral-7b': 'Mistral-7B', 'gemma-7b': 'gemma-7b', 'yi-6b': 'Yi-6B'}
LABELS = ['resolved_O', 'resolved_V', 'resolved_U', 'reversal', 'newly_resolved', 'lost', 'opposite_parts']


def rng_for(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big'))


def rec(analysis, subject, quantity, value):
    if isinstance(value, (float, np.floating)):
        value = f'{float(value):.6g}'
    RECORDS.append((analysis, subject, quantity, str(value)))


def say(line=''):
    print(line, flush=True)
    SUMMARY.append(line)


def control_set(acc_all, focal, allowed):
    cand = np.array([j for j in allowed if j != focal])
    dist = np.abs(acc_all[cand] - acc_all[focal])
    near = cand[dist <= WINDOW]
    return near if len(near) >= MIN_CONTROLS else cand[np.argsort(dist, kind='stable')[:MIN_CONTROLS]]


def subset_p(values, size, observed, key):
    values = np.asarray(values, dtype=float)
    if len(values) < size:
        return float('nan'), 0, 'pool smaller than the filter set'
    count = 1
    for k in range(1, size + 1):
        count = count * (len(values) - k + 1) // k
    if count <= MAX_SUBSETS:
        means = np.array([values[list(c)].mean() for c in itertools.combinations(range(len(values)), size)])
        return float(np.mean(means >= observed - 1e-12)), count, 'exact'
    gen = rng_for(key)
    means = np.array([values[gen.choice(len(values), size=size, replace=False)].mean() for _ in range(MAX_SUBSETS)])
    return float(np.mean(means >= observed - 1e-12)), MAX_SUBSETS, 'random subsets'


def line_of(j):
    if sets[j] == 'lineage':
        return LINE[tags[j]]
    low = names[j].lower()
    for key, label in (('llama-2-7b', 'Llama-2-7b'), ('llama-2-13b', 'Llama-2-13b'), ('mistral-7b', 'Mistral-7B'),
                       ('gemma-7b', 'gemma-7b'), ('yi-6b', 'Yi-6B')):
        if key in low:
            return label
    return names[j]


# ---------------------------------------------------------------------------------------------------------------
# Data and scores
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
    raise SystemExit('a used subject has no removed question: the removed score is undefined (design, Section 11)')
omega = n_kept[used] / n_kept[used].sum()
n_used_all, n_used_kept = int(n_all[used].sum()), int(n_kept[used].sum())

s_v = sum(w * Y[:, cols[s]].mean(axis=1) for w, s in zip(omega, used))
s_o = sum(w * Y[:, cols[s][kept[cols[s]]]].mean(axis=1) for w, s in zip(omega, used))
s_u = sum(w * Y[:, cols[s][~kept[cols[s]]]].mean(axis=1) for w, s in zip(omega, used))
p_v, p_o, p_u = Y.mean(axis=1), Y[:, kept].mean(axis=1), Y[:, ~kept].mean(axis=1)

r = rng_for('mmlu_pairs/bootstrap')
num_v, num_o, num_u = np.zeros((B, N)), np.zeros((B, N)), np.zeros((B, N))
den, den_u = np.zeros(B), np.zeros(B)
pl_all, pl_kept, pl_den = np.zeros((B, N)), np.zeros((B, N)), np.zeros(B)
empty_removed = 0
for s in range(S):
    c = cols[s]
    Ws = r.multinomial(len(c), np.full(len(c), 1.0 / len(c)), size=B).astype(np.float64)
    k_s = kept[c].astype(np.float64)
    all_s = Ws @ Y[:, c].T
    kept_den = Ws @ k_s
    kept_num = (Ws * k_s) @ Y[:, c].T
    pl_all += all_s
    pl_kept += kept_num
    pl_den += kept_den
    if s in used:
        rem_den = len(c) - kept_den
        ok = rem_den > 0
        empty_removed += int(np.sum(~ok))
        num_v += kept_den[:, None] * all_s / len(c)
        num_o += kept_num
        den += kept_den
        num_u += np.where(ok, kept_den, 0.0)[:, None] * (all_s - kept_num) / np.where(ok, rem_den, 1.0)[:, None]
        den_u += np.where(ok, kept_den, 0.0)
b_v, b_o, b_u = (num_v / den[:, None]).T, (num_o / den[:, None]).T, (num_u / den_u[:, None]).T
bp_v, bp_o, bp_u = (pl_all / Q).T, (pl_kept / pl_den[:, None]).T, ((pl_all - pl_kept) / (Q - pl_den)[:, None]).T
STD = ({'O': s_o, 'U': s_u, 'V': s_v}, {'O': b_o, 'U': b_u, 'V': b_v})
PLAIN = ({'O': p_o, 'U': p_u, 'V': p_v}, {'O': bp_o, 'U': bp_u, 'V': bp_v})

idx_f = np.flatnonzero(sets == 'filter')
clean = np.flatnonzero(sets == 'clean')
clean_ns = np.flatnonzero((sets == 'clean') & (tags != 'sibling'))
lineage = np.flatnonzero(sets == 'lineage')
known = np.flatnonzero(sets != 'random')
POP = {'A': np.flatnonzero((sets == 'filter') | (sets == 'clean')), 'K': known, 'B': np.arange(N)}
POP_TITLE = {'A': 'Population A: the filter models and the clean pool',
             'K': 'Population K: the systems of known provenance (A and the lineage pool)',
             'B': 'Population B: all systems (description only)'}


# ---------------------------------------------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------------------------------------------
def pair_statistics(rows, est, boot):
    """Pairs of the systems in rows, oriented so that the gap on the kept questions is not negative."""
    ia, ib = np.triu_indices(len(rows), 1)
    ia, ib = rows[ia], rows[ib]
    first = est['O'][ia] >= est['O'][ib]
    a, b = np.where(first, ia, ib), np.where(first, ib, ia)
    out = {'a': a, 'b': b}
    for name in 'OUV':
        out[name] = est[name][a] - est[name][b]
        for part in ('lo_', 'hi_', 'se_'):
            out[part + name] = np.empty(len(a))
    max_ov, max_ou = np.zeros(B), np.zeros(B)
    for start in range(0, len(a), CHUNK):
        sl = slice(start, start + CHUNK)
        t = {}
        for name in 'OUV':
            d = boot[name][a[sl]] - boot[name][b[sl]]
            out['lo_' + name][sl], out['hi_' + name][sl] = np.percentile(d, [2.5, 97.5], axis=1)
            se = d.std(axis=1, ddof=1)
            out['se_' + name][sl] = se
            with np.errstate(divide='ignore', invalid='ignore'):
                t[name] = np.where(se[:, None] > 0, np.abs(d - out[name][sl][:, None]) / se[:, None], 0.0).max(axis=0)
        max_ov = np.maximum(max_ov, np.maximum(t['O'], t['V']))
        max_ou = np.maximum(max_ou, np.maximum(t['O'], t['U']))
    out['q_ov'], out['q_ou'] = float(np.percentile(max_ov, 95)), float(np.percentile(max_ou, 95))
    out['tied_O'] = out['O'] == 0
    return out


def labels(p, simultaneous):
    """Decision language of the design, Section 11, for oriented pairs."""
    def resolved(name, q):
        if simultaneous:
            lo = np.where(p['se_' + name] > 0, p[name] - q * p['se_' + name], -np.inf)
            hi = np.where(p['se_' + name] > 0, p[name] + q * p['se_' + name], np.inf)
        else:
            lo, hi = p['lo_' + name], p['hi_' + name]
        return (lo > 0) | (hi < 0)

    res_o, res_v = resolved('O', p['q_ov']), resolved('V', p['q_ov'])
    res_o2, res_u = resolved('O', p['q_ou']), resolved('U', p['q_ou'])
    return {'resolved_O': res_o, 'resolved_V': res_v, 'resolved_U': res_u,
            'reversal': res_o & res_v & (p['O'] * p['V'] < 0),
            'newly_resolved': ~res_o & res_v,
            'lost': res_o & ~res_v,
            'opposite_parts': res_o2 & res_u & (p['O'] * p['U'] < 0)}


def order_stats(sub, whole, comp, iu):
    d_sub = (sub[:, None] - sub[None, :])[iu]
    d_all = (whole[:, None] - whole[None, :])[iu]
    d_comp = (comp[:, None] - comp[None, :])[iu]
    s_sub, s_all = np.sign(d_sub), np.sign(d_all)
    denom = np.sqrt(float(np.sum(s_sub != 0)) * float(np.sum(s_all != 0)))
    tau = float(np.sum(s_sub * s_all) / denom) if denom > 0 else float('nan')
    return tau, int(np.sum(d_sub * d_all < 0)), int(np.sum(d_sub * d_comp < 0))


def describe_pair(p, i, cross=False):
    def cell(name):
        return f'{100 * p[name][i]:+.2f} [{100 * p["lo_" + name][i]:+.2f}, {100 * p["hi_" + name][i]:+.2f}]'
    a, b = p['a'][i], p['b'][i]
    return (f'{names[a]} ({sets[a]}) minus {names[b]} ({sets[b]}): kept {cell("O")}, all {cell("V")}, removed {cell("U")}')


def against(case, slug, label, observed, ref, digits):
    lo, hi = np.percentile(ref, [2.5, 97.5])
    pct = float(np.mean(ref <= observed))
    inside = bool(lo <= observed <= hi)
    for q, v in ((slug, observed), (slug + '_reference_lo', lo), (slug + '_reference_hi', hi), (slug + '_percentile', pct),
                 (slug + '_inside', int(inside))):
        rec(case, 'reference', q, v)
    return (f'{label} {observed:.{digits}f} against a reference of {lo:.{digits}f} to {hi:.{digits}f} '
            f'({"inside" if inside else "outside"}; percentile {100 * pct:.1f})'), inside, lo, hi


def order_analysis(case, title, rows, scores, ref):
    est, boot = scores
    p = pair_statistics(rows, est, boot)
    lab_u, lab_s = labels(p, False), labels(p, True)
    n_pairs, not_tied = len(p['a']), ~p['tied_O']
    iu = np.triu_indices(len(rows), 1)
    tau, opp_v, opp_u = order_stats(est['O'][rows], est['V'][rows], est['U'][rows], iu)
    say(f'### {title}')
    say(f'- {len(rows)} systems, {n_pairs} pairs, {int(p["tied_O"].sum())} tied on the kept questions.')
    se = {name: 100 * float(np.median(p['se_' + name])) for name in 'OUV'}
    swing_v = 100 * float(np.median(p['q_ov'] * (p['se_O'] + p['se_V'])))
    swing_u = 100 * float(np.median(p['q_ou'] * (p['se_O'] + p['se_U'])))
    say(f'- Precision: the standard error of a gap is {se["V"]:.2f} points on all questions, {se["O"]:.2f} on the kept and '
        f'{se["U"]:.2f} on the removed ones (medians). Simultaneous critical values: {p["q_ov"]:.2f} for kept and all, '
        f'{p["q_ou"]:.2f} for kept and removed. A reversal can be established only if the gap changes by about {swing_v:.1f} '
        f'points between all and kept questions; opposite parts need about {swing_u:.1f} points between removed and kept.')
    say(f'- Point estimates: Kendall tau_b between the scores on kept and on all questions {tau:.4f}; {opp_v} of '
        f'{int(not_tied.sum())} pairs have opposite signs on kept and on all questions; {opp_u} have opposite signs on kept and on '
        f'removed questions.')
    for q, v in (('n_systems', len(rows)), ('n_pairs', n_pairs), ('n_tied_on_kept', int(p['tied_O'].sum())), ('tau_b_kept_all', tau),
                 ('opposite_sign_kept_all', opp_v), ('opposite_sign_kept_removed', opp_u), ('q_kept_all', p['q_ov']),
                 ('q_kept_removed', p['q_ou']), ('median_se_all', se['V'] / 100), ('median_se_kept', se['O'] / 100),
                 ('median_se_removed', se['U'] / 100), ('median_change_needed_reversal', swing_v / 100),
                 ('median_change_needed_opposite_parts', swing_u / 100)):
        rec(case, 'all', q, v)
    for tag, la in (('unadjusted', lab_u), ('simultaneous', lab_s)):
        n = {k: int(v.sum()) for k, v in la.items()}
        say(f'- {tag.capitalize()} 95% intervals: resolved on kept {n["resolved_O"]}, on all {n["resolved_V"]}, on removed '
            f'{n["resolved_U"]}; **reversals {n["reversal"]}**; newly resolved {n["newly_resolved"]}; lost {n["lost"]}; '
            f'**opposite parts {n["opposite_parts"]}**.')
        for k, v in n.items():
            rec(case, 'all', f'{k}_{tag}', v)
    inside_v = None
    if ref is not None:
        sub, comp, whole, ref_name = ref
        values = np.array([order_stats(sub[rows, k], whole[rows], comp[rows, k], iu) for k in range(sub.shape[1])])
        text_tau, _, _, _ = against(case, 'tau_b', 'tau_b', tau, values[:, 0], 4)
        text_v, inside_v, lo_v, hi_v = against(case, 'opposite_kept_all', 'pairs with opposite signs on the subset and on all questions',
                                               opp_v, values[:, 1], 0)
        text_u, inside_u, lo_u, hi_u = against(case, 'opposite_kept_removed', 'pairs with opposite signs on the subset and on its complement',
                                               opp_u, values[:, 2], 0)
        say(f'- Random subsets of the same size ({ref_name}; central 95% of {sub.shape[1]} subsets): {text_tau}; {text_v}; {text_u}.')
        n_rev = int(lab_s['reversal'].sum())
        if n_rev == 0:
            if inside_v:
                outcome = ('No reversal is established under simultaneous intervals, and the sign disagreement is within what '
                           'same-size random subsets give.')
            else:
                word = 'more' if opp_v > hi_v else 'fewer'
                outcome = (f'No reversal is established under simultaneous intervals; {word} pairs change sign than same-size random '
                           f'subsets give ({opp_v} against {lo_v:.0f} to {hi_v:.0f}).')
        else:
            word = 'within' if inside_v else ('above' if opp_v > hi_v else 'below')
            outcome = (f'{n_rev} reversal{"s are" if n_rev > 1 else " is"} established under simultaneous intervals; the number of '
                       f'pairs that change sign is {word} what same-size random subsets give ({opp_v} against {lo_v:.0f} to {hi_v:.0f}).')
        say(f'- **Outcome in the wording fixed in the design:** {outcome}')
        rec(case, 'all', 'outcome', outcome)
    return p, lab_u, lab_s


def list_pairs(p, flags, title, key):
    chosen = np.flatnonzero(flags)
    if not len(chosen):
        return
    order = chosen[np.argsort(-np.abs(p[key][chosen] - p['O'][chosen]))]
    say(f'- {title} ({len(chosen)}' + (f'; the {LISTED} with the largest change of gap' if len(chosen) > LISTED else '') + '):')
    for i in order[:LISTED]:
        say('  - ' + describe_pair(p, i))


# ---------------------------------------------------------------------------------------------------------------
# Reference subsets
# ---------------------------------------------------------------------------------------------------------------
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
REF_STD = (ref_sub, ref_comp, s_v, 'the same number of questions per subject as the kept set')

# ---------------------------------------------------------------------------------------------------------------
# Report: order and the two parts
# ---------------------------------------------------------------------------------------------------------------
say('# Order of systems on all questions and on the kept questions of original MMLU: results')
say()
say(f'Specification: `experiments/MMLU_RECONSTRUCTION_DESIGN.md`, Section 11 (written and approved 2026-11-15, before any of it was '
    f'computed; exploratory). {N} systems; {len(used)} subjects with {n_used_all} original questions, of which {n_used_kept} kept and '
    f'{n_used_all - n_used_kept} removed. Scores are standardised to the kept questions\' subject mix. A pair\'s gap is the difference of '
    f'the two scores, oriented so that it is not negative on the kept questions. {B} bootstrap replicates over questions within '
    f'subject; 95% percentile intervals in brackets; simultaneous intervals are estimate ± q × SD over all pairs of a population. '
    f'1 point = 0.01 of accuracy.')
say()
say('Labels. Resolved: the interval of the gap excludes zero. Reversal: resolved on kept and on all questions, with opposite signs. '
    'Newly resolved: not resolved on kept, resolved on all. Lost: resolved on kept, not on all. Opposite parts: resolved on kept and '
    'on removed questions, with opposite signs; this is weaker than a reversal.')
say()
say('Corrections of 2026-11-16 (design, Section 12; wording only, no estimate changed). In part 2, the reference tail share is the share '
    'of subsets of the placebo pool, of the size of the filter set, whose mean is at least the observed value. It says where that '
    'value lies among those subsets. It is not a calibrated p-value: the filter models are not exchangeable with the pool systems. '
    'The reading there is the label that the rule fixed in the design gives.')
say()
scores_file = OUT / 'systems.tsv'
if scores_file.exists():
    table = {}
    with scores_file.open(encoding='utf-8') as fh:
        head = next(fh).rstrip('\n').split('\t')
        for text_line in fh:
            cells = text_line.rstrip('\n').split('\t')
            table[cells[0]] = (float(cells[head.index('a_all')]), float(cells[head.index('a_kept')]))
    gap = max(max(abs(table[n][0] - s_v[j]), abs(table[n][1] - s_o[j])) for j, n in enumerate(names) if n in table)
    say(f'Check: a(all) and a(kept) of {sum(n in table for n in names)} systems agree with `systems.tsv` of Section 9 within '
        f'{gap:.1e}. Bootstrap replicates in which a subject had no removed question: {empty_removed}.')
    say()
    rec('check', 'all', 'max_difference_from_systems_tsv', gap)
rec('check', 'all', 'replicates_with_empty_removed_subject', empty_removed)

say('## 1. Order (question 1) and the two parts (question 2)')
say()
PAIRS = {}
for key in ('A', 'K', 'B'):
    PAIRS[key] = order_analysis(key, POP_TITLE[key], POP[key], STD, REF_STD if key in ('A', 'B') else None)
    p, lab_u, lab_s = PAIRS[key]
    if key != 'B':
        list_pairs(p, lab_s['reversal'], 'Reversals under simultaneous intervals', 'V')
        list_pairs(p, lab_u['reversal'], 'Reversals under unadjusted intervals', 'V')
        list_pairs(p, lab_s['opposite_parts'], 'Opposite parts under simultaneous intervals', 'U')
    say()

r = rng_for('mmlu_pairs/reference/plain')
rows_a = POP['A']
plain_sub, plain_comp = np.zeros((N, K)), np.zeros((N, K))
total = Y[rows_a].sum(axis=1)
n_plain = int(kept.sum())
for k in range(K):
    chosen = r.choice(Q, size=n_plain, replace=False)
    corr = Y[np.ix_(rows_a, chosen)].sum(axis=1)
    plain_sub[rows_a, k] = corr / n_plain
    plain_comp[rows_a, k] = (total - corr) / (Q - n_plain)
order_analysis('A_plain', f'Sensitivity for population A: plain accuracy without standardisation (all {Q} questions of the {S} subjects '
               f'against the {n_plain} kept ones)', rows_a, PLAIN, (plain_sub, plain_comp, p_v, 'drawn uniformly'))
say()


# ---------------------------------------------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------------------------------------------
lo_r, hi_r = s_v[idx_f].min() - WINDOW, s_v[idx_f].max() + WINDOW


def places_lost(C):
    """For every system of known provenance: opponents of C ahead of it on all and on kept questions, and the difference."""
    L, Lb, ahead = {}, {}, {}
    for x in known:
        opp = C[C != x]
        av = float(np.sum(s_v[opp] > s_v[x]) + 0.5 * np.sum(s_v[opp] == s_v[x]))
        ao = float(np.sum(s_o[opp] > s_o[x]) + 0.5 * np.sum(s_o[opp] == s_o[x]))
        bv = (b_v[opp] > b_v[x]).sum(axis=0) + 0.5 * (b_v[opp] == b_v[x]).sum(axis=0)
        bo = (b_o[opp] > b_o[x]).sum(axis=0) + 0.5 * (b_o[opp] == b_o[x]).sum(axis=0)
        L[x], Lb[x], ahead[x] = ao - av, bo - bv, (av, ao, len(opp))
    return L, Lb, ahead


def excess(x, C, L, Lb):
    ctrl = control_set(s_v, x, C)
    return L[x] - float(np.mean([L[k] for k in ctrl])), Lb[x] - np.mean([Lb[k] for k in ctrl], axis=0), ctrl


def builders(tag, title, C):
    L, Lb, ahead = places_lost(C)
    pool = [j for j in C if lo_r <= s_v[j] <= hi_r]
    res = {m: excess(m, C, L, Lb) for m in idx_f}
    n_m = np.array([res[m][0] for m in idx_f])
    reps = np.array([res[m][1] for m in idx_f])
    placebo = np.array([excess(q, C, L, Lb)[0] for q in pool])
    p_value, n_subsets, how = subset_p(placebo, len(idx_f), n_m.mean(), f'mmlu_pairs/{tag}/subsets')
    b_lo, b_hi = np.percentile(reps.mean(axis=0), [2.5, 97.5])
    passes = p_value < 0.05 and b_lo > 0
    reading = 'confirmed' if passes and n_m.mean() >= 1 else ('detectable but small' if passes else 'not detected')
    say(f'### {title}')
    say(f'- Opponents: {len(C)} clean-pool systems. Controls per filter model: ' + ', '.join(str(len(res[m][2])) for m in idx_f)
        + f'; placebo pool: {len(pool)} systems.')
    say(f'- **Excess places lost N = {n_m.mean():+.2f}** (question bootstrap 95% interval {b_lo:+.2f} to {b_hi:+.2f}); placebo values '
        f'of the pool: mean {placebo.mean():+.2f}, SD {placebo.std(ddof=1):.2f}, range {placebo.min():+.2f} to {placebo.max():+.2f}; '
        f'**reference tail share: {round(p_value * n_subsets)} of {n_subsets} subsets of the pool ({p_value:.5f}; {how})**. '
        f'**Reading fixed in the design: {reading}.**')
    say()
    say('| Filter model | Place on all questions | Place on kept questions | Places lost | Mean places lost by its controls | Excess N_m | 95% interval |')
    say('|---|---:|---:|---:|---:|---:|---|')
    for k, m in enumerate(idx_f):
        av, ao, n_opp = ahead[m]
        lo_k, hi_k = np.percentile(reps[k], [2.5, 97.5])
        say(f'| {names[m]} | {av + 1:g} of {n_opp + 1} | {ao + 1:g} of {n_opp + 1} | {L[m]:+g} | {L[m] - n_m[k]:+.2f} | {n_m[k]:+.2f} | '
            f'{lo_k:+.2f} to {hi_k:+.2f} |')
        for q, v in (('place_all', av + 1), ('place_kept', ao + 1), ('places_lost', L[m]), ('excess', n_m[k]), ('excess_lo', lo_k),
                     ('excess_hi', hi_k), ('n_controls', len(res[m][2]))):
            rec(tag, names[m], q, v)
    say()
    say(f'- Pool systems with a placebo value at least as large as N: {int(np.sum(placebo >= n_m.mean()))} of {len(pool)}; filter '
        f'models whose excess is above every placebo value: {int(np.sum(n_m > placebo.max()))} of {len(idx_f)}.')
    say()
    for q, v in (('N', n_m.mean()), ('N_lo', b_lo), ('N_hi', b_hi), ('placebo_p', p_value), ('pool_size', len(pool)),
                 ('placebo_mean', placebo.mean()), ('placebo_sd', placebo.std(ddof=1)), ('placebo_max', placebo.max()),
                 ('reading', reading)):
        rec(tag, 'all', q, v)
    for k, q in enumerate(pool):
        rec(tag, names[q], 'placebo_excess', placebo[k])
    return {'L': L, 'Lb': Lb, 'ahead': ahead, 'pool': pool, 'placebo': placebo, 'excess': {m: res[m][0] for m in idx_f}}


say('## 2. Builders (question 3)')
say()
say('Places lost by a system: the number of clean-pool systems ahead of it on the kept questions minus the number ahead of it on all '
    'questions (a tie counts one half). The excess N_m of a filter model is its places lost minus the mean places lost by its '
    'controls, the clean-pool systems with a(all) within 3 points (Section 4). Section 9 already measured the penalty in points; this '
    'part translates it into places.')
say()
main = builders('places', 'Clean pool as opponents and controls, siblings included', clean)
no_sib = builders('places_no_siblings', 'Sensitivity: siblings removed from the opponents and the controls', clean_ns)

ref = no_sib['placebo']
say('### By line: the lineage pool')
say(f'- Excess places lost of each lineage system, with opponents and controls from the clean pool without siblings. Reference: the '
    f'{len(ref)} placebo values of that pool (mean {ref.mean():+.2f}, SD {ref.std(ddof=1):.2f}, maximum {ref.max():+.2f}). Only lineage '
    f'systems inside the filter models\' range of a(all), widened by 0.03, are counted. A line is a model line read from names. '
    f'Fine-tunes of one base are not independent of each other, so the Mann–Whitney p-value has no independent-sample reading; it only '
    f'describes how far the two sets of numbers are apart.')
say()
say('| Line | Systems in range (outside) | Mean excess | Median | Share above zero | Share above the pool maximum | Mann–Whitney p (descriptive) |')
say('|---|---:|---:|---:|---:|---:|---:|')
lineage_excess = {j: excess(j, clean_ns, no_sib['L'], no_sib['Lb'])[0] for j in lineage}
for label in sorted(set(line_of(j) for j in lineage)) + ['all lines']:
    members = [j for j in lineage if label == 'all lines' or line_of(j) == label]
    inside = [j for j in members if lo_r <= s_v[j] <= hi_r]
    if not inside:
        say(f'| {label} | 0 ({len(members)}) | | | | | |')
        continue
    values = np.array([lineage_excess[j] for j in inside])
    p_mw = stats.mannwhitneyu(values, ref, alternative='greater').pvalue if len(values) >= 3 else float('nan')
    say(f'| {label} | {len(inside)} ({len(members) - len(inside)}) | {values.mean():+.2f} | {np.median(values):+.2f} | '
        f'{100 * np.mean(values > 0):.0f}% | {100 * np.mean(values > ref.max()):.0f}% | {p_mw:.4g} |')
    for q, v in (('n_in_range', len(inside)), ('n_outside', len(members) - len(inside)), ('mean_excess', values.mean()),
                 ('median_excess', float(np.median(values))), ('share_above_zero', float(np.mean(values > 0))),
                 ('share_above_pool_max', float(np.mean(values > ref.max()))), ('mann_whitney_p', p_mw)):
        rec('places_lineage', label, q, v)
say()


def direction(key, title, builder_rows, groups):
    p, lab_u, lab_s = PAIRS[key]
    a_b, b_b = np.isin(p['a'], builder_rows), np.isin(p['b'], builder_rows)
    a_c, b_c = np.isin(p['a'], clean), np.isin(p['b'], clean)
    cross = (a_b & b_c) | (b_b & a_c)
    sign = np.where(a_b, 1.0, -1.0)
    d_o, d_v, d_u = sign * p['O'], sign * p['V'], sign * p['U']
    who = np.where(a_b, p['a'], p['b'])
    fall, rise = cross & (d_v > 0) & (d_o < 0), cross & (d_v < 0) & (d_o > 0)
    down, up = cross & (d_u > 0) & (d_o < 0), cross & (d_u < 0) & (d_o > 0)
    say(f'### {title}')
    say('- Gaps are the filter-line system minus the clean-pool system. "Falls behind": ahead on all questions, behind on the kept '
        'ones. "Moves ahead": the opposite. For opposite parts, "ahead on removed": ahead on the removed questions and behind on the '
        'kept ones.')
    say()
    say('| Filter-line systems | Cross pairs | Opposite signs, all and kept: falls behind / moves ahead | Reversals, unadjusted: falls / '
        'moves ahead | Reversals, simultaneous: falls / moves ahead | Opposite parts, unadjusted: ahead on removed / ahead on kept | '
        'Opposite parts, simultaneous: ahead on removed / ahead on kept |')
    say('|---|---:|---|---|---|---|---|')
    for label, members in groups:
        sel = cross & np.isin(who, members)
        cells = [f'{int(np.sum(sel & fall))} / {int(np.sum(sel & rise))}',
                 f'{int(np.sum(sel & fall & lab_u["reversal"]))} / {int(np.sum(sel & rise & lab_u["reversal"]))}',
                 f'{int(np.sum(sel & fall & lab_s["reversal"]))} / {int(np.sum(sel & rise & lab_s["reversal"]))}',
                 f'{int(np.sum(sel & down & lab_u["opposite_parts"]))} / {int(np.sum(sel & up & lab_u["opposite_parts"]))}',
                 f'{int(np.sum(sel & down & lab_s["opposite_parts"]))} / {int(np.sum(sel & up & lab_s["opposite_parts"]))}']
        say(f'| {label} ({len(members)}) | {int(sel.sum())} | ' + ' | '.join(cells) + ' |')
        for q, v in (('cross_pairs', int(sel.sum())), ('falls_behind', int(np.sum(sel & fall))), ('moves_ahead', int(np.sum(sel & rise))),
                     ('reversal_unadjusted_falls', int(np.sum(sel & fall & lab_u['reversal']))),
                     ('reversal_unadjusted_moves_ahead', int(np.sum(sel & rise & lab_u['reversal']))),
                     ('reversal_simultaneous_falls', int(np.sum(sel & fall & lab_s['reversal']))),
                     ('reversal_simultaneous_moves_ahead', int(np.sum(sel & rise & lab_s['reversal']))),
                     ('opposite_parts_unadjusted_ahead_on_removed', int(np.sum(sel & down & lab_u['opposite_parts']))),
                     ('opposite_parts_unadjusted_ahead_on_kept', int(np.sum(sel & up & lab_u['opposite_parts']))),
                     ('opposite_parts_simultaneous_ahead_on_removed', int(np.sum(sel & down & lab_s['opposite_parts']))),
                     ('opposite_parts_simultaneous_ahead_on_kept', int(np.sum(sel & up & lab_s['opposite_parts'])))):
            rec(f'direction_{key}', label, q, v)
    both_clean = a_c & b_c
    say()
    say(f'- For comparison, the {int(both_clean.sum())} pairs of two clean-pool systems: {int(np.sum(both_clean & (p["O"] * p["V"] < 0)))} '
        f'with opposite signs on all and kept questions; reversals {int(np.sum(both_clean & lab_u["reversal"]))} unadjusted and '
        f'{int(np.sum(both_clean & lab_s["reversal"]))} simultaneous; opposite parts {int(np.sum(both_clean & lab_u["opposite_parts"]))} '
        f'unadjusted and {int(np.sum(both_clean & lab_s["opposite_parts"]))} simultaneous.')
    say()
    rec(f'direction_{key}', 'clean_pairs', 'pairs', int(both_clean.sum()))
    rec(f'direction_{key}', 'clean_pairs', 'opposite_sign_kept_all', int(np.sum(both_clean & (p['O'] * p['V'] < 0))))
    rec(f'direction_{key}', 'clean_pairs', 'reversal_unadjusted', int(np.sum(both_clean & lab_u['reversal'])))
    rec(f'direction_{key}', 'clean_pairs', 'reversal_simultaneous', int(np.sum(both_clean & lab_s['reversal'])))
    rec(f'direction_{key}', 'clean_pairs', 'opposite_parts_unadjusted', int(np.sum(both_clean & lab_u['opposite_parts'])))
    rec(f'direction_{key}', 'clean_pairs', 'opposite_parts_simultaneous', int(np.sum(both_clean & lab_s['opposite_parts'])))
    return cross, sign, who, fall, rise, down, up


groups_a = [(names[m], [m]) for m in idx_f] + [('all filter models', list(idx_f))]
direction('A', 'Direction among cross pairs, population A (families of population A)', idx_f, groups_a)
builder_rows = np.concatenate([idx_f, lineage])
lines = sorted(set(line_of(j) for j in builder_rows))
groups_k = [(label, [j for j in builder_rows if line_of(j) == label]) for label in lines] + [('all filter-line systems', list(builder_rows))]
cross_k = direction('K', 'Direction among cross pairs, population K, by line (families of population K)', builder_rows, groups_k)

# ---------------------------------------------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------------------------------------------
p, lab_u, lab_s = PAIRS['A']
with (OUT / 'pairs_A.tsv').open('w', encoding='utf-8', newline='\n') as fh:
    stat_cols = [f'{part}{name}' for name in 'OVU' for part in ('', 'lo_', 'hi_', 'se_')]
    head = ['system_a', 'set_a', 'system_b', 'set_b'] + [c.replace('O', 'kept').replace('V', 'all').replace('U', 'removed') for c in stat_cols]
    fh.write('\t'.join(head + [f'{k}_unadjusted' for k in LABELS] + [f'{k}_simultaneous' for k in LABELS]) + '\n')
    for i in range(len(p['a'])):
        row = [names[p['a'][i]], sets[p['a'][i]], names[p['b'][i]], sets[p['b'][i]]] + [f'{p[c][i]:.6f}' for c in stat_cols]
        row += [str(int(lab_u[k][i])) for k in LABELS] + [str(int(lab_s[k][i])) for k in LABELS]
        fh.write('\t'.join(row) + '\n')

p, lab_u, lab_s = PAIRS['K']
cross, sign, who, fall, rise, down, up = cross_k
with (OUT / 'pairs_K_cross.tsv').open('w', encoding='utf-8', newline='\n') as fh:
    fh.write('filter_line_system\tset\tline\tclean_system\tsibling\tgap_all\tgap_kept\tgap_removed\tse_all\tse_kept\tse_removed\t'
             'direction_all_kept\tdirection_removed_kept\treversal_unadjusted\treversal_simultaneous\topposite_parts_unadjusted\t'
             'opposite_parts_simultaneous\n')
    for i in np.flatnonzero(cross):
        other = p['b'][i] if who[i] == p['a'][i] else p['a'][i]
        d1 = 'falls_behind' if fall[i] else ('moves_ahead' if rise[i] else 'same_sign')
        d2 = 'ahead_on_removed' if down[i] else ('ahead_on_kept' if up[i] else 'same_sign')
        fh.write('\t'.join([names[who[i]], sets[who[i]], line_of(who[i]), names[other], str(int(tags[other] == 'sibling')),
                            f'{sign[i] * p["V"][i]:.6f}', f'{sign[i] * p["O"][i]:.6f}', f'{sign[i] * p["U"][i]:.6f}',
                            f'{p["se_V"][i]:.6f}', f'{p["se_O"][i]:.6f}', f'{p["se_U"][i]:.6f}', d1, d2,
                            str(int(lab_u['reversal'][i])), str(int(lab_s['reversal'][i])), str(int(lab_u['opposite_parts'][i])),
                            str(int(lab_s['opposite_parts'][i]))]) + '\n')

with (OUT / 'places.tsv').open('w', encoding='utf-8', newline='\n') as fh:
    fh.write('system\tset\ttag\tline\ta_all\ta_kept\ta_removed\tin_range\tahead_all\tahead_kept\topponents\tplaces_lost\tn_controls\t'
             'excess_places_lost\tplaces_lost_no_siblings\texcess_no_siblings\n')
    for j in sorted(known, key=lambda j: -s_v[j]):
        av, ao, n_opp = main['ahead'][j]
        e1, _, c1 = excess(j, clean, main['L'], main['Lb'])
        e2 = excess(j, clean_ns, no_sib['L'], no_sib['Lb'])[0]
        fh.write('\t'.join([names[j], sets[j], tags[j], line_of(j) if sets[j] != 'clean' else '', f'{s_v[j]:.6f}', f'{s_o[j]:.6f}',
                            f'{s_u[j]:.6f}', str(int(lo_r <= s_v[j] <= hi_r)), f'{av:g}', f'{ao:g}', str(n_opp), f'{main["L"][j]:g}',
                            str(len(c1)), f'{e1:.4f}', f'{no_sib["L"][j]:g}', f'{e2:.4f}']) + '\n')

with (OUT / 'records_pairs.tsv').open('w', encoding='utf-8', newline='\n') as fh:
    fh.write('analysis\tsubject\tquantity\tvalue\n')
    for row in RECORDS:
        fh.write('\t'.join(row) + '\n')
(OUT / 'summary_pairs.md').write_text('\n'.join(SUMMARY) + '\n', encoding='utf-8')
print(f'wrote {len(RECORDS)} records')
