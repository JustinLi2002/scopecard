"""Are MMLU-Pro's filter models penalised on the questions they filtered? (experiments/MMLUPRO_FILTER_DESIGN.md)

Reads the archives fetched by mmlupro_fetch.py in memory; nothing is extracted or executed.

Usage (from experiments/):  python -E -P scripts/mmlupro_filter_test.py results/mmlupro_filter/raw results/mmlupro_filter
"""
import hashlib
import itertools
import json
import pathlib
import sys
import zipfile

import numpy as np

RAW, OUT = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
FILTER = ['Llama-2-7b-hf_5shots', 'Llama-2-13b-hf_5shots', 'Mistral-7B-v0.1_5shots', 'gemma-7b_5shots', 'Yi-6B_5shots',
          'Yi-6b-Chat_5shots']
SIBLINGS = ['Llama-2-70b-hf_5shots', 'Mistral-7B-v0.2-hf_5shots', 'Mistral-7B-Instruct-v0.1_5shots',
            'Mistral-7B-Instruct-v0.2_5shots', 'Yi-34B_5shots']
WINDOW, MIN_PER_SOURCE, COVERAGE, BOOT, MAX_SUBSETS, PANELS = 0.10, 100, 0.95, 1000, 200000, 2000
MISALIGNED = 0.05   # design, Section 9, correction 1 of 2026-11-16: share of recorded gold answers that differ from the table
RECORDS, SUMMARY, ACCEPT = [], [], []


def rng_for(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big'))


def rec(analysis, subject, quantity, value):
    if isinstance(value, (float, np.floating)):
        value = f'{float(value):.6g}'
    RECORDS.append((analysis, subject, quantity, str(value)))


def say(line=''):
    print(line, flush=True)
    SUMMARY.append(line)


def accept(ok, text):
    line = ('PASS  ' if ok else ('INFO  ' if ok is None else 'FAIL  ')) + text
    ACCEPT.append(line)
    print(line, flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------------------------------------------
questions = [json.loads(line) for line in (RAW / 'mmlu_pro_test_questions.jsonl').open(encoding='utf-8')]
qid = [q['question_id'] for q in questions]
pos = {q: k for k, q in enumerate(qid)}
is_m = np.array([q['src'].startswith('ori_mmlu') for q in questions])
category = np.array([q['category'] for q in questions])
gold = [str(q['answer']).strip() for q in questions]
accept(len(qid) == 12032 and len(pos) == 12032, f'question table: {len(qid)} questions; MMLU-origin {int(is_m.sum())}, other sources {int((~is_m).sum())}')

names, rows, dropped, misaligned, gold_changed = [], [], [], [], {}
for path in sorted((RAW / 'eval_results').glob('*.zip')):
    name = path.name[len('model_outputs_'):]
    for suffix in ('.json.zip', '.zip'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    y = np.full(len(qid), np.nan)
    differs = 0
    with zipfile.ZipFile(path) as zf:
        members = [i.filename for i in zf.infolist() if i.filename.endswith('.json') and '__MACOSX' not in i.filename
                   and not i.filename.endswith('summary.json')]
        for member in members:
            with zf.open(member) as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                continue
            for r in data:
                if not isinstance(r, dict) or r.get('question_id') not in pos:
                    continue
                k = pos[r['question_id']]
                if not np.isnan(y[k]):
                    continue                                   # first occurrence of a question is kept
                answer = str(r.get('answer')).strip()
                differs += answer != gold[k]
                pred = r.get('pred')
                y[k] = float(pred is not None and str(pred).strip() == answer)
    coverage = float(np.mean(~np.isnan(y)))
    if coverage < COVERAGE:
        dropped.append(f'{name} ({100 * coverage:.1f}%)')
        continue
    matched = int(np.sum(~np.isnan(y)))
    if differs > MISALIGNED * matched:
        misaligned.append(f'{name} ({differs} of {matched} recorded gold answers differ from the table)')
        continue
    names.append(name)
    rows.append(y)
    gold_changed[name] = differs
Y = np.array(rows)
accept(None, f'systems kept: {len(names)}; dropped for coverage below 95%: {", ".join(dropped) or "none"}')
accept(None, f'excluded because more than {100 * MISALIGNED:.0f}% of the recorded gold answers differ from the question table, so the '
       f'question ids do not correspond to the current questions (design, Section 9, correction 1): {"; ".join(misaligned) or "none"}')
accept(all(f in names for f in FILTER), f'filter models present: {sum(f in names for f in FILTER)} of {len(FILTER)}')
accept(all(s in names for s in SIBLINGS), f'siblings present: {sum(s in names for s in SIBLINGS)} of {len(SIBLINGS)}')
accept(None, 'records whose gold answer differs from the current table (older dataset versions): '
       + ', '.join(f'{n} {v}' for n, v in gold_changed.items() if v) or 'none')
overall = {n: float(np.nanmean(Y[k])) for k, n in enumerate(names)}
accept(None, 'overall accuracy of gpt4o(2024-05-13)_5shots: '
       f'{100 * overall.get("gpt4o(2024-05-13)_5shots", float("nan")):.1f}% (the paper reports 72.6% for GPT-4o)')
role = np.array(['filter' if n in FILTER else ('sibling' if n in SIBLINGS else 'other') for n in names])
is_filter = role == 'filter'

cats = sorted(set(category))
eligible = [c for c in cats if np.sum(is_m & (category == c)) >= MIN_PER_SOURCE and np.sum(~is_m & (category == c)) >= MIN_PER_SOURCE]
accept(len(eligible) >= 3, f'eligible categories (at least {MIN_PER_SOURCE} questions from each source): {", ".join(eligible)}; '
       f'MMLU-origin {sum(int(np.sum(is_m & (category == c))) for c in eligible)}, other '
       f'{sum(int(np.sum(~is_m & (category == c))) for c in eligible)} questions')


# ---------------------------------------------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------------------------------------------
def standardised(Yw, weights=None, use=None):
    """a(M), a(O) per system over the categories in `use`, standardised to a common category mix."""
    use = use or eligible
    a_m, a_o = np.zeros(Yw.shape[0]), np.zeros(Yw.shape[0])
    total = sum(np.sum(category == c) for c in use)
    for c in use:
        share = np.sum(category == c) / total
        for source, acc in ((is_m, a_m), (~is_m, a_o)):
            cell = source & (category == c)
            y = Yw[:, cell]
            w = np.ones(cell.sum()) if weights is None else weights[cell]
            present = ~np.isnan(y)
            acc += share * (np.where(present, y, 0.0) @ w) / (present @ w)
    return a_m, a_o


def pooled(Yw):
    return np.nanmean(Yw[:, is_m], axis=1), np.nanmean(Yw[:, ~is_m], axis=1)


def control_sets(a_o, focal, allowed):
    """Indices of the systems in `allowed` within WINDOW of the focal system's a(O); at least the three nearest."""
    cand = np.array([j for j in allowed if j != focal])
    near = cand[np.abs(a_o[cand] - a_o[focal]) <= WINDOW]
    return near if len(near) >= 3 else cand[np.argsort(np.abs(a_o[cand] - a_o[focal]))[:3]]


def penalty(a_m, a_o, focal, controls):
    return (a_m[controls].mean() - a_m[focal]) - (a_o[controls].mean() - a_o[focal])


def subset_p(values, size, observed, key):
    """Share of size-`size` subsets of `values` whose mean is at least `observed`."""
    values = np.asarray(values)
    count = 1
    for k in range(1, size + 1):
        count = count * (len(values) - k + 1) // k
    if len(values) < size:
        return float('nan'), 0, 'pool smaller than the filter set'
    if count <= MAX_SUBSETS:
        means = np.array([values[list(c)].mean() for c in itertools.combinations(range(len(values)), size)])
        return float(np.mean(means >= observed)), count, 'exact'
    r = rng_for(key)
    means = np.array([values[r.choice(len(values), size=size, replace=False)].mean() for _ in range(MAX_SUBSETS)])
    return float(np.mean(means >= observed)), MAX_SUBSETS, 'random subsets'


def analysis(tag, title, a_m, a_o, exclude_siblings=False, bootstrap=None):
    idx_f = [j for j in range(len(names)) if role[j] == 'filter']
    allowed = [j for j in range(len(names)) if role[j] == 'other' or (role[j] == 'sibling' and not exclude_siblings)]
    lo, hi = a_o[idx_f].min() - WINDOW, a_o[idx_f].max() + WINDOW
    pool = [j for j in allowed if lo <= a_o[j] <= hi]
    controls = {m: control_sets(a_o, m, allowed) for m in idx_f}
    p_m = np.array([penalty(a_m, a_o, m, controls[m]) for m in idx_f])
    placebo_controls = {q: control_sets(a_o, q, allowed) for q in pool}
    placebo = np.array([penalty(a_m, a_o, q, placebo_controls[q]) for q in pool])
    p_value, n_subsets, how = subset_p(placebo, len(idx_f), p_m.mean(), f'mmlupro/{tag}/subsets')
    say(f'### {title}')
    say(f'- Filter models: a(O) from {100 * a_o[idx_f].min():.1f}% to {100 * a_o[idx_f].max():.1f}%; controls per filter model: '
        + ', '.join(str(len(controls[m])) for m in idx_f) + f'; placebo pool: {len(pool)} non-filter systems.')
    text = f'- **Penalty P = {100 * p_m.mean():+.2f} points**'
    if bootstrap is not None:
        reps = np.array([[penalty(bm, bo, m, controls[m]) for m in idx_f] for bm, bo in bootstrap])
        b_lo, b_hi = np.percentile(reps.mean(axis=1), [2.5, 97.5])
        text += f' (question bootstrap 95% interval {100 * b_lo:+.2f} to {100 * b_hi:+.2f})'
        rec(tag, 'all', 'P_lo', b_lo)
        rec(tag, 'all', 'P_hi', b_hi)
    say(text + f'; placebo penalties of the pool: mean {100 * placebo.mean():+.2f}, SD {100 * placebo.std(ddof=1):.2f}, range '
        f'{100 * placebo.min():+.2f} to {100 * placebo.max():+.2f}; **reference tail share: {round(p_value * n_subsets)} of {n_subsets} '
        f'subsets of the pool ({p_value:.4f}; {how})**.')
    per_model = []
    for k, m in enumerate(idx_f):
        cell = f'{names[m]} {100 * p_m[k]:+.2f}'
        if bootstrap is not None:
            lo_k, hi_k = np.percentile(reps[:, k], [2.5, 97.5])
            cell += f' [{100 * lo_k:+.2f}, {100 * hi_k:+.2f}]'
        per_model.append(cell)
        rec(tag, names[m], 'penalty', p_m[k])
    say('- By filter model: ' + '; '.join(per_model) + '.')
    sib = [(names[q], placebo[k]) for k, q in enumerate(pool) if role[q] == 'sibling']
    if sib:
        say('- Siblings (placebo penalties): ' + '; '.join(f'{n} {100 * v:+.2f}' for n, v in sib) + '.')
    rank = int(np.sum(placebo >= p_m.mean()))
    say(f'- Pool systems with a placebo penalty at least as large as P: {rank} of {len(pool)}.')
    rec(tag, 'all', 'P', p_m.mean())
    rec(tag, 'all', 'placebo_p', p_value)
    rec(tag, 'all', 'pool_size', len(pool))
    rec(tag, 'all', 'placebo_mean', placebo.mean())
    rec(tag, 'all', 'placebo_sd', placebo.std(ddof=1))
    for k, q in enumerate(pool):
        rec(tag, names[q], 'placebo_penalty', placebo[k])
    result = {'P': float(p_m.mean()), 'p': p_value, 'pool': pool, 'placebo': placebo, 'idx_f': idx_f}
    if bootstrap is not None:
        result['ci'] = (float(b_lo), float(b_hi))
    say()
    return result


OUT.mkdir(parents=True, exist_ok=True)
a_m, a_o = standardised(Y)
say('# MMLU-Pro filter study: results')
say()
pin = (RAW / 'PINNED_COMMIT.txt').read_text(encoding='utf-8').strip()
say(f'Design: `experiments/MMLUPRO_FILTER_DESIGN.md` (fixed 2026-11-15). Outputs from TIGER-AI-Lab/MMLU-Pro at commit `{pin[:10]}`; '
    f'{len(names)} systems kept. M = MMLU-origin questions (filtered by the eight models), O = questions from other sources. '
    f'Accuracies are standardised over {len(eligible)} categories that have at least {MIN_PER_SOURCE} questions from each source. '
    f'1 point = 0.01.')
say()
say(f'Corrections of 2026-11-16 (design, Section 9). (1) Excluded from every analysis because the question ids do not correspond to '
    f'the current questions: {"; ".join(misaligned) or "none"}. (2) The reference tail share is the share of subsets of the placebo '
    f'pool, of the size of the filter set, whose mean is at least the observed value. It says where that value lies among those '
    f'subsets. It is not a calibrated p-value: the filter models are not exchangeable with the pool systems. The readings below are '
    f'the labels that the rule fixed in the design gives.')
say()
low = int(np.sum(a_o[is_filter] < 0.15))
gate = low <= is_filter.sum() / 2
accept(gate, f'non-degeneracy gate: {low} of {int(is_filter.sum())} filter models have a(O) below 15% (more than half would make '
       f'the primary analysis uninformative)')
(OUT / 'acceptance.txt').write_text('\n'.join(ACCEPT) + '\n', encoding='utf-8')
with (OUT / 'systems.tsv').open('w', encoding='utf-8', newline='\n') as fh:
    fh.write('system\trole\tcoverage\ta_M\ta_O\toverall\n')
    for k in np.argsort(-a_o):
        fh.write(f'{names[k]}\t{role[k]}\t{np.mean(~np.isnan(Y[k])):.4f}\t{a_m[k]:.6f}\t{a_o[k]:.6f}\t{overall[names[k]]:.6f}\n')
if not gate:
    say('**The non-degeneracy gate failed: ability cannot be matched on the other-source questions. The primary analysis is '
        'uninformative and is not reported; see systems.tsv for the descriptive table.**')
    (OUT / 'summary.md').write_text('\n'.join(SUMMARY) + '\n', encoding='utf-8')
    raise SystemExit(0)

# question bootstrap within category and source, eligible categories
r = rng_for('mmlupro/bootstrap')
boot = []
for _ in range(BOOT):
    w = np.zeros(len(qid))
    for c in eligible:
        for source in (is_m, ~is_m):
            cell = np.flatnonzero(source & (category == c))
            w[cell] = r.multinomial(len(cell), np.full(len(cell), 1.0 / len(cell)))
    boot.append(standardised(Y, weights=w))

say('## Primary analysis')
say()
main = analysis('primary', 'Standardised over eligible categories, siblings among the controls', a_m, a_o, bootstrap=boot)
ci = main['ci']
passes = main['p'] < 0.05 and ci[0] > 0
reading = 'confirmed' if passes and main['P'] >= 0.01 else ('detectable but small' if passes else 'not detected')
say(f'**Reading fixed in the design: {reading}.**')
say()
rec('primary', 'all', 'reading', reading)

say('## Secondary analyses')
say()
analysis('no_siblings', '1. Siblings removed from the controls and from the placebo pool', a_m, a_o, exclude_siblings=True, bootstrap=boot)

# 2. regression version
idx_n = np.array([j for j in range(len(names)) if role[j] != 'filter'])
design = lambda x: np.column_stack([np.ones_like(x), x, x ** 2])
coef = np.linalg.lstsq(design(a_o[idx_n]), a_m[idx_n], rcond=None)[0]
resid_f = a_m[main['idx_f']] - design(a_o[main['idx_f']]) @ coef
loo = []
for q in main['pool']:
    others = idx_n[idx_n != q]
    c = np.linalg.lstsq(design(a_o[others]), a_m[others], rcond=None)[0]
    loo.append(a_m[q] - design(a_o[[q]]) @ c)
loo = np.array(loo).ravel()
p_reg, n_sub, how = subset_p(-loo, len(resid_f), float(-resid_f.mean()), 'mmlupro/regression/subsets')
say('### 2. Regression version')
say(f'- a(M) regressed on a(O) and its square over the {len(idx_n)} non-filter systems. Mean shortfall of the filter models '
    f'below the fitted curve: {100 * -resid_f.mean():+.2f} points (by model: '
    + ', '.join(f'{names[m]} {100 * -v:+.2f}' for m, v in zip(main['idx_f'], resid_f))
    + f'); leave-one-out shortfalls of the pool: mean {100 * -loo.mean():+.2f}, SD {100 * loo.std(ddof=1):.2f}; reference tail share: '
    f'{round(p_reg * n_sub)} of {n_sub} subsets of the pool ({p_reg:.4f}; {how}).')
say()
rec('regression', 'all', 'shortfall', float(-resid_f.mean()))
rec('regression', 'all', 'p', p_reg)
rec('regression', 'all', 'n_systems_in_fit', len(idx_n))
rec('data', 'all', 'excluded_misaligned', '; '.join(misaligned) or 'none')

# 3. dependence footprint
elig_mask = np.isin(category, eligible)


def mean_phi(members, source):
    cols = elig_mask & source
    block = Y[np.ix_(members, np.flatnonzero(cols))]
    block = block[:, ~np.isnan(block).any(axis=0)]
    c = np.corrcoef(block)
    return float(np.nanmean(c[np.triu_indices(len(members), 1)]))


def footprint(members):
    return mean_phi(members, is_m) - mean_phi(members, ~is_m)


obs = footprint(main['idx_f'])
r = rng_for('mmlupro/footprint')
pool = np.array(main['pool'])
ref = np.array([footprint(list(r.choice(pool, size=len(main['idx_f']), replace=False))) for _ in range(PANELS)]) \
    if len(pool) >= len(main['idx_f']) else np.array([np.nan])
say('### 3. Dependence footprint')
say(f'- Mean pairwise phi among the filter models: on M {mean_phi(main["idx_f"], is_m):.3f}, on O {mean_phi(main["idx_f"], ~is_m):.3f}; '
    f'difference {obs:+.3f}. The same difference for {PANELS} random sets of {len(main["idx_f"])} pool systems: mean '
    f'{np.nanmean(ref):+.3f}, central 95% {np.nanpercentile(ref, 2.5):+.3f} to {np.nanpercentile(ref, 97.5):+.3f}; share at or below '
    f'the filter models\' value: {float(np.mean(ref <= obs)):.4f}.')
say()
rec('footprint', 'all', 'difference', obs)
rec('footprint', 'all', 'reference_share_at_or_below', float(np.mean(ref <= obs)))

# 4. all categories, no standardisation
p_m_all, p_o_all = pooled(Y)
analysis('all_categories', '4. All categories, no standardisation', p_m_all, p_o_all)

say('## Systems in the ability range (standardised accuracies, in %)')
say()
say('| System | Role | a(O) | a(M) | a(M) minus a(O) |')
say('|---|---|---:|---:|---:|')
show = sorted(set(main['idx_f']) | set(main['pool']), key=lambda j: a_o[j])
for j in show:
    say(f'| {names[j]} | {role[j]} | {100 * a_o[j]:.1f} | {100 * a_m[j]:.1f} | {100 * (a_m[j] - a_o[j]):+.1f} |')
say()
with (OUT / 'records.tsv').open('w', encoding='utf-8', newline='\n') as fh:
    fh.write('analysis\tsubject\tquantity\tvalue\n')
    for row in RECORDS:
        fh.write('\t'.join(row) + '\n')
(OUT / 'summary.md').write_text('\n'.join(SUMMARY) + '\n', encoding='utf-8')
print(f'wrote {len(RECORDS)} records')
