"""Own-panel penalty of a majority filter as a function of panel size, on original MMLU
(experiments/MMLU_RECONSTRUCTION_DESIGN.md, Section 10; method of PANEL_FILTER_DESIGN.md, Section 12).

Exploratory extension. Reads the correctness matrix written by mmlu_reconstruction_test.py. No download, no training.

Usage (from experiments/):  python -E -P -X utf8 scripts/mmlu_panel_size_curve.py results/mmlu_reconstruction
"""
import hashlib
import pathlib
import sys

import numpy as np

OUT = pathlib.Path(sys.argv[1])
N_PANELS, MAX_MEMBERS, N_OUTSIDE, BOOT = 100, 16, 10, 100
RECORDS, SUMMARY = [], []
QUANTITIES = ['T_self', 'T_total', 'T_family', 'placebo', 'flip_excess', 'T_self_common']
CONTAINS = [('tinyllama', ('tinyllama',)), ('openllama', ('open_llama',)), ('llama', ('llama-2', 'llama2', 'meta-llama-3')),
            ('stablelm', ('stablelm',)), ('mistral', ('mistral', 'mixtral')), ('gemma', ('gemma',)), ('qwen', ('qwen',)),
            ('falcon', ('falcon',)), ('deepseek', ('deepseek',)), ('redpajama', ('redpajama',)), ('cerebras', ('cerebras',)),
            ('starcoder', ('starcoder',)), ('command-r', ('command-r',)), ('yi', ('yi-6b', 'yi-34b')), ('phi', ()),
            ('mpt', ('mpt-',)), ('opt', ('__opt-',)), ('pythia', ('pythia', 'dolly'))]
STARTS = {'llama': ('huggyllama__',), 'yi': ('01-ai__',), 'phi': ('microsoft__phi',)}
LINEAGE = {'llama2-7b': 'llama', 'llama2-13b': 'llama', 'mistral-7b': 'mistral', 'gemma-7b': 'gemma', 'yi-6b': 'yi'}


def rng_for(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big'))


def rec(analysis, subject, quantity, value):
    if isinstance(value, (float, np.floating)):
        value = f'{float(value):.6g}'
    RECORDS.append((analysis, subject, quantity, str(value)))


def say(line=''):
    print(line, flush=True)
    SUMMARY.append(line)


def family(name, kind, tag):
    if kind == 'lineage':
        return LINEAGE[tag]
    low = name.lower()
    for label, keys in CONTAINS:
        if any(k in low for k in keys) or any(low.startswith(k) for k in STARTS.get(label, ())):
            return label
    return 'own:' + name


def make_plan(key, fam, sizes):
    """Every random choice of the study, made without looking at any outcome."""
    n = len(fam)
    plan = {}
    for p in sizes:
        panels = []
        for j in range(N_PANELS):
            r = rng_for(f'mmlu_panel_size/{key}/{p}/{j}')
            panel = r.choice(n, size=p, replace=False)
            inside = np.zeros(n, dtype=bool)
            inside[panel] = True
            chosen = panel if p <= MAX_MEMBERS else r.choice(panel, size=MAX_MEMBERS, replace=False)
            members = []
            for m in chosen:
                same = fam == fam[m]
                siblings = np.flatnonzero(inside & same & (np.arange(n) != m))
                pool = np.flatnonzero(~inside & ~same)
                others = np.flatnonzero(inside & ~same)
                if len(pool) < 1 or len(others) < 1:
                    members.append(None)
                    continue
                g = len(siblings)
                both = len(pool) >= g + 1 and len(others) >= g + 1
                incoming = r.choice(pool, size=g + 1 if both else 1, replace=False)
                outgoing = r.choice(others, size=g + 1 if both else 1, replace=False)
                candidates = np.setdiff1d(np.flatnonzero(~inside), incoming)
                outside = r.choice(candidates, size=min(N_OUTSIDE, len(candidates)), replace=False)
                members.append((int(m), siblings, incoming, outgoing, both, outside))
            panels.append((panel, members))
        plan[p] = panels
    return plan


def mean_on(row, questions):
    return float(row[questions].mean()) if len(questions) else float('nan')


def evaluate(Y, plan, strict, levels=False):
    def subset(counts, p):
        return np.flatnonzero(2 * counts < p) if strict else np.flatnonzero(2 * counts <= p)

    out = {}
    for p, panels in plan.items():
        d_self, d_total, d_family, d_common, placebo = [], [], [], [], []
        flips_m = valid_m = flips_0 = valid_0 = 0
        sizes, level, skipped, no_total, with_siblings, evaluated = [], [], 0, 0, 0, 0
        for panel, members in panels:
            counts = Y[panel].sum(axis=0)
            h_std = subset(counts, p)
            if levels:
                sizes.append(len(h_std))
                level.append(float(Y[:, h_std].mean()) if len(h_std) else float('nan'))
            for item in members:
                if item is None:
                    skipped += 1
                    continue
                m, siblings, incoming, outgoing, both, outside = item
                evaluated += 1
                h_self = subset(counts - Y[m] + Y[incoming[0]], p)
                s_self = mean_on(Y[m], h_self)
                s_ref = mean_on(Y[m], subset(counts - Y[outgoing[0]] + Y[incoming[0]], p))
                d_self.append(s_self - s_ref)
                if both:
                    if len(siblings):
                        with_siblings += 1
                        gain = Y[incoming].sum(axis=0)
                        s_tot = mean_on(Y[m], subset(counts - Y[m] - Y[siblings].sum(axis=0) + gain, p))
                        s_tref = mean_on(Y[m], subset(counts - Y[outgoing].sum(axis=0) + gain, p))
                    else:
                        s_tot, s_tref = s_self, s_ref
                    d_total.append(s_tot - s_tref)
                    d_family.append((s_tot - s_tref) - (s_self - s_ref))
                    d_common.append(s_self - s_ref)             # own participation on the comparisons that have the total
                else:
                    no_total += 1
                if len(h_std) and len(h_self) and len(outside) >= 2:
                    block = Y[outside]
                    q_std, q_new = block[:, h_std].mean(axis=1), block[:, h_self].mean(axis=1)
                    placebo.append(float(np.mean(q_new - q_std)))
                    a0, a1 = mean_on(Y[m], h_std) - q_std, s_self - q_new
                    ok = a0 != 0
                    valid_m += int(ok.sum())
                    flips_m += int(np.sum(ok & (a0 * a1 < 0)))
                    iu = np.triu_indices(len(outside), 1)
                    b0, b1 = (q_std[:, None] - q_std[None, :])[iu], (q_new[:, None] - q_new[None, :])[iu]
                    ok = b0 != 0
                    valid_0 += int(ok.sum())
                    flips_0 += int(np.sum(ok & (b0 * b1 < 0)))
        share_m = flips_m / valid_m if valid_m else float('nan')
        share_0 = flips_0 / valid_0 if valid_0 else float('nan')
        out[p] = {
            'T_self': float(np.nanmean(d_self)) if d_self else float('nan'),
            'T_total': float(np.nanmean(d_total)) if d_total else float('nan'),
            'T_family': float(np.nanmean(d_family)) if d_family else float('nan'),
            'T_self_common': float(np.nanmean(d_common)) if d_common else float('nan'),
            'placebo': float(np.nanmean(placebo)) if placebo else float('nan'),
            'flip_member': share_m, 'flip_baseline': share_0, 'flip_excess': share_m - share_0,
            'evaluated': evaluated, 'skipped': skipped, 'no_total': no_total, 'with_siblings': with_siblings,
        }
        if levels:
            out[p]['subset_size'] = float(np.mean(sizes))
            out[p]['subset_size_min'] = float(np.min(sizes))
            out[p]['level'] = float(np.nanmean(level))
    return out


def reading(estimate, lo, hi, unit):
    if lo > 0 or hi < 0:
        return 'material' if abs(estimate) >= unit else 'small'
    return 'not detected'


def run(key, label, rows, sizes, strict, with_family):
    Y = Y_ALL[rows]
    fam = FAM[rows]
    plan = make_plan(key, fam, sizes)
    est = evaluate(Y, plan, strict, levels=True)
    r = rng_for(f'mmlu_panel_size/{key}/bootstrap')
    reps = {p: {q: [] for q in QUANTITIES} for p in sizes}
    for b in range(BOOT):
        idx = np.concatenate([r.choice(c, size=len(c)) for c in COLS])
        e = evaluate(Y[:, idx], plan, strict)
        for p in sizes:
            for q in QUANTITIES:
                reps[p][q].append(e[p][q])
        if (b + 1) % 20 == 0:
            print(f'   {key}: {b + 1} of {BOOT} bootstrap replicates', flush=True)
    say(f'## {label}: {len(rows)} systems, {len(set(fam))} model lines (name rules)')
    say()
    if with_family:
        say('| Builders | Questions kept (mean; minimum) | Mean score of all systems on them | Own participation, all evaluated '
            'members, points | Own participation, members with the same-line comparison, points | With same-line builders too, same '
            'members, points | Same-line part, same members, points | Placebo, points | Sign changes: member pairs / baseline | '
            'Excess, percentage points | Reading on size: own / total |')
        say('|---:|---|---:|---|---|---|---|---|---|---:|---|')
    else:
        say('| Builders | Questions kept (mean; minimum) | Mean score of all systems on them | Own participation, points | Placebo, '
            'points | Sign changes: member pairs / baseline | Excess, percentage points | Reading: own |')
        say('|---:|---|---:|---|---|---|---:|---|')
    for p in sizes:
        e = est[p]
        ci = {q: tuple(np.nanpercentile(reps[p][q], [2.5, 97.5])) for q in QUANTITIES}

        def cell(q):
            return f'{100 * e[q]:+.2f} [{100 * ci[q][0]:+.2f}, {100 * ci[q][1]:+.2f}]'

        own, total = reading(e['T_self'], *ci['T_self'], 0.01), reading(e['T_total'], *ci['T_total'], 0.01)
        head = f'| {p} | {e["subset_size"]:.0f}; {e["subset_size_min"]:.0f} | {100 * e["level"]:.1f}% | {cell("T_self")} | '
        tail = (f'{cell("placebo")} | {100 * e["flip_member"]:.2f}% / {100 * e["flip_baseline"]:.2f}% | '
                f'{100 * e["flip_excess"]:+.2f} | ')
        if with_family:
            say(head + f'{cell("T_self_common")} | {cell("T_total")} | {cell("T_family")} | ' + tail + f'{own} / {total} |')
        else:
            say(head + tail + f'{own} |')
        for q in QUANTITIES:
            rec(key, f'p={p}', q, e[q])
            rec(key, f'p={p}', q + '_lo', ci[q][0])
            rec(key, f'p={p}', q + '_hi', ci[q][1])
        for q in ('flip_member', 'flip_baseline', 'evaluated', 'skipped', 'no_total', 'with_siblings', 'subset_size',
                  'subset_size_min', 'level'):
            rec(key, f'p={p}', q, e[q])
        rec(key, f'p={p}', 'reading_own', own)
        rec(key, f'p={p}', 'reading_total', total)
        rec(key, f'p={p}', 'T_self_times_p', e['T_self'] * p)
    say()
    say('- Members evaluated, of which with a same-line builder in the panel, and members without the sibling comparison: '
        + '; '.join(f'p = {p}: {est[p]["evaluated"]}, {est[p]["with_siblings"]}, {est[p]["no_total"]}' for p in sizes) + '.')
    say('- Own participation times the number of builders (constant if the penalty is proportional to 1/p): '
        + '; '.join(f'p = {p}: {100 * est[p]["T_self"] * p:.1f}' for p in sizes) + '.')
    say()


data = np.load(OUT / 'correct_matrix.npz', allow_pickle=False)
Y_ALL = data['Y'].astype(np.int64)
NAMES, SETS, TAGS = [str(x) for x in data['systems']], [str(x) for x in data['sets']], [str(x) for x in data['tags']]
SUBJECT = data['subject']
COLS = [np.flatnonzero(SUBJECT == s) for s in sorted(set(SUBJECT.tolist()))]
FAM = np.array([family(n, k, t) for n, k, t in zip(NAMES, SETS, TAGS)])
A = np.array([j for j in range(len(NAMES)) if SETS[j] in ('filter', 'clean')])
B = np.arange(len(NAMES))
with (OUT / 'families_panel_size.tsv').open('w', encoding='utf-8', newline='\n') as fh:
    fh.write('system\tset\ttag\tfamily\tpopulation_A\n')
    for j in np.argsort(FAM, kind='stable'):
        fh.write(f'{NAMES[j]}\t{SETS[j]}\t{TAGS[j]}\t{FAM[j]}\t{int(j in set(A.tolist()))}\n')
say('# Own-panel penalty by panel size on original MMLU: results')
say()
say(f'Specification: `experiments/MMLU_RECONSTRUCTION_DESIGN.md`, Section 10 (fixed 2026-11-15; exploratory). {Y_ALL.shape[1]} questions. '
    f'{N_PANELS} random panels per size; every member evaluated up to {MAX_MEMBERS} per panel; {BOOT} question-bootstrap replicates '
    f'within subject; 95% percentile intervals in brackets. 1 point = 0.01 of accuracy. Only point estimates of the sign changes '
    f'are to be read.')
say()
say('Corrections of 2026-11-16 (design, Section 12; no estimate changed). (1) Own participation is given twice: over every evaluated '
    'member, and over the members that have the same-line comparison, which are the ones the next two columns use; only on those does '
    'the total equal own participation plus the same-line part. (2) The same outcomes vote and score here, so own participation cannot '
    'be negative in any comparison; the intervals and the placebo say nothing about its sign, only about its size. (3) "Same-line" '
    'means the same model line read from names.')
say()
run('A', 'Population A, MMLU-Pro\'s rule (kept if at most half of the panel is right)', A, [4, 8, 16, 32, 64], False, True)
run('A_strict', 'Population A, sensitivity: kept if fewer than half of the panel is right', A, [4, 8, 16, 32, 64], True, True)
run('B', 'Population B (all systems; lineage mostly unknown), MMLU-Pro\'s rule', B, [4, 8, 16, 32, 64, 96], False, False)
with (OUT / 'records_panel_size.tsv').open('w', encoding='utf-8', newline='\n') as fh:
    fh.write('analysis\tsubject\tquantity\tvalue\n')
    for row in RECORDS:
        fh.write('\t'.join(row) + '\n')
(OUT / 'summary_panel_size.md').write_text('\n'.join(SUMMARY) + '\n', encoding='utf-8')
print(f'wrote {len(RECORDS)} records')
