"""Own-panel penalty of a solve-rate filter as a function of panel size (PANEL_FILTER_DESIGN.md, Section 12).

Exploratory extension. Reads the BigCodeBench per-task tables and the family table written by
panel_filter_study.py. No download, no training.

Usage (from experiments/):  python -E -P scripts/panel_size_curve.py results/subsets/raw results/panel_filter
"""
import hashlib
import json
import pathlib
import sys

import numpy as np

RAW, OUT = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
SIZES, N_PANELS, MAX_MEMBERS, N_OUTSIDE, BOOT = [4, 8, 16, 32, 64, 96], 100, 16, 10, 100
RECORDS, SUMMARY = [], []
QUANTITIES = ['T_self', 'T_total', 'T_family', 'placebo', 'flip_excess', 'T_self_common']


def rng_for(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big'))


def rec(analysis, subject, quantity, value):
    if isinstance(value, (float, np.floating)):
        value = f'{float(value):.6g}'
    RECORDS.append((analysis, subject, quantity, str(value)))


def say(line=''):
    print(line, flush=True)
    SUMMARY.append(line)


def load(split):
    tasks = [json.loads(line)['task_id'] for line in (RAW / 'huggingface' / 'bigcodebench_v0.1.4.jsonl').open(encoding='utf-8')]
    rows = [json.loads(line) for line in (RAW / 'huggingface' / f'bigcodebench_perf_{split}.jsonl').open(encoding='utf-8')]
    return [r['Model'] for r in rows], np.array([[int(r[t]) for t in tasks] for r in rows], dtype=np.int64)


def make_plan(split, fam):
    """Every random choice of the study, made without looking at any outcome."""
    n = len(fam)
    plan = {}
    for p in SIZES:
        panels = []
        for j in range(N_PANELS):
            r = rng_for(f'panel_size/{split}/{p}/{j}')
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


def subset(counts, p):
    return np.flatnonzero(2 * counts < p)


def mean_on(row, tasks):
    return float(row[tasks].mean()) if len(tasks) else float('nan')


def evaluate(Y, plan, levels=False):
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
                    q_std, q_new = Y[outside][:, h_std].mean(axis=1), Y[outside][:, h_self].mean(axis=1)
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


def run(split, label):
    names, Y = load(split)
    fam = np.array([FAMILY[name] for name in names])
    plan = make_plan(split, fam)
    est = evaluate(Y, plan, levels=True)
    r = rng_for(f'panel_size/{split}/bootstrap')
    t = Y.shape[1]
    reps = {p: {q: [] for q in QUANTITIES} for p in SIZES}
    for _ in range(BOOT):
        e = evaluate(Y[:, r.integers(0, t, t)], plan)
        for p in SIZES:
            for q in QUANTITIES:
                reps[p][q].append(e[p][q])
    say(f'## {label}: {len(names)} models')
    say()
    say('| Builders | Tasks kept (mean; minimum) | Mean score of all models on them | Own participation, all evaluated members, '
        'points | Own participation, members with the same-line comparison, points | With same-line builders too, same members, '
        'points | Same-line part, same members, points | Placebo, points | Sign changes: member pairs / baseline | Excess, percentage '
        'points (point estimate) | Reading on size: own / total |')
    say('|---:|---|---:|---|---|---|---|---|---|---:|---|')
    for p in SIZES:
        e = est[p]
        ci = {q: tuple(np.nanpercentile(reps[p][q], [2.5, 97.5])) for q in QUANTITIES}

        def cell(q):
            return f'{100 * e[q]:+.2f} [{100 * ci[q][0]:+.2f}, {100 * ci[q][1]:+.2f}]'

        # The reading on ranking is withdrawn (design, Section 13, correction 1): the bootstrap of the sign-change share is biased.
        verdicts = (reading(e['T_self'], *ci['T_self'], 0.01), reading(e['T_total'], *ci['T_total'], 0.01),
                    'withdrawn')
        say(f'| {p} | {e["subset_size"]:.0f}; {e["subset_size_min"]:.0f} | {100 * e["level"]:.1f}% | {cell("T_self")} | '
            f'{cell("T_self_common")} | {cell("T_total")} | {cell("T_family")} | {cell("placebo")} | {100 * e["flip_member"]:.2f}% / '
            f'{100 * e["flip_baseline"]:.2f}% | {100 * e["flip_excess"]:+.2f} | {verdicts[0]} / {verdicts[1]} |')
        for q in QUANTITIES:
            rec(split, f'p={p}', q, e[q])
            rec(split, f'p={p}', q + '_lo', ci[q][0])
            rec(split, f'p={p}', q + '_hi', ci[q][1])
        for q in ('flip_member', 'flip_baseline', 'evaluated', 'skipped', 'no_total', 'with_siblings', 'subset_size',
                  'subset_size_min', 'level'):
            rec(split, f'p={p}', q, e[q])
        for q, v in zip(('own', 'total', 'ranking'), verdicts):
            rec(split, f'p={p}', 'reading_' + q, v)
    say()
    say('- Members evaluated, of which with a same-line builder in the panel, and members without the sibling comparison: '
        + '; '.join(f'p = {p}: {est[p]["evaluated"]}, {est[p]["with_siblings"]}, {est[p]["no_total"]}' for p in SIZES) + '.')
    say('- Own participation times the number of builders (constant if the penalty is proportional to 1/p): '
        + '; '.join(f'p = {p}: {100 * est[p]["T_self"] * p:.1f}' for p in SIZES) + '.')
    say()


FAMILY = {}
with (OUT / 'families.tsv').open(encoding='utf-8') as fh:
    next(fh)
    for line in fh:
        cols = line.rstrip('\n').split('\t')
        FAMILY[cols[0]] = cols[2]
say('# Own-panel penalty by panel size: results')
say()
say(f'Specification: `experiments/PANEL_FILTER_DESIGN.md`, Section 12 (fixed 2026-11-15; exploratory). Rule: tasks with a solve '
    f'rate below 50% over the panel. {N_PANELS} random panels per size; every member evaluated up to {MAX_MEMBERS} per panel; '
    f'{BOOT} task-bootstrap replicates; 95% percentile intervals in brackets. 1 point = 0.01 of score.')
say()
say('Corrections of 2026-11-16 (design, Section 13; no estimate changed). (1) The share of sign changes is given as a point estimate '
    'only and carries no reading: 7 of the 12 point estimates of its excess lie outside their own bootstrap intervals, so that '
    'bootstrap is biased. (2) Own participation is given twice: over every evaluated member, and over the members that have the '
    'same-line comparison, which are the ones the next two columns use; only on those does the total equal own participation plus the '
    'same-line part. (3) The same outcomes vote and score here, so own participation cannot be negative in any comparison; the '
    'intervals and the placebo say nothing about its sign, only about its size. (4) "Same-line" means the same model line read from '
    'names.')
say()
run('complete', 'Complete')
run('instruct', 'Instruct (same tasks; 124 models in common with Complete)')
with (OUT / 'records_panel_size.tsv').open('w', encoding='utf-8', newline='\n') as fh:
    fh.write('split\tsubject\tquantity\tvalue\n')
    for row in RECORDS:
        fh.write('\t'.join(row) + '\n')
(OUT / 'summary_panel_size.md').write_text('\n'.join(SUMMARY) + '\n', encoding='utf-8')
print(f'wrote {len(RECORDS)} records')
