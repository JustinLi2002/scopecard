"""Panel composition study (experiments/PANEL_COMPOSITION_DESIGN.md): concentrated against spread filter panels of the same size and
ability on BigCodeBench; Complete votes, Instruct scores.

Usage (from experiments/):
  python -E -P -X utf8 scripts/panel_composition.py --check   # checks of Section 11 with synthetic scoring vectors; reads no Instruct outcome
  python -E -P -X utf8 scripts/panel_composition.py           # the run
"""
import csv
import hashlib
import json
import pathlib
import sys

import numpy as np

PAPER = pathlib.Path(__file__).resolve().parents[1]
RAW = PAPER / 'results/subsets/raw'
OUT = PAPER / 'results/panel_composition'
MIN_ACC, N_REP, N_CONTROLS, N_BOOT, SIZES = 0.25, 100, 5, 1000, (8, 16)
CHECK = '--check' in sys.argv
RECORDS, SUMMARY = [], []


def rng_for(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big'))


def rec(analysis, group, quantity, value):
    if isinstance(value, (float, np.floating)):
        value = f'{float(value):.6g}'
    RECORDS.append((analysis, group, quantity, str(value)))


def say(text=''):
    print(text, flush=True)
    SUMMARY.append(text)


def load(modes):
    with (RAW / 'manifest.tsv').open(encoding='utf-8') as f:
        manifest = {r['path']: r for r in csv.DictReader(f, delimiter='\t')}
    inputs = ['huggingface/bigcodebench_v0.1.4.jsonl'] + [f'huggingface/bigcodebench_perf_{m}.jsonl' for m in ('complete', 'instruct')]
    for name in inputs:
        if hashlib.sha256((RAW / name).read_bytes()).hexdigest() != manifest[name]['sha256']:
            raise SystemExit(f'input hash differs from the manifest: {name}')
    tasks = [json.loads(s)['task_id'] for s in (RAW / inputs[0]).read_text(encoding='utf-8').splitlines()]
    raw = {}
    for mode in ('complete', 'instruct'):
        rows = [json.loads(s) for s in (RAW / f'huggingface/bigcodebench_perf_{mode}.jsonl').read_text(encoding='utf-8').splitlines()]
        raw[mode] = {r['Model']: r for r in rows}
    names = sorted(raw['complete'].keys() & raw['instruct'].keys())
    if len(names) != 124 or len(tasks) != 1140:
        raise SystemExit('unexpected table sizes')
    mats = {mode: np.array([[int(raw[mode][n][t]) for t in tasks] for n in names], dtype=np.float64) for mode in modes}
    return names, tasks, mats


names, tasks, mats = load(('complete',) if CHECK else ('complete', 'instruct'))
T_ = len(tasks)
lines_of = {}
with (PAPER / 'results/panel_filter/families.tsv').open(encoding='utf-8') as fh:
    for row in csv.DictReader(fh, delimiter='\t'):
        lines_of[row['model']] = row['line']
L = np.array([lines_of[n] for n in names])
acc_c = mats['complete'].mean(axis=1)

# Pools, as in scripts/panel_composition_check.py
gen = rng_for('panel_composition/partition')
pool = {}
for ln in sorted(set(L)):
    members = np.flatnonzero(L == ln)
    order = members[gen.permutation(len(members))]
    k = len(members) // 2
    for j in order[:k]:
        pool[int(j)] = 'E'
    for j in order[k:]:
        pool[int(j)] = 'B'
E = np.array([j for j in range(len(names)) if pool[j] == 'E'])
B_ALL = [j for j in range(len(names)) if pool[j] == 'B']
B_FLOOR = [j for j in B_ALL if acc_c[j] >= MIN_ACC]
if len(E) != 57:
    raise SystemExit('the evaluation pool differs from the design')
e_pos = {int(j): i for i, j in enumerate(E)}
TIE = rng_for('panel_composition/controls').random(len(names))


def focal_lines(p, eligible_b):
    c = p // 2
    return [ln for ln in sorted(set(L), key=lambda x: (-np.sum(L == x), x))
            if sum(1 for j in eligible_b if L[j] == ln) >= c and sum(1 for j in E if L[j] == ln) >= 2]


def make_pair(ln, p, rep, eligible_b):
    g = rng_for(f'panel_composition/{p}/{ln}/{rep}')
    c = p // 2
    b_line = [j for j in eligible_b if L[j] == ln]
    b_other = [j for j in eligible_b if L[j] != ln]
    voters = [int(x) for x in g.choice(b_line, c, replace=False)]
    others, used = [], {ln}
    for j in g.permutation(b_other):
        if L[j] not in used and len(others) < p - c:
            others.append(int(j))
            used.add(L[j])
    if len(others) < p - c:
        return None
    stay = voters[int(g.integers(c))]
    spread, new_lines, gaps = [stay] + others, [], []
    for v in [x for x in voters if x != stay]:
        cand = [j for j in b_other if L[j] not in used]
        if not cand:
            return None
        dist = np.abs(acc_c[cand] - acc_c[v])
        best = [cand[i] for i in np.flatnonzero(dist == dist.min())]
        r = int(g.choice(best))
        spread.append(r)
        used.add(L[r])
        new_lines.append(str(L[r]))
        gaps.append(abs(acc_c[r] - acc_c[v]))
    return {'conc': voters + others, 'spread': spread, 'voters': voters, 'stay': stay, 'others': others, 'new_lines': new_lines,
            'gaps': gaps}


def controls_for(targets, unchanged_e, acc_t, k):
    out = {}
    for w in targets:
        cand = np.array(unchanged_e)
        order = np.lexsort((TIE[cand], np.abs(acc_t[cand] - acc_t[w])))
        out[w] = cand[order[:k]]
    return out


def item_total(Z):
    """Corrected item-total correlation of every task over the evaluation pool; nan where the task is constant over it."""
    ze = Z[E]
    total = ze.sum(axis=1, keepdims=True)
    rest = total - ze
    zc, rc = ze - ze.mean(axis=0), rest - rest.mean(axis=0)
    den = np.sqrt((zc ** 2).sum(axis=0) * (rc ** 2).sum(axis=0))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, (zc * rc).sum(axis=0) / den, np.nan)


def run(Ysrc, Z, eligible_b, k_controls, weights, label):
    """All replicates of all focal lines and sizes; returns per-contrast bootstrap arrays and per-replicate rows."""
    acc_t = Z.mean(axis=1)
    r_item = item_total(Z)
    ZE = Z[E]
    out, rows = {}, []
    for p in SIZES:
        for ln in focal_lines(p, eligible_b):
            t_line, t_new, reps = [], [], []
            for rep in range(N_REP):
                pair = make_pair(ln, p, rep, eligible_b)
                if pair is None:
                    continue
                Hc = 2 * Ysrc[pair['conc']].sum(axis=0) < p
                Hs = 2 * Ysrc[pair['spread']].sum(axis=0) < p
                if not Hc.any() or not Hs.any():
                    raise SystemExit('an empty kept set')
                sc = ((weights * Hc) @ ZE.T) / (weights @ Hc)[:, None]
                ss = ((weights * Hs) @ ZE.T) / (weights @ Hs)[:, None]
                d = ss - sc
                unchanged = set(L) - {ln} - set(pair['new_lines'])
                unchanged_e = [int(j) for j in E if L[j] in unchanged]
                focal_e = [int(j) for j in E if L[j] == ln]
                new_e = [int(j) for j in E if L[j] in set(pair['new_lines'])]
                ctrl = controls_for(focal_e + new_e, unchanged_e, acc_t, k_controls)

                def cost(group):
                    return np.mean([d[:, e_pos[w]] - d[:, [e_pos[int(x)] for x in ctrl[w]]].mean(axis=1) for w in group], axis=0)
                t_line.append(cost(focal_e))
                if new_e:
                    t_new.append(cost(new_e))
                orig = 0
                for name, H in (('conc', Hc), ('spread', Hs)):
                    sel = H & ~np.isnan(r_item)
                    rows.append({'label': label, 'p': p, 'line': ln, 'rep': rep, 'panel': name, 'kept': int(H.sum()),
                                 'difficulty': float(ZE[:, H].mean()),
                                 'discrimination': float(np.mean(r_item[sel])) if sel.any() else float('nan'),
                                 'constant_tasks': int(np.sum(H & np.isnan(r_item))), 'score_sd': float(ZE[:, H].mean(axis=1).std(ddof=1)),
                                 'panel_ability': float(acc_c[pair[name]].mean()), 'jaccard': float((Hc & Hs).sum() / (Hc | Hs).sum()),
                                 'cost_line': float(t_line[-1][orig]), 'cost_new': float(t_new[-1][orig]) if new_e else float('nan'),
                                 'new_lines': ','.join(pair['new_lines']), 'max_gap': float(max(pair['gaps']))})
            out[(p, ln)] = {'line': np.mean(t_line, axis=0), 'new': np.mean(t_new, axis=0) if t_new else None,
                            'reps_line': np.array([x[0] for x in t_line]), 'reps_new': np.array([x[0] for x in t_new]),
                            'n_reps': len(t_line)}
    return out, rows


def weights_for(n_boot):
    w = np.ones((1 + n_boot, T_))
    g = rng_for('panel_composition/bootstrap')
    w[1:] = g.multinomial(T_, np.full(T_, 1.0 / T_), size=n_boot)
    return w


# ---------------------------------------------------------------------------------------------------------------
# Checks (Section 11): synthetic scoring vectors; no Instruct outcome is read
# ---------------------------------------------------------------------------------------------------------------
if CHECK:
    Yc = mats['complete']
    w = weights_for(200)
    ok = True
    res, _ = run(Yc, np.ones_like(Yc), B_FLOOR, N_CONTROLS, w[:21], 'constant')
    worst = max(float(np.abs(v['line']).max()) for v in res.values())
    print(f'constant scoring vectors: largest |estimate| {worst:.2e} (expected 0)')
    ok &= worst < 1e-12
    g = rng_for('panel_composition/check/independent')
    Zr = (g.random(Yc.shape) < Yc.mean(axis=1, keepdims=True)).astype(np.float64)
    res, _ = run(Yc, Zr, B_FLOOR, N_CONTROLS, w, 'independent')
    for (p, ln), v in res.items():
        mc = v['reps_line'].std(ddof=1) / np.sqrt(v['n_reps'])
        sd_b = v['line'][1:].std(ddof=1)
        print(f'independent scoring vectors, p={p} {ln}: estimate {100 * v["line"][0]:+.3f} points, task-bootstrap SD '
              f'{100 * sd_b:.3f} (ratio {v["line"][0] / sd_b:+.2f}); panel-replicate SE {100 * mc:.3f}')
        ok &= abs(v['line'][0]) < 3.5 * sd_b
    Zp = Yc.copy()
    for ln in focal_lines(8, B_FLOOR):
        v_star = max((j for j in B_FLOOR if L[j] == ln), key=lambda j: acc_c[j])
        for j in E:
            if L[j] == ln:
                Zp[j] = Yc[v_star]
    res, _ = run(Yc, Zp, B_FLOOR, N_CONTROLS, w, 'planted')
    for (p, ln), v in res.items():
        print(f'planted (focal E models score with a voter\'s votes), p={p} {ln}: estimate {100 * v["line"][0]:+.3f} points')
        ok &= v['line'][0] > 0
    # Bootstrap weights against explicit resampling, a few replicates
    g = rng_for('panel_composition/check/resampling')
    worst = 0.0
    for rep in range(3):
        pair = make_pair('qwen', 8, rep, B_FLOOR)
        idx = g.integers(0, T_, size=T_)
        counts = np.bincount(idx, minlength=T_).astype(np.float64)
        for panel in (pair['conc'], pair['spread']):
            H = 2 * Yc[panel].sum(axis=0) < 8
            weighted = ((counts * H) @ Zr[E].T) / (counts @ H)
            Hx = 2 * Yc[panel][:, idx].sum(axis=0) < 8
            explicit = Zr[E][:, idx][:, Hx].mean(axis=1)
            worst = max(worst, float(np.abs(weighted - explicit).max()))
    print(f'bootstrap weights against explicit resampling: largest difference {worst:.2e}')
    ok &= worst < 1e-12
    print('ALL CHECKS PASSED' if ok else 'A CHECK FAILED')
    sys.exit(0 if ok else 1)

# ---------------------------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------------------------
OUT.mkdir(parents=True, exist_ok=True)
Yc, Yi = mats['complete'], mats['instruct']
W = weights_for(N_BOOT)
say('# Panel composition: results')
say()
say(f'Specification: `experiments/PANEL_COMPOSITION_DESIGN.md` (fixed 2026-11-17 before any quantity was computed; exploratory). '
    f'BigCodeBench v0.1.4, {len(names)} models, {T_} tasks. Evaluation pool {len(E)} models (never vote); builder pool {len(B_ALL)}, of '
    f'which {len(B_FLOOR)} have a Complete pass rate of at least {MIN_ACC:.0%}. {N_REP} panel pairs per focal line and size; '
    f'{N_BOOT} task-bootstrap replicates. Concentration cost: how much more the focal line\'s non-voting models gain on the spread '
    f'panel\'s kept set than their {N_CONTROLS} nearest models of unchanged lines; positive means that concentration costs the line. '
    f'1 point = 0.01.')
say()

main, rows = run(Yc, Yi, B_FLOOR, N_CONTROLS, W, 'primary')
keys = list(main.keys())
boot = np.array([main[k]['line'][1:] for k in keys])
est = np.array([main[k]['line'][0] for k in keys])
sd = boot.std(axis=1, ddof=1)
q = float(np.percentile(np.max(np.abs(boot - est[:, None]) / sd[:, None], axis=0), 95))


def reading(lo, hi, value):
    if lo > 0 or hi < 0:
        return 'material' if abs(value) >= 0.01 else 'detectable but small'
    return 'not detected'


focal8 = [k for k in keys if k[0] == 8]
pooled = np.mean([main[k]['line'] for k in focal8], axis=0)
lo, hi = np.percentile(pooled[1:], [2.5, 97.5])
say('## 1. Primary: concentration cost at 8 builders, mean over the focal lines')
say()
say(f'**T(8) = {100 * pooled[0]:+.2f} points** (task-bootstrap 95% interval {100 * lo:+.2f} to {100 * hi:+.2f}), over '
    f'{len(focal8)} lines with equal weight. **Reading fixed in the design: {reading(lo, hi, pooled[0])}.**')
rec('primary', 'T8', 'estimate', pooled[0])
rec('primary', 'T8', 'lo', lo)
rec('primary', 'T8', 'hi', hi)
rec('primary', 'T8', 'reading', reading(lo, hi, pooled[0]))
say()
say('## 2. By line')
say()
say(f'Simultaneous intervals over the {len(keys)} contrasts: estimate ± {q:.2f} × bootstrap SD. Cost of spreading: the same quantity '
    f'for the evaluation-pool models of the lines that gain a voter in the spread panel (negative: they lose). Panel-replicate SE: '
    f'the spread over the {N_REP} panel pairs only, not the precision of an estimate (design, Deviation 1).')
say()
say('| Size | Focal line | E models of the line | Replicates | Concentration cost, points | 95% interval | Simultaneous interval | '
    'Reading | Panel-replicate SE | Cost of spreading, points (95% interval) |')
say('|---:|---|---:|---:|---:|---|---|---|---:|---|')
for i, k in enumerate(keys):
    p, ln = k
    v = main[k]
    b_lo, b_hi = np.percentile(v['line'][1:], [2.5, 97.5])
    s_lo, s_hi = est[i] - q * sd[i], est[i] + q * sd[i]
    mc = v['reps_line'].std(ddof=1) / np.sqrt(v['n_reps'])
    new_text = ''
    if v['new'] is not None:
        n_lo, n_hi = np.percentile(v['new'][1:], [2.5, 97.5])
        new_text = f'{100 * v["new"][0]:+.2f} ({100 * n_lo:+.2f} to {100 * n_hi:+.2f})'
        rec(f'line_{p}', ln, 'cost_of_spreading', v['new'][0])
        rec(f'line_{p}', ln, 'cost_of_spreading_lo', n_lo)
        rec(f'line_{p}', ln, 'cost_of_spreading_hi', n_hi)
    n_e = sum(1 for j in E if L[j] == ln)
    say(f'| {p} | {ln} | {n_e} | {v["n_reps"]} | {100 * est[i]:+.2f} | {100 * b_lo:+.2f} to {100 * b_hi:+.2f} | '
        f'{100 * s_lo:+.2f} to {100 * s_hi:+.2f} | {reading(s_lo, s_hi, est[i])} | {100 * mc:.2f} | {new_text} |')
    for qn, val in (('estimate', est[i]), ('lo', b_lo), ('hi', b_hi), ('simultaneous_lo', s_lo), ('simultaneous_hi', s_hi),
                    ('reading', reading(s_lo, s_hi, est[i])), ('panel_replicate_se', mc), ('replicates', v['n_reps'])):
        rec(f'line_{p}', ln, qn, val)
say()

say('## 3. What the two panels keep (means over replicates)')
say()
say('| Size | Focal line | Panel | Panel ability, Complete % | Tasks kept | Jaccard overlap | Difficulty: E models\' mean Instruct pass '
    'rate, % | Discrimination: mean corrected item–total correlation | SD of E models\' scores, points | Largest replacement gap, points |')
say('|---:|---|---|---:|---:|---:|---:|---:|---:|---:|')
for p, ln in keys:
    for name in ('conc', 'spread'):
        sel = [r for r in rows if r['p'] == p and r['line'] == ln and r['panel'] == name]
        say(f'| {p} | {ln} | {"concentrated" if name == "conc" else "spread"} | {100 * np.mean([r["panel_ability"] for r in sel]):.1f} | '
            f'{np.mean([r["kept"] for r in sel]):.0f} | {np.mean([r["jaccard"] for r in sel]):.3f} | '
            f'{100 * np.mean([r["difficulty"] for r in sel]):.1f} | {np.mean([r["discrimination"] for r in sel]):.3f} | '
            f'{100 * np.mean([r["score_sd"] for r in sel]):.1f} | {100 * np.mean([r["max_gap"] for r in sel]):.1f} |')
        for qn in ('panel_ability', 'kept', 'jaccard', 'difficulty', 'discrimination', 'score_sd', 'constant_tasks'):
            rec(f'kept_{p}_{name}', ln, qn, float(np.mean([r[qn] for r in sel])))
say()
say('Distribution over replicates of the concentration cost (point estimates per replicate), points:')
say()
for p, ln in keys:
    x = 100 * main[(p, ln)]['reps_line']
    say(f'- {p} builders, {ln}: median {np.median(x):+.2f}, quartiles {np.percentile(x, 25):+.2f} and {np.percentile(x, 75):+.2f}, '
        f'range {x.min():+.2f} to {x.max():+.2f}, share above zero {100 * np.mean(x > 0):.0f}%.')
say()

say('## 4. Sensitivity (reported, not read)')
say()
say('| Variant | ' + ' | '.join(f'{p}: {ln}' for p, ln in keys) + ' | T(8), the primary focal lines |')
say('|---|' + '---:|' * (len(keys) + 1))
notes = []
for label, Ysrc, Z, eligible, k in (('Instruct votes, Complete scores', Yi, Yc, B_FLOOR, N_CONTROLS),
                                    ('three controls', Yc, Yi, B_FLOOR, 3), ('ten controls', Yc, Yi, B_FLOOR, 10),
                                    ('no 25% floor for panel members', Yc, Yi, B_ALL, N_CONTROLS)):
    res, rows_s = run(Ysrc, Z, eligible, k, W[:1], label)
    cells = []
    for p, ln in keys:
        cells.append(f'{100 * res[(p, ln)]["line"][0]:+.2f}' if (p, ln) in res else 'not focal')
        if (p, ln) in res:
            rec(f'sensitivity/{label}', f'{p}_{ln}', 'estimate', res[(p, ln)]['line'][0])
    f8 = [res[key]['line'][0] for key in focal8 if key in res]
    say(f'| {label} | ' + ' | '.join(cells) + f' | {100 * np.mean(f8):+.2f} |')
    rec(f'sensitivity/{label}', 'T8', 'estimate', float(np.mean(f8)))
    for key in sorted(key for key in res if key not in main):
        notes.append(f'- {label}: {key[1]} is a focal line at {key[0]} builders only in this variant: '
                     f'{100 * res[key]["line"][0]:+.2f} points (not in T(8)).')
        rec(f'sensitivity/{label}', f'{key[0]}_{key[1]}', 'estimate_extra_line', res[key]['line'][0])
    if label.startswith('no 25%'):
        gaps = [r['max_gap'] for r in rows_s if r['panel'] == 'conc']
        diffs = [rs['panel_ability'] - rc['panel_ability'] for rc, rs in zip(rows_s[0::2], rows_s[1::2])]
        notes.append(f'- Without the floor: largest replacement gap up to {100 * max(gaps):.1f} points; panel mean ability, spread '
                     f'minus concentrated, {100 * min(diffs):+.2f} to {100 * max(diffs):+.2f} points.')
say()
for text in notes:
    say(text)
say()

with (OUT / 'replicates.tsv').open('w', encoding='utf-8', newline='\n') as fh:
    head = ['label', 'p', 'line', 'rep', 'panel', 'kept', 'difficulty', 'discrimination', 'constant_tasks', 'score_sd', 'panel_ability',
            'jaccard', 'cost_line', 'cost_new', 'new_lines', 'max_gap']
    fh.write('\t'.join(head) + '\n')
    for r in rows:
        fh.write('\t'.join(f'{r[h]:.6g}' if isinstance(r[h], float) else str(r[h]) for h in head) + '\n')
with (OUT / 'pools.tsv').open('w', encoding='utf-8', newline='\n') as fh:
    fh.write('model\tline\tpool\tcomplete_pass_rate\tinstruct_pass_rate\tpanel_eligible\n')
    for j, n in enumerate(names):
        fh.write(f'{n}\t{L[j]}\t{pool[j]}\t{acc_c[j]:.6f}\t{Yi[j].mean():.6f}\t{int(j in B_FLOOR)}\n')
with (OUT / 'records.tsv').open('w', encoding='utf-8', newline='\n') as fh:
    fh.write('analysis\tgroup\tquantity\tvalue\n')
    for row in RECORDS:
        fh.write('\t'.join(row) + '\n')
(OUT / 'summary.md').write_text('\n'.join(SUMMARY) + '\n', encoding='utf-8')
print(f'wrote {len(RECORDS)} records')
