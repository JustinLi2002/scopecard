"""Feasibility check for the panel-composition study (experiments/PANEL_COMPOSITION_DESIGN.md). Reads no outcome of the study: only each
model's pass rate over all 1,140 tasks on the two splits, the model lines of PANEL_FILTER_DESIGN.md (Section 3), and, for the line
coherence check, the agreement of pass/fail patterns on the voting split. No kept set is built and nothing is scored on one.

Usage (from experiments/): python -E -P -X utf8 scripts/panel_composition_check.py
"""
import csv
import hashlib
import json
import pathlib

import numpy as np

PAPER = pathlib.Path(__file__).resolve().parents[1]
RAW = PAPER / 'results/subsets/raw'
MIN_ACC = 0.25  # panel members: B models with a Complete pass rate of at least 25%


def rng_for(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big'))


def load():
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
    mats = {mode: np.array([[int(raw[mode][n][t]) for t in tasks] for n in names], dtype=np.float64) for mode in raw}
    return names, tasks, mats


names, tasks, mats = load()
line = {}
with (PAPER / 'results/panel_filter/families.tsv').open(encoding='utf-8') as fh:
    for row in csv.DictReader(fh, delimiter='\t'):
        line[row['model']] = row['line']
L = np.array([line[n] for n in names])
acc = {m: mats[m].mean(axis=1) for m in mats}
lines = sorted(set(L), key=lambda x: (-np.sum(L == x), x))
print(f'{len(names)} models in both splits, {len(tasks)} tasks, {len(lines)} lines.')
print()

# Partition: in every line of at least two models, floor(n/2) models drawn at random go to the evaluation pool E, the rest to the
# builder pool B; a line of one model goes to B.
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
print('| Line | Models | B | E | Complete pass rate, B: min / median / max | Complete, E | Instruct, E |')
print('|---|---:|---:|---:|---|---|---|')
for ln in lines:
    b = [j for j in np.flatnonzero(L == ln) if pool[int(j)] == 'B']
    e = [j for j in np.flatnonzero(L == ln) if pool[int(j)] == 'E']

    def rng3(v):
        return ' / '.join(f'{100 * x:.1f}' for x in (np.min(v), np.median(v), np.max(v))) if len(v) else ''
    print(f'| {ln} | {np.sum(L == ln)} | {len(b)} | {len(e)} | {rng3(acc["complete"][b])} | {rng3(acc["complete"][e])} | '
          f'{rng3(acc["instruct"][e])} |')
print()

# Line coherence on the voting split: mean phi coefficient of pass/fail patterns between two models of one line, against pairs of
# models of different lines whose pass rates differ by at most 3 points.
y = mats['complete']
z = (y - y.mean(axis=1, keepdims=True))
sd = z.std(axis=1)
phi = (z @ z.T) / len(tasks) / np.outer(sd, sd)
print('| Line | Within-line pairs within 3 points of each other\'s pass rate | Mean phi within the line | Mean phi, cross-line pairs within 3 points of each other\'s pass rate |')
print('|---|---:|---:|---:|')
for ln in lines:
    idx = np.flatnonzero(L == ln)
    if len(idx) < 2:
        continue
    within = [phi[a, b] for i, a in enumerate(idx) for b in idx[i + 1:] if abs(acc['complete'][a] - acc['complete'][b]) <= 0.03]
    cross = [phi[a, b] for a in idx for b in np.flatnonzero(L != ln) if abs(acc['complete'][a] - acc['complete'][b]) <= 0.03]
    if within:
        print(f'| {ln} | {len(within)} | {np.mean(within):.3f} | {np.mean(cross):.3f} ({len(cross)} pairs) |')
print()

# Feasibility of the swap: for each focal line and panel size, how far the nearest-ability replacement from a line not yet in the
# panel lies from the replaced member (Complete pass rate), over 200 trial panels drawn as in the design.
for p in (8, 16):
    c = p // 2
    for ln in lines:
        b_line = [j for j in np.flatnonzero(L == ln) if pool[int(j)] == 'B' and acc['complete'][j] >= MIN_ACC]
        e_line = [j for j in np.flatnonzero(L == ln) if pool[int(j)] == 'E']
        if len(b_line) < c or len(e_line) < 2:
            continue
        g = rng_for(f'panel_composition/check/{p}/{ln}')
        gaps, fails, mean_diff = [], 0, []
        b_other = [j for j in range(len(names)) if pool[j] == 'B' and L[j] != ln and acc['complete'][j] >= MIN_ACC]
        for _ in range(200):
            voters = list(g.choice(b_line, c, replace=False))
            others, used_lines = [], {ln}
            for j in g.permutation(b_other):
                if L[j] not in used_lines and len(others) < p - c:
                    others.append(int(j))
                    used_lines.add(L[j])
            if len(others) < p - c:
                fails += 1
                continue
            stay = voters[int(g.integers(c))]
            spread, ok = [stay] + others, True
            for v in [x for x in voters if x != stay]:
                cand = [j for j in b_other if L[j] not in used_lines]
                if not cand:
                    ok = False
                    break
                dist = np.abs(acc['complete'][cand] - acc['complete'][v])
                best = [cand[i] for i in np.flatnonzero(dist == dist.min())]
                r = int(g.choice(best))
                gaps.append(abs(acc['complete'][r] - acc['complete'][v]))
                spread.append(r)
                used_lines.add(L[r])
            if not ok:
                fails += 1
                continue
            mean_diff.append(acc['complete'][spread].mean() - acc['complete'][voters + others].mean())
        if gaps:
            print(f'p={p} {ln}: B at least 25% {len(b_line)}, E {len(e_line)}; failed panels {fails} of 200; replacement gap, points: median '
                  f'{100 * np.median(gaps):.1f}, 90th percentile {100 * np.percentile(gaps, 90):.1f}, max {100 * np.max(gaps):.1f}; panel mean '
                  f'ability, spread minus concentrated: median {100 * np.median(mean_diff):+.2f}, range {100 * np.min(mean_diff):+.2f} to '
                  f'{100 * np.max(mean_diff):+.2f}')
        else:
            print(f'p={p} {ln}: B {len(b_line)}, E {len(e_line)}; every trial panel failed')
