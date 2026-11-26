"""Frozen-source / other-prompt scoring experiment. See SEPARATED_SCORING_DESIGN.en.md.

Only writes experiments/results/separated_scoring. No model inference or downloads.
Run from the repository: python -E -P -X utf8 experiments/scripts/separated_scoring.py
"""
from __future__ import annotations

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'experiments'
RAW = PAPER / 'results/subsets/raw'
OUT = PAPER / 'results/separated_scoring'
DESIGN = PAPER / 'SEPARATED_SCORING_DESIGN.en.md'  # public copy: the translation (the original is not shipped)
SIZES = [4, 8, 16, 32, 64]
N_PANELS, MAX_MEMBERS, N_OUTSIDE, BOOT = 100, 16, 10, 2000
LABELS = ['source', 'target', 'outside', 'excess']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rng(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big'))


def dump(path, value):
    def convert(v):
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert), encoding='utf-8')


def write_tsv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def load():
    with (RAW / 'manifest.tsv').open(encoding='utf-8') as f:
        manifest = {r['path']: r for r in csv.DictReader(f, delimiter='\t')}
    inputs = ['huggingface/bigcodebench_v0.1.4.jsonl'] + [
        f'huggingface/bigcodebench_perf_{mode}.jsonl' for mode in ['complete', 'instruct']]
    provenance = {}
    for name in inputs:
        actual = sha(RAW / name)
        if actual != manifest[name]['sha256']:
            raise ValueError(f'Input hash differs from existing manifest: {name}')
        provenance[name] = {**manifest[name], 'verified_sha256': actual}
    tasks = [json.loads(s)['task_id'] for s in (RAW / inputs[0]).read_text(encoding='utf-8').splitlines()]
    assert len(tasks) == len(set(tasks)) == 1140
    raw = {}
    for mode in ['complete', 'instruct']:
        rows = [json.loads(s) for s in (RAW / f'huggingface/bigcodebench_perf_{mode}.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len({r['Model'] for r in rows}) == len(rows)
        raw[mode] = {r['Model']: r for r in rows}
        for row in rows:
            assert set(row) == {'Model', *tasks}
            assert all(row[t] in (0, 1, '0', '1') for t in tasks)
    names = sorted(raw['complete'].keys() & raw['instruct'].keys())
    assert len(names) == 124
    matrices = {mode: np.array([[int(raw[mode][name][t]) for t in tasks] for name in names], dtype=np.int64)
                for mode in raw}
    return names, tasks, matrices, provenance


def make_plan(n, sizes=SIZES, n_panels=N_PANELS):
    result = {}
    for p in sizes:
        panels = []
        for j in range(n_panels):
            random = rng(f'separated_scoring/v1/{p}/{j}')
            panel = random.choice(n, p, replace=False)
            outside_panel = np.setdiff1d(np.arange(n), panel)
            chosen = panel if p <= MAX_MEMBERS else random.choice(panel, MAX_MEMBERS, replace=False)
            members = []
            for m in chosen:
                u = int(random.choice(panel[panel != m]))
                replacement = int(random.choice(outside_panel))
                pool = outside_panel[outside_panel != replacement]
                outside = random.choice(pool, min(N_OUTSIDE, len(pool)), replace=False)
                members.append({'m': int(m), 'u': u, 'r': replacement, 'outside': outside.tolist()})
            panels.append({'panel': panel.tolist(), 'members': members})
        result[p] = panels
    return result


def masks(source, panel, item):
    """Only source outcomes are accessible here."""
    m, u, r = item['m'], item['u'], item['r']
    assert m != u and m in panel and u in panel and r not in panel
    counts = source[panel].sum(0)
    p = len(panel)
    return 2 * (counts - source[m] + source[r]) < p, 2 * (counts - source[u] + source[r]) < p


def contrast(y, left, right, w=None):
    w = np.ones(y.shape[-1]) if w is None else w
    return float((y * w)[left].sum() / w[left].sum() - (y * w)[right].sum() / w[right].sum())


def vector_bootstrap(left, right, score_vectors, weights):
    """Equal event weights; each panel has the same fixed number of members.

    score_vectors is [event, task, channel], with channels source, target,
    mean target of the outside models. Return [weight replicate, channel].
    """
    A, B = left.astype(float), right.astype(float)
    den_a, den_b = weights @ A.T, weights @ B.T
    if np.any(den_a <= 0) or np.any(den_b <= 0):
        raise ValueError('An empty selected set occurred under task resampling')
    means = np.empty((len(weights), score_vectors.shape[2]))
    for k in range(score_vectors.shape[2]):
        score = score_vectors[:, :, k]
        numerator = weights @ (A * score).T
        numerator /= den_a
        second = weights @ (B * score).T
        second /= den_b
        means[:, k] = np.mean(numerator - second, axis=1)
    return np.column_stack([means, means[:, 1] - means[:, 2]])


def tests():
    random = rng('separated_scoring/synthetic_checks')
    source = random.integers(0, 2, (20, 101))
    target = random.integers(0, 2, (20, 101))
    plan = make_plan(20, sizes=[4], n_panels=3)
    left, right, vectors = [], [], []
    tested = 0
    max_identity_error = 0.
    for row in plan[4]:
        for item in row['members']:
            a, b = masks(source, row['panel'], item)
            m = item['m']
            enters, exits = a & ~b, b & ~a
            assert np.all(source[m, enters] == 1) and np.all(source[m, exits] == 0)
            sref = source[m, b].mean()
            predicted = (enters.sum()*(1-sref) + exits.sum()*sref) / a.sum()
            observed = contrast(source[m], a, b)
            max_identity_error = max(max_identity_error, abs(observed-predicted))
            assert observed >= -1e-12
            assert abs(contrast(1-source[m], a, b)+observed) < 1e-12
            assert abs(contrast(np.ones(101), a, b)) < 1e-12
            # Regenerating masks after manipulating target cannot change source selection.
            changed_target = 1-target
            assert np.array_equal(a, masks(source, row['panel'], item)[0])
            assert changed_target.shape == target.shape
            left.append(a); right.append(b)
            vectors.append(np.column_stack([source[m], target[m], target[item['outside']].mean(0)]))
            tested += 1
    left, right, vectors = np.array(left), np.array(right), np.array(vectors)
    draws = random.multinomial(101, np.full(101, 1/101), size=7)
    weights = np.vstack([np.ones(101), draws])
    fast = vector_bootstrap(left, right, vectors, weights)
    slow = []
    for w in weights:
        slots = np.repeat(np.arange(101), w.astype(int))
        event = [[contrast(vectors[i, slots, k], left[i, slots], right[i, slots]) for k in range(3)]
                 for i in range(tested)]
        means = np.mean(event, axis=0)
        slow.append([*means, means[1]-means[2]])
    error = float(np.max(np.abs(fast-slow)))
    assert error < 1e-12 and max_identity_error < 1e-12
    return {'synthetic_member_events': tested, 'explicit_resampling_comparisons': int(fast.size),
            'maximum_explicit_resampling_error': error, 'maximum_source_identity_error': max_identity_error,
            'source_nonnegative_complement_negative_constant_zero': True}


def rank_counts(before, after):
    valid = before != 0
    flip = valid & (before * after < 0)
    return {'valid': int(valid.sum()), 'flips': int(flip.sum()),
            'flips_both_1pp': int((flip & (abs(before) >= .01) & (abs(after) >= .01)).sum()),
            'flips_both_2pp': int((flip & (abs(before) >= .02) & (abs(after) >= .02)).sum())}


def evaluate_group(source, target, panels, names, p, direction, weights):
    left, right, vectors, events = [], [], [], []
    members_per_panel = min(p, MAX_MEMBERS)
    for j, panel in enumerate(panels):
        assert len(panel['members']) == members_per_panel
        for item in panel['members']:
            m, outside = item['m'], item['outside']
            assert len(outside) == N_OUTSIDE and not (set(outside) & (set(panel['panel']) | {item['r']}))
            a, b = masks(source, panel['panel'], item)
            if not a.any() or not b.any():
                raise ValueError(f'Empty source-selected set: {direction}, p={p}, panel={j}, model={names[m]}')
            v = np.column_stack([source[m], target[m], target[outside].mean(0)])
            values = [contrast(v[:, k], a, b) for k in range(3)]
            entry, leave = a & ~b, b & ~a
            sref = float(source[m, b].mean())
            assert np.all(source[m, entry] == 1) and np.all(source[m, leave] == 0)
            assert abs(values[0] - (entry.sum()*(1-sref) + leave.sum()*sref)/a.sum()) < 1e-12
            out_ref, out_self = target[outside][:, b].mean(1), target[outside][:, a].mean(1)
            member_rank = rank_counts(target[m, b].mean()-out_ref, target[m, a].mean()-out_self)
            tri = np.triu_indices(N_OUTSIDE, 1)
            outside_rank = rank_counts((out_ref[:, None]-out_ref[None, :])[tri],
                                       (out_self[:, None]-out_self[None, :])[tri])
            event = {'direction': direction, 'p': p, 'panel': j, 'model': names[m],
                     'reference_removed': names[item['u']], 'replacement': names[item['r']],
                     **dict(zip(LABELS[:3], values)), 'excess': values[1]-values[2],
                     'n_self': int(a.sum()), 'n_reference': int(b.sum()), 'entries': int(entry.sum()), 'exits': int(leave.sum()),
                     'target_right_entries': int(target[m, entry].sum()), 'target_right_exits': int(target[m, leave].sum()),
                     **{'member_'+k: value for k, value in member_rank.items()},
                     **{'outside_'+k: value for k, value in outside_rank.items()}}
            events.append(event); left.append(a); right.append(b); vectors.append(v)
    left, right, vectors = np.array(left), np.array(right), np.array(vectors)
    fast = vector_bootstrap(left, right, vectors, weights)
    point_direct = np.array([[e[k] for k in LABELS] for e in events]).mean(0)
    if np.max(np.abs(fast[0]-point_direct)) > 1e-12:
        raise AssertionError('Point and vectorized calculations disagree')
    # Real-data check: explicit resampling of eight events for three bootstrap replicates.
    for w in weights[1:4]:
        slots = np.repeat(np.arange(source.shape[1]), w.astype(int))
        for i in range(8):
            actual_panel = panels[i // members_per_panel]
            actual_item = actual_panel['members'][i % members_per_panel]
            a2, b2 = masks(source[:, slots], actual_panel['panel'], actual_item)
            assert np.array_equal(a2, left[i, slots]) and np.array_equal(b2, right[i, slots])
            for k in range(3):
                assert abs(contrast(vectors[i, :, k], left[i], right[i], w) -
                           contrast(vectors[i, slots, k], a2, b2)) < 1e-12
    return fast, events


def summarize_group(direction, p, values, events):
    summary = {'direction': direction, 'p': p, 'events': len(events), 'panels': N_PANELS,
               'mean_self_size': float(np.mean([e['n_self'] for e in events])),
               'min_selected_size': min(min(e['n_self'], e['n_reference']) for e in events),
               'target_negative_event_share': float(np.mean([e['target'] < 0 for e in events])),
               'excess_negative_event_share': float(np.mean([e['excess'] < 0 for e in events]))}
    for i, key in enumerate(LABELS):
        low, high = np.percentile(values[1:, i], [2.5, 97.5])
        panel_means = np.array([np.mean([e[key] for e in events if e['panel'] == j]) for j in range(N_PANELS)])
        summary[key] = {'estimate': float(values[0, i]), 'low': float(low), 'high': float(high),
                        'panel_mc_se': float(panel_means.std(ddof=1)/np.sqrt(N_PANELS))}
    summary['target_over_source'] = float(values[0, 1]/values[0, 0]) if values[0, 0] > 0 else None
    for kind in ['member', 'outside']:
        counts = {k: sum(e[kind+'_'+k] for e in events) for k in ['valid', 'flips', 'flips_both_1pp', 'flips_both_2pp']}
        counts['flip_rate'] = counts['flips']/counts['valid'] if counts['valid'] else None
        counts['flipped_share_both_1pp'] = counts['flips_both_1pp']/counts['flips'] if counts['flips'] else None
        counts['flipped_share_both_2pp'] = counts['flips_both_2pp']/counts['flips'] if counts['flips'] else None
        summary[kind+'_rank'] = counts
    for region in ['entries', 'exits']:
        total = sum(e[region] for e in events)
        summary['target_accuracy_'+region] = sum(e['target_right_'+region] for e in events)/total if total else None
    return summary


def add_joint_intervals(groups, bootstraps):
    keys = [(i, metric) for i in range(len(groups)) for metric in ['target', 'excess']]
    points = np.array([groups[i][metric]['estimate'] for i, metric in keys])
    distribution = np.column_stack([bootstraps[i][1:, LABELS.index(metric)] for i, metric in keys])
    sd = distribution.std(0, ddof=1)
    if np.any(sd <= 0):
        raise ValueError('Degenerate target contrast in simultaneous interval family')
    maximum = np.max(abs((distribution-points)/sd), axis=1)
    q = float(np.quantile(maximum, .95))
    for k, (i, metric) in enumerate(keys):
        groups[i][metric]['joint_low'] = float(points[k]-q*sd[k])
        groups[i][metric]['joint_high'] = float(points[k]+q*sd[k])
    return {'contrasts': len(keys), 'replicates': len(distribution), 'critical_value': q,
            'type': 'Approximate centered max-absolute-t; task bootstrap, fixed models and panels'}


def make_plot(groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, direction, title in zip(axes, ['complete_to_instruct', 'instruct_to_complete'],
                                   ['Complete selects; Instruct scores', 'Instruct selects; Complete scores']):
        rows = [g for g in groups if g['direction'] == direction]
        for key, label, color, offset in [('source', 'Same-response effect', '#9b9b9b', -.11),
                                            ('target', 'Other-prompt effect', '#2374ab', 0),
                                            ('excess', 'Other-prompt excess over outsiders', '#bd4f32', .11)]:
            x = np.arange(len(rows))+offset
            y = np.array([g[key]['estimate']*100 for g in rows])
            if key == 'source':
                ax.plot(x, y, 'o--', color=color, label=label, markersize=4)
            else:
                low = np.array([g[key]['joint_low']*100 for g in rows])
                high = np.array([g[key]['joint_high']*100 for g in rows])
                ax.errorbar(x, y, yerr=[y-low, high-y], fmt='o-', color=color, label=label, markersize=4, capsize=3)
        ax.axhline(0, color='black', linewidth=.6)
        ax.axhline(1, color='#cccccc', linestyle=':', linewidth=.8)
        ax.set_xticks(np.arange(len(rows)), [g['p'] for g in rows])
        ax.set_xlabel('Number of selection models')
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel('Score difference (percentage points)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8, frameon=False)
    fig.suptitle('Outcome separation on 124 paired models and 1,140 shared tasks', fontsize=12)
    fig.tight_layout(rect=(0, .08, 1, .96))
    fig.savefig(OUT / 'cross_prompt_effects.png', dpi=180)
    fig.savefig(OUT / 'cross_prompt_effects.pdf')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    checks = tests()
    print('Synthetic separation, algebra and explicit-bootstrap checks passed.', flush=True)
    if '--self-test' in sys.argv:
        print(json.dumps(checks, indent=2)); return
    names, tasks, matrices, provenance = load()
    # Public copy (2026-11-17): the guard reads the package's own checksum list, SHA256SUMS, instead of the
    # authors' list of protected working files, and checks that no shipped file changes during the run.
    package = Path(os.environ.get('SCOPECARD_PACKAGE_ROOT', ROOT))
    protected = {path: digest for digest, path in (line.split('  ', 1) for line in (package / 'SHA256SUMS').read_text(encoding='utf-8').splitlines() if line)}
    protection_before = {p: sha(package / p) for p in protected}
    monitored = [PAPER / name for name in ['PANEL_FILTER_DESIGN.md', 'MMLUPRO_FILTER_DESIGN.md', 'MMLU_RECONSTRUCTION_DESIGN.md']]
    for directory in ['panel_filter', 'mmlu_reconstruction', 'mmlupro_filter']:
        monitored += [p for p in (PAPER / 'results' / directory).glob('*') if p.is_file() and p.suffix in {'.tsv', '.md', '.npz'}]
    monitored_before = {str(p.relative_to(ROOT)): sha(p) for p in monitored}
    plan = make_plan(len(names))
    dump(OUT / 'panel_plan.json', {'models': names, 'tasks': tasks, 'panels': plan})
    stamp = {'time_started': datetime.now().astimezone().isoformat(), 'design_sha256': sha(DESIGN),
             'script_sha256': sha(Path(__file__)), 'panel_plan_sha256': sha(OUT / 'panel_plan.json'),
             'inputs': provenance, 'bootstrap_replicates': BOOT,
             'primary': 'complete_to_instruct/p=8/excess', 'status': 'plan recorded before effect computation'}
    dump(OUT / 'run_manifest.json', stamp)
    weights = rng('separated_scoring/v1/task-bootstrap').multinomial(len(tasks), np.full(len(tasks), 1/len(tasks)), size=BOOT)
    weights = np.vstack([np.ones(len(tasks)), weights]).astype(float)
    all_events, groups, bootstraps = [], [], []
    for source_name, target_name in [('complete', 'instruct'), ('instruct', 'complete')]:
        direction = source_name+'_to_'+target_name
        source, target = matrices[source_name], matrices[target_name]
        for p in SIZES:
            result, events = evaluate_group(source, target, plan[p], names, p, direction, weights)
            summary = summarize_group(direction, p, result, events)
            groups.append(summary); bootstraps.append(result); all_events.extend(events)
            print(f'{direction} p={p}: source={100*result[0,0]:.3f}, target={100*result[0,1]:.3f}, '
                  f'excess={100*result[0,3]:.3f} pp ({len(events)} events)', flush=True)
    joint = add_joint_intervals(groups, bootstraps)
    write_tsv(OUT / 'events.tsv', all_events)
    model_rows = []
    for g in groups:
        for name in names:
            rows = [e for e in all_events if e['direction'] == g['direction'] and e['p'] == g['p'] and e['model'] == name]
            if rows:
                model_rows.append({'direction': g['direction'], 'p': g['p'], 'model': name, 'events': len(rows),
                                   **{key: float(np.mean([e[key] for e in rows])) for key in LABELS}})
    write_tsv(OUT / 'per_model.tsv', model_rows)
    compact = []
    for g in groups:
        for metric in LABELS:
            compact.append({'direction': g['direction'], 'p': g['p'], 'metric': metric,
                            **{k: g[metric].get(k, '') for k in ['estimate','low','high','joint_low','joint_high','panel_mc_se']}})
    write_tsv(OUT / 'effects.tsv', compact)
    np.savez_compressed(OUT / 'bootstrap.npz', values=np.array(bootstraps),
                        directions=np.array([g['direction'] for g in groups]), sizes=np.array([g['p'] for g in groups]),
                        metrics=np.array(LABELS), task_weights=weights[1:].astype(np.int16))
    primary = next(g for g in groups if g['direction'] == 'complete_to_instruct' and g['p'] == 8)
    pe = primary['excess']
    decision = 'positive_cross_prompt_excess' if pe['low'] > 0 else ('negative_cross_prompt_excess' if pe['high'] < 0 else 'unresolved')
    checks.update({'real_explicit_resampling_checks': len(groups)*3*8*3,
                   'input_models': len(names), 'input_tasks': len(tasks),
                   'models_with_identical_response_vectors': [names[i] for i in range(len(names)) if np.array_equal(matrices['complete'][i], matrices['instruct'][i])],
                   'within_model_disagreement_min_mean_max': [float(v) for v in [
                       np.mean(matrices['complete'] != matrices['instruct'], axis=1).min(),
                       np.mean(matrices['complete'] != matrices['instruct']),
                       np.mean(matrices['complete'] != matrices['instruct'], axis=1).max()]],
                   'protected_files_checked': len(protected),
                   'protected_changes_during_run': [p for p, s in protection_before.items() if sha(package/p) != s],
                   'protected_changes_from_prior_baseline': [p for p, s in protected.items() if sha(package/p) != s],
                   'monitored_original_changes_during_run': [p for p, s in monitored_before.items() if sha(ROOT/p) != s]})
    assert not checks['protected_changes_during_run']
    dump(OUT / 'verification.json', checks)
    full = {'primary': primary, 'primary_decision': decision, 'primary_lower_bound_at_least_one_pp': pe['low'] >= .01,
            'joint_interval_family': joint, 'groups': groups,
            'scope': 'Exploratory cross-prompt transfer; fixed archived runs, models, panels. Not independent repeated inference.',
            'runtime_seconds_before_plot': time.perf_counter()-started}
    dump(OUT / 'summary.json', full)
    lines = ['# Separated selection and scoring: cross-prompt results', '',
             'Exploratory. Fixed archived runs; 124 exact-name common models, 1,140 tasks. '
             'Source responses alone select; target responses never vote. Each direction has 100 panels per size.', '',
             'All score differences and intervals below are in percentage points. '
             'Curves use approximate simultaneous 95% intervals across 20 target/excess contrasts; '
             'source is the mechanically nonnegative same-response control.', '',
             '| Selection → scoring | Panel size | Source | Target [joint 95%] | Outside | Excess [joint 95%] |',
             '|---|---:|---:|---|---:|---|']
    for g in groups:
        cell = lambda k: f"{100*g[k]['estimate']:.3f} [{100*g[k]['joint_low']:.3f}, {100*g[k]['joint_high']:.3f}]"
        lines.append(f"| {g['direction']} | {g['p']} | {100*g['source']['estimate']:.3f} | {cell('target')} | "
                     f"{100*g['outside']['estimate']:.3f} | {cell('excess')} |")
    lines += ['', f"Primary Complete → Instruct, p=8, excess: {100*pe['estimate']:.3f} pp, "
              f"pointwise percentile 95% [{100*pe['low']:.3f}, {100*pe['high']:.3f}]. Decision: {decision}.",
              f"Lower bound supports at least 1 pp: {pe['low'] >= .01}.", '',
              'Outside is measured on the same H_self vs H_reference contrast. Positive excess '
              'does not identify a correct target or prove historical benchmark unfairness. '
              'Cross-prompt outcomes share models and tasks; this is not a same-protocol independent replicate.', '',
              'Ranking columns in summary.json are descriptive repeated-comparison events only, not jointly resolved pair reversals.', '',
              f"Synthetic checks and {checks['real_explicit_resampling_checks']} real explicit-resampling comparisons passed. "
              f"Protected changes during run: {checks['protected_changes_during_run']}."]
    (OUT / 'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    make_plot(groups)
    artifact_hashes = {str(p.relative_to(ROOT)): sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'SHA256SUMS.json'}
    artifact_hashes[str(DESIGN.relative_to(ROOT))] = sha(DESIGN)
    artifact_hashes[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    dump(OUT / 'SHA256SUMS.json', artifact_hashes)
    print(f'Completed in {time.perf_counter()-started:.1f} s. {decision}. Outputs: {OUT}', flush=True)


if __name__ == '__main__':
    main()
