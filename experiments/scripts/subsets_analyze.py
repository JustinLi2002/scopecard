"""Acceptance checks and every analysis of the released-subset study (experiments/SUBSET_C_DESIGN.md).

Reads only the files fetched by subsets_fetch.py. Nothing downloaded is imported or executed: the SWE-bench Lite
criteria are re-implemented below from a reading of swebench/collect/make_lite/criteria.py and make_lite.py.

Usage (from experiments/):  python -E -P scripts/subsets_analyze.py results/subsets/raw results/subsets
"""
import ast
import csv
import hashlib
import json
import pathlib
import re
import sys

import numpy as np

RAW, OUT = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
B = 2000                      # bootstrap replicates (design, Section 6)
K = 2000                      # random subsets (design, Section 7)
HARD_RELEASE = '2024-07-24'   # date of the BigCodeBench-Hard announcement
RECORDS = []                  # (case, analysis, subject, quantity, value)
SUMMARY = []                  # lines of summary.md
ACCEPT = []                   # lines of acceptance.txt
csv.field_size_limit(10 ** 9)


def rng_for(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big'))


def rec(case, analysis, subject, quantity, value):
    if isinstance(value, (float, np.floating)):
        value = f'{float(value):.6g}'
    RECORDS.append((case, analysis, subject, quantity, str(value)))


def say(line=''):
    print(line, flush=True)
    SUMMARY.append(line)


def accept(ok, text):
    line = ('PASS  ' if ok else 'FAIL  ') + text
    ACCEPT.append(line)
    print(line, flush=True)
    return ok


def read_jsonl(name):
    with (RAW / 'huggingface' / name).open(encoding='utf-8') as fh:
        return [json.loads(line) for line in fh]


# ---------------------------------------------------------------------------------------------------------------
# SWE-bench Lite criteria, re-implemented (criteria.py and make_lite.py at the pinned commit)
# ---------------------------------------------------------------------------------------------------------------
PR_KEYWORDS = {'close', 'closes', 'closed', 'fix', 'fixes', 'fixed', 'resolve', 'resolves', 'resolved'}
IMAGE_EXT = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.svg', '.webp', '.ico', '.heif', '.bpg', '.avif']
VIDEO_EXT = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
RE_IMAGE = re.compile(r'\b({})\b'.format('|'.join(re.escape(e) for e in IMAGE_EXT)), flags=re.IGNORECASE)
RE_VIDEO = re.compile(r'\b({})\b'.format('|'.join(re.escape(e) for e in VIDEO_EXT)), flags=re.IGNORECASE)
RE_HUNK = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')


def contains_git_commit_hash(text):
    return bool(re.search(r'(?<!/)\b[0-9a-f]{40}\b', text) or re.search(r'\[[0-9a-f]{23}\]', text))


def contains_hyperlinks(text, repo):
    prefix = re.escape(f'http://github.com/{repo}')
    return bool(re.search(r'(?:https?://(?!{}).+)|(?:www\.(?!{}).+)'.format(prefix, prefix), text))


def contains_image(text):
    return RE_IMAGE.search(text) is not None or RE_VIDEO.search(text) is not None


def issue_reference(text):
    """Returns (certain, hash_number). `certain`: a closing keyword before #n, or a GitLab issue URL.
    `hash_number`: the text has some #n. The original code keeps an instance with #n only if an HTTP request for
    that issue does not return 200; that request cannot be reproduced, so both readings are carried."""
    refs = dict(re.findall(r'(\w+)\s+\#(\d+)', text))
    certain = any(word.lower() in PR_KEYWORDS for word in refs) or bool(re.search(r'https?:\/\/gitlab.com\/(.*)\/issues', text))
    return certain, bool(re.search(r'#\d+', text))


def parse_patch(text):
    """Minimal unified-diff reader with the file classes of unidiff.PatchSet (added, removed, modified)."""
    files, cur, left_src, left_tgt = [], None, 0, 0
    for line in text.splitlines():
        if left_src > 0 or left_tgt > 0:                 # inside a hunk body
            if line.startswith('\\'):
                continue
            if line.startswith('+'):
                left_tgt -= 1
            elif line.startswith('-'):
                left_src -= 1
            else:
                left_src -= 1
                left_tgt -= 1
            continue
        if line.startswith('diff --git '):
            m = re.match(r'^diff --git (a/.+?) (b/.+)$', line)
            cur = {'source': m.group(1) if m else None, 'target': m.group(2) if m else None, 'hunks': []}
            files.append(cur)
        elif line.startswith('new file mode') and cur is not None:
            cur['source'] = '/dev/null'
        elif line.startswith('deleted file mode') and cur is not None:
            cur['target'] = '/dev/null'
        elif line.startswith('--- '):
            if cur is None or cur['hunks']:
                cur = {'source': None, 'target': None, 'hunks': []}
                files.append(cur)
            cur['source'] = line[4:].split('\t')[0].strip()
        elif line.startswith('+++ ') and cur is not None:
            cur['target'] = line[4:].split('\t')[0].strip()
        else:
            m = RE_HUNK.match(line)
            if m and cur is not None:
                s_start, s_len = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
                t_start, t_len = int(m.group(3)), int(m.group(4) if m.group(4) is not None else 1)
                cur['hunks'].append((s_start, s_len, t_start, t_len))
                left_src, left_tgt = s_len, t_len
    for f in files:
        one = len(f['hunks']) == 1
        f['added'] = f['source'] == '/dev/null' or (one and f['hunks'][0][0] == 0 and f['hunks'][0][1] == 0)
        f['removed'] = f['target'] == '/dev/null' or (one and f['hunks'][0][2] == 0 and f['hunks'][0][3] == 0)
        f['modified'] = not (f['added'] or f['removed'])
    return files


def contains_pytest_match_arg(test_patch):
    if any(x in test_patch for x in ('pytest.raises', 'pytest.warns', 'pytest.deprecated_call')):
        return 'match' in test_patch
    return any(x in test_patch for x in ('assertOutput', 'assertRaises', 'checks.Error'))


def lite_flags(inst):
    """One boolean per stated filter: True means the filter excludes the instance."""
    ps, repo = inst['problem_statement'], inst['repo']
    files = parse_patch(inst['patch'])
    modified = [f for f in files if f['modified']]
    hunks = sum(len(f['hunks']) for f in modified)
    certain, hash_number = issue_reference(ps)
    return {
        'words_le_40': len(ps.split()) <= 40,
        'hyperlink': contains_hyperlinks(ps, repo),
        'issue_ref_certain': certain,
        'issue_ref_hash_number': hash_number and not certain,
        'commit_hash': contains_git_commit_hash(ps),
        'image': contains_image(ps),
        'adds_or_removes_file': any(f['added'] or f['removed'] for f in files),
        'more_than_1_file': len(modified) > 1,
        'hunks_not_1_to_3': not (0 < hunks <= 3),
        'test_error_message': contains_pytest_match_arg(inst['test_patch']),
    }


# ---------------------------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------------------------
def kendall_tau_b(x, y):
    iu = np.triu_indices(len(x), 1)
    sx = np.sign(x[:, None] - x[None, :])[iu]
    sy = np.sign(y[:, None] - y[None, :])[iu]
    denom = np.sqrt(float(np.sum(sx != 0)) * float(np.sum(sy != 0)))
    return float(np.sum(sx * sy) / denom) if denom > 0 else float('nan')


def boot_weights(key, n, kind='plain', in_o=None, clusters=None):
    """B x n matrix of instance multiplicities."""
    rng = rng_for(key)
    if kind == 'plain':
        return rng.multinomial(n, np.full(n, 1.0 / n), size=B)
    if kind == 'stratified':
        w = np.zeros((B, n), dtype=np.int64)
        for mask in (in_o, ~in_o):
            m = int(mask.sum())
            w[:, mask] = rng.multinomial(m, np.full(m, 1.0 / m), size=B)
        return w
    if kind == 'cluster':
        names = sorted(set(clusters))
        index = np.array([names.index(c) for c in clusters])
        counts = rng.multinomial(len(names), np.full(len(names), 1.0 / len(names)), size=B)
        return counts[:, index]
    raise ValueError(kind)


def region_scores(Y, w, mask):
    """Systems x replicates matrix of mean outcomes over the masked instances under multiplicities w."""
    wm = w[:, mask].astype(np.float64)
    return (Y[:, mask].astype(np.float64) @ wm.T) / wm.sum(axis=1)


def pair_statistics(s_o, s_u, s_v, b_o, b_u, b_v, big_w, chunk=1500):
    """Per-pair estimates, percentile intervals and simultaneous critical values (design, Sections 4-6)."""
    n = len(s_o)
    ia, ib = np.triu_indices(n, 1)
    sign = np.where(s_o[ia] >= s_o[ib], 1.0, -1.0)           # orientation: A has the higher score on O
    a = np.where(sign > 0, ia, ib)
    b = np.where(sign > 0, ib, ia)
    est = {'O': s_o[a] - s_o[b], 'U': s_u[a] - s_u[b], 'V': s_v[a] - s_v[b]}
    est['c'] = est['O'] - est['U']
    out = {'a': a, 'b': b, 'tied_O': est['O'] == 0}
    for name in ('O', 'U', 'V', 'c'):
        out[name] = est[name]
        out['lo_' + name] = np.empty(len(a))
        out['hi_' + name] = np.empty(len(a))
        out['se_' + name] = np.empty(len(a))
    out['rho_lo'] = np.full(len(a), np.nan)
    out['rho_hi'] = np.full(len(a), np.nan)
    max_t_ov, max_t_c = np.zeros(B), np.zeros(B)
    for start in range(0, len(a), chunk):
        sl = slice(start, start + chunk)
        d = {'O': b_o[a[sl]] - b_o[b[sl]], 'U': b_u[a[sl]] - b_u[b[sl]], 'V': b_v[a[sl]] - b_v[b[sl]]}
        d['c'] = d['O'] - d['U']
        for name in ('O', 'U', 'V', 'c'):
            if np.isnan(d[name]).any():                      # a replicate with an empty region (cluster bootstrap)
                lo, hi = np.nanpercentile(d[name], [2.5, 97.5], axis=1)
                out['se_' + name][sl] = np.nanstd(d[name], axis=1, ddof=1)
                d[name] = np.where(np.isnan(d[name]), est[name][sl][:, None], d[name])
            else:
                lo, hi = np.percentile(d[name], [2.5, 97.5], axis=1)
                out['se_' + name][sl] = d[name].std(axis=1, ddof=1)
            out['lo_' + name][sl], out['hi_' + name][sl] = lo, hi
        for name, target in (('O', 'ov'), ('V', 'ov'), ('c', 'c')):
            se = out['se_' + name][sl][:, None]
            with np.errstate(divide='ignore', invalid='ignore'):
                t = np.where(se > 0, np.abs(d[name] - est[name][sl][:, None]) / se, 0.0)
            if target == 'ov':
                max_t_ov = np.maximum(max_t_ov, t.max(axis=0))
            else:
                max_t_c = np.maximum(max_t_c, t.max(axis=0))
        with np.errstate(divide='ignore', invalid='ignore'):
            rho = np.where(d['O'] > 0, 1.0 - d['U'] / d['O'], np.nan)
        enough = np.sum(~np.isnan(rho), axis=1) >= 0.5 * B
        if enough.any():
            lo, hi = np.nanpercentile(rho[enough], [2.5, 97.5], axis=1)
            idx = np.arange(start, min(start + chunk, len(a)))[enough]
            out['rho_lo'][idx], out['rho_hi'][idx] = lo, hi
    out['q_ov'] = float(np.percentile(max_t_ov, 95))
    out['q_c'] = float(np.percentile(max_t_c, 95))
    out['c_star'] = np.abs(est['O']) / big_w
    with np.errstate(divide='ignore', invalid='ignore'):
        out['rho'] = np.where(est['O'] != 0, 1.0 - est['U'] / est['O'], np.nan)
    return out


def labels(p, simultaneous):
    """Decision language of the design, Section 5, for oriented pairs (the estimate of Delta_O is >= 0)."""
    if simultaneous:
        lo = {k: p[k] - (p['q_c'] if k == 'c' else p['q_ov']) * p['se_' + k] for k in ('O', 'V', 'c')}
        hi = {k: p[k] + (p['q_c'] if k == 'c' else p['q_ov']) * p['se_' + k] for k in ('O', 'V', 'c')}
        for k in ('O', 'V', 'c'):                           # a statistic with zero SD is left unresolved
            lo[k] = np.where(p['se_' + k] > 0, lo[k], -np.inf)
            hi[k] = np.where(p['se_' + k] > 0, hi[k], np.inf)
    else:
        lo = {k: p['lo_' + k] for k in ('O', 'V', 'c')}
        hi = {k: p['hi_' + k] for k in ('O', 'V', 'c')}
    res_o = (lo['O'] > 0) | (hi['O'] < 0)
    res_v = (lo['V'] > 0) | (hi['V'] < 0)
    return {
        'resolved_O': res_o,
        'resolved_V': res_v,
        'reversal': res_o & res_v & (np.sign(p['O']) != np.sign(p['V'])),
        'newly_resolved': ~res_o & res_v,
        'lost': res_o & ~res_v,
        'attenuated': res_o & (lo['c'] > 0),
        'amplified': res_o & (hi['c'] < 0),
    }


def subset_statistics(Y, mask):
    """tau_b, opposite-sign pairs and mean |c| of a subset against the whole (design, Section 7)."""
    s_sub, s_rest, s_all = Y[:, mask].mean(axis=1), Y[:, ~mask].mean(axis=1), Y.mean(axis=1)
    iu = np.triu_indices(Y.shape[0], 1)
    d_sub = (s_sub[:, None] - s_sub[None, :])[iu]
    d_rest = (s_rest[:, None] - s_rest[None, :])[iu]
    d_all = (s_all[:, None] - s_all[None, :])[iu]
    return kendall_tau_b(s_sub, s_all), int(np.sum(d_sub * d_all < 0)), float(np.mean(np.abs(d_sub - d_rest)))


def reference(key, Y, mask, strata=None):
    rng = rng_for(key)
    n, n_o = Y.shape[1], int(mask.sum())
    stats = np.empty((K, 3))
    for k in range(K):
        m = np.zeros(n, dtype=bool)
        if strata is None:
            m[rng.choice(n, size=n_o, replace=False)] = True
        else:
            for value in sorted(set(strata)):
                pool = np.flatnonzero(strata == value)
                m[rng.choice(pool, size=int(mask[pool].sum()), replace=False)] = True
        stats[k] = subset_statistics(Y, m)
    return stats


def describe(values):
    q = np.percentile(values, [5, 25, 50, 75, 95])
    return {'p05': q[0], 'q1': q[1], 'median': q[2], 'q3': q[3], 'p95': q[4]}


def analyse(case, analysis, title, Y, names, mask, strata=None, clusters=None, write_pairs=True):
    """One (V, O) analysis: Y is systems x instances over V, mask marks O."""
    n_sys, n = Y.shape
    n_o = int(mask.sum())
    big_w = 1.0 - n_o / n
    key = f'subset_c/{case}/{analysis}'
    s_o, s_u, s_v = Y[:, mask].mean(axis=1), Y[:, ~mask].mean(axis=1), Y.mean(axis=1)
    w = boot_weights(key + '/bootstrap', n)
    b_o, b_u = region_scores(Y, w, mask), region_scores(Y, w, ~mask)
    b_v = region_scores(Y, w, np.ones(n, dtype=bool))
    p = pair_statistics(s_o, s_u, s_v, b_o, b_u, b_v, big_w)
    lab, lab_sim = labels(p, False), labels(p, True)
    n_pairs, not_tied = len(p['a']), ~p['tied_O']
    opposite = int(np.sum(not_tied & (p['O'] * p['V'] < 0)))
    tau, opp_check, mean_abs_c = subset_statistics(Y, mask)

    say(f'### {title}')
    say(f'- {n_sys} systems, {n} instances; kept {n_o}, excluded {n - n_o}; W = {big_w:.4f}; 1/W = {1 / big_w:.3f}.')
    for q, v in (('n_systems', n_sys), ('n_instances', n), ('n_kept', n_o), ('W', big_w), ('n_pairs', n_pairs),
                 ('n_pairs_tied_on_O', int(p['tied_O'].sum())), ('pairs_opposite_sign', opposite), ('tau_b_O_vs_V', tau),
                 ('mean_abs_c', mean_abs_c), ('q_simultaneous_OV', p['q_ov']), ('q_simultaneous_c', p['q_c'])):
        rec(case, analysis, 'all', q, v)
    shift = s_u - s_o
    shift_b = b_u - b_o
    lo_s, hi_s = np.percentile(shift_b, [2.5, 97.5], axis=1)
    say(f'- Level: score on the excluded part minus score on the kept part, mean over systems {shift.mean():+.4f} '
        f'(range {shift.min():+.4f} to {shift.max():+.4f}); interval excludes zero for '
        f'{int(np.sum((lo_s > 0) | (hi_s < 0)))} of {n_sys} systems '
        f'({int(np.sum(hi_s < 0))} lower on the excluded part, {int(np.sum(lo_s > 0))} higher).')
    rec(case, analysis, 'all', 'level_shift_mean', shift.mean())
    rec(case, analysis, 'all', 'level_shift_n_lower_on_U', int(np.sum(hi_s < 0)))
    rec(case, analysis, 'all', 'level_shift_n_higher_on_U', int(np.sum(lo_s > 0)))
    top_o = [names[i] for i in np.flatnonzero(s_o == s_o.max())]
    top_v = [names[i] for i in np.flatnonzero(s_v == s_v.max())]
    say(f'- Ranking: Kendall tau_b between kept-part and parent scores {tau:.4f}; {opposite} of {int(not_tied.sum())} '
        f'pairs not tied on the kept part have opposite signs; top on kept part: {", ".join(top_o)}; top on parent: '
        f'{", ".join(top_v)}.')
    rec(case, analysis, 'all', 'top_same', int(top_o == top_v))
    for tag, la in (('unadjusted', lab), ('simultaneous', lab_sim)):
        counts = {k: int(v.sum()) for k, v in la.items()}
        say(f'- {tag.capitalize()} 95% intervals, {n_pairs} pairs: resolved on kept part {counts["resolved_O"]}, on parent '
            f'{counts["resolved_V"]}; reversals {counts["reversal"]}; newly resolved {counts["newly_resolved"]}; lost '
            f'{counts["lost"]}; attenuated {counts["attenuated"]}; amplified {counts["amplified"]}.')
        for k, v in counts.items():
            rec(case, analysis, 'all', f'{k}_{tag}', v)
    sel = lab['resolved_O']
    if sel.sum() >= 5:
        d_rho, d_c = describe(p['rho'][sel]), describe(np.abs(p['c'][sel]))
        above_1, above_w = float(np.mean(p['rho'][sel] > 1)), float(np.mean(p['rho'][sel] > 1 / big_w))
        say(f'- rho over the {int(sel.sum())} pairs resolved on the kept part: median {d_rho["median"]:.3f} (quartiles '
            f'{d_rho["q1"]:.3f} to {d_rho["q3"]:.3f}; 5th to 95th percentile {d_rho["p05"]:.3f} to {d_rho["p95"]:.3f}); '
            f'share above 1: {above_1:.3f}; share above 1/W: {above_w:.3f}. |c| median {d_c["median"]:.4f} '
            f'(5th to 95th percentile {d_c["p05"]:.4f} to {d_c["p95"]:.4f}).')
        for k, v in d_rho.items():
            rec(case, analysis, 'resolved_on_O', f'rho_{k}', v)
        for k, v in d_c.items():
            rec(case, analysis, 'resolved_on_O', f'abs_c_{k}', v)
        rec(case, analysis, 'resolved_on_O', 'rho_share_above_1', above_1)
        rec(case, analysis, 'resolved_on_O', 'rho_share_above_1_over_W', above_w)
    else:
        say(f'- rho: only {int(sel.sum())} pairs are resolved on the kept part; no distribution is reported.')
    for ref_name, ref_strata in (('uniform', None), ('by_repository', strata)):
        if ref_name == 'by_repository' and strata is None:
            continue
        ref = reference(f'{key}/reference/{ref_name}', Y, mask, ref_strata)
        parts = []
        for col, (label, obs) in enumerate((('tau_b', tau), ('opposite-sign pairs', opposite), ('mean |c|', mean_abs_c))):
            lo, hi = np.percentile(ref[:, col], [2.5, 97.5])
            pct = float(np.mean(ref[:, col] <= obs))
            parts.append(f'{label} {obs:.4g} against a reference of {lo:.4g} to {hi:.4g} (percentile {100 * pct:.1f})')
            slug = label.replace(' ', '_').replace('|', '').replace('-', '_')
            rec(case, analysis, f'reference_{ref_name}', f'{slug}_lo', lo)
            rec(case, analysis, f'reference_{ref_name}', f'{slug}_hi', hi)
            rec(case, analysis, f'reference_{ref_name}', f'{slug}_percentile_of_observed', pct)
        say(f'- Random subsets of the same size ({ref_name.replace("_", " ")}, central 95%): ' + '; '.join(parts) + '.')
    for kind in ('stratified', 'cluster'):
        if kind == 'cluster' and clusters is None:
            continue
        w2 = boot_weights(f'{key}/bootstrap/{kind}', n, kind, mask, clusters)
        p2 = pair_statistics(s_o, s_u, s_v, region_scores(Y, w2, mask), region_scores(Y, w2, ~mask),
                             region_scores(Y, w2, np.ones(n, dtype=bool)), big_w)
        c2 = {k: int(v.sum()) for k, v in labels(p2, False).items()}
        say(f'- Sensitivity, {kind} bootstrap (unadjusted): resolved on kept part {c2["resolved_O"]}, on parent '
            f'{c2["resolved_V"]}; reversals {c2["reversal"]}; newly resolved {c2["newly_resolved"]}; lost {c2["lost"]}; '
            f'attenuated {c2["attenuated"]}; amplified {c2["amplified"]}.')
        for k, v in c2.items():
            rec(case, analysis, f'sensitivity_{kind}', f'{k}_unadjusted', v)
    for tag, la in (('unadjusted', lab), ('simultaneous', lab_sim)):
        for i in np.flatnonzero(la['reversal']):
            say(f'  - Reversal ({tag}): {names[p["a"][i]]} minus {names[p["b"][i]]}: kept {p["O"][i]:+.4f} '
                f'[{p["lo_O"][i]:+.4f}, {p["hi_O"][i]:+.4f}], excluded {p["U"][i]:+.4f}, parent {p["V"][i]:+.4f} '
                f'[{p["lo_V"][i]:+.4f}, {p["hi_V"][i]:+.4f}], c {p["c"][i]:+.4f}, c* {p["c_star"][i]:.4f}.')
    say()
    if write_pairs:
        with (OUT / f'pairs_{case}_{analysis}.tsv').open('w', encoding='utf-8', newline='\n') as fh:
            cols = ['O', 'lo_O', 'hi_O', 'U', 'lo_U', 'hi_U', 'V', 'lo_V', 'hi_V', 'c', 'lo_c', 'hi_c', 'c_star', 'rho',
                    'rho_lo', 'rho_hi']
            fh.write('\t'.join(['system_a', 'system_b'] + cols + [f'{k}_unadj' for k in lab] + [f'{k}_sim' for k in lab_sim]) + '\n')
            for i in range(n_pairs):
                row = [names[p['a'][i]], names[p['b'][i]]] + [f'{p[c][i]:.6f}' for c in cols]
                row += [str(int(lab[k][i])) for k in lab] + [str(int(lab_sim[k][i])) for k in lab_sim]
                fh.write('\t'.join(row) + '\n')
        with (OUT / f'systems_{case}_{analysis}.tsv').open('w', encoding='utf-8', newline='\n') as fh:
            fh.write('system\tscore_kept\tscore_excluded\tscore_parent\tshift\tshift_lo\tshift_hi\n')
            for i in np.argsort(-s_v):
                fh.write(f'{names[i]}\t{s_o[i]:.6f}\t{s_u[i]:.6f}\t{s_v[i]:.6f}\t{shift[i]:.6f}\t{lo_s[i]:.6f}\t{hi_s[i]:.6f}\n')
    return p, lab, lab_sim


def matched_analysis(case, analysis, title, Y, names, mask, strata_labels):
    """Mechanism check: the excluded part reweighted to the kept part's distribution over a label."""
    n = Y.shape[1]
    key = f'subset_c/{case}/{analysis}'
    values = sorted(set(strata_labels))
    common = [v for v in values if np.any(mask & (strata_labels == v)) and np.any(~mask & (strata_labels == v))]
    dropped = [v for v in values if v not in common]
    keep = np.isin(strata_labels, common)
    w = boot_weights(key + '/bootstrap', n).astype(np.float64)

    def scores(weights):
        """Returns kept, excluded-raw and excluded-matched scores; each systems x (replicates or 1)."""
        tot_o = (weights[:, mask & keep]).sum(axis=1)
        s_o = (Y[:, mask & keep].astype(np.float64) @ weights[:, mask & keep].T) / tot_o
        s_u = (Y[:, ~mask & keep].astype(np.float64) @ weights[:, ~mask & keep].T) / weights[:, ~mask & keep].sum(axis=1)
        s_m = np.zeros_like(s_o)
        share_sum = np.zeros(weights.shape[0])
        for v in common:
            cell_o, cell_u = mask & (strata_labels == v), ~mask & (strata_labels == v)
            n_u = weights[:, cell_u].sum(axis=1)
            share = np.where(n_u > 0, weights[:, cell_o].sum(axis=1) / tot_o, 0.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                mean_u = np.where(n_u > 0, (Y[:, cell_u].astype(np.float64) @ weights[:, cell_u].T) / n_u, 0.0)
            s_m += mean_u * share
            share_sum += share
        return s_o, s_u, s_m / share_sum

    s_o, s_u, s_m = (x[:, 0] for x in scores(np.ones((1, n))))
    b_o, b_u, b_m = scores(w)
    ia, ib = np.triu_indices(len(names), 1)
    sign = np.where(s_o[ia] >= s_o[ib], 1.0, -1.0)
    a, b = np.where(sign > 0, ia, ib), np.where(sign > 0, ib, ia)
    out = {}
    for tag, su, bu in (('raw', s_u, b_u), ('matched', s_m, b_m)):
        c = (s_o[a] - s_o[b]) - (su[a] - su[b])
        cb = (b_o[a] - b_o[b]) - (bu[a] - bu[b])
        lo, hi = np.percentile(cb, [2.5, 97.5], axis=1)
        lo_o, hi_o = np.percentile(b_o[a] - b_o[b], [2.5, 97.5], axis=1)
        res_o = lo_o > 0
        out[tag] = (c, lo, hi, res_o)
    say(f'### {title}')
    say(f'- {len(names)} systems; {int((mask & keep).sum())} kept and {int((~mask & keep).sum())} excluded instances '
        f'with labels present on both sides ({", ".join(common)}); labels dropped: {", ".join(dropped) or "none"}.')
    for tag in ('raw', 'matched'):
        c, lo, hi, res_o = out[tag]
        say(f'- {tag.capitalize()} excluded part: mean c over pairs {c.mean():+.4f}, median |c| {np.median(np.abs(c)):.4f}; among the '
            f'{int(res_o.sum())} pairs resolved on the kept part, attenuated {int(np.sum(res_o & (lo > 0)))}, amplified '
            f'{int(np.sum(res_o & (hi < 0)))}.')
        rec(case, analysis, tag, 'mean_c', c.mean())
        rec(case, analysis, tag, 'median_abs_c', np.median(np.abs(c)))
        rec(case, analysis, tag, 'attenuated_unadjusted', int(np.sum(res_o & (lo > 0))))
        rec(case, analysis, tag, 'amplified_unadjusted', int(np.sum(res_o & (hi < 0))))
    raw_c, matched_c = out['raw'][0], out['matched'][0]
    denom = float(np.sum(raw_c ** 2))
    if denom > 0:
        share = 1.0 - float(np.sum(matched_c ** 2)) / denom
        say(f'- Share of the sum of squared c removed by matching: {share:.3f}.')
        rec(case, analysis, 'matched', 'share_of_squared_c_removed', share)
    level_raw, level_matched = float(np.mean(s_u - s_o)), float(np.mean(s_m - s_o))
    say(f'- Level: mean over systems of excluded minus kept score, raw {level_raw:+.4f}, matched {level_matched:+.4f}.')
    rec(case, analysis, 'raw', 'level_shift_mean', level_raw)
    rec(case, analysis, 'matched', 'level_shift_mean', level_matched)
    say()


# ---------------------------------------------------------------------------------------------------------------
# Load and accept: SWE-bench
# ---------------------------------------------------------------------------------------------------------------
OUT.mkdir(parents=True, exist_ok=True)
test_rows = sorted(read_jsonl('swebench_test.jsonl'), key=lambda r: r['instance_id'])
ids = [r['instance_id'] for r in test_rows]
pos = {i: k for k, i in enumerate(ids)}
repo = np.array([r['repo'] for r in test_rows])
lite_ids = {r['instance_id'] for r in read_jsonl('swebench_lite_test.jsonl')}
verified_rows = read_jsonl('swebench_verified_test.jsonl')
verified_ids = {r['instance_id'] for r in verified_rows}
with (RAW / 'openai_annotations' / 'extracted' / 'ensembled_annotations_public.csv').open(encoding='utf-8', newline='') as fh:
    annot = {r['instance_id']: r for r in csv.DictReader(fh)}

accept(len(ids) == 2294 and len(set(ids)) == 2294, f'SWE-bench test split: {len(ids)} instances (2,294 expected), {len(set(repo))} repositories')
accept(len(lite_ids) == 300 and lite_ids <= set(ids), f'Lite: {len(lite_ids)} instances, all in the test split: {lite_ids <= set(ids)}; repositories {len(set(repo[[pos[i] for i in lite_ids]]))}')
accept(len(verified_ids) == 500 and verified_ids <= set(ids), f'Verified: {len(verified_ids)} instances, all in the test split: {verified_ids <= set(ids)}')
accept(len(annot) == 1699 and set(annot) <= set(ids), f'OpenAI annotations: {len(annot)} instances, all in the test split: {set(annot) <= set(ids)}')
accept(verified_ids <= set(annot), f'Verified inside the annotated set: {len(verified_ids & set(annot))} of {len(verified_ids)}')
ACCEPT.append(f'INFO  Lite instances that are annotated: {len(lite_ids & set(annot))}; Lite and Verified share {len(lite_ids & verified_ids)} instances')

sub_dirs = sorted(p for p in (RAW / 'swebench_experiments' / 'test').iterdir() if p.is_dir())
sys_names, columns, outside = [], [], 0
for d in sub_dirs:
    data = json.loads((d / 'results.json').read_text(encoding='utf-8'))
    if not isinstance(data.get('resolved'), list):
        ACCEPT.append(f'INFO  dropped (no resolved list): {d.name}')
        continue
    y = np.zeros(len(ids), dtype=np.int8)
    for i in set(data['resolved']):
        if i in pos:
            y[pos[i]] = 1
        else:
            outside += 1
    sys_names.append(d.name)
    columns.append(y)
Y_A = np.array(columns)
accept(len(sys_names) == 24, f'SWE-bench systems with a resolved list on the full split: {len(sys_names)} (24 expected); resolved ids outside the split: {outside}')
pin = (RAW / 'swebench_experiments' / 'PINNED_COMMIT.txt').read_text(encoding='utf-8').strip()
ACCEPT.append(f'INFO  SWE-bench/experiments commit {pin}')

# ---------------------------------------------------------------------------------------------------------------
# Rule diagnosis A, and the regions of case A
# ---------------------------------------------------------------------------------------------------------------
flags = [lite_flags(r) for r in test_rows]
flag_names = list(flags[0])
F = np.array([[f[k] for k in flag_names] for f in flags])
certain_cols = [k != 'issue_ref_hash_number' for k in flag_names]
excluded_loose = F[:, certain_cols].any(axis=1)             # ignores a bare #n (the HTTP check would return "absent")
excluded_strict = F.any(axis=1)                             # treats every #n as an existing issue
in_lite = np.array([i in lite_ids for i in ids])
in_verified = np.array([i in verified_ids for i in ids])
in_annot = np.array([i in annot for i in ids])
E_strict, E_loose = ~excluded_strict, ~excluded_loose
contain_strict = float(np.mean(E_strict[in_lite]))
contain_loose = float(np.mean(E_loose[in_lite]))

say('# Released-subset study: results')
say()
say(f'Design: `experiments/SUBSET_C_DESIGN.md` (fixed 2026-11-14). SWE-bench/experiments at commit `{pin[:10]}`. '
    f'{B} bootstrap replicates and {K} random subsets per analysis. Scores are proportions (0 to 1).')
say()
say('## Rule diagnosis')
say()
say('### Case A: SWE-bench Lite')
say(f'- One stated filter cannot be recomputed: an instance with a bare `#n` is excluded only if an HTTP request for '
    f'that issue returned 200 when Lite was built. Two readings bracket it: every `#n` excludes (strict), or no bare '
    f'`#n` excludes (loose).')
say(f'- Instances passing the re-implemented filters: strict {int(E_strict.sum())}, loose {int(E_loose.sum())}, of {len(ids)}.')
say(f'- Lite instances inside the recomputed eligible set: strict {int(E_strict[in_lite].sum())} of 300 '
    f'({100 * contain_strict:.1f}%), loose {int(E_loose[in_lite].sum())} of 300 ({100 * contain_loose:.1f}%).')
alone = {k: int(np.sum(F[:, j] & ~np.delete(F, j, axis=1).any(axis=1))) for j, k in enumerate(flag_names)}
total = {k: int(F[:, j].sum()) for j, k in enumerate(flag_names)}
lite_hit = {k: int(F[in_lite, j].sum()) for j, k in enumerate(flag_names)}
say('- Instances each filter excludes (in total / alone / among the 300 Lite instances): ' + '; '.join(
    f'{k} {total[k]} / {alone[k]} / {lite_hit[k]}' for k in flag_names) + '.')
say(f'- W of the filters alone: strict {1 - E_strict.mean():.4f}, loose {1 - E_loose.mean():.4f}; W of the release: {1 - 300 / len(ids):.4f}.')
say()
rec('A', 'rule', 'strict', 'n_eligible', int(E_strict.sum()))
rec('A', 'rule', 'loose', 'n_eligible', int(E_loose.sum()))
rec('A', 'rule', 'strict', 'lite_containment', contain_strict)
rec('A', 'rule', 'loose', 'lite_containment', contain_loose)
for k in flag_names:
    rec('A', 'rule', k, 'n_excluded_total', total[k])
    rec('A', 'rule', k, 'n_excluded_alone', alone[k])
    rec('A', 'rule', k, 'n_lite_flagged', lite_hit[k])
faithful = contain_loose >= 0.95
accept(faithful, f'Lite inside the recomputed eligible set (loose reading) {100 * contain_loose:.1f}% (95% needed for the three-region analyses)')
E_LOOSE_A = E_loose | in_lite                              # design, Section 14, deviation 1

# ---------------------------------------------------------------------------------------------------------------
# Rule diagnosis B
# ---------------------------------------------------------------------------------------------------------------
annot_ids = [i for i in ids if i in annot]
recomputed = np.array([float(annot[i]['underspecified']) >= 2 or float(annot[i]['false_negative']) >= 2
                       or float(annot[i]['other_major_issues']) >= 1 for i in annot_ids])
given = np.array([annot[i]['filter_out'] == 'True' for i in annot_ids])
ver_annot = np.array([i in verified_ids for i in annot_ids])
difficulty = np.array([annot[i]['difficulty'] for i in annot_ids])
hard_labels = np.isin(difficulty, ['1-4 hours', '>4 hours'])
say('### Case B: SWE-bench Verified')
say(f'- Recomputed filter (either criterion 2 or above, or another major issue) agrees with the published `filter_out` '
    f'on {int(np.sum(recomputed == given))} of {len(annot_ids)} annotated instances; {int(given.sum())} are filtered out '
    f'({100 * given.mean():.1f}%) and {int((~given).sum())} pass.')
say(f'- Verified instances that pass the filter: {int(np.sum(ver_annot & ~given))} of {int(ver_annot.sum())}; '
    f'{int(ver_annot.sum())} of the {int((~given).sum())} passing instances were selected '
    f'({100 * ver_annot.sum() / (~given).sum():.1f}%).')
say(f'- Passing instances labelled 1-4 hours or >4 hours: {int(np.sum(~given & hard_labels))}, of which '
    f'{int(np.sum(~given & hard_labels & ver_annot))} are in Verified; passing instances not selected: '
    f'{int(np.sum(~given & ~ver_annot))}, of which {int(np.sum(~given & ~ver_annot & hard_labels))} carry those labels.')
say()
rec('B', 'rule', 'all', 'filter_agreement', int(np.sum(recomputed == given)))
rec('B', 'rule', 'all', 'n_filtered_out', int(given.sum()))
rec('B', 'rule', 'all', 'n_pass', int((~given).sum()))
rec('B', 'rule', 'all', 'verified_passing', int(np.sum(ver_annot & ~given)))

# ---------------------------------------------------------------------------------------------------------------
# Load and accept: BigCodeBench; rule diagnosis C
# ---------------------------------------------------------------------------------------------------------------
tasks = read_jsonl('bigcodebench_v0.1.4.jsonl')
task_ids = [t['task_id'] for t in tasks]
tpos = {t: k for k, t in enumerate(task_ids)}
hard_ids = {t['task_id'] for t in read_jsonl('bigcodebench_hard_v0.1.4.jsonl')}
in_hard = np.array([t in hard_ids for t in task_ids])
published = {r['model']: r for r in read_jsonl('bigcodebench_results.jsonl')}
accept(len(task_ids) == 1140 and len(hard_ids) == 148 and hard_ids <= set(task_ids),
       f'BigCodeBench: {len(task_ids)} tasks, Hard {len(hard_ids)}, Hard inside the task list: {hard_ids <= set(task_ids)}')
PERF = {}
for split in ('complete', 'instruct'):
    rows = read_jsonl(f'bigcodebench_perf_{split}.jsonl')
    cols = [c for c in rows[0] if c != 'Model']
    same_tasks = set(cols) == set(task_ids)
    cell_values = sorted({str(r[c]) for r in rows for c in cols})
    names, mat, dropped = [], [], []
    for r in rows:
        if any(str(r[c]) not in ('0', '1') for c in cols):
            dropped.append(r['Model'])
            continue
        names.append(r['Model'])
        mat.append([int(r[t]) for t in task_ids])
    Yc = np.array(mat, dtype=np.int8)
    matched = [(m, 100 * Yc[k].mean(), published[m][split]) for k, m in enumerate(names)
               if m in published and published[m][split] is not None]
    agree = sum(1 for _, mine, theirs in matched if abs(mine - theirs) <= 0.5)
    accept(same_tasks and cell_values == ['0', '1'], f'BigCodeBench {split}: {len(rows)} models, task columns equal the task list: '
           f'{same_tasks}; cell values {cell_values} (mapping fixed: "1" passed, "0" not); dropped for other values: {len(dropped)}')
    accept(len(matched) > 0 and agree >= 0.95 * len(matched), f'BigCodeBench {split}: table mean against the published full-set score, '
           f'{agree} of {len(matched)} matched models within 0.5 points ({len(names) - len(matched)} models not matched by name)')
    dates = np.array([published[m]['date'] if m in published and published[m].get('date') else '' for m in names])
    PERF[split] = (names, Yc, dates)

libs_n = np.array([len(ast.literal_eval(t['libs'])) for t in tasks])
sol_chars = np.array([len(t['canonical_solution']) for t in tasks])
sol_tokens = np.array([len(t['canonical_solution'].split()) for t in tasks])
rates = {}
for name in ('solve_rate_complete_20240618', 'solve_rate_instruct_20240618', 'solve_rate_complete', 'solve_rate_instruct'):
    table = {r['task_id']: float(r['solve_rate']) for r in read_jsonl(name + '.jsonl')}
    rates[name] = np.array([table[t] for t in task_ids])
say('### Case C: BigCodeBench-Hard')
say(f'- Hard tasks that satisfy each computable filter: more than two libraries {int(np.sum(libs_n[in_hard] > 2))} of 148; '
    f'solution longer than the mean in characters {int(np.sum(sol_chars[in_hard] > sol_chars.mean()))}, in whitespace tokens '
    f'{int(np.sum(sol_tokens[in_hard] > sol_tokens.mean()))}; ' + '; '.join(
        f'{k} below 50%: {int(np.sum(v[in_hard] < 50))}' for k, v in rates.items()) + '.')
rec('C', 'rule', 'hard', 'n_libs_gt_2', int(np.sum(libs_n[in_hard] > 2)))
rec('C', 'rule', 'hard', 'n_solution_longer_than_mean_chars', int(np.sum(sol_chars[in_hard] > sol_chars.mean())))
rec('C', 'rule', 'hard', 'n_solution_longer_than_mean_tokens', int(np.sum(sol_tokens[in_hard] > sol_tokens.mean())))
for k, v in rates.items():
    rec('C', 'rule', 'hard', f'n_{k}_below_50', int(np.sum(v[in_hard] < 50)))
for length_name, length in (('characters', sol_chars), ('whitespace tokens', sol_tokens)):
    for k in ('solve_rate_complete_20240618', 'solve_rate_instruct_20240618'):
        rule = (libs_n > 2) & (length > length.mean()) & (rates[k] < 50)
        jac = float(np.sum(rule & in_hard) / np.sum(rule | in_hard))
        say(f'- Computable part of the stated rule (libraries, length in {length_name}, {k}): {int(rule.sum())} tasks; '
            f'{int(np.sum(rule & in_hard))} of the 148 Hard tasks are among them; Jaccard with the release {jac:.3f}.')
        rec('C', 'rule', f'{length_name.replace(" ", "_")}/{k}', 'n_rule', int(rule.sum()))
        rec('C', 'rule', f'{length_name.replace(" ", "_")}/{k}', 'jaccard', jac)
say('- The Stack Overflow match (`q_idx`, `score`) is published for the 148 Hard tasks only, so the 626-task pool cannot be '
    'listed and the stated rule cannot be recomputed in full.')
say()

(OUT / 'acceptance.txt').write_text('\n'.join(ACCEPT) + '\n', encoding='utf-8')
if any(line.startswith('FAIL') for line in ACCEPT if 'eligible set' not in line and 'published full-set score' not in line):
    raise SystemExit('a blocking acceptance check failed; no analysis was run')

# ---------------------------------------------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------------------------------------------
say('## Case A: SWE-bench Lite (scope rule)')
say()
analyse('A', 'main', 'A, release against parent: Lite against the rest of the test split', Y_A, sys_names, in_lite,
        strata=repo, clusters=list(repo))
if faithful:
    analyse('A', 'control', 'A, negative control: Lite against eligible instances that were not sampled (strict reading)',
            Y_A[:, E_strict], sys_names, in_lite[E_strict], strata=repo[E_strict], clusters=list(repo[E_strict]))
    analyse('A', 'rule_alone_strict', 'A, the filters alone, strict reading: eligible instances against the rest',
            Y_A, sys_names, E_strict, strata=repo, clusters=list(repo))
    analyse('A', 'rule_alone_loose', 'A, the filters alone, loose reading: eligible instances against the rest',
            Y_A, sys_names, E_LOOSE_A, strata=repo, clusters=list(repo))
else:
    say('The three-region analyses are not available: the re-implemented filters do not contain 95% of Lite.')
    say()
diff_all = np.array([annot[i]['difficulty'] if i in annot else '' for i in ids])
matched_analysis('A', 'mechanism', 'A, mechanism check on annotated instances: excluded part matched to Lite on difficulty labels',
                 Y_A[:, in_annot], sys_names, in_lite[in_annot], diff_all[in_annot])

say('## Case B: SWE-bench Verified (validity filter, contrast)')
say()
say(f'Test instances without annotations (outside this case): {int((~in_annot).sum())}.')
say()
analyse('B', 'main', 'B, Verified against the other annotated instances', Y_A[:, in_annot], sys_names, in_verified[in_annot],
        strata=repo[in_annot], clusters=list(repo[in_annot]))
filtered = np.array([i in annot and annot[i]['filter_out'] == 'True' for i in ids])
sel = in_verified | filtered
analyse('B', 'fail_only', 'B, Verified against the annotated instances that fail the filter', Y_A[:, sel], sys_names,
        in_verified[sel], strata=repo[sel], clusters=list(repo[sel]))

say('## Case C: BigCodeBench-Hard (difficulty rule)')
say()
for split, label in (('complete', 'Complete, primary'), ('instruct', 'Instruct, replication')):
    names, Yc, dates = PERF[split]
    after, before = dates > HARD_RELEASE, (dates != '') & (dates <= HARD_RELEASE)
    say(f'{label}: {len(names)} models; dated on or before {HARD_RELEASE}: {int(before.sum())}; after: {int(after.sum())}; '
        f'undated or unmatched: {int((dates == "").sum())}.')
    say()
    rec('C', split, 'all', 'n_models_before_release', int(before.sum()))
    rec('C', split, 'all', 'n_models_after_release', int(after.sum()))
    analyse('C', split, f'C, {label}: Hard against the other tasks, all models', Yc, names, in_hard)
    if after.sum() >= 3:
        analyse('C', f'{split}_after', f'C, {label}: models dated after the Hard release only', Yc[after],
                [n for n, keep in zip(names, after) if keep], in_hard, write_pairs=False)
    if before.sum() >= 3:
        analyse('C', f'{split}_before', f'C, {label}: models dated on or before the Hard release only', Yc[before],
                [n for n, keep in zip(names, before) if keep], in_hard, write_pairs=False)

with (OUT / 'records.tsv').open('w', encoding='utf-8', newline='\n') as fh:
    fh.write('case\tanalysis\tsubject\tquantity\tvalue\n')
    for row in RECORDS:
        fh.write('\t'.join(row) + '\n')
(OUT / 'summary.md').write_text('\n'.join(SUMMARY) + '\n', encoding='utf-8')
print(f'wrote {len(RECORDS)} records')
