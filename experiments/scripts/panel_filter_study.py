"""Panel-dependence of a solve-rate difficulty filter (experiments/PANEL_FILTER_DESIGN.md).

Reads the BigCodeBench per-task tables fetched by subsets_fetch.py. No download, no training.

Usage (from experiments/):  python -E -P scripts/panel_filter_study.py results/subsets/raw results/panel_filter
"""
import hashlib
import json
import pathlib
import re
import sys

import numpy as np
from scipy.stats import rankdata

RAW, OUT = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
K_MAIN, DRAWS, PERMUTATIONS, BOOT = 148, 30, 2000, 300
RECORDS, SUMMARY = [], []
LINE_RULES = [
    ('gpt', r'^(GPT-|Quasar-Alpha$)'),
    ('claude', r'^Claude-'),
    ('gemini', r'^Gemini-'),
    ('gemma', r'^(Gemma-|CodeGemma-)'),
    ('codellama', r'^(CodeLlama-|Phind-CodeLlama|ReflectionCoder-CL-)'),
    ('llama', r'^(Llama-|Hermes-2-|OpenChat-3\.6|Athene-70B|DeepSeek-R1-Distill-Llama-)'),
    ('deepseek-coder', r'^(DeepSeek-Coder-(1\.3B|6\.7B|33B)-|Magicoder-S-DS|OpenCodeInterpreter-DS|AutoCoder$|AutoCoder-S-|'
                       r'Artigenz-Coder-DS|ReflectionCoder-DS-|WaveCoder-Ultra|WarriorCoder|WhiteRabbitNeo-33B)'),
    ('deepseek-v2plus', r'^(DeepSeek-Coder-V2|DeepSeek-V2|DeepSeek-V3)'),
    ('qwen-coder', r'^(CodeQwen|Qwen2\.5-Coder-|AutoCoder-QW|Nxcode-CQ)'),
    ('qwen', r'^(Qwen|QwQ-|Tess-|Athene-V2|Sky-T1|DeepSeek-R1-Distill-Qwen-|DeepCoder-)'),
    ('mistral', r'^(Mistral-|Mixtral-|Codestral-)'),
    ('phi', r'^Phi-'),
    ('granite', r'^Granite-'),
    ('yi', r'^Yi-'),
    ('starcoder2', r'^StarCoder2'),
    ('cohere', r'^(C4AI-Command|Command R)'),
    ('codegeex', r'^CodeGeex'),
    ('internlm', r'^InternLM'),
    ('opencoder', r'^OpenCoder-'),
]
ORG_MERGE = {'gemini': 'google', 'gemma': 'google', 'llama': 'meta', 'codellama': 'meta', 'deepseek-coder': 'deepseek',
             'deepseek-v2plus': 'deepseek', 'qwen': 'alibaba', 'qwen-coder': 'alibaba'}


def rng_for(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big'))


def rec(analysis, subject, quantity, value):
    if isinstance(value, (float, np.floating)):
        value = f'{float(value):.6g}'
    RECORDS.append((analysis, subject, quantity, str(value)))


def say(line=''):
    print(line, flush=True)
    SUMMARY.append(line)


def family_line(name):
    for family, pattern in LINE_RULES:
        if re.match(pattern, name):
            return family
    return 'unmatched:' + name


def family_lexical(name):
    return re.match(r'[A-Za-z]+', name).group(0).lower()


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def build(counts, n_builders, k, threshold):
    """The hard subset of a panel: the k tasks solved by fewest builders, or every task below a 50% solve rate."""
    if threshold:
        return np.flatnonzero(2 * counts < n_builders)
    return np.argsort(counts, kind='stable')[:k]


def score(rows, subset):
    return rows[..., subset].mean(axis=-1) if len(subset) else np.full(rows.shape[:-1], np.nan)


def make_draws(key, fam, with_pairs=True):
    """Random builder sets, fixed once per analysis and reused in every bootstrap replicate."""
    n = len(fam)
    families = sorted(set(fam))
    members = {f: np.flatnonzero(fam == f) for f in families}
    outside = {f: np.flatnonzero(fam != f) for f in families}
    model_draws = {}
    for f in families:
        for m in members[f]:
            r = rng_for(f'{key}/model/{m}')
            model_draws[m] = np.array([r.choice(outside[f], size=len(members[f]), replace=False) for _ in range(DRAWS)])
    pair_draws, skipped = {}, []
    if with_pairs:
        for i, fa in enumerate(families):
            for fb in families[i + 1:]:
                size = len(members[fa]) + len(members[fb])
                pool = np.flatnonzero((fam != fa) & (fam != fb))
                if len(pool) < size:
                    skipped.append((fa, fb))
                    continue
                r = rng_for(f'{key}/pair/{fa}/{fb}')
                pair_draws[(fa, fb)] = np.array([r.choice(pool, size=size, replace=False) for _ in range(DRAWS)])
    return families, members, model_draws, pair_draws, skipped, n


def evaluate(Y, draws, k, threshold, with_pairs=True):
    families, members, model_draws, pair_draws, _, n = draws
    colsum = Y.sum(axis=0)
    fam_sum = {f: Y[members[f]].sum(axis=0) for f in families}
    h_all = build(colsum, n, k, threshold)
    s_std = score(Y, h_all)
    s_fam, s_self, s_ref = np.empty(n), np.empty(n), np.empty(n)
    for f in families:
        size = len(members[f])
        h = build(colsum - fam_sum[f], n - size, k, threshold)
        s_fam[members[f]] = score(Y[members[f]], h)
        for m in members[f]:
            acc_self, acc_ref = 0.0, 0.0
            for others in model_draws[m]:
                partial = Y[others[:size - 1]].sum(axis=0) if size > 1 else 0
                acc_self += score(Y[m], build(colsum - partial - Y[m], n - size, k, threshold))
                acc_ref += score(Y[m], build(colsum - partial - Y[others[size - 1]], n - size, k, threshold))
            s_self[m], s_ref[m] = acc_self / DRAWS, acc_ref / DRAWS
    out = {'s_std': s_std, 's_fam': s_fam, 's_self': s_self, 's_ref': s_ref}
    sizes = np.array([len(members[f]) for f in families])
    size_of = np.empty(n)
    for f, sz in zip(families, sizes):
        size_of[members[f]] = sz
    multi = size_of > 1
    out['T_self'] = float(np.nanmean(s_self - s_ref))
    out['T_family'] = float(np.nanmean(s_fam - s_self))
    out['T_family_multi'] = float(np.nanmean((s_fam - s_self)[multi])) if multi.any() else float('nan')
    out['T_total'] = float(np.nanmean(s_fam - s_ref))
    out['own_family_minus_standard'] = float(np.nanmean(s_fam - s_std))
    out['spearman_model'] = spearman(s_fam - s_ref, size_of)
    if with_pairs:
        valid_n, flips_cf, flips_ref, shifts, size_diffs = 0, 0.0, 0.0, [], []
        for (fa, fb), sets in pair_draws.items():
            ia, ib = members[fa], members[fb]
            d_std = s_std[ia][:, None] - s_std[ib][None, :]
            valid = d_std != 0
            sc = score(Y, build(colsum - fam_sum[fa] - fam_sum[fb], n - len(ia) - len(ib), k, threshold))
            d_cf = sc[ia][:, None] - sc[ib][None, :]
            valid_n += int(valid.sum())
            flips_cf += float(np.sum(valid & (d_cf * d_std < 0)))
            for others in sets:
                sr = score(Y, build(colsum - Y[others].sum(axis=0), n - len(others), k, threshold))
                flips_ref += float(np.sum(valid & ((sr[ia][:, None] - sr[ib][None, :]) * d_std < 0))) / DRAWS
            shifts.append((d_cf - d_std).ravel())
            size_diffs.append(np.full(d_std.size, len(ia) - len(ib), dtype=float))
        out['n_cross_pairs'] = valid_n
        out['flip_share_cf'] = flips_cf / valid_n if valid_n else float('nan')
        out['flip_share_ref'] = flips_ref / valid_n if valid_n else float('nan')
        out['flip_share_excess'] = out['flip_share_cf'] - out['flip_share_ref']
        out['spearman_pair'] = spearman(np.concatenate(shifts), np.concatenate(size_diffs)) if shifts else float('nan')
    return out


def permutation_test(key, Y, fam, k, threshold):
    """Shuffle family labels over models (sizes kept); statistic: mean over models of s_fam - s_std."""
    n = len(fam)
    colsum = Y.sum(axis=0)
    s_std = score(Y, build(colsum, n, k, threshold))
    families = sorted(set(fam))
    sizes = [int(np.sum(fam == f)) for f in families]

    def statistic(order):
        total, start = 0.0, 0
        for size in sizes:
            idx = order[start:start + size]
            start += size
            h = build(colsum - Y[idx].sum(axis=0), n - size, k, threshold)
            total += float(np.nansum(score(Y[idx], h) - s_std[idx]))
        return total / n

    observed = statistic(np.concatenate([np.flatnonzero(fam == f) for f in families]))
    r = rng_for(f'{key}/permutation')
    null = np.array([statistic(r.permutation(n)) for _ in range(PERMUTATIONS)])
    return observed, float(null.mean()), float((1 + np.sum(null >= observed)) / (1 + PERMUTATIONS))


def verdict(estimate, lo, hi, unit=0.01):
    if lo > 0 or hi < 0:
        return 'material' if abs(estimate) >= unit else 'detectable but small'
    return 'not detected'


def run(analysis, title, names, Y, fam, k=K_MAIN, threshold=False, bootstrap=False, tables=False):
    key = f'panel_filter/{analysis}'
    draws = make_draws(key, fam, with_pairs=True)
    families, members, _, pair_draws, skipped, n = draws
    est = evaluate(Y, draws, k, threshold)
    observed, null_mean, p_value = permutation_test(key, Y, fam, k, threshold)
    sizes = sorted((len(members[f]) for f in families), reverse=True)
    h_all = build(Y.sum(axis=0), n, k, threshold)
    say(f'### {title}')
    say(f'- {n} models in {len(families)} families (sizes {", ".join(str(s) for s in sizes[:8])}{", ..." if len(sizes) > 8 else ""}; '
        f'{sum(1 for s in sizes if s == 1)} singletons); standard hard subset {len(h_all)} tasks; mean standard score '
        f'{np.nanmean(est["s_std"]):.4f}.')
    intervals = {}
    if bootstrap:
        r = rng_for(f'{key}/bootstrap')
        t = Y.shape[1]
        keys = ['T_self', 'T_family', 'T_family_multi', 'T_total', 'flip_share_excess', 'spearman_model', 'spearman_pair']
        reps = {q: [] for q in keys}
        for _ in range(BOOT):
            e = evaluate(Y[:, r.integers(0, t, t)], draws, k, threshold)
            for q in keys:
                reps[q].append(e[q])
        intervals = {q: tuple(np.nanpercentile(reps[q], [2.5, 97.5])) for q in keys}

    def fmt(q, points=True):
        scale = 100 if points else 1
        text = f'{scale * est[q]:+.3f}'
        if q in intervals:
            lo, hi = intervals[q]
            text += f' [{scale * lo:+.3f}, {scale * hi:+.3f}]'
            rec(analysis, 'all', q + '_lo', lo)
            rec(analysis, 'all', q + '_hi', hi)
        rec(analysis, 'all', q, est[q])
        return text

    unit = ' points' if not bootstrap else ' points (bootstrap 95% interval)'
    say(f'- Own participation, T_self: {fmt("T_self")}{unit}.')
    say(f'- Family dependence beyond own participation, T_family: {fmt("T_family")} points over all models; '
        f'{fmt("T_family_multi")} over models whose family has at least two members.')
    say(f'- Total, T_total: {fmt("T_total")} points.')
    say(f'- Permutation test of family structure ({PERMUTATIONS} shuffles): observed mean of (own family left out minus standard) '
        f'{100 * observed:+.3f} points, mean under shuffled families {100 * null_mean:+.3f}; p = {p_value:.4f}.')
    say(f'- Spearman over models between (s_fam - s_ref) and family size: {fmt("spearman_model", points=False)}.')
    say(f'- Cross-family pairs with a non-zero standard difference: {est["n_cross_pairs"]}; share whose sign changes when both '
        f'families are left out {100 * est["flip_share_cf"]:.2f}%, when the same number of unrelated models is left out '
        f'{100 * est["flip_share_ref"]:.2f}%; excess {fmt("flip_share_excess")} percentage points. Spearman between the shift '
        f'of the pair difference and the difference of family sizes: {fmt("spearman_pair", points=False)}.')
    if skipped:
        say(f'- Family pairs skipped because too few unrelated models remain: {len(skipped)}.')
    for q in ('flip_share_cf', 'flip_share_ref', 'n_cross_pairs'):
        rec(analysis, 'all', q, est[q])
    rec(analysis, 'all', 'permutation_observed', observed)
    rec(analysis, 'all', 'permutation_null_mean', null_mean)
    rec(analysis, 'all', 'permutation_p', p_value)
    rec(analysis, 'all', 'n_models', n)
    rec(analysis, 'all', 'n_families', len(families))
    if bootstrap:
        reading = {q: verdict(est[q], *intervals[q]) for q in ('T_self', 'T_family', 'T_total')}
        lo, hi = intervals['flip_share_excess']
        ranking = 'material' if (lo > 0 or hi < 0) and abs(est['flip_share_excess']) >= 0.01 else (
            'detectable but small' if (lo > 0 or hi < 0) else 'not detected')
        say(f'- **Reading fixed in the design:** own participation {reading["T_self"]}; family dependence {reading["T_family"]}; '
            f'total {reading["T_total"]}; ranking consequence {ranking}.')
        for q, v in reading.items():
            rec(analysis, 'all', q + '_reading', v)
        rec(analysis, 'all', 'ranking_reading', ranking)
    if tables:
        rows = []
        for f in families:
            idx = members[f]
            rows.append((f, len(idx), np.nanmean((est['s_fam'] - est['s_ref'])[idx]), np.nanmean((est['s_self'] - est['s_ref'])[idx]),
                         np.nanmean((est['s_fam'] - est['s_self'])[idx])))
        rows.sort(key=lambda x: -x[1])
        say('- By family (members; total, own participation, family dependence, in points): ' + '; '.join(
            f'{f} ({size}; {100 * a:+.2f}, {100 * b:+.2f}, {100 * c:+.2f})' for f, size, a, b, c in rows if size >= 3) + '.')
        with (OUT / f'models_{analysis}.tsv').open('w', encoding='utf-8', newline='\n') as fh:
            fh.write('model\tfamily\tfamily_size\ts_std\ts_fam\ts_self\ts_ref\n')
            for m in range(n):
                fh.write(f'{names[m]}\t{fam[m]}\t{int(np.sum(fam == fam[m]))}\t{est["s_std"][m]:.6f}\t{est["s_fam"][m]:.6f}\t'
                         f'{est["s_self"][m]:.6f}\t{est["s_ref"][m]:.6f}\n')
        with (OUT / f'families_{analysis}.tsv').open('w', encoding='utf-8', newline='\n') as fh:
            fh.write('family\tmembers\ttotal\town_participation\tfamily_dependence\n')
            for f, size, a, b, c in rows:
                fh.write(f'{f}\t{size}\t{a:.6f}\t{b:.6f}\t{c:.6f}\n')
    say()
    return est


def load(split):
    tasks = [json.loads(line)['task_id'] for line in (RAW / 'huggingface' / 'bigcodebench_v0.1.4.jsonl').open(encoding='utf-8')]
    rows = [json.loads(line) for line in (RAW / 'huggingface' / f'bigcodebench_perf_{split}.jsonl').open(encoding='utf-8')]
    names = [r['Model'] for r in rows]
    return names, np.array([[int(r[t]) for t in tasks] for r in rows], dtype=np.int64)


OUT.mkdir(parents=True, exist_ok=True)
say('# Panel dependence of a solve-rate difficulty filter: results')
say()
say(f'Design: `experiments/PANEL_FILTER_DESIGN.md` (fixed 2026-11-15). Hard subset = the k tasks solved by fewest builders. '
    f'{DRAWS} random builder sets per condition, {PERMUTATIONS} label shuffles, {BOOT} task-bootstrap replicates for the two '
    f'bootstrapped analyses. Scores are proportions; effects are given in points (1 point = 0.01). In each draw the '
    f'"self" and "reference" conditions share n - 1 removed models and differ in the last one (the model itself against one '
    f'more unrelated model).')
say()
say('Correction of 2026-11-16 (design, Section 13). The readings "ranking consequence" below are withdrawn: they rest on a bootstrap '
    'interval of the share of sign changes, and that bootstrap is biased. The shares stay as point estimates. Under the rule "solve '
    'rate below 50%" the same outcomes vote and score, so own participation (T_self) cannot be negative in any comparison; its '
    'interval says nothing about its sign, only about its size.')
say()
names_c, Y_c = load('complete')
names_i, Y_i = load('instruct')
fam_c = {d: np.array([fn(n) for n in names_c]) for d, fn in (('line', family_line), ('lexical', family_lexical))}
fam_c['organisation'] = np.array([ORG_MERGE.get(f, f) for f in fam_c['line']])
fam_i_line = np.array([family_line(n) for n in names_i])
unmatched = sorted({f for f in list(fam_c['line']) + list(fam_i_line) if f.startswith('unmatched:')})
say(f'Names that matched no family rule: {", ".join(unmatched) if unmatched else "none"}.')
say()
with (OUT / 'families.tsv').open('w', encoding='utf-8', newline='\n') as fh:
    fh.write('model\tsplits\tline\tlexical\torganisation\n')
    for name in sorted(set(names_c) | set(names_i), key=str.lower):
        line = family_line(name)
        fh.write(f'{name}\t{"C" if name in names_c else ""}{"I" if name in names_i else ""}\t{line}\t{family_lexical(name)}\t'
                 f'{ORG_MERGE.get(line, line)}\n')

if len(sys.argv) > 3 and sys.argv[3] == 'deviation':
    # Design, Section 11, deviation 1 (post hoc): the published rule as the main rule.
    say('## Deviation 1 (post hoc): the published rule, solve rate below 50%, as the main rule')
    say()
    run('complete_line_threshold', 'Complete, model-line families, solve rate below 50%', names_c, Y_c, fam_c['line'],
        threshold=True, bootstrap=True, tables=True)
    run('instruct_line_threshold', 'Instruct, model-line families, solve rate below 50%', names_i, Y_i, fam_i_line,
        threshold=True, bootstrap=True, tables=True)
    run('complete_lexical_threshold', 'Complete, lexical families, solve rate below 50%', names_c, Y_c, fam_c['lexical'],
        threshold=True)
    run('complete_organisation_threshold', 'Complete, organisation families, solve rate below 50%', names_c, Y_c,
        fam_c['organisation'], threshold=True)
    with (OUT / 'records_deviation.tsv').open('w', encoding='utf-8', newline='\n') as fh:
        fh.write('analysis\tsubject\tquantity\tvalue\n')
        for row in RECORDS:
            fh.write('\t'.join(row) + '\n')
    (OUT / 'summary_deviation.md').write_text('\n'.join(SUMMARY) + '\n', encoding='utf-8')
    raise SystemExit(0)

say('## Primary analysis')
say()
run('complete_line_k148', 'Complete, model-line families, k = 148 (primary)', names_c, Y_c, fam_c['line'], bootstrap=True, tables=True)
say('## Replication on Instruct (same tasks, 124 models in common)')
say()
run('instruct_line_k148', 'Instruct, model-line families, k = 148', names_i, Y_i, fam_i_line, bootstrap=True, tables=True)
say('## Sensitivity analyses (point estimates and permutation test)')
say()
run('complete_line_k74', 'Complete, model-line families, k = 74', names_c, Y_c, fam_c['line'], k=74)
run('complete_line_k296', 'Complete, model-line families, k = 296', names_c, Y_c, fam_c['line'], k=296)
run('complete_line_threshold', 'Complete, model-line families, solve rate below 50%', names_c, Y_c, fam_c['line'], threshold=True)
run('complete_lexical_k148', 'Complete, lexical families, k = 148', names_c, Y_c, fam_c['lexical'])
run('complete_organisation_k148', 'Complete, organisation families, k = 148', names_c, Y_c, fam_c['organisation'])
with (OUT / 'records.tsv').open('w', encoding='utf-8', newline='\n') as fh:
    fh.write('analysis\tsubject\tquantity\tvalue\n')
    for row in RECORDS:
        fh.write('\t'.join(row) + '\n')
(OUT / 'summary.md').write_text('\n'.join(SUMMARY) + '\n', encoding='utf-8')
print(f'wrote {len(RECORDS)} records')
