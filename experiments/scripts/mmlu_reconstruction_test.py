"""The MMLU-Pro filter on original MMLU: do the filter models score lower on the questions that were kept?
(experiments/MMLU_RECONSTRUCTION_DESIGN.md, with the deviations of its Section 8.)

Parquet and zip files are read in memory; nothing is extracted or executed.

Usage (from experiments/):
  python -E -P -X utf8 scripts/mmlu_reconstruction_test.py results/mmlu_reconstruction/raw results/mmlupro_filter/raw results/mmlu_reconstruction [gates]
  python -E -P -X utf8 scripts/mmlu_reconstruction_test.py --from-matrix MATRIX_DIR OUT
With "gates" only the data gates are run: question texts and row counts, no accuracy is compared.
With --from-matrix (added 2026-11-17) the run starts from MATRIX_DIR/correct_matrix.npz and the systems.tsv written
with it by a full run, skips the data gates (they passed before that matrix was written) and computes every
estimate again with the same code. Only the run labels of systems.tsv are taken from that systems.tsv.
"""
import collections
import hashlib
import itertools
import json
import pathlib
import sys
import zipfile

import numpy as np
import pyarrow.parquet as pq
from scipy import stats

if len(sys.argv) > 1 and sys.argv[1] == '--from-matrix':
    FROM_MATRIX, MATRIX_DIR, OUT = True, pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    RAW = PRO = None
    GATES_ONLY = False
else:
    FROM_MATRIX, MATRIX_DIR = False, None
    RAW, PRO, OUT = (pathlib.Path(p) for p in sys.argv[1:4])
    GATES_ONLY = len(sys.argv) > 4 and sys.argv[4] == 'gates'
FILTER = ['meta-llama__Llama-2-7b-hf', 'meta-llama__Llama-2-7b-chat-hf', 'meta-llama__Llama-2-13b-hf',
          'meta-llama__Llama-2-13b-chat-hf', 'mistralai__Mistral-7B-v0.1', 'google__gemma-7b', '01-ai__Yi-6B']
WINDOW, MIN_CONTROLS, MIN_KEPT, MATCH_GATE, BOOT, MAX_SUBSETS = 0.03, 5, 10, 0.95, 1000, 200000
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


def norm(text):
    return ' '.join(str(text).lower().split())


def line_of(name):
    return 'llama2' if 'llama-2' in name.lower() else name


OUT.mkdir(parents=True, exist_ok=True)

if not FROM_MATRIX:
    # ---------------------------------------------------------------------------------------------------------------
    # Questions of original MMLU
    # ---------------------------------------------------------------------------------------------------------------
    mmlu = pq.read_table(RAW / 'cais_mmlu_all_test.parquet').to_pylist()
    Q = len(mmlu)
    subjects = sorted({row['subject'] for row in mmlu})
    S = len(subjects)
    subject_index = {name: s for s, name in enumerate(subjects)}
    subject_of = np.array([subject_index[row['subject']] for row in mmlu])
    cols = [np.flatnonzero(subject_of == s) for s in range(S)]
    qnorm = [norm(row['question']) for row in mmlu]


    def expected(row):
        return row['question'].strip() + ''.join(f'\n{letter}. {c}' for letter, c in zip('ABCD', row['choices'])) + '\nAnswer:'


    exact = [collections.defaultdict(list) for _ in range(S)]
    loose = [collections.defaultdict(list) for _ in range(S)]
    for q, row in enumerate(mmlu):
        exact[subject_of[q]][expected(row)].append(q)
        loose[subject_of[q]][norm(expected(row))].append(q)
    accept(Q == 14042 and S == 57, f'cais/mmlu test: {Q} questions in {S} subjects')

    # ---------------------------------------------------------------------------------------------------------------
    # Texts of MMLU-Pro's MMLU-origin questions, from the output archives (most frequent text per question id)
    # ---------------------------------------------------------------------------------------------------------------
    cache = OUT / 'mmlupro_origin_texts.json'
    if cache.exists():
        pro_text = json.loads(cache.read_text(encoding='utf-8'))
    else:
        with (PRO / 'mmlu_pro_test_questions.jsonl').open(encoding='utf-8') as fh:
            table = {}
            for text_line in fh:
                item = json.loads(text_line)
                if item['src'].startswith('ori_mmlu'):
                    table[item['question_id']] = item['src']
        votes = {qid: collections.Counter() for qid in table}
        for path in sorted((PRO / 'eval_results').glob('*.zip')):
            with zipfile.ZipFile(path) as zf:
                members = [i.filename for i in zf.infolist() if i.filename.endswith('.json') and '__MACOSX' not in i.filename
                           and not i.filename.endswith('summary.json')]
                for member in members:
                    with zf.open(member) as fh:
                        data = json.load(fh)
                    if not isinstance(data, list):
                        continue
                    for r in data:
                        if isinstance(r, dict) and r.get('question_id') in table and r.get('src') == table[r['question_id']] \
                                and isinstance(r.get('question'), str):
                            votes[r['question_id']][norm(r['question'])] += 1
            print(f'   read texts from {path.name}', flush=True)
        pro_text = {str(qid): {'src': table[qid], 'text': votes[qid].most_common(1)[0][0] if votes[qid] else None,
                               'variants': len(votes[qid])} for qid in table}
        cache.write_text(json.dumps(pro_text), encoding='utf-8')

    orig_texts = [set(qnorm[q] for q in cols[s]) for s in range(S)]
    pro_texts = [set() for _ in range(S)]
    by_subject = collections.defaultdict(lambda: [0, 0])
    n_ids = len(pro_text)
    with_text = matched = unknown_subject = 0
    for item in pro_text.values():
        name = item['src'].split('-', 1)[1]
        if name not in subject_index:
            unknown_subject += 1
            continue
        s = subject_index[name]
        by_subject[name][0] += 1
        if item['text'] is None:
            continue
        with_text += 1
        pro_texts[s].add(item['text'])
        if item['text'] in orig_texts[s]:
            matched += 1
            by_subject[name][1] += 1
    kept = np.array([qnorm[q] in pro_texts[subject_of[q]] for q in range(Q)])
    n_kept = np.array([int(kept[cols[s]].sum()) for s in range(S)])
    variants = sum(item['variants'] > 1 for item in pro_text.values())
    accept(None, f'MMLU-Pro questions of MMLU origin: {n_ids}; with a text in the output archives: {with_text}; with more than one '
           f'text across archives: {variants}; with a subject unknown to cais/mmlu: {unknown_subject}')
    match_gate = matched / n_ids >= MATCH_GATE
    accept(match_gate, f'text match gate: {matched} of {n_ids} MMLU-origin questions of MMLU-Pro ({100 * matched / n_ids:.2f}%) equal an '
           f'original question of the same subject after lower-casing and collapsing whitespace (gate: at least {100 * MATCH_GATE:.0f}%)')
    worst = sorted(by_subject.items(), key=lambda kv: kv[1][1] / kv[1][0])[:5]
    accept(None, 'subjects with the lowest match rate: ' + '; '.join(f'{name} {m}/{n}' for name, (n, m) in worst))
    shared = collections.Counter((subject_of[q], qnorm[q]) for q in range(Q))
    kept_shared = sum(1 for q in range(Q) if kept[q] and shared[(subject_of[q], qnorm[q])] > 1)
    accept(None, f'original questions marked kept: {int(kept.sum())} of {Q} ({100 * kept.mean():.1f}%); kept rows whose question text '
           f'occurs more than once in the subject: {kept_shared}; subjects without a kept question: '
           + (', '.join(subjects[s] for s in range(S) if n_kept[s] == 0) or 'none'))

    # ---------------------------------------------------------------------------------------------------------------
    # Per-question results (latest complete run that passes the row gate; design, Section 8, deviations 2 and 3)
    # ---------------------------------------------------------------------------------------------------------------
    PERM = {}


    def mapping(s, examples):
        used = collections.Counter()
        perm = np.empty(len(examples), dtype=int)
        for k, e in enumerate(examples):
            key, rows = e, exact[s].get(e)
            if rows is None:
                key = norm(e)
                rows = loose[s].get(key)
                if rows is None:
                    return None
            if used[key] >= len(rows):
                return None
            perm[k] = rows[used[key]]
            used[key] += 1
        return perm if len(set(perm.tolist())) == len(perm) else None


    def load_run(folder):
        y = np.full(Q, np.nan)
        for s, subject in enumerate(subjects):
            path = folder / f'{subject}.parquet'
            if not path.exists():
                return None, f'{subject}: no file'
            pf = pq.ParquetFile(path)
            fields = pf.schema_arrow.names
            if 'acc' in fields:
                tab = pf.read(columns=['example', 'acc'])
                acc = tab.column('acc').to_pylist()
            elif 'metrics' in fields:
                tab = pf.read(columns=['example', 'metrics'])
                acc = [m.get('acc') if m else None for m in tab.column('metrics').to_pylist()]
            else:
                return None, f'{subject}: no accuracy column'
            examples = tab.column('example').to_pylist()
            if len(examples) != len(cols[s]):
                return None, f'{subject}: {len(examples)} rows, expected {len(cols[s])}'
            key = (s, tuple(examples))
            perm = PERM.get(key)
            if perm is None:
                perm = mapping(s, examples)
                if perm is None:
                    return None, f'{subject}: rows do not match cais/mmlu'
                PERM[key] = perm
            try:
                values = np.array([float(a) for a in acc])
            except (TypeError, ValueError):
                return None, f'{subject}: unreadable accuracy'
            if not np.all((values == 0) | (values == 1)):
                return None, f'{subject}: accuracy not 0 or 1'
            y[perm] = values
        return y, 'ok'


    def candidates(repo):
        found = []
        main = RAW / 'details' / repo
        if (main / 'run.json').exists():
            meta = json.loads((main / 'run.json').read_text(encoding='utf-8'))
            if meta.get('status') == 'ok':
                found.append((meta['run'], main))
        alt = RAW / 'details_alt' / repo
        if alt.exists():
            for folder in sorted((p for p in alt.iterdir() if (p / 'run.json').exists()), key=lambda p: p.name, reverse=True):
                found.append((folder.name, folder))
        return found


    selection = []
    with (RAW / 'selection.tsv').open(encoding='utf-8') as fh:
        next(fh)
        for text_line in fh:
            repo, kind, tag, present = text_line.rstrip('\n').split('\t')
            selection.append((repo, kind, tag, present == '1'))
    names, sets, tags, runs, rows, dropped, alternates = [], [], [], [], [], [], []
    for k, (repo, kind, tag, present) in enumerate(selection):
        if not present:
            continue
        reasons = []
        options = candidates(repo)
        for n_try, (run, folder) in enumerate(options):
            y, why = load_run(folder)
            if y is not None:
                names.append(repo[len('details_'):])
                sets.append(kind)
                tags.append(tag)
                runs.append(run)
                rows.append(y)
                if n_try:
                    alternates.append(f'{repo[len("details_"):]} (run {run}; later run rejected: {reasons[0]})')
                break
            reasons.append(f'{run}: {why}')
        else:
            dropped.append((repo[len('details_'):], kind, '; '.join(reasons) or 'no MMLU records'))
        if (k + 1) % 40 == 0:
            print(f'   loaded {k + 1} of {len(selection)} selection lines', flush=True)
    Y = np.array(rows)
    sets, tags = np.array(sets), np.array(tags)
    N = len(names)
    index = {name: j for j, name in enumerate(names)}
    for kind in ('filter', 'clean', 'lineage', 'random'):
        chosen = sum(1 for s in selection if s[1] == kind and s[3])
        accept(None, f'{kind}: {int(np.sum(sets == kind))} of {chosen} selected repositories loaded'
               + (f' ({int(np.sum((sets == "clean") & (tags == "sibling")))} siblings)' if kind == 'clean' else ''))
    accept(None, 'dropped: ' + ('; '.join(f'{n} [{kind}: {why[:70]}]' for n, kind, why in dropped) or 'none'))
    accept(None, 'earlier run used: ' + ('; '.join(alternates) or 'none'))
    idx_f = [index[f] for f in FILTER if f in index]
    accept(len(idx_f) >= 3, f'filter models with per-question MMLU records: {len(idx_f)} of 8 ('
           + ', '.join(names[j] for j in idx_f) + '); without: ' + ', '.join(f for f in FILTER if f not in index) + ', 01-ai__Yi-6B-Chat')
    accept(not np.isnan(Y).any(), 'every loaded system has all 14,042 questions')
    (OUT / 'acceptance.txt').write_text('\n'.join(ACCEPT) + '\n', encoding='utf-8')
    if GATES_ONLY:
        print('gates only: stopping before any accuracy is compared')
        raise SystemExit(0)
    if not match_gate:
        say('# MMLU reconstruction study: stopped at the text-match gate')
        say()
        say('Fewer than 95% of MMLU-Pro\'s MMLU-origin questions equal an original question, so the kept set is unreliable and, as fixed '
            'in the design, no accuracy is compared. See acceptance.txt.')
        (OUT / 'summary.md').write_text('\n'.join(SUMMARY) + '\n', encoding='utf-8')
        raise SystemExit(0)
    np.savez_compressed(OUT / 'correct_matrix.npz', Y=Y.astype(np.uint8), systems=np.array(names), sets=sets, tags=tags, kept=kept,
                        subject=np.array([subjects[s] for s in subject_of]))
else:
    # Matrix entry: the matrix and its systems.tsv come from a full run; every estimate below is computed again.
    with np.load(MATRIX_DIR / 'correct_matrix.npz', allow_pickle=False) as data:
        Y = data['Y'].astype(np.float64)
        names = [str(x) for x in data['systems']]
        sets, tags, kept = data['sets'], data['tags'], data['kept']
        subject_names = [str(x) for x in data['subject']]
    Q = len(kept)
    subjects = sorted(set(subject_names))
    S = len(subjects)
    subject_index = {name: s for s, name in enumerate(subjects)}
    subject_of = np.array([subject_index[name] for name in subject_names])
    cols = [np.flatnonzero(subject_of == s) for s in range(S)]
    n_kept = np.array([int(kept[cols[s]].sum()) for s in range(S)])
    N = len(names)
    index = {name: j for j, name in enumerate(names)}
    idx_f = [index[f] for f in FILTER if f in index]
    run_of = {}
    with (MATRIX_DIR / 'systems.tsv').open(encoding='utf-8') as fh:
        next(fh)
        for text_line in fh:
            fields = text_line.rstrip('\n').split('\t')
            run_of[fields[0]] = fields[3]
    if sorted(run_of) != sorted(names):
        raise SystemExit('the systems.tsv next to the matrix does not list the systems of the matrix')
    runs = [run_of[name] for name in names]
    matrix_sha = hashlib.sha256((MATRIX_DIR / 'correct_matrix.npz').read_bytes()).hexdigest()
    accept(None, f'started from correct_matrix.npz (sha256 {matrix_sha}); the data gates were not rerun')

# ---------------------------------------------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------------------------------------------
used = [s for s in range(S) if n_kept[s] >= MIN_KEPT]
omega = n_kept[used] / n_kept[used].sum()
a_all = sum(w * Y[:, cols[s]].mean(axis=1) for w, s in zip(omega, used))
a_kept = sum(w * Y[:, cols[s][kept[cols[s]]]].mean(axis=1) for w, s in zip(omega, used))
p_all, p_kept = Y.mean(axis=1), Y[:, kept].mean(axis=1)
leaderboard = np.mean([Y[:, cols[s]].mean(axis=1) for s in range(S)], axis=0)

r = rng_for('mmlu_reconstruction/bootstrap')
W = np.zeros((BOOT, Q))
for s in range(S):
    W[:, cols[s]] = r.multinomial(len(cols[s]), np.full(len(cols[s]), 1.0 / len(cols[s])), size=BOOT)
num_std, den_std, kept_std = np.zeros((BOOT, N)), np.zeros(BOOT), np.zeros((BOOT, N))
B_p_all, B_p_kept, den_pool = np.zeros((BOOT, N)), np.zeros((BOOT, N)), np.zeros(BOOT)
for s in range(S):
    Ws, Ys = W[:, cols[s]], Y[:, cols[s]]
    k_s = kept[cols[s]].astype(float)
    all_s = Ws @ Ys.T
    kept_num = (Ws * k_s) @ Ys.T
    kept_den = Ws @ k_s
    B_p_all += all_s / Q
    B_p_kept += kept_num
    den_pool += kept_den
    if s in used:
        num_std += kept_den[:, None] * all_s / len(cols[s])
        kept_std += kept_num
        den_std += kept_den
B_all, B_kept = num_std / den_std[:, None], kept_std / den_std[:, None]
B_p_kept /= den_pool[:, None]


def control_set(acc_all, focal, allowed):
    cand = np.array([j for j in allowed if j != focal])
    dist = np.abs(acc_all[cand] - acc_all[focal])
    near = cand[dist <= WINDOW]
    return near if len(near) >= MIN_CONTROLS else cand[np.argsort(dist, kind='stable')[:MIN_CONTROLS]]


def penalty(acc_kept, acc_all, focal, controls):
    return (acc_kept[controls].mean() - acc_kept[focal]) - (acc_all[controls].mean() - acc_all[focal])


def subset_p(values, size, observed, key):
    values = np.asarray(values)
    if len(values) < size:
        return float('nan'), 0, 'pool smaller than the filter set'
    count = 1
    for k in range(1, size + 1):
        count = count * (len(values) - k + 1) // k
    if count <= MAX_SUBSETS:
        means = np.array([values[list(c)].mean() for c in itertools.combinations(range(len(values)), size)])
        return float(np.mean(means >= observed)), count, 'exact'
    gen = rng_for(key)
    means = np.array([values[gen.choice(len(values), size=size, replace=False)].mean() for _ in range(MAX_SUBSETS)])
    return float(np.mean(means >= observed)), MAX_SUBSETS, 'random subsets'


def analysis(tag, title, acc_kept, acc_all, allowed, boot):
    lo, hi = acc_all[idx_f].min() - WINDOW, acc_all[idx_f].max() + WINDOW
    pool = [j for j in allowed if lo <= acc_all[j] <= hi]
    controls = {m: control_set(acc_all, m, allowed) for m in idx_f}
    p_m = np.array([penalty(acc_kept, acc_all, m, controls[m]) for m in idx_f])
    placebo = np.array([penalty(acc_kept, acc_all, q, control_set(acc_all, q, allowed)) for q in pool])
    p_value, n_subsets, how = subset_p(placebo, len(idx_f), p_m.mean(), f'mmlu_reconstruction/{tag}/subsets')
    bk, ba = boot
    reps = np.array([(bk[:, controls[m]].mean(axis=1) - bk[:, m]) - (ba[:, controls[m]].mean(axis=1) - ba[:, m]) for m in idx_f]).T
    b_lo, b_hi = np.percentile(reps.mean(axis=1), [2.5, 97.5])
    say(f'### {title}')
    say(f'- Filter models: a(all) from {100 * acc_all[idx_f].min():.1f}% to {100 * acc_all[idx_f].max():.1f}%; controls per filter '
        f'model: ' + ', '.join(str(len(controls[m])) for m in idx_f) + f'; placebo pool: {len(pool)} systems.')
    say(f'- **Penalty P = {100 * p_m.mean():+.2f} points** (question bootstrap 95% interval {100 * b_lo:+.2f} to {100 * b_hi:+.2f}); '
        f'placebo penalties of the pool: mean {100 * placebo.mean():+.2f}, SD {100 * placebo.std(ddof=1):.2f}, range '
        f'{100 * placebo.min():+.2f} to {100 * placebo.max():+.2f}; **reference tail share: {round(p_value * n_subsets)} of {n_subsets} '
        f'subsets of the pool ({p_value:.5f}; {how})**.')
    cells = []
    for k, m in enumerate(idx_f):
        lo_k, hi_k = np.percentile(reps[:, k], [2.5, 97.5])
        cells.append(f'{names[m]} {100 * p_m[k]:+.2f} [{100 * lo_k:+.2f}, {100 * hi_k:+.2f}]')
        rec(tag, names[m], 'penalty', p_m[k])
        rec(tag, names[m], 'penalty_lo', lo_k)
        rec(tag, names[m], 'penalty_hi', hi_k)
        rec(tag, names[m], 'n_controls', len(controls[m]))
    say('- By filter model: ' + '; '.join(cells) + '.')
    sib = [(names[q], placebo[k]) for k, q in enumerate(pool) if tags[q] == 'sibling']
    if sib:
        say('- Siblings in the pool (placebo penalties): ' + '; '.join(f'{n} {100 * v:+.2f}' for n, v in sib) + '.')
    say(f'- Pool systems with a placebo penalty at least as large as P: {int(np.sum(placebo >= p_m.mean()))} of {len(pool)}; '
        f'filter models whose penalty exceeds every placebo penalty: {int(np.sum(p_m > placebo.max()))} of {len(idx_f)}.')
    say()
    for quantity, value in (('P', p_m.mean()), ('P_lo', b_lo), ('P_hi', b_hi), ('placebo_p', p_value), ('pool_size', len(pool)),
                            ('placebo_mean', placebo.mean()), ('placebo_sd', placebo.std(ddof=1)), ('placebo_max', placebo.max())):
        rec(tag, 'all', quantity, value)
    for k, q in enumerate(pool):
        rec(tag, names[q], 'placebo_penalty', placebo[k])
    return {'P': float(p_m.mean()), 'p': p_value, 'ci': (float(b_lo), float(b_hi)), 'pool': pool, 'placebo': placebo, 'p_m': p_m}


# ---------------------------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------------------------
clean = [j for j in range(N) if sets[j] == 'clean']
clean_ns = [j for j in clean if tags[j] != 'sibling']
random_pool = [j for j in range(N) if sets[j] == 'random']
lineage = [j for j in range(N) if sets[j] == 'lineage']
accept(None, 'mean of the 57 subject accuracies (the old leaderboard\'s MMLU score) of the filter models: '
       + ', '.join(f'{names[j]} {100 * leaderboard[j]:.1f}%' for j in idx_f))
accept(None, f'subjects used (at least {MIN_KEPT} kept questions): {len(used)} of {S}, holding {int(n_kept[used].sum())} kept and '
       f'{int(sum(len(cols[s]) for s in used))} original questions')
(OUT / 'acceptance.txt').write_text('\n'.join(ACCEPT) + '\n', encoding='utf-8')

say('# MMLU reconstruction study: results')
say()
say(f'Design: `experiments/MMLU_RECONSTRUCTION_DESIGN.md` (fixed 2026-11-15; deviations in its Section 8). Per-question results on original '
    f'MMLU (four options, five-shot) from the archive `open-llm-leaderboard-old`: {N} systems, of which {len(idx_f)} of the eight filter '
    f'models, {len(clean)} of the clean pool ({len(clean) - len(clean_ns)} siblings), {len(lineage)} of the lineage pool and '
    f'{len(random_pool)} of the random pool. An original question is *kept* if MMLU-Pro holds a question with the same text and subject: '
    f'{int(kept.sum())} of {Q}. a(all) and a(kept) are a system\'s accuracy over all and over the kept original questions, standardised '
    f'to the kept questions\' subject mix ({len(used)} subjects). The penalty of a filter model is how much lower it scores on the kept '
    f'questions than clean-pool systems with the same a(all). 1 point = 0.01.')
say()
say('Corrections of 2026-11-16 (design, Section 12; wording only, no estimate changed). The reference tail share is the share of '
    'subsets of the placebo pool, of the size of the filter set, whose mean is at least the observed value. It says where that value '
    'lies among those subsets. It is not a calibrated p-value: the filter models are not exchangeable with the pool systems. The '
    'readings below are the labels that the rule fixed in the design gives. In the leave-one-out of secondary analysis 4 the same '
    'outcomes vote and score, so the penalty of voting cannot be negative in any comparison; only its size is a result.')
say()
say('## Primary analysis')
say()
main = analysis('primary', 'Clean pool as controls, siblings included', a_kept, a_all, clean, (B_kept, B_all))
passes = main['p'] < 0.05 and main['ci'][0] > 0
reading = 'confirmed' if passes and main['P'] >= 0.01 else ('detectable but small' if passes else 'not detected')
say(f'**Reading fixed in the design: {reading}.**')
say()
rec('primary', 'all', 'reading', reading)

say('## Sensitivity analyses (fixed)')
say()
no_sib = analysis('no_siblings', '1. Siblings removed from the clean pool', a_kept, a_all, clean_ns, (B_kept, B_all))
analysis('random_pool', '2. Random pool in place of the clean pool', a_kept, a_all, random_pool, (B_kept, B_all))
analysis('pooled', '3. No subject standardisation (all 57 subjects pooled)', p_kept, p_all, clean, (B_p_kept, B_p_all))

say('## Secondary analyses')
say()
# 1. fidelity
n_f = len(idx_f)
votes_f = Y[idx_f].sum(axis=0).astype(int)
panel, taken = [], set()
for m in idx_f:
    order = sorted((j for j in clean if j not in taken), key=lambda j: abs(a_all[j] - a_all[m]))
    panel.append(order[0])
    taken.add(order[0])
votes_c = Y[panel].sum(axis=0).astype(int)


def auc_table(not_kept_counts, kept_counts):
    """P(votes on a not-kept question > votes on a kept one) + half the ties, from counts by number of votes."""
    total = 0.0
    for v in range(len(kept_counts)):
        total += not_kept_counts[v] * (kept_counts[:v].sum() + 0.5 * kept_counts[v])
    return total / (not_kept_counts.sum() * kept_counts.sum())


def counts(votes, weights=None):
    w = np.ones(Q) if weights is None else weights
    return (np.array([np.sum(w[(votes == v) & ~kept]) for v in range(n_f + 1)]),
            np.array([np.sum(w[(votes == v) & kept]) for v in range(n_f + 1)]))


auc_f, auc_c = auc_table(*counts(votes_f)), auc_table(*counts(votes_c))
ind = {(tagname, v, flag): ((votes == v) & (kept if flag else ~kept)).astype(float)
       for tagname, votes in (('f', votes_f), ('c', votes_c)) for v in range(n_f + 1) for flag in (0, 1)}
boot_counts = {key: W @ vec for key, vec in ind.items()}
diff = []
for b in range(BOOT):
    pair = []
    for tagname in ('f', 'c'):
        nk = np.array([boot_counts[(tagname, v, 0)][b] for v in range(n_f + 1)])
        kk = np.array([boot_counts[(tagname, v, 1)][b] for v in range(n_f + 1)])
        pair.append(auc_table(nk, kk))
    diff.append(pair[0] - pair[1])
d_lo, d_hi = np.percentile(diff, [2.5, 97.5])
say('### 1. Fidelity of the public runs')
say(f'- c = number of the {n_f} available filter models that are right on a question. Under the published rule (removed if more than '
    f'four of eight are right) a question with c = {n_f} was removed whatever the missing models did, and a question with c ≤ 1 was '
    f'kept by the filter. The matched clean panel is an addition to the design: ' + ', '.join(names[j] for j in panel) + '.')
say()
say('| Number right | Questions (filter models) | Share kept | Questions (matched clean panel) | Share kept |')
say('|---:|---:|---:|---:|---:|')
for v in range(n_f + 1):
    sel_f, sel_c = votes_f == v, votes_c == v
    say(f'| {v} | {int(sel_f.sum())} | {100 * kept[sel_f].mean():.1f}% | {int(sel_c.sum())} | {100 * kept[sel_c].mean():.1f}% |')
    rec('fidelity', f'c={v}', 'questions_filter', int(sel_f.sum()))
    rec('fidelity', f'c={v}', 'share_kept_filter', float(kept[sel_f].mean()))
    rec('fidelity', f'c={v}', 'questions_matched', int(sel_c.sum()))
    rec('fidelity', f'c={v}', 'share_kept_matched', float(kept[sel_c].mean()))
say()
say(f'- How well the number right separates not-kept from kept questions (area under the curve; 0.5 = no information): filter models '
    f'{auc_f:.3f}, matched clean panel {auc_c:.3f}; difference {auc_f - auc_c:+.3f} (question bootstrap 95% interval {d_lo:+.3f} to '
    f'{d_hi:+.3f}).')
say(f'- Share kept among questions that all {n_f} available filter models answer correctly: {100 * kept[votes_f == n_f].mean():.1f}% '
    f'(0% if the public runs equalled the filter\'s runs); among questions with c ≤ 1: {100 * kept[votes_f <= 1].mean():.1f}%.')
say()
rec('fidelity', 'all', 'auc_filter', auc_f)
rec('fidelity', 'all', 'auc_matched', auc_c)
rec('fidelity', 'all', 'auc_difference_lo', d_lo)
rec('fidelity', 'all', 'auc_difference_hi', d_hi)

# 2. lineage
lo, hi = a_all[idx_f].min() - WINDOW, a_all[idx_f].max() + WINDOW
ref = no_sib['placebo']
say('### 2. Lineage')
say(f'- Penalties of the lineage-pool systems (fine-tunes and copies named after a filter model), each against clean-pool controls '
    f'without siblings. Reference: the {len(ref)} placebo penalties of that pool (mean {100 * ref.mean():+.2f}, SD '
    f'{100 * ref.std(ddof=1):.2f}, maximum {100 * ref.max():+.2f}). Only lineage systems inside the filter models\' range of a(all), '
    f'widened by 0.03, are counted. A line is a model line read from names. The one-sided Mann–Whitney test is an addition to the '
    f'design. Fine-tunes of one base are not independent of each other, so its p-value has no independent-sample reading; it only '
    f'describes how far the two sets of numbers are apart.')
say()
say('| Line | Systems in range (outside) | Mean penalty | Median | Share above zero | Share above the pool maximum | Mann–Whitney p (descriptive) |')
say('|---|---:|---:|---:|---:|---:|---:|')
lineage_pen = {j: penalty(a_kept, a_all, j, control_set(a_all, j, clean_ns)) for j in lineage}
for label in sorted(set(tags[lineage])) + ['all lines']:
    members = [j for j in lineage if label == 'all lines' or tags[j] == label]
    inside = [j for j in members if lo <= a_all[j] <= hi]
    if not inside:
        say(f'| {label} | 0 ({len(members)}) | | | | | |')
        continue
    values = np.array([lineage_pen[j] for j in inside])
    p_mw = stats.mannwhitneyu(values, ref, alternative='greater').pvalue if len(values) >= 3 else float('nan')
    say(f'| {label} | {len(inside)} ({len(members) - len(inside)}) | {100 * values.mean():+.2f} | {100 * np.median(values):+.2f} | '
        f'{100 * np.mean(values > 0):.0f}% | {100 * np.mean(values > ref.max()):.0f}% | {p_mw:.4g} |')
    rec('lineage', label, 'n_in_range', len(inside))
    rec('lineage', label, 'mean_penalty', values.mean())
    rec('lineage', label, 'median_penalty', float(np.median(values)))
    rec('lineage', label, 'mann_whitney_p', p_mw)
say()
for j in lineage:
    rec('lineage', names[j], 'penalty', lineage_pen[j])

# 4. leave-one-out on the real panel
threshold = n_f // 2
kept0 = votes_f <= threshold
say('### 4. Leave-one-out on the panel of available filter models')
say(f'- Rule: a question is removed if more than half of the {n_f} models are right, so it is kept if at most {threshold} are. The '
    f'panel\'s own kept set holds {int(kept0.sum())} of {Q} questions ({100 * kept0.mean():.1f}%). For each member, it is replaced by '
    f'the clean-pool system nearest in a(all); the reference replaces another member, not of its line, by the same system. Plain '
    f'accuracy over all subjects.')
say()
say('| Member | Replaced by | Score on the panel\'s kept set | Score when it does not vote | Reference (another member replaced) | Penalty of voting | 95% interval |')
say('|---|---|---:|---:|---:|---:|---:|')
t_reps, t_point, replacements, masks = [], [], [], {}
for m in idx_f:
    repl = min(clean, key=lambda j: abs(a_all[j] - a_all[m]))
    replacements.append(repl)
    k_self = (votes_f - Y[m] + Y[repl]) <= threshold
    others = [u for u in idx_f if u != m and line_of(names[u]) != line_of(names[m])]
    k_ref = [(votes_f - Y[u] + Y[repl]) <= threshold for u in others]
    masks[m] = (k_self, k_ref)
    self_score = Y[m, k_self].mean()
    ref_score = float(np.mean([Y[m, k].mean() for k in k_ref]))
    b_self = (W @ (Y[m] * k_self)) / (W @ k_self.astype(float))
    b_ref = np.mean([(W @ (Y[m] * k)) / (W @ k.astype(float)) for k in k_ref], axis=0)
    t_reps.append(b_self - b_ref)
    t_point.append(self_score - ref_score)
    lo_m, hi_m = np.percentile(b_self - b_ref, [2.5, 97.5])
    say(f'| {names[m]} | {names[repl]} | {100 * Y[m, kept0].mean():.1f}% | {100 * self_score:.1f}% | {100 * ref_score:.1f}% | '
        f'{100 * (self_score - ref_score):+.2f} | {100 * lo_m:+.2f} to {100 * hi_m:+.2f} |')
    rec('leave_one_out', names[m], 'replacement', names[repl])
    rec('leave_one_out', names[m], 'score_own_panel', float(Y[m, kept0].mean()))
    rec('leave_one_out', names[m], 'score_not_voting', float(self_score))
    rec('leave_one_out', names[m], 'score_reference', ref_score)
    rec('leave_one_out', names[m], 'penalty', float(self_score - ref_score))
    rec('leave_one_out', names[m], 'plain_change', float(self_score - Y[m, kept0].mean()))
t_reps = np.array(t_reps)
m_lo, m_hi = np.percentile(t_reps.mean(axis=0), [2.5, 97.5])
outside = [j for j in main['pool'] if j not in replacements]
placebo_loo = np.array([np.mean([Y[x, masks[m][0]].mean() - np.mean([Y[x, k].mean() for k in masks[m][1]]) for m in idx_f])
                        for x in outside])
plain = np.mean([Y[m, masks[m][0]].mean() - Y[m, kept0].mean() for m in idx_f])
say()
say(f'- **Mean penalty of voting: {100 * np.mean(t_point):+.2f} points** (95% interval {100 * m_lo:+.2f} to {100 * m_hi:+.2f}); the '
    f'plain change against the original panel is {100 * plain:+.2f} points. The same quantity for {len(outside)} clean-pool systems '
    f'that never vote: mean {100 * placebo_loo.mean():+.2f}, SD {100 * placebo_loo.std(ddof=1):.2f} (it should be zero).')
say()
rec('leave_one_out', 'all', 'mean_penalty', float(np.mean(t_point)))
rec('leave_one_out', 'all', 'mean_penalty_lo', m_lo)
rec('leave_one_out', 'all', 'mean_penalty_hi', m_hi)
rec('leave_one_out', 'all', 'plain_change', float(plain))
rec('leave_one_out', 'all', 'placebo_mean', float(placebo_loo.mean()))
rec('leave_one_out', 'all', 'kept_set_size', int(kept0.sum()))

say('## Systems in the ability range (standardised accuracies, in %)')
say()
say('| System | Set | a(all) | a(kept) | a(kept) minus a(all) | Penalty against the clean pool |')
say('|---|---|---:|---:|---:|---:|')
placebo_of = dict(zip(main['pool'], main['placebo']))
placebo_of.update(dict(zip(idx_f, main['p_m'])))
for j in sorted(set(idx_f) | set(main['pool']), key=lambda j: a_all[j]):
    label = sets[j] + (' (sibling)' if tags[j] == 'sibling' else '')
    say(f'| {names[j]} | {label} | {100 * a_all[j]:.1f} | {100 * a_kept[j]:.1f} | {100 * (a_kept[j] - a_all[j]):+.1f} | '
        f'{100 * placebo_of[j]:+.2f} |')
say()
with (OUT / 'systems.tsv').open('w', encoding='utf-8', newline='\n') as fh:
    fh.write('system\tset\ttag\trun\ta_all\ta_kept\tpooled_all\tpooled_kept\tmean_of_subjects\n')
    for j in np.argsort(-a_all):
        fh.write(f'{names[j]}\t{sets[j]}\t{tags[j]}\t{runs[j]}\t{a_all[j]:.6f}\t{a_kept[j]:.6f}\t{p_all[j]:.6f}\t{p_kept[j]:.6f}\t'
                 f'{leaderboard[j]:.6f}\n')
with (OUT / 'records.tsv').open('w', encoding='utf-8', newline='\n') as fh:
    fh.write('analysis\tsubject\tquantity\tvalue\n')
    for row in RECORDS:
        fh.write('\t'.join(row) + '\n')
(OUT / 'summary.md').write_text('\n'.join(SUMMARY) + '\n', encoding='utf-8')
print(f'wrote {len(RECORDS)} records')
