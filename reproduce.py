"""Reproduce ScopeCard's stored results with one command, and time each step.

usage: python reproduce.py [--tier quick|paper|cards|memory|filters|filters-upstream|filters-figure|all] [--out DIR]
                           [--fetch] [--keep-going] [--diagnose-input-change]

Tiers ('all' runs paper, cards and filters):
  quick    the tests that need no data: synthetic cases, registry, schema and site checks.
  paper    the WN18RR measured-c analysis from the stored scorer outputs (experiments/results/wn18rr_runs). It rebuilds
           the declared target from WordNet 3.0 and WN18RR and checks it against the recorded hashes, runs the
           acceptance checks, reruns the prespecified analysis and compares every record with the stored
           records.tsv, then redraws Figures 2-4, Tables 1-2 and Box 1, and recomputes the GPU-hours from the job
           records. Needs WordNet 3.0 and WN18RR (41 MB).
  cards    every scope card of the card library from the public sources, the example cards, the browser page,
           the library site and the full test suite (golden tests included), each output compared with the
           shipped copy; then the memory step. Needs all inputs in inputs.json (3.9 GB).
  memory   the YAGO3-10 and Amazon-Book audits alone, each in a child process that reads its own peak resident
           memory when the audit ends (Windows: peak working set; Linux and macOS: getrusage ru_maxrss); their
           cards are compared with the shipped ones. Needs those two benchmarks' inputs in inputs.json (1.4 GB).
  filters  the studies of filters that use model outputs (experiments/results/subsets, panel_filter, separated_scoring,
           panel_composition, mmlupro_filter, mmlu_reconstruction), from their recorded inputs, which
           replay_inputs.py downloads again and checks (filter_inputs.json; 255 MB). The MMLU analysis starts from
           the stored per-question matrix (correct_matrix.npz). Every stored output is compared with the rerun's.
           Finally the figure on filtering by model outputs is drawn from the rerun's records, and every value it
           draws is checked against them.
  filters-upstream
           filters, plus the per-question MMLU records of the 268 leaderboard systems (6.3 GB, 15,334 files),
           from which the MMLU matrix is rebuilt and compared field by field with the stored matrix.
  filters-figure
           the figure on filtering by model outputs (fig_filters.pdf), drawn from the stored records of the filter
           studies: its list of drawn values (fig_filters_values.tsv) is compared with the stored list, and every
           value is checked against its record and each record file against its SHA-256. Needs no inputs.
GPU training and scoring (runs E1, E2, S3, deploy) are not rerun: experiments/scripts/hpc/ shows how they were run,
and their per-query outputs are the stored files that the paper tier starts from.

Inputs are read from experiments/pilot_data/<path> as listed in inputs.json (filters: in filter_inputs.json), and
every file is checked against its SHA-256 first. --fetch downloads missing inputs from the recorded sources.
Outputs go to --out (default: repro_out/ next to this script). Nothing shipped is overwritten: the cards tier works
on a copy of tool/, and the filter studies run in a staged copy of their scripts under <out>/filters/. Each step's
time and result go to <out>/timing.tsv and the logs to <out>/logs/.
A step fails when its output differs from the stored copy. A failed input check fails every step that reads that
input; --keep-going runs the steps that do not, and the run still ends as failed. --diagnose-input-change runs the
studies on inputs that failed their check, labels those steps 'input change diagnosis', and ends as failed: such a
run is a diagnosis, not a reproduction.
Needs Python >= 3.9 with numpy (paper, cards, filters), matplotlib (paper, filters, filters-figure), scipy and
pyarrow (filters) and pytest (quick, cards); experiments/env/requirements-filters.txt lists the versions used for the
filters tier. The paper, filters and filters-figure tiers also need LaTeX with the helvet, type1cm and underscore
packages: the figures are typeset with
Matplotlib's usetex so that they embed only Type 1 fonts. PyMuPDF is optional; without it the figures are drawn as
PDF only.
"""
import argparse
import fnmatch
import filecmp
import gzip
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


def find_root():
    for d in (HERE, os.path.dirname(HERE)):
        if os.path.isdir(os.path.join(d, 'tool', 'src', 'scopecard')) and os.path.isdir(os.path.join(d, 'experiments', 'scripts')):
            return d
    sys.exit('reproduce.py: tool/ and experiments/ not found next to this script or its parent folder')


ROOT = find_root()
TOOL, P4 = os.path.join(ROOT, 'tool'), os.path.join(ROOT, 'experiments')
SCRIPTS, DATA = os.path.join(P4, 'scripts'), os.path.join(P4, 'pilot_data')
RESULTS = os.path.join(P4, 'results')
MANIFEST = os.path.join(HERE, 'inputs.json')
FILTER_INDEX = os.path.join(HERE, 'filter_inputs.json')
REPLAY = os.path.join(HERE, 'replay_inputs.py')
EXAMPLES = {'amazon_book': 'amazon-book', 'gowalla': 'gowalla', 'ogbl_citation2': 'ogbl-citation2',
            'trecdl2019': 'trec-dl-2019-passage', 'yago3_10': 'yago3-10'}      # example card = library card
WN18RR_EXAMPLE = ['--delta-O', '-0.0142', '--metric', 'auc']                    # the WN18RR example adds a comparison
RECORD_TOL = 1e-9         # records are printed to 10 decimals; a last-digit change across numpy builds is allowed
CLI = 'import sys; sys.path.insert(0, sys.argv.pop(1)); from scopecard.cli import main; sys.exit(main(sys.argv[1:]))'


class StepFailed(Exception):
    pass


def data(rel):
    return os.path.join(DATA, *rel.split('/'))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def clock(s):
    return f'{s:.1f} s' if s < 90 else f'{int(s // 60)} min {int(round(s % 60)):02d} s'


class Runner:
    def __init__(self, out, keep_going):
        self.out, self.keep_going, self.rows = out, keep_going, []
        self.logs = os.path.join(out, 'logs')
        os.makedirs(self.logs, exist_ok=True)
        self.env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')

    def step(self, name, fn):
        print(f'... {name}', flush=True)
        t0 = time.time()
        try:
            note, ok = fn() or '', True
        except StepFailed as e:
            note, ok = str(e), False
        except OSError as e:                                  # a file that could not be read, written or copied
            note, ok = f'{type(e).__name__}: {e}'[:600], False
        dt = time.time() - t0
        self.rows.append((name, dt, 'ok' if ok else 'FAIL', note))
        print(f"{'ok  ' if ok else 'FAIL'} {name} ({clock(dt)}){': ' + note if note else ''}", flush=True)
        self.write_timing()
        if not ok and not self.keep_going:
            self.finish()
            sys.exit(1)
        return ok

    def run(self, name, args, cwd=None, env=None):
        """Run a Python command; its output goes to logs/<name>.log. Fails the step on a non-zero exit."""
        log = os.path.join(self.logs, name + '.log')
        with open(log, 'w', encoding='utf-8', newline='\n') as f:
            f.write('$ python ' + ' '.join(args) + '\n')
            f.flush()
            # -X utf8 and -B as flags, because -E and -I ignore PYTHONUTF8 and PYTHONDONTWRITEBYTECODE
            r = subprocess.run([sys.executable, '-X', 'utf8', '-B'] + args, cwd=cwd or ROOT, env=dict(self.env, **(env or {})),
                               stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            tail = open(log, encoding='utf-8', errors='replace').read().strip().splitlines()[-5:]
            raise StepFailed(f'exit code {r.returncode}; see {os.path.relpath(log, self.out)}: ' + ' | '.join(tail))
        return open(log, encoding='utf-8', errors='replace').read()

    def write_timing(self):
        with open(os.path.join(self.out, 'timing.tsv'), 'w', encoding='utf-8', newline='\n') as f:
            f.write('step\tseconds\tstatus\tnote\n')
            for name, dt, st, note in self.rows:
                f.write(f'{name}\t{dt:.1f}\t{st}\t{note}\n')

    def finish(self, diagnosis=False):
        width = max(len(r[0]) for r in self.rows) if self.rows else 10
        print('\nstep'.ljust(width + 1) + '  time        status')
        for name, dt, st, _ in self.rows:
            print(f'{name.ljust(width)}  {clock(dt).ljust(11)} {st}')
        total = sum(r[1] for r in self.rows)
        failed = [r[0] for r in self.rows if r[2] != 'ok']
        print(f"{'total'.ljust(width)}  {clock(total)}")
        print('all steps passed' if not failed else f'{len(failed)} step(s) failed: ' + ', '.join(failed))
        if diagnosis:
            print('inputs differed from their manifests: this run is an input change diagnosis, not a reproduction')
        print(f"timing: {os.path.join(self.out, 'timing.tsv')}")


# ---------------------------------------------------------------- inputs

def load_manifest():
    return json.load(open(MANIFEST, encoding='utf-8'))['inputs']


def wanted(tier, x):
    return tier in x['tiers']


def download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + '.part'
    req = urllib.request.Request(url, headers={'User-Agent': 'scopecard-reproduce/1'})
    with urllib.request.urlopen(req, timeout=60) as r, open(part, 'wb') as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(part, dest)


def inside(base, rel):
    p = os.path.normpath(os.path.join(base, rel))
    if os.path.commonpath([os.path.abspath(base), os.path.abspath(p)]) != os.path.abspath(base):
        raise StepFailed(f'archive member outside its folder: {rel}')
    return p


def fetch(x):
    fe = x['fetch']
    kind = fe['kind']
    if kind == 'manual':
        raise StepFailed(f"{x['id']}: no direct download; place {x['path']} by hand ({x.get('note', '')})")
    if kind == 'file':
        download(fe['url'], data(x['path']))
        return
    if kind == 'files':
        for f in x['files']:
            download(fe['url'] + f.rsplit('/', 1)[1], data(f))
        return
    archive = data(fe['archive'])
    if not (os.path.exists(archive) and sha256(archive) == fe['sha256']):
        download(fe['url'], archive)
        if sha256(archive) != fe['sha256']:
            raise StepFailed(f"{x['id']}: {fe['archive']} does not have the recorded SHA-256")
    base = data(fe['into'])
    if kind == 'tar.gz':
        with tarfile.open(archive, 'r:gz') as t:
            for m in t.getmembers():
                name = m.name.lstrip('./')
                if m.isfile() and name in fe['members']:
                    dest = inside(base, name)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with t.extractfile(m) as src, open(dest, 'wb') as out:
                        shutil.copyfileobj(src, out, 1 << 20)
    else:
        with zipfile.ZipFile(archive) as z:
            for name in fe['members']:
                dest = inside(base, name)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with z.open(name) as src, open(dest, 'wb') as out:
                    shutil.copyfileobj(src, out, 1 << 20)


def check_inputs(tier, do_fetch, ids=None):
    def go():
        n = size = 0
        skipped = []
        for x in load_manifest():
            if not wanted(tier, x) or (ids is not None and x['id'] not in ids):
                continue
            absent = [f for f in x['files'] if not os.path.exists(data(f))]
            if absent and do_fetch and x['fetch']['kind'] != 'manual':
                print(f"    fetching {x['id']} from {x['fetch']['url']}", flush=True)
                fetch(x)
                absent = [f for f in x['files'] if not os.path.exists(data(f))]
            if absent:
                if x.get('optional'):
                    skipped.append(x['id'])
                    continue
                raise StepFailed(f"{x['id']}: {len(absent)} file(s) missing under experiments/pilot_data/{x['path']}"
                                 + ('' if do_fetch else '; rerun with --fetch'))
            for f, h in x['files'].items():
                if sha256(data(f)) != h:
                    raise StepFailed(f'{f}: SHA-256 differs from inputs.json (a different version of the file?)')
                n += 1
                size += os.path.getsize(data(f))
        return f'{n} files, {size / 1e6:.0f} MB, all hashes match' + (f"; optional, absent: {', '.join(skipped)}" if skipped else '')
    return go


# ---------------------------------------------------------------- comparisons

def read_sha_list(path):
    out = {}
    for line in open(path, encoding='utf-8'):
        if line.strip():
            h, name = line.split(None, 1)
            out[name.strip()] = h
    return out


def check_hash_list(new_dir, sha_file):
    want = read_sha_list(sha_file)
    bad = [n for n, h in want.items() if sha256(os.path.join(new_dir, n)) != h]
    if bad:
        raise StepFailed('differs from the recorded hashes: ' + ', '.join(bad))
    return f'{len(want)} files equal the recorded SHA-256'


def same_content(a, b):
    """'bytes' if the files are identical, 'content' if a .gz or .zip differs only in its compressed bytes
    (deflate output depends on the zlib build, e.g. zlib vs zlib-ng), None otherwise."""
    if filecmp.cmp(a, b, shallow=False):
        return 'bytes'
    if a.endswith('.gz'):
        with gzip.open(a) as fa, gzip.open(b) as fb:
            return 'content' if fa.read() == fb.read() else None
    if a.endswith('.zip'):
        with zipfile.ZipFile(a) as za, zipfile.ZipFile(b) as zb:
            ia, ib = za.infolist(), zb.infolist()
            same = [(i.filename, i.date_time, i.external_attr) for i in ia] == [(i.filename, i.date_time, i.external_attr) for i in ib]
            return 'content' if same and all(za.read(i.filename) == zb.read(i.filename) for i in ia) else None
    return None


def same_files(new_dir, ref_dir, names):
    how = {n: same_content(os.path.join(new_dir, n), os.path.join(ref_dir, n)) for n in names}
    differ = [n for n, h in how.items() if h is None]
    if differ:
        raise StepFailed(f'{len(differ)} of {len(names)} differ: ' + ', '.join(differ[:8]))
    note = f"{len(names)} file{'s' if len(names) != 1 else ''} byte-identical"
    recompressed = [n for n, h in how.items() if h == 'content']
    if recompressed:
        note = (f"{len(names)} files identical; {len(recompressed)} compressed file(s) equal after decompression, "
                f"their bytes depend on the zlib build: " + ', '.join(recompressed[:4]))
    return note


def tree(d):
    out = []
    for base, _, names in os.walk(d):
        out += [os.path.relpath(os.path.join(base, n), d).replace(os.sep, '/') for n in names]
    return sorted(out)


def same_tree(new_dir, ref_dir):
    a, b = tree(new_dir), tree(ref_dir)
    if a != b:
        extra, missing = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        raise StepFailed(f'file lists differ: new only {extra[:5]}, shipped only {missing[:5]}')
    return same_files(new_dir, ref_dir, a)


def compare_records(new, ref):
    if filecmp.cmp(new, ref, shallow=False):
        n = sum(1 for _ in open(ref, encoding='utf-8')) - 1
        return f'{n} records, byte-identical'

    def load(p):
        with open(p, encoding='utf-8') as f:
            head = f.readline().rstrip('\n').split('\t')
            rows = [dict(zip(head, line.rstrip('\n').split('\t'))) for line in f]
        return {tuple(r[k] for k in ('design', 'estimand', 'subject', 'seed', 'quantity', 'cluster')): r for r in rows}
    A, B = load(new), load(ref)
    if set(A) != set(B):
        raise StepFailed(f'record keys differ: {len(set(A) - set(B))} new only, {len(set(B) - set(A))} shipped only')
    worst, where = 0.0, None
    for k, r in B.items():
        if A[k]['n_queries'] != r['n_queries']:
            raise StepFailed(f'n_queries differs for {k}')
        for col in ('value', 'lo', 'hi'):
            if (A[k][col] == '') != (r[col] == ''):
                raise StepFailed(f'{col} present in one copy only for {k}')
            if r[col] != '':
                d = abs(float(A[k][col]) - float(r[col]))
                if d > worst:
                    worst, where = d, (k, col)
    if worst > RECORD_TOL:
        raise StepFailed(f'largest difference {worst:.2e} at {where}')
    return f'{len(B)} records; not byte-identical, largest difference {worst:.1e} (<= {RECORD_TOL:g})'


# Comparisons of the filter studies. A file is byte-identical, or else equal apart from numbers that differ by at most
# one unit in the last digit printed in the stored copy (text), or by at most 1e-9 relative (JSON, arrays). Which
# files needed the tolerance is reported.
NUMBER = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')


def last_digit(text):
    m = re.fullmatch(r'[-+]?\d*(?:\.(\d*))?(?:[eE]([-+]?\d+))?', text)
    return 10.0 ** (int(m.group(2) or 0) - len(m.group(1) or ''))


def text_within_last_digit(new, ref):
    """None if the texts differ outside numbers or by more than one unit in a number's last printed digit; else the
    largest difference."""
    a, b = open(new, encoding='utf-8').read(), open(ref, encoding='utf-8').read()
    if NUMBER.split(a) != NUMBER.split(b):
        return None
    worst = 0.0
    for x, y in zip(NUMBER.findall(a), NUMBER.findall(b)):
        if x != y:
            d = abs(float(x) - float(y))
            if d > last_digit(y) * (1 + 1e-9):
                return None
            worst = max(worst, d)
    return worst


def json_equal(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a) == list(b) and all(json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or abs(a - b) <= 1e-9 * max(1.0, abs(b))
    return type(a) is type(b) and a == b


def arrays_equal(new, ref):
    import numpy as np
    with np.load(new, allow_pickle=False) as x, np.load(ref, allow_pickle=False) as y:
        if sorted(x.files) != sorted(y.files):
            return None
        for k in y.files:
            u, v = x[k], y[k]
            if u.dtype != v.dtype or u.shape != v.shape:
                return None
            if np.issubdtype(v.dtype, np.floating):
                if not np.allclose(u, v, rtol=1e-9, atol=0, equal_nan=True):
                    return None
            elif not np.array_equal(u, v):
                return None
        return sorted(y.files)


def package_hash_count():
    sums = os.path.join(ROOT, 'SHA256SUMS')
    return sum(1 for line in open(sums, encoding='utf-8') if line.strip()) if os.path.exists(sums) else None


# Shipped files that a rerun does not write again (the stdout of the original runs, with their times, and the fetch
# logs), figures (drawn, not compared: PDF files carry their creation time), and fields left out of the comparison of
# a file, with what is checked in their place. Declared here, before any rerun of version 0.1.0; design_sha256 was
# added on 2026-11-17, after the first reruns, when the Chinese original of the cross-prompt design was left out.
HISTORICAL, DRAWN = ('*.log',), ('*.png', '*.pdf')
EXCLUDED = {
    'separated_scoring/run_manifest.json': {
        'time_started': ('the start time of the run', None),
        'script_sha256': ("the stored value is the SHA-256 of the authors' original script (COPY_MAP.tsv names the "
                          "change); checked: the rerun's value equals the SHA-256 of this package's copy",
                          lambda v: v == sha256(os.path.join(SCRIPTS, 'separated_scoring.py'))),
        'design_sha256': ('the stored value is the SHA-256 of the design as fixed, in Chinese, which this package does '
                          'not ship (its English translation names it); checked: the rerun\'s value equals the SHA-256 '
                          'of the design file the rerun read',
                          lambda v: v == sha256(os.path.join(P4, SEPARATED_DESIGN)))},
    'separated_scoring/verification.json': {
        'protected_files_checked': ("the original run checked 24 files of the authors' working tree; checked: the rerun's "
                                    'value equals the number of files in SHA256SUMS', lambda v: v == package_hash_count())},
    'separated_scoring/summary.json': {'runtime_seconds_before_plot': ('the run time', None)},
}
# written again by the rerun but checked against the files it names, not against the stored list: the stored list
# was written after the independent check and names files that are not in this package (COPY_MAP.tsv)
SELF_HASHED = {'separated_scoring/SHA256SUMS.json'}
ORIGIN_TEXTS_SHA256 = '2c899c24a4766741733ea8afe4df68c09a85bdcc2a512e41e93eb77dbb07db70'   # not shipped (question texts)


def compare_study(study, new_dir, special=None, skip=()):
    """Compare every shipped file of results/<study> (its raw/ folder apart) with the rerun's copy."""
    ref_dir = os.path.join(RESULTS, study)
    special = special or {}
    names = [n for n in tree(ref_dir) if not n.startswith('raw/') and n not in skip]
    counts = {'identical': 0, 'tolerance': [], 'fields excluded': [], 'drawn': 0, 'historical': 0, 'special': []}
    bad = []
    for n in names:
        key, new, ref = f'{study}/{n}', os.path.join(new_dir, n), os.path.join(ref_dir, n)
        if any(fnmatch.fnmatch(n, p) for p in HISTORICAL):
            counts['historical'] += 1
            continue
        if not os.path.exists(new):
            bad.append(f'{n}: not written by the rerun')
            continue
        if any(fnmatch.fnmatch(n, p) for p in DRAWN):
            counts['drawn'] += 1
            continue
        if n in special:
            msg = special[n](new, ref)
            if msg is None:
                bad.append(f'{n}: differs')
            else:
                counts['special'].append(f'{n}: {msg}')
            continue
        if key in SELF_HASHED:
            listed = json.load(open(new, encoding='utf-8'))
            stage_root = os.path.dirname(os.path.dirname(os.path.dirname(new_dir)))
            wrong = [p for p, h in listed.items() if sha256(os.path.join(stage_root, *re.split(r'[\\/]', p))) != h]
            if wrong:
                bad.append(f'{n}: names {len(wrong)} file(s) with another SHA-256')
            else:
                counts['special'].append(f'{n}: its {len(listed)} hashes match the files it names')
            continue
        if filecmp.cmp(new, ref, shallow=False):
            counts['identical'] += 1
            continue
        if n.endswith('.npz'):
            if arrays_equal(new, ref) is None:
                bad.append(f'{n}: arrays differ')
            else:
                counts['special'].append(f'{n}: same arrays (compressed bytes differ)')
            continue
        if n.endswith('.json'):
            a, b = json.load(open(new, encoding='utf-8')), json.load(open(ref, encoding='utf-8'))
            excluded = EXCLUDED.get(key, {})
            for field, (why, check) in excluded.items():
                if check is not None and not check(a.get(field)):
                    bad.append(f'{n}: {field} = {a.get(field)!r} fails its check')
                a.pop(field, None)
                b.pop(field, None)
            if not json_equal(a, b):
                bad.append(f'{n}: differs' + (f' (apart from {", ".join(excluded)})' if excluded else ''))
            elif excluded:
                counts['fields excluded'].append(f'{n} [{", ".join(excluded)}]')
            else:
                counts['tolerance'].append(n)
            continue
        worst = text_within_last_digit(new, ref)
        if worst is None:
            bad.append(f'{n}: differs')
        else:
            counts['tolerance'].append(n)
    if bad:
        raise StepFailed(f'{len(bad)} of {len(names)} stored files differ: ' + '; '.join(bad[:6]))
    note = f"{counts['identical']} files byte-identical"
    if counts['tolerance']:
        note += (f"; {len(counts['tolerance'])} equal within the tolerance (last printed digit, or 1e-9 relative): "
                 + ', '.join(counts['tolerance'][:4]))
    if counts['fields excluded']:
        note += '; equal apart from declared fields: ' + '; '.join(counts['fields excluded'])
    if counts['special']:
        note += '; ' + '; '.join(counts['special'])
    if counts['drawn']:
        note += f"; {counts['drawn']} figure files drawn"
    if counts['historical']:
        note += f"; {counts['historical']} logs of the original runs not rewritten"
    return note


# ---------------------------------------------------------------- tiers

def tier_quick(R):
    nodata = os.path.join(R.out, 'no_inputs')     # golden tests skip; they run in the cards tier
    R.step('quick: tests without data', lambda: last_line(R.run(
        'quick_tests', ['-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_synthetic.py', 'tests/test_library.py'],
        cwd=TOOL, env={'SCOPECARD_PILOT_DATA': nodata})))


def last_line(text):
    lines = [x for x in text.strip().splitlines() if x.strip()]
    return lines[-1].strip() if lines else ''


def tier_paper(R, do_fetch):
    O = os.path.join(R.out, 'experiments')
    tgt, tgt3 = os.path.join(O, 'results', 'wn18rr_target'), os.path.join(O, 'results', 'wn18rr_target_s3')
    ana, figs = os.path.join(O, 'results', 'wn18rr_analysis'), os.path.join(O, 'figures')
    wn_dict, wn18rr = data('kg/wordnet30/WordNet-3.0/dict'), data('kg/conve/WN18RR')
    s = lambda name: os.path.join(SCRIPTS, name)
    for d in (tgt, tgt3):
        shutil.rmtree(d, ignore_errors=True)

    R.step('paper: check inputs', check_inputs('paper', do_fetch))

    def build():
        R.run('target_build', ['-I', s('build_wn18rr_target.py'), wn_dict, wn18rr, tgt])
        return check_hash_list(tgt, os.path.join(RESULTS, 'wn18rr_target', 'target_files.sha256'))
    R.step('paper: build declared target', build)
    R.step('paper: target acceptance checks', lambda: last_line(R.run(
        'target_accept', ['-I', s('accept_wn18rr_target.py'), tgt, wn18rr, wn_dict, s('build_wn18rr_target.py')])))

    def describe():
        R.run('target_describe', ['-I', s('describe_wn18rr_target.py'), tgt])
        return same_files(tgt, os.path.join(RESULTS, 'wn18rr_target'), ['composition.txt'])
    R.step('paper: target composition', describe)

    def build_s3():
        R.run('target_s3_build', ['-I', s('build_wn18rr_target_s3.py'), tgt, tgt3])
        return check_hash_list(tgt3, os.path.join(RESULTS, 'wn18rr_target_s3', 'target_files.sha256'))
    R.step('paper: build S3 target', build_s3)
    R.step('paper: S3 acceptance checks', lambda: last_line(R.run(
        'target_s3_accept', ['-I', s('accept_wn18rr_target_s3.py'), tgt3, tgt, wn18rr, s('build_wn18rr_target_s3.py')])))

    def analyze():
        R.run('analysis', ['-E', '-P', s('analyze_wn18rr.py'), os.path.join(RESULTS, 'wn18rr_runs'), tgt, ana,
                           '--target-s3', tgt3])
        return compare_records(os.path.join(ana, 'records.tsv'), os.path.join(RESULTS, 'wn18rr_analysis', 'records.tsv'))
    R.step('paper: prespecified analysis', analyze)

    def summarize():
        R.run('summary', ['-E', '-P', s('summarize_wn18rr.py'), ana])
        return same_files(ana, os.path.join(RESULTS, 'wn18rr_analysis'), ['summary.md'])
    R.step('paper: summary of records', summarize)

    def figures():
        # make_figures.py reads fixed paths next to itself; a copy placed in <out>/experiments/scripts reads the
        # records just computed, the shipped card library and the pilot table
        os.makedirs(os.path.join(O, 'scripts'), exist_ok=True)
        shutil.copyfile(s('make_figures.py'), os.path.join(O, 'scripts', 'make_figures.py'))
        shutil.copyfile(os.path.join(P4, 'pilot_decidability.tsv'), os.path.join(O, 'pilot_decidability.tsv'))
        lib = os.path.join(R.out, 'tool', 'library')
        shutil.rmtree(lib, ignore_errors=True)
        shutil.copytree(os.path.join(TOOL, 'library', 'cards'), os.path.join(lib, 'cards'))
        shutil.copyfile(os.path.join(TOOL, 'library', 'benchmarks.json'), os.path.join(lib, 'benchmarks.json'))
        R.run('figures', ['-E', '-P', os.path.join(O, 'scripts', 'make_figures.py'), figs])
        # the tables and Box 1 that make_figures.py writes; Figure 1 (fig1_schematic.tex) is drawn by hand
        text = sorted(n for n in os.listdir(figs) if n.endswith(('.md', '.tex')))
        drawn = sorted(n for n in os.listdir(figs) if n.endswith(('.pdf', '.png')))
        return same_files(figs, os.path.join(P4, 'figures'), text) + f' (tables, Box 1); {len(drawn)} figure files drawn'
    R.step('paper: figures, tables, Box 1', figures)

    def gpu_hours():
        # gpu_hours.py reads and writes results/compute next to its own folder; a copy in <out>/experiments/scripts reads
        # copies of the two job records
        d = os.path.join(O, 'results', 'compute')
        os.makedirs(d, exist_ok=True)
        for n in ('sacct_20261028.txt', 'e1_seed0_summary.txt'):
            shutil.copyfile(os.path.join(RESULTS, 'compute', n), os.path.join(d, n))
        os.makedirs(os.path.join(O, 'scripts'), exist_ok=True)
        shutil.copyfile(s('gpu_hours.py'), os.path.join(O, 'scripts', 'gpu_hours.py'))
        R.run('gpu_hours', ['-I', os.path.join(O, 'scripts', 'gpu_hours.py')])
        return same_files(d, os.path.join(RESULTS, 'compute'), ['gpu_hours.md'])
    R.step('paper: GPU-hours from the job records', gpu_hours)

    tex = os.path.join(P4, 'manuscript', 'main.tex')
    if os.path.exists(tex):
        R.step('paper: numbers in the manuscript', lambda: last_line(R.run(
            'manuscript_check', ['-E', '-P', os.path.join(SCRIPTS, 'check_manuscript.py'), tex])))


def tier_cards(R, do_fetch):
    stage = os.path.join(R.out, 'stage', 'tool')
    wn_dict, wn18rr = data('kg/wordnet30/WordNet-3.0/dict'), data('kg/conve/WN18RR')
    env = {'SCOPECARD_PILOT_DATA': DATA}

    R.step('cards: check inputs', check_inputs('cards', do_fetch))

    def copy_tool():
        shutil.rmtree(stage, ignore_errors=True)
        shutil.copytree(TOOL, stage, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache', 'dist', 'site', 'cards', 'entries'))
        return 'tool/ copied without its built outputs (cards, entries, site, web/dist)'
    R.step('cards: working copy of tool/', copy_tool)

    def cards():
        R.run('make_entries', ['-E', '-P', os.path.join(stage, 'library', 'make_entries.py'), DATA,
                               os.path.join(P4, 'pilot_decidability.tsv')])
        a = same_tree(os.path.join(stage, 'library', 'cards'), os.path.join(TOOL, 'library', 'cards'))
        b = same_tree(os.path.join(stage, 'library', 'entries'), os.path.join(TOOL, 'library', 'entries'))
        return f'cards: {a}; entries: {b}'
    R.step('cards: library cards from public sources', cards)

    def examples():
        ex = os.path.join(stage, 'examples')
        shutil.rmtree(ex)
        os.makedirs(ex)
        cli = ['-E', '-P', '-c', CLI, os.path.join(stage, 'src')]
        R.run('example_wn18rr', cli + ['kg-wordnet', '--release', wn18rr, '--wordnet', wn_dict, '--name', 'WN18RR']
              + WN18RR_EXAMPLE + ['--out', os.path.join(ex, 'wn18rr_card.json'), '--md', os.path.join(ex, 'wn18rr_card.md')])
        render = ('import json, sys; sys.path.insert(0, sys.argv[1]); from scopecard.card import to_markdown\n'
                  'for src, dst in zip(sys.argv[2::2], sys.argv[3::2]):\n'
                  '    open(dst, "w", encoding="utf-8", newline="\\n").write(to_markdown(json.load(open(src, encoding="utf-8"))))')
        pairs = []
        for name, slug in EXAMPLES.items():
            card = os.path.join(stage, 'library', 'cards', slug + '.json')
            shutil.copyfile(card, os.path.join(ex, name + '_card.json'))
            pairs += [card, os.path.join(ex, name + '_card.md')]
        R.run('examples_md', ['-E', '-P', '-c', render, os.path.join(stage, 'src')] + pairs)
        return same_tree(ex, os.path.join(TOOL, 'examples'))
    R.step('cards: example cards', examples)

    def web():
        R.run('web_build', ['-E', '-P', os.path.join(stage, 'web', 'build.py'), wn_dict])
        return same_tree(os.path.join(stage, 'web', 'dist'), os.path.join(TOOL, 'web', 'dist'))
    R.step('cards: browser page', web)

    def site():
        R.run('site_build', ['-E', '-P', os.path.join(stage, 'library', 'build_site.py')])
        return same_tree(os.path.join(stage, 'site'), os.path.join(TOOL, 'site'))
    R.step('cards: card library site', site)

    R.step('cards: full test suite', lambda: last_line(R.run(
        'tests_full', ['-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests'], cwd=stage, env=env)))
    memory_step(R, 'cards')


# ---------------------------------------------------------------- V7: peak memory of two audits

MEMORY_CARDS = ('yago3-10', 'amazon-book')
MEMORY_INPUTS = ('yago3-10', 'yago3-facts', 'amazon-book-release', 'amazon-2014-books-ratings')   # ids in inputs.json


# The audit runs in a child process that reads its own peak memory when the audit ends and writes it to a file.
# The parent cannot measure it: in a virtual environment on Windows, python.exe is a launcher that starts the
# interpreter as a further process, so the process the parent starts is not the one that runs the audit.
# The audit uses no further processes.
MEASURED_AUDIT = '''
import sys
peak_file = sys.argv.pop(1)
sys.path.insert(0, sys.argv.pop(1))
from scopecard.cli import main
code = main(sys.argv[1:])
if sys.platform == 'win32':
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (n, ctypes.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                                           'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                                           'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    c = Counters()
    c.cb = ctypes.sizeof(c)
    kernel32 = ctypes.WinDLL('kernel32')
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    get = ctypes.WinDLL('psapi').GetProcessMemoryInfo
    get.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    ok = get(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    peak = f'{c.PeakWorkingSetSize} {c.PeakPagefileUsage}' if ok else 'unread'
else:
    import resource
    peak = str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))
with open(peak_file, 'w', encoding='utf-8') as f:
    f.write(peak)
sys.exit(code)
'''
# On Windows the peak private memory is recorded as well. It counts all memory the process committed, used or not,
# so it is at least the private part of the working set, and unlike the working set it is not lowered when Windows
# trims working sets because other processes need memory.
MEMORY_MEASURE = {'win32': 'peak working set of the audit process, read by that process when the audit ended '
                           '(GetProcessMemoryInfo); private: its peak private memory (PeakPagefileUsage)'
                  }.get(sys.platform, 'maximum resident set size of the audit process, read by that process when the '
                                      'audit ended (getrusage ru_maxrss)')


def peak_memory_run(args, log, peak_file, env):
    """Run one audit in a child process; return its exit code and the peak memory, in bytes, that the process
    running the audit read for itself: (resident, private), private None outside Windows, or None if unread."""
    if os.path.exists(peak_file):
        os.remove(peak_file)
    with open(log, 'w', encoding='utf-8', newline='\n') as f:
        f.write('$ python -E -P -c <MEASURED_AUDIT in reproduce.py> ' + ' '.join([peak_file] + args) + '\n')
        f.flush()
        code = subprocess.call([sys.executable, '-X', 'utf8', '-B', '-E', '-P', '-c', MEASURED_AUDIT, peak_file] + args,
                               cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    try:
        values = [int(v) for v in open(peak_file, encoding='utf-8').read().split()]
        peak = (values[0], values[1] if len(values) > 1 else None)
    except (OSError, ValueError, IndexError):
        peak = None
    return code, peak


def memory_step(R, tier, do_fetch=False):
    def go():
        reg = {b['slug']: b for b in json.load(open(os.path.join(TOOL, 'library', 'benchmarks.json'), encoding='utf-8'))['benchmarks']}
        out = os.path.join(R.out, 'memory')
        os.makedirs(out, exist_ok=True)
        rows = []
        for slug in MEMORY_CARDS:
            b = reg[slug]
            local = {i['id']: data(i['local']) for i in b['inputs']}
            args = [local[a[1:-1]] if a.startswith('{') and a.endswith('}') else a for a in b['args']]
            t0 = time.time()
            code, peak = peak_memory_run([os.path.join(TOOL, 'src')] + args + ['--out', os.path.join(out, slug + '.json')],
                                         os.path.join(R.logs, f'memory_{slug}.log'),
                                         os.path.join(R.logs, f'memory_{slug}_peak.txt'), dict(R.env))
            if code != 0:
                raise StepFailed(f'{slug}: exit code {code}; see logs/memory_{slug}.log')
            if peak is None:
                raise StepFailed(f'{slug}: the peak memory could not be read')
            rows.append((slug, time.time() - t0, peak, MEMORY_MEASURE))
        cards = same_files(out, os.path.join(TOOL, 'library', 'cards'), [s + '.json' for s in MEMORY_CARDS])
        with open(os.path.join(out, 'memory.tsv'), 'w', encoding='utf-8', newline='\n') as f:
            f.write('card\tseconds\tpeak_bytes\tpeak_GB\tprivate_peak_bytes\tprivate_peak_GB\tmeasure\tpython\t'
                    'platform\tcpus\n')
            for slug, dt, (peak, private), how in rows:
                priv = f'{private}\t{private / 1e9:.2f}' if private is not None else '\t'
                f.write(f'{slug}\t{dt:.1f}\t{peak}\t{peak / 1e9:.2f}\t{priv}\t{how}\t{platform.python_version()}\t'
                        f'{platform.platform()}\t{os.cpu_count()}\n')
        return '; '.join(f'{s}: {p[0] / 1e9:.2f} GB peak' + (f' ({p[1] / 1e9:.2f} GB private)' if p[1] is not None else '')
                         + f', {clock(dt)}' for s, dt, p, _ in rows) + f'; cards: {cards}'
    if tier == 'memory':
        R.step('memory: check inputs', check_inputs('cards', do_fetch, ids=MEMORY_INPUTS))
    R.step('memory: peak memory of the YAGO3-10 and Amazon-Book audits', go)


# ---------------------------------------------------------------- filters

P4NAME = os.path.basename(P4)                 # 'experiments' in the package
FILTER_SCRIPTS = ('subsets_analyze', 'panel_filter_study', 'panel_size_curve', 'separated_scoring',
                  'check_separated_scoring', 'panel_composition', 'panel_composition_check',
                  'panel_composition_calibration', 'mmlupro_filter_test', 'mmlu_reconstruction_test',
                  'mmlu_panel_size_curve', 'mmlu_pairs_test', 'mmlu_pairs_posthoc', 'mmlu_pairs_precision_check',
                  'mmlu_difficulty_check', 'mmlu_difficulty_reference_test')
# The cross-prompt design was fixed in Chinese. The package ships its English translation instead, and the package's
# copy of separated_scoring.py reads the translation (COPY_MAP.tsv names the change); the authors' working tree has
# the original.
SEPARATED_DESIGN = ('SEPARATED_SCORING_DESIGN.en.md' if os.path.exists(os.path.join(P4, 'SEPARATED_SCORING_DESIGN.en.md'))
                    else 'SEPARATED_SCORING_DESIGN.md')
FILTER_DESIGNS = ('SUBSET_C_DESIGN.md', 'PANEL_FILTER_DESIGN.md', SEPARATED_DESIGN,
                  'PANEL_COMPOSITION_DESIGN.md', 'MMLUPRO_FILTER_DESIGN.md', 'MMLU_RECONSTRUCTION_DESIGN.md')
NEEDS = {'subsets': ['subsets'], 'mmlupro_filter': ['mmlupro_filter'], 'mmlu': ['mmlu_questions'],
         'mmlu-full': ['mmlu_details', 'mmlu_details_alt', 'mmlu_questions', 'mmlupro_filter']}


def filter_index():
    return json.load(open(FILTER_INDEX, encoding='utf-8'))


def filter_inputs(index, sid):
    """Where a set of filter inputs lives: experiments/pilot_data/filters/..., or under SCOPECARD_FILTER_INPUTS."""
    into = next(s['into'] for s in index['sets'] if s['id'] == sid).split('/')
    base = os.environ.get('SCOPECARD_FILTER_INPUTS')
    return os.path.join(base, *into[3:]) if base else os.path.join(ROOT, *into)


def long_paths_ok():
    """False on Windows when the 260-character limit on paths applies (long paths not enabled)."""
    if os.name != 'nt':
        return True
    try:
        import ctypes
        return bool(ctypes.WinDLL('ntdll').RtlAreLongPathsEnabled())
    except (AttributeError, OSError):
        return False


def longest_input_path(index, sid):
    """The longest path a replay of this set writes, with the '.part' suffix of a download in progress."""
    entry = next(s for s in index['sets'] if s['id'] == sid)
    into = os.path.abspath(filter_inputs(index, sid))
    with open(os.path.join(ROOT, *entry['manifest'].split('/')), encoding='utf-8') as f:
        paths = [line.split('\t', 1)[0] for line in f.read().splitlines()[1:] if line]
    if entry.get('only'):
        paths = [x for x in paths if x in entry['only']]
    return len(into) + 1 + max(len(x) for x in paths) + len('.part')


def longest_stage_path(index, sp):
    """The longest path the filter studies write under their staged copy sp (<out>/filters/<P4NAME>): the staged
    inputs of the subsets and MMLU-Pro studies, and every output, which goes where the stored copy is."""
    rels = []
    for sid in ('subsets', 'mmlupro_filter'):
        entry = next(s for s in index['sets'] if s['id'] == sid)
        with open(os.path.join(ROOT, *entry['manifest'].split('/')), encoding='utf-8') as f:
            rels += [f'results/{sid}/raw/' + line.split('\t', 1)[0] for line in f.read().splitlines()[1:] if line]
        rels += [f'results/{sid}/raw/{n}' for n in tree(os.path.join(RESULTS, sid, 'raw'))]
    for study in ('subsets', 'panel_filter', 'separated_scoring', 'panel_composition', 'mmlupro_filter',
                  'mmlu_reconstruction'):
        rels += [f'results/{study}/{n}' for n in tree(os.path.join(RESULTS, study)) if not n.startswith('raw/')]
    return len(os.path.abspath(sp)) + 1 + max(len(r) for r in rels)


def tier_filters(R, do_fetch, upstream, diagnose):
    index = filter_index()
    stage = os.path.join(R.out, 'filters')
    SP = os.path.join(stage, P4NAME)

    def path_lengths():
        longest = longest_stage_path(index, SP)
        if longest > 259 and not long_paths_ok():
            raise StepFailed(f'the studies would write paths of up to {longest} characters under {stage}, more than '
                             'Windows allows here (260); choose a shorter --out, for example C:\\bso, or enable long '
                             'paths in Windows')
        return f'the longest path the studies write has {longest} characters'
    R.step('filters: path lengths', path_lengths)
    res = lambda study: os.path.join(SP, 'results', study)
    sc = lambda name: os.path.join(SP, 'scripts', name + '.py')
    failed = set()
    sets = ['subsets', 'mmlupro_filter', 'mmlu_questions'] + (['mmlu_details', 'mmlu_details_alt'] if upstream else [])

    for sid in sets:
        def check_set(sid=sid):
            longest = longest_input_path(index, sid)
            if longest > 259 and not long_paths_ok():
                failed.add(sid)
                raise StepFailed(f'{sid}: its files would have paths of up to {longest} characters, more than Windows '
                                 'allows here (260); set SCOPECARD_FILTER_INPUTS to a short folder, for example '
                                 'C:\\bsf, or enable long paths in Windows')
            args = ['-E', '-P', REPLAY, 'replay', FILTER_INDEX, sid, '--into', filter_inputs(index, sid),
                    '--report', os.path.join(R.logs, f'replay_{sid}.tsv')] + ([] if do_fetch else ['--check-only'])
            try:
                return last_line(R.run(f'replay_{sid}', args))
            except StepFailed as e:
                failed.add(sid)
                raise StepFailed(str(e) + ('' if do_fetch else '; rerun with --fetch to download missing inputs'))
        R.step(f'filters: inputs, {sid}', check_set)

    def study(name, needs, fn):
        bad = sorted({s for n in needs for s in NEEDS[n]} & failed)
        if bad and not diagnose:
            def not_run():
                raise StepFailed('not run: input set ' + ', '.join(bad) + ' failed its check')
            R.step(name, not_run)
            return
        R.step(('input change diagnosis: ' if bad else '') + name, fn)

    def setup():
        shutil.rmtree(stage, ignore_errors=True)
        os.makedirs(os.path.join(SP, 'scripts'))
        for n in FILTER_SCRIPTS:
            shutil.copyfile(os.path.join(SCRIPTS, n + '.py'), sc(n))
        for n in FILTER_DESIGNS:
            shutil.copyfile(os.path.join(P4, n), os.path.join(SP, n))
        # the subsets inputs where the scripts expect them: the shipped records of raw/ (manifest, pinned commits),
        # then the replayed files
        raw = os.path.join(res('subsets'), 'raw')
        shutil.copytree(os.path.join(RESULTS, 'subsets', 'raw'), raw)
        shutil.copytree(filter_inputs(index, 'subsets'), raw, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('replay_report.tsv'))
        return f'{len(FILTER_SCRIPTS)} scripts, {len(FILTER_DESIGNS)} designs and the subsets inputs staged in {os.path.relpath(stage, R.out)}'
    R.step('filters: staged copy', setup)
    raw = os.path.join(res('subsets'), 'raw')

    def subsets():
        R.run('subsets', ['-E', '-P', sc('subsets_analyze'), raw, res('subsets')])
        return compare_study('subsets', res('subsets'))
    study('filters: released subsets of SWE-bench and BigCodeBench', ['subsets'], subsets)

    def panel_filter():
        out = res('panel_filter')
        R.run('panel_filter', ['-E', '-P', sc('panel_filter_study'), raw, out])
        R.run('panel_filter_deviation', ['-E', '-P', sc('panel_filter_study'), raw, out, 'deviation'])
        R.run('panel_filter_size', ['-E', '-P', sc('panel_size_curve'), raw, out])
        return compare_study('panel_filter', out)
    study('filters: panel filter on BigCodeBench', ['subsets'], panel_filter)

    def separated():
        R.run('separated_scoring', ['-E', '-P', sc('separated_scoring')], env={'SCOPECARD_PACKAGE_ROOT': ROOT})
        R.run('separated_scoring_check', ['-E', '-P', sc('check_separated_scoring')])
        return compare_study('separated_scoring', res('separated_scoring'))
    study('filters: cross-prompt scoring', ['subsets'], separated)

    def composition():
        R.run('panel_composition_checks', ['-E', '-P', sc('panel_composition'), '--check'])
        R.run('panel_composition', ['-E', '-P', sc('panel_composition')])
        R.run('panel_composition_check', ['-E', '-P', sc('panel_composition_check')])
        R.run('panel_composition_calibration', ['-E', '-P', sc('panel_composition_calibration')])
        return compare_study('panel_composition', res('panel_composition'))
    study('filters: panel composition', ['subsets'], composition)

    def mmlupro():
        pro = os.path.join(res('mmlupro_filter'), 'raw')
        shutil.copytree(os.path.join(RESULTS, 'mmlupro_filter', 'raw'), pro, dirs_exist_ok=True)
        shutil.copytree(filter_inputs(index, 'mmlupro_filter'), pro, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('replay_report.tsv'))
        R.run('mmlupro_filter', ['-E', '-P', sc('mmlupro_filter_test'), pro, res('mmlupro_filter')])
        return compare_study('mmlupro_filter', res('mmlupro_filter'))
    study('filters: MMLU-Pro filter models', ['mmlupro_filter'], mmlupro)

    out = res('mmlu_reconstruction')
    parquet = os.path.join(filter_inputs(index, 'mmlu_questions'), 'cais_mmlu_all_test.parquet')
    stored_npz = os.path.join(RESULTS, 'mmlu_reconstruction', 'correct_matrix.npz')

    def matrix_acceptance(new, ref):
        a = open(new, encoding='utf-8').read().splitlines()
        b = open(ref, encoding='utf-8').read().splitlines()
        first = f'INFO  started from correct_matrix.npz (sha256 {sha256(stored_npz)}); the data gates were not rerun'
        tail = a[1:]
        if not a or a[0] != first or not tail or b[-len(tail):] != tail:
            return None
        return (f'{len(tail)} analysis lines identical; the {len(b) - len(tail)} lines of the data gates and of the '
                f'loading of runs belong to the build of the matrix, which this entry does not rerun')

    def mmlu():
        matrix_in = os.path.join(SP, 'matrix_in')
        os.makedirs(matrix_in, exist_ok=True)
        for n in ('correct_matrix.npz', 'systems.tsv'):
            shutil.copyfile(os.path.join(RESULTS, 'mmlu_reconstruction', n), os.path.join(matrix_in, n))
        R.run('mmlu_from_matrix', ['-E', '-P', sc('mmlu_reconstruction_test'), '--from-matrix', matrix_in, out])
        shutil.copyfile(stored_npz, os.path.join(out, 'correct_matrix.npz'))      # the matrix the analyses below read
        R.run('mmlu_panel_size', ['-E', '-P', sc('mmlu_panel_size_curve'), out])
        R.run('mmlu_pairs', ['-E', '-P', sc('mmlu_pairs_test'), out])
        R.run('mmlu_pairs_posthoc', ['-E', '-P', sc('mmlu_pairs_posthoc'), out, parquet])
        R.run('mmlu_pairs_precision_check', ['-E', '-P', sc('mmlu_pairs_precision_check'), os.path.join(out, 'correct_matrix.npz')])
        R.run('mmlu_difficulty_check', ['-E', '-P', sc('mmlu_difficulty_check'), out, parquet])
        R.run('mmlu_difficulty_reference', ['-E', '-P', sc('mmlu_difficulty_reference_test'), out, parquet])
        # correct_matrix.npz is this entry's input, a copy of the stored matrix: not compared
        return compare_study('mmlu_reconstruction', out, special={'acceptance.txt': matrix_acceptance},
                             skip=('correct_matrix.npz',))
    study('filters: MMLU reconstruction from the stored matrix', ['mmlu'], mmlu)

    def figure():
        # the figure on filtering by model outputs, from the rerun's records; compare_study has compared those records
        # with the stored ones (within the last printed digit), so the drawn values are checked against the rerun's
        figs = os.path.join(SP, 'figures')
        R.run('fig_filters', ['-E', '-P', os.path.join(SCRIPTS, 'make_fig_filters.py'), '--results',
                              os.path.join(SP, 'results'), '--out', figs])
        return last_line(R.run('fig_filters_check', ['-E', '-P', os.path.join(SCRIPTS, 'check_fig_filters.py'),
                                                     '--results', os.path.join(SP, 'results'), '--values',
                                                     os.path.join(figs, 'fig_filters_values.tsv')]))
    study('filters: figure on filtering by model outputs, from the rerun records', ['subsets', 'mmlu'], figure)

    if not upstream:
        return

    def full():
        raw_mmlu = filter_inputs(index, 'mmlu_details')
        R.run('mmlu_run_json', ['-E', '-P', REPLAY, 'run-json', FILTER_INDEX, raw_mmlu])
        shutil.copyfile(os.path.join(RESULTS, 'mmlu_reconstruction', 'raw', 'selection.tsv'), os.path.join(raw_mmlu, 'selection.tsv'))
        pro = os.path.join(res('mmlupro_filter'), 'raw')
        if not os.path.isdir(pro):
            raise StepFailed('the MMLU-Pro inputs were not staged (the MMLU-Pro step did not run)')
        full_out = res('mmlu_reconstruction_full')
        R.run('mmlu_full_build', ['-E', '-P', sc('mmlu_reconstruction_test'), raw_mmlu, pro, full_out])
        fields = arrays_equal(os.path.join(full_out, 'correct_matrix.npz'), stored_npz)
        if fields is None:
            raise StepFailed('the rebuilt matrix differs from the stored correct_matrix.npz')
        same = same_files(full_out, os.path.join(RESULTS, 'mmlu_reconstruction'),
                          ['acceptance.txt', 'records.tsv', 'summary.md', 'systems.tsv'])
        if sha256(os.path.join(full_out, 'mmlupro_origin_texts.json')) != ORIGIN_TEXTS_SHA256:
            raise StepFailed('mmlupro_origin_texts.json does not have the SHA-256 of the original run')
        return (f"matrix rebuilt from the per-question records: fields {', '.join(fields)} equal the stored ones "
                f"(values, dtypes, shapes, system and question order); {same}; mmlupro_origin_texts.json (not shipped) "
                f"has the original run's SHA-256")
    study('filters-upstream: MMLU matrix rebuilt from the leaderboard records', ['mmlu-full'], full)


def tier_filters_figure(R):
    """The figure on filtering by model outputs, drawn from the stored records; no inputs are needed."""
    figs = os.path.join(R.out, 'experiments', 'figures')

    def go():
        R.run('fig_filters', ['-E', '-P', os.path.join(SCRIPTS, 'make_fig_filters.py'), '--results', RESULTS,
                              '--out', figs])
        check = last_line(R.run('fig_filters_check', ['-E', '-P', os.path.join(SCRIPTS, 'check_fig_filters.py'),
                                                      '--results', RESULTS, '--values',
                                                      os.path.join(figs, 'fig_filters_values.tsv')]))
        same = same_files(figs, os.path.join(P4, 'figures'), ['fig_filters_values.tsv'])
        return f'{check}; {same}'
    R.step('filters-figure: figure on filtering by model outputs, from the stored records', go)


def environment():
    lines = [f'python {platform.python_version()} ({platform.python_implementation()})',
             f'platform {platform.platform()}', f'cpus {os.cpu_count()}']
    for mod in ('numpy', 'scipy', 'pyarrow', 'matplotlib', 'pytest'):
        try:
            lines.append(f'{mod} {__import__(mod).__version__}')
        except ImportError:
            lines.append(f'{mod} not installed')
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--tier', choices=['quick', 'paper', 'cards', 'memory', 'filters', 'filters-upstream', 'filters-figure',
                                       'all'], default='paper')
    ap.add_argument('--out', default=os.path.join(HERE, 'repro_out'))
    ap.add_argument('--fetch', action='store_true', help='download missing inputs from their recorded sources')
    ap.add_argument('--keep-going', action='store_true', help='run the remaining steps after a failure; the run still fails')
    ap.add_argument('--diagnose-input-change', action='store_true',
                    help='run the filter studies on inputs that failed their check, as a diagnosis (the run fails)')
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    env = environment()
    with open(os.path.join(out, 'environment.txt'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(env + [f'tier {a.tier}', f'root {ROOT}']) + '\n')
    print('\n'.join(env) + f'\ntier {a.tier}; outputs in {out}\n', flush=True)
    R = Runner(out, a.keep_going or a.diagnose_input_change)
    if a.tier == 'quick':
        tier_quick(R)
    if a.tier in ('paper', 'all'):
        tier_paper(R, a.fetch)
    if a.tier in ('cards', 'all'):
        tier_cards(R, a.fetch)
    if a.tier == 'memory':
        memory_step(R, 'memory', a.fetch)
    if a.tier in ('filters', 'filters-upstream', 'all'):
        tier_filters(R, a.fetch, a.tier == 'filters-upstream', a.diagnose_input_change)
    if a.tier == 'filters-figure':
        tier_filters_figure(R)
    diagnosis = any(r[0].startswith('input change diagnosis') for r in R.rows)
    R.finish(diagnosis)
    sys.exit(0 if all(r[2] == 'ok' for r in R.rows) and not diagnosis else 1)


if __name__ == '__main__':
    main()
