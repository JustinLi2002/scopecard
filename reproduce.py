"""Reproduce BenchScope's stored results with one command, and time each step.

usage: python reproduce.py [--tier quick|paper|cards|all] [--out DIR] [--fetch] [--keep-going]

Tiers ('all' runs paper, then cards):
  quick  the tests that need no data: synthetic cases, registry, schema and site checks.
  paper  the WN18RR measured-c analysis from the stored scorer outputs (experiments/results/wn18rr_runs). It rebuilds
         the declared target from WordNet 3.0 and WN18RR and checks it against the recorded hashes, runs the
         acceptance checks, reruns the prespecified analysis and compares every record with the stored
         records.tsv, then redraws Figures 2-4, Tables 1-2 and Box 1. Needs WordNet 3.0 and WN18RR (41 MB).
  cards  every scope card of the card library from the public sources, the example cards, the browser page,
         the library site and the full test suite (golden tests included), each output compared with the
         shipped copy. Needs all inputs in inputs.json (3.9 GB).
GPU training and scoring (runs E1, E2, S3, deploy) are not rerun: experiments/scripts/hpc/ shows how they were run,
and their per-query outputs are the stored files that the paper tier starts from.

Inputs are read from experiments/pilot_data/<path> as listed in inputs.json, and every file is checked against its
SHA-256 first. --fetch downloads missing inputs from the public sources listed there (3.8 GB for 'cards').
Outputs go to --out (default: repro_out/ next to this script). Nothing shipped is overwritten: the cards tier
works on a copy of tool/. Each step's time and result go to <out>/timing.tsv and the logs to <out>/logs/.
Needs Python >= 3.9 with numpy (paper, cards), matplotlib (paper) and pytest (quick, cards). The paper tier also
needs LaTeX with the helvet, type1cm and underscore packages: the figures are typeset with Matplotlib's usetex so
that they embed only Type 1 fonts. PyMuPDF is optional; without it the figures are drawn as PDF only.
"""
import argparse
import filecmp
import gzip
import hashlib
import json
import os
import platform
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
        if os.path.isdir(os.path.join(d, 'tool', 'src', 'benchscope_audit')) and os.path.isdir(os.path.join(d, 'experiments', 'scripts')):
            return d
    sys.exit('reproduce.py: tool/ and experiments/ not found next to this script or its parent folder')


ROOT = find_root()
TOOL, P4 = os.path.join(ROOT, 'tool'), os.path.join(ROOT, 'experiments')
SCRIPTS, DATA = os.path.join(P4, 'scripts'), os.path.join(P4, 'pilot_data')
RESULTS = os.path.join(P4, 'results')
MANIFEST = os.path.join(HERE, 'inputs.json')
EXAMPLES = {'amazon_book': 'amazon-book', 'gowalla': 'gowalla', 'ogbl_citation2': 'ogbl-citation2',
            'trecdl2019': 'trec-dl-2019-passage', 'yago3_10': 'yago3-10'}      # example card = library card
WN18RR_EXAMPLE = ['--delta-O', '-0.0142', '--metric', 'auc']                    # the WN18RR example adds a comparison
RECORD_TOL = 1e-9         # records are printed to 10 decimals; a last-digit change across numpy builds is allowed


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
        dt = time.time() - t0
        self.rows.append((name, dt, 'ok' if ok else 'FAIL', note))
        print(f"{'ok  ' if ok else 'FAIL'} {name} ({clock(dt)}){': ' + note if note else ''}", flush=True)
        self.write_timing()
        if not ok and not self.keep_going:
            self.finish()
            sys.exit(1)

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

    def finish(self):
        width = max(len(r[0]) for r in self.rows) if self.rows else 10
        print('\nstep'.ljust(width + 1) + '  time        status')
        for name, dt, st, _ in self.rows:
            print(f'{name.ljust(width)}  {clock(dt).ljust(11)} {st}')
        total = sum(r[1] for r in self.rows)
        failed = [r[0] for r in self.rows if r[2] != 'ok']
        print(f"{'total'.ljust(width)}  {clock(total)}")
        print('all steps passed' if not failed else f'{len(failed)} step(s) failed: ' + ', '.join(failed))
        print(f"timing: {os.path.join(self.out, 'timing.tsv')}")


# ---------------------------------------------------------------- inputs

def load_manifest():
    return json.load(open(MANIFEST, encoding='utf-8'))['inputs']


def wanted(tier, x):
    return tier in x['tiers']


def download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + '.part'
    req = urllib.request.Request(url, headers={'User-Agent': 'benchscope-reproduce/1'})
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


def check_inputs(tier, do_fetch):
    def go():
        n = size = 0
        skipped = []
        for x in load_manifest():
            if not wanted(tier, x):
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


# ---------------------------------------------------------------- tiers

def tier_quick(R):
    nodata = os.path.join(R.out, 'no_inputs')     # golden tests skip; they run in the cards tier
    R.step('quick: tests without data', lambda: last_line(R.run(
        'quick_tests', ['-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_synthetic.py', 'tests/test_library.py'],
        cwd=TOOL, env={'BENCHSCOPE_PILOT_DATA': nodata})))


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

    tex = os.path.join(P4, 'manuscript', 'main.tex')
    if os.path.exists(tex):
        R.step('paper: numbers in the manuscript', lambda: last_line(R.run(
            'manuscript_check', ['-E', '-P', os.path.join(SCRIPTS, 'check_manuscript.py'), tex])))


def tier_cards(R, do_fetch):
    stage = os.path.join(R.out, 'stage', 'tool')
    wn_dict, wn18rr = data('kg/wordnet30/WordNet-3.0/dict'), data('kg/conve/WN18RR')
    env = {'BENCHSCOPE_PILOT_DATA': DATA}

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
        cli = ['-E', '-P', '-c', 'import sys; sys.path.insert(0, sys.argv.pop(1)); from benchscope_audit.cli import main; '
               'sys.exit(main(sys.argv[1:]))', os.path.join(stage, 'src')]
        R.run('example_wn18rr', cli + ['kg-wordnet', '--release', wn18rr, '--wordnet', wn_dict, '--name', 'WN18RR']
              + WN18RR_EXAMPLE + ['--out', os.path.join(ex, 'wn18rr_card.json'), '--md', os.path.join(ex, 'wn18rr_card.md')])
        render = ('import json, sys; sys.path.insert(0, sys.argv[1]); from benchscope_audit.card import to_markdown\n'
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


def environment():
    lines = [f'python {platform.python_version()} ({platform.python_implementation()})',
             f'platform {platform.platform()}', f'cpus {os.cpu_count()}']
    for mod in ('numpy', 'matplotlib', 'pytest'):
        try:
            lines.append(f'{mod} {__import__(mod).__version__}')
        except ImportError:
            lines.append(f'{mod} not installed')
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--tier', choices=['quick', 'paper', 'cards', 'all'], default='paper')
    ap.add_argument('--out', default=os.path.join(HERE, 'repro_out'))
    ap.add_argument('--fetch', action='store_true', help='download missing inputs listed in inputs.json')
    ap.add_argument('--keep-going', action='store_true', help='run the remaining steps after a failure')
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    env = environment()
    with open(os.path.join(out, 'environment.txt'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(env + [f'tier {a.tier}', f'root {ROOT}']) + '\n')
    print('\n'.join(env) + f'\ntier {a.tier}; outputs in {out}\n', flush=True)
    R = Runner(out, a.keep_going)
    if a.tier == 'quick':
        tier_quick(R)
    if a.tier in ('paper', 'all'):
        tier_paper(R, a.fetch)
    if a.tier in ('cards', 'all'):
        tier_cards(R, a.fetch)
    R.finish()
    sys.exit(0 if all(r[2] == 'ok' for r in R.rows) else 1)


if __name__ == '__main__':
    main()
