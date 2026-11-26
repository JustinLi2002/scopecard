"""Replay the recorded downloads of the filter studies, and check every file.

usage:
  python replay_inputs.py replay INDEX SET [--into DIR] [--check-only] [--force] [--report FILE] [--workers N]
  python replay_inputs.py run-json INDEX RAW
  python replay_inputs.py index ROOT OUT

INDEX is filter_inputs.json (next to this script). Each of its input sets names a manifest of this package: the
list that the study's fetch script wrote when it downloaded the files, one row per file with its path, size,
SHA-256, source, upstream revision and date. `replay` downloads the files of one set again from the recorded sources
into DIR (default: the set's folder under experiments/pilot_data/filters/) and checks each one. The fetch scripts
themselves are not run. A file already in DIR with the recorded size and SHA-256 is kept ('present') unless --force
is given; --check-only downloads nothing and fails on a missing file. The report (default DIR/replay_report.tsv)
has one line per file. Any failure makes the command exit with status 1: a set with a failed file has not been
replayed, and nothing is retried with a looser check.

How each kind of source is replayed and checked:
  pinned    a GitHub file at a commit, or a Hugging Face file at a commit (.../resolve/<commit>/...): downloaded;
            it must have the recorded size and SHA-256.
  listing   a GitHub directory listing at a commit (api.github.com ...?ref=<commit>): downloaded; it must have the
            recorded SHA-256, or else the normalized JSON hash (rule J) recorded in INDEX.
  table     a table that the fetch script read through the Hugging Face rows API. That API serves only the current
            revision and does not fix its serialization, so the table is rebuilt from the dataset's parquet file at
            the recorded revision by the converter below. The parquet file must have the size and SHA-256 that the
            Hugging Face hub lists for it at that revision (recorded in INDEX). The table must have the recorded
            SHA-256, or else the normalized content hash (rule T) recorded in INDEX.
  unpinned  a file at a URL without a revision (cdn.openai.com): downloaded; it must have the recorded SHA-256.
  member    a file of a replayed zip archive; member names are checked to stay inside the folder; it must have
            the recorded SHA-256.
  derived   archive_members.tsv, rebuilt from the replayed listing and archives as the fetch script built it; it
            must have the recorded SHA-256.

Rule T (tables of one JSON object per line): parse each line; write each row back with sorted keys, compact
separators and non-ASCII characters kept, json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(',', ':'));
join the rows in their original order, each followed by a newline; take the SHA-256 of the UTF-8 bytes. Equal
hashes mean the same rows in the same order with the same fields and values (identifiers, answers and every other
field); only key order, spacing and escaping are ignored. Numbers compare as parsed: 1 and 1.0 differ.
Rule J (JSON documents): rule T applied to the whole document as one row.
Converter: read the parquet file with pyarrow; keep the columns recorded in INDEX (all columns where the fetch script
kept all); for MMLU-Pro add n_options = the number of options, as the fetch script did; a float NaN becomes null
(JSON has no NaN); write the columns in the order recorded in INDEX from the archived table ('keys'), one
json.dumps(row, ensure_ascii=False) and a newline per row.
INDEX, these rules and the converter were fixed on 2026-11-17 before any file was downloaded again; the normalized
hashes and the column orders in INDEX were computed from the archived copies. Revision of the converter, 2026-11-17,
after the first replay: that replay wrote the columns in the parquet file's order. Twelve of the thirteen tables then
had the recorded SHA-256; the Instruct table of bigcode/bigcodebench-perf matched under rule T only, because its
parquet file orders the 1,141 columns differently from the rows API. Since the cross-prompt and composition scripts
check their inputs' SHA-256 themselves, the converter now writes the recorded column order. The checks are unchanged.

`run-json` writes the run.json files of the MMLU study's raw folder after its two manifests have been replayed.
The fetch scripts derived these files from the leaderboard repositories and did not list them. For each repository
in fetch_status.json with status "ok", details/<repository>/run.json is {"status": "ok", "sha": <revision>, "run":
<run folder>}, with the revision and the run folder read from the URLs of its files; for the other repositories it
is {"status": <status>}. Each folder of details_alt/ gets the same three fields from the URLs of its own files. The
analysis reads only "status" and "run"; the fetch script also wrote "runs_complete" or "runs", which nothing reads.

`index` is how INDEX was made, kept for the record: it reads the manifests and the archived files under ROOT (the
authors' working tree) and the file lists of the Hugging Face revisions (metadata only).

Needs Python >= 3.9; pyarrow for the tables (replay of the subsets and mmlupro_filter sets).
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import pathlib
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime

UA = 'scopecard-replay/0.1'
HEX40 = re.compile(r'^[0-9a-f]{40}$')
REPORT_HEAD = 'path\tkind\tstatus\tcheck\tbytes_recorded\tbytes\tsha256_recorded\tsha256\tnote\n'


# ---------------------------------------------------------------- hashes and rules

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')) + '\n'


def rule_t(path):
    """Normalized content hash of a table with one JSON object per line (rule T), its row count and key order."""
    h, rows, keys = hashlib.sha256(), 0, None
    with open(path, encoding='utf-8', newline='') as f:
        for line in f:
            if not line.endswith('\n'):
                raise ValueError(f'{path}: last line has no newline')
            row = json.loads(line[:-1])
            if keys is None:
                keys = list(row)
            h.update(canonical(row).encode('utf-8'))
            rows += 1
    return h.hexdigest(), rows, keys


def rule_j(path):
    with open(path, encoding='utf-8') as f:
        return hashlib.sha256(canonical(json.load(f)).encode('utf-8')).hexdigest()


# ---------------------------------------------------------------- manifests and sources

def read_manifest(path):
    with open(path, encoding='utf-8', newline='') as f:          # one manifest was written with CRLF line ends
        head = f.readline().rstrip('\r\n').split('\t')
        if head != ['path', 'bytes', 'sha256', 'url', 'upstream_revision', 'retrieved']:
            raise SystemExit(f'{path}: unexpected header {head}')
        return [dict(zip(head, line.rstrip('\r\n').split('\t'))) for line in f if line.strip()]


def kind_of(url):
    if url.startswith('(derived'):
        return 'derived'
    if url.endswith(' (member)'):
        return 'member'
    u = urllib.parse.urlsplit(url.split(' ')[0])
    parts = u.path.split('/')
    if u.netloc == 'datasets-server.huggingface.co' and u.path == '/rows':
        return 'table'
    if u.netloc == 'api.github.com' and '/contents/' in u.path and re.search(r'(^|&)ref=[0-9a-f]{40}$', u.query):
        return 'listing'
    if u.netloc == 'raw.githubusercontent.com' and len(parts) > 4 and HEX40.match(parts[3]):
        return 'pinned'
    if u.netloc == 'huggingface.co' and 'resolve' in parts and HEX40.match((parts + [''])[parts.index('resolve') + 1]):
        return 'pinned'
    if u.scheme == 'https':
        return 'unpinned'
    raise SystemExit(f'unknown kind of source: {url}')


def inside(base, rel):
    """The path of a manifest file or archive member under base; anything that could leave base is refused."""
    parts = pathlib.PurePosixPath(rel).parts
    if not rel or rel.startswith(('/', '\\')) or '..' in parts or ':' in rel or '\\' in rel:
        raise SystemExit(f'unsafe path: {rel!r}')
    return os.path.join(base, *parts)


class Fetcher:
    """Downloads with a cap on the request rate, and retries on rate limits, server errors and broken connections
    (never on a content mismatch)."""

    def __init__(self, rate):
        self.lock, self.next_slot, self.interval = threading.Lock(), 0.0, 1.0 / rate

    def _turn(self):
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_slot)
            self.next_slot = slot + self.interval
        if slot > now:
            time.sleep(slot - now)

    def get(self, url, dest, accept=None, tries=8):
        headers = {'User-Agent': UA}
        if accept:
            headers['Accept'] = accept
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        part = dest + '.part'
        for attempt in range(tries):
            self._turn()
            try:
                h, n = hashlib.sha256(), 0
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r, \
                        open(part, 'wb') as f:
                    for block in iter(lambda: r.read(1 << 20), b''):
                        f.write(block)
                        h.update(block)
                        n += len(block)
                os.replace(part, dest)
                return n, h.hexdigest()
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                    raise
                try:
                    wait = int(e.headers.get('Retry-After') or 0)
                except ValueError:
                    wait = 0
                wait, why = min(max(wait, 15 * (attempt + 1)), 300), f'HTTP {e.code}'
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == tries - 1:
                    raise
                wait, why = 10 * (attempt + 1), str(getattr(e, 'reason', e))
            print(f'   {why}; retrying in {wait} s: {url[:110]}', flush=True)
            time.sleep(wait)


# ---------------------------------------------------------------- table converter

def no_nan(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, list):
        return [no_nan(x) for x in v]
    if isinstance(v, dict):
        return {k: no_nan(x) for k, x in v.items()}
    return v


def convert_table(spec, parquet, dest):
    import pyarrow.parquet as pq
    table = pq.read_table(parquet)
    names = table.column_names
    cols = names if spec['columns'] == 'all' else spec['columns']
    absent = [c for c in cols if c not in names]
    if absent:
        raise ValueError(f'columns not in the parquet file: {absent}')
    add = spec.get('add') or {}
    if set(add) - {'n_options'}:
        raise ValueError(f'unknown derived columns: {sorted(add)}')
    order = spec['keys']                     # the column order of the archived table (revision of 2026-11-17)
    if sorted(order) != sorted(list(cols) + list(add)):
        raise ValueError('the recorded column order does not name the converted columns')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + '.part'
    with open(part, 'w', encoding='utf-8', newline='\n') as f:
        for row in table.to_pylist():
            out = {c: no_nan(row[c]) for c in cols}
            if 'n_options' in add:
                out['n_options'] = len(row['options'])
            f.write(json.dumps({k: out[k] for k in order}, ensure_ascii=False) + '\n')
    os.replace(part, dest)


# ---------------------------------------------------------------- replay

def find_root(index_path):
    """The folder the index's manifest paths are relative to: the package root (the index's own folder) or, in the
    authors' working tree, the parent of release/."""
    index = json.load(open(index_path, encoding='utf-8'))
    probe = index['sets'][0]['manifest']
    here = os.path.dirname(os.path.abspath(index_path))
    for d in (here, os.path.dirname(here)):
        if os.path.exists(os.path.join(d, *probe.split('/'))):
            return index, d
    raise SystemExit(f'{probe} not found next to {index_path} or in its parent folder')


def replay(args):
    index, root = find_root(args.index)
    sets = {s['id']: s for s in index['sets']}
    if args.set not in sets:
        raise SystemExit(f'no set {args.set!r} in {args.index}; sets: {", ".join(sets)}')
    spec = sets[args.set]
    manifest_path = os.path.join(root, *spec['manifest'].split('/'))
    if sha256_file(manifest_path) != spec['manifest_sha256']:
        raise SystemExit(f'{spec["manifest"]} does not have the SHA-256 recorded in the index')
    rows = read_manifest(manifest_path)
    if spec.get('only'):
        rows = [r for r in rows if r['path'] in spec['only']]
    into = os.path.abspath(args.into or os.path.join(root, *spec['into'].split('/')))
    os.makedirs(into, exist_ok=True)
    cache = os.path.join(into + '.upstream')        # parquet files the tables are rebuilt from
    report = args.report or os.path.join(into, 'replay_report.tsv')
    fetcher = Fetcher(args.rate)
    results = {}
    by_url = {r['url']: r for r in rows}

    def record(row, kind, status, check='', size='', digest='', note=''):
        results[row['path']] = (row['path'], kind, status, check, row['bytes'], str(size), row['sha256'], digest, note)

    def present(row, dest):
        return (not args.force and os.path.exists(dest) and os.path.getsize(dest) == int(row['bytes'])
                and sha256_file(dest) == row['sha256'])

    def do(row):
        kind = kind_of(row['url'])
        dest = inside(into, row['path'])
        try:
            if kind in ('member', 'derived'):
                return row, kind, None                    # after the downloads
            if present(row, dest):
                record(row, kind, 'present', 'bytes', row['bytes'], row['sha256'])
                return row, kind, True
            if args.check_only:
                why = 'missing' if not os.path.exists(dest) else 'differs from the recorded SHA-256'
                record(row, kind, 'FAIL', '', note=why + '; run with --fetch to replay it')
                return row, kind, False
            if kind == 'table':
                return row, kind, replay_table(row, dest)
            n, digest = fetcher.get(row['url'], dest, 'application/vnd.github+json' if kind == 'listing' else None)
            if n == int(row['bytes']) and digest == row['sha256']:
                record(row, kind, 'ok', 'bytes', n, digest)
                return row, kind, True
            if kind == 'listing':
                want = spec['json_documents'][row['path']]['normalized_sha256']
                try:
                    got = rule_j(dest)
                except ValueError as e:
                    got = f'not JSON ({e})'
                if got == want:
                    record(row, kind, 'ok', 'normalized', n, digest, 'bytes differ; normalized JSON content equal (rule J)')
                    return row, kind, True
                record(row, kind, 'FAIL', 'normalized', n, digest, f'normalized JSON hash {got[:16]} != {want[:16]}')
                return row, kind, False
            record(row, kind, 'FAIL', 'bytes', n, digest, 'size or SHA-256 differs from the manifest')
            return row, kind, False
        except Exception as e:                            # a network error ends this file, not the others
            record(row, kind, 'FAIL', '', note=f'{type(e).__name__}: {e}'[:300])
            return row, kind, False

    def replay_table(row, dest):
        t = spec['tables'][row['path']]
        url = (f"https://huggingface.co/datasets/{t['dataset']}/resolve/{t['revision']}/"
               + urllib.parse.quote(t['parquet']))
        # short names: Windows refuses paths over 260 characters unless long paths are enabled
        pq_dest = os.path.join(cache, f"{t['dataset'].replace('/', '__')}@{t['revision'][:12]}",
                               os.path.basename(t['parquet']))
        if not (os.path.exists(pq_dest) and sha256_file(pq_dest) == t['parquet_sha256']):
            fetcher.get(url, pq_dest)
        pq_size, pq_sha = os.path.getsize(pq_dest), sha256_file(pq_dest)
        if (pq_size, pq_sha) != (t['parquet_bytes'], t['parquet_sha256']):
            record(row, 'table', 'FAIL', '', note=f"parquet {t['parquet']} at {t['revision'][:10]}: size or SHA-256 "
                                                  f"differs from the hub's listing ({pq_sha[:16]})")
            return False
        try:
            convert_table(t, pq_dest, dest)
        except Exception as e:
            record(row, 'table', 'FAIL', '', note=f'converter: {type(e).__name__}: {e}'[:300])
            return False
        n, digest = os.path.getsize(dest), sha256_file(dest)
        src = f"from {t['dataset']}@{t['revision'][:10]}/{t['parquet']} (sha256 {pq_sha[:16]})"
        if n == int(row['bytes']) and digest == row['sha256']:
            record(row, 'table', 'ok', 'bytes', n, digest, src)
            return True
        got, rows_n, _ = rule_t(dest)
        if got == t['normalized_sha256'] and rows_n == t['rows']:
            record(row, 'table', 'ok', 'normalized', n, digest, f'bytes differ; content equal under rule T ({got[:16]}); {src}')
            return True
        record(row, 'table', 'FAIL', 'normalized', n, digest,
               f"rule T hash {got[:16]} != {t['normalized_sha256'][:16]}, rows {rows_n} vs {t['rows']}; {src}")
        return False

    t0 = time.time()
    first = [r for r in rows if kind_of(r['url']) not in ('member', 'derived')]
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row, kind, ok in pool.map(do, first):
            done += 1
            if done % 500 == 0 or done == len(first):
                print(f'   {done} of {len(first)} files ({time.time() - t0:.0f} s)', flush=True)
    for row in rows:                                     # archive members and derived listings
        kind = kind_of(row['url'])
        if kind not in ('member', 'derived'):
            continue
        dest = inside(into, row['path'])
        try:
            if present(row, dest):
                record(row, kind, 'present', 'bytes', row['bytes'], row['sha256'])
                continue
            if args.check_only:
                record(row, kind, 'FAIL', '', note='missing or different; run with --fetch to replay it')
                continue
            if kind == 'member':
                rebuild_member(row, dest, by_url, into)
            else:
                rebuild_archive_members(spec['derived'][row['path']], dest, into)
            n, digest = os.path.getsize(dest), sha256_file(dest)
            ok = n == int(row['bytes']) and digest == row['sha256']
            record(row, kind, 'ok' if ok else 'FAIL', 'bytes', n, digest, '' if ok else 'differs from the manifest')
        except Exception as e:
            record(row, kind, 'FAIL', '', note=f'{type(e).__name__}: {e}'[:300])
    with open(report, 'w', encoding='utf-8', newline='\n') as f:
        f.write(REPORT_HEAD)
        for r in rows:
            f.write('\t'.join(results[r['path']]) + '\n')
    status = [results[r['path']][2] for r in rows]
    failed = [r['path'] for r in rows if results[r['path']][2] == 'FAIL']
    normalized = sum(1 for r in rows if results[r['path']][3] == 'normalized' and results[r['path']][2] == 'ok')
    size = sum(int(r['bytes']) for r in rows)
    print(f"{args.set}: {len(rows)} files, {size / 1e6:.1f} MB; ok {status.count('ok')} (normalized {normalized}), "
          f"present {status.count('present')}, failed {len(failed)}; {time.time() - t0:.0f} s; report {report}")
    for p in failed[:10]:
        print(f'   FAIL {p}: {results[p][8]}')
    return 1 if failed else 0


def rebuild_member(row, dest, by_url, into):
    archive_row = by_url[row['url'][:-len(' (member)')]]
    folder = os.path.dirname(archive_row['path'])
    prefix = (folder + '/' if folder else '') + 'extracted/'
    if not row['path'].startswith(prefix):
        raise ValueError(f'member path not under {prefix}')
    member = row['path'][len(prefix):]
    inside(into, member)                                  # refuses names that could leave the folder
    with zipfile.ZipFile(inside(into, archive_row['path'])) as z, z.open(member) as src:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest + '.part', 'wb') as out:
            for block in iter(lambda: src.read(1 << 20), b''):
                out.write(block)
    os.replace(dest + '.part', dest)


def rebuild_archive_members(rule, dest, into):
    """archive_members.tsv as the fetch script wrote it: for each file of the listing, sorted by name, the members
    of the zip archive (name and uncompressed size) or '(not a zip archive)'."""
    listing = json.load(open(inside(into, rule['listing']), encoding='utf-8'))
    files = sorted((x['name'], x['size']) for x in listing if x['type'] == 'file')
    rows = []
    for name, size in files:
        target = inside(into, f"{rule['folder']}/{name}")
        if zipfile.is_zipfile(target):
            with zipfile.ZipFile(target) as zf:
                rows += [(name, info.filename, info.file_size) for info in zf.infolist()]
        else:
            rows.append((name, '(not a zip archive)', size))
    with open(dest + '.part', 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('archive\tmember\tbytes\n')
        for r in rows:
            fh.write(f'{r[0]}\t{r[1]}\t{r[2]}\n')
    os.replace(dest + '.part', dest)


# ---------------------------------------------------------------- run.json of the MMLU study

def run_of_url(url):
    parts = urllib.parse.urlsplit(url).path.split('/')
    i = parts.index('resolve')
    return parts[i + 1], urllib.parse.unquote(parts[i + 2])


def derive_run_json(manifest_rows, alt_rows, status):
    """{relative path: content} of every run.json, by the rule in the module docstring."""
    out, main = {}, {}
    for r in manifest_rows:
        if r['path'].startswith('details/'):
            repo = r['path'].split('/')[1]
            main.setdefault(repo, set()).add(run_of_url(r['url']))
    for repo, st in status.items():
        if st == 'ok':
            runs = main.get(repo, set())
            if len(runs) != 1:
                raise SystemExit(f'{repo}: {len(runs)} (revision, run) pairs in the manifest URLs, expected 1')
            (sha, run), = runs
            out[f'details/{repo}/run.json'] = {'status': 'ok', 'sha': sha, 'run': run}
        else:
            out[f'details/{repo}/run.json'] = {'status': st}
    alt = {}
    for r in alt_rows:
        parts = r['path'].split('/')
        alt.setdefault('/'.join(parts[:3]), set()).add(run_of_url(r['url']))
    for folder, runs in alt.items():
        if len(runs) != 1:
            raise SystemExit(f'{folder}: {len(runs)} (revision, run) pairs in the manifest URLs, expected 1')
        (sha, run), = runs
        out[f'{folder}/run.json'] = {'status': 'ok', 'sha': sha, 'run': run}
    return out


def run_json(args):
    index, root = find_root(args.index)
    rule = index['run_json']
    manifest = read_manifest(os.path.join(root, *rule['manifest'].split('/')))
    alt = read_manifest(os.path.join(root, *rule['manifest_alt'].split('/')))
    status = json.load(open(os.path.join(root, *rule['fetch_status'].split('/')), encoding='utf-8'))
    files = derive_run_json(manifest, alt, status)
    for rel, content in files.items():
        dest = inside(args.raw, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, 'w', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(content))
    print(f"wrote {len(files)} run.json files ({sum(1 for c in files.values() if c['status'] == 'ok')} with status ok)")
    return 0


# ---------------------------------------------------------------- index (how filter_inputs.json was made)

SETS = [
    # id, tiers, manifest (relative to ROOT), only these paths (None = all), folder the files are replayed into
    ('subsets', ['filters', 'filters-upstream'], 'experiments/results/subsets/raw/manifest.tsv', None,
     'experiments/pilot_data/filters/subsets/raw'),
    ('mmlupro_filter', ['filters', 'filters-upstream'], 'experiments/results/mmlupro_filter/raw/manifest.tsv', None,
     'experiments/pilot_data/filters/mmlupro_filter/raw'),
    ('mmlu_questions', ['filters', 'filters-upstream'], 'experiments/results/mmlu_reconstruction/raw/manifest.tsv',
     ['cais_mmlu_all_test.parquet'], 'experiments/pilot_data/filters/mmlu_reconstruction/raw'),
    ('mmlu_details', ['filters-upstream'], 'experiments/results/mmlu_reconstruction/raw/manifest.tsv', None,
     'experiments/pilot_data/filters/mmlu_reconstruction/raw'),
    ('mmlu_details_alt', ['filters-upstream'], 'experiments/results/mmlu_reconstruction/raw/manifest_alt.tsv', None,
     'experiments/pilot_data/filters/mmlu_reconstruction/raw'),
]


def hub_files(dataset, revision):
    url = f'https://huggingface.co/api/datasets/{dataset}/revision/{revision}?blobs=true'
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA}), timeout=60) as r:
        info = json.load(r)
    if info.get('sha') != revision:
        raise SystemExit(f'{dataset}: the hub answered for revision {info.get("sha")}, not {revision}')
    return {s['rfilename']: s for s in info['siblings']}


def make_index(args):
    root = os.path.abspath(args.root)
    out = {'schema': 'scopecard-filter-inputs/1',
           'about': ('Inputs of the filter studies. Every downloaded file is listed, with its size and SHA-256, in '
                     'the manifest that the study\'s fetch script wrote at download time; this index names those '
                     'manifests, the folders the files are replayed into, and, for the tables that came through the '
                     'Hugging Face rows API, the parquet file each is rebuilt from and its normalized content hash. '
                     'Written before any replay, from the archived copies. replay_inputs.py states the rules. '
                     'inputs.json, the inputs of the WN18RR study and of the scope cards, is unchanged.'),
           'created': datetime.now().astimezone().isoformat(timespec='seconds'),
           'sets': []}
    hub_cache = {}
    for sid, tiers, manifest, only, into in SETS:
        mpath = os.path.join(root, *manifest.split('/'))
        rows = read_manifest(mpath)
        if only:
            rows = [r for r in rows if r['path'] in only]
            if len(rows) != len(only):
                raise SystemExit(f'{sid}: not every path of {only} is in {manifest}')
        raw = os.path.dirname(mpath)
        entry = {'id': sid, 'tiers': tiers, 'manifest': manifest, 'manifest_sha256': sha256_file(mpath),
                 'into': into, 'files': len(rows), 'bytes': sum(int(r['bytes']) for r in rows),
                 'kinds': {}}
        if only:
            entry['only'] = only
        for r in rows:
            k = kind_of(r['url'])
            entry['kinds'][k] = entry['kinds'].get(k, 0) + 1
        tables, docs, derived = {}, {}, {}
        for r in rows:
            k = kind_of(r['url'])
            local = os.path.join(raw, *r['path'].split('/'))
            if k in ('table', 'listing', 'derived'):
                if sha256_file(local) != r['sha256']:
                    raise SystemExit(f'{sid}: archived {r["path"]} does not have its recorded SHA-256')
            if k == 'table':
                q = urllib.parse.parse_qs(urllib.parse.urlsplit(r['url'].split(' ')[0]).query)
                dataset, split = q['dataset'][0], q['split'][0]
                m = re.search(r'\(columns: (.*)\)$', r['url'])
                if dataset == 'TIGER-Lab/MMLU-Pro':
                    columns, add = ['question_id', 'category', 'src', 'answer', 'answer_index'], {'n_options': 'number of options'}
                elif m.group(1) == 'all':
                    columns, add = 'all', None
                else:
                    columns, add = m.group(1).split(','), None
                key = (dataset, r['upstream_revision'])
                if key not in hub_cache:
                    hub_cache[key] = hub_files(dataset, r['upstream_revision'])
                    time.sleep(1)
                files = hub_cache[key]
                shards = sorted(f for f in files if f.startswith(f'data/{split}-') and f.endswith('.parquet'))
                if shards != [f'data/{split}-00000-of-00001.parquet']:
                    raise SystemExit(f'{dataset} {split}: parquet files {shards}, expected one shard')
                lfs = files[shards[0]].get('lfs') or {}
                normalized, n_rows, keys = rule_t(local)
                tables[r['path']] = {'dataset': dataset, 'split': split, 'revision': r['upstream_revision'],
                                     'parquet': shards[0], 'parquet_bytes': lfs.get('size', files[shards[0]].get('size')),
                                     'parquet_sha256': lfs.get('sha256'), 'columns': columns, 'add': add,
                                     'rows': n_rows, 'keys': keys, 'sha256': r['sha256'],
                                     'normalized_sha256': normalized}
                if not tables[r['path']]['parquet_sha256']:
                    raise SystemExit(f'{dataset}: the hub lists no SHA-256 for {shards[0]}')
            elif k == 'listing':
                docs[r['path']] = {'sha256': r['sha256'], 'normalized_sha256': rule_j(local)}
            elif k == 'derived':
                listing = next(x['path'] for x in rows if kind_of(x['url']) == 'listing')
                folder = urllib.parse.urlsplit(next(x['url'] for x in rows if kind_of(x['url']) == 'listing')).path.split('/contents/')[1]
                derived[r['path']] = {'rule': 'archive_members', 'listing': listing, 'folder': folder}
        if tables:
            entry['tables'] = tables
        if docs:
            entry['json_documents'] = docs
        if derived:
            entry['derived'] = derived
        out['sets'].append(entry)
    # the run.json rule, checked against the archived files
    mraw = os.path.join(root, 'experiments', 'results', 'mmlu_reconstruction', 'raw')
    manifest = read_manifest(os.path.join(mraw, 'manifest.tsv'))
    alt = read_manifest(os.path.join(mraw, 'manifest_alt.tsv'))
    status = json.load(open(os.path.join(mraw, 'fetch_status.json'), encoding='utf-8'))
    files = derive_run_json(manifest, alt, status)
    same = 0
    for rel, content in files.items():
        old = json.load(open(os.path.join(mraw, *rel.split('/')), encoding='utf-8'))
        if all(old.get(k) == v for k, v in content.items()):
            same += 1
    if same != len(files):
        raise SystemExit(f'run.json rule: {same} of {len(files)} archived files agree')
    on_disk = sum(1 for _ in pathlib.Path(mraw).rglob('run.json'))
    out['run_json'] = {'manifest': 'experiments/results/mmlu_reconstruction/raw/manifest.tsv',
                       'manifest_alt': 'experiments/results/mmlu_reconstruction/raw/manifest_alt.tsv',
                       'fetch_status': 'experiments/results/mmlu_reconstruction/raw/fetch_status.json',
                       'check': (f'the rule gives {len(files)} run.json files; each agrees with the archived file in '
                                 f'every field it writes ({on_disk} archived run.json files in all)')}
    with open(args.out, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write('\n')
    print(f'wrote {args.out}: ' + '; '.join(f"{s['id']} {s['files']} files {s['bytes'] / 1e6:.1f} MB" for s in out['sets']))
    print(out['run_json']['check'])
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('replay')
    r.add_argument('index')
    r.add_argument('set')
    r.add_argument('--into')
    r.add_argument('--check-only', action='store_true')
    r.add_argument('--force', action='store_true')
    r.add_argument('--report')
    r.add_argument('--workers', type=int, default=4)
    r.add_argument('--rate', type=float, default=6.0, help='requests per second at most (default 6)')
    j = sub.add_parser('run-json')
    j.add_argument('index')
    j.add_argument('raw')
    i = sub.add_parser('index')
    i.add_argument('root')
    i.add_argument('out')
    a = ap.parse_args()
    return {'replay': replay, 'run-json': run_json, 'index': make_index}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
