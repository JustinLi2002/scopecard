"""Download the public inputs of the released-subset study (experiments/SUBSET_C_DESIGN.md, Section 3).

Standard library only. Nothing downloaded is executed: the SWE-bench Lite filter code is saved with a .txt
suffix and read by a person, and the annotation archive is extracted with path checks into its own directory.

Hugging Face tables are read through the datasets-server rows API (JSON), because this machine has no parquet
reader. Only the columns named below are kept, one JSON object per line; the hub revision of each dataset is
recorded in the manifest.

Usage:  python -I scripts/subsets_fetch.py <output dir>      (run from experiments/)
Existing files are kept; delete a file to fetch it again.
"""
import datetime
import hashlib
import json
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

OUT = pathlib.Path(sys.argv[1])
UA = 'Mozilla/5.0 (compatible; research-tool)'
TODAY = datetime.date.today().isoformat()
MANIFEST = []          # (relative path, bytes, sha256, url, upstream revision, date)

GH_EXPERIMENTS = 'SWE-bench/experiments'
GH_SWEBENCH = 'SWE-bench/SWE-bench'
LITE_CODE = ['swebench/collect/make_lite/criteria.py', 'swebench/collect/make_lite/make_lite.py',
             'swebench/collect/make_lite/README.md']
ANNOTATIONS_URL = 'https://cdn.openai.com/introducing-swe-bench-verified/swe-bench-annotation-results.zip'
# dataset id, split, columns kept (None = all), output file name
HF_TABLES = [
    ('princeton-nlp/SWE-bench', 'test',
     ['instance_id', 'repo', 'created_at', 'version', 'problem_statement', 'patch', 'test_patch'], 'swebench_test.jsonl'),
    ('princeton-nlp/SWE-bench_Lite', 'test', ['instance_id', 'repo'], 'swebench_lite_test.jsonl'),
    ('princeton-nlp/SWE-bench_Verified', 'test', ['instance_id', 'repo', 'difficulty'], 'swebench_verified_test.jsonl'),
    ('bigcode/bigcodebench', 'v0.1.4', ['task_id', 'libs', 'canonical_solution'], 'bigcodebench_v0.1.4.jsonl'),
    ('bigcode/bigcodebench-hard', 'v0.1.4', ['task_id', 'libs', 'q_idx', 'score'], 'bigcodebench_hard_v0.1.4.jsonl'),
    ('bigcode/bigcodebench-perf', 'complete', None, 'bigcodebench_perf_complete.jsonl'),
    ('bigcode/bigcodebench-perf', 'instruct', None, 'bigcodebench_perf_instruct.jsonl'),
    ('bigcode/bigcodebench-solve-rate', 'complete', None, 'solve_rate_complete.jsonl'),
    ('bigcode/bigcodebench-solve-rate', 'instruct', None, 'solve_rate_instruct.jsonl'),
    ('bigcode/bigcodebench-complete-solve-rate', 'train', None, 'solve_rate_complete_20240618.jsonl'),
    ('bigcode/bigcodebench-instruct-solve-rate', 'train', None, 'solve_rate_instruct_20240618.jsonl'),
    ('bigcode/bigcodebench-results', 'train', None, 'bigcodebench_results.jsonl'),
]


def get(url, accept=None, tries=5):
    headers = {'User-Agent': UA}
    if accept:
        headers['Accept'] = accept
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as resp:
                return resp.read()
        except urllib.error.HTTPError as err:
            if err.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
            wait = 20 * (attempt + 1)
            print(f'   HTTP {err.code}, waiting {wait} s: {url[:90]}', flush=True)
            time.sleep(wait)
        except urllib.error.URLError as err:
            if attempt == tries - 1:
                raise
            print(f'   {err.reason}, retrying: {url[:90]}', flush=True)
            time.sleep(10)


def record(path, url, revision):
    data = path.read_bytes()
    MANIFEST.append((path.relative_to(OUT).as_posix(), str(len(data)), hashlib.sha256(data).hexdigest(), url, revision, TODAY))


def save(path, url, revision, accept=None):
    """Fetch url into path unless it exists; record it either way."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(get(url, accept))
        time.sleep(0.3)
    record(path, url, revision)


def github_head(repo):
    info = json.loads(get(f'https://api.github.com/repos/{repo}', 'application/vnd.github+json'))
    branch = info['default_branch']
    commit = json.loads(get(f'https://api.github.com/repos/{repo}/commits/{branch}', 'application/vnd.github+json'))
    return branch, commit['sha']


def fetch_swebench_outcomes():
    pin = OUT / 'swebench_experiments' / 'PINNED_COMMIT.txt'
    if pin.exists():
        sha = pin.read_text(encoding='utf-8').split()[0]
    else:
        _, sha = github_head(GH_EXPERIMENTS)
        pin.parent.mkdir(parents=True, exist_ok=True)
        pin.write_text(sha + '\n', encoding='utf-8')
    listing_path = OUT / 'swebench_experiments' / 'evaluation_test_listing.json'
    listing_url = f'https://api.github.com/repos/{GH_EXPERIMENTS}/contents/evaluation/test?ref={sha}'
    save(listing_path, listing_url, sha, 'application/vnd.github+json')
    names = sorted(x['name'] for x in json.loads(listing_path.read_text(encoding='utf-8')) if x['type'] == 'dir')
    print(f'SWE-bench experiments at {sha[:10]}: {len(names)} submissions under evaluation/test', flush=True)
    missing = []
    for name in names:
        for rel in ('results/results.json', 'metadata.yaml'):
            url = f'https://raw.githubusercontent.com/{GH_EXPERIMENTS}/{sha}/evaluation/test/{name}/{rel}'
            target = OUT / 'swebench_experiments' / 'test' / name / rel.split('/')[-1]
            try:
                save(target, url, sha)
            except urllib.error.HTTPError as err:
                missing.append(f'{name}/{rel} (HTTP {err.code})')
    if missing:
        print('   not found:', '; '.join(missing), flush=True)


def fetch_lite_code():
    pin = OUT / 'swebench_make_lite' / 'PINNED_COMMIT.txt'
    if pin.exists():
        sha = pin.read_text(encoding='utf-8').split()[0]
    else:
        _, sha = github_head(GH_SWEBENCH)
        pin.parent.mkdir(parents=True, exist_ok=True)
        pin.write_text(sha + '\n', encoding='utf-8')
    for rel in LITE_CODE:
        url = f'https://raw.githubusercontent.com/{GH_SWEBENCH}/{sha}/{rel}'
        save(OUT / 'swebench_make_lite' / (rel.split('/')[-1] + '.txt'), url, sha)
    print(f'SWE-bench Lite filter code at {sha[:10]}: {len(LITE_CODE)} files (saved as .txt, not executed)', flush=True)


def fetch_hf_table(dataset, split, columns, name):
    target = OUT / 'huggingface' / name
    revision = json.loads(get(f'https://huggingface.co/api/datasets/{dataset}'))['sha']
    base = 'https://datasets-server.huggingface.co/rows?' + urllib.parse.urlencode(
        {'dataset': dataset, 'config': 'default', 'split': split})
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        rows, offset, total, truncated = [], 0, None, 0
        while total is None or offset < total:
            page = json.loads(get(f'{base}&offset={offset}&length=100'))
            total = page['num_rows_total']
            if not page['rows']:
                break
            for item in page['rows']:
                keep = columns or list(item['row'])
                truncated += sum(1 for c in item.get('truncated_cells', []) if c in keep)
                rows.append({c: item['row'][c] for c in keep})
            offset += len(page['rows'])
            time.sleep(1.0)
        if truncated:
            raise SystemExit(f'{dataset} {split}: {truncated} truncated cells in kept columns; not saved')
        if len(rows) != total:
            raise SystemExit(f'{dataset} {split}: got {len(rows)} of {total} rows; not saved')
        with target.open('w', encoding='utf-8', newline='\n') as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    n = sum(1 for _ in target.open(encoding='utf-8'))
    record(target, f'{base} (columns: {"all" if columns is None else ",".join(columns)})', revision)
    print(f'{dataset} [{split}] at {revision[:10]}: {n} rows -> {name}', flush=True)


def fetch_annotations():
    folder = OUT / 'openai_annotations'
    archive = folder / 'swe-bench-annotation-results.zip'
    save(archive, ANNOTATIONS_URL, 'Last-Modified 2024-08-12')
    extracted = folder / 'extracted'
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        total = sum(i.file_size for i in infos)
        if total > 300_000_000:
            raise SystemExit(f'annotation archive expands to {total} bytes; not extracted')
        for info in infos:
            parts = pathlib.PurePosixPath(info.filename).parts
            if info.filename.startswith(('/', '\\')) or '..' in parts or ':' in info.filename:
                raise SystemExit(f'unsafe path in archive: {info.filename!r}')
        if not extracted.exists():
            extracted.mkdir(parents=True)
            zf.extractall(extracted)
    for path in sorted(p for p in extracted.rglob('*') if p.is_file()):
        record(path, ANNOTATIONS_URL + ' (member)', 'Last-Modified 2024-08-12')
    print(f'OpenAI annotations: {len(infos)} archive members, {total} bytes extracted', flush=True)


fetch_swebench_outcomes()
fetch_lite_code()
fetch_annotations()
for spec in HF_TABLES:
    fetch_hf_table(*spec)
OUT.mkdir(parents=True, exist_ok=True)
with (OUT / 'manifest.tsv').open('w', encoding='utf-8', newline='\n') as fh:
    fh.write('path\tbytes\tsha256\turl\tupstream_revision\tretrieved\n')
    for row in MANIFEST:
        fh.write('\t'.join(row) + '\n')
print(f'manifest: {len(MANIFEST)} files, {sum(int(r[1]) for r in MANIFEST) / 1e6:.1f} MB')
