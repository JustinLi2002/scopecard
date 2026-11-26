"""Download the public inputs of the MMLU-Pro filter study (experiments/MMLUPRO_FILTER_DESIGN.md, Section 2).

Standard library only. The archives are saved as they are and never extracted to disk or executed; the analysis
reads their members in memory. The question table is read through the Hugging Face datasets-server rows API and
only the columns needed are kept.

Usage:  python -I scripts/mmlupro_fetch.py <output dir>      (run from experiments/)
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
REPO = 'TIGER-AI-Lab/MMLU-Pro'
DATASET = 'TIGER-Lab/MMLU-Pro'
MANIFEST = []


def get(url, accept=None, tries=6):
    headers = {'User-Agent': UA}
    if accept:
        headers['Accept'] = accept
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=180) as resp:
                return resp.read()
        except urllib.error.HTTPError as err:
            if err.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
            wait = 20 * (attempt + 1)
            print(f'   HTTP {err.code}, waiting {wait} s: {url[:90]}', flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as err:
            if attempt == tries - 1:
                raise
            print(f'   {err}, retrying: {url[:90]}', flush=True)
            time.sleep(10)


def record(path, url, revision):
    data = path.read_bytes()
    MANIFEST.append((path.relative_to(OUT).as_posix(), str(len(data)), hashlib.sha256(data).hexdigest(), url, revision, TODAY))


def fetch_outputs():
    folder = OUT / 'eval_results'
    folder.mkdir(parents=True, exist_ok=True)
    pin = OUT / 'PINNED_COMMIT.txt'
    if pin.exists():
        sha = pin.read_text(encoding='utf-8').split()[0]
    else:
        info = json.loads(get(f'https://api.github.com/repos/{REPO}', 'application/vnd.github+json'))
        commit = json.loads(get(f'https://api.github.com/repos/{REPO}/commits/{info["default_branch"]}', 'application/vnd.github+json'))
        sha = commit['sha']
        pin.write_text(sha + '\n', encoding='utf-8')
    listing = OUT / 'eval_results_listing.json'
    if not listing.exists():
        listing.write_bytes(get(f'https://api.github.com/repos/{REPO}/contents/eval_results?ref={sha}', 'application/vnd.github+json'))
    record(listing, f'https://api.github.com/repos/{REPO}/contents/eval_results?ref={sha}', sha)
    files = sorted((x['name'], x['size']) for x in json.loads(listing.read_text(encoding='utf-8')) if x['type'] == 'file')
    print(f'{REPO} at {sha[:10]}: {len(files)} files, {sum(s for _, s in files) / 1e6:.1f} MB listed', flush=True)
    members = []
    for name, size in files:
        target = folder / name
        url = f'https://raw.githubusercontent.com/{REPO}/{sha}/eval_results/{urllib.parse.quote(name)}'
        if not target.exists():
            target.write_bytes(get(url))
            time.sleep(0.5)
        if target.stat().st_size != size:
            raise SystemExit(f'{name}: {target.stat().st_size} bytes on disk, {size} listed')
        record(target, url, sha)
        if zipfile.is_zipfile(target):
            with zipfile.ZipFile(target) as zf:
                for info in zf.infolist():
                    members.append((name, info.filename, info.file_size))
        else:
            members.append((name, '(not a zip archive)', size))
    with (OUT / 'archive_members.tsv').open('w', encoding='utf-8', newline='\n') as fh:
        fh.write('archive\tmember\tbytes\n')
        for row in members:
            fh.write(f'{row[0]}\t{row[1]}\t{row[2]}\n')
    record(OUT / 'archive_members.tsv', '(derived: member listing of the archives)', sha)
    print(f'   {len(members)} archive members, {sum(m[2] for m in members) / 1e6:.0f} MB uncompressed (not extracted)', flush=True)


def fetch_questions():
    target = OUT / 'mmlu_pro_test_questions.jsonl'
    revision = json.loads(get(f'https://huggingface.co/api/datasets/{DATASET}'))['sha']
    base = 'https://datasets-server.huggingface.co/rows?' + urllib.parse.urlencode({'dataset': DATASET, 'config': 'default', 'split': 'test'})
    if not target.exists():
        rows, offset, total = [], 0, None
        while total is None or offset < total:
            page = json.loads(get(f'{base}&offset={offset}&length=100'))
            total = page['num_rows_total']
            if not page['rows']:
                break
            for item in page['rows']:
                row = item['row']
                if any(c in item.get('truncated_cells', []) for c in ('question_id', 'category', 'src', 'answer', 'answer_index', 'options')):
                    raise SystemExit(f'truncated cell in row {item["row_idx"]}; not saved')
                rows.append({'question_id': row['question_id'], 'category': row['category'], 'src': row['src'],
                             'answer': row['answer'], 'answer_index': row['answer_index'], 'n_options': len(row['options'])})
            offset += len(page['rows'])
            time.sleep(1.0)
        if len(rows) != total:
            raise SystemExit(f'got {len(rows)} of {total} rows; not saved')
        with target.open('w', encoding='utf-8', newline='\n') as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    n = sum(1 for _ in target.open(encoding='utf-8'))
    record(target, base + ' (columns: question_id, category, src, answer, answer_index; n_options = number of options)', revision)
    print(f'{DATASET} [test] at {revision[:10]}: {n} rows', flush=True)


OUT.mkdir(parents=True, exist_ok=True)
fetch_outputs()
fetch_questions()
with (OUT / 'manifest.tsv').open('w', encoding='utf-8', newline='\n') as fh:
    fh.write('path\tbytes\tsha256\turl\tupstream_revision\tretrieved\n')
    for row in MANIFEST:
        fh.write('\t'.join(row) + '\n')
print(f'manifest: {len(MANIFEST)} files, {sum(int(r[1]) for r in MANIFEST) / 1e6:.1f} MB')
