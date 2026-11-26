"""Fetch an earlier complete run for a repository whose latest complete run fails the row-count gate
(experiments/MMLU_RECONSTRUCTION_DESIGN.md, Section 8, deviation 2). Standard library only; nothing is executed.

Each call fetches, for every repository named, the latest complete run that is not on disk yet, into
<raw dir>/details_alt/<repository>/<run>/ (":" in a run name becomes "-"). Files are pinned to the revision
recorded by mmlu_reconstruction_fetch.py and listed in <raw dir>/manifest_alt.tsv.

Usage:  python -I scripts/mmlu_reconstruction_fetch_alt.py <raw dir> <repository> [<repository> ...]      (run from experiments/)
"""
import datetime
import hashlib
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

OUT = pathlib.Path(sys.argv[1])
UA = 'Mozilla/5.0 (compatible; research-tool)'
TODAY = datetime.date.today().isoformat()
ORG, N_SUBJECTS = 'open-llm-leaderboard-old', 57
MANIFEST = OUT / 'manifest_alt.tsv'


def get(url, tries=7):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA}), timeout=180) as resp:
                return resp.read()
        except urllib.error.HTTPError as err:
            if err.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
            wait = 30 * (attempt + 1)
            print(f'   HTTP {err.code}, waiting {wait} s', flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
            if attempt == tries - 1:
                raise
            print(f'   {type(err).__name__}, retrying', flush=True)
            time.sleep(15)


def safe(run):
    return run.replace(':', '-')


if not MANIFEST.exists():
    MANIFEST.write_text('path\tbytes\tsha256\turl\tupstream_revision\tretrieved\n', encoding='utf-8')
for repo in sys.argv[2:]:
    primary = json.loads((OUT / 'details' / repo / 'run.json').read_text(encoding='utf-8'))
    sha = primary['sha']
    base = OUT / 'details_alt' / repo
    earlier = sorted((r for r in primary.get('runs_complete', []) if r != primary.get('run')), key=safe, reverse=True)
    todo = [r for r in earlier if not (base / safe(r) / 'run.json').exists()]
    if not todo:
        print(f'{repo}: no further complete run to fetch (complete runs: {primary.get("runs_complete", [])})', flush=True)
        continue
    run = todo[0]
    info = json.loads(get(f'https://huggingface.co/api/datasets/{ORG}/{repo}/revision/{sha}'))
    files = {}
    for s in info.get('siblings', []):
        m = re.match(r'^([^/]+)/details_harness\|hendrycksTest-([a-z_]+)\|5_.*\.parquet$', s['rfilename'])
        if m and m.group(1) == run:
            files[m.group(2)] = s['rfilename']
    if len(files) != N_SUBJECTS:
        print(f'{repo}: run {run} lists {len(files)} subject files at revision {sha[:10]}, expected {N_SUBJECTS}; skipped', flush=True)
        continue
    folder = base / safe(run)
    folder.mkdir(parents=True, exist_ok=True)
    size = 0
    for subject, path in sorted(files.items()):
        target = folder / f'{subject}.parquet'
        url = f'https://huggingface.co/datasets/{ORG}/{repo}/resolve/{sha}/{urllib.parse.quote(path)}'
        if not target.exists():
            data = get(url)
            target.write_bytes(data)
            with MANIFEST.open('a', encoding='utf-8', newline='\n') as fh:
                fh.write(f'{target.relative_to(OUT).as_posix()}\t{len(data)}\t{hashlib.sha256(data).hexdigest()}\t{url}\t{sha}\t{TODAY}\n')
            time.sleep(0.05)
        size += target.stat().st_size
    (folder / 'run.json').write_text(json.dumps({'status': 'ok', 'sha': sha, 'run': run}), encoding='utf-8')
    print(f'{repo}: fetched run {run} ({size / 1e6:.1f} MB); complete runs still not on disk: {len(todo) - 1}', flush=True)
