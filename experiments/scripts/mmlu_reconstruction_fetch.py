"""Download the public inputs of the MMLU reconstruction study (experiments/MMLU_RECONSTRUCTION_DESIGN.md, Sections 2-3).

Standard library only. Parquet files are saved as they are; nothing is executed. The selection of models uses
repository names only and is made before any result is read. Resumable: existing files are kept.

Usage:  python -I scripts/mmlu_reconstruction_fetch.py <output dir>      (run from experiments/)
"""
import datetime
import hashlib
import json
import pathlib
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

OUT = pathlib.Path(sys.argv[1])
UA = 'Mozilla/5.0 (compatible; research-tool)'
TODAY = datetime.date.today().isoformat()
ORG = 'open-llm-leaderboard-old'
N_RANDOM, N_LINEAGE, N_SUBJECTS = 120, 80, 57
FILTER = ['meta-llama__Llama-2-7b-hf', 'meta-llama__Llama-2-7b-chat-hf', 'meta-llama__Llama-2-13b-hf',
          'meta-llama__Llama-2-13b-chat-hf', 'mistralai__Mistral-7B-v0.1', 'google__gemma-7b', '01-ai__Yi-6B']
# Clean pool: not derived from any filter model's weights (design, Section 3). True marks a same-line sibling.
CLEAN = {name: False for name in '''
tiiuae__falcon-7b tiiuae__falcon-7b-instruct tiiuae__falcon-40b tiiuae__falcon-40b-instruct tiiuae__falcon-rw-1b
mosaicml__mpt-7b mosaicml__mpt-7b-instruct mosaicml__mpt-30b mosaicml__mpt-30b-instruct
EleutherAI__pythia-2.8b EleutherAI__pythia-6.9b EleutherAI__pythia-12b EleutherAI__gpt-j-6b EleutherAI__gpt-neo-2.7B
gpt2-large gpt2-xl facebook__opt-6.7b facebook__opt-13b facebook__xglm-7.5B bigscience__bloom-7b1
databricks__dolly-v2-7b databricks__dolly-v2-12b
stabilityai__stablelm-base-alpha-7b stabilityai__stablelm-3b-4e1t stabilityai__stablelm-zephyr-3b stabilityai__stablelm-2-zephyr-1_6b
togethercomputer__RedPajama-INCITE-7B-Base togethercomputer__RedPajama-INCITE-7B-Instruct
openlm-research__open_llama_3b openlm-research__open_llama_7b openlm-research__open_llama_13b
microsoft__phi-1_5 microsoft__phi-2 microsoft__Phi-3-mini-4k-instruct microsoft__Phi-3-mini-128k-instruct
Qwen__Qwen-7B Qwen__Qwen-14B Qwen__Qwen-72B Qwen__Qwen1.5-0.5B Qwen__Qwen1.5-1.8B Qwen__Qwen1.5-4B Qwen__Qwen1.5-7B
Qwen__Qwen1.5-14B Qwen__Qwen1.5-32B Qwen__Qwen1.5-72B Qwen__Qwen1.5-7B-Chat Qwen__Qwen1.5-14B-Chat
deepseek-ai__deepseek-llm-7b-base deepseek-ai__deepseek-llm-7b-chat deepseek-ai__deepseek-llm-67b-base deepseek-ai__deepseek-llm-67b-chat
internlm__internlm-7b internlm__internlm-20b internlm__internlm2-7b internlm__internlm2-20b internlm__internlm2-base-7b
baichuan-inc__Baichuan2-7B-Base baichuan-inc__Baichuan2-13B-Base THUDM__chatglm2-6b
Salesforce__xgen-7b-4k-base Salesforce__xgen-7b-8k-base cerebras__Cerebras-GPT-6.7B cerebras__Cerebras-GPT-13B
bigcode__starcoder bigcode__starcoderbase TinyLlama__TinyLlama-1.1B-Chat-v1.0 TinyLlama__TinyLlama-1.1B-intermediate-step-1431k-3T
huggyllama__llama-7b huggyllama__llama-13b huggyllama__llama-30b huggyllama__llama-65b
meta-llama__Meta-Llama-3-8B meta-llama__Meta-Llama-3-8B-Instruct meta-llama__Meta-Llama-3-70B meta-llama__Meta-Llama-3-70B-Instruct
CohereForAI__c4ai-command-r-v01 CohereForAI__c4ai-command-r-plus
'''.split()}
CLEAN.update({name: True for name in '''
meta-llama__Llama-2-70b-hf meta-llama__Llama-2-70b-chat-hf 01-ai__Yi-9B 01-ai__Yi-34B 01-ai__Yi-34B-200K
google__gemma-2b google__gemma-2b-it mistralai__Mixtral-8x7B-v0.1 mistralai__Mixtral-8x7B-Instruct-v0.1
'''.split()})
LINEAGE = [('llama2-7b', ('llama-2-7b', 'llama2-7b')), ('llama2-13b', ('llama-2-13b', 'llama2-13b')),
           ('mistral-7b', ('mistral-7b',)), ('gemma-7b', ('gemma-7b',)), ('yi-6b', ('yi-6b',))]
MANIFEST = OUT / 'manifest.tsv'


def keyed(key):
    return random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big'))


def get(url, tries=7):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA}), timeout=180) as resp:
                return resp.read(), resp.headers.get('Link', '')
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


def manifest_line(path, url, revision):
    data = path.read_bytes()
    with MANIFEST.open('a', encoding='utf-8', newline='\n') as fh:
        fh.write(f'{path.relative_to(OUT).as_posix()}\t{len(data)}\t{hashlib.sha256(data).hexdigest()}\t{url}\t{revision}\t{TODAY}\n')


def listing():
    cache = OUT / 'archive_listing.json'
    if cache.exists():
        return json.loads(cache.read_text(encoding='utf-8'))
    names, url = [], f'https://huggingface.co/api/datasets?author={ORG}&limit=1000'
    while url:
        body, link = get(url)
        names += [d['id'].split('/', 1)[1] for d in json.loads(body)]
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
        time.sleep(1)
    cache.write_text(json.dumps(sorted(names)), encoding='utf-8')
    return sorted(names)


def select(names):
    """Returns [(repo, set, tag)], using names only (design, Section 3)."""
    details = [n for n in names if n.startswith('details_')]
    by_lower = {n.lower(): n for n in details}
    chosen, absent = [], []
    for model in FILTER:
        repo = by_lower.get(('details_' + model).lower())
        (chosen if repo else absent).append((repo or model, 'filter', ''))
    for model, sibling in CLEAN.items():
        repo = by_lower.get(('details_' + model).lower())
        (chosen if repo else absent).append((repo or model, 'clean', 'sibling' if sibling else ''))
    taken = {c[0] for c in chosen}

    def tag(name):
        low = name.lower()
        for label, keys in LINEAGE:
            if any(k in low for k in keys):
                return label
        return ''

    candidates = sorted(n for n in details if n not in taken and tag(n))
    lineage = keyed('mmlu_reconstruction/lineage').sample(candidates, min(N_LINEAGE, len(candidates)))
    chosen += [(n, 'lineage', tag(n)) for n in sorted(lineage)]
    rest = sorted(n for n in details if n not in taken and not tag(n))
    chosen += [(n, 'random', '') for n in sorted(keyed('mmlu_reconstruction/random').sample(rest, N_RANDOM))]
    with (OUT / 'selection.tsv').open('w', encoding='utf-8', newline='\n') as fh:
        fh.write('repository\tset\ttag\tin_archive\n')
        for repo, kind, label in chosen:
            fh.write(f'{repo}\t{kind}\t{label}\t1\n')
        for repo, kind, label in absent:
            fh.write(f'{repo}\t{kind}\t{label}\t0\n')
    print(f'archive: {len(details)} detail repositories; selected {len(chosen)} '
          f'({sum(c[1] == "filter" for c in chosen)} filter, {sum(c[1] == "clean" for c in chosen)} clean, '
          f'{sum(c[1] == "lineage" for c in chosen)} of {len(candidates)} lineage, {sum(c[1] == "random" for c in chosen)} random); '
          f'named but absent: {len(absent)}', flush=True)
    return chosen


def fetch_repo(repo):
    folder = OUT / 'details' / repo
    done = folder / 'run.json'
    if done.exists():
        return json.loads(done.read_text(encoding='utf-8')).get('status', 'ok')
    folder.mkdir(parents=True, exist_ok=True)
    info = json.loads(get(f'https://huggingface.co/api/datasets/{ORG}/{repo}')[0])
    sha = info.get('sha', '')
    runs = {}
    for s in info.get('siblings', []):
        path = s['rfilename']
        m = re.match(r'^([^/]+)/details_harness\|hendrycksTest-([a-z_]+)\|5_.*\.parquet$', path)
        if m:
            runs.setdefault(m.group(1), {})[m.group(2)] = path
    complete = {run: files for run, files in runs.items() if len(files) == N_SUBJECTS}
    if not complete:
        done.write_text(json.dumps({'status': 'no complete run', 'sha': sha, 'runs': {r: len(f) for r, f in runs.items()}}), encoding='utf-8')
        return 'no complete run'
    run = max(complete, key=lambda r: r.replace(':', '-'))
    for subject, path in sorted(complete[run].items()):
        target = folder / f'{subject}.parquet'
        url = f'https://huggingface.co/datasets/{ORG}/{repo}/resolve/{sha}/{urllib.parse.quote(path)}'
        if not target.exists():
            target.write_bytes(get(url)[0])
            manifest_line(target, url, sha)
            time.sleep(0.05)
    done.write_text(json.dumps({'status': 'ok', 'sha': sha, 'run': run, 'runs_complete': sorted(complete)}), encoding='utf-8')
    return 'ok'


OUT.mkdir(parents=True, exist_ok=True)
if not MANIFEST.exists():
    MANIFEST.write_text('path\tbytes\tsha256\turl\tupstream_revision\tretrieved\n', encoding='utf-8')
selected = select(listing())
mmlu = OUT / 'cais_mmlu_all_test.parquet'
if not mmlu.exists():
    info = json.loads(get('https://huggingface.co/api/datasets/cais/mmlu')[0])
    paths = [s['rfilename'] for s in info['siblings'] if re.match(r'^all/test-.*\.parquet$', s['rfilename'])]
    if len(paths) != 1:
        raise SystemExit(f'cais/mmlu: expected one all/test parquet file, found {paths}')
    url = f'https://huggingface.co/datasets/cais/mmlu/resolve/{info["sha"]}/{paths[0]}'
    mmlu.write_bytes(get(url)[0])
    manifest_line(mmlu, url, info['sha'])
print(f'cais/mmlu all/test: {mmlu.stat().st_size / 1e6:.1f} MB', flush=True)
status = {}
for k, (repo, kind, label) in enumerate(selected):
    try:
        status[repo] = fetch_repo(repo)
    except Exception as err:                      # one failing repository must not stop the rest
        status[repo] = f'error: {type(err).__name__} {err}'[:200]
    if (k + 1) % 20 == 0 or k + 1 == len(selected):
        ok = sum(v == 'ok' for v in status.values())
        size = sum(p.stat().st_size for p in (OUT / 'details').rglob('*.parquet')) / 1e6
        print(f'{k + 1}/{len(selected)} repositories: {ok} complete, {size:.0f} MB so far', flush=True)
(OUT / 'fetch_status.json').write_text(json.dumps(status, indent=1), encoding='utf-8')
bad = {r: s for r, s in status.items() if s != 'ok'}
print(f'done: {len(status) - len(bad)} complete, {len(bad)} not: ' + '; '.join(f'{r} [{s}]' for r, s in list(bad.items())[:12]))
