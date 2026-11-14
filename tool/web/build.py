"""Build the static browser page into web/dist/ (host the folder as is, e.g. on GitHub Pages).

dist/index.html                 the page (copied from web/index.html)
dist/worker.js                  runs Pyodide off the page's main thread (copied from web/worker.js)
dist/benchscope_audit.zip       the package source; Pyodide unpacks it, so the page runs the same code as the CLI
dist/wordnet30_wn18.tsv.gz      WordNet 3.0 synsets and WN18-relation triples, with WordNet's licence notice
dist/example_wn18rr_card.*      the WN18RR example card (tool/examples), shown by "Show an example card"

Both archives are written without timestamps, so a rebuild from the same sources gives the same bytes.

usage: python web/build.py <WordNet-3.0/dict>
"""
import hashlib
import os
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'src')
sys.path.insert(0, SRC)
from benchscope_audit import __version__  # noqa: E402
from benchscope_audit.universes import export_wordnet30_table  # noqa: E402


def zip_package(out_path):
    root = os.path.join(SRC, 'benchscope_audit')
    files = []
    for d, _, fs in os.walk(root):
        files += [os.path.join(d, f) for f in fs if f.endswith('.py')]
    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            info = zipfile.ZipInfo(os.path.relpath(p, SRC).replace('\\', '/'), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(p, 'rb') as f:
                z.writestr(info, f.read())
    return len(files)


def main(wordnet_dict):
    dist = os.path.join(HERE, 'dist')
    os.makedirs(dist, exist_ok=True)
    for f in ('index.html', 'worker.js'):
        shutil.copyfile(os.path.join(HERE, f), os.path.join(dist, f))
    # the page's "Show an example card": the command line's WN18RR card (tool/examples)
    for ext in ('json', 'md'):
        shutil.copyfile(os.path.join(HERE, '..', 'examples', f'wn18rr_card.{ext}'), os.path.join(dist, f'example_wn18rr_card.{ext}'))
    n = zip_package(os.path.join(dist, 'benchscope_audit.zip'))
    lines = export_wordnet30_table(wordnet_dict, os.path.join(dist, 'wordnet30_wn18.tsv.gz'))
    print(f'benchscope-audit {__version__}: {n} modules zipped; WordNet table {lines} lines')
    for f in sorted(os.listdir(dist)):
        p = os.path.join(dist, f)
        print(f'  {f}  {os.path.getsize(p):,} bytes  sha256 {hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]}')


if __name__ == '__main__':
    main(sys.argv[1])
