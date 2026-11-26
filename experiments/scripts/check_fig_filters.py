"""Check fig_filters_values.tsv, the list of every value drawn in fig_filters.pdf, against the records it names.

usage: python -E -P check_fig_filters.py [--results DIR] [--values FILE]
       (defaults: results/ and figures/fig_filters_values.tsv in the folder above this script)

The checks read the records themselves and share no code with make_fig_filters.py:
  1. sources: each of the four source files has the SHA-256 that the list gives, so the figure was drawn from them;
  2. values: every listed estimate, lower and upper end is 100 times the stored field that its reference
     (key#column) names, printed to ten significant digits, and every key occurs once in its file;
  3. label: the label printed with the pooled estimate of panel c is that estimate and its interval to two decimals;
  4. coverage: exactly the expected elements are listed (a: BigCodeBench Complete and Instruct at 4 to 96 builders,
     MMLU at 4 to 64; b: the outputs that voted and the Instruct outputs at 4 to 64; c: the pooled D(8), the seven
     lines at 8 builders and Qwen at 16, and the label).
Prints one line per check and a last line with the verdict; exits with 1 if any check fails.
"""
import argparse
import csv
import hashlib
import os
import sys

P4 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
KEYS = {   # source file: the columns that make a record's key
    'panel_filter/records_panel_size.tsv': ('split', 'subject', 'quantity'),
    'mmlu_reconstruction/records_panel_size.tsv': ('analysis', 'subject', 'quantity'),
    'separated_scoring/effects.tsv': ('direction', 'p', 'metric'),
    'panel_composition/records.tsv': ('analysis', 'group', 'quantity'),
}
LINES = ('Qwen', 'Llama', 'DeepSeek-Coder', 'GPT', 'Mistral', 'Code Llama', 'Qwen-Coder')
EXPECTED = sorted(
    [('a', 'BigCodeBench, Complete', str(p)) for p in (4, 8, 16, 32, 64, 96)]
    + [('a', 'BigCodeBench, Instruct', str(p)) for p in (4, 8, 16, 32, 64, 96)]
    + [('a', "MMLU, MMLU-Pro's rule", str(p)) for p in (4, 8, 16, 32, 64)]
    + [('b', 'Complete outputs (voted)', str(p)) for p in (4, 8, 16, 32, 64)]
    + [('b', 'Instruct outputs (did not vote)', str(p)) for p in (4, 8, 16, 32, 64)]
    + [('c', 'seven lines, pooled', 'pooled'), ('c', 'printed under the label of the pooled row', 'pooled'),
       ('c', 'Qwen, 16 builders (supplementary)', 'line, 16 builders')]
    + [('c', name, 'line, 8 builders') for name in LINES])


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(results):
    rows, dup = {}, []
    for rel, cols in KEYS.items():
        with open(os.path.join(results, *rel.split('/')), newline='', encoding='utf-8') as fh:
            for r in csv.DictReader(fh, delimiter='\t'):
                key = (rel, '|'.join(r[c] for c in cols))
                if key in rows:
                    dup.append(f'{rel}: {key[1]}')
                rows[key] = r
    return rows, dup


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results', default=os.path.join(P4, 'results'))
    ap.add_argument('--values', default=os.path.join(P4, 'figures', 'fig_filters_values.tsv'))
    a = ap.parse_args()
    with open(a.values, encoding='utf-8') as fh:
        text = fh.read().splitlines()
    listed_sha = {p[1]: p[2] for p in (x.split('\t') for x in text if x.startswith('# sha256\t'))}
    body = [x for x in text if not x.startswith('#')]
    head, entries = body[0].split('\t'), [dict(zip(body[0].split('\t'), x.split('\t'))) for x in body[1:]]
    fails = []

    # 1. sources
    bad = [rel for rel in KEYS if listed_sha.get(rel) != sha256(os.path.join(a.results, *rel.split('/')))]
    extra = sorted(set(listed_sha) - set(KEYS))
    fails += [f'sources: {rel} does not have the listed SHA-256' for rel in bad]
    fails += [f'sources: {rel} is listed but is not a source' for rel in extra]
    print(f'sources: {len(KEYS) - len(bad)} of {len(KEYS)} files have the listed SHA-256')

    # 2. values
    rows, dup = load(a.results)
    fails += [f'values: key occurs twice, {d}' for d in dup]
    stored = lambda rel, ref: rows[(rel, ref.split('#')[0])][ref.split('#')[1]]  # noqa: E731
    n = 0
    for e in entries:
        refs = [e['ref_estimate'], e['ref_lo'], e['ref_hi']]
        try:
            raw = [stored(e['source'], r) for r in refs]
        except KeyError as err:
            fails.append(f"values: {e['panel']}, {e['element']}, {e['x']}: no record for {err}")
            continue
        if e['element'] == 'printed under the label of the pooled row':
            continue                                                       # check 3
        for col, r in zip(('estimate', 'lo', 'hi'), raw):
            n += 1
            if e[col] != f'{100 * float(r):.10g}':
                fails.append(f"values: {e['panel']}, {e['element']}, {e['x']}, {col}: listed {e[col]}, "
                             f"record gives {100 * float(r):.10g}")
    print(f'values: {n} listed values checked against their records')

    # 3. label
    labels = [e for e in entries if e['element'] == 'printed under the label of the pooled row']
    for e in labels:
        v, lo, hi = (float(f'{100 * float(stored(e["source"], r)):.10g}')
                     for r in (e['ref_estimate'], e['ref_lo'], e['ref_hi']))
        want = f'{v:.2f} [{lo:.2f}, {hi:.2f}]'
        if e['estimate'] != want:
            fails.append(f"label: printed {e['estimate']!r}, records give {want!r}")
    print(f'label: {len(labels)} printed label(s): ' + ', '.join(e['estimate'] for e in labels))

    # 4. coverage
    got = sorted((e['panel'], e['element'], e['x']) for e in entries)
    if got != EXPECTED or tuple(head) != ('panel', 'element', 'x', 'estimate', 'lo', 'hi', 'interval', 'source',
                                          'ref_estimate', 'ref_lo', 'ref_hi'):
        missing, unexpected = sorted(set(EXPECTED) - set(got)), sorted(set(got) - set(EXPECTED))
        fails.append(f'coverage: missing {missing}, unexpected {unexpected}, '
                     f'{len(got) - len(set(got))} listed twice')
    print(f'coverage: {len(got)} elements listed, {len(EXPECTED)} expected')

    for f in fails:
        print('FAIL', f)
    print(f'{len(fails)} failures: every value drawn matches its record' if not fails else
          f'{len(fails)} failures')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
