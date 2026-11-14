"""Make the card library's entries from the registry (benchmarks.json), the pilot table and the public files.

For each benchmark with a command, the CLI runs in its own process (large universes need several GB, so one
at a time) and writes library/cards/<slug>.json; the card must pass validate_card. Every benchmark, with or
without a card, gets library/entries/<slug>.json: the registry fields plus its pilot-table row, copied as is.

usage: python library/make_entries.py <pilot data folder> <pilot_decidability.tsv> [--only slug ...] [--no-cards]
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'src')
sys.path.insert(0, SRC)
from benchscope_audit.card import validate_card  # noqa: E402

PILOT_FIELDS = ('domain', 'sources_checked', 'negative_source', 'rule_stated_in_paper', 'rule_found_in_code_or_release',
                'decidability', 'verification', 'W_pilot', 'evaluation_protocol', 'notes')


def read_pilot(path):
    with open(path, encoding='utf-8') as f:
        head = f.readline().rstrip('\n').split('\t')
        return {r['benchmark']: r for r in (dict(zip(head, line.rstrip('\n').split('\t'))) for line in f if line.strip())}


def resolve(args, inputs, root):
    local = {i['id']: os.path.join(root, *i['local'].split('/')) for i in inputs}
    out = []
    for a in args:
        if a.startswith('{') and a.endswith('}'):
            a = local[a[1:-1]]
        out.append(a)
    return out


def make_card(b, root, cards_dir):
    for i in b['inputs']:
        p = os.path.join(root, *i['local'].split('/'))
        if not os.path.exists(p):
            raise FileNotFoundError(f"{b['slug']}: input {i['id']} not found at {p} (download: {i['source']})")
    out = os.path.join(cards_dir, b['slug'] + '.json')
    cmd = [sys.executable, '-E', '-P', '-X', 'utf8', '-c', 'import sys; sys.path.insert(0, sys.argv[1]); '
           'from benchscope_audit.cli import main; sys.exit(main(sys.argv[2:]))', SRC]
    cmd += resolve(b['args'], b['inputs'], root) + ['--out', out]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    card = json.load(open(out, encoding='utf-8'))
    problems = validate_card(card)
    if problems:
        raise ValueError(f"{b['slug']}: invalid card: {problems}")
    print(f"  {b['slug']}: card in {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pilot_data')
    ap.add_argument('pilot_tsv')
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--no-cards', action='store_true', help='rewrite entries only; keep the existing cards')
    a = ap.parse_args()
    reg = json.load(open(os.path.join(HERE, 'benchmarks.json'), encoding='utf-8'))['benchmarks']
    pilot = read_pilot(a.pilot_tsv)
    cards_dir, entries_dir = os.path.join(HERE, 'cards'), os.path.join(HERE, 'entries')
    os.makedirs(cards_dir, exist_ok=True)
    os.makedirs(entries_dir, exist_ok=True)
    for b in reg:
        if a.only and b['slug'] not in a.only:
            continue
        if b['pilot_row'] not in pilot:
            raise KeyError(f"{b['slug']}: no pilot row {b['pilot_row']!r}")
        has_card = b['args'] is not None
        if has_card and not a.no_cards:
            make_card(b, a.pilot_data, cards_dir)
        if has_card and not os.path.exists(os.path.join(cards_dir, b['slug'] + '.json')):
            raise FileNotFoundError(f"{b['slug']}: no card yet; run without --no-cards")
        entry = {k: v for k, v in b.items() if k not in ('args',)}
        entry['command'] = None
        if has_card:
            disp = {i['id']: i['display'] for i in b['inputs']}
            entry['command'] = ['benchscope-audit'] + [disp[x[1:-1]] if x.startswith('{') else x for x in b['args']] \
                + ['--out', 'card.json']
        entry['card'] = b['slug'] + '.json' if has_card else None
        entry['pilot'] = {k: pilot[b['pilot_row']][k] for k in PILOT_FIELDS}
        with open(os.path.join(entries_dir, b['slug'] + '.json'), 'w', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(entry, indent=1, ensure_ascii=False, sort_keys=True) + '\n')
    print('entries:', len(os.listdir(entries_dir)), '| cards:', len(os.listdir(cards_dir)))


if __name__ == '__main__':
    main()
