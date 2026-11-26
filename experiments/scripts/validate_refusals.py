"""V3 (OUTLINE.md v3, Section 3): does the audit refuse or warn where the files do not support a claim?

Each scenario is a small set of input files with one defect (or a control without it), the command line that
audits them, the behaviour the tool is specified to show, and a check of what it did. The files are written to
results/validation/v3_fixtures/<scenario>/, so the browser page can be run on the same files (the page calls the
same main(); its rows are added by the browser check and recorded separately).

Behaviours checked:
  refused   exit code 2, one line 'scopecard: input error: ...' naming the problem, no traceback, no card
  warned    a card is written, the problem is listed under warnings and repeated after the scope statement
  withheld  a card is written, but the comparison gives no break-even (c* is null) and says why
  coverage  the card reports coverage only (support.structural false)

Outputs: results/validation/v3_refusals.tsv and v3_refusals.md.
usage: python -E -P validate_refusals.py [OUT=results/validation]
"""
import contextlib
import io
import json
import os
import shutil
import sys

P4 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(P4, '..', 'tool', 'src'))
from scopecard.card import validate_card  # noqa: E402
from scopecard.cli import main as cli_main  # noqa: E402

UNI = ''.join(f'e{i}\tr\te{j}\n' for i in range(12) for j in (i + 1, i + 3) if j < 12)       # a small triple universe
REL = 'e0\tr\te1\ne1\tr\te2\ne2\tr\te3\ne3\tr\te4\n'                                         # a release inside it
QRELS = lambda judged, extra='': ''.join(f'{q} 0 d{i} {g}\n' for q in judged for i, g in enumerate((2, 2, 2, 0))) + extra  # noqa: E731
TOPICS = ''.join(f'{q}\tquery {q}\n' for q in range(1, 11))


def LGCN(train):                                    # LightGCN release (remapped ids) and its ratings universe
    return {'release/train.txt': train, 'release/test.txt': '0 0 1\n1 1\n',
            'release/user_list.txt': 'org_id remap_id\nu0 0\nu1 1\n', 'release/item_list.txt': 'org_id remap_id\ni0 0\ni1 1\n',
            'ratings.csv': 'u0,i0,5,1\nu1,i1,4,1\nu0,i1,3,1\nu2,i0,2,1\n'}


def kg(release_train, release_rest=REL):
    return {'universe.tsv': UNI, 'release/train.txt': release_train, 'release/valid.txt': release_rest,
            'release/test.txt': release_rest}


KG = ['kg-triples', '--release', '{d}/release', '--universe', '{d}/universe.tsv', '--format', 'tsv', '--name', 'toy KG']
QR = ['query-qrels', '--topics', '{d}/topics.tsv', '--qrels', '{d}/qrels.txt', '--corpus-size', '1000', '--name', 'toy']
PC = ['pair-classification', '--pairs', '{d}/pairs.csv', '--a-col', 'x', '--b-col', 'y', '--universe-units', '{d}/units.txt',
      '--name', 'toy pairs']

SCENARIOS = [
    # id, scenario, files, argv, expected behaviour, check(code, err, card) -> observed or None if it failed
    ('R1', 'Release does not match the universe (KG ids)',
     {'universe.tsv': UNI, 'release/train.txt': 'x0\tr\tx1\nx1\tr\tx2\n', 'release/valid.txt': 'x2\tr\tx3\n',
      'release/test.txt': 'x3\tr\tx4\n'}, KG, 'refused: fewer than 50% of release ids occur in the universe',
     lambda c, e, k: c == 2 and 'input error: Only 0 of the' in e and k is None),
    ('R2', 'Release does not match the universe (query level)',
     {'topics.tsv': TOPICS, 'qrels.txt': QRELS((21, 22, 23))}, QR, 'refused: no judged topic is a released topic',
     lambda c, e, k: c == 2 and 'Only 0 of the 3 judged topics' in e and k is None),
    ('R3', 'Release partly outside the universe',
     {'topics.tsv': TOPICS, 'qrels.txt': QRELS(tuple(range(1, 10)) + (99,))}, QR,
     'warned: between 50% and 95% of release ids occur in the universe',
     lambda c, e, k: c == 0 and any('Only 9 of the 10' in w for w in k['warnings']) and 'Caution' in k['scope_statement']),
    ('R4', 'Malformed release line', kg('e0\tr\te1\ne1 r e2\n'), KG, 'refused, naming the file and the line',
     lambda c, e, k: c == 2 and 'train.txt, line 2' in e and k is None),
    ('R5', 'One empty split in a KG release (train.txt)', kg(''), KG, 'refused, naming the empty file',
     lambda c, e, k: c == 2 and 'train.txt has no data lines' in e and k is None),
    ('R5b', 'One empty split in a LightGCN release (train.txt)', LGCN(''),
     ['recsys-ratings', '--release', '{d}/release', '--ratings', '{d}/ratings.csv', '--name', 'toy'],
     'refused, naming the empty file', lambda c, e, k: c == 2 and 'train.txt has no data lines' in e and k is None),
    ('R5c', 'All release files empty', kg('', ''), KG, 'refused, naming an empty file',
     lambda c, e, k: c == 2 and 'train.txt has no data lines' in e and k is None),
    ('R5d', 'Empty universe unit list', {'units.txt': '', 'pairs.csv': 'x,y\na,b\n'}, PC, 'refused, naming the empty file',
     lambda c, e, k: c == 2 and 'units.txt has no data lines' in e and k is None),
    ('R5e', 'Pair file with a header only', {'units.txt': 'a\nb\n', 'pairs.csv': 'x,y\n'}, PC,
     'refused, naming the empty file', lambda c, e, k: c == 2 and 'pairs.csv has no data lines' in e and k is None),
    ('R5f', 'Empty topic list', {'topics.tsv': '\n', 'qrels.txt': QRELS((1, 2))}, QR, 'refused, naming the empty file',
     lambda c, e, k: c == 2 and 'topics.tsv has no data lines' in e and k is None),
    ('R5g', 'LightGCN split with user ids but no items', LGCN('0\n1\n'),
     ['recsys-ratings', '--release', '{d}/release', '--ratings', '{d}/ratings.csv', '--name', 'toy'],
     'refused, naming the file', lambda c, e, k: c == 2 and 'train.txt lists no user-item pairs' in e and k is None),
    ('R6', 'Raw file with lines of the wrong format',
     {'topics.tsv': TOPICS, 'qrels.txt': QRELS((1, 2, 3), 'garbage line\n1 0 dX notanumber\n')}, QR,
     'warned: skipped lines counted per file, first one named',
     lambda c, e, k: c == 0 and k['inputs']['qrels.txt']['skipped'] == 2 and any('were skipped' in w for w in k['warnings'])),
    ('R7', 'Raw file with no line of the right format', {'topics.tsv': TOPICS, 'qrels.txt': 'a b\nc d\n'}, QR,
     'refused: no line of the file fits its format', lambda c, e, k: c == 2 and 'none of its 2 lines' in e and k is None),
    ('R8', 'Pair file with an unused label column',
     {'units.txt': 'a\nb\nc\n', 'pairs.csv': 'x,y,label\na,b,1\nb,c,0\n'}, PC, 'warned: a label column was not used',
     lambda c, e, k: c == 0 and any("column named 'label'" in w for w in k['warnings'])),
    ('R9', 'Sampled pairs with an unknown sampling rule',
     {'units.txt': 'a\nb\nc\nd\n', 'pairs.csv': 'x,y\na,b\nc,d\n'}, PC + ['--delta-O', '0.02', '--metric', 'auc'],
     'coverage only, and no break-even',
     lambda c, e, k: c == 0 and k['support']['structural'] is False and k['comparison']['c_star'] is None
     and not k['comparison']['applicable'] and 'coverage' in k['scope_statement']),
    ('R10', 'Declared candidate rule that the file contradicts',
     {'units.txt': 'a\nb\nc\nd\n', 'pairs.csv': 'x,y\na,b\nc,d\na,c\n'},
     PC + ['--candidate-rule', 'all-pairs', '--delta-O', '0.02', '--metric', 'auc'],
     'warned, and the break-even withheld: the file lists fewer pairs than the declaration implies',
     lambda c, e, k: c == 0 and k['support']['declaration']['consistent'] is False and any('lists only 3 of the 6 pairs' in w
                                                                                            for w in k['warnings'])
     and k['comparison']['c_star'] is None and 'not supported by the file' in k['comparison']['statement']),
    ('R11', 'Declared candidate rule that the file supports (control)',
     {'units.txt': 'a\nb\nc\nd\n', 'pairs.csv': 'x,y\na,b\na,c\na,d\nb,c\nb,d\nc,d\n'},
     PC + ['--candidate-rule', 'all-pairs'], 'no warning; the exclusion is structural',
     lambda c, e, k: c == 0 and k['support']['declaration']['consistent'] is True and k['warnings'] == []
     and k['support']['structural'] is True),
    ('R12', 'Break-even asked for a metric that does not decompose (MRR)', kg(REL), KG + ['--delta-O', '0.01', '--metric', 'mrr'],
     'withheld: MRR is not linear in the candidate mix',
     lambda c, e, k: c == 0 and k['comparison']['c_star'] is None and not k['comparison']['applicable']
     and 'not linear' in k['comparison']['statement']),
    ('R13', 'Model difference given without its metric', kg(REL), KG + ['--delta-O', '0.01'],
     'refused by the argument parser: --delta-O needs --metric',
     lambda c, e, k: c == 2 and '--delta-O needs --metric' in e and k is None),
    ('R14', 'No unobserved pair in the universe (zero candidates)',
     {'units.txt': 'a\nb\nc\n', 'pairs.csv': 'x,y\na,b\nb,c\n', 'observed.tsv': 'a\tb\na\tc\nb\tc\n'},
     PC + ['--universe-pairs', '{d}/observed.tsv'], 'W_pairs undefined with a note, no crash',
     lambda c, e, k: c == 0 and k['exclusion'].get('W_pairs') is None and 'undefined' in json.dumps(k['exclusion'])),
    ('R15', 'Nothing excluded (W = 0) with a model difference',
     {'units.txt': 'a\nb\nc\nd\n', 'pairs.csv': 'x,y\na,b\na,c\na,d\nb,c\nb,d\nc,d\n'},
     PC + ['--candidate-rule', 'all-pairs', '--delta-O', '0.02', '--metric', 'auc'],
     'no crash; the comparison says nothing excluded can reverse it',
     lambda c, e, k: c == 0 and k['comparison']['applicable'] and k['comparison']['c_star'] is None
     and 'nothing' in k['comparison']['statement']),
    # added 2026-10-25 to 2026-10-27
    ('R16', 'Two unit sets that share ids (bipartite, all pairs listed)',
     {'a.txt': '0\n1\n', 'b.txt': '0\n1\n', 'pairs.csv': 'x,y\n0,0\n0,1\n1,0\n1,1\n'},
     ['pair-classification', '--pairs', '{d}/pairs.csv', '--a-col', 'x', '--b-col', 'y', '--universe-units', '{d}/a.txt',
      '--universe-units-b', '{d}/b.txt', '--bipartite', '--candidate-rule', 'all-pairs', '--name', 'toy'],
     'no warning; coverage 100%; the declaration holds (equal ids on the two sides are different units)',
     lambda c, e, k: c == 0 and k['exclusion']['pair_coverage'] == 1 and k['support']['declaration']['consistent']
     and k['warnings'] == []),
    ('R17', "Recommendation release counted with the pilot table's release positives",
     {'release/user_list.txt': 'org_id remap_id\nu1 0\nu2 1\n', 'release/item_list.txt': 'org_id remap_id\ni1 0\ni2 1\n',
      'release/train.txt': '0 0\n', 'release/test.txt': '1 1\n', 'ratings.csv': 'u1,i1,5,1\nu1,i2,5,2\nu2,i2,5,3\nu3,i3,5,4\n'},
     ['recsys-ratings', '--release', '{d}/release', '--ratings', '{d}/ratings.csv', '--positives', 'release', '--name', 'toy'],
     'the headline is the share (80%); the pilot value (60%) is kept beside it, not reported as excluded',
     lambda c, e, k: c == 0 and k['exclusion']['W_pairs'] == 0.8 and '80.0% of the 5' in k['scope_statement']
     and k['exclusion']['W_pairs_release_convention'] == 0.6),
    ('R18', 'Corpus size smaller than the documents judged for one topic',
     {'topics.tsv': TOPICS, 'qrels.txt': QRELS((1, 2))}, [a if a != '1000' else '3' for a in QR],
     'refused, naming the corpus size (an unjudged share would be negative)',
     lambda c, e, k: c == 2 and 'smaller than the 4 distinct documents judged' in e and k is None),
    ('R19', 'Model difference that is not a finite difference in [-1, 1]', kg(REL), KG + ['--delta-O', 'nan', '--metric', 'auc'],
     'refused as an input error (also in the page, which cannot exit)',
     lambda c, e, k: c == 2 and 'input error: --delta-O must be a finite difference' in e and k is None),
    # added 2026-10-27
    ('R20', 'Model difference while some test triples fall outside the universe',
     kg('e0\tr\te1\ne1\tr\te2\n') | {'release/test.txt': 'e0\tr\te1\ne1\tr\te2\ne0\tr\tzz\n'},
     KG + ['--delta-O', '-0.02', '--metric', 'auc'],
     'warned: the break-even uses fewer queries than the difference and carries no guarantee',
     lambda c, e, k: c == 0 and k['comparison']['applicable']
     and any('1 of the 3 test triples (an entity outside the universe)' in w and 'no guarantee' in w
             for w in k['warnings'])),
    ('R21', 'A test query with no kept candidate (W_q = 1)',
     {'universe.tsv': 'a\tr\ta\na\tr\tb\nb\tr\tc\nc\tr\td\n', 'release/train.txt': 'a\tr\ta\n',
      'release/valid.txt': 'a\tr\tb\n', 'release/test.txt': 'a\tr\tb\n'},
     KG + ['--delta-O', '-0.02', '--metric', 'auc'],
     'withheld: the benchmark difference is undefined on that query',
     lambda c, e, k: c == 0 and k['exclusion']['per_query']['queries_without_kept'] == 1
     and k['comparison']['c_star'] is None and 'keep no candidate of the release' in k['comparison']['statement']),
]


def run(sid, files, argv, root):
    d = os.path.join(root, sid)
    shutil.rmtree(d, ignore_errors=True)
    for rel, text in files.items():
        p = os.path.join(d, *rel.split('/'))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
    out = os.path.join(d, 'card.json')
    args = [a.replace('{d}', d) for a in argv] + ['--out', out]
    err, std = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(std):
        try:
            code = cli_main(args)
        except SystemExit as e:                     # argparse errors exit through SystemExit
            code = e.code
    card = json.load(open(out, encoding='utf-8')) if os.path.exists(out) else None
    if card is not None:
        os.remove(out)                              # the fixtures folder keeps inputs only
    return code, err.getvalue(), card


def main(out):
    root = os.path.join(out, 'v3_fixtures')
    os.makedirs(root, exist_ok=True)
    rows = []
    for sid, name, files, argv, expected, check in SCENARIOS:
        code, err, card = run(sid, files, argv, root)
        ok = bool(check(code, err, card)) and 'Traceback' not in err
        if card is not None:
            ok = ok and validate_card(card) == []
        observed = (f'exit {code}; ' + (err.strip().splitlines()[-1][:160] if code else
                                         f"warnings {len(card['warnings'])}; structural {card['support']['structural']}; "
                                         f"c* {(card.get('comparison') or {}).get('c_star')}"))
        rows.append((sid, name, ' '.join(['scopecard'] + [a.replace('{d}/', '') for a in argv]), expected, observed,
                     'pass' if ok else 'FAIL'))
        print(f"{sid:<4} {'pass' if ok else 'FAIL'}  {name}", flush=True)
    with open(os.path.join(out, 'v3_refusals.tsv'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('id\tscenario\tcommand\texpected\tobserved\tresult\n')
        f.writelines('\t'.join(r) + '\n' for r in rows)
    md = ['# V3: refusal and warning suite (command line)', '',
          f'{sum(r[5] == "pass" for r in rows)} of {len(rows)} scenarios behave as specified. Every card written also '
          'passes validate_card; no scenario prints a traceback. Fixtures: v3_fixtures/<id>/.', '',
          '| id | scenario | expected | observed | result |', '|---|---|---|---|---|']
    md += [f'| {r[0]} | {r[1]} | {r[3]} | {r[4].replace("|", "/")} | {r[5]} |' for r in rows]
    with open(os.path.join(out, 'v3_refusals.md'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(md) + '\n')
    return 0 if all(r[5] == 'pass' for r in rows) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(P4, 'results', 'validation')))
