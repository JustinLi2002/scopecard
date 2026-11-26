"""The card library (Layer 3): registry, cards, entries, schema and site build."""
import json
import os
import sys

import pytest

from conftest import PILOT
from scopecard.card import REQUIRED, TASKS, VERDICTS, validate_card
from scopecard.cli import main as cli_main

LIB = os.path.join(os.path.dirname(__file__), '..', 'library')
PILOT_TSV = os.path.join(PILOT, '..', 'pilot_decidability.tsv')
REG = json.load(open(os.path.join(LIB, 'benchmarks.json'), encoding='utf-8'))['benchmarks']
sys.path.insert(0, LIB)
import build_site  # noqa: E402
from build_site import SCHEMA_FILE  # noqa: E402
import make_entries  # noqa: E402


def test_registry_is_consistent():
    slugs = [b['slug'] for b in REG]
    assert len(slugs) == len(set(slugs)) == 10
    for b in REG:
        ids = {i['id'] for i in b['inputs']}
        if b['args'] is None:
            assert b['why_no_card'] and not b['inputs']
            continue
        assert b['why_no_card'] is None
        assert {a[1:-1] for a in b['args'] if a.startswith('{')} == ids
        assert all(i['source'].startswith(('https://', 'http://')) for i in b['inputs'])


@pytest.mark.skipif(not os.path.exists(PILOT_TSV), reason='pilot table not present')
def test_entries_copy_the_pilot_table():
    pilot = make_entries.read_pilot(PILOT_TSV)
    for b in REG:
        e = json.load(open(os.path.join(LIB, 'entries', b['slug'] + '.json'), encoding='utf-8'))
        assert e['pilot'] == {k: pilot[b['pilot_row']][k] for k in make_entries.PILOT_FIELDS}
        assert (e['card'] is None) == (b['args'] is None)


def test_library_cards_validate():
    for b in REG:
        if b['args']:
            card = json.load(open(os.path.join(LIB, 'cards', b['slug'] + '.json'), encoding='utf-8'))
            assert validate_card(card) == [], b['slug']
            assert card['benchmark']['name'] == b['name']


def test_validator_catches_inconsistent_cards():
    card = json.load(open(os.path.join(LIB, 'cards', 'wn18rr.json'), encoding='utf-8'))
    bad = json.loads(json.dumps(card))
    bad['exclusion']['W'] = 0.5
    assert any('W does not equal' in p for p in validate_card(bad))
    bad = json.loads(json.dumps(card))
    bad['rule_diagnosis']['families'][0]['verdict'] = 'reproduced'
    assert any('verdict does not match' in p for p in validate_card(bad))
    bad = {k: v for k, v in card.items() if k != 'hygiene'}
    assert validate_card(bad) == ['missing fields: hygiene']


def test_schema_file_matches_validator():
    s = json.load(open(os.path.join(LIB, 'schema', SCHEMA_FILE), encoding='utf-8'))
    assert tuple(s['required']) == REQUIRED
    assert tuple(s['properties']['task']['enum']) == TASKS
    assert tuple(s['$defs']['rule']['properties']['verdict']['enum']) == VERDICTS


def test_site_build_is_deterministic(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    build_site.main(str(a))
    build_site.main(str(b))
    files = sorted(os.path.relpath(os.path.join(d, f), a) for d, _, fs in os.walk(a) for f in fs)
    assert files == sorted(os.path.relpath(os.path.join(d, f), b) for d, _, fs in os.walk(b) for f in fs)
    assert all((a / f).read_bytes() == (b / f).read_bytes() for f in files)
    index = (a / 'index.html').read_text(encoding='utf-8')
    for x in REG:
        assert f'cards/{x["slug"]}.html' in index
        page = (a / 'cards' / f'{x["slug"]}.html').read_text(encoding='utf-8')
        if x['args']:
            card = json.load(open(os.path.join(LIB, 'cards', x['slug'] + '.json'), encoding='utf-8'))
            assert build_site.esc(card['scope_statement']) in page
        else:
            assert 'No card yet.' in page


@pytest.mark.parametrize('slug', ['wn18rr', 'trec-dl-2019-passage', 'trec-dl-2020-passage'])
def test_fast_library_cards_are_current(tmp_path, slug):
    """The quick cards equal a fresh run of their registry command (the large ones are covered by the golden tests)."""
    b = next(x for x in REG if x['slug'] == slug)
    paths = [os.path.join(PILOT, *i['local'].split('/')) for i in b['inputs']]
    if not all(os.path.exists(p) for p in paths):
        pytest.skip('inputs not present')
    out = tmp_path / 'card.json'
    cli_main(make_entries.resolve(b['args'], b['inputs'], PILOT) + ['--out', str(out)])
    assert out.read_text(encoding='utf-8') == open(os.path.join(LIB, 'cards', slug + '.json'), encoding='utf-8').read()
