"""Synthetic checks with known answers."""
import itertools
import json
import os
import pickle
import zipfile

import numpy as np
import pytest

from benchscope_audit import (Universe, c_star, derived_scope, map_triples, pair_exclusion, rule_units,
                              search_rules, unit_exclusion)
from benchscope_audit.card import jac, pct, to_json, to_markdown, validate_card
from benchscope_audit.cli import main as cli_main
from benchscope_audit.io import InputError, read_kg_release, read_lightgcn_release, read_torch_numpy_dict
from benchscope_audit.mapping import match_triples_exact
from benchscope_audit.pipelines import audit_pair_classification, audit_query_qrels
from benchscope_audit.rules import search_depth_threshold
from benchscope_audit.scope import query_shares, user_shares
from benchscope_audit.universes import from_pair_stream, from_pairs, from_triples, from_unit_pairs


def random_bipartite(seed=0, users=300, items=200, p=0.03):
    rng = np.random.default_rng(seed)
    # heterogeneous activity so that cores are non-trivial
    pu = rng.pareto(1.5, users) + 0.2
    pi = rng.pareto(1.5, items) + 0.2
    prob = np.clip(np.outer(pu, pi) / np.outer(pu, pi).mean() * p, 0, 1)
    m = rng.random((users, items)) < prob
    return from_pairs([(f'u{a}', f'i{b}') for a, b in zip(*np.nonzero(m))], name='toy')


@pytest.mark.parametrize('family,T', [('kcore', 3), ('kcore', 5), ('threshold', 4), ('both_ends', 3), ('seeds_neighbours', 6)])
def test_rule_search_recovers_generating_rule(family, T):
    uni = random_bipartite()
    release = rule_units(uni, family, T)
    assert release.sum() > 0
    res = search_rules(uni, release, thresholds=range(1, 12))
    hit = [r for r in res['families'] if r['family'] == family][0]
    assert hit['jaccard'] == 1.0 and hit['verdict'] == 'reproduced'
    # the generating threshold itself must give Jaccard 1 (another T may tie, e.g. when cores coincide)
    assert np.array_equal(rule_units(uni, family, hit['T']), release)


def test_unit_exclusion_counts():
    uni = Universe('toy', units=np.array(list('abcdef'), dtype=object), edges=np.array([[0, 1], [1, 2], [3, 4]]),
                   strata={'kind': np.array(list('xxxyyy'), dtype=object)})
    mask = np.array([1, 1, 0, 0, 0, 0], bool)
    out = unit_exclusion(uni, mask)
    assert out['W'] == pytest.approx(4 / 6)
    assert out['by_strata']['kind']['x']['W'] == pytest.approx(1 / 3)
    out2 = unit_exclusion(uni, mask, among=uni.depth() > 0)        # 'f' has no relation
    assert out2['units'] == 5 and out2['W'] == pytest.approx(3 / 5)


def test_pair_exclusion_matches_enumeration():
    uni = random_bipartite(seed=1, users=40, items=30, p=0.1)
    mask = rule_units(uni, 'kcore', 3)
    got = pair_exclusion(uni, mask)['W_pairs']
    users = [i for i in range(uni.n) if uni.part[i] == 0]
    items = [i for i in range(uni.n) if uni.part[i] == 1]
    obs = {tuple(e) for e in uni.edges.tolist()}
    cand_all = [(a, b) for a, b in itertools.product(users, items) if (a, b) not in obs]
    cand_rel = [(a, b) for a, b in cand_all if mask[a] and mask[b]]
    assert got == pytest.approx(1 - len(cand_rel) / len(cand_all))


def test_derived_scope_dmin():
    uni = random_bipartite(seed=2)
    mask = rule_units(uni, 'kcore', 4)
    depth = uni.depth()
    sc = derived_scope(uni, mask, depth, thresholds=range(0, 30))
    ws = {r['d']: r['W'] for r in sc['curve']}
    assert sc['d_min'] is not None and ws[sc['d_min']] <= 0.5
    assert all(ws[d] > 0.5 for d in ws if d < sc['d_min'])


def test_c_star_gives_both_thresholds():
    """Review counterexample: delta_O 0.1, W 0.8. The ordering reverses once delta_U < -0.025 (not -0.125):
    c* = 0.125 bounds the between-region change c = delta_O - delta_U, not the excluded-region difference."""
    r = c_star(0.1, 0.8, 'auc')
    assert r['c_star'] == pytest.approx(0.125) and r['delta_U_break_even'] == pytest.approx(0.025)
    target = lambda dU: 0.2 * 0.1 + 0.8 * dU
    assert target(-0.04) < 0 < target(-0.02)                       # either side of -0.025
    assert (0.1 - -0.04) > r['c_star'] > (0.1 - -0.02)             # the same two cases measured by c
    assert '0.025' in r['statement'] and '0.125' in r['statement'] and r['applicable']
    assert c_star(-0.0142, 0.6337, 'auc')['c_star'] == pytest.approx(0.0142 / 0.6337)


def test_c_star_refuses_what_it_cannot_support():
    w0 = c_star(0.02, 0.0, 'auc')                                  # nothing excluded: no error, no threshold
    assert w0['applicable'] and w0['c_star'] is None and 'nothing' in w0['statement']
    for metric in ('mrr', 'hits'):                                 # not linear in the candidate mix
        r = c_star(0.02, 0.6, metric)
        assert not r['applicable'] and r['c_star'] is None and 'not linear' in r['statement']
    assert not c_star(0.0, 0.6, 'auc')['applicable'] and not c_star(0.02, None, 'auc')['applicable']
    with pytest.raises(ValueError):
        c_star(0.02, 0.6, 'ndcg')
    with pytest.raises(SystemExit):                                # the CLI requires the metric with a difference
        cli_main(['kg-wordnet', '--release', '.', '--wordnet', '.', '--delta-O', '0.01', '--out', 'x.json'])


def test_pair_coverage_is_not_exclusion(tmp_path):
    """Review counterexample: universe a, b, c, d and a release of ab and cd. Every unit is used, yet 4 of the
    6 pairs never appear; the release cannot tell whether they were impossible or just not drawn."""
    (tmp_path / 'u.txt').write_text('a\nb\nc\nd\n')
    (tmp_path / 'r.csv').write_text('x,y\na,b\nc,d\n')
    run = lambda **kw: audit_pair_classification([str(tmp_path / 'r.csv')], 'x', 'y', 'toy',
                                                 universe_units=str(tmp_path / 'u.txt'), **kw)
    card = run()
    ex = card['exclusion']
    assert card['support']['structural'] is False and ex['W'] == 0 and ex['W_pairs'] == 0
    assert (ex['release_pairs_in_universe'], ex['universe_pairs']) == (2, 6) and ex['pair_coverage'] == pytest.approx(1 / 3)
    assert 'coverage' in card['scope_statement'] and 'leave out' not in card['scope_statement']
    assert 'Coverage' in to_markdown(card)
    cmp_ = run(delta_O=0.02, metric='auc')['comparison']           # no break-even without an established exclusion
    assert not cmp_['applicable'] and cmp_['c_star'] is None
    # 2026-10-21 user test: declaring "every pair is evaluated" while the file lists 2 of the 6 pairs among its units
    # must not yield a confident "nothing excluded" verdict
    declared = run(candidate_rule='all_pairs', delta_O=0.02, metric='auc')
    assert declared['support']['declaration'] == {'pairs_listed': 2, 'pairs_expected': 6, 'consistent': False}
    assert any('lists only 2 of the 6 pairs' in w for w in declared['warnings']) and 'Caution:' in declared['scope_statement']
    assert not declared['comparison']['applicable'] and 'not supported by the file' in declared['comparison']['statement']
    (tmp_path / 'full.csv').write_text('x,y\na,b\na,c\na,d\nb,c\nb,d\nc,d\n')    # the full list supports the declaration
    full = audit_pair_classification([str(tmp_path / 'full.csv')], 'x', 'y', 'toy', universe_units=str(tmp_path / 'u.txt'),
                                     candidate_rule='all_pairs', delta_O=0.02, metric='auc')
    assert full['support']['declaration']['consistent'] and full['warnings'] == []
    assert full['support']['structural'] and 'leave out 0.0%' in full['scope_statement']
    assert full['comparison']['applicable'] and full['comparison']['c_star'] is None              # W = 0: no error
    assert validate_card(card) == [] and validate_card(declared) == [] and validate_card(full) == []


def test_zero_candidates_and_release_convention():
    full = from_pairs([('u1', 'i1'), ('u1', 'i2'), ('u2', 'i1'), ('u2', 'i2')], name='complete 2x2')
    r = pair_exclusion(full, np.ones(full.n, bool))                # no unobserved pair: undefined, no ZeroDivisionError
    assert r['candidate_pairs'] == 0 and r['W_pairs'] != r['W_pairs'] and 'undefined' in r['note']
    # Review counterexample: 3x3 universe, 5 observed pairs, release keeps 3 of them and all 6 units
    u3 = from_pairs([('u1', 'i1'), ('u2', 'i2'), ('u3', 'i3'), ('u1', 'i2'), ('u2', 'i3')], name='3x3')
    r = pair_exclusion(u3, np.ones(u3.n, bool), release_pairs=3)
    assert r['W_pairs'] == 0.0 and r['W_pairs_release_convention'] != r['W_pairs_release_convention']  # -0.5 is not reported
    assert 'not a share' in r['release_convention_note']


def test_universe_fingerprint_tracks_content(tmp_path):
    """Review counterexample: the same file name with different units must give different cards."""
    (tmp_path / 'r.csv').write_text('x,y\nP1,P2\n')
    cards = []
    for units in ('P1\nP2\nP3\n', 'P1\nP2\nZ9\n', 'P3\nP1\nP2\n'):
        (tmp_path / 'units.txt').write_text(units)
        cards.append(audit_pair_classification([str(tmp_path / 'r.csv')], 'x', 'y', 'x', universe_units=str(tmp_path / 'units.txt')))
    fp = [c['universe']['fingerprint'] for c in cards]
    assert fp[0] != fp[1] and to_json(cards[0]) != to_json(cards[1])
    assert fp[0] == fp[2]                                          # the same units in another order: same universe


def test_map_triples_reports_merged_ids():
    # 'n1' and 'v1' share the bare id '1'; the release uses '1' for both through different relations
    units = np.array(['n1', 'v1', 'n2', 'v2'], dtype=object)
    uni = Universe('toy', units=units, edges=np.array([[0, 2], [1, 3]]), edge_types=np.array(['r', 's'], dtype=object))
    rel, rep = map_triples({'train': [('1', 'r', '2'), ('1', 's', '2'), ('9', 'r', '2')]}, uni, key=lambda x: x[1:])
    assert rep.n_matched_triples == 2 and len(rep.unmatched_triples) == 1
    assert rep.merged_ids == {'1': ['n1', 'v1'], '2': ['n2', 'v2']}
    assert rel.unit_mask.all()


def test_match_triples_exact_agrees_with_set_lookup():
    rng = np.random.default_rng(3)
    trip = {(f'e{a}', f'r{r}', f'e{b}') for a, r, b in zip(rng.integers(0, 50, 400), rng.integers(0, 4, 400), rng.integers(0, 50, 400))}
    uni = from_triples(trip, name='toy')
    pool = sorted(trip)
    release = {'train': pool[:100] + [('e1', 'r9', 'e2'), ('zz', 'r0', 'e1')],
               'test': [('e3', 'r1', 'e4'), pool[-1]]}
    rep, by_split = match_triples_exact(release, uni)
    for s, ts in release.items():
        assert by_split[s] == sum(t in trip for t in ts)
    assert rep.n_release_triples == 104 and rep.n_matched_triples == sum(by_split.values())
    assert set(rep.unmatched_triples) == {t for ts in release.values() for t in ts if t not in trip}


def test_unmapped_release_units_count_against_rules():
    uni = random_bipartite(seed=4)
    release = rule_units(uni, 'kcore', 3)
    res = search_rules(uni, release, families=('kcore',), thresholds=[3], stated={'family': 'kcore', 'T': 3},
                       unmapped_release_units=10)
    k = int(release.sum())
    assert res['stated']['jaccard'] == pytest.approx(k / (k + 10))
    assert res['families'][0]['release'] == k + 10 and res['stated']['verdict'] != 'reproduced'


def _torch_like_files(tmp_path, obj):
    """The two torch.save layouts, written without torch: zip (archive/data.pkl) and legacy (pickled header)."""
    zp = tmp_path / 'zipfmt.pt'
    with zipfile.ZipFile(zp, 'w') as z:
        z.writestr('archive/data.pkl', pickle.dumps(obj, protocol=2))
        z.writestr('archive/version', '3\n')
    lp = tmp_path / 'legacy.pt'
    with open(lp, 'wb') as f:
        for x in (0x1950a86a20f9469cfc6c, 1001, {'protocol_version': 1001}, obj, []):
            pickle.dump(x, f, protocol=2)
    return zp, lp


def test_read_torch_numpy_dict_both_layouts(tmp_path):
    obj = {'source_node': np.arange(5, dtype=np.int64), 'target_node_neg': np.arange(12, dtype=np.int64).reshape(4, 3)}
    for p in _torch_like_files(tmp_path, obj):
        got = read_torch_numpy_dict(str(p))
        assert set(got) == set(obj) and all(np.array_equal(got[k], obj[k]) for k in obj)


def test_read_torch_numpy_dict_refuses_other_globals(tmp_path):
    class Evil:
        def __reduce__(self):
            return (os.getcwd, ())
    for p in _torch_like_files(tmp_path, {'x': Evil()}):
        with pytest.raises(pickle.UnpicklingError, match='refusing'):
            read_torch_numpy_dict(str(p))


def test_pair_stream_equals_from_pairs():
    rng = np.random.default_rng(5)
    pairs = [(f'u{a}', f'i{b}') for a, b in zip(rng.integers(0, 40, 600), rng.integers(0, 30, 600))]   # with duplicates
    a, b = from_pairs(pairs, name='x'), from_pair_stream(iter(pairs), name='x')
    as_set = lambda u: {(u.units[h], u.units[t]) for h, t in u.edges.tolist()}
    assert set(a.units.tolist()) == set(b.units.tolist()) and as_set(a) == as_set(b) and len(b.edges) == len(set(pairs))
    assert {u: p for u, p in zip(a.units.tolist(), a.part.tolist())} == {u: p for u, p in zip(b.units.tolist(), b.part.tolist())}
    mask_a, mask_b = rule_units(a, 'kcore', 3), rule_units(b, 'kcore', 3)
    assert set(a.units[mask_a].tolist()) == set(b.units[mask_b].tolist())


def _random_pairs(rng, n, m, prefix='p'):
    return [(f'{prefix}{a}', f'{prefix}{b}') for a, b in zip(rng.integers(0, n, m), rng.integers(0, n, m))]


def test_unipartite_pair_exclusion_matches_enumeration():
    rng = np.random.default_rng(6)
    units = [f'p{i}' for i in range(40)]
    uni, info = from_unit_pairs(units, None, _random_pairs(rng, 40, 120), symmetric=True)
    mask = rng.random(uni.n) < 0.6
    obs = {tuple(sorted(e)) for e in uni.edges.tolist()}
    cand = [(a, b) for a, b in itertools.combinations(range(uni.n), 2) if (a, b) not in obs]
    want = 1 - sum(mask[a] and mask[b] for a, b in cand) / len(cand)
    assert pair_exclusion(uni, mask)['W_pairs'] == pytest.approx(want)
    assert info['observed_pairs_kept'] == len(obs) and info['self_pairs_dropped'] >= 0
    # derived scope at d = 1 restricts to units with an observed partner
    d = uni.depth()
    keep = d >= 1
    cand1 = [(a, b) for a, b in cand if keep[a] and keep[b]]
    want1 = 1 - sum(mask[a] and mask[b] for a, b in cand1) / len(cand1)
    sc = derived_scope(uni, mask, d, thresholds=[1], pairs=True)
    assert sc['curve'][0]['W'] == pytest.approx(want1)


def test_pair_classification_pipeline_symmetric(tmp_path):
    (tmp_path / 'units.txt').write_text('\n'.join(f'P{i}' for i in range(10)) + '\n')
    (tmp_path / 'observed.tsv').write_text('P0\tP1\nP1\tP2\nP2\tP3\nP7\tP8\nP9\tX\n')     # P9-X leaves the universe
    rows = ['a,b,y', 'P0,P1,1', 'P1,P0,1', 'P2,P3,0', 'P2,P4,0', 'P4,P5,1', 'P4,P5,0', 'P5,P5,0', 'P0,Q1,0']
    (tmp_path / 'release.csv').write_text('\n'.join(rows) + '\n')
    card = audit_pair_classification([str(tmp_path / 'release.csv')], 'a', 'b', 'toy', label_col='y',
                                      universe_units=str(tmp_path / 'units.txt'), universe_pairs=str(tmp_path / 'observed.tsv'))
    ex, m, h = card['exclusion'], card['mapping'], card['hygiene']['pairs']
    assert (ex['included'], ex['units']) == (6, 10) and m['unmapped_ids'] == 1          # P0..P5 in; Q1 not in the universe
    assert card['universe']['observed_pairs_outside_universe'] == 1 and ex['observed_pairs'] == 4
    n, nr = 10, 6
    assert ex['W_pairs'] == pytest.approx(1 - (nr * (nr - 1) // 2 - 3) / (n * (n - 1) // 2 - 4))   # P0-P1, P1-P2, P2-P3 inside
    assert (h['rows'], h['distinct_pairs'], h['duplicate_rows'], h['self_pairs']) == (8, 6, 2, 1)
    assert h['pairs_with_conflicting_labels'] == 1                                      # P4-P5 labelled 1 and 0
    assert h['negatives_observed_as_interactions'] == 1                                 # P2-P3 is an observed interaction
    assert validate_card(card) == []


def test_pair_classification_pipeline_bipartite(tmp_path):
    (tmp_path / 'drugs.txt').write_text('\n'.join(f'D{i}' for i in range(6)) + '\n')
    (tmp_path / 'targets.txt').write_text('\n'.join(f'T{i}' for i in range(5)) + '\n')
    (tmp_path / 'release.tsv').write_text('D0\tT0\t1\nD1\tT1\t0\nD1\tT0\t0\nD2\tT2\t1\n')
    card = audit_pair_classification([str(tmp_path / 'release.tsv')], 0, 1, 'dti', label_col=2, header=False,
                                      universe_units=str(tmp_path / 'drugs.txt'), universe_units_b=str(tmp_path / 'targets.txt'),
                                      symmetric=False)
    ex = card['exclusion']
    assert (ex['part0_retained'], ex['part1_retained'], ex['part0_units'], ex['part1_units']) == (3, 3, 6, 5)
    assert ex['W_pairs'] == pytest.approx(1 - 9 / 30) and card['rule_diagnosis'] is None    # no observed pairs: no depth
    assert card['hygiene']['pairs']['by_label'] == {'0': 2, '1': 2}


def test_pct_never_rounds_to_the_bounds():
    assert pct(0.999742) == '99.97%' and pct(0.633642) == '63.4%' and pct(0.00004) == '0.004%'
    assert pct(1.0) == '100.0%' and pct(0.0) == '0.0%'
    assert jac(70839 / 70846) == '0.9999' and jac(1.0) == '1.000' and jac(0.96727) == '0.967'


def test_depth_threshold_search():
    depth = np.array([0, 1, 5, 6, 6, 9, 30])
    mask = depth >= 6
    res = search_depth_threshold(depth, mask, thresholds=range(1, 11))
    assert (res['best']['T'], res['best']['jaccard'], res['best']['verdict']) == (6, 1.0, 'reproduced')


def test_query_qrels_pipeline(tmp_path):
    (tmp_path / 'topics.tsv').write_text(''.join(f'{q}\tquery {q}\n' for q in range(1, 11)))
    rows = []
    for q, grades in {1: [3, 2, 2, 0, 0, 0], 2: [2, 2, 2, 2, 1], 3: [2, 0, 0]}.items():
        rows += [f'{q} 0 d{i} {g}\n' for i, g in enumerate(grades)]
    (tmp_path / 'qrels.txt').write_text(''.join(rows))
    card = audit_query_qrels(str(tmp_path / 'topics.tsv'), str(tmp_path / 'qrels.txt'), 1000, 'toy',
                             relevant_grade=2, min_relevant=3, max_ratio=0.6)
    ex, rc = card['exclusion'], card['rule_check']
    assert (ex['included'], ex['units'], ex['W']) == (3, 10, 0.7)
    assert ex['within_query']['unjudged_share']['min'] == pytest.approx(1 - 6 / 1000)
    # topic 2: 4 of 5 relevant (ratio 0.8); topic 3: 1 relevant
    assert rc['violations'] == 2 and rc['below_min_relevant'] == 1 and rc['at_or_above_max_ratio'] == 1
    assert rc['verdict'].startswith('violated by 2')


# ---- input checks and card wording (fixes after the 2026-10-21 user test)

def _qrels_files(tmp_path, topics=10, judged=(1, 2, 3), extra=''):
    (tmp_path / 'topics.tsv').write_text(''.join(f'{q}\tquery {q}\n' for q in range(1, topics + 1)))
    rows = [f'{q} 0 d{i} {g}\n' for q in judged for i, g in enumerate((2, 2, 2, 0))]
    (tmp_path / 'qrels.txt').write_text(''.join(rows) + extra)
    return str(tmp_path / 'topics.tsv'), str(tmp_path / 'qrels.txt')


def test_scope_statement_names_what_w_counts(tmp_path):
    """The statement names the counted population with its size, never just the universe's name."""
    t, q = _qrels_files(tmp_path)
    card = audit_query_qrels(t, q, 1000, 'toy')
    assert 'leave out 70.0% of the 10 toy test topics' in card['scope_statement']
    assert card['counted'] == 'the 10 toy test topics' and card['warnings'] == [] and validate_card(card) == []
    md = to_markdown(card)
    assert '**W is a share of:** the 10 toy test topics' in md and '**Definitions:**' in md


def test_release_that_does_not_match_the_universe_is_refused(tmp_path):
    """User-test counterexample: an unrelated release (no id in the universe) must not get a card saying it
    leaves out 100% of the universe; a release partly outside it gets a warning."""
    t, q = _qrels_files(tmp_path, topics=10, judged=(21, 22, 23))
    with pytest.raises(InputError, match=r'Only 0 of the 3 judged topics \(0\.0%\)'):
        audit_query_qrels(t, q, 1000, 'toy')
    t, q = _qrels_files(tmp_path, topics=10, judged=tuple(range(1, 10)) + (99,))          # 9 of 10 in the list
    card = audit_query_qrels(t, q, 1000, 'toy')
    assert len(card['warnings']) == 1 and 'Only 9 of the 10 judged topics (90.0%)' in card['warnings'][0]
    assert 'Caution: Only 9 of the 10' in card['scope_statement'] and '## Warnings' in to_markdown(card)


def test_skipped_input_lines_are_counted_and_reported(tmp_path):
    """User-test counterexample: lines of a raw file that do not fit were dropped silently."""
    t, q = _qrels_files(tmp_path, extra='garbage line\n1 0 dX notanumber\n')
    card = audit_query_qrels(t, q, 1000, 'toy')
    e = card['inputs']['qrels.txt']
    assert (e['lines'], e['skipped'], e['first_skipped']['line']) == (14, 2, 13)
    assert card['warnings'] == ['2 of the 14 lines of qrels.txt did not have the expected format (4 whitespace-separated '
                                'fields: topic, iteration, document, integer grade) and were skipped (first at line 13).']
    assert validate_card(card) == []
    bad = dict(card, warnings=[])
    assert 'inputs record skipped lines, but the card has no warning' in validate_card(bad)
    (tmp_path / 'qrels.txt').write_text('a b\nc d\n')
    with pytest.raises(InputError, match='none of its 2 lines has the expected format'):
        audit_query_qrels(t, q, 1000, 'toy')


def test_malformed_release_lines_name_the_file_and_line(tmp_path):
    """User-test counterexample: a space-separated KG release gave a raw Python unpacking error."""
    d = tmp_path / 'rel'
    d.mkdir()
    for s in ('train', 'valid', 'test'):
        (d / f'{s}.txt').write_text('a\tr\tb\nc r d\n')
    with pytest.raises(InputError, match=r"train\.txt, line 2: expected 3 tab-separated fields .* found 1"):
        read_kg_release(str(d))
    lg = tmp_path / 'lg'
    lg.mkdir()
    (lg / 'user_list.txt').write_text('org_id remap_id\nu0 0\n')
    (lg / 'item_list.txt').write_text('org_id remap_id\ni0 0\n')
    (lg / 'train.txt').write_text('0 0\n')
    (lg / 'test.txt').write_text('0 7\n')
    with pytest.raises(InputError, match=r'test\.txt, line 1: item id 7 is not in item_list\.txt'):
        read_lightgcn_release(str(lg))


def test_cli_reports_input_errors_without_a_traceback(tmp_path, capsys):
    t, q = _qrels_files(tmp_path, judged=(21, 22, 23))
    rc = cli_main(['query-qrels', '--topics', t, '--qrels', q, '--corpus-size', '100', '--out', str(tmp_path / 'c.json')])
    err = capsys.readouterr().err
    assert rc == 2 and 'benchscope-audit: input error: Only 0 of the 3 judged topics' in err and 'Traceback' not in err
    assert not (tmp_path / 'c.json').exists()


def test_pair_file_with_an_unused_label_column_is_flagged(tmp_path):
    """User-test counterexample: a file with a 'label' column, read without one, became negatives-only silently.
    Also: one conflicting pair reads '1 pair', and the default universe name is not a file name."""
    (tmp_path / 'units.txt').write_text('a\nb\nc\n')
    (tmp_path / 'r.csv').write_text('x,y,label\na,b,1\nb,a,0\n')
    card = audit_pair_classification([str(tmp_path / 'r.csv')], 'x', 'y', 'toy', universe_units=str(tmp_path / 'units.txt'))
    assert any("has a column named 'label', but no label column was given" in w for w in card['warnings'])
    assert card['universe']['name'] == 'the declared universe' and 'the 3 declared units (units.txt)' in card['scope_statement']
    labelled = audit_pair_classification([str(tmp_path / 'r.csv')], 'x', 'y', 'toy', label_col='label',
                                         universe_units=str(tmp_path / 'units.txt'))
    assert labelled['warnings'] == [] and '1 pair with conflicting labels' in to_markdown(labelled)


def test_empty_release_files_and_unit_lists_are_refused(tmp_path):
    """V3 gap (2026-10-24): a release with one empty split was audited from the other splits without a warning."""
    d = tmp_path / 'kg'
    d.mkdir()
    for s, text in (('train', ''), ('valid', 'a\tr\tb\n'), ('test', 'a\tr\tb\n')):
        (d / f'{s}.txt').write_text(text)
    with pytest.raises(InputError, match=r'train\.txt has no data lines'):
        read_kg_release(str(d))
    lg = tmp_path / 'lg'
    lg.mkdir()
    (lg / 'user_list.txt').write_text('org_id remap_id\nu0 0\n')
    (lg / 'item_list.txt').write_text('org_id remap_id\ni0 0\n')
    (lg / 'test.txt').write_text('0 0\n')
    for train, msg in (('', r'train\.txt has no data lines'), ('0\n', r'train\.txt lists no user-item pairs')):
        (lg / 'train.txt').write_text(train)
        with pytest.raises(InputError, match=msg):
            read_lightgcn_release(str(lg))
    (tmp_path / 'units.txt').write_text('\n')
    (tmp_path / 'r.csv').write_text('x,y\na,b\n')
    with pytest.raises(InputError, match=r'units\.txt has no data lines'):
        audit_pair_classification([str(tmp_path / 'r.csv')], 'x', 'y', 'toy', universe_units=str(tmp_path / 'units.txt'))


def test_bipartite_sides_may_share_ids(tmp_path):
    """Review counterexample (2026-10-24): with ids 0 and 1 on both sides, (0, 0) and (1, 1) are drug-target pairs,
    not self pairs; dropping them halved the coverage and contradicted a true all-pairs declaration."""
    cards = []
    for b_ids in ('0\n1\n', 'x\ny\n'):                            # the same release with the B side renamed
        (tmp_path / 'a.txt').write_text('0\n1\n')
        (tmp_path / 'b.txt').write_text(b_ids)
        b0, b1 = b_ids.split()
        (tmp_path / 'r.csv').write_text(f'a,b\n0,{b0}\n0,{b1}\n1,{b0}\n1,{b1}\n')
        cards.append(audit_pair_classification([str(tmp_path / 'r.csv')], 'a', 'b', 'toy', universe_units=str(tmp_path / 'a.txt'),
                                                universe_units_b=str(tmp_path / 'b.txt'), symmetric=False,
                                                candidate_rule='all_pairs', delta_O=0.1, metric='auc'))
    for card in cards:
        assert card['exclusion']['pair_coverage'] == 1 and card['support']['declaration']['consistent']
        assert card['hygiene']['pairs']['self_pairs'] == 0 and card['comparison']['applicable']
        assert validate_card(card) == []


def test_release_convention_never_heads_the_card(tmp_path):
    """Review counterexample (2026-10-24): 3 x 3 users and items with 4 observed pairs; the release keeps a 2 x 2
    block holding 3 of them but lists 2. Of the 5 unobserved pairs, 1 is kept: 80% are excluded. The pilot
    table's release-positives value (60%) is kept only beside the share."""
    rel = tmp_path / 'rel'
    rel.mkdir()
    (rel / 'user_list.txt').write_text('org_id remap_id\nu1 0\nu2 1\n')
    (rel / 'item_list.txt').write_text('org_id remap_id\ni1 0\ni2 1\n')
    (rel / 'train.txt').write_text('0 0\n')
    (rel / 'test.txt').write_text('1 1\n')
    (tmp_path / 'ratings.csv').write_text('u1,i1,5,1\nu1,i2,5,2\nu2,i2,5,3\nu3,i3,5,4\n')
    for positives in ('universe', 'release'):
        out = tmp_path / f'{positives}.json'
        assert cli_main(['recsys-ratings', '--release', str(rel), '--ratings', str(tmp_path / 'ratings.csv'),
                         '--positives', positives, '--name', 'toy', '--out', str(out)]) == 0
        card = json.loads(out.read_text(encoding='utf-8'))
        ex = card['exclusion']
        assert ex['W_pairs'] == pytest.approx(0.8) and '80.0% of the 5 unobserved' in card['scope_statement']
        assert ex.get('W_pairs_release_convention') == (pytest.approx(0.6) if positives == 'release' else None)
        assert validate_card(card) == []


def test_impossible_inputs_are_refused(tmp_path, capsys):
    """Review counterexamples (2026-10-24): a corpus smaller than one topic's judgments gave an unjudged share of
    -100%, and NaN, infinite or out-of-range differences got a break-even."""
    t, q = _qrels_files(tmp_path, judged=(1,))                   # topic 1 has 4 distinct judged documents
    with pytest.raises(InputError, match='smaller than the 4 distinct documents judged for one topic'):
        audit_query_qrels(t, q, 3, 'toy')
    for d in (float('nan'), float('inf'), 2.0):
        r = c_star(d, 0.5, 'auc')
        assert not r['applicable'] and r['c_star'] is None and 'finite difference in [-1, 1]' in r['statement']
    assert 'its magnitude exceeds c*' in c_star(-0.02, 0.5, 'auc')['statement']    # c is negative when delta_O is
    (tmp_path / 'units.txt').write_text('a\nb\n')
    (tmp_path / 'r.csv').write_text('x,y\na,b\n')
    for d in ('nan', 'inf', '1.5'):                                # an input error, not SystemExit (the page calls main)
        assert cli_main(['pair-classification', '--pairs', str(tmp_path / 'r.csv'), '--a-col', 'x', '--b-col', 'y',
                         '--universe-units', str(tmp_path / 'units.txt'), '--delta-O', d, '--metric', 'auc',
                         '--out', str(tmp_path / 'c.json')]) == 2
        assert 'input error: --delta-O must be a finite difference between -1 and 1' in capsys.readouterr().err
    card = audit_query_qrels(t, q, 1000, 'toy')
    card['exclusion']['within_query']['unjudged_share']['min'] = -1.0
    card['comparison'] = {'metric': 'auc', 'delta_O': float('nan'), 'W': 0.5, 'c_star': None}
    assert validate_card(card) == ['exclusion.within_query.unjudged_share must lie in [0, 1]',
                                   'comparison.delta_O must be a finite difference in [-1, 1]']


def _enumerated_query_shares(uni, mask, test, known):
    """W_q by listing every candidate of every filtered query (the definition, for the tests)."""
    trip = {(uni.units[a], r, uni.units[b]) for (a, b), r in zip(uni.edges.tolist(), uni.edge_types.tolist())}
    trip |= {(uni.units[a], r, uni.units[b]) for a, r, b in list(known) + list(test)}
    out = []
    for a, r, b in test:
        h, t = uni.units[a], uni.units[b]
        for fixed, side in ((h, 'tail'), (t, 'head')):
            cand = [v for v in uni.units if ((fixed, r, v) if side == 'tail' else (v, r, fixed)) not in trip]
            out.append(sum(not mask[uni.index[v]] for v in cand) / len(cand))
    return out


def test_query_shares_follow_the_filtered_protocol():
    """Review counterexample (2026-10-24): entities a-f, release {a, b, c}, so the entity share is 1/2. The test
    tail query (a, r, ?) has b, d, e and f as known answers; its candidates a and c are all kept, so W_q = 0,
    while the head query (?, r, b) keeps b, c of the five candidates b-f: W_q = 3/5."""
    uni = from_triples([('a', 'r', 'b'), ('a', 'r', 'd'), ('a', 'r', 'e'), ('a', 'r', 'f'), ('b', 'r', 'c'),
                        ('b', 'r', 'a')], 'six')
    mask = np.array([u in 'abc' for u in uni.units])
    ix = uni.index
    r = query_shares(uni, mask, [(ix['a'], 'r', ix['b'])])
    assert r['W_q'] == {'min': 0.0, 'median': 0.3, 'max': 0.6} and r['W_bar'] == pytest.approx(0.3)
    assert r['queries'] == 2 and r['queries_without_excluded'] == 1
    rng = np.random.default_rng(3)                                 # random typed graph against enumeration
    names = [f'e{i}' for i in range(40)]
    trips = {(names[a], f'r{k}', names[b]) for a, b, k in rng.integers(0, [40, 40, 3], size=(150, 3)) if a != b}
    uni = from_triples(sorted(trips), 'random')
    mask = rng.random(uni.n) < 0.4
    tr = [(uni.index[h], r, uni.index[t]) for h, r, t in sorted(trips)[:20]]
    extra = [(uni.index['e1'], 'r9', uni.index['e2'])]               # a release triple the universe lacks
    got = query_shares(uni, mask, tr, extra)
    want = _enumerated_query_shares(uni, mask, tr, extra)
    assert got['queries'] == len(want) and got['W_bar'] == pytest.approx(np.mean(want), abs=1e-12)
    assert got['W_q']['min'] == pytest.approx(min(want)) and got['W_q']['max'] == pytest.approx(max(want))


def test_user_shares_follow_all_item_ranking():
    """A test user's candidates are the universe's items minus every item the user is known to have used."""
    rng = np.random.default_rng(5)
    pairs = sorted({(f'u{a}', f'i{b}') for a, b in rng.integers(0, [15, 25], size=(120, 2))})
    uni = from_pairs(pairs, 'toy')
    mask = rng.random(uni.n) < 0.5
    tests = [uni.index[f'u:u{k}'] for k in range(8) if f'u:u{k}' in uni.index]
    known = [(uni.index['u:u0'], uni.index[f'i:i{k}']) for k in range(25) if f'i:i{k}' in uni.index][:3]
    got = user_shares(uni, mask, tests, known)
    items = [i for i in range(uni.n) if uni.part[i] == 1]
    seen = {(a, b) for a, b in uni.edges.tolist()} | set(known)
    want = []
    for u in tests:
        cand = [i for i in items if (u, i) not in seen]
        want.append(sum(not mask[i] for i in cand) / len(cand))
    assert got['queries'] == len(tests) and got['W_bar'] == pytest.approx(np.mean(want), abs=1e-12)


def test_break_even_uses_the_mean_per_query_share(tmp_path):
    """The KG cards compute c* from the mean of the per-query shares, which makes it exact for Corollary 1's c."""
    (tmp_path / 'u.tsv').write_text(''.join(f'e{i}\tr\te{j}\n' for i in range(12) for j in (i + 1, i + 3) if j < 12))
    rel = tmp_path / 'rel'
    rel.mkdir()
    for s in ('train', 'valid', 'test'):
        (rel / f'{s}.txt').write_text('e0\tr\te1\ne1\tr\te2\ne2\tr\te3\ne3\tr\te4\n')
    out = tmp_path / 'c.json'
    assert cli_main(['kg-triples', '--release', str(rel), '--universe', str(tmp_path / 'u.tsv'), '--format', 'tsv',
                     '--delta-O', '-0.02', '--metric', 'auc', '--out', str(out)]) == 0
    card = json.loads(out.read_text(encoding='utf-8'))
    pq, cmp_ = card['exclusion']['per_query'], card['comparison']
    assert pq['queries'] == 8 and cmp_['W'] == pq['W_bar'] != card['exclusion']['W']
    assert cmp_['c_star'] == pytest.approx(0.02 / pq['W_bar'], abs=1e-6) and 'exactly when' in cmp_['statement']
    assert any('Cov(W_q, c_q)' in a for a in cmp_['assumptions']) and validate_card(card) == []
    assert 'Per query, on the universe' in to_markdown(card)
    assert 'the same 8 test queries' in cmp_['statement'] and "benchmark's own filter it carries no" in cmp_['statement']


def test_queries_without_candidates_or_kept_units_are_counted():
    """Review counterexamples (2026-10-27): a query whose every unit is a known answer has no candidate at all and
    was dropped silently; a query whose kept units are all known answers has W_q = 1, so the benchmark difference
    is undefined on it, and a break-even was still given."""
    uni = from_triples([('a', 'r', 'a'), ('a', 'r', 'b'), ('a', 'r', 'c'), ('b', 'r', 'c')], 'all known')
    ix = uni.index
    r = query_shares(uni, np.array([u in 'ab' for u in uni.units]), [(ix['a'], 'r', ix['b'])])
    assert r['queries_without_candidates'] == 1 and r['queries'] == 1          # the tail query (a, r, ?) has none
    uni = from_triples([('a', 'r', 'a'), ('a', 'r', 'b'), ('b', 'r', 'c'), ('c', 'r', 'd')], 'none kept')
    ix = uni.index
    r = query_shares(uni, np.array([u in 'ab' for u in uni.units]), [(ix['a'], 'r', ix['b'])])
    assert r['queries_without_kept'] == 1 and r['W_q']['max'] == 1.0          # (a, r, ?) keeps neither a nor b
    cs = c_star(-0.02, r['W_bar'], 'auc', per_query=r)
    assert not cs['applicable'] and cs['c_star'] is None and 'keep no candidate of the release' in cs['statement']


def test_break_even_warns_when_test_triples_are_skipped(tmp_path):
    """Review counterexample (2026-10-27): YAGO3-10's shares use 4,827 of its 5,000 test triples, while a model
    difference from a leaderboard covers all of them; the card must say that the break-even then carries no
    guarantee (wording changed from "approximate" on 2026-10-28: there is no bound on the mismatch)."""
    (tmp_path / 'u.tsv').write_text(''.join(f'e{i}\tr\te{j}\n' for i in range(12) for j in (i + 1, i + 3) if j < 12))
    rel = tmp_path / 'rel'
    rel.mkdir()
    for s in ('train', 'valid'):
        (rel / f'{s}.txt').write_text('e0\tr\te1\ne1\tr\te2\ne2\tr\te3\ne3\tr\te4\n')
    (rel / 'test.txt').write_text('e0\tr\te1\ne1\tr\te2\ne0\tr\tzz\n')               # zz is not in the universe
    args = ['kg-triples', '--release', str(rel), '--universe', str(tmp_path / 'u.tsv'), '--format', 'tsv']
    out = tmp_path / 'c.json'
    assert cli_main(args + ['--out', str(out)]) == 0
    assert not any('Left out of the per-query shares' in w for w in json.loads(out.read_text())['warnings'])
    assert cli_main(args + ['--delta-O', '0.01', '--metric', 'auc', '--out', str(out)]) == 0
    card = json.loads(out.read_text(encoding='utf-8'))
    assert card['exclusion']['per_query']['test_triples_used'] == 2
    assert any('1 of the 3 test triples (an entity outside the universe)' in w and 'no guarantee' in w
               for w in card['warnings'])
