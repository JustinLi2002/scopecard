"""Golden tests: the package must reproduce the numbers of the 2026-10 pilot
(experiments/pilot_decidability.tsv, results/kg/wn18_scope.txt, results/kg/yago3_10_scope.txt,
results/recsys/{gowalla,amazon-book}_scope.txt, results/ir/trecdl20{19,20}_passage_scope.txt,
results/graph/ogbl_citation2_scope.txt) and of the declared
WN18RR target (results/wn18rr_target/summary.txt) from the same public inputs."""
import os
import shutil

import numpy as np
import pytest

from conftest import PILOT
from scopecard import map_triples, search_rules, unit_exclusion
from scopecard.io import read_kg_release
from scopecard.pipelines import (audit_kg_triples, audit_kg_wordnet, audit_ogbl_citation2, audit_query_qrels,
                                        audit_recsys_checkins, audit_recsys_ratings)
from scopecard.card import to_json
from scopecard.universes import (WN18_RELATIONS, WN18RR_RELATIONS, export_wordnet30_table, load_wordnet30,
                                        load_wordnet30_table)

WORDNET = os.path.join(PILOT, 'kg', 'wordnet30', 'WordNet-3.0', 'dict')
WN18RR = os.path.join(PILOT, 'kg', 'conve', 'WN18RR')
WN18 = os.path.join(PILOT, 'kg', 'wn18_orig', 'wordnet-mlj12')
YAGO3_10 = os.path.join(PILOT, 'kg', 'conve', 'YAGO3-10')
YAGO_FACTS = os.path.join(PILOT, 'kg', 'yago3', 'yagoFacts.ttl')
GOWALLA_REL = os.path.join(PILOT, 'recsys', 'lightgcn', 'gowalla')
GOWALLA_RAW = os.path.join(PILOT, 'recsys', 'raw_gowalla', 'loc-gowalla_totalCheckins.txt.gz')
AMAZON_REL = os.path.join(PILOT, 'recsys', 'lightgcn', 'amazon-book')
AMAZON_RAW = os.path.join(PILOT, 'recsys', 'raw_amazon2014', 'ratings_Books.csv')
TREC = os.path.join(PILOT, 'ir')
CITATION2 = os.path.join(PILOT, 'graph', 'citation2', 'citation-v2')

need_wordnet = pytest.mark.skipif(not (os.path.isdir(WORDNET) and os.path.isdir(WN18RR)), reason='WordNet / WN18RR not present')
need_gowalla = pytest.mark.skipif(not (os.path.exists(GOWALLA_RAW) and os.path.isdir(GOWALLA_REL)), reason='Gowalla not present')


@need_wordnet
def test_wn18rr_declared_target():
    """POS-aware identity: O 41,105, U 71,094, W_rule 0.633642; 161 merged ids; 5 unmatched triples."""
    card = audit_kg_wordnet(WN18RR, WORDNET, 'WN18RR', identity='pos_offset')
    ex, m = card['exclusion'], card['mapping']
    assert (ex['included'], ex['excluded'], ex['units']) == (41105, 71094, 112199)
    assert round(ex['W'], 6) == 0.633642
    assert m['merged_ids'] == 161 and m['unmatched_triples'] == 5 and m['release_ids'] == 40943
    assert card['hygiene']['unseen_test_units']['test_triples_affected'] == 210
    # per-query shares under the target's filter: the measured experiment's own analysis (analyze_wn18rr.py,
    # records.tsv E2/primary W over its 6,266 bench_test queries) gives 0.6337159107
    pq = ex['per_query']
    assert pq['queries'] == 6266 and pq['W_bar'] == 0.633716 and pq['queries_without_excluded'] == 0


@need_wordnet
def test_wordnet_table_equals_dict(tmp_path):
    """The compact table the browser page ships gives the same universe as the dict folder (both identities,
    both relation sets) and the same WN18RR card byte for byte; rebuilding it gives the same bytes."""
    t1, t2 = tmp_path / 'a.tsv.gz', tmp_path / 'b.tsv.gz'
    export_wordnet30_table(WORDNET, str(t1))
    export_wordnet30_table(WORDNET, str(t2))
    assert t1.read_bytes() == t2.read_bytes()
    for rels in (tuple(WN18_RELATIONS), WN18RR_RELATIONS):
        for identity in ('pos_offset', 'bare_offset'):
            for iso in (True, False):
                a = load_wordnet30(WORDNET, rels, identity, iso)
                b = load_wordnet30_table(str(t1), rels, identity, iso)
                assert a.version == b.version and np.array_equal(a.units, b.units) and np.array_equal(a.edges, b.edges)
                assert np.array_equal(a.edge_types, b.edge_types) and np.array_equal(a.strata['pos'], b.strata['pos'])
    c1 = audit_kg_wordnet(WN18RR, WORDNET, 'WN18RR', delta_O=-0.0142, metric='auc')
    c2 = audit_kg_wordnet(WN18RR, str(t1), 'WN18RR', delta_O=-0.0142, metric='auc')
    assert c1['universe']['fingerprint'] == c2['universe']['fingerprint'] and to_json(c1) == to_json(c2)
    w = c1['exclusion']['per_query']['W_bar']                     # the break-even uses the mean per-query share
    assert c1['comparison']['W'] == w != c1['exclusion']['W']
    assert c1['comparison']['delta_U_break_even'] == pytest.approx((1 - w) * 0.0142 / w, abs=2e-6)
    assert c1['comparison']['c_star'] == pytest.approx(0.0142 / w, abs=2e-6)


@need_wordnet
def test_wn18rr_benchmark_identity():
    """The benchmark's bare-offset keying: 71,032 of 111,975 excluded (0.6344); 76,431 of 117,374 over all offsets."""
    card = audit_kg_wordnet(WN18RR, WORDNET, 'WN18RR', identity='bare_offset')
    ex = card['exclusion']
    assert (ex['excluded'], ex['units']) == (71032, 111975)
    assert round(ex['W_all_synsets'] * 117374) == 76431


@need_wordnet
@pytest.mark.skipif(not os.path.isdir(WN18), reason='WN18 release not present')
def test_wn18_rule_does_not_reproduce(tmp_path):
    """WN18's stated '< 15 triplets' rule: Jaccard 0.150; best both-ends threshold T=5 at 0.597;
    best seeds+neighbours T=16 at 0.418 (18-relation graph, bare offsets)."""
    for s in ('train', 'valid', 'test'):
        shutil.copy(os.path.join(WN18, f'wordnet-mlj12-{s}.txt'), tmp_path / f'{s}.txt')
    trips, _ = read_kg_release(str(tmp_path))
    uni = load_wordnet30(WORDNET, identity='bare_offset', include_isolated=False)
    rel, rep = map_triples(trips, uni)
    assert rel.unit_mask.sum() == 40943
    res = search_rules(uni, rel.unit_mask, families=('both_ends', 'seeds_neighbours'), stated={'family': 'both_ends', 'T': 15})
    fam = {r['family']: r for r in res['families']}
    assert round(res['stated']['jaccard'], 3) == 0.150
    assert fam['both_ends']['T'] == 5 and round(fam['both_ends']['jaccard'], 3) == 0.597
    assert fam['seeds_neighbours']['T'] == 16 and round(fam['seeds_neighbours']['jaccard'], 3) == 0.418
    assert res['best']['verdict'] == 'not reproduced'


@pytest.mark.skipif(not (os.path.exists(YAGO_FACTS) and os.path.isdir(YAGO3_10)), reason='YAGO 3.0.2 / YAGO3-10 not present')
def test_yago3_10_against_yago302():
    """YAGO3-10 vs YAGO 3.0.2 yagoFacts (about 3 minutes): 121,138 of 123,182 release entities in the universe,
    1,054,372 of 1,089,040 triples found; W 0.9540 (2,514,177 of 2,635,315 entities excluded); 'both ends with
    >= 10 facts' selects 123,189 entities at Jaccard 0.967, and the search finds that rule unprompted."""
    card = audit_kg_triples(YAGO3_10, YAGO_FACTS, 'yago-ttl', 'YAGO3-10', stated={'family': 'both_ends', 'T': 10})
    m, ex, rd = card['mapping'], card['exclusion'], card['rule_diagnosis']
    assert (m['release_ids'], m['mapped_ids'], m['unmapped_ids']) == (123182, 121138, 2044)
    assert (m['release_triples'], m['matched_triples']) == (1089040, 1054372)
    assert m['matched_triples_by_split'] == {'train': 1044687, 'valid': 4858, 'test': 4827}
    assert (ex['excluded'], ex['units']) == (2514177, 2635315) and round(ex['W'], 4) == 0.9540
    st = rd['stated']
    assert (st['selected'], st['overlap']) == (123189, 121136) and round(st['jaccard'], 3) == 0.967
    assert st['verdict'] == 'approximately reproduced'
    assert (rd['best']['family'], rd['best']['T']) == ('both_ends', 10)


@pytest.mark.parametrize('year,evaluated,W,max_ratio,unjudged', [
    ('2019', 43, 0.785, 0.534, (0.999934, 0.999985)),
    ('2020', 54, 0.730, 0.350, (0.999958, 0.999983)),
])
def test_trec_dl_passage(year, evaluated, W, max_ratio, unjudged):
    """Topic-level W over the 200 released test topics; the 2019 overview's topic rule holds for every
    released topic (min relevant exactly 3); per-topic unjudged share of the 8,841,823 passages."""
    topics, qrels = os.path.join(TREC, f'msmarco-test{year}-queries.tsv.gz'), os.path.join(TREC, f'{year}qrels-pass.txt')
    if not (os.path.exists(topics) and os.path.exists(qrels)):
        pytest.skip('TREC DL files not present')
    card = audit_query_qrels(topics, qrels, 8841823, f'TREC DL {year} passage', relevant_grade=2, min_relevant=3, max_ratio=0.6)
    ex, rc = card['exclusion'], card['rule_check']
    assert (ex['included'], ex['units']) == (evaluated, 200) and round(ex['W'], 3) == W
    assert card['mapping']['unmapped_ids'] == 0
    assert rc['violations'] == 0 and rc['observed_min_relevant'] == 3 and round(rc['observed_max_ratio'], 3) == max_ratio
    u = ex['within_query']['unjudged_share']
    assert (round(u['min'], 6), round(u['max'], 6)) == unjudged


@pytest.mark.skipif(not (os.path.exists(AMAZON_RAW) and os.path.isdir(AMAZON_REL)), reason='Amazon-Book / 2014 Books ratings not present')
def test_amazon_book_against_2014_ratings():
    """Amazon-Book vs the 2014 Books ratings (22,507,155 distinct pairs; several minutes): every release user, item and
    interaction is in the raw ratings; pair-level W 0.999742 (release-positives convention); the stated 10-core keeps
    158,650 users and 128,939 items and contains the whole release (52,643 + 91,599 units), so it does not reproduce it."""
    card = audit_recsys_ratings(AMAZON_REL, AMAZON_RAW, 'Amazon-Book', positives='release')
    m, ex, st = card['mapping'], card['exclusion'], card['rule_diagnosis']['stated']
    assert (m['release_ids'], m['unmapped_ids']) == (52643 + 91599, 0)
    assert (ex['part0_units'], ex['part1_units'], ex['observed_pairs']) == (8026324, 2330066, 22507155)
    assert (ex['part0_retained'], ex['part1_retained'], ex['observed_pairs_retained']) == (52643, 91599, 2984108)
    assert round(ex['W_pairs'], 6) == 0.999742
    assert (st['family'], st['T'], st['selected'], st['overlap']) == ('kcore', 10, 158650 + 128939, 52643 + 91599)
    assert round(st['jaccard'], 6) == round(144242 / 287589, 6) and st['verdict'] == 'not reproduced'
    assert card['rule_diagnosis']['best']['verdict'] == 'not reproduced'
    assert 'leave out 99.97%' in card['scope_statement']


@pytest.mark.skipif(not os.path.isdir(os.path.join(CITATION2, 'split', 'time')), reason='ogbl-citation2 not present')
def test_ogbl_citation2_source_rule():
    """Source papers vs all 2019 papers (about 1 minute; OGB .pt files read without torch): 21,995 of 108,591
    excluded (0.2025); 'at least 6 in-subset references' reproduces the sources exactly; negatives are never
    later than the source (3,211,832 / 3,212,415 same year) and uniform over nodes (lower-degree half 0.500)."""
    card = audit_ogbl_citation2(CITATION2)
    ex, rd, h = card['exclusion'], card['rule_diagnosis'], card['hygiene']
    assert (ex['excluded'], ex['units']) == (21995, 108591) and round(ex['W'], 4) == 0.2025
    assert (rd['best']['family'], rd['best']['T'], rd['best']['verdict']) == ('threshold', 6, 'reproduced')
    assert h['sources_identical_in_valid_and_test'] and h['source_years'] == [2019]
    ns = h['negative_sampling']
    assert [ns[s]['same_year_as_source'] for s in ('valid', 'test')] == [3211832, 3212415]
    assert all(ns[s]['later_than_source'] == 0 and round(ns[s]['share_in_lower_degree_half'], 3) == 0.5 for s in ns)
    assert [ns[s]['with_no_training_edge'] for s in ('valid', 'test')] == [39579, 39850]


@need_gowalla
def test_gowalla_against_raw_checkins():
    """Pair-level W 0.991088 (release-positives convention of the pilot); users exact; 10-core reproduces
    the release up to 7 items (units Jaccard 70,839 / 70,846)."""
    card = audit_recsys_checkins(GOWALLA_REL, GOWALLA_RAW, 'Gowalla', positives='release')
    ex = card['exclusion']
    assert round(ex['W_pairs'], 6) == 0.991088
    assert (ex['part0_retained'], ex['part1_retained']) == (29858, 40981)
    st = card['rule_diagnosis']['stated']
    assert st['family'] == 'kcore' and round(st['jaccard'], 6) == round(70839 / 70846, 6)
    assert st['verdict'] == 'approximately reproduced'
