"""End-to-end audits for the task types of version 1 that have a built-in universe."""
import os
from typing import Dict, Optional, Sequence

import numpy as np

from .card import build_card
from .compare import c_star
from .core import Universe, counts
from .hygiene import pair_release_checks, reverse_leakage, unseen_test_units
from .io import (InputError, InputLog, basename, progress, read_int_column, read_kg_release, read_lightgcn_release,
                 read_pair_rows, read_topic_ids, read_torch_numpy_dict, read_trec_qrels, read_unit_list, sha256)
from .judgments import check_topic_rule, judgment_profile
from .mapping import check_mapping, map_triples, map_units, match_triples_exact
from .rules import search_depth_threshold, search_rules
from .scope import depth_profile, derived_scope, pair_exclusion, query_shares, unit_exclusion, user_shares
from .universes import (WN18_RELATIONS, WN18RR_RELATIONS, from_pairs, from_unit_pairs, load_ratings_csv,
                        load_snap_checkins, load_triples_tsv, load_wordnet30_source, load_yago3_facts)

# How each task chooses its evaluation candidates (card field `support`). Only when candidates are enumerated
# over the release's units is a unit absent from the release structurally excluded.
SUPPORT = {
    'kg': {'candidates': 'every entity of the release, in filtered head and tail ranking', 'structural': True,
           'basis': 'Link prediction ranks each test query against the release entity set, so a universe entity '
                    'absent from the release is never a candidate.'},
    'recsys': {'candidates': 'every item of the release, for each test user of the release (all-item ranking)',
               'structural': True,
               'basis': 'NGCF and LightGCN rank all release items for each test user, so users and items absent '
                        'from the release are never evaluated.'},
    'qrels': {'candidates': 'the judged topics; within a topic, every corpus document', 'structural': True,
              'basis': 'Only topics with judgments are scored, so a released test topic without judgments is never '
                       'evaluated; inside a topic, unjudged documents count as non-relevant.'},
    'citation2': {'candidates': 'the valid/test source papers, each ranked against its sampled negatives',
                  'structural': True,
                  'basis': 'Only source papers are queried, so a paper of the source year that is not a source is '
                           'never evaluated as a query.'},
    'pairs_unknown': {'candidates': 'the pairs listed in the release', 'structural': False,
                      'basis': 'The release lists its evaluated pairs but not how they were drawn, so an absent unit '
                               'or pair may have had a nonzero chance of being drawn.'},
    'pairs_all': {'candidates': 'every pair of the release units (declared by the user)', 'structural': True,
                  'basis': 'Declared: the benchmark evaluates every pair among its units, so a pair with an end outside '
                           'the release is never a candidate.'},
}


def comparison(delta_O: Optional[float], metric: Optional[str], W: Optional[float], W_source: str,
               structural: bool = True, per_query: Optional[Dict] = None) -> Optional[Dict]:
    """The card's comparison block. The metric must be stated; no break-even without an established excluded share."""
    if delta_O is None:
        return None
    if metric is None:
        raise ValueError('state the metric of delta_O (auc, mrr or hits): the break-even exists only for AUC-type metrics')
    if not structural:
        return {'metric': metric, 'delta_O': delta_O, 'W': None, 'W_source': W_source, 'applicable': False,
                'c_star': None, 'delta_U_break_even': None,
                'statement': 'No break-even is given: the release does not show how its candidates were drawn, so no '
                             'excluded share is established.'}
    return c_star(delta_O, W, metric, W_source, per_query)


def skipped_test_warning(per_query: Dict, test_triples: int, delta_O: Optional[float],
                         why: str = 'an entity outside the universe') -> list:
    """With a model difference, the queries the break-even uses must be the ones the difference was measured on."""
    used = per_query.get('test_triples_used', test_triples)
    if delta_O is None or used >= test_triples:
        return []
    return [f'Left out of the per-query shares and the break-even: {test_triples - used:,} of the {test_triples:,} test '
            f'triples ({why}); they use the other {used:,}. A model difference measured on all test triples is on a '
            f'different query set, for which the break-even carries no guarantee; recompute the difference on the '
            f'{used:,} triples the shares use.']


def audit_kg_wordnet(release_dir: str, wordnet_dict: str, name: str, relations: str = 'wn18rr',
                     identity: str = 'pos_offset', delta_O: Optional[float] = None, metric: Optional[str] = None) -> Dict:
    """A WordNet-derived KG benchmark (WN18, WN18RR) against WordNet 3.0.

    identity 'pos_offset' (correct synset identity; release ids, which are bare offsets, are mapped
    through the triples) or 'bare_offset' (the benchmark's own keying)."""
    rels = WN18RR_RELATIONS if relations == 'wn18rr' else tuple(WN18_RELATIONS)
    progress('reading the release')
    trips, hashes = read_kg_release(release_dir)
    progress('reading WordNet 3.0')
    uni = load_wordnet30_source(wordnet_dict, rels, identity=identity, include_isolated=True)
    key = (lambda s: s[1:]) if identity == 'pos_offset' else None
    progress('mapping release ids through the triples')
    rel, rep = map_triples(trips, uni, key=key, name=name)
    warnings = check_mapping(rep.to_dict(), 'WordNet 3.0 (through the triples of the chosen relation set)')
    progress('computing exclusion and depth')
    depth = uni.depth()
    connected = depth > 0
    excl = unit_exclusion(uni, rel.unit_mask, among=connected)
    excl['W_all_synsets'] = float(1 - rel.unit_mask.sum() / uni.n)
    progress('computing per-query excluded shares')
    as_triples = lambda rows: list(zip(rel.edges[rows, 0].tolist(), rel.edge_types[rows].tolist(), rel.edges[rows, 1].tolist()))
    excl['per_query'] = query_shares(uni, rel.unit_mask, as_triples(rel.splits['test']), as_triples(np.arange(len(rel.edges))),
                                     among=connected, test_triples=len(trips['test']))
    warnings += skipped_test_warning(excl['per_query'], len(trips['test']), delta_O,
                                     'no matching triple in the universe')
    what = 'synsets' if identity == 'pos_offset' else 'bare synset offsets'
    counted = f"the {excl['units']:,} WordNet 3.0 {what} that carry at least one triple of its {len(rels)} relations"
    # WN18RR inherits WN18's entity set, whose stated rule counts triples of all 18 WN18 relations:
    # diagnose rules on that graph, with the same release units
    progress('searching construction rules')
    rule_uni = uni if relations == 'wn18' else load_wordnet30_source(wordnet_dict, tuple(WN18_RELATIONS), identity=identity)
    rule_mask = np.zeros(rule_uni.n, bool)
    rule_mask[[rule_uni.index[u] for u in uni.units[rel.unit_mask].tolist()]] = True
    rule_mask &= rule_uni.depth() > 0
    card = build_card(
        benchmark={'name': name, 'files_sha256': hashes, 'splits': {s: len(t) for s, t in trips.items()}},
        universe={'name': uni.name, 'version': uni.version, 'units': uni.n, 'units_with_relations': int(connected.sum()),
                  'relations': len(rels), 'fingerprint': uni.fingerprint()},
        task='kg_link_prediction', mapping=rep.to_dict(), exclusion=excl,
        depth=depth_profile(uni, rel.unit_mask, depth, among=connected),
        rules=search_rules(rule_uni, rule_mask, stated={'family': 'both_ends', 'T': 15},
                           unmapped_release_units=len(rep.unmapped_ids)),
        scope=derived_scope(uni, rel.unit_mask, depth, among=connected, thresholds=range(1, 41)),
        comparison=comparison(delta_O, metric, excl['per_query']['W_bar'], 'filtered test queries on the universe',
                              per_query=excl['per_query']),
        hygiene={'reverse_leakage': reverse_leakage(trips['test'], trips['train']),
                 'unseen_test_units': unseen_test_units(trips['test'], trips['train'])},
        support=SUPPORT['kg'], counted=counted, warnings=warnings,
        notes='Universe = WordNet 3.0 synsets carrying >= 1 triple of the benchmark relations (W); '
              'W_all_synsets also counts synsets with none. Depth = number of universe triples on the synset.')
    return card


def audit_recsys_checkins(release_dir: str, checkins_path: str, name: str, positives: str = 'universe',
                          k_core: int = 10) -> Dict:
    """A LightGCN/NGCF-format recommendation release against raw SNAP check-ins (e.g. Gowalla)."""
    log = InputLog()
    progress('reading the raw check-ins')
    uni = load_snap_checkins(checkins_path, name='SNAP Gowalla check-ins', log=log)
    return audit_recsys(release_dir, uni, name, positives, k_core, log=log, item_word='locations',
                        notes='Universe = distinct (user, location) pairs in the raw check-ins; candidates = unobserved pairs.')


def audit_recsys_ratings(release_dir: str, ratings_path: str, name: str, positives: str = 'universe',
                         k_core: int = 10, universe_name: str = '') -> Dict:
    """A LightGCN/NGCF-format recommendation release against an Amazon 2014 ratings CSV (e.g. Amazon-Book vs
    ratings_Books.csv). About 22.5 M interactions for Books: expect minutes and several GB of memory."""
    log = InputLog()
    progress('reading the raw ratings')
    uni = load_ratings_csv(ratings_path, name=universe_name, log=log)
    return audit_recsys(release_dir, uni, name, positives, k_core, log=log,
                        notes='Universe = distinct (user, item) pairs in the raw ratings; candidates = unobserved pairs.')


def audit_recsys(release_dir: str, uni: Universe, name: str, positives: str = 'universe', k_core: int = 10,
                 notes: str = '', log: Optional[InputLog] = None, item_word: str = 'items') -> Dict:
    """A LightGCN/NGCF-format recommendation release against a bipartite universe of raw interactions.
    `k_core` is the stated rule (NGCF / LightGCN: the 10-core)."""
    progress('reading the release')
    splits, hashes = read_lightgcn_release(release_dir)
    rel_ids = {'u:' + u for s in splits.values() for u, _ in s} | {'i:' + i for s in splits.values() for _, i in s}
    progress('mapping release ids')
    rel, rep = map_units(sorted(rel_ids), uni, name=name)
    warnings = check_mapping(rep.to_dict(), uni.name, 'release users and items')
    progress('computing exclusion and depth')
    rel_pairs = {p for s in splits.values() for p in s}
    excl = unit_exclusion(uni, rel.unit_mask, by_strata=False)
    excl.update(pair_exclusion(uni, rel.unit_mask, release_pairs=len(rel_pairs) if positives == 'release' else None))
    bp = excl['by_part']
    progress('computing per-query excluded shares')
    ix = uni.index
    known = [(ix['u:' + u], ix['i:' + i]) for u, i in rel_pairs if 'u:' + u in ix and 'i:' + i in ix]
    tests = sorted({ix['u:' + u] for u, _ in splits['test'] if 'u:' + u in ix})
    # users the release drops are whole queries removed (Proposition 1(a)); W_q covers the users it keeps
    excl['per_query'] = user_shares(uni, rel.unit_mask, tests, known, excluded_queries_share=bp['0']['W'])
    counted = (f"the {excl.get('candidate_pairs', 0):,} unobserved user-item pairs among the {bp['0']['units']:,} users and "
               f"{bp['1']['units']:,} {item_word} of {uni.name}")
    depth = uni.depth()
    log = log or InputLog()
    progress('searching construction rules')
    card = build_card(
        benchmark={'name': name, 'files_sha256': hashes, 'interactions': len(rel_pairs)},
        universe={'name': uni.name, 'version': uni.version, 'units': uni.n, 'pairs': int(len(uni.edges)),
                  'fingerprint': uni.fingerprint()},
        task='recommendation', support=SUPPORT['recsys'], mapping=rep.to_dict(), exclusion=excl,
        depth=depth_profile(uni, rel.unit_mask, depth),
        rules=search_rules(uni, rel.unit_mask, families=('kcore', 'threshold', 'both_ends'), thresholds=range(1, 21),
                           stated={'family': 'kcore', 'T': k_core}, unmapped_release_units=len(rep.unmapped_ids)),
        scope=derived_scope(uni, rel.unit_mask, depth, thresholds=range(1, 21), pairs=True),
        counted=counted, warnings=warnings + log.warnings(), inputs=log.to_dict(), notes=notes)
    return card


def audit_query_qrels(topics_path: str, qrels_path: str, corpus_size: int, name: str, relevant_grade: int = 1,
                      min_relevant: Optional[int] = None, max_ratio: Optional[float] = None,
                      universe_name: str = '', rule_source: str = '') -> Dict:
    """A TREC-style query-level benchmark: the released test topics are the universe, the topics with
    judgments are the release. TREC DL passage: relevant_grade 2; the 2019 overview's topic rule is
    min_relevant 3, max_ratio 0.6."""
    log = InputLog()
    progress('reading topics and qrels')
    topics = read_topic_ids(topics_path)
    qrels = read_trec_qrels(qrels_path, log=log)
    uni = Universe(name=universe_name or f'{name} test topics', version=basename(topics_path),
                   units=np.array(sorted(set(topics)), dtype=object), edges=np.zeros((0, 2), dtype=np.int64))
    progress('mapping judged topics onto the topic list')
    most = max(len(d) for d in qrels.values())
    if corpus_size < most:                      # the unjudged share of a topic would be negative
        raise InputError(f'--corpus-size {corpus_size:,} is smaller than the {most:,} distinct documents judged for one '
                         f'topic. Give the number of documents in the collection the topics are ranked against '
                         f'(8,841,823 for the MS MARCO passage corpus).')
    rel, rep = map_units(sorted(qrels), uni, name=name)
    warnings = check_mapping(rep.to_dict(), f'the topic list ({basename(topics_path)})', 'judged topics') + log.warnings()
    excl = unit_exclusion(uni, rel.unit_mask, by_strata=False)
    excl['within_query'] = judgment_profile(qrels, corpus_size, relevant_grade)
    check = None
    if min_relevant is not None or max_ratio is not None:
        text = ' and '.join(x for x in (f'>= {min_relevant} relevant (grade >= {relevant_grade})' if min_relevant is not None else '',
                                         f'relevant/judged < {max_ratio}' if max_ratio is not None else '') if x)
        check = check_topic_rule(qrels, relevant_grade, min_relevant, max_ratio, text, rule_source)
    hashes = {os.path.basename(p): sha256(p) for p in (topics_path, qrels_path)}
    return build_card(
        benchmark={'name': name, 'files_sha256': hashes, 'evaluated_queries': len(qrels)},
        universe={'name': uni.name, 'version': uni.version, 'units': uni.n, 'fingerprint': uni.fingerprint()},
        task='query_level', mapping=rep.to_dict(), exclusion=excl, rule_check=check, support=SUPPORT['qrels'],
        counted=f'the {uni.n:,} {uni.name}', warnings=warnings, inputs=log.to_dict(),
        notes='Universe = the released test topics; release = topics with judgments. Within an evaluated topic, '
              'unjudged documents are scored as non-relevant by trec_eval (implicit negatives).')


def negative_sampling_profile(src: np.ndarray, neg: np.ndarray, year: np.ndarray, train_deg: np.ndarray,
                              ref_keys: np.ndarray, chunk: int = 10000) -> Dict:
    """Checks on sampled negatives (OGB citation style): later than the source, same year, actually
    cited by the source (ref_keys = sorted source * n + target over all release edges), share in the
    lower-degree half of all nodes (0.5 for uniform sampling), and negatives with no training edge."""
    n = len(year)
    low = np.zeros(n, bool)
    low[np.argsort(train_deg, kind='stable')[: n // 2]] = True
    later = same = cited = in_low = no_edge = 0
    for a in range(0, len(src), chunk):
        s, m = src[a:a + chunk], neg[a:a + chunk]
        ys, ym = year[s][:, None], year[m]
        later += int((ym > ys).sum())
        same += int((ym == ys).sum())
        k = (s[:, None] * n + m).ravel()
        pos = np.searchsorted(ref_keys, k)
        cited += int((ref_keys[np.minimum(pos, len(ref_keys) - 1)] == k).sum())
        in_low += int(low[m].sum())
        no_edge += int((train_deg[m] == 0).sum())
    total = int(neg.size)
    return {'sources': int(len(src)), 'negatives_per_source': int(neg.shape[1]), 'negatives': total,
            'later_than_source': later, 'same_year_as_source': same, 'cited_by_source': cited,
            'share_in_lower_degree_half': in_low / total, 'with_no_training_edge': no_edge}


def audit_ogbl_citation2(dataset_dir: str, name: str = 'ogbl-citation2', split: str = 'time',
                         source_year: Optional[int] = None, negatives: bool = True) -> Dict:
    """ogbl-citation2 (OGB folder with raw/node_year.csv.gz and split/<split>/{train,valid,test}.pt; .npz with the
    same keys also works). Query level: the units are the papers of the source year (default: the latest year);
    the release's queries are the valid/test source papers. Depth = references inside the subset, over the
    train, valid and test edges. The split files are read without torch (numpy arrays only)."""
    def split_file(s):
        for ext in ('.pt', '.npz'):
            p = os.path.join(dataset_dir, 'split', split, s + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f'split/{split}/{s}.pt or .npz in {dataset_dir}')
    year_path = os.path.join(dataset_dir, 'raw', 'node_year.csv.gz')
    progress('reading node years and split files')
    year = read_int_column(year_path)
    n = len(year)
    paths = {s: split_file(s) for s in ('train', 'valid', 'test')}
    sp = {s: read_torch_numpy_dict(p) for s, p in paths.items()}
    src_all = np.concatenate([sp[s]['source_node'] for s in sp])
    dst_all = np.concatenate([sp[s]['target_node'] for s in sp])
    refs = counts(src_all, n)
    y = int(year.max()) if source_year is None else int(source_year)
    papers = np.flatnonzero(year == y)
    uni = Universe(name=f'{name} papers published in {y}', version=f'node_year.csv.gz, split {split}',
                   units=np.array([str(i) for i in papers], dtype=object), edges=np.zeros((0, 2), dtype=np.int64))
    sources = {s: np.unique(sp[s]['source_node']) for s in ('valid', 'test')}
    progress('mapping source papers')
    rel, rep = map_units([str(i) for i in np.union1d(sources['valid'], sources['test'])], uni, name=name)
    warnings = check_mapping(rep.to_dict(), uni.name, 'source papers')
    progress('computing exclusion, depth and the negative-sampling checks')
    excl = unit_exclusion(uni, rel.unit_mask, by_strata=False)
    depth = refs[papers]
    hygiene = {'sources_identical_in_valid_and_test': bool(np.array_equal(sources['valid'], sources['test'])),
               'source_years': sorted({int(v) for s in ('valid', 'test') for v in np.unique(year[sp[s]['source_node']])})}
    if negatives and all('target_node_neg' in sp[s] for s in ('valid', 'test')):
        train_deg = counts(sp['train']['source_node'], n) + counts(sp['train']['target_node'], n)
        ref_keys = np.unique(src_all * n + dst_all)
        hygiene['negative_sampling'] = {s: negative_sampling_profile(sp[s]['source_node'], sp[s]['target_node_neg'], year,
                                                                     train_deg, ref_keys) for s in ('valid', 'test')}
    hashes = {os.path.relpath(p, dataset_dir).replace('\\', '/'): sha256(p) for p in [year_path] + list(paths.values())}
    return build_card(
        benchmark={'name': name, 'files_sha256': hashes, 'nodes': n, 'edges': {s: int(len(sp[s]['source_node'])) for s in sp},
                   'queries': int(rel.unit_mask.sum())},
        universe={'name': uni.name, 'version': uni.version, 'units': uni.n, 'fingerprint': uni.fingerprint()},
        task='query_level', mapping=rep.to_dict(), exclusion=excl, support=SUPPORT['citation2'],
        depth=depth_profile(uni, rel.unit_mask, depth),
        rules=search_depth_threshold(depth, rel.unit_mask, thresholds=range(1, 41), unmapped_release_units=len(rep.unmapped_ids)),
        scope=derived_scope(uni, rel.unit_mask, depth, thresholds=range(1, 41)),
        hygiene=hygiene, counted=f'the {uni.n:,} {uni.name}', warnings=warnings,
        notes=f'Universe = papers of {y}, the source year; release = valid/test source papers. Depth = references inside the '
              'subset (train + valid + test edges). The rule search is the threshold family on that depth.')


def audit_pair_classification(pair_files: Sequence[str], a_col, b_col, name: str, label_col=None, positive_label: str = '1',
                              header: bool = True, sep: Optional[str] = None, universe_units: Optional[str] = None,
                              universe_units_b: Optional[str] = None, universe_pairs: Optional[str] = None,
                              pair_cols=(0, 1), pair_header: bool = False, symmetric: bool = True,
                              stated: Optional[Dict] = None, delta_O: Optional[float] = None, universe_name: str = '',
                              metric: Optional[str] = None, candidate_rule: str = 'unknown') -> Dict:
    """A pair-classification release (labelled pairs, or a negatives-only file) against a declared universe of units,
    optionally with observed pairs (known interactions) that give depth. symmetric: one unit set and unordered pairs
    (protein-protein); otherwise drug-target style with two unit sets.

    A release lists its pairs but not how they were drawn, so by default (candidate_rule 'unknown') W and W_pairs
    are coverage: a unit or pair absent from the release is not shown to be excluded. With candidate_rule
    'all_pairs' the user declares that the benchmark evaluates every pair among its units; W_pairs is then the
    excluded share of the universe's unobserved pairs."""
    if candidate_rule not in ('unknown', 'all_pairs'):
        raise ValueError("candidate_rule must be 'unknown' or 'all_pairs'")
    log, warnings = InputLog(), []
    rows, hashes = [], {}
    progress('reading the release pairs')
    for p in pair_files:
        cols: list = []
        rows += read_pair_rows(p, a_col, b_col, label_col, header, sep, log=log, columns=cols)
        hashes[basename(p)] = sha256(p)
        low = {c.strip().lower(): c for c in cols if c not in (a_col, b_col)}
        named = [low[k] for k in ('label', 'labels', 'class', 'target', 'interaction', 'y') if k in low]
        if label_col is None and named:
            warnings.append(f'{basename(p)} has a column named {named[0]!r}, but no label column was given, so every '
                            f'row was read as a negative. If the file holds labelled pairs, name the label column.')
    if universe_units is None and universe_pairs is None:
        raise ValueError('declare the universe: a unit list (--universe-units), observed pairs (--universe-pairs), or both')
    progress('reading the universe')
    ua = read_unit_list(universe_units) if universe_units else None
    ub = read_unit_list(universe_units_b) if universe_units_b else None
    obs = ([(a, b) for a, b, _ in read_pair_rows(universe_pairs, pair_cols[0], pair_cols[1], None, pair_header, log=log)]
           if universe_pairs else [])
    srcs = [basename(p) for p in (universe_units, universe_units_b, universe_pairs) if p]
    uni, info = from_unit_pairs(ua, ub, obs, symmetric, name=universe_name or 'the declared universe',
                                version=', '.join(srcs))
    counted = f"the {uni.n:,} declared units" + (f" of {universe_name}" if universe_name else f" ({', '.join(srcs)})")
    if symmetric:
        ids = {x for a, b, _ in rows for x in (a, b)}
    else:
        ids = {'a:' + a for a, _, _ in rows} | {'b:' + b for _, b, _ in rows}
    progress('mapping release units')
    rel, rep = map_units(sorted(ids), uni, name=name)
    warnings = check_mapping(rep.to_dict(), universe_name or f"the declared units ({', '.join(srcs)})",
                             'release units') + warnings + log.warnings()
    progress('computing coverage and depth')
    excl = unit_exclusion(uni, rel.unit_mask, by_strata=False)
    excl.update(pair_exclusion(uni, rel.unit_mask))
    if candidate_rule == 'all_pairs':        # the statement then gives W_pairs, a share of pairs, not of units
        counted = f"the {excl.get('candidate_pairs', 0):,} unobserved pairs among " + counted
    key = (lambda a, b: (min(a, b), max(a, b))) if symmetric else (lambda a, b: ('a:' + a, 'b:' + b))
    # a == b is a self pair only in one unit set; across two sets the same id names two different units
    inside = {key(a, b) for a, b, _ in rows if (a != b or not symmetric) and all(x in uni.index for x in key(a, b))}
    if symmetric:
        all_pairs = uni.n * (uni.n - 1) // 2
    else:
        all_pairs = int((uni.part == 0).sum()) * int((uni.part == 1).sum())
    excl.update({'release_pairs_in_universe': len(inside), 'universe_pairs': all_pairs,
                 'pair_coverage': len(inside) / all_pairs if all_pairs else float('nan')})
    support = dict(SUPPORT['pairs_all' if candidate_rule == 'all_pairs' else 'pairs_unknown'])
    cmp_block = comparison(delta_O, metric, excl['W_pairs'], 'pair share', support['structural'])
    if candidate_rule == 'all_pairs':
        # the declaration says every pair among the release units is evaluated, so the file should list them all
        n_a = int((rel.unit_mask & (uni.part == 0)).sum()) if not symmetric else int(rel.unit_mask.sum())
        n_b = int((rel.unit_mask & (uni.part == 1)).sum()) if not symmetric else 0
        expected = n_a * (n_a - 1) // 2 if symmetric else n_a * n_b
        listed = len(inside)
        support['declaration'] = {'pairs_listed': listed, 'pairs_expected': expected, 'consistent': listed >= expected}
        if listed < expected:
            units_txt = f'its {n_a:,} units' if symmetric else f'its {n_a:,} x {n_b:,} units'
            support['basis'] += (f' The file does not support this declaration: it lists {listed:,} of the {expected:,} pairs '
                                 f'among {units_txt}.')
            warnings.append(f'You declared that every pair among the release units is evaluated, but the file lists only '
                            f'{listed:,} of the {expected:,} pairs among {units_txt}. The exclusion figures rest on the '
                            f'declaration, not on the file. If the file is only a sample of the evaluated pairs, upload the '
                            f'full list to check the declaration.')
            if cmp_block is not None:
                cmp_block = {**cmp_block, 'applicable': False, 'c_star': None, 'delta_U_break_even': None,
                             'statement': f'No break-even is given: the declared candidate rule (every pair among the '
                                          f'release units is evaluated) is not supported by the file, which lists '
                                          f'{listed:,} of {expected:,} such pairs.'}
                cmp_block.pop('assumptions', None)
    names = uni.units.tolist()
    observed = {(names[h], names[t]) for h, t in uni.edges.tolist()}
    if symmetric:
        observed |= {(t, h) for h, t in observed}
    else:
        observed = {(h[2:], t[2:]) for h, t in observed}
    has_depth = len(uni.edges) > 0
    depth = uni.depth() if has_depth else None
    return build_card(
        benchmark={'name': name, 'files_sha256': hashes, 'rows': len(rows), 'symmetric': symmetric},
        universe={'name': uni.name, 'version': uni.version, 'units': uni.n, 'fingerprint': uni.fingerprint(), **info},
        task='pair_classification', mapping=rep.to_dict(), exclusion=excl, support=support,
        depth=depth_profile(uni, rel.unit_mask, depth) if has_depth else None,
        rules=search_rules(uni, rel.unit_mask, stated=stated, unmapped_release_units=len(rep.unmapped_ids)) if has_depth else None,
        scope=derived_scope(uni, rel.unit_mask, depth, thresholds=range(1, 41), pairs=True) if has_depth else None,
        comparison=cmp_block,
        hygiene={'pairs': pair_release_checks(rows, observed, symmetric, positive_label, label_col is not None)},
        counted=counted, warnings=warnings, inputs=log.to_dict(),
        notes='Universe = the declared units' + (' with their observed pairs (depth = observed partners)' if has_depth else '')
              + '; W_pairs counts unobserved pairs of distinct units' + ('' if symmetric else ' (one from each side)')
              + '. Without a declared candidate rule, W, W_pairs and pair_coverage describe coverage, not exclusion.')


def load_kg_universe(path: str, fmt: str, relations=None, log: Optional[InputLog] = None) -> Universe:
    """A triple universe from a file: fmt 'yago-ttl' (YAGO 3 Turtle theme) or 'tsv' (head, relation, tail)."""
    if fmt == 'yago-ttl':
        return load_yago3_facts(path, relations=relations)
    if fmt == 'tsv':
        return load_triples_tsv(path, relations=relations, log=log)
    raise ValueError(f'unknown universe format {fmt}')


def audit_kg_triples(release_dir: str, universe_path: str, fmt: str, name: str, stated: Optional[Dict] = None,
                     restrict_relations: bool = True, delta_O: Optional[float] = None,
                     thresholds=range(1, 41), metric: Optional[str] = None) -> Dict:
    """A KG benchmark whose entity ids are the universe's own ids (e.g. YAGO3-10 vs YAGO 3.0.2 yagoFacts).

    The universe is the entities carrying at least one universe triple of the release's relation types
    (all relation types if restrict_relations is False). `stated` is the paper's rule as a rule family,
    e.g. {'family': 'both_ends', 'T': 10} for YAGO3-10's 'a minimum of 10 relations'."""
    log = InputLog()
    progress('reading the release')
    trips, hashes = read_kg_release(release_dir)
    rels = {r for s in trips.values() for _, r, _ in s}
    progress('reading the universe triples')
    uni = load_kg_universe(universe_path, fmt, relations=rels if restrict_relations else None, log=log)
    ids = sorted({e for s in trips.values() for h, _, t in s for e in (h, t)})
    progress('mapping release ids')
    rel, rep = map_units(ids, uni, name=name)
    rep, matched = match_triples_exact(trips, uni, rep)
    mapping = rep.to_dict()
    mapping['matched_triples_by_split'] = matched
    warnings = check_mapping(mapping, uni.name, 'release entities') + log.warnings()
    progress('computing exclusion, depth and rules')
    excl = unit_exclusion(uni, rel.unit_mask, by_strata=False)
    ix = uni.index
    as_triples = lambda ts: [(ix[h], r, ix[t]) for h, r, t in ts if h in ix and t in ix]
    excl['per_query'] = query_shares(uni, rel.unit_mask, as_triples(trips['test']),
                                     as_triples([x for s in trips.values() for x in s]), test_triples=len(trips['test']))
    warnings += skipped_test_warning(excl['per_query'], len(trips['test']), delta_O)
    counted = (f"the {uni.n:,} {uni.name} entities that occur in at least one triple"
               + (f" of its {len(rels)} relation types" if restrict_relations else ''))
    depth = uni.depth()
    card = build_card(
        benchmark={'name': name, 'files_sha256': hashes, 'splits': {s: len(t) for s, t in trips.items()},
                   'relations': len(rels)},
        universe={'name': uni.name, 'version': uni.version, 'units': uni.n, 'triples': int(len(uni.edges)),
                  'relations': len(set(uni.edge_types.tolist())), 'fingerprint': uni.fingerprint()},
        task='kg_link_prediction', mapping=mapping, exclusion=excl, support=SUPPORT['kg'],
        depth=depth_profile(uni, rel.unit_mask, depth),
        rules=search_rules(uni, rel.unit_mask, thresholds=thresholds, stated=stated,
                           unmapped_release_units=len(rep.unmapped_ids)),
        scope=derived_scope(uni, rel.unit_mask, depth, thresholds=thresholds),
        comparison=comparison(delta_O, metric, excl['per_query']['W_bar'], 'filtered test queries on the universe',
                              per_query=excl['per_query']),
        hygiene={'reverse_leakage': reverse_leakage(trips['test'], trips['train']),
                 'unseen_test_units': unseen_test_units(trips['test'], trips['train'])},
        counted=counted, warnings=warnings, inputs=log.to_dict(),
        notes='Universe = entities carrying >= 1 universe triple' + (' of the release relation types' if restrict_relations else '')
              + '. Depth = number of universe triples on the entity. Release entities absent from the universe count '
                'against every rule in the Jaccard.')
    return card
