"""Command line: benchscope-audit {kg-wordnet, kg-triples, recsys-checkins, recsys-ratings, query-qrels,
ogbl-citation2, pair-classification} ... --out card.json [--md card.md]"""
import argparse
import sys

from .card import to_json, to_markdown
from .io import InputError
from .pipelines import (audit_kg_triples, audit_kg_wordnet, audit_ogbl_citation2, audit_pair_classification,
                        audit_query_qrels, audit_recsys_checkins, audit_recsys_ratings)


def main(argv=None, catch_errors: bool = True):
    """Run one audit. An InputError (a file without the expected format, or a release that does not match
    the universe) is printed as one line on stderr with exit code 2; catch_errors=False re-raises it
    (the browser page shows it itself)."""
    ap = argparse.ArgumentParser(prog='benchscope-audit', description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    k = sub.add_parser('kg-wordnet', help='WordNet-derived KG benchmark (WN18, WN18RR) vs WordNet 3.0')
    k.add_argument('--release', required=True, help='folder with train.txt, valid.txt, test.txt')
    k.add_argument('--wordnet', required=True, help='WordNet-3.0/dict folder, or the table from export_wordnet30_table')
    k.add_argument('--relations', choices=['wn18', 'wn18rr'], default='wn18rr')
    k.add_argument('--identity', choices=['pos_offset', 'bare_offset'], default='pos_offset')
    k.add_argument('--name', default='WN18RR')
    t = sub.add_parser('kg-triples', help='KG benchmark whose entity ids are the universe ids (e.g. YAGO3-10 vs YAGO 3.0.2)')
    t.add_argument('--release', required=True, help='folder with train.txt, valid.txt, test.txt')
    t.add_argument('--universe', required=True, help='universe triples: YAGO 3 Turtle theme or a head/relation/tail TSV')
    t.add_argument('--format', choices=['yago-ttl', 'tsv'], default='tsv')
    t.add_argument('--name', default='KG benchmark')
    t.add_argument('--stated-family', choices=['threshold', 'both_ends', 'seeds_neighbours', 'kcore'], default=None,
                   help="the paper's rule as a rule family (YAGO3-10: both_ends)")
    t.add_argument('--stated-T', type=int, default=None, help="the stated rule's threshold (YAGO3-10: 10)")
    t.add_argument('--all-relations', action='store_true', help='do not restrict the universe to the release relation types')
    for p in (k, t):
        p.add_argument('--delta-O', type=float, default=None, help='a model difference on the benchmark (first minus second)')
        p.add_argument('--metric', choices=['auc', 'mrr', 'hits'], default=None,
                       help='the metric of --delta-O (required with it); break-even values exist only for AUC')
    r = sub.add_parser('recsys-checkins', help='LightGCN-format release vs raw SNAP check-ins')
    r.add_argument('--release', required=True)
    r.add_argument('--checkins', required=True)
    r.add_argument('--name', default='Gowalla (LightGCN)')
    r.add_argument('--positives', choices=['universe', 'release'], default='universe',
                   help="'release' also reports the 2026-10 pilot table's value (W_pairs_release_convention); "
                        'W_pairs always uses the universe\'s positives')
    ra = sub.add_parser('recsys-ratings', help='LightGCN-format release vs an Amazon 2014 ratings CSV (e.g. Amazon-Book)')
    ra.add_argument('--release', required=True)
    ra.add_argument('--ratings', required=True, help='user,item,rating,timestamp CSV, e.g. ratings_Books.csv')
    ra.add_argument('--name', default='Amazon-Book (LightGCN)')
    ra.add_argument('--universe-name', default='', help='e.g. "Amazon 2014 Books ratings" (default: the file name)')
    ra.add_argument('--positives', choices=['universe', 'release'], default='universe',
                    help="'release' also reports the 2026-10 pilot table's value (W_pairs_release_convention); "
                         'W_pairs always uses the universe\'s positives')
    for p in (r, ra):
        p.add_argument('--k-core', type=int, default=10, help='the stated k-core (NGCF / LightGCN: 10)')
    q = sub.add_parser('query-qrels', help='TREC-style query-level benchmark: released topics vs topics with judgments')
    q.add_argument('--topics', required=True, help='test topic list (.tsv or .tsv.gz, id in the first field)')
    q.add_argument('--qrels', required=True)
    q.add_argument('--corpus-size', type=int, required=True)
    q.add_argument('--name', default='TREC benchmark')
    q.add_argument('--relevant-grade', type=int, default=1, help='lowest relevant grade (TREC DL passage: 2)')
    q.add_argument('--min-relevant', type=int, default=None, help='stated topic rule: minimum relevant judgments')
    q.add_argument('--max-ratio', type=float, default=None, help='stated topic rule: relevant/judged must be below this')
    q.add_argument('--rule-source', default='', help='where the topic rule is stated (recorded on the card)')
    pc = sub.add_parser('pair-classification', help='labelled pairs (protein-protein, drug-target) vs a declared universe of units')
    pc.add_argument('--pairs', required=True, nargs='+', help='release pair files (CSV/TSV, .gz allowed); negatives-only files work too')
    pc.add_argument('--a-col', required=True, help='column of the first unit (name, or 0-based index with --no-header)')
    pc.add_argument('--b-col', required=True, help='column of the second unit')
    pc.add_argument('--label-col', default=None, help='label column (omit for a file of negatives or positives only)')
    pc.add_argument('--positive-label', default='1')
    pc.add_argument('--no-header', action='store_true', help='the pair files have no header row')
    pc.add_argument('--sep', default=None, help="column separator (default: ',' for .csv, tab otherwise)")
    pc.add_argument('--universe-units', default=None, help='declared units, one per line (the first side if --bipartite)')
    pc.add_argument('--universe-units-b', default=None, help='declared units of the second side (--bipartite)')
    pc.add_argument('--universe-pairs', default=None, help='observed pairs in the universe (two columns, no header), for depth')
    pc.add_argument('--bipartite', action='store_true', help='two unit sets (drug-target); default: one set, unordered pairs')
    pc.add_argument('--universe-name', default='')
    pc.add_argument('--name', default='pair benchmark')
    pc.add_argument('--stated-family', choices=['threshold', 'both_ends', 'seeds_neighbours', 'kcore'], default=None)
    pc.add_argument('--stated-T', type=int, default=None)
    pc.add_argument('--delta-O', type=float, default=None, help='a model difference on the benchmark (first minus second)')
    pc.add_argument('--metric', choices=['auc', 'mrr', 'hits'], default=None, help='the metric of --delta-O (required with it)')
    pc.add_argument('--candidate-rule', choices=['unknown', 'all-pairs'], default='unknown',
                    help="how the benchmark draws its pairs: 'unknown' (default; W is then coverage) or 'all-pairs' "
                         "(every pair among its units is evaluated)")
    c = sub.add_parser('ogbl-citation2', help='ogbl-citation2: source papers vs all papers of the source year')
    c.add_argument('--dataset', required=True, help='OGB folder with raw/node_year.csv.gz and split/time/*.pt (no torch needed)')
    c.add_argument('--name', default='ogbl-citation2')
    c.add_argument('--no-negatives', action='store_true', help='skip the checks on sampled negatives')
    for p in (k, t, r, ra, q, c, pc):
        p.add_argument('--out', required=True)
        p.add_argument('--md', default=None)
    a = ap.parse_args(argv)
    if getattr(a, 'delta_O', None) is not None and a.metric is None:
        ap.error('--delta-O needs --metric (auc, mrr or hits)')
    try:
        if getattr(a, 'delta_O', None) is not None and not -1 <= a.delta_O <= 1:     # also rejects nan and inf
            raise InputError('--delta-O must be a finite difference between -1 and 1')  # not ap.error: the page has no exit
        if a.cmd == 'kg-wordnet':
            card = audit_kg_wordnet(a.release, a.wordnet, a.name, a.relations, a.identity, a.delta_O, a.metric)
        elif a.cmd == 'kg-triples':
            if (a.stated_family is None) != (a.stated_T is None):
                ap.error('--stated-family and --stated-T go together')
            stated = {'family': a.stated_family, 'T': a.stated_T} if a.stated_family else None
            card = audit_kg_triples(a.release, a.universe, a.format, a.name, stated=stated,
                                    restrict_relations=not a.all_relations, delta_O=a.delta_O, metric=a.metric)
        elif a.cmd == 'query-qrels':
            card = audit_query_qrels(a.topics, a.qrels, a.corpus_size, a.name, a.relevant_grade, a.min_relevant, a.max_ratio,
                                     rule_source=a.rule_source)
        elif a.cmd == 'ogbl-citation2':
            card = audit_ogbl_citation2(a.dataset, a.name, negatives=not a.no_negatives)
        elif a.cmd == 'pair-classification':
            if (a.stated_family is None) != (a.stated_T is None):
                ap.error('--stated-family and --stated-T go together')
            card = audit_pair_classification(a.pairs, a.a_col, a.b_col, a.name, a.label_col, a.positive_label, not a.no_header,
                                             a.sep, a.universe_units, a.universe_units_b, a.universe_pairs,
                                             symmetric=not a.bipartite,
                                             stated={'family': a.stated_family, 'T': a.stated_T} if a.stated_family else None,
                                             delta_O=a.delta_O, universe_name=a.universe_name, metric=a.metric,
                                             candidate_rule=a.candidate_rule.replace('-', '_'))
        elif a.cmd == 'recsys-ratings':
            card = audit_recsys_ratings(a.release, a.ratings, a.name, a.positives, a.k_core, a.universe_name)
        else:
            card = audit_recsys_checkins(a.release, a.checkins, a.name, a.positives, a.k_core)
    except InputError as e:
        if not catch_errors:
            raise
        print(f'benchscope-audit: input error: {e}', file=sys.stderr)
        return 2
    with open(a.out, 'w', encoding='utf-8', newline='\n') as f:
        f.write(to_json(card))
    if a.md:
        with open(a.md, 'w', encoding='utf-8', newline='\n') as f:
            f.write(to_markdown(card))
    print(card['scope_statement'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
