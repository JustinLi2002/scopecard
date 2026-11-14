"""Write experiments/pilot_decidability.tsv (one row per selected benchmark, coded 2026-10-18).

Same columns as [P3]'s pilot table plus `evaluation_protocol`. Every number in
`verification` and `W_pilot` is checked against the row's result file(s) before
writing: the script refuses to write a number that does not occur there.

  wn18_scope.py          -> results/kg/wn18_scope.txt
  fb15k237_scope.py      -> results/kg/fb15k237_scope.txt
  yago3_10_scope.py      -> results/kg/yago3_10_scope.txt
  ogbl_collab_scope.py   -> results/graph/ogbl_collab_scope.txt
  ogbl_citation2_scope.py-> results/graph/ogbl_citation2_scope.txt
  recsys_kcore_scope.py  -> results/recsys/{gowalla,yelp2018,amazon-book}_scope.txt
  trecdl_scope.py        -> results/ir/trecdl{2019,2020}_passage_scope.txt

Usage: python -I build_pilot_tsv.py <experiments dir>
"""
import csv
import os
import re
import sys

COLS = [
    "benchmark", "domain", "sources_checked", "negative_source",
    "rule_stated_in_paper", "rule_found_in_code_or_release", "text_sufficient",
    "entity", "candidate", "depth", "threshold_T", "declared_universe",
    "decidability", "verification", "support_at_evaluated_unit",
    "W_pilot", "W_target_used", "needed_for_full_W", "evaluation_protocol", "notes",
]

ROWS = [
    {
        "benchmark": "FB15k-237 (Toutanova & Chen 2015)",
        "domain": "knowledge-graph link prediction",
        "result_files": ["results/kg/fb15k237_scope.txt"],
        "sources_checked": "Bordes et al. 2013 NeurIPS (TransE, Sec. 4.1); Toutanova & Chen 2015 CVSC (Sec. 4.3); FB15k release (everest.hds.utc.fr fb15k.tgz via Wayback, README); FB15K-237.2.zip (Microsoft download, README); ConvE repo FB15k-237.tar.gz",
        "negative_source": "implicit: every corrupted triple over the entity set, filtered of known triples",
        "rule_stated_in_paper": "FB15k: Freebase entities present in Wikilinks with >= 100 Freebase mentions (entities and relations); FB15k-237: the 401 most frequent FB15k relations, near-duplicate/inverse relations (>= 97% pair overlap) reduced to one each (237), and valid/test triples dropped if their entity pair is directly linked in training",
        "rule_found_in_code_or_release": "relation rule consistent: 401st relation on FB15k train has 100 triples (3 tied), every one of the 164 dropped relations has a kept partner with >= 97% overlap, but 28 kept pairs also overlap >= 97% of the smaller set (3 of the larger); pair rule exact: 0 valid/test triples linked in train, the rule removes 0.372 / 0.375 of FB15k valid/test triples on the 237 relations; NOT stated: 71 entities that still carry triples on the 237 relations were removed with all 4555 of their training triples",
        "text_sufficient": "no: the 97% denominator, which relation of a group is kept, and the entity removal are not stated",
        "entity": "Freebase entity (head or tail slot of a query)",
        "candidate": "(query, candidate entity)",
        "depth": "Freebase mention count (FB15k seed rule); training link between the pair (test rule)",
        "threshold_T": "100 mentions + Wikilinks (entity, inherited); pair must have 0 training links (test positives)",
        "declared_universe": "none declared (Freebase)",
        "decidability": "partial",
        "verification": "relation and test-pair rules checked against FB15k and the release (all 310116 FB15k-237 triples are FB15k triples); entity seed rule not checkable without a 2013 Freebase snapshot and Wikilinks",
        "support_at_evaluated_unit": "violated at entity level (inherited mention filter); the test-pair rule removes positives, not negatives",
        "W_pilot": "not computed (entity universe unbounded without a declared convention)",
        "W_target_used": "",
        "needed_for_full_W": "declared entity universe (e.g. Freebase topics with a Wikipedia page) and Freebase mention counts",
        "evaluation_protocol": "filtered ranking of head and tail against all 14541 entities; MRR, Hits@k",
        "notes": "the ConvE copy is triple-for-triple identical to the Microsoft release; the undocumented entity removal (one entity with 2867 triples) is the same class of silent change as [P3]'s MolTrans input blanking",
    },
    {
        "benchmark": "WN18RR (Dettmers et al. 2018, from WN18)",
        "domain": "knowledge-graph link prediction",
        "result_files": ["results/kg/wn18_scope.txt"],
        "sources_checked": "Bordes et al. 2014 Mach Learn 94:233 (Sec. 5.1, Table 2); Bordes et al. 2012 AISTATS; TransE 2013 (Sec. 4.1); Dettmers et al. 2018 AAAI (ConvE); wordnet-mlj12 release (README: 'extracted from WordNet 3.0'); ConvE repo WN18RR.tar.gz; Princeton WordNet-3.0.tar.gz",
        "negative_source": "implicit: every corrupted triple over the entity set, filtered of known triples",
        "rule_stated_in_paper": "'we filtered out the synsets appearing in less that 15 triplets' over the 18 relation types, giving 40,943 synsets (Mach Learn 2014); WN18RR keeps WN18's entities and drops inverse relations (11 relations)",
        "rule_found_in_code_or_release": "WN18 is the induced subgraph of WordNet 3.0's 18-relation graph on its entity set (151437 of 151442 triples rebuilt; 58 induced triples absent), so the entity set is the rule; but 34393 of the 40943 synsets have < 15 triples in that graph and 38117 in the release itself; no degree threshold reproduces the set (best single-pass Jaccard 0.597, 15-core empty); entity ids are bare offsets, which WordNet reuses across POS: 161 WN18 entities merge triples of two different synsets",
        "text_sufficient": "no: the stated threshold does not reproduce the release; the counting source is not stated",
        "entity": "WordNet synset",
        "candidate": "(query, candidate synset)",
        "depth": "stated: triplet count (source unknown)",
        "threshold_T": "15 (stated, not reproducible)",
        "declared_universe": "WordNet 3.0 (named in the release README)",
        "decidability": "partial",
        "verification": "the rule cannot be reimplemented, but the release fixes the entity set exactly and WordNet 3.0 holds every excluded synset and triple (92998 of 93003 WN18RR triples rebuilt)",
        "support_at_evaluated_unit": "violated: excluded share by POS n 0.6092, v 0.4397, a 0.9391, r 0.9914",
        "W_pilot": "entity share excluded 0.6344 (synsets with >= 1 triple of the 11 relations) / 0.6512 (all WordNet 3.0 offsets); 125326 of 218367 rebuilt triples touch an excluded synset",
        "W_target_used": "WordNet 3.0 synsets, keyed by bare offset as the release keys them; uniform over candidate entities for a fixed query",
        "needed_for_full_W": "",
        "evaluation_protocol": "filtered ranking of head and tail against all 40943 entities; MRR, Hits@k; 210 of 3134 test triples have an entity absent from train",
        "notes": "excluded region fully observable and finite: candidate for the measured c (PLAN.md Sect. 4); check whether the offset collision is documented",
    },
    {
        "benchmark": "YAGO3-10 (Dettmers et al. 2018)",
        "domain": "knowledge-graph link prediction",
        "result_files": ["results/kg/yago3_10_scope.txt"],
        "sources_checked": "Dettmers et al. 2018 AAAI (ConvE, datasets section); Rossi et al. 2021 TKDD (Sec. 4.1); ConvE repo YAGO3-10.tar.gz; YAGO 3.0.2 yagoFacts.ttl (yago-knowledge.org/data/yago3/tmp)",
        "negative_source": "implicit: every corrupted triple over the entity set, filtered of known triples",
        "rule_stated_in_paper": "'a subset of YAGO3 which consists of entities which have a minimum of 10 relations each' (YAGO3 version not stated); Rossi et al. read it as >= 10 different relations plus all facts of their neighbours",
        "rule_found_in_code_or_release": "entities with >= 10 yagoFacts facts, keeping only facts whose BOTH ends qualify (one pass): 123189 entities (Jaccard 0.967 with the release), 1089132 triples; the release equals the 3.0.2 facts induced on its entity set (0 induced facts missing); the 'distinct relations' reading gives 546 seeds (Jaccard 0.004) and the 'neighbours' reading 2338580 entities (Jaccard 0.052)",
        "text_sufficient": "no: 'relations' means facts, not relation types, and neighbours are not included; version unstated",
        "entity": "YAGO entity",
        "candidate": "(query, candidate entity)",
        "depth": "number of yagoFacts facts on the entity",
        "threshold_T": "10, both ends of every fact",
        "declared_universe": "YAGO3 (version unstated; 3.0.2 used here)",
        "decidability": "partial",
        "verification": "rule reimplemented on YAGO 3.0.2 (Jaccard 0.967); not exact: 1054372 of 1089040 release triples are 3.0.2 facts and 2046 release entities have < 10 facts in 3.0.2, plausibly a version difference that the unstated version prevents checking",
        "support_at_evaluated_unit": "violated at entity level",
        "W_pilot": "entity share excluded 0.9540 (YAGO 3.0.2 entities with >= 1 fact on the 37 relations); 4573794 of 5628166 facts touch an excluded entity",
        "W_target_used": "YAGO 3.0.2 entities with >= 1 entity-entity fact; uniform over candidate entities for a fixed query",
        "needed_for_full_W": "the exact YAGO3 version used",
        "evaluation_protocol": "filtered ranking of head and tail against all 123182 entities; MRR, Hits@k",
        "notes": "inside the release 52437 entities have < 10 triples: the threshold applies in the source KG, not in the benchmark graph",
    },
    {
        "benchmark": "ogbl-collab (OGB, Hu et al. 2020)",
        "domain": "graph link prediction",
        "result_files": ["results/graph/ogbl_collab_scope.txt"],
        "sources_checked": "Hu et al. 2020 NeurIPS (OGB, Sec. 5.2); ogb.stanford.edu linkprop page; collab.zip v1 (raw, split/time, mapping)",
        "negative_source": "sampled: 100,000 random node pairs per split (Hits@50)",
        "rule_stated_in_paper": "'a subset of the collaboration network between authors indexed by MAG'; train <= 2017, valid 2018, test 2019",
        "rule_found_in_code_or_release": "every one of the 235868 authors has a training (<= 2017) collaboration, and 0 valid/test positives touch an author without one; negatives are uniform over node pairs (lower-degree half share 0.502); 0.366 of test positives repeat a training pair",
        "text_sufficient": "no: the author subset rule is not stated anywhere checked",
        "entity": "author",
        "candidate": "(author, author) pair",
        "depth": "pre-2018 collaborations",
        "threshold_T": "1 on both authors (found); subset rule unknown",
        "declared_universe": "none declared (MAG authors)",
        "decidability": "undecidable",
        "verification": "release only: the T = 1 restriction is a property of the release; whether it is the subset rule or a consequence of it cannot be told",
        "support_at_evaluated_unit": "violated for authors whose first collaboration is in 2018-2019",
        "W_pilot": "not computed",
        "W_target_used": "",
        "needed_for_full_W": "MAG / OpenAlex author records (the release maps nodes to MAG author ids)",
        "evaluation_protocol": "Hits@50 against 100,000 sampled negative pairs",
        "notes": "",
    },
    {
        "benchmark": "ogbl-citation2 (OGB, Hu et al. 2020)",
        "domain": "graph link prediction",
        "result_files": ["results/graph/ogbl_citation2_scope.txt"],
        "sources_checked": "Hu et al. 2020 NeurIPS (OGB, Sec. 5.4 and footnote 17); ogb.stanford.edu linkprop page; citation-v2.zip v1 (raw, split/time; CRC-checked)",
        "negative_source": "sampled: 1,000 earlier papers per source not cited by it (MRR)",
        "rule_stated_in_paper": "'a subset of papers extracted from MAG'; source papers are 2019 papers, two references of each dropped for validation and test; negatives 'randomly-sampled from all the previous papers that are not referenced by the source paper'",
        "rule_found_in_code_or_release": "a 2019 paper is a source exactly when it has >= 6 in-subset references: all 86596 papers with >= 6 are sources and 0 of the 21995 non-sources have 6 or more; negatives are never later than the source (same year allowed), uniform over nodes (lower-degree half share 0.500)",
        "text_sufficient": "no: the six-reference requirement and the paper subset rule are not stated",
        "entity": "source paper (query); candidate paper",
        "candidate": "(source paper, earlier paper)",
        "depth": "in-subset references of the source paper",
        "threshold_T": "6 references on the source (found); paper subset rule unknown",
        "declared_universe": "none declared (MAG papers); query side: the 2019 papers in the subset",
        "decidability": "partial",
        "verification": "source rule verified exactly on the release; the paper subset rule needs MAG / OpenAlex",
        "support_at_evaluated_unit": "violated at query level: 2019 papers with fewer than six in-subset references are never evaluated",
        "W_pilot": "query-level excluded share 0.2025 (21995 of 108591 2019 papers); candidate side not computed",
        "W_target_used": "2019 papers in the subset, uniform (query side only)",
        "needed_for_full_W": "MAG / OpenAlex paper records for the candidate side (the release maps nodes to MAG paper ids)",
        "evaluation_protocol": "MRR of the held-out reference against 1,000 sampled negatives",
        "notes": "the older ogbl-citation was deprecated for a bug in its validation/test negatives",
    },
    {
        "benchmark": "Gowalla (NGCF / LightGCN 10-core)",
        "domain": "recommendation",
        "result_files": ["results/recsys/gowalla_scope.txt"],
        "sources_checked": "Wang et al. 2019 SIGIR (NGCF, Sec. 4.1.1); He et al. 2020 SIGIR (LightGCN, Sec. 4); kuandeng/LightGCN Data/gowalla (train, test, user_list, item_list); SNAP loc-gowalla_totalCheckins",
        "negative_source": "implicit: all items the user has not interacted with in train",
        "rule_stated_in_paper": "'we use the 10-core setting, i.e., retaining users and items with at least ten interactions'",
        "rule_found_in_code_or_release": "iterated 10-core on distinct (user, location) check-in pairs reproduces the users exactly (29858) and items to within 7 (40988 vs 40981; pairs 1027464 vs 1027370); one released user has 9 interactions",
        "text_sufficient": "yes, up to the definition of an interaction (distinct user-location pair)",
        "entity": "user; item",
        "candidate": "(user, item) pair",
        "depth": "interaction count (core number)",
        "threshold_T": "10 on users and items",
        "declared_universe": "SNAP Gowalla check-ins (raw source)",
        "decidability": "partial",
        "verification": "Jaccard users 1.0000, items 0.9998, pairs 0.9999 against the raw 10-core; every release interaction is a raw pair; not exact: 7 core items absent and one released user below the threshold, cause not established (agrees with the independent audit in RECOMMENDATION_AUDIT.md)",
        "support_at_evaluated_unit": "violated: retained users 0.2788, items 0.0320",
        "W_pilot": "0.991088 (unobserved user-item pairs)",
        "W_target_used": "raw users x raw items minus observed pairs, uniform ([P2]'s recommendation convention)",
        "needed_for_full_W": "",
        "evaluation_protocol": "full ranking over all items not in the user's training set; Recall@20, NDCG@20",
        "notes": "[P2] reported Amazon 2023 five-cores only; this is a different source and threshold",
    },
    {
        "benchmark": "Yelp2018 (NGCF / LightGCN 10-core)",
        "domain": "recommendation",
        "result_files": ["results/recsys/yelp2018_scope.txt"],
        "sources_checked": "NGCF Sec. 4.1.1 and footnote 2; LightGCN Sec. 4; kuandeng/LightGCN Data/yelp2018",
        "negative_source": "implicit: all items the user has not interacted with in train",
        "rule_stated_in_paper": "2018 Yelp challenge, same 10-core setting; the revised version also removes cold-start items from the test set",
        "rule_found_in_code_or_release": "the release is a 10-core internally (min user degree 18, min item degree 10); 0 test interactions on items or users absent from train",
        "text_sufficient": "yes for the rule; the raw source is no longer distributed",
        "entity": "user; item",
        "candidate": "(user, item) pair",
        "depth": "interaction count",
        "threshold_T": "10 on users and items; test items must occur in train",
        "declared_universe": "Yelp 2018 challenge data (not available)",
        "decidability": "decidable, unverified",
        "verification": "release-internal checks only; the 2018 challenge file is not public and the current Yelp Open Dataset requires accepting its terms (not done)",
        "support_at_evaluated_unit": "violated (by construction of a 10-core)",
        "W_pilot": "not computed",
        "W_target_used": "",
        "needed_for_full_W": "the 2018 Yelp challenge review file",
        "evaluation_protocol": "full ranking over all items not in the user's training set; Recall@20, NDCG@20",
        "notes": "",
    },
    {
        "benchmark": "Amazon-Book (NGCF / LightGCN 10-core)",
        "domain": "recommendation",
        "result_files": ["results/recsys/amazon-book_scope.txt", "results/recsys/amazon-book_variants.txt"],
        "sources_checked": "NGCF Sec. 4.1.1 (cites He & McAuley 2016); LightGCN Sec. 4; kuandeng/LightGCN Data/amazon-book; Amazon 2014 ratings_Books.csv and reviews_Books_5.json.gz (mcauleylab.ucsd.edu)",
        "negative_source": "implicit: all items the user has not interacted with in train",
        "rule_stated_in_paper": "'we use the 10-core setting to ensure that each user and item have at least ten interactions'",
        "rule_found_in_code_or_release": "every release interaction is a 2014 Books rating (all 2984108), ratings 1-5 and years 1996-2014 all present, and the release is a 10-core internally; but the raw 10-core has 158650 users and 128939 items, of which the release keeps 52643 and 91599 (all inside the raw 10-core); rating thresholds, time windows and the reviews-with-text subset do not reproduce it",
        "text_sufficient": "no: an additional reduction to about one third of the 10-core users is not stated",
        "entity": "user; item",
        "candidate": "(user, item) pair",
        "depth": "interaction count",
        "threshold_T": "10 on users and items, plus an unstated user reduction",
        "declared_universe": "Amazon 2014 Books ratings (raw source)",
        "decidability": "partial",
        "verification": "release checked against the raw 10-core (Jaccard users 0.3318, items 0.7104, pairs 0.6347; release inside the core); the extra reduction is not identified",
        "support_at_evaluated_unit": "violated: retained users 0.0066, items 0.0393",
        "W_pilot": "0.999742 (unobserved user-item pairs)",
        "W_target_used": "raw users x raw items minus observed pairs, uniform ([P2]'s recommendation convention)",
        "needed_for_full_W": "",
        "evaluation_protocol": "full ranking over all items not in the user's training set; Recall@20, NDCG@20",
        "notes": "different release and threshold from [P2]'s Amazon 2023 five-cores; no [P2] number re-used",
    },
    {
        "benchmark": "TREC DL 2019 passage (Craswell et al. 2020)",
        "domain": "information retrieval",
        "result_files": ["results/ir/trecdl2019_passage_scope.txt"],
        "sources_checked": "Craswell et al. 2020 overview (arXiv 2003.07820, judging section and Table 5); 2019qrels-pass.txt (trec.nist.gov); msmarco-test2019-queries.tsv.gz",
        "negative_source": "measured (judged grade 0-1) plus implicit (unjudged passages scored non-relevant by trec_eval)",
        "rule_stated_in_paper": "52 of 200 topics judged: median MRR of submitted document runs under sparse MARCO labels in (0, 0.5]; evaluation set keeps topics with >= 3 relevant and relevant/judged < 0.6; judged pool = top-10 of all runs + MARCO-judged passages + HiCAL selections",
        "rule_found_in_code_or_release": "43 topics in qrels; 0 with < 3 relevant (min exactly 3) and 0 with ratio >= 0.6 (max 0.534)",
        "text_sufficient": "yes",
        "entity": "topic (query)",
        "candidate": "(topic, passage)",
        "depth": "relevant passages found (grade >= 2)",
        "threshold_T": "3 relevant, plus a difficulty window and relevant/judged < 0.6",
        "declared_universe": "the 200 test topics",
        "decidability": "verified",
        "verification": "evaluation-set rule checked on the qrels; the median-MRR pre-selection needs the submitted runs (not checked)",
        "support_at_evaluated_unit": "violated at topic level; at passage level unjudged passages are scored, not excluded (contamination, not positivity)",
        "W_pilot": "topic-level excluded share 0.785 (157 of 200)",
        "W_target_used": "the 200 released test topics, uniform",
        "needed_for_full_W": "",
        "evaluation_protocol": "NDCG@10 (main), AP, RR over 8.8M passages; unjudged = non-relevant",
        "notes": "the topic rule has [P2]'s form exactly: entities (topics) with fewer than T = 3 positives are excluded",
    },
    {
        "benchmark": "TREC DL 2020 passage (Craswell et al. 2021)",
        "domain": "information retrieval",
        "result_files": ["results/ir/trecdl2020_passage_scope.txt"],
        "sources_checked": "Craswell et al. 2021 overview (arXiv 2102.07662, Sec. 2); Craswell, Mitra, Yilmaz, Campos, Voorhees & Soboroff 2021, SIGIR 2369-2375, doi:10.1145/3404835.3463249 (arXiv 2104.09399, Sec. 3.2 'Test Sets'); Craswell et al. 2020 overview for the 2019 thresholds; Voorhees, Craswell & Lin 2022, SIGIR, doi:10.1145/3477495.3531728; 2020qrels-pass.txt (trec.nist.gov); msmarco-test2020-queries.tsv.gz",
        "negative_source": "measured (judged grade 0-1) plus implicit (unjudged passages scored non-relevant by trec_eval)",
        "rule_stated_in_paper": "the 2020 overview says only that NIST chose a subset of the queries for judging (54 passage topics); the companion SIGIR paper (Sec. 3.2) states 'The same judgment process was used in 2020' as in 2019: candidates chosen by the median MRR of document runs (excluded if > 0.5 or 0.0), and candidates dropped when their number of relevant documents fell outside the range; the thresholds are the 2019 overview's (>= 3 relevant, relevant/judged < 0.6); Voorhees et al. 2022 describe the density cut as 'more than one half'",
        "rule_found_in_code_or_release": "54 topics in qrels; consistent with the stated rule: 0 with < 3 relevant (min exactly 3), 0 with ratio >= 0.6 (max 0.350)",
        "text_sufficient": "yes, by reference: the companion SIGIR paper states that the 2019 judgment process was reused, and the 2019 overview gives its thresholds",
        "entity": "topic (query)",
        "candidate": "(topic, passage)",
        "depth": "relevant passages found (grade >= 2)",
        "threshold_T": "3 relevant, plus the median-MRR window and relevant/judged < 0.6 (the 2019 process, reused in 2020)",
        "declared_universe": "the 200 test topics",
        "decidability": "verified",
        "verification": "evaluation-set rule checked on the qrels (max relevant/judged 0.350, so the 'more than one half' density wording also holds); the median-MRR pre-selection needs the submitted runs (not checked)",
        "support_at_evaluated_unit": "violated at topic level",
        "W_pilot": "topic-level excluded share 0.730 (146 of 200)",
        "W_target_used": "the 200 released test topics, uniform",
        "needed_for_full_W": "",
        "evaluation_protocol": "NDCG@10 (main), AP, RR over 8.8M passages; unjudged = non-relevant",
        "notes": "recoded 2026-10-19 (prior-work check): first coded text_sufficient = no and decidability partial because the 2020 overview does not state the rule; the companion SIGIR paper states that the 2019 process was reused",
    },
]

NUM = re.compile(r'(?<![\w.])\d[\d,]*(?:\.\d+)?')


def check(row, base):
    text = ''
    for rf in row['result_files']:
        with open(os.path.join(base, rf), encoding='utf-8') as f:
            text += f.read().replace(',', '')
    missing = []
    for col in ('verification', 'W_pilot', 'rule_found_in_code_or_release'):
        for n in NUM.findall(row[col]):
            n = n.replace(',', '')
            if len(n.replace('.', '')) < 3 or re.fullmatch(r'(19|20)\d\d', n):
                continue          # thresholds, small counts and years stated in the text
            if n not in text:
                missing.append((col, n))
    return missing


def main(base):
    bad = {r['benchmark']: check(r, base) for r in ROWS}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        for k, v in bad.items():
            print('NOT IN RESULT FILE:', k, v)
        sys.exit(1)
    out = os.path.join(base, 'pilot_decidability.tsv')
    with open(out, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, COLS, delimiter='\t', extrasaction='ignore')
        w.writeheader()
        w.writerows(ROWS)
    print('wrote', out, len(ROWS), 'rows')


if __name__ == '__main__':
    main(sys.argv[1])
