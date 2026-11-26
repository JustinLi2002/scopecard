# Scope card: WN18RR

> Comparisons on WN18RR license claims about the entities it can evaluate, which leave out 63.4% of the 112,199 WordNet 3.0 synsets that carry at least one triple of its 11 relations. The closest simple rule (both_ends, T = 5) is not reproduced (Jaccard 0.595). Caution: Left out of the per-query shares and the break-even: 1 of the 3,134 test triples (no matching triple in the universe); they use the other 3,133. A model difference measured on all test triples is on a different query set, for which the break-even carries no guarantee; recompute the difference on the 3,133 triples the shares use.

## Warnings

- Left out of the per-query shares and the break-even: 1 of the 3,134 test triples (no matching triple in the universe); they use the other 3,133. A model difference measured on all test triples is on a different query set, for which the break-even carries no guarantee; recompute the difference on the 3,133 triples the shares use.

- **Task:** kg_link_prediction
- **Universe:** WordNet 3.0 pos_offset, 11 relations
- **W is a share of:** the 112,199 WordNet 3.0 synsets that carry at least one triple of its 11 relations
- **Definitions:** Universe = WordNet 3.0 synsets carrying >= 1 triple of the benchmark relations (W); W_all_synsets also counts synsets with none. Depth = number of universe triples on the synset.
- **Tool:** scopecard 0.0.2 (schema scopecard/0.3)
- **Universe fingerprint:** `sha256:0e77d7b548e8f9dfa3db8792cbe0cb6d373395cda3e9e7bf51383f5049806f10`
- **Candidates:** every entity of the release, in filtered head and tail ranking (excluded units cannot be evaluated). Link prediction ranks each test query against the release entity set, so a universe entity absent from the release is never a candidate.

## Mapping

40943 of 40943 release ids mapped; 0 unmapped; 161 ids stand for more than one universe unit; 5 of 93003 release triples have no universe match.

## Exclusion

Units: 71094 of 112199 excluded, **W = 0.6336**.
By pos: a 0.9277, n 0.6085, r 0.8981, v 0.4324.
Per query, on the universe under the benchmark's protocol: mean W_q = **0.6337** over 6,266 queries (from 0.6328 to 0.6361, median 0.6337); 0 have no excluded candidate.

## Rule diagnosis

| family | best T | Jaccard | verdict |
|---|---|---|---|
| both_ends | 5 | 0.595 | not reproduced |
| threshold | 5 | 0.592 | not reproduced |
| kcore | 4 | 0.546 | not reproduced |
| seeds_neighbours | 16 | 0.418 | not reproduced |

Stated rule (both_ends, T = 15): Jaccard 0.149, not reproduced.

## Derived scope

d_min = 3 (smallest depth threshold with W <= 0.5, units level). This is a coverage threshold: it says where the benchmark covers at least half of the restricted universe, not that rankings hold there.

## Hygiene

Reverse-edge leakage: 1,086 of 3,134 test triples (34.7%) have their reverse (tail, relation, head) in training.
By relation: _derivationally_related_form 1,011 of 1,074 (94%); _verb_group 38 of 39 (97%); _also_see 34 of 56 (61%); _similar_to 3 of 3 (100%).
Unseen in training: 209 test entities never occur in a training triple; they touch 210 test triples of 3,134.

## Comparison

The second model leads by 0.0142 AUC on the benchmark. On the declared universe that ordering reverses exactly when the change between the two regions, c (the per-query changes delta_O,q - delta_U,q averaged with weights W_q), has the sign of delta_O and its magnitude exceeds c* = |delta_O| / mean(W_q) = 0.02241. This holds for a delta_O measured on the same 6,266 test queries, each ranked against its kept candidates under the filter above; for a difference from another query set or from the benchmark's own filter it carries no guarantee, and delta_O has to be recomputed on these queries. If the per-query shares do not co-vary with the per-query changes, the ordering reverses when the models' difference on the excluded part is in the opposite direction and larger than 0.008208.
Assumes a fixed positive reference; excluded candidates weighted like the benchmark candidates; delta_O is the mean per-query difference over the same 6,266 test queries, each ranked against its kept candidates under the filter of exclusion.per_query; W = 0.6337 is the mean of the per-query excluded shares W_q (filtered test queries on the universe); c* is then exact for c, the W_q-weighted mean of the per-query changes (Corollary 1, which writes it as delta_O - delta_U + Cov(W_q, c_q) / W); the delta_U threshold also takes that covariance as zero.
