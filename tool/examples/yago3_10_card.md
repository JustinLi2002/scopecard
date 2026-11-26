# Scope card: YAGO3-10

> Comparisons on YAGO3-10 license claims about the entities it can evaluate, which leave out 95.4% of the 2,635,315 YAGO 3.0.2 entities that occur in at least one triple of its 37 relation types. The closest simple rule (both_ends, T = 10) is approximately reproduced (Jaccard 0.967).

- **Task:** kg_link_prediction
- **Universe:** YAGO 3.0.2 yagoFacts.ttl
- **W is a share of:** the 2,635,315 YAGO 3.0.2 entities that occur in at least one triple of its 37 relation types
- **Definitions:** Universe = entities carrying >= 1 universe triple of the release relation types. Depth = number of universe triples on the entity. Release entities absent from the universe count against every rule in the Jaccard.
- **Tool:** scopecard 0.0.2 (schema scopecard/0.3)
- **Universe fingerprint:** `sha256:8708a2b249492f1eef68d5d165768804efc3861b8c0178b5da78afcd9a648c2c`
- **Candidates:** every entity of the release, in filtered head and tail ranking (excluded units cannot be evaluated). Link prediction ranks each test query against the release entity set, so a universe entity absent from the release is never a candidate.

## Mapping

121138 of 123182 release ids mapped; 2044 unmapped; 0 ids stand for more than one universe unit; 34668 of 1089040 release triples have no universe match.

## Exclusion

Units: 2514177 of 2635315 excluded, **W = 0.9540**.
Per query, on the universe under the benchmark's protocol: mean W_q = **0.9544** over 9,654 queries (from 0.9529 to 0.9647, median 0.9540); 0 have no excluded candidate.

## Rule diagnosis

| family | best T | Jaccard | verdict |
|---|---|---|---|
| both_ends | 10 | 0.967 | approximately reproduced |
| threshold | 10 | 0.950 | approximately reproduced |
| kcore | 7 | 0.534 | not reproduced |
| seeds_neighbours | 24 | 0.053 | not reproduced |

Stated rule (both_ends, T = 10): Jaccard 0.967, approximately reproduced.

## Derived scope

d_min = 7 (smallest depth threshold with W <= 0.5, units level). This is a coverage threshold: it says where the benchmark covers at least half of the restricted universe, not that rankings hold there.

## Hygiene

Reverse-edge leakage: 148 of 5,000 test triples (3.0%) have their reverse (tail, relation, head) in training.
By relation: isConnectedTo 99 of 150 (66%); isLocatedIn 27 of 411 (7%); isMarriedTo 17 of 17 (100%); hasChild 2 of 24 (8%); hasNeighbor 2 of 2 (100%); dealsWith 1 of 7 (14%).
Unseen in training: 18 test entities never occur in a training triple; they touch 18 test triples of 5,000.
