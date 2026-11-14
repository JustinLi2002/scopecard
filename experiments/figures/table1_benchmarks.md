| Benchmark | Domain | Unit | Regime | Decidability | W | Rule diagnosis | Construction finding (prior work) |
|---|---|---|---|---|---|---|---|
| FB15k-237 | knowledge-graph link prediction | entity | (iii) | partial | not computable | — | entities with triples on the kept relations removed, unstated (new [2]) |
| WN18RR | knowledge-graph link prediction | entity | (i) | partial | 0.6336 | stated: J = 0.149; best both-ends threshold, T = 5: J = 0.595 | stated rule does not reproduce the entity set; ids merge synsets (new / partly known [1]) |
| YAGO3-10 | knowledge-graph link prediction | entity | (i) | partial | 0.9540 | best both-ends threshold, T = 10: J = 0.967 | every description misstates the rule (facts, not relation types) (new [3]) |
| ogbl-collab | graph link prediction | author pair | (iii) | undecidable | not computable | — | author subset rule not stated anywhere checked (—) |
| ogbl-citation2 | graph link prediction | 2019 source paper | (i) | partial | 0.2025 | best threshold, T = 6: J = 1.000 | a source paper needs a minimum of in-subset references, unstated (new [4]) |
| Gowalla | recommendation | user–item pair | (i) | partial | 0.9911 | best k-core, T = 10: J = 0.9999 | 10-core approximately reproduced (—) |
| Yelp2018 | recommendation | user–item pair | (iii) | decidable, unverified | not computable | — | raw data not public; release-internal checks only (—) |
| Amazon-Book | recommendation | user–item pair | (i) | partial | 0.9997 | stated: J = 0.502; best k-core, T = 14: J = 0.726 | release is a strict subset of the stated 10-core; the extra user reduction is unexplained (partly known [5]) |
| TREC DL 2019 passage | information retrieval | topic | (i) | verified | 0.7850 | stated topic rule: 0 violations on 43 judged topics | topic selection by judged-relevant counts and density (known [6]) |
| TREC DL 2020 passage | information retrieval | topic | (i) | verified | 0.7300 | stated topic rule: 0 violations on 54 judged topics | the 2019 selection, reused by reference (known [6]) |

W and rule diagnosis from the tool cards (tool/library/cards); regime from the card (support) or the registry (no card); domain and decidability from pilot_decidability.tsv; prior-work status from PRIOR_WORK.md (section in brackets).
