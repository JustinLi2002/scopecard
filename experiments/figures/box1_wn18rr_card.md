**Box 1. The WN18RR scope card, abridged** (`tool/library/cards/wn18rr.json`)

- **Scope statement:** Comparisons on WN18RR license claims about the entities it can evaluate, which leave out 63.4% of the 112,199 WordNet 3.0 synsets that carry at least one triple of its 11 relations. The closest simple rule (both_ends, T = 5) is not reproduced (Jaccard 0.595).
- **Mapping:** 40,943 of 40,943 ids mapped through 92,998 of 93,003 triples; 161 ids merge synsets of different POS.
- **Exclusion:** W = 0.6336 (71,094 of 112,199 synsets); by POS: a 0.928, n 0.608, r 0.898, v 0.432; per filtered test query on the universe, mean W_q = 0.6337 over 6,266 queries.
- **Rule diagnosis:** stated (both-ends threshold, T = 15): Jaccard 0.149, not reproduced; best found (both-ends threshold, T = 5): 0.595, not reproduced.
- **Derived scope:** d_min = 3.
- **Hygiene:** 1,086 of 3,134 test triples (34.7%) have their reverse in training, among them 1,011 of the 1,074 _derivationally_related_form test triples; 209 test entities never occur in training (210 test triples).
