# Scope card: Amazon-Book

> Comparisons on Amazon-Book license claims about the user-item pairs it can evaluate, which leave out 99.97% of the 18,701,842,150,229 unobserved user-item pairs among the 8,026,324 users and 2,330,066 items of Amazon 2014 Books ratings. The closest simple rule (kcore, T = 14) is not reproduced (Jaccard 0.726).

- **Task:** recommendation
- **Universe:** Amazon 2014 Books ratings ratings_Books.csv
- **W is a share of:** the 18,701,842,150,229 unobserved user-item pairs among the 8,026,324 users and 2,330,066 items of Amazon 2014 Books ratings
- **Definitions:** Universe = distinct (user, item) pairs in the raw ratings; candidates = unobserved pairs.
- **Tool:** scopecard 0.0.2 (schema scopecard/0.3)
- **Universe fingerprint:** `sha256:c19b8537a93fd6eac241da55a8db7086374968336aa32c3b60c8ea0da1d4ed9a`
- **Candidates:** every item of the release, for each test user of the release (all-item ranking) (excluded units cannot be evaluated). NGCF and LightGCN rank all release items for each test user, so users and items absent from the release are never evaluated.

## Mapping

144242 of 144242 release ids mapped; 0 unmapped; 0 ids stand for more than one universe unit.

## Exclusion

Units: 10212148 of 10356390 excluded, **W = 0.9861**.
Unobserved pairs outside the retained rectangle (52643 of 8026324 and 91599 of 2330066 units retained): **W_pairs = 0.999742**.
The 2026-10 pilot table counted the release's own positives instead: 0.999742. Pilot-table convention: release positives inside the retained units, universe positives in the denominator. The two counts come from different candidate sets, so this is not a share and is kept only for comparison with the 2026-10 pilot table; W_pairs is the share.
Per query, on the universe under the benchmark's protocol: mean W_q = **0.9607** over 52,639 queries (from 0.9607 to 0.9660, median 0.9607); 0 have no excluded candidate. The release also drops 99.3% of the universe's users, whose queries it never asks.

## Rule diagnosis

| family | best T | Jaccard | verdict |
|---|---|---|---|
| kcore | 14 | 0.726 | not reproduced |
| both_ends | 20 (top of the searched range; a higher T may fit better) | 0.451 | not reproduced |
| threshold | 20 (top of the searched range; a higher T may fit better) | 0.441 | not reproduced |

Stated rule (kcore, T = 10): Jaccard 0.502, not reproduced.

## Derived scope

No depth threshold up to 20 brings W to 0.5 or below (pairs level; W = 0.787642 at d = 20). This is a coverage threshold: it says where the benchmark covers at least half of the restricted universe, not that rankings hold there.

## Input files

- **ratings_Books.csv:** 22,507,155 lines read.
