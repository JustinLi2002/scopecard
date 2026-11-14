# Scope card: Gowalla

> Comparisons on Gowalla license claims about the user-item pairs it can evaluate, which leave out 99.1% of the 137,177,550,814 unobserved user-item pairs among the 107,092 users and 1,280,969 locations of SNAP Gowalla check-ins. The closest simple rule (kcore, T = 10) is approximately reproduced (Jaccard 0.9999).

- **Task:** recommendation
- **Universe:** SNAP Gowalla check-ins loc-gowalla_totalCheckins.txt.gz
- **W is a share of:** the 137,177,550,814 unobserved user-item pairs among the 107,092 users and 1,280,969 locations of SNAP Gowalla check-ins
- **Definitions:** Universe = distinct (user, location) pairs in the raw check-ins; candidates = unobserved pairs.
- **Tool:** benchscope-audit 0.0.1 (schema benchscope-card/0.3)
- **Universe fingerprint:** `sha256:87c5b8d253be6f28cbb9be2b67f2e75e6e39415ee7d3529393afc0b7bac80709`
- **Candidates:** every item of the release, for each test user of the release (all-item ranking) (excluded units cannot be evaluated). NGCF and LightGCN rank all release items for each test user, so users and items absent from the release are never evaluated.

## Mapping

70839 of 70839 release ids mapped; 0 unmapped; 0 ids stand for more than one universe unit.

## Exclusion

Units: 1317222 of 1388061 excluded, **W = 0.9490**.
Unobserved pairs outside the retained rectangle (29858 of 107092 and 40981 of 1280969 units retained): **W_pairs = 0.991088**.
The 2026-10 pilot table counted the release's own positives instead: 0.991088. Pilot-table convention: release positives inside the retained units, universe positives in the denominator. The two counts come from different candidate sets, so this is not a share and is kept only for comparison with the 2026-10 pilot table; W_pairs is the share.
Per query, on the universe under the benchmark's protocol: mean W_q = **0.9680** over 29,858 queries (from 0.9680 to 0.9688, median 0.9680); 0 have no excluded candidate. The release also drops 72.1% of the universe's users, whose queries it never asks.

## Rule diagnosis

| family | best T | Jaccard | verdict |
|---|---|---|---|
| kcore | 10 | 0.9999 | approximately reproduced |
| both_ends | 12 | 0.610 | not reproduced |
| threshold | 12 | 0.590 | not reproduced |

Stated rule (kcore, T = 10): Jaccard 0.9999, approximately reproduced.

## Derived scope

d_min = 17 (smallest depth threshold with W <= 0.5, pairs level). This is a coverage threshold: it says where the benchmark covers at least half of the restricted universe, not that rankings hold there.

## Input files

- **loc-gowalla_totalCheckins.txt.gz:** 6,442,892 lines read.
