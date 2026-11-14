# Scope card: ogbl-citation2

> Comparisons on ogbl-citation2 license claims about the queries it can evaluate, which leave out 20.3% of the 108,591 ogbl-citation2 papers published in 2019. The closest simple rule (threshold, T = 6) is reproduced (Jaccard 1.000).

- **Task:** query_level
- **Universe:** ogbl-citation2 papers published in 2019 node_year.csv.gz, split time
- **W is a share of:** the 108,591 ogbl-citation2 papers published in 2019
- **Definitions:** Universe = papers of 2019, the source year; release = valid/test source papers. Depth = references inside the subset (train + valid + test edges). The rule search is the threshold family on that depth.
- **Tool:** benchscope-audit 0.0.1 (schema benchscope-card/0.3)
- **Universe fingerprint:** `sha256:3238cf865abe65c01fc46c32a4009261c2d662275373be93f5d0b177395d0134`
- **Candidates:** the valid/test source papers, each ranked against its sampled negatives (excluded units cannot be evaluated). Only source papers are queried, so a paper of the source year that is not a source is never evaluated as a query.

## Mapping

86596 of 86596 release ids mapped; 0 unmapped; 0 ids stand for more than one universe unit.

## Exclusion

Units: 21995 of 108591 excluded, **W = 0.2025**.

## Rule diagnosis

| family | best T | Jaccard | verdict |
|---|---|---|---|
| threshold | 6 | 1.000 | reproduced |

## Derived scope

d_min = 1 (smallest depth threshold with W <= 0.5, units level). This is a coverage threshold: it says where the benchmark covers at least half of the restricted universe, not that rankings hold there.

## Sampled negatives

- **valid:** 1000 per source for 86596 sources. Later than the source: 0; same year: 3211832; cited by the source: 0. Share in the lower-degree half of all nodes: 0.500 (uniform sampling gives 0.5).
- **test:** 1000 per source for 86596 sources. Later than the source: 0; same year: 3212415; cited by the source: 0. Share in the lower-degree half of all nodes: 0.500 (uniform sampling gives 0.5).
