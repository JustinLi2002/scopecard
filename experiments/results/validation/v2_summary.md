# V2: synthetic benchmarks with known answers

3240 runs (2694 used, 546 skipped because the release kept under 5% or over 95% of the units: kcore 60, seeds_neighbours 486); 2334 rule-generated; 180 random controls, 180 edge_sample controls; 11.1 minutes.

## The answers are independent of the tool

- Releases and the recovery check come from a plain-Python reference implementation of the four families. On all 30 graphs, the reference and the tool select the same units for 2400 of the 2400 family-threshold pairs of the search grid (4 families, thresholds 1-20).

## W is exact

- Unit-level W equals the direct count in 2694 of 2694 runs.
- Pair-level W equals explicit enumeration of every unobserved pair in 864 of 864 small-graph runs (up to 110,601 candidate pairs).

## Recovery of the generating rule

| noise eps | runs | generating rule recovered (same units) | same family and T | verdict reproduced | approximately | not reproduced | mean Jaccard of the generating rule |
|---|---|---|---|---|---|---|---|
| 0 | 389 | 100.0% | 79.4% | 100.0% | 0.0% | 0.0% | 1.0000 |
| 0.005 | 389 | 99.7% | 79.2% | 5.9% | 94.1% | 0.0% | 0.9903 |
| 0.01 | 389 | 100.0% | 79.4% | 2.6% | 97.4% | 0.0% | 0.9807 |
| 0.02 | 389 | 99.5% | 78.9% | 0.3% | 98.2% | 1.5% | 0.9604 |
| 0.05 | 389 | 91.3% | 70.7% | 0.0% | 0.0% | 100.0% | 0.9050 |
| 0.1 | 389 | 81.5% | 62.5% | 0.0% | 0.0% | 100.0% | 0.8195 |

Without noise, by family (recovered = the best rule selects exactly the generating units):

| family | runs | recovered | same family and T |
|---|---|---|---|
| threshold | 120 | 120 | 120 |
| both_ends | 120 | 120 | 41 |
| seeds_neighbours | 39 | 39 | 39 |
| kcore | 110 | 110 | 109 |

## Controls: random releases

| noise eps | runs | verdict reproduced | approximately | not reproduced | max best Jaccard |
|---|---|---|---|---|---|
| 0 | 30 | 0 | 0 | 30 | 0.6688 |
| 0.005 | 30 | 0 | 0 | 30 | 0.6667 |
| 0.01 | 30 | 0 | 0 | 30 | 0.6667 |
| 0.02 | 30 | 0 | 0 | 30 | 0.6688 |
| 0.05 | 30 | 0 | 0 | 30 | 0.6667 |
| 0.1 | 30 | 0 | 0 | 30 | 0.6667 |

## Controls: releases built by a rule outside the four families (random relations, both ends kept)

| noise eps | runs | verdict reproduced | approximately | not reproduced | max best Jaccard |
|---|---|---|---|---|---|
| 0 | 30 | 0 | 0 | 30 | 0.7000 |
| 0.005 | 30 | 0 | 0 | 30 | 0.7163 |
| 0.01 | 30 | 0 | 0 | 30 | 0.7122 |
| 0.02 | 30 | 0 | 0 | 30 | 0.6905 |
| 0.05 | 30 | 0 | 0 | 30 | 0.6840 |
| 0.1 | 30 | 0 | 0 | 30 | 0.6688 |

## End to end (kg-triples on files)

- 12 runs; exit code 0 in 12; the card's W equals the known W (6 decimals) in 12; the best rule selects exactly the released units in 12.
