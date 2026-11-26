# Separated selection and scoring: cross-prompt results

Exploratory. Fixed archived runs; 124 exact-name common models, 1,140 tasks. Source responses alone select; target responses never vote. Each direction has 100 panels per size.

All score differences and intervals below are in percentage points. Curves use approximate simultaneous 95% intervals across 20 target/excess contrasts; source is the mechanically nonnegative same-response control.

| Selection → scoring | Panel size | Source | Target [joint 95%] | Outside | Excess [joint 95%] |
|---|---:|---:|---|---:|---|
| complete_to_instruct | 4 | 7.778 | 2.435 [2.099, 2.772] | 0.002 | 2.433 [2.102, 2.764] |
| complete_to_instruct | 8 | 4.046 | 1.244 [1.053, 1.435] | 0.020 | 1.225 [1.036, 1.414] |
| complete_to_instruct | 16 | 2.128 | 0.676 [0.552, 0.800] | 0.001 | 0.675 [0.553, 0.798] |
| complete_to_instruct | 32 | 1.108 | 0.348 [0.271, 0.426] | -0.006 | 0.354 [0.277, 0.432] |
| complete_to_instruct | 64 | 0.583 | 0.183 [0.132, 0.233] | 0.005 | 0.178 [0.128, 0.227] |
| instruct_to_complete | 4 | 5.991 | 2.026 [1.746, 2.306] | 0.034 | 1.991 [1.719, 2.264] |
| instruct_to_complete | 8 | 3.138 | 1.038 [0.882, 1.194] | -0.025 | 1.063 [0.906, 1.220] |
| instruct_to_complete | 16 | 1.619 | 0.536 [0.439, 0.633] | -0.015 | 0.551 [0.453, 0.648] |
| instruct_to_complete | 32 | 0.836 | 0.282 [0.222, 0.342] | 0.002 | 0.280 [0.220, 0.339] |
| instruct_to_complete | 64 | 0.447 | 0.149 [0.106, 0.191] | 0.000 | 0.149 [0.106, 0.191] |

Primary Complete → Instruct, p=8, excess: 1.225 pp, pointwise percentile 95% [1.084, 1.362]. Decision: positive_cross_prompt_excess.
Lower bound supports at least 1 pp: True.

Outside is measured on the same H_self vs H_reference contrast. Positive excess does not identify a correct target or prove historical benchmark unfairness. Cross-prompt outcomes share models and tasks; this is not a same-protocol independent replicate.

Ranking columns in summary.json are descriptive repeated-comparison events only, not jointly resolved pair reversals.

Synthetic checks and 720 real explicit-resampling comparisons passed. Protected changes during run: [].
