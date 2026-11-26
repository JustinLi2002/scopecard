# Panel dependence of a solve-rate difficulty filter: results

Design: `experiments/PANEL_FILTER_DESIGN.md` (fixed 2026-11-15). Hard subset = the k tasks solved by fewest builders. 30 random builder sets per condition, 2000 label shuffles, 300 task-bootstrap replicates for the two bootstrapped analyses. Scores are proportions; effects are given in points (1 point = 0.01). In each draw the "self" and "reference" conditions share n - 1 removed models and differ in the last one (the model itself against one more unrelated model).

Correction of 2026-11-16 (design, Section 13). The readings "ranking consequence" below are withdrawn: they rest on a bootstrap interval of the share of sign changes, and that bootstrap is biased. The shares stay as point estimates. Under the rule "solve rate below 50%" the same outcomes vote and score, so own participation (T_self) cannot be negative in any comparison; its interval says nothing about its sign, only about its size.

Names that matched no family rule: none.

## Primary analysis

### Complete, model-line families, k = 148 (primary)
- 153 models in 19 families (sizes 23, 19, 17, 11, 10, 9, 8, 8, ...; 2 singletons); standard hard subset 148 tasks; mean standard score 0.0023.
- Own participation, T_self: +0.197 [+0.125, +0.270] points (bootstrap 95% interval).
- Family dependence beyond own participation, T_family: +0.040 [-0.001, +0.102] points over all models; +0.041 [-0.001, +0.103] over models whose family has at least two members.
- Total, T_total: +0.237 [+0.148, +0.334] points.
- Permutation test of family structure (2000 shuffles): observed mean of (own family left out minus standard) +0.230 points, mean under shuffled families +0.192; p = 0.0195.
- Spearman over models between (s_fam - s_ref) and family size: -0.050 [-0.158, +0.106].
- Cross-family pairs with a non-zero standard difference: 4734; share whose sign changes when both families are left out 3.13%, when the same number of unrelated models is left out 0.88%; excess +2.246 [+0.694, +4.655] percentage points. Spearman between the shift of the pair difference and the difference of family sizes: +0.058 [-0.118, +0.158].
- **Reading fixed in the design:** own participation detectable but small; family dependence not detected; total detectable but small; ranking consequence material.
- By family (members; total, own participation, family dependence, in points): qwen (23; +0.24, +0.23, +0.01); llama (19; +0.14, +0.16, -0.01); deepseek-coder (17; +0.37, +0.11, +0.27); codellama (11; +0.18, +0.19, -0.01); mistral (10; +0.00, +0.01, -0.01); granite (9; +0.21, +0.21, -0.01); gpt (8; +0.90, +0.91, -0.00); qwen-coder (8; +0.08, +0.06, +0.03); yi (8; +0.16, +0.19, -0.03); deepseek-v2plus (7; +0.55, +0.58, -0.03); claude (6; +0.15, +0.21, -0.05); gemini (6; +0.56, +0.15, +0.41); phi (6; +0.03, +0.03, +0.00); gemma (5; +0.05, +0.05, -0.00); starcoder2 (4; +0.00, +0.01, -0.01).

## Replication on Instruct (same tasks, 124 models in common)

### Instruct, model-line families, k = 148
- 126 models in 19 families (sizes 23, 16, 14, 8, 8, 7, 7, 6, ...; 3 singletons); standard hard subset 148 tasks; mean standard score 0.0000.
- Own participation, T_self: +0.156 [+0.102, +0.216] points (bootstrap 95% interval).
- Family dependence beyond own participation, T_family: +0.075 [+0.008, +0.146] points over all models; +0.076 [+0.008, +0.150] over models whose family has at least two members.
- Total, T_total: +0.231 [+0.137, +0.316] points.
- Permutation test of family structure (2000 shuffles): observed mean of (own family left out minus standard) +0.231 points, mean under shuffled families +0.155; p = 0.0005.
- Spearman over models between (s_fam - s_ref) and family size: +0.044 [-0.149, +0.188].
- Cross-family pairs with a non-zero standard difference: 0; share whose sign changes when both families are left out nan%, when the same number of unrelated models is left out nan%; excess +nan [+nan, +nan] percentage points. Spearman between the shift of the pair difference and the difference of family sizes: +0.115 [-0.121, +0.192].
- **Reading fixed in the design:** own participation detectable but small; family dependence detectable but small; total detectable but small; ranking consequence not detected.
- By family (members; total, own participation, family dependence, in points): qwen (23; +0.26, +0.17, +0.09); llama (16; +0.30, +0.14, +0.15); deepseek-coder (14; +0.39, +0.12, +0.27); gpt (8; +0.00, +0.01, -0.01); mistral (8; +0.17, +0.18, -0.01); codellama (7; +0.19, +0.19, +0.00); qwen-coder (7; +0.00, +0.01, -0.01); claude (6; +0.45, +0.45, +0.00); deepseek-v2plus (6; +0.23, +0.20, +0.03); gemini (6; +0.23, +0.23, -0.01); phi (6; +0.11, +0.12, -0.00); yi (5; +0.14, +0.15, -0.01); granite (4; +0.17, +0.17, +0.00); gemma (3; +0.23, +0.23, +0.00).

## Sensitivity analyses (point estimates and permutation test)

### Complete, model-line families, k = 74
- 153 models in 19 families (sizes 23, 19, 17, 11, 10, 9, 8, 8, ...; 2 singletons); standard hard subset 74 tasks; mean standard score 0.0000.
- Own participation, T_self: +0.207 points.
- Family dependence beyond own participation, T_family: +0.049 points over all models; +0.050 over models whose family has at least two members.
- Total, T_total: +0.256 points.
- Permutation test of family structure (2000 shuffles): observed mean of (own family left out minus standard) +0.256 points, mean under shuffled families +0.212; p = 0.0155.
- Spearman over models between (s_fam - s_ref) and family size: +0.038.
- Cross-family pairs with a non-zero standard difference: 0; share whose sign changes when both families are left out nan%, when the same number of unrelated models is left out nan%; excess +nan percentage points. Spearman between the shift of the pair difference and the difference of family sizes: +0.015.

### Complete, model-line families, k = 296
- 153 models in 19 families (sizes 23, 19, 17, 11, 10, 9, 8, 8, ...; 2 singletons); standard hard subset 296 tasks; mean standard score 0.0309.
- Own participation, T_self: +0.297 points.
- Family dependence beyond own participation, T_family: +0.195 points over all models; +0.197 over models whose family has at least two members.
- Total, T_total: +0.492 points.
- Permutation test of family structure (2000 shuffles): observed mean of (own family left out minus standard) +0.459 points, mean under shuffled families +0.286; p = 0.0005.
- Spearman over models between (s_fam - s_ref) and family size: -0.002.
- Cross-family pairs with a non-zero standard difference: 10109; share whose sign changes when both families are left out 2.40%, when the same number of unrelated models is left out 1.89%; excess +0.510 percentage points. Spearman between the shift of the pair difference and the difference of family sizes: +0.013.

### Complete, model-line families, solve rate below 50%
- 153 models in 19 families (sizes 23, 19, 17, 11, 10, 9, 8, 8, ...; 2 singletons); standard hard subset 616 tasks; mean standard score 0.1771.
- Own participation, T_self: +0.215 points.
- Family dependence beyond own participation, T_family: +0.264 points over all models; +0.268 over models whose family has at least two members.
- Total, T_total: +0.479 points.
- Permutation test of family structure (2000 shuffles): observed mean of (own family left out minus standard) +0.339 points, mean under shuffled families +0.086; p = 0.0005.
- Spearman over models between (s_fam - s_ref) and family size: -0.016.
- Cross-family pairs with a non-zero standard difference: 10710; share whose sign changes when both families are left out 0.69%, when the same number of unrelated models is left out 0.46%; excess +0.233 percentage points. Spearman between the shift of the pair difference and the difference of family sizes: +0.072.

### Complete, lexical families, k = 148
- 153 models in 39 families (sizes 19, 16, 13, 9, 8, 8, 7, 6, ...; 15 singletons); standard hard subset 148 tasks; mean standard score 0.0023.
- Own participation, T_self: +0.200 points.
- Family dependence beyond own participation, T_family: -0.006 points over all models; -0.007 over models whose family has at least two members.
- Total, T_total: +0.194 points.
- Permutation test of family structure (2000 shuffles): observed mean of (own family left out minus standard) +0.185 points, mean under shuffled families +0.191; p = 0.6772.
- Spearman over models between (s_fam - s_ref) and family size: -0.018.
- Cross-family pairs with a non-zero standard difference: 4881; share whose sign changes when both families are left out 2.68%, when the same number of unrelated models is left out 0.56%; excess +2.127 percentage points. Spearman between the shift of the pair difference and the difference of family sizes: +0.008.

### Complete, organisation families, k = 148
- 153 models in 15 families (sizes 31, 30, 24, 11, 10, 9, 8, 8, ...; 2 singletons); standard hard subset 148 tasks; mean standard score 0.0023.
- Own participation, T_self: +0.204 points.
- Family dependence beyond own participation, T_family: +0.023 points over all models; +0.023 over models whose family has at least two members.
- Total, T_total: +0.227 points.
- Permutation test of family structure (2000 shuffles): observed mean of (own family left out minus standard) +0.212 points, mean under shuffled families +0.195; p = 0.1994.
- Spearman over models between (s_fam - s_ref) and family size: -0.101.
- Cross-family pairs with a non-zero standard difference: 4539; share whose sign changes when both families are left out 2.45%, when the same number of unrelated models is left out 1.30%; excess +1.146 percentage points. Spearman between the shift of the pair difference and the difference of family sizes: +0.002.

