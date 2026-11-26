# Own-panel penalty by panel size on original MMLU: results

Specification: `experiments/MMLU_RECONSTRUCTION_DESIGN.md`, Section 10 (fixed 2026-11-15; exploratory). 14042 questions. 100 random panels per size; every member evaluated up to 16 per panel; 100 question-bootstrap replicates within subject; 95% percentile intervals in brackets. 1 point = 0.01 of accuracy. Only point estimates of the sign changes are to be read.

Corrections of 2026-11-16 (design, Section 12; no estimate changed). (1) Own participation is given twice: over every evaluated member, and over the members that have the same-line comparison, which are the ones the next two columns use; only on those does the total equal own participation plus the same-line part. (2) The same outcomes vote and score here, so own participation cannot be negative in any comparison; the intervals and the placebo say nothing about its sign, only about its size. (3) "Same-line" means the same model line read from names.

## Population A, MMLU-Pro's rule (kept if at most half of the panel is right): 75 systems, 23 model lines (name rules)

| Builders | Questions kept (mean; minimum) | Mean score of all systems on them | Own participation, all evaluated members, points | Own participation, members with the same-line comparison, points | With same-line builders too, same members, points | Same-line part, same members, points | Placebo, points | Sign changes: member pairs / baseline | Excess, percentage points | Reading on size: own / total |
|---:|---|---:|---|---|---|---|---|---|---:|---|
| 4 | 9113; 4792 | 39.2% | +9.80 [+9.68, +9.92] | +9.67 [+9.55, +9.79] | +9.92 [+9.80, +10.03] | +0.25 [+0.24, +0.27] | -0.07 [-0.09, -0.06] | 17.93% / 5.17% | +12.75 | material / material |
| 8 | 8310; 4595 | 36.4% | +6.48 [+6.38, +6.58] | +6.48 [+6.38, +6.58] | +7.17 [+7.07, +7.27] | +0.69 [+0.67, +0.70] | -0.06 [-0.07, -0.05] | 15.96% / 4.07% | +11.90 | material / material |
| 16 | 7715; 5390 | 34.1% | +3.80 [+3.73, +3.85] | +3.80 [+3.73, +3.85] | +4.81 [+4.73, +4.89] | +1.01 [+0.99, +1.03] | +0.03 [+0.02, +0.04] | 13.30% / 2.88% | +10.42 | material / material |
| 32 | 7232; 5308 | 32.2% | +2.06 [+2.01, +2.11] | +2.06 [+2.01, +2.11] | +3.32 [+3.25, +3.39] | +1.26 [+1.22, +1.30] | +0.03 [+0.02, +0.04] | 9.61% / 2.00% | +7.61 | material / material |
| 64 | 7059; 6469 | 31.3% | +1.03 [+0.99, +1.08] | +1.00 [+0.96, +1.04] | +1.67 [+1.60, +1.74] | +0.67 [+0.63, +0.71] | -0.00 [-0.01, +0.01] | 6.65% / 1.33% | +5.32 | material / material |

- Members evaluated, of which with a same-line builder in the panel, and members without the sibling comparison: p = 4: 400, 60, 12; p = 8: 800, 245, 0; p = 16: 1600, 844, 0; p = 32: 1600, 1164, 0; p = 64: 1600, 1158, 279.
- Own participation times the number of builders (constant if the penalty is proportional to 1/p): p = 4: 39.2; p = 8: 51.8; p = 16: 60.7; p = 32: 66.0; p = 64: 66.0.

## Population A, sensitivity: kept if fewer than half of the panel is right: 75 systems, 23 model lines (name rules)

| Builders | Questions kept (mean; minimum) | Mean score of all systems on them | Own participation, all evaluated members, points | Own participation, members with the same-line comparison, points | With same-line builders too, same members, points | Same-line part, same members, points | Placebo, points | Sign changes: member pairs / baseline | Excess, percentage points | Reading on size: own / total |
|---:|---|---:|---|---|---|---|---|---|---:|---|
| 4 | 5528; 2669 | 31.8% | +16.58 [+16.43, +16.78] | +16.64 [+16.49, +16.84] | +17.55 [+17.39, +17.75] | +0.91 [+0.89, +0.93] | -0.18 [-0.20, -0.15] | 37.61% / 10.07% | +27.54 | material / material |
| 8 | 6083; 3844 | 31.2% | +8.51 [+8.40, +8.63] | +8.51 [+8.40, +8.63] | +9.57 [+9.46, +9.73] | +1.07 [+1.04, +1.10] | -0.03 [-0.05, -0.02] | 24.59% / 6.66% | +17.93 | material / material |
| 16 | 6157; 4305 | 30.0% | +4.39 [+4.31, +4.47] | +4.39 [+4.31, +4.47] | +5.56 [+5.47, +5.66] | +1.17 [+1.14, +1.21] | +0.04 [+0.03, +0.05] | 17.81% / 3.99% | +13.82 | material / material |
| 32 | 6616; 5462 | 30.5% | +2.19 [+2.13, +2.25] | +2.19 [+2.13, +2.25] | +3.41 [+3.32, +3.48] | +1.22 [+1.17, +1.25] | +0.05 [+0.04, +0.05] | 11.26% / 2.31% | +8.94 | material / material |
| 64 | 6722; 6149 | 30.4% | +1.09 [+1.04, +1.13] | +1.06 [+1.01, +1.10] | +1.79 [+1.71, +1.87] | +0.73 [+0.69, +0.77] | +0.00 [-0.00, +0.01] | 7.04% / 1.42% | +5.62 | material / material |

- Members evaluated, of which with a same-line builder in the panel, and members without the sibling comparison: p = 4: 400, 64, 3; p = 8: 800, 265, 0; p = 16: 1600, 851, 0; p = 32: 1600, 1154, 0; p = 64: 1600, 1181, 241.
- Own participation times the number of builders (constant if the penalty is proportional to 1/p): p = 4: 66.3; p = 8: 68.0; p = 16: 70.2; p = 32: 70.1; p = 64: 69.8.

## Population B (all systems; lineage mostly unknown), MMLU-Pro's rule: 268 systems, 118 model lines (name rules)

| Builders | Questions kept (mean; minimum) | Mean score of all systems on them | Own participation, points | Placebo, points | Sign changes: member pairs / baseline | Excess, percentage points | Reading: own |
|---:|---|---:|---|---|---|---:|---|
| 4 | 7830; 5319 | 35.5% | +8.98 [+8.86, +9.10] | -0.05 [-0.07, -0.04] | 26.86% / 9.55% | +17.31 | material |
| 8 | 6958; 4781 | 31.3% | +5.53 [+5.42, +5.62] | +0.10 [+0.08, +0.11] | 24.96% / 7.74% | +17.22 | material |
| 16 | 6584; 5659 | 29.0% | +2.88 [+2.81, +2.95] | +0.03 [+0.02, +0.04] | 18.69% / 5.17% | +13.53 | material |
| 32 | 6359; 5746 | 27.6% | +1.48 [+1.43, +1.51] | +0.00 [-0.00, +0.01] | 11.30% / 3.20% | +8.10 | material |
| 64 | 6245; 5695 | 26.9% | +0.75 [+0.72, +0.77] | -0.00 [-0.00, +0.00] | 6.32% / 1.87% | +4.45 | small |
| 96 | 6216; 5859 | 26.6% | +0.51 [+0.48, +0.53] | +0.00 [+0.00, +0.01] | 4.45% / 1.41% | +3.04 | small |

- Members evaluated, of which with a same-line builder in the panel, and members without the sibling comparison: p = 4: 400, 44, 15; p = 8: 800, 242, 10; p = 16: 1600, 698, 0; p = 32: 1600, 772, 0; p = 64: 1600, 871, 0; p = 96: 1600, 949, 0.
- Own participation times the number of builders (constant if the penalty is proportional to 1/p): p = 4: 35.9; p = 8: 44.2; p = 16: 46.1; p = 32: 47.3; p = 64: 47.9; p = 96: 48.8.

