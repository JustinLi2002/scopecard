# Panel composition: results

Specification: `experiments/PANEL_COMPOSITION_DESIGN.md` (fixed 2026-11-17 before any quantity was computed; exploratory). BigCodeBench v0.1.4, 124 models, 1140 tasks. Evaluation pool 57 models (never vote); builder pool 67, of which 63 have a Complete pass rate of at least 25%. 100 panel pairs per focal line and size; 1000 task-bootstrap replicates. Concentration cost: how much more the focal line's non-voting models gain on the spread panel's kept set than their 5 nearest models of unchanged lines; positive means that concentration costs the line. 1 point = 0.01.

## 1. Primary: concentration cost at 8 builders, mean over the focal lines

**T(8) = +0.66 points** (task-bootstrap 95% interval +0.51 to +0.80), over 7 lines with equal weight. **Reading fixed in the design: detectable but small.**

## 2. By line

Simultaneous intervals over the 8 contrasts: estimate ± 2.77 × bootstrap SD. Cost of spreading: the same quantity for the evaluation-pool models of the lines that gain a voter in the spread panel (negative: they lose). Panel-replicate SE: the spread over the 100 panel pairs only, not the precision of an estimate (design, Deviation 1).

| Size | Focal line | E models of the line | Replicates | Concentration cost, points | 95% interval | Simultaneous interval | Reading | Panel-replicate SE | Cost of spreading, points (95% interval) |
|---:|---|---:|---:|---:|---|---|---|---:|---|
| 8 | qwen | 11 | 100 | +0.33 | +0.12 to +0.55 | +0.03 to +0.64 | detectable but small | 0.03 | -0.07 (-0.16 to +0.03) |
| 8 | llama | 8 | 100 | +0.40 | +0.14 to +0.65 | +0.04 to +0.76 | detectable but small | 0.04 | -0.09 (-0.22 to +0.04) |
| 8 | deepseek-coder | 6 | 100 | +1.00 | +0.66 to +1.35 | +0.52 to +1.48 | detectable but small | 0.03 | -0.13 (-0.27 to -0.01) |
| 8 | gpt | 4 | 100 | +0.81 | +0.32 to +1.34 | +0.07 to +1.56 | detectable but small | 0.05 | +0.08 (-0.24 to +0.40) |
| 8 | mistral | 4 | 100 | +0.40 | -0.03 to +0.79 | -0.18 to +0.99 | not detected | 0.04 | -0.26 (-0.46 to -0.07) |
| 8 | codellama | 3 | 100 | +0.83 | +0.38 to +1.32 | +0.16 to +1.50 | detectable but small | 0.05 | -0.05 (-0.21 to +0.11) |
| 8 | qwen-coder | 3 | 100 | +0.83 | +0.39 to +1.28 | +0.21 to +1.45 | detectable but small | 0.04 | -0.17 (-0.38 to +0.05) |
| 16 | qwen | 11 | 100 | +0.40 | +0.13 to +0.71 | -0.00 to +0.81 | not detected | 0.03 | -0.12 (-0.22 to -0.02) |

## 3. What the two panels keep (means over replicates)

| Size | Focal line | Panel | Panel ability, Complete % | Tasks kept | Jaccard overlap | Difficulty: E models' mean Instruct pass rate, % | Discrimination: mean corrected item–total correlation | SD of E models' scores, points | Largest replacement gap, points |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | qwen | concentrated | 45.9 | 552 | 0.866 | 13.3 | 0.165 | 4.7 | 1.5 |
| 8 | qwen | spread | 46.1 | 554 | 0.866 | 13.5 | 0.170 | 4.8 | 1.5 |
| 8 | llama | concentrated | 46.3 | 549 | 0.860 | 13.4 | 0.167 | 4.7 | 1.5 |
| 8 | llama | spread | 46.3 | 552 | 0.860 | 13.4 | 0.169 | 4.8 | 1.5 |
| 8 | deepseek-coder | concentrated | 46.2 | 559 | 0.858 | 14.0 | 0.182 | 5.3 | 1.2 |
| 8 | deepseek-coder | spread | 46.2 | 556 | 0.858 | 13.6 | 0.176 | 5.0 | 1.2 |
| 8 | gpt | concentrated | 52.4 | 470 | 0.846 | 10.2 | 0.117 | 3.0 | 1.8 |
| 8 | gpt | spread | 52.1 | 474 | 0.846 | 9.9 | 0.116 | 3.0 | 1.8 |
| 8 | mistral | concentrated | 46.0 | 555 | 0.861 | 13.7 | 0.172 | 5.0 | 0.7 |
| 8 | mistral | spread | 46.0 | 561 | 0.861 | 13.8 | 0.176 | 5.1 | 0.7 |
| 8 | codellama | concentrated | 45.5 | 568 | 0.856 | 14.5 | 0.192 | 5.5 | 1.1 |
| 8 | codellama | spread | 45.7 | 562 | 0.856 | 13.9 | 0.179 | 5.2 | 1.1 |
| 8 | qwen-coder | concentrated | 45.7 | 562 | 0.864 | 13.7 | 0.176 | 5.1 | 0.6 |
| 8 | qwen-coder | spread | 45.8 | 558 | 0.864 | 13.7 | 0.172 | 4.9 | 0.6 |
| 16 | qwen | concentrated | 46.4 | 562 | 0.887 | 13.1 | 0.164 | 4.7 | 5.2 |
| 16 | qwen | spread | 46.6 | 567 | 0.887 | 13.5 | 0.172 | 5.0 | 5.2 |

Distribution over replicates of the concentration cost (point estimates per replicate), points:

- 8 builders, qwen: median +0.35, quartiles +0.15 and +0.54, range -0.71 to +1.13, share above zero 87%.
- 8 builders, llama: median +0.43, quartiles +0.10 and +0.62, range -0.35 to +1.22, share above zero 85%.
- 8 builders, deepseek-coder: median +0.95, quartiles +0.78 and +1.24, range +0.26 to +1.87, share above zero 100%.
- 8 builders, gpt: median +0.83, quartiles +0.51 and +1.15, range -0.35 to +1.71, share above zero 96%.
- 8 builders, mistral: median +0.37, quartiles +0.18 and +0.69, range -0.61 to +1.26, share above zero 83%.
- 8 builders, codellama: median +0.86, quartiles +0.45 and +1.19, range -0.61 to +2.04, share above zero 96%.
- 8 builders, qwen-coder: median +0.82, quartiles +0.54 and +1.11, range -0.13 to +2.04, share above zero 98%.
- 16 builders, qwen: median +0.39, quartiles +0.22 and +0.55, range -0.12 to +1.07, share above zero 94%.

## 4. Sensitivity (reported, not read)

| Variant | 8: qwen | 8: llama | 8: deepseek-coder | 8: gpt | 8: mistral | 8: codellama | 8: qwen-coder | 16: qwen | T(8), the primary focal lines |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Instruct votes, Complete scores | +0.28 | +0.42 | +0.99 | +0.01 | +0.01 | +0.26 | +1.07 | +0.45 | +0.43 |
| three controls | +0.31 | +0.36 | +1.02 | +0.82 | +0.43 | +0.90 | +0.83 | +0.39 | +0.67 |
| ten controls | +0.32 | +0.36 | +1.01 | +0.75 | +0.43 | +0.80 | +0.88 | +0.37 | +0.65 |
| no 25% floor for panel members | +0.24 | +0.33 | +0.91 | +0.68 | +0.45 | +0.71 | +0.75 | +0.31 | +0.58 |

- no 25% floor for panel members: llama is a focal line at 16 builders only in this variant: +0.28 points (not in T(8)).
- Without the floor: largest replacement gap up to 37.2 points; panel mean ability, spread minus concentrated, -0.93 to +6.02 points.

