# Panel composition: does concentrating a filter panel in one model line cost that line's other models? — study design

**Status:** written 2026-11-17, before any quantity defined below was computed. Justin asked for this study, as a small scoped version that does not presume that a spread panel is better. **Exploratory:** on BigCodeBench tables already analysed (`SUBSET_C_DESIGN.md`, `PANEL_FILTER_DESIGN.md`, `results/separated_scoring/summary.md`).

**What had been looked at before this design was fixed.** Everything in those three studies, including the cross-prompt effect of own participation (about 1.2 points at 8 builders, Complete votes and Instruct scores). For this design, by `scripts/panel_composition_check.py`, which builds no kept set and scores nothing on one: every model's pass rate over all tasks on both splits, the partition below, the trial panels' abilities, and the agreement of pass/fail patterns on the voting split within and across lines.

## 1. Question

A filter that keeps the tasks most of a panel fails penalises the panel's members, and the MMLU study suggests it also penalises models of the members' lines that did not vote (`MMLU_RECONSTRUCTION_DESIGN.md`, Sections 9 and 11: the Llama-2 line, four of eight votes). Holding the panel's size and ability fixed:

1. **Concentrated against spread.** Do the non-voting models of a line score lower, relative to equally able models of unaffected lines, when half of the panel comes from their line than when one member does and the others come from distinct lines?
2. **Where the cost goes.** When the panel is spread, how much do the lines that gain a voter lose?
3. **What the panel keeps.** How do the two panels differ in the number, difficulty and discrimination of the tasks kept?

A spread panel is not presumed to be better. Spreading moves the cost from one line to several; the study measures both sides.

## 2. Data

BigCodeBench v0.1.4, 1,140 tasks, the Complete and Instruct tables of `bigcode/bigcodebench-perf` (`results/subsets/raw/`, hashes checked against `manifest.tsv`), the 124 models in both. **Votes come from Complete and scores from Instruct**, as in `SEPARATED_SCORING_DESIGN.md`: the scored outputs never voted. Instruct votes with Complete scores is a secondary direction on the same panels; it is not an independent replication.

## 3. Lines and pools

- **Lines:** the primary model-line rules of `PANEL_FILTER_DESIGN.md`, Section 3, read from names (19 lines). *Check without names:* among pairs of models whose Complete pass rates differ by at most 3 points, pass/fail patterns agree more within a line than across lines (mean phi coefficient): qwen 0.563 against 0.525, llama 0.616 against 0.522, deepseek-coder 0.661 against 0.525, gpt 0.622 against 0.577, qwen-coder 0.683 against 0.547, codellama 0.544 against 0.504, mistral 0.513 against 0.505. Lines therefore share failures beyond ability, weakly for mistral.
- **Pools, fixed once:** in every line of at least two models, ⌊n/2⌋ models drawn with a keyed generator form the **evaluation pool E**; the others form the **builder pool B**; a line of one model goes to B. E has 57 models and B 67. **No model of E ever votes.**
- **Panel members:** models of B with a Complete pass rate of at least 25%. Without this rule the weakest qwen and llama models (8% and 11%) have no model of similar ability in other lines, and the two panels of a pair differ in mean ability by up to 6 points (check).
- **Focal lines:** p = 8 (c = 4 voters of the line): the lines with at least four panel-eligible models in B and at least two models in E, namely qwen, llama, deepseek-coder, gpt, mistral, codellama and qwen-coder. p = 16 (c = 8): qwen only, the one line with eight panel-eligible models in B.

## 4. Panels

For a focal line L, panel size p and c = p/2, each replicate draws (keyed generator per p, line and replicate):

1. **Voters of L:** c panel-eligible B models of L, uniformly without replacement.
2. **Others:** p − c panel-eligible B models of other lines, at most one per line: B models outside L in a random order, each taken if its line is not yet in the panel.
3. **Concentrated panel** P_c: the c voters of L and the p − c others.
4. **Spread panel** P_s: one voter of L, drawn at random, stays. Each of the other c − 1, in random order, is replaced by the panel-eligible B model nearest to it in Complete pass rate among the lines not yet in the panel (ties drawn at random).

P_s holds one model of L and p − 1 models of distinct lines. The two panels share p − c members; they differ by c − 1 models of L against c − 1 models of **new lines**, matched one by one on ability. Feasibility check (200 trial panels per line): median gap of a replacement 0.3 to 0.7 points, panel mean abilities within 1.1 points at p = 8 and 1.4 points at p = 16.

100 replicates per focal line and size. Panel mean ability (Complete) of both panels is reported for every replicate.

## 5. Rule and scores

The published rule of BigCodeBench-Hard on the panel: a task is kept if fewer than half of the panel's models pass it on the voting split, H(P) = {t : 2 Σ_{j∈P} y_jt < p}. A model's score on H is its mean Instruct pass rate there. Nothing on the scoring split enters a panel, a pool or a kept set.

## 6. Estimands

For a model e of E, d_e = S_{H(P_s)}(z_e) − S_{H(P_c)}(z_e): how much better it does on the spread panel's kept set.

- **Unchanged lines** of a replicate: all lines except L and the new lines. Their models are the same in both panels (none, or the same member), so their E models are the controls.
- **Controls of an E model:** the five E models of unchanged lines nearest to it in Instruct pass rate over all tasks (ties broken by a keyed random number). They take out the change that the two kept sets cause for any model of that ability.
- **Concentration cost of line L** in a replicate: the mean over the E models w of L of [d_w − mean of d over w's controls]. **T(L, p)** is its mean over replicates. Positive means that L's non-voting models lose more on the concentrated panel's kept set than equally able models of unaffected lines. **T(8)**, the mean of T(L, 8) over the seven focal lines with equal weight, is the **primary estimand**.
- **Cost of spreading:** for the E models of the new lines, the mean over replicates of the same difference, T_new(L, p). Negative means that a line gaining one voter loses. Lines of one model have no E model and do not enter.

## 7. Inference

Task bootstrap, 1,000 replicates: multinomial weights over the 1,140 tasks, keyed, shared by all models, panels and replicates; panels, kept status and controls fixed (the rule is per task, so this equals resampling tasks and applying it again). Percentile 95% intervals. For the eight contrasts T(L, p), simultaneous intervals from the bootstrap maximum of |t| over the eight. The spread of the estimates over the 100 panel replicates is reported as a Monte Carlo standard error; it is not an interval over a population of panels. Models are fixed; one run per model and split.

## 8. Readings, fixed in advance

As in `PANEL_FILTER_DESIGN.md`, Section 7, in points (1 point = 0.01):

- **material:** the interval excludes zero and the estimate is at least 1 point;
- **detectable but small:** the interval excludes zero and the estimate is below 1 point;
- **not detected:** otherwise; not evidence of absence, and the interval is the bound.

Applied to T(8) with its percentile interval (primary) and to each T(L, p) with simultaneous intervals. A null or negative result is reported as it is. A positive T(8) shows that concentrating the panel in a line costs that line's other models more than equally able models of unaffected lines; it does not show that a spread panel gives the right ranking, and it is read together with T_new and with what the panels keep (Section 9).

## 9. Descriptions, fixed

For both panels of each replicate, means over replicates: the number of tasks kept and the overlap of the two kept sets (Jaccard); difficulty, the mean Instruct pass rate of all E models on the kept set; discrimination, the mean corrected item–total correlation over E models of the kept tasks (Instruct; tasks that every E model passes or fails are left out and counted), and the standard deviation of the E models' scores on the kept set; panel mean ability (Complete). The distribution over replicates of the concentration cost and of T_new.

## 10. Sensitivity, fixed (reported, not read)

The secondary direction (Instruct votes, Complete scores; same panels and controls chosen on Complete pass rates); three and ten controls in place of five; no 25% floor for panel members (the matching then fails for some replicates, and the panel mean abilities are reported).

*A detail the design left open, settled before the run (2026-11-17):* in every variant T(8) is the mean over the seven focal lines of the primary analysis; a line that becomes focal only in a variant (without the floor) is reported separately and does not enter T(8). The panel-replicate standard error is shown under that name (Deviation 1).

## 11. Checks before the run

On the real tables with synthetic scoring vectors: a constant scoring vector gives zero for every estimand; scoring vectors drawn independently of the votes give estimates within Monte Carlo error of zero; when the scoring vector of each E model of the focal line is replaced by the voting vector of a voter of its line, T(L, p) is positive. Bootstrap weights against explicit resampling of tasks for a few replicates. Then the real run, once.

**Deviation 1** (2026-11-17, from the first check run, before any Instruct outcome was read). With independent scoring vectors, six of the eight estimates lay more than four panel-replicate standard errors from zero (largest −0.31 points against 0.05). The criterion was wrong, not the code: for a fixed scoring vector the two panels of every replicate differ on much the same tasks, so the spread over panel replicates leaves out the task-level noise of that vector, which is the uncertainty this study's intervals carry. The check now compares each estimate with its task-bootstrap standard deviation (200 replicates; criterion 3.5 SD for each of the eight). Consequence for the run: the panel-replicate standard error of Section 7 describes only the variation over panels and is not to be read as the precision of an estimate.

**Null calibration** (added 2026-11-17 before the run). A dry run of the whole script with one synthetic scoring split drawn independently of the votes, to test the reporting code, gave T(8) = −0.20 points with a percentile interval excluding zero (about 2.4 bootstrap SD). Forty further independent splits (`scripts/panel_composition_calibration.py`, 200 bootstrap replicates each):

- T(8) had mean +0.01 points and SD 0.078 over the draws, against a mean bootstrap SD of 0.083 (ratio 0.94), and its percentile interval excluded zero in 3 of 40 draws.
- Over the eight contrasts, the ratio of estimate to bootstrap SD had SD 0.98, and 16 of 320 percentile intervals excluded zero (5%).
- At least one of the eight simultaneous intervals excluded zero in 3 of 40 draws.

The intervals are close to nominal under the null, and the dry-run draw was a chance excess. No reading is changed.

## 12. Stated in advance (my expectations)

- T(8) positive and below 1 point: lines share failures beyond ability only weakly (Section 3), and the cross-prompt effect of a member's own vote at 8 builders is about 1.2 points; three more voters of the line act on its non-voting models through that weaker link.
- Larger for the lines whose patterns agree most (deepseek-coder, qwen-coder), near zero for mistral.
- T_new negative and smaller per line than T(L, 8) is positive, since each new line gains one voter.
- The two panels keep similar numbers of tasks of similar difficulty, since they are matched on ability.

## 13. Limits stated in advance

- Lines are read from names; the agreement check supports them but is not proof of shared weights.
- B and E are one fixed partition; E has two to eleven models per focal line, and three for codellama and qwen-coder.
- At 16 builders only qwen can be concentrated; the size curve of this question is not estimated.
- Cross-prompt, not repeated runs of one protocol; one run per model.
- Random panels within the rules, not panels that builders chose; the 25% floor and the ability matching are choices of this design.
- Exploratory, on tables already analysed.

## 14. Outputs

`scripts/panel_composition_check.py` (feasibility check, run); `scripts/panel_composition.py`; `results/panel_composition/` (summary.md, records.tsv, replicates.tsv, pools.tsv). No manuscript file is changed.

## 15. Run

2026-11-17, `python -E -P -X utf8 scripts/panel_composition.py` from `experiments/`, once, after the checks of Section 11 had passed (second check run, with Deviation 1) and after the null calibration. Outputs are in `results/panel_composition/`.

- Input hashes matched the manifest. E = 57 models and B = 67, of which 63 are panel-eligible.
- The focal lines were seven at 8 builders and qwen at 16, as designed. All 100 panel pairs were built for each of the eight contrasts.
- 1,000 bootstrap replicates; the simultaneous multiplier was 2.77.

Changes to the script after the design was fixed:

- the check criterion (Deviation 1);
- a guard for an empty selection in the discrimination mean (it affects only the constant check);
- the name of the panel-replicate SE column;
- T(8) over the primary focal lines in the sensitivity table (the note in Section 10).

None of these was made after an Instruct outcome had been read.

## 16. Results by the fixed readings

*The cost of spreading and the kept-set comparison are corrected in Section 19 (2026-11-17).*

**Primary.** T(8) = +0.66 points (95% interval +0.51 to +0.80): **detectable but small**.

**By line**, with simultaneous intervals over the eight contrasts:

| Size | Line | E models | Concentration cost, points | Simultaneous interval | Reading | Cost of spreading, points (95% interval) |
|---:|---|---:|---:|---|---|---|
| 8 | qwen | 11 | +0.33 | +0.03 to +0.64 | detectable but small | −0.07 (−0.16 to +0.03) |
| 8 | llama | 8 | +0.40 | +0.04 to +0.76 | detectable but small | −0.09 (−0.22 to +0.04) |
| 8 | deepseek-coder | 6 | +1.00 (0.997) | +0.52 to +1.48 | detectable but small, just below the 1-point threshold | −0.13 (−0.27 to −0.01) |
| 8 | gpt | 4 | +0.81 | +0.07 to +1.56 | detectable but small | +0.08 (−0.24 to +0.40) |
| 8 | mistral | 4 | +0.40 | −0.18 to +0.99 | not detected | −0.26 (−0.46 to −0.07) |
| 8 | codellama | 3 | +0.83 | +0.16 to +1.50 | detectable but small | −0.05 (−0.21 to +0.11) |
| 8 | qwen-coder | 3 | +0.83 | +0.21 to +1.45 | detectable but small | −0.17 (−0.38 to +0.05) |
| 16 | qwen | 11 | +0.40 | −0.00 to +0.81 | not detected | −0.12 (−0.22 to −0.02) |

No contrast reaches "material".

- **Over panel pairs.** The share of pairs with a positive concentration cost is 83% to 100% for each contrast (100% for deepseek-coder).
- **Cost of spreading.** It is negative for six of the seven lines at 8 builders and for qwen at 16, at −0.05 to −0.26 points for each line that gains a voter; gpt is +0.08. The 95% interval excludes zero for deepseek-coder, mistral and qwen at 16. A spread panel gives one voter each to three new lines at 8 builders and to seven at 16.
- **What the panels keep.** Spread minus concentrated, as means over pairs:
  - tasks kept: −6 to +5 tasks out of 470–568;
  - Jaccard overlap of the two kept sets: 0.85–0.89;
  - difficulty (E models' mean Instruct pass rate on the kept set): −0.6 to +0.4 points;
  - mean corrected item–total correlation: −0.013 to +0.008;
  - panel mean ability: −0.25 to +0.22 points.

  Neither composition keeps a set that is consistently easier, harder or more discriminating.
- **Sensitivity (reported, not read).** T(8) is +0.43 in the reverse direction, +0.67 with three controls, +0.65 with ten and +0.58 without the floor. In the reverse direction gpt and mistral fall to +0.01. Every other entry of every variant is positive.
- **Post hoc, not read.** Within a line, the per-pair concentration cost correlates with the per-pair difference in kept-set difficulty at −0.46 (codellama) to +0.10 (qwen at 16).

## 17. Against what was stated in advance

- **T(8) positive and below 1 point:** as stated (+0.66).
- **Larger for deepseek-coder and qwen-coder:** partly right.
  - deepseek-coder is the largest (+1.00).
  - qwen-coder (+0.83) is level with codellama (+0.83) and gpt (+0.81).
  - Across lines, the order does not follow the coherence numbers of Section 3 closely: gpt and codellama, with small coherence gaps, are as large as qwen-coder.
- **Near zero for mistral:** wrong as a point estimate. It is +0.40, as large as qwen and llama, although not detected.
- **T_new negative and smaller per line:** as stated for six of the seven lines and at 16 builders; gpt is +0.08.
- **Similar tasks kept:** as stated.

## 18. Reading (exploratory)

*Items 2 and 3 and the wording are corrected in Section 19 (2026-11-17).*

1. **The cost.** At 8 builders, with ability matched, compare a panel with half its members from one line against a panel spread across lines. The concentrated panel keeps tasks on which that line's non-voting models lose about 0.66 points more than equally able models of unaffected lines. By the fixed reading this is detectable but small. It is positive in six of the seven lines (mistral is not detected) and in every sensitivity variant.
2. **Spreading moves the cost.** Lines that gain one voter lose about 0.1 point each, at most 0.26. Spreading the panel does not remove the cost; it divides it into smaller pieces across more lines. This study does not show that a spread panel is better.
3. **What is kept.** The two compositions keep sets of the same size, difficulty and discrimination, within about half a point and 0.013. The difference in cost therefore comes without an easier or less discriminating kept set. What changes is whose failures define the set.
4. **Which outputs are scored.** The estimate concerns non-voting models of a voter's line, scored on outputs that never voted (cross-prompt). For scale only: the cross-prompt effect of a member's own vote at 8 builders is about 1.2 points (`results/separated_scoring/summary.md`). That is a different quantity.
5. **What this does not show:**
   - that a spread panel gives the right ranking or a better benchmark;
   - that the effect extends beyond BigCodeBench, these lines or panels of 8. At 16 builders only qwen could be concentrated, and its estimate (+0.40) is not detected.

   Lines are read from names.

**Wording for the paper.** On BigCodeBench, with panel size and ability held fixed, concentrating half of an 8-model filter panel in one model line cost that line's non-voting models about 0.7 points (95% interval 0.5 to 0.8) relative to equally able models of unaffected lines. Spreading the panel moved smaller costs, about 0.1 point each, to the lines that gained a voter. The two compositions kept tasks of the same number, difficulty and discrimination. Do not write that diversification helps or removes the effect.

## 19. Corrections (2026-11-17)

At Justin's request, before any manuscript text; nothing was recomputed.

1. **The cost of spreading is pooled over lines.** T_new(L, p) is the mean over all evaluation-pool models of the lines that gained a voter, taken together (`scripts/panel_composition.py`, where `new_e` is built). Lines with more E models therefore weigh more, and the quantity is not the loss of each new line. "For each line that gains a voter" (Section 16) and "Lines that gain one voter lose about 0.1 point each" (Section 18) are withdrawn. The correct reading: the evaluation models of the lines that gained a voter, pooled, changed by −0.05 to −0.26 points relative to their controls (gpt +0.08).
2. **No conservation.** The study shows that spreading reduced the focal line's disadvantage relative to its controls. It also shows that the pooled evaluation models of the lines that gained a seat had smaller negative changes. It shows neither that the effect is conserved nor that spreading can only redistribute it. "Spreading the panel does not remove the cost; it divides it into smaller pieces across more lines" (Section 18, item 2) is withdrawn, together with the paper wording below it.
3. **No equivalence test.** "The same size, difficulty and discrimination" (Sections 16 to 18) is withdrawn. The observed mean differences were small and had no consistent direction (ranges in Section 16). No equivalence test was run.
4. **Wording for the paper** (replaces the wording at the end of Section 18). On BigCodeBench, with panel size and mean panel ability held fixed, a panel with half its members from one model line put that line's non-voting models at a disadvantage of about 0.7 points (95% interval 0.5 to 0.8). The comparison is against equally able models of unaffected lines, and against a panel spread across lines. On the spread panel, the evaluation models of the lines that gained a voter had smaller negative changes on average (−0.05 to −0.26 points; one line +0.08). The two kept sets differed little on average in size, difficulty and discrimination, with no consistent direction.
5. **Do not write** that the effect is conserved, that spreading only redistributes it, that the kept sets are the same, or what each line loses.
