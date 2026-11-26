# Does a difficulty filter built from models' outcomes depend on the models that built it? — study design

**Status:** design fixed 2026-11-15, before any quantity defined below was computed. Justin asked for the study on that day. Internal, dated document: the word used is "prespecified".

**What had been seen before:** the BigCodeBench performance tables were analysed on 2026-11-14 (`SUBSET_C_DESIGN.md`, `results/subsets/`): scores on BigCodeBench-Hard and on the other tasks, pair statistics, and a grouping of models by date. In that analysis the models dated on or before the Hard release disagreed with the parent ranking more than random subsets did on Complete, and not clearly on Instruct. That observation motivates this study, so the study is **prespecified on data that have already been analysed**: no leave-out construction, no solve-rate filter of our own and no family grouping had been computed. The list of model names was read to write the family rules in Section 3; no outcome was read with it.

## 1. Question

BigCodeBench-Hard keeps tasks whose solve rate over the leaderboard's models is below 50%, and the same leaderboard then ranks those models on the tasks kept. A task is therefore kept partly because the models being ranked failed it.

1. **Own participation.** Does a model score lower on a hard subset that its own outcomes helped to select than on one selected without it?
2. **Family dependence.** Does it score lower when models that share its base weights helped to select the subset, beyond its own participation?
3. **Consequence.** Does scoring every comparison on a subset built without the two models' families change which model is ahead, more than changing the panel by the same number of unrelated models does?
4. **Remedy.** How far does cross-fitting (building the subset without the models being compared) remove the dependence?

Only the outcome-based part of the rule is studied. The published rule also requires more than two libraries, a long solution and a Stack Overflow match; those do not depend on the models.

## 2. Data

`bigcode/bigcodebench-perf` as downloaded on 2026-11-14 (`results/subsets/raw/`, hashes in `manifest.tsv`): Complete (153 models × 1,140 tasks, primary) and Instruct (126 models, replication on the same tasks, with 124 models in common, so not an independent replication). Cell value "1" is a pass. All models are used; none is selected by performance.

## 3. Families (fixed before any outcome)

**Primary: model line.** Models that share base weights or are fine-tuned from the same base are one family. The assignment is by the ordered rules below (first match on the model name), written from knowledge of each model's provenance and not checked against model cards:

| Family | Rule on the name |
|---|---|
| gpt | starts with `GPT-`, or is `Quasar-Alpha` |
| claude | starts with `Claude-` |
| gemini | starts with `Gemini-` |
| gemma | starts with `Gemma-` or `CodeGemma-` |
| codellama | starts with `CodeLlama-`, `Phind-CodeLlama` or `ReflectionCoder-CL-` |
| llama | starts with `Llama-`, `Hermes-2-` (Llama-3 fine-tunes), `OpenChat-3.6`, `Athene-70B` or `DeepSeek-R1-Distill-Llama-` |
| deepseek-coder | `DeepSeek-Coder-1.3B/6.7B/33B-*`, `Magicoder-S-DS`, `OpenCodeInterpreter-DS`, `AutoCoder`, `AutoCoder-S-`, `Artigenz-Coder-DS`, `ReflectionCoder-DS-`, `WaveCoder-Ultra`, `WarriorCoder`, `WhiteRabbitNeo-33B` |
| deepseek-v2plus | starts with `DeepSeek-Coder-V2`, `DeepSeek-V2` or `DeepSeek-V3` |
| qwen-coder | starts with `CodeQwen`, `Qwen2.5-Coder-`, `AutoCoder-QW` or `Nxcode-CQ` |
| qwen | starts with `Qwen`, `QwQ-`, `Tess-` (Qwen2 fine-tune), `Athene-V2`, `Sky-T1`, `DeepSeek-R1-Distill-Qwen-` or `DeepCoder-` |
| mistral | starts with `Mistral-`, `Mixtral-` or `Codestral-` |
| phi, granite, yi, starcoder2 | starts with `Phi-`, `Granite-`, `Yi-`, `StarCoder2` |
| cohere | starts with `C4AI-Command` or `Command R` |
| codegeex, internlm, opencoder | starts with `CodeGeex`, `InternLM`, `OpenCoder-` |

A name that matches no rule forms its own family and is listed in the output.

**Sensitivity:** (a) lexical families, the leading run of letters of the name in lower case, which needs no knowledge of provenance and splits fine-tunes from their bases; (b) organisation of the base weights, obtained by merging gemini with gemma, llama with codellama, deepseek-coder with deepseek-v2plus, and qwen with qwen-coder.

## 4. Construction

For a set S of builder models, the solve rate of task t is the mean of y over S, and **H(S)** is the k tasks with the lowest solve rate, ties broken by the order of the task list. **k = 148**, the size of BigCodeBench-Hard. M is the set of all models.

For each model m, with family F of size n:

- **standard:** s_std = score on H(M);
- **own family left out:** s_fam = score on H(M \ F);
- **self left out:** s_self = score on H(M \ ({m} ∪ R)), R being n − 1 models drawn at random outside F, averaged over 30 draws;
- **reference:** s_ref = score on H(M \ R′), R′ being n models drawn at random outside F with m kept in, averaged over 30 draws.

All three leave-out conditions remove the same number of builders, so they differ only in who is removed.

## 5. Estimands

Means over models:

- **T_self = mean(s_self − s_ref):** own participation.
- **T_family = mean(s_fam − s_self):** family dependence beyond own participation. It is zero by construction for a family of one, so it is also reported over the models whose family has at least two members.
- **T_total = mean(s_fam − s_ref).**

Also: the three by family; Spearman correlation over models between (s_fam − s_ref) and family size.

**Pairs.** For two models of different families, Δ_std is their difference on H(M), Δ_cf their difference on H(M \ (F_a ∪ F_b)), and Δ_ref their difference on H(M \ R″), R″ being |F_a ∪ F_b| models drawn at random outside both families, 30 draws.

- **Sign changes:** the share of cross-family pairs with Δ_std ≠ 0 whose Δ_cf has the opposite sign, against the same share for Δ_ref (mean over draws). The excess is the consequence attributable to who built the subset.
- **Direction:** Spearman correlation over cross-family pairs between (Δ_cf − Δ_std) and (n_a − n_b). If builders are penalised, the model from the larger family gains more when both families are left out, so the correlation is positive.

## 6. Inference

- **Family structure (primary test for T_family):** a permutation test on the fixed data. Family labels are shuffled over models 2,000 times, keeping the family sizes; the statistic is the mean over models of (s_fam − s_std). The p-value is the share of shuffles whose statistic is at least the observed one. Under shuffling a model's "family" is itself plus random others, which is exactly the self-left-out condition.
- **Sampling of tasks:** a bootstrap over tasks, 300 replicates, in which every subset is rebuilt on the resampled tasks; the random model draws are keyed and identical across replicates. Percentile 95% intervals for T_self, T_family, T_total, the excess share of sign changes and the two Spearman correlations.
- Models are fixed. One run per model: run-to-run variation is not in any interval.

## 7. How the result will be read (fixed in advance)

For T_self, T_family and T_total, in score points (1 point = 0.01):

- **material:** the bootstrap interval excludes zero and the estimate is at least 1 point;
- **detectable but small:** the interval excludes zero and the estimate is below 1 point;
- **not detected:** otherwise. Not detected is not evidence of absence; the interval is reported as the bound.

For the ranking consequence: **material** if the excess share of sign changes is at least 1% of cross-family pairs with an interval that excludes zero.

Both outcomes are reportable. A material dependence means the standard practice of ranking the builders on their own hard subset penalises well-represented families, and cross-fitting is the remedy. No material dependence means that with a panel this large the filter's dependence on any one family is bounded by the interval reported.

## 8. Sensitivity analyses (fixed)

- k = 74 and k = 296.
- The threshold rule itself: H(S) = tasks with a solve rate below 50% over S (the subset size then varies between conditions).
- The two alternative family definitions of Section 3.
- Instruct.

## 9. Limits stated in advance

- The real Hard panel was the leaderboard of mid-2024, not these 153 models, and the real rule has three more filters. This study measures the mechanism on the current tables, not the bias of the published subset.
- Family rules rest on recalled provenance; errors would blur families and bias T_family towards zero. The lexical definition is the check that needs no such knowledge.
- A model's own influence on a solve rate over about 150 models is small by arithmetic, so T_self is expected to be small.
- Related work to read in full before any citation: Phang et al., "Adversarially Constructed Evaluation Sets Are More Challenging, but May Not Be Fair" (arXiv 2111.08181), where filtering with a single adversary model penalises that model; Zheng and Yang (arXiv 2609.00482), family-based item selection with held-out halves.

## 10. Outputs

`experiments/scripts/panel_filter_study.py`; `experiments/results/panel_filter/` (summary.md, records.tsv, families.tsv, per-model and per-family tables). No manuscript file is changed.

## 11. Deviations

**1. The primary specification was degenerate; the published rule becomes the main rule (recorded 2026-11-15, after the prespecified run, so everything under this deviation is post hoc).**

- **What went wrong.** The design fixed the hard subset as the k = 148 tasks with the lowest solve rate, to hold the size constant. BigCodeBench has more than 148 tasks that almost no model solves, so that subset sits at the floor: the mean score on it is 0.23% on Complete and exactly 0 on Instruct, where every cross-family pair is tied and the pair analysis is undefined. Ties at a solve count of zero are then broken by task order. The primary results (Complete and Instruct at k = 148, and k = 74) describe unsolved tasks, not a hard subset like the published one, and are not to be used as the study's finding. The floor should have been checked before k was fixed; it was not.
- **What was seen before this deviation was written:** the whole prespecified run, including the planned sensitivity analysis under the published rule (solve rate below 50%; 616 tasks; mean score 17.7%) for model-line families on Complete: T_self +0.215, T_family +0.264, T_total +0.479 points; permutation p = 0.0005; excess share of sign changes +0.23 percentage points.
- **What is added.** Under the rule "solve rate below 50%": the task bootstrap (300 replicates) for model-line families on Complete and on Instruct, and point estimates with the permutation test for the lexical and organisation families on Complete. The Complete, model-line analysis keeps its random streams, so its point estimates are the ones already seen. The readings of Section 7 are applied unchanged.
- Outputs: `results/panel_filter/summary_deviation.md` and `records_deviation.tsv`.

## 12. Extension: the effect as a function of panel size

**Status:** specification fixed 2026-11-15, before any of it was computed. Justin asked for it on that day. **Exploratory:** it was added after the results of Sections 5 to 11 had been seen, on the same tables.

**Question.** The analyses above use a panel of about 150 builders, where one model moves a solve rate by under one point. Benchmarks are also filtered with a handful of models. How large is the penalty of taking part in the filter when the panel has p builders, and from what size is it below one point?

**Rule.** The published one: H(P) is the set of tasks whose solve rate over the panel P is below 50%. Nothing is fixed by rank, so the floor problem of deviation 1 cannot recur; the size of H(P) and the mean score on it are reported for every p as a check.

**Panels.** p in {4, 8, 16, 32, 64, 96}. For each p, 100 panels drawn uniformly at random without replacement from all models of the split. From each panel, every member is evaluated if p ≤ 16, otherwise 16 members drawn at random.

**Conditions for a member m** with g same-line models also in the panel (families as in Section 3, primary definition):

- **self:** m is replaced by r; **reference:** another member u, not of m's family, is replaced by r instead. r is drawn at random among models outside the panel and outside m's family, so the panel keeps its size and loses or keeps m.
- **self and siblings:** m and its g siblings are replaced by g + 1 such outside models; **reference:** g + 1 members not of m's family are replaced by the same outside models and m stays. If too few eligible models exist, the member is left out of this comparison and counted.

**Estimands, per p** (means over panels and members; scores of m on the subset built under each condition):

- T_self(p) = self − reference;
- T_total(p) = (self and siblings) − its reference; T_family(p) = T_total(p) − T_self(p) on the members that have both;
- **placebo:** for 10 models outside the panel, the change of their score when one member is replaced. It should be zero and checks the machinery;
- **ranking:** for a member against those 10 outside models, the share of non-zero differences on H(P) whose sign changes when the member is replaced; the same share for pairs among the 10 outside models is the baseline, and the excess is reported.

**Inference.** Task bootstrap, 100 replicates, with the panels and all random choices fixed across replicates; percentile 95% intervals. The readings of Section 7 apply (material: at least 1 point with an interval that excludes zero).

**Splits.** Complete, then Instruct as before.

**Stated in advance.** The penalty should fall as p grows, roughly in proportion to 1/p. Random panels are not how builders choose panels, which usually favour strong or popular models; a panel of strong models from few families would be penalised differently. Outputs: `scripts/panel_size_curve.py`, `results/panel_filter/summary_panel_size.md`, `records_panel_size.tsv`.

## 13. Corrections (2026-11-16)

Justin approved the corrections on 2026-11-16. They were written before `scripts/panel_size_curve.py` was run again. No estimate changes.

1. **The readings on ranking are withdrawn.** In the panel-size run, 7 of the 12 point estimates of the excess share of sign changes lie outside their own bootstrap intervals (Complete at 4, 8 and 16 builders; Instruct at 4, 8, 32 and 64). The task bootstrap of that share is biased, so its intervals are not shown any more and no reading is attached to them. The same holds for the reading "ranking consequence" of Section 7 as applied in `summary.md` and `summary_deviation.md`: it rests on an interval of the same kind and is withdrawn. The shares of sign changes stay as point estimates. Whether an order changes is to be judged for named pairs with simultaneous intervals, as in `MMLU_RECONSTRUCTION_DESIGN.md`, Section 11.
2. **One set of comparisons for the same-line columns** (Section 12). Own participation was averaged over every evaluated member, the same-line total over the members that have that comparison, so the difference of the two columns was not the same-line part. A column now gives own participation on the members that have the same-line total.
3. **The sign of own participation is an identity, not a finding** (Sections 5 and 12). One set of outcomes both votes and scores. The self and reference subsets then differ only on tasks where the votes common to both are at the threshold and the two swapped members disagree; the tasks that enter are ones the member solved and the tasks that leave are ones it failed. With e tasks entering, l leaving, n tasks in the self subset and s the member's score on the reference subset, the difference is [e(1 − s) + l·s]/n, which cannot be negative. It was derived here and checked on 34,800 comparisons. The placebo and the intervals of T_self therefore say nothing about its sign. What they report is its size. The identity does not cover scores from runs that did not vote.
4. **"Family" means a model line read from names** (Section 3). It is not proof of shared base weights.
