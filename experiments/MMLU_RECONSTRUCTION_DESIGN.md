# The MMLU-Pro filter on original MMLU — reconstruction study design

**Status:** design fixed 2026-11-15, before any per-question result on original MMLU was downloaded or seen. Justin asked for this study on that day, approving the larger download and the installation of a parquet reader.

**What is known beforehand.**

- The rule (arXiv 2406.01574, Section 3.2): eight models were run on MMLU and a question answered correctly by more than four was removed.
- The test on MMLU-Pro's own outputs (`MMLUPRO_FILTER_DESIGN.md`, 2026-11-15): penalty +1.57 points in the predicted direction, not detected by the primary placebo test (p = 0.14); +3.37 points and p < 0.005 when same-family systems are not used as controls; the two Llama-2 filter models carry it (+7.2 and +4.1), and two near-clones of filter models show the same deficit. That test was on ten-option questions with other prompts than the filter's.
- From repository listings only (names, sizes): the archive `open-llm-leaderboard-old` on Hugging Face holds per-question results of about 7,000 models on original MMLU from one harness. Seven of the eight filter models are there; Yi-6B-Chat is not.

No per-question outcome on original MMLU has been looked at.

## 1. Questions

1. **Penalty in the original format.** On the original four-option questions that MMLU-Pro kept, do the filter models score lower than systems of the same ability?
2. **Fidelity.** Do the public runs of the filter models reproduce the published filter, that is, does the number of filter models that are right on a question predict whether it was kept?
3. **Lineage.** Do models fine-tuned from a filter model share the penalty?

## 2. Data

- **Per-question results:** for each selected model, the latest run directory of its `details_*` repository that contains all 57 files `details_harness|hendrycksTest-<subject>|5_*.parquet` (five-shot). Correctness is the harness's accuracy metric for the row. A question is identified by subject and row position, checked against the row hashes where the files carry them.
- **Question texts:** `cais/mmlu`, configuration `all`, test split (14,042 questions expected).
- **Kept status:** an original question is *kept* if a question of MMLU-Pro with `src` = `ori_mmlu-<subject>` has the same text after lower-casing and collapsing whitespace, within the same subject. MMLU-Pro texts come from the outputs downloaded on 2026-11-15. Everything else is *not kept* (removed by the filter or by the later review; the two cannot be told apart).
- **Gates, checked before any accuracy is compared:**
  - at least 95% of MMLU-Pro's MMLU-origin questions must match an original question, otherwise the kept set is unreliable and the study stops at a descriptive report;
  - a model whose row count differs from `cais/mmlu` in any subject is dropped and listed.

## 3. Systems (fixed before any outcome)

- **Filter models (7 of 8):** meta-llama/Llama-2-7b-hf, Llama-2-7b-chat-hf, Llama-2-13b-hf, Llama-2-13b-chat-hf; mistralai/Mistral-7B-v0.1; google/gemma-7b; 01-ai/Yi-6B.
- **Clean pool (primary controls and placebo pool):** a list of models that are not derived from any filter model's weights, written from knowledge of provenance and fixed in `scripts/mmlu_reconstruction_fetch.py` (`CLEAN`). It holds base models of other lines (Falcon, MPT, Pythia, GPT-J, GPT-Neo, GPT-2, OPT, XGLM, BLOOM, StableLM, RedPajama, OpenLLaMA, Phi, Qwen, DeepSeek-LLM, InternLM, Baichuan, XGen, Cerebras-GPT, StarCoder, TinyLlama, Llama-1, Llama-3, Command-R) and **same-line models with other weights** (Llama-2-70b, Yi-34B, Yi-9B, Gemma-2b, Mixtral), which are flagged as siblings. Names absent from the archive are skipped and listed.
- **Random pool:** 120 repositories drawn at random from the archive listing (keyed generator on the sorted names), outside the sets above. It contains hidden derivatives of filter models, since most leaderboard entries are fine-tunes of Llama-2 or Mistral-7B.
- **Lineage pool:** 80 repositories drawn at random among those whose lower-cased name contains `llama-2-7b`, `llama2-7b`, `llama-2-13b`, `llama2-13b`, `mistral-7b`, `gemma-7b` or `yi-6b`, tagged by the substring matched.

## 4. Estimands and primary analysis

- **Subjects used:** those with at least 10 kept questions. Both accuracies of a system are standardised to the kept questions' subject mix (weights proportional to the number of kept questions per subject): **a(all)**, over every original question of the subject, and **a(kept)**, over the kept ones.
- **Controls of a filter model m:** the clean-pool systems with a(all) within 0.03 of a_m(all); if fewer than five, the five nearest.
- **Penalty:** P_m = [mean over controls of a(kept) − a_m(kept)] − [mean over controls of a(all) − a_m(all)], and **P = mean of P_m over the filter models.** Positive means the filter model does worse on the kept questions than equally able systems.
- **Prediction, written in advance:** P > 0 and larger than on MMLU-Pro, because the questions and format are the filter's own; largest for the four Llama-2 models, which supplied half the votes. If the public runs differ much from the filter's runs (Section 5), P shrinks.

**Inference.** Placebo test as in `MMLUPRO_FILTER_DESIGN.md`, Section 4: every clean-pool system inside the filter models' range of a(all), widened by 0.03, gets a placebo penalty with itself left out of the controls; the p-value is the share of subsets of the pool, of the size of the filter set, whose mean is at least P (exact up to 200,000 subsets, otherwise 200,000 random ones). A bootstrap over questions within subject (1,000 replicates, control sets fixed) gives intervals. **Reading:** confirmed if P is at least 1 point, placebo p is below 0.05 and the interval is above zero; detectable but small if the tests pass below 1 point; not detected otherwise.

**Sensitivity, fixed:** siblings removed from the clean pool; the random pool in place of the clean pool; no subject standardisation.

## 5. Secondary analyses (fixed)

1. **Fidelity of the public runs.** For each original question, c = how many of the seven available filter models are right. Table of the share kept by c. With seven of eight models, the rule implies kept for c ≤ 3 whatever the eighth did, and removed for c ≥ 5; c = 4 depends on the eighth.
2. **Lineage.** Placebo penalties of the lineage pool by tag, against the clean pool without siblings.
3. **By filter model.**
4. **Leave-one-out on the real panel.** With the seven public runs and the rule "kept if at most three of seven are right" (the eighth model is missing, so this is a seven-model panel, not the published one): for each member, the change of its score on the kept set when it is replaced by the clean-pool system nearest in a(all). This repeats `PANEL_FILTER_DESIGN.md`, Section 12, on MMLU with the real builders.

## 6. Limits stated in advance

- The public runs are not the filter's runs: same weights, another harness and prompt. Section 5.1 measures how far apart they are.
- Yi-6B-Chat is missing. Running it, or all eight with one harness, on HPC would close that gap; it needs model licences accepted by Justin and is not part of this design.
- "Not kept" mixes the filter with the later expert review.
- The clean list rests on recalled provenance. A derivative wrongly listed as clean would carry part of the penalty and bias P towards zero.
- One harness, one run per model.

## 7. Outputs

`experiments/scripts/mmlu_reconstruction_fetch.py`, `experiments/scripts/mmlu_reconstruction_test.py`; `experiments/results/mmlu_reconstruction/` (raw/ with a manifest and the selection lists, acceptance.txt, summary.md, records.tsv). No manuscript file is changed.

## 8. Deviations

Recorded 2026-11-15, from file structure and question texts only. No accuracy of any model had been read or compared when these were written.

1. **Download size.** About 22 MB per repository, so about 6 GB in total, against the 1 to 3 GB stated to Justin before the download.
2. **Question identity is the text, not the row position.** The detail files are not in the row order of `cais/mmlu` (54 of 14,042 rows sit at the same position); the order is the same in 200 of the 201 repositories checked. Every detail row carries the field `example`, which equals the question followed by `A.` to `D.` and `Answer:`, and all 14,042 rows of `cais/mmlu` are matched exactly by this text. Rows are therefore mapped to `cais/mmlu` by `example`, repeated texts in order of occurrence. The row-count gate becomes: every subject must hold exactly the rows of `cais/mmlu`.
3. **Run selection.** The latest complete run of `meta-llama/Llama-2-7b-hf` (2023-09-15) holds 10 questions per subject, a debug run, and fails the row-count gate. The rule becomes, for every repository: the latest complete run that passes the row-count gate; a repository with none is dropped and listed. The earlier run 2023-08-29 of this repository was fetched into `raw/details_alt/` with `scripts/mmlu_reconstruction_fetch_alt.py` (26.4 MB, listed in `raw/manifest_alt.tsv`); it passes the row-count gate and is the run used. No other repository needed an earlier run.

4. **Five filter models, not seven.** The repositories of `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf` hold per-question records for DROP, GSM8K and Winogrande only, none for MMLU. The statement above that seven of the eight filter models are in the archive was made from repository names and was wrong. No re-upload of the same weights is in the archive under a name that identifies it (the nearest names are a quantised copy and fine-tunes), so no stand-in is used. Five filter models remain: Llama-2-7b-hf, Llama-2-13b-hf, Mistral-7B-v0.1, gemma-7b and Yi-6B. Fixed before any accuracy was read:
   - P is the mean over these five. The prediction "largest for the Llama-2 models" now concerns two models.
   - Section 5.1: c counts five models. Under the published rule a question with c = 5 was removed whatever the three missing models did, and a question with c ≤ 1 was kept by the filter; for c from 2 to 4 the outcome depends on the missing models.
   - Section 5.4: a five-model panel with the rule "removed if more than half of the panel is right", which is the published rule at eight models and the rule of Section 5.4 at seven; at five, a question is kept if at most two are right. As in `PANEL_FILTER_DESIGN.md`, Section 12, the penalty reported is the member's score on the kept set built without it minus its score on the kept set built when another member, not of its line, is replaced by the same system. The plain change against the original panel is reported beside it.
5. **Repositories without MMLU records.** Ten further selected repositories hold no MMLU details and are dropped: two of the clean pool (falcon-40b-instruct, internlm-20b), one sibling (Llama-2-70b-chat-hf), one of the lineage pool and six of the random pool. Thirteen names of the clean list are not in the archive (listed in `raw/selection.tsv`).

**Added, not in the design** (decided 2026-11-15 before any accuracy was read; labelled as additions in the outputs):

- A reference for Section 5.1: the same table for a panel of five clean-pool systems matched to the filter models on a(all) (for each filter model the nearest one not yet used). Any model's errors predict which questions are hard, so without this reference the table cannot separate the filter's footprint from plain difficulty.
- For Section 5.2, the lineage systems inside the filter models' range of a(all), widened by 0.03, are compared with the clean pool without siblings by a one-sided Mann–Whitney test. The lineage groups are larger than the pool, so the subset test of Section 4 is undefined for them.

Notes on the data, same date:

- The MMLU-Pro question table saved on 2026-11-15 holds no question texts. As Section 2 says, texts come from the output archives: for each question id, the most frequent text among the records whose `src` equals the table's.
- 87 rows of `cais/mmlu` share their question text with another row of the same subject (27 of them also share the choices). Kept status is defined on the question text, so such rows share it.
- 56 of the 57 subjects occur as `ori_mmlu-<subject>` in MMLU-Pro (6,810 questions); `moral_scenarios` has no kept question.
- Gates, run before any accuracy was compared: all 6,810 MMLU-origin questions of MMLU-Pro equal an original question of the same subject (100%). MMLU-Pro holds 286 question texts under two or more ids of one source subject (576 ids, mostly consecutive, with different option sets), so the 6,810 ids are 6,520 distinct texts; they mark 6,580 original rows as kept (46.9%). 340 ids have two text versions across the output archives; the more frequent one is used.
- 268 systems pass the row-count gate: 5 filter models, 70 of the clean pool (8 siblings), 79 of the lineage pool and 114 of the random pool.

## 9. Run

One run, 2026-11-15, after the gates: `python -E -P -X utf8 scripts/mmlu_reconstruction_test.py results/mmlu_reconstruction/raw results/mmlupro_filter/raw results/mmlu_reconstruction`. Results in `results/mmlu_reconstruction/summary.md`; the correctness matrix of the 268 systems is saved as `correct_matrix.npz` for later analyses.

## 10. Extension: the penalty as a function of panel size on MMLU

**Status:** specification fixed 2026-11-15, before any of it was computed. Justin asked for it on that day. **Exploratory:** it was added after the results of Section 9 had been seen, on the same matrix (`results/mmlu_reconstruction/correct_matrix.npz`). No download.

**Question.** `PANEL_FILTER_DESIGN.md`, Section 12, found on BigCodeBench that taking part in an outcome-based filter costs a builder about 0.3/p of score with p builders. Does the same law hold on original MMLU, a second benchmark with other models and a four-option format, and with what constant?

**Rule.** MMLU-Pro's rule for a panel of any size: a question is removed if more than half of the panel is right, so the kept set H(P) holds the questions on which at most half of the panel P is right. Sensitivity: the rule of the BigCodeBench study, kept if fewer than half are right. The size of H(P) and the mean score of all systems on it are reported for every p.

**Populations.**

- **A (primary): the 5 filter models and the 70 clean-pool systems**, whose provenance is known. p in {4, 8, 16, 32, 64}; at p = 64 the panels share most of their members, since the population has 75 systems.
- **B (secondary): all 268 systems**, p in {4, 8, 16, 32, 64, 96}. Most of them are fine-tunes whose base cannot be read from the name, so near-copies of a member stay in the panel when it is replaced. Only the own-participation estimand is reported for B.

**Families** (model line, by the first rule that matches the lower-cased name; written from knowledge of provenance, before any outcome of this extension):

| Family | Rule on the name |
|---|---|
| tinyllama, openllama | contains `tinyllama`; contains `open_llama` |
| llama | contains `llama-2`, `llama2` or `meta-llama-3`, or starts with `huggyllama__` |
| stablelm | contains `stablelm` |
| mistral | contains `mistral` or `mixtral` |
| gemma, qwen, falcon, deepseek, redpajama, cerebras, starcoder, command-r | contains that word |
| yi | starts with `01-ai__` or contains `yi-6b` or `yi-34b` |
| phi | starts with `microsoft__phi` |
| mpt, opt | contains `mpt-`; contains `__opt-` |
| pythia | contains `pythia` or `dolly` (Dolly v2 is fine-tuned from Pythia) |

A lineage-pool system takes the family of its tag (llama, mistral, gemma, yi). A name that matches no rule forms its own family.

**Panels, conditions and estimands** are those of `PANEL_FILTER_DESIGN.md`, Section 12, unchanged: 100 random panels per size; every member evaluated if p ≤ 16, otherwise 16 members drawn at random; **self** (the member m is replaced by a system r from outside the panel and outside its family) against **reference** (another member, not of m's family, is replaced by r); **self and siblings** against its reference; placebo for 10 systems outside the panel; share of sign changes between a member and those 10 systems against the baseline among the 10. Scores are plain accuracy on the kept set.

**Inference.** Question bootstrap within subject, 100 replicates, with the panels and every random choice fixed; percentile 95% intervals. Reading as before: material if at least 1 point with an interval that excludes zero. The sign-change intervals were biased in the BigCodeBench study, so only point estimates of the sign changes are to be read.

**Stated in advance.**

- The penalty falls roughly as 1/p. The five public builders gave 8.3 points at p = 5 (Section 9), that is, penalty × p of about 41. If that constant carries over to population A: about 10 points at 4 builders, 5 at 8, 2.6 at 16, 1.3 at 32 and 0.6 at 64. Random panels from the clean pool mix systems at chance level with far stronger ones, unlike the real panel, so the constant may differ.
- In population B the penalty at the same p should be smaller, because near-copies of a member remain in the panel.
- The placebo should be near zero.

**Outputs.** `scripts/mmlu_panel_size_curve.py`; `results/mmlu_reconstruction/summary_panel_size.md`, `records_panel_size.tsv`, `families_panel_size.tsv`.

**Run.** One run, 2026-11-15: `python -E -P -X utf8 scripts/mmlu_panel_size_curve.py results/mmlu_reconstruction`. No member was skipped and nothing deviated from this section.

**Against what was stated in advance.**

- *Form.* Under the sensitivity rule, penalty × p is 66 to 70 at every size, so the penalty is proportional to 1/p. Under MMLU-Pro's rule it is 39 at 4 builders, 52 at 8, 61 at 16 and 66 at 32 and at 64: the 1/p form holds from about 16 builders, not below.
- *Constant.* The constant of 41 did not carry over. The value at 4 builders agrees (9.8 points against 10). At 8 to 64 builders the penalties are 6.5, 3.8, 2.1 and 1.0 points, 1.3 to 1.6 times what a constant of 41 gives.
- *Population B.* Smaller at every size, as stated: 9.0, 5.5, 2.9, 1.5 and 0.75 points at 4 to 64 builders and 0.51 at 96; penalty × p is 36 to 49.
- *Placebo.* Within 0.1 point of zero under MMLU-Pro's rule in both populations, and −0.18 at 4 builders under the sensitivity rule, as stated. Several of its intervals exclude zero; the intervals reflect the sampling of questions only, with the 100 panels fixed.

**Post hoc (not in the specification; descriptions of the same run).**

- MMLU-Pro's rule keeps ties, so at 4 builders it keeps 65% of the questions against 39% under the sensitivity rule. Penalty × p × share of questions kept is alike under the two rules at every size: 25 and 26 at 4 builders, 31 and 29 at 8, 33 and 31 at 16, 34 and 33 at 32, 33 and 33 at 64. The lower constant of MMLU-Pro's rule at small panels follows from how many questions the rule keeps.
- With same-line builders counted, the penalty does not fall as 1/p in population A: 9.9, 7.2, 4.8, 3.3 and 1.7 points at 4 to 64 builders (times p: 40, 57, 77, 106 and 107), because a larger panel holds more builders of one line (population A has 12 Qwen and 11 Llama systems among 75).

## 11. Extension: the order of systems on all questions and on the kept questions

**Status:** specification written 2026-11-15 for Justin's review and approved by him unchanged on the same day, before any of it was computed. **Exploratory:** it comes after Sections 9 and 10, on the same matrix (`results/mmlu_reconstruction/correct_matrix.npz`). No download.

**What had been looked at before this section was written.** Every system's a(all) and a(kept) and the penalties of Section 9, including that Llama-2-13b has a lower a(kept) than three clean-pool systems it leads on a(all). The precision check below. No pairwise interval, no count of sign changes and no random-subset reference had been computed.

**Questions.**

1. **Order.** Between all questions and the kept questions, how many pairs of systems change sign, is that more than a random subset of the same size gives, and is any reversal established?
2. **The two parts.** Do the kept and the removed questions order some pairs oppositely?
3. **Builders.** Do the filter models, and the models fine-tuned from them, lose places on the kept questions relative to systems of the same ability?

**Scores and regions.** As in Section 4: the subjects with at least 10 kept questions, every score standardised to the kept questions' subject mix. Three regions: all original questions (V), the kept ones (O) and the removed ones (U = V without O). For two systems, Δ_R is the difference of their scores on region R; the pair is oriented so that Δ_O ≥ 0, and a pair with Δ_O = 0 does not enter sign counts.

**Populations.**

- **A (primary):** the 5 filter models and the 70 clean-pool systems; 2,775 pairs.
- **K (secondary):** A and the 79 lineage-pool systems, the 154 systems of known provenance; 11,781 pairs.
- **B (description only):** all 268 systems; 35,778 pairs.

**Decision language** (that of `SUBSET_C_DESIGN.md`, Section 5, with one label added):

- **Resolved** on a region: the 95% interval of Δ_R excludes zero.
- **Reversal:** Δ_O and Δ_V are both resolved, with opposite signs.
- **Newly resolved:** Δ_O unresolved and Δ_V resolved. **Lost:** Δ_O resolved and Δ_V unresolved.
- **Opposite parts (added):** Δ_O and Δ_U are both resolved, with opposite signs. This is weaker than a reversal: the two parts disagree, and all questions together side with one of them.

Every label is reported under unadjusted per-pair intervals and under simultaneous intervals. Wording of the outcome: if no reversal is established, the text says so and adds whether the sign disagreement is within what same-size random subsets give. It never says that a ranking holds or transfers.

**Uncertainty.** Systems are fixed; questions are the sampling unit. Bootstrap over questions within subject, 2,000 replicates, the same replicates for every system, pair and region; percentile intervals. Simultaneous intervals as in `SUBSET_C_DESIGN.md`, Section 6: the bootstrap maximum of |t| over all pairs of the population and the two statistics of a label (Δ_O and Δ_V for reversal, newly resolved and lost; Δ_O and Δ_U for opposite parts), with the interval estimate ± q × SD and q the 95th percentile of the maximum. Each population has its own families. Run-to-run variation is not in the intervals: each system has one run.

**Reference: random subsets.** 2,000 subsets of the questions with the same number per subject as the kept set. For each: Kendall's τ_b between the scores on the subset and on all questions; the number of pairs with opposite signs on the subset and on all questions; the number with opposite signs on the subset and on its complement. **Reading, fixed in advance** (`SUBSET_C_DESIGN.md`, Section 7): a value of the kept set inside the central 95% of the reference means that, on that statistic, the kept set is not distinguishable from a random subset of its size. Reported for A and B.

**Builders.**

- **Opponents:** the clean pool, without the system itself.
- **Places lost** by a system x: L_x = (opponents ahead of x on the kept questions) − (opponents ahead of x on all questions). A tie counts one half. Negative means places gained.
- **Against equally able systems:** for a filter model m with the controls of Section 4, N_m = L_m − mean of L over its controls, and **N = mean of N_m over the filter models.** Positive means that the filter model loses more places than systems of the same a(all).
- **Inference as in Section 4:** every clean-pool system inside the filter models' range of a(all), widened by 0.03, gets a placebo N with itself left out of its controls; the p-value is the share of subsets of that pool, of the size of the filter set, whose mean is at least N (exact). Bootstrap intervals with the control sets fixed.
- **Reading:** confirmed if N is at least 1 place, the placebo p is below 0.05 and the interval is above zero; detectable but small if the tests pass below 1 place; not detected otherwise.
- **By filter model,** and **by line** for the lineage pool: the placebo N of each lineage system against the clean pool without siblings; mean, median, share above the placebo pool's maximum, and a one-sided Mann–Whitney test against the pool.
- **Direction among cross pairs** (a filter or lineage system with a clean-pool system; 350 pairs in A, 5,880 in K): of the pairs with opposite signs on all and on kept questions, how many have the filter-line system falling behind and how many have it moving ahead; the same split for reversals and for opposite parts.

**Sensitivity, fixed.** For question 1 in population A: plain accuracy without standardisation (all 14,042 questions, uniform random subsets of 6,580). For question 3: siblings removed from the opponents and the controls.

**Precision, checked before fixing** (2026-11-15, `scripts/mmlu_pairs_precision_check.py`; it reads no order outcome).

- Population A: the standard error of a gap is 0.56 points on all questions and 0.75 on the kept ones (medians), and the simultaneous critical value is about 4.3. A gap is resolved from about 2.4 points on all questions and 3.2 on the kept ones, so **an established reversal needs the gap to change by about 5.6 points** (4.7 to 6.1 across pairs). With unadjusted intervals: 2.6 points.
- Population K: critical value about 4.6, change needed 5.8 points (4.6 to 6.5). Population B: about 4.8, 6.0 points (4.6 to 6.8).
- Clean-pool systems within 6 points below and above each filter model on all questions: Llama-2-7b 7 and 6; Llama-2-13b 3 and 5; Mistral-7B 5 and 9; gemma-7b 3 and 12; Yi-6B 6 and 9. Lineage systems: 5 below (1 to 22) and 5 above (3 to 10), medians and ranges. Places can therefore be lost and gained; the count is coarse.

**Stated in advance.**

- **Reversals between all and kept questions under simultaneous intervals: none or very few, whatever the truth.** The largest penalty of Section 9 is 5.4 points (Llama-2-13b), at the edge of what can be established, and three clean-pool systems lie within 6 points below it. A count of zero would not show that the order is unaffected. The quantities with power are the sign changes against the random-subset reference and the places lost.
- Sign changes between all and kept questions: more than the random-subset reference gives.
- Places lost: N > 0, carried by the two Llama-2 models; the Llama-2 fine-tunes lose places and the Mistral-7B fine-tunes do not. Section 9 gave their penalties, so this part is a translation into places, not a new test of the penalty.
- Opposite parts: established for some pairs, mostly with a Llama-2-line system ahead on the removed questions and behind on the kept ones. Between the removed and the kept questions the change of a gap is about twice the change between all and kept.

**Limits.**

- "Kept" mixes the filter with the later review (Section 6).
- The records are archived runs in the original four-option format, not MMLU-Pro's ten-option format.
- All questions contain the kept ones, so the two orders are not independent. The bootstrap carries that dependence.
- The opposite-parts label was added after the precision check showed that reversals can hardly be established here. By the filter's construction a builder's right answers sit more often among the removed questions, so opposite parts are expected for builders.
- Places depend on how many opponents are near. Controls within 3 points share that.
- One run per system. The section is informed by the results of Sections 9 and 10.

**Outputs.** `scripts/mmlu_pairs_test.py`; `results/mmlu_reconstruction/summary_pairs.md`, `records_pairs.tsv`, `pairs_A.tsv`, `pairs_K_cross.tsv`, `places.tsv`. No manuscript file is changed.

**Details fixed while the script was written, before the run** (2026-11-15; from question counts and system lists only):

- The 56 used subjects hold 13,147 questions, 6,580 kept and 6,567 removed. Every used subject has at least 16 removed questions, so the score on the removed questions is defined in every subject.
- In a bootstrap replicate the subject weights follow the replicate's own kept counts, as in Section 4.
- By line, as in Section 9: only lineage systems inside the filter models' range of a(all), widened by 0.03, are counted, and the number outside is given. The lines are those of the tags (Llama-2-7b 11 systems, Llama-2-13b 21, Mistral-7B 45, gemma-7b 2).
- The sensitivity without standardisation resamples questions within subject over all 57 subjects, as in Section 4.
- The script is first run on a synthetic matrix of random outcomes to find coding errors. That run reads no real record.

**Run.** One run, 2026-11-16, after the test on the synthetic matrix: `python -E -P -X utf8 scripts/mmlu_pairs_test.py results/mmlu_reconstruction`. a(all) and a(kept) agree with `systems.tsv` of Section 9 within 5e-7. No deviation.

**Results by the fixed readings** (`results/mmlu_reconstruction/summary_pairs.md`).

- **Order, population A.** No reversal is established under simultaneous intervals (36 under unadjusted intervals); more pairs change sign than same-size random subsets give: 261 of 2,773 against 16 to 39. Kendall's τ_b between the scores on kept and on all questions is 0.81 against 0.97 to 0.99. Without standardisation: 262 against 18 to 41, none established.
- **Order, populations K and B.** 36 reversals are established in K and 67 in B. In B 2,610 of 35,765 pairs change sign, against 287 to 443.
- **Opposite parts** under simultaneous intervals: 4 in A, 494 in K, 874 in B.
- **Builders.** N = +5.31 places (2.37 to 6.34), placebo p = 0 of 53,130 subsets: confirmed by the fixed reading. By model: Llama-2-7b +23.3 (9.7 to 28.9), Llama-2-13b +3.0 (1.0 to 4.8), Mistral-7B 0.0, gemma-7b +0.3, Yi-6B 0.0. Without siblings: +4.96 (2.22 to 6.03), p = 0 of 42,504.
- **By line.** Llama-2-7b fine-tunes +26.1 places (median +33.1, 80% above the pool maximum, p = 2.2e-5); Llama-2-13b +1.3 (median +1.0, p = 0.0036); Mistral-7B −0.5; gemma-7b −0.5.
- **Direction.** In A, 28 cross pairs change sign, all with the filter model falling behind (25 for Llama-2-7b, 3 for Llama-2-13b). In K, 375 fall behind and 28 move ahead; the 31 reversals established among cross pairs all have a system of the Llama-2-7b line falling behind a clean-pool system.

**Against what was stated in advance.**

- *"Reversals: none or very few, whatever the truth."* Right for A (none). **Wrong for K (36) and B (67).** The statement rested on the penalty against equally able systems (at most 5.4 points) and on Llama-2-13b. It left out pairs of unequal ability. Near-copies of Llama-2-7b-chat score 23.0 to 23.5% on the kept questions, 3 to 4 points below systems that guess, and lead those systems by about 19 points on all questions.
- *Sign changes above the reference:* as stated.
- *Places lost carried by the two Llama-2 models, Llama-2 fine-tunes losing places, Mistral-7B fine-tunes not:* as stated. Not foreseen: the size for Llama-2-7b, 23 places from a penalty of 2.0 points (Section 9).
- *Opposite parts:* as stated for the Llama-2 lines. Of the 96 established among cross pairs in K, 92 have the filter-line system ahead on the removed questions.

**Post hoc descriptions** (rules written 2026-11-16, after the run above and before any of them was computed; `scripts/mmlu_pairs_posthoc.py`, which recomputes the labels with the same generator keys and checks them against `records_pairs.tsv`).

What prompted them, seen in `places.tsv`: the 29 clean-pool systems lowest on all questions (24.5 to 40.6%) all score between 24.2 and 27.5% on the kept questions, close to the 25% of guessing among four options. On the kept questions the lower part of the field is therefore not separated from guessing, and places there turn on differences of one or two points.

1. **The floor.** The share of each answer letter among all and among the kept questions (what a system that always gives one letter scores). The systems whose bootstrap interval of a(kept) lies below 25%, by set and line.
2. **Sign changes by floor status.** A system is above the floor at a cut c if a(kept) ≥ c. Three cuts are reported, 28%, 30% and 35%, since any single cut is arbitrary. For A, K and B and each cut: the pairs of two systems above the floor, how many of them have opposite signs on kept and on all questions, the reference from the same 2,000 random subsets restricted to those systems, τ_b with its reference, and the reversals among them under the labels already computed.
3. **The established reversals.** Every reversal under simultaneous intervals in K and in B is listed (`pairs_reversals_simultaneous.tsv`), with the set, line and scores of the two systems; counts of the distinct systems that fall behind and of those that lead on the kept questions.

**Post hoc results** (2026-11-16; `results/mmlu_reconstruction/summary_pairs_posthoc.md`; the recomputed labels agree with the run).

- *Floor.* All 31 clean-pool systems below 40% on all questions score 24.2 to 27.3% on the kept ones; the 9 between 40 and 50% score 26.1 to 32.1%. A system that always gives one letter scores 23.4 to 26.6% on the kept questions, so a score below 25% is not by itself below guessing. A is the key of 26.6% of the kept questions and of 19.1% of the removed ones (untested observation: the panel seems to have chosen A less often).
- *Population A above the floor.* With both systems at 30% or more on the kept questions (35 systems, 595 pairs), 7 pairs change sign against a reference of 1 to 8, τ_b is 0.977 against 0.973 to 0.997, and there is no reversal even under unadjusted intervals. At the 35% cut: 7 against 1 to 8. At the 28% cut: 12 against 1 to 8. Of the 261 sign changes in A, 249 to 254 involve a system below the cut.
- *Populations K and B above the floor.* More sign changes than the reference at every cut: in B at 35%, 617 of 12,090 pairs against 167 to 304; in K at 35%, 145 of 3,240 against 33 to 73. Reversals established with both systems above the cut: 5, 3 and 0 in K and 11, 7 and 0 in B at the cuts 28%, 30% and 35%.
- *The established reversals.* In all 36 of K the system that falls behind is a fine-tune or near-copy of Llama-2 (7 systems of the Llama-2-7b line, 5 of the Llama-2-13b line). In B it is so in 66 of 67 (16 of 17 systems; the other is a random-pool system). The systems that lead on the kept questions score 26 to 38% there and 23 to 50% on all questions. Near-copies of Llama-2-7b-chat score about 45% on all questions and 23.0 to 23.6% on the kept ones, and are last of 71 there.

**Reading** (mine; exploratory).

1. The kept questions do not separate the lower half of the field from guessing. In population A the excess of sign changes over the reference lies there; above the floor the sign disagreement is within what same-size random subsets give.
2. The orders that are established to reverse are those of the Llama-2 line, which held four of the eight votes: its fine-tunes fall to the level of guessing on the kept questions, some below every system that guesses.
3. N is confirmed by the fixed reading, but 23 of its places come from one model landing among the systems at the floor, where a point or two decides twenty places. Places are a poor scale here; the penalty in points (Section 9) is the better one.
4. The cuts that define the floor were chosen after the run. Three are reported, and the conclusion for population A needs the 30% or the 35% cut.

## 12. Corrections (2026-11-16)

Justin approved the corrections on 2026-11-16. They were written before the scripts were run again. No estimate changes: the scripts are run again to change wording and to add one column, and their records are compared with the earlier ones.

1. **The placebo share is not a calibrated p-value** (Sections 4, 9 and 11). The share of subsets of the placebo pool whose mean is at least the observed value says where that value lies among those subsets. It controls no error rate: the filter models were not assigned at random and are not exchangeable with the pool systems, the pool systems are not independent of each other, and the controls are not matched again for each subset. Complete enumeration makes the share exact as a count and nothing more. The summaries now print it as a reference tail share, "k of n subsets". "Confirmed" in Sections 9 and 11 is the label that the rule fixed in advance gives. The evidence is the size of the estimate, its question-bootstrap interval, and where each filter model lies among the pool.
2. **Lineage** (Sections 9 and 11). Fine-tunes of one base are not independent of each other, so the Mann–Whitney p-values have no independent-sample reading. They stay in the tables as a description of how far two sets of numbers are apart. A "line" is a model line read from names, which is not proof of shared base weights.
3. **One set of comparisons for the same-line columns** (Section 10). Own participation was averaged over every evaluated member, the same-line total over the members that have that comparison (400 and 388 at 4 builders, 1,600 and 1,321 at 64), so the difference of the two columns was not the same-line part. A column now gives own participation on the members that have the same-line total.
4. **The sign of the same-outcome penalty is an identity** (the leave-one-out of Section 9 and Section 10). One set of outcomes both votes and scores there. The self and reference sets then differ only on questions where the votes common to both are at the threshold and the two swapped members disagree; the questions that enter are ones the member got right and those that leave are ones it got wrong. With e questions entering, l leaving, n questions in the self set and s the member's score on the reference set, the difference is [e(1 − s) + l·s]/n, which cannot be negative. The placebo and the intervals of those sections therefore say nothing about the sign of the penalty. What they report is its size. The primary analysis of Section 4 and the whole of Section 11 score runs that did not vote on the released kept set, so the identity does not apply to them; there a filter model can gain, and Yi-6B does.

## 13. Extension: a difficulty-matched reference for the order of Section 11

**Status:** specification written 2026-11-16 for Justin's review; revised the same day, before any order statistic on matched subsets was computed (Revision 1, at the end of this section). **Exploratory.** Nothing of it has been computed except the matching check below. That check reads no order outcome, but it uses the scored halves' scores on the kept set and the expected scores on matched subsets, and the estimators and strata were chosen from it. The analysis is therefore tuned on evaluation-side levels and then fixed: blind to the order statistics, not to the evaluation side. Same matrix (`results/mmlu_reconstruction/correct_matrix.npz`); no download.

**Why.** Section 11 compares the kept set with random subsets that have the same number of questions per subject. The kept set is harder than those subsets (by about 8 points for the clean-pool systems, matching check below), so its excess of sign changes can come from difficulty alone: on any hard subset the lower part of the field falls to guessing. Zheng & Yang (arXiv 2609.00482 v2, read in full 2026-11-16) match their random subtests on item group and easiness strata. This section adds that match, and a match on the answer key, to tell the effect of hard questions apart from the effect of the panel's choice among questions of the same difficulty.

**What had been looked at before this section was written.** All results of Sections 9 to 11, including the sign changes by floor status (Section 11, post hoc). The matching check below. No sign change, τ_b or other pairwise statistic has been computed on a difficulty-matched subset or on the pair sets defined here.

**Question.** For the pairs of Section 11, are the sign changes between all and kept questions more than random subsets give that have the same subject composition (R0, the reference of Section 11), the same subject and difficulty composition (R1), and the same subject, difficulty and answer-key composition (R2)? Reported by pair type. Population B is left out: its random-pool systems are of unknown lineage, mostly fine-tunes of the filter models' lines.

**Difficulty without circularity.**

- *Halves.* The 70 clean-pool systems are split into two halves by model line (the name rules of Section 10; every line in one half), balancing the number of systems, the number below 40% on all questions and the number of eligible estimators (greedy over lines in order of size, keyed tie-breaks). Half 1: 36 systems (cerebras, command-r, llama, mpt, gpt-j, xglm, gpt2-xl, phi, pythia, redpajama, stablelm, yi). Half 2: 34 systems (deepseek, falcon, gemma, mistral, openllama, opt, gpt-neo, bloom, qwen, starcoder, tinyllama).
- *Estimators.* In each half, the clean-pool systems whose line holds no filter model (not llama, mistral, gemma or yi by the name rules, which also leaves out Llama-1 and Llama-3, by name rather than by provenance) and that score at least 40% on all questions: 10 in half 1, 16 in half 2 (11 of them Qwen). Systems below 40% are left out because on hard questions they answer near guessing and add noise (matching check).
- *Folds.* Fold 1: difficulty from the estimators of half 1; the systems scored are the five filter models, all of half 2 and, for population K, the 79 lineage systems. Fold 2: the reverse. No system is scored on strata that its own outcomes helped to form. A question's difficulty is the share of the fold's estimators that answer it right.
- *Strata.* Within each used subject, questions are ordered by difficulty (ties broken by a keyed random number) and cut into near-equal strata. R1: 8 strata per subject. R2: the same 8 strata, each split by answer key, so that R1 and R2 differ only by the match on the key (Revision 1; R1 had 16 strata). A cell is a subject × stratum (× key). Difficulty from 10 binary outcomes takes at most 11 values and from 16 at most 17, so a stratum boundary can fall inside a group of tied questions, and the tie-break then decides their stratum (diagnostics and sensitivity below).

**References.** In each fold, 2,000 subsets drawn independently: in every cell as many questions as the kept set has there, uniformly without replacement. R0 is the 2,000 subsets of Section 11 (same generator key), evaluated on the same pairs. Subset i of fold 1 and subset i of fold 2 together give reference value i. Every subset has the kept set's number of questions per subject, so scores are standardised exactly as in Section 11, and the complement of a matched subset has the removed set's composition.

**Pairs and folds** (population A: 5 filter and 70 clean-pool systems; K adds the 79 lineage systems):

| Pair type | Pairs | Scored in |
|---|---:|---|
| clean–clean, same half | 1,191 | one fold |
| clean–clean, different halves | 1,224 | none: not evaluated |
| filter–clean | 350 | one fold |
| filter–filter | 10 | both folds |
| (K) lineage–clean | 5,530 | one fold |
| (K) lineage–lineage and lineage–filter | 3,476 | both folds |

A pair scored in both folds counts with the mean over the two folds of its indicator. The kept set's count does not depend on the fold.

**Statistics.** For each pair type: the number of pairs with opposite signs on the subset and on all questions, and on the subset and its complement (for the kept set: kept and removed), as in Section 11. For populations A and K: Kendall's τ_b between the scores on the subset and on all questions, per fold over the systems scored in that fold, averaged over the two folds.

**Readings, fixed in advance** (revised in Revision 1). As in Section 11, statistic by statistic, a kept-set value lies inside, above or below the central 95% of a reference.

- *Inside a reference:* on that statistic and at this precision, the kept set is compatible with random subsets of that composition; no excess beyond them is detected. This does not show that difficulty, or difficulty and the key, account for the whole change.
- *Above:* more than random subsets of that composition give.
- *Below:* fewer than they give.

No monotone relation between the difficulty of a subset and its number of sign changes is assumed. Two systems can change order more than once as the questions get harder: 80% against 70% on easier questions, 40% against 50% on harder ones, 30% against 20% on the hardest. The matched subsets stay up to about 2 points easier than the kept set for the scored clean-pool systems above 40% on all questions (matching check). That residual is reported as unresolved matching bias and is not read as a bound in either direction.

For the sign changes between subset and all questions, by pair type:

- *above R0, inside R1 and R2:* no excess is detected beyond random subsets of the same subject and estimated-difficulty composition;
- *above R1, inside R2:* an excess beyond the difficulty match that is not detected once the answer-key mix is also matched (a mix that is itself a product of the panel's choice);
- *above R2:* a residual difference remains after matching on subject, estimated difficulty stratum and answer key. It is not attributed to the panel: the error of the difficulty estimates, item discrimination, content and the later expert review (the kept set mixes the filter with the review) all remain possible sources;
- any other combination is described as it is.

The share of the excess over the R0 median that the R1 and R2 medians reach says how far each matching condition reduces the observed excess, and nothing about conditions that are not matched. These readings are given for clean–clean and filter–clean pairs (A) and for lineage–clean and lineage–lineage-or-filter pairs (K). The ten filter–filter pairs are described only. Wording as in Section 11: never that a ranking holds or transfers, and never that difficulty explains a change that is only not detected beyond a reference.

**Descriptions, fixed.** The medians of R0, R1 and R2 beside the kept value, and the share of the kept set's excess over the R0 median that the R1 and R2 medians reach. By floor status at the cuts of Section 11 (28%, 30% and 35% on the kept questions): clean–clean and filter–clean pairs with both systems above the cut and with at least one below. For K, lineage–clean pairs by line (Llama-2-7b, Llama-2-13b, Mistral-7B, gemma-7b). Matching diagnostics per fold: level gaps of the scored clean-pool systems below and above 40% on all questions, expected overlap with the kept set, key shares. Tie diagnostics per fold: the number of distinct difficulty values and the share of questions whose stratum depends on the tie-break.

**Matching check, before fixing the choices** (2026-11-16, `scripts/mmlu_difficulty_check.py`; it reads no order outcome: only the estimators' outcomes, the kept set, the answer keys of `cais/mmlu`, and the expected scores of the scored clean-pool systems on matched subsets, which are exact and need no draw).

- Answer keys of the kept questions: A 26.6%, B 24.6%, C 23.4%, D 25.3%; of all questions of the used subjects: A 22.9%, B 24.7%, C 25.6%, D 26.9%. Subsets matched on difficulty alone (4 strata or more) have A 22.2 to 24.1% and D 27.2 to 30.4%: the panel's choice leans to questions keyed A beyond what difficulty gives.
- Level gap, the expected score of the other half's clean-pool systems on a matched subset minus their score on the kept set (folds 1 and 2). Subject composition only: +7.8 and +8.5 points (+3.1 and +3.4 for systems below 40% on all questions, +12.0 and +12.2 above). All eligible systems (24 and 30) as estimators, 8 strata: −1.2 and +0.2 below 40%, +3.9 and +3.8 above. Eligible systems of at least 40% (10 and 16): R1 at 8 strata +0.1 and +1.2 below, +1.9 and +2.0 above; R1 at 16 strata +0.1 and +1.1 below, +1.7 and +1.8 above; R2 at 8 strata +0.8 and +1.0 below, +1.6 and +1.8 above. The whole half as estimators, lines of filter models included: R2 at 8 strata at most +1.8; R1 at 16 strata up to +3.0 above 40%.
- Expected overlap of a matched subset with the kept set: R0 0.556; R1 at 8 strata 0.69; R1 at 16 strata 0.70; R2 at 8 strata 0.73 (R2 at 16 strata 0.77, with about 30% of its cells entirely kept).
- **Choices fixed from this check:** the eligible estimators of at least 40%, which give the smaller largest gap of the two estimator sets that leave out the lines of the filter models; 8 strata for R1 and for R2. First fixed were 16 strata for R1 and 8 for R2, the finest of the grid (2, 4, 8, 16) that keep the expected overlap below 0.75; R1 and R2 then differed in two respects, so Revision 1 puts both at 8 strata and R1 at 16 strata among the sensitivities. The bound on the overlap still holds. For these choices the matched subsets stay up to about 2.0 points (R1) and 1.8 points (R2) easier than the kept set for the clean-pool systems above 40% on all questions, so a fully matched reference is not available from these estimators. The readings assume no direction for the effect of that residual.

**Sensitivity, fixed** (reported, not read; revised in Revision 1): R1 at 4 and 16 strata and R2 at 4 strata (R2 at 16 strata has an expected overlap of 0.77 and is not run); as estimators all eligible systems of the half, and the whole half (lines of filter models included); **difficulty with equal weight per line**, the share right averaged within each line of the fold's estimators and then over lines, so that the 11 Qwen estimators of fold 2 count as one line; **ten further tie-break keys**; five further splits of the clean pool by line (random assignment of lines with keyed generators, accepted when the halves differ by at most six systems and by at most six estimators and the split differs from the primary one, from its mirror image and from the splits already accepted; the primary split differs by two and six). Except for the strata variants, every variant uses R1 and R2 at 8 strata. 2,000 subsets per fold for the strata, estimator and line-weight variants, 500 for the tie keys and the splits. For each: the kept value against R1 and R2 for the four pair types that are read; for the tie keys, the spread of the reference bounds and medians over the eleven keys.

**Stated in advance** (my expectations, written before Revision 1 for R1 at 16 strata and kept unchanged; they are read against R1 at 8 strata. Two earlier advance statements of this study were partly wrong):

- Clean–clean pairs: the excess over R0 shrinks against R1 and further against R2, and is inside R2. Whether it is inside R1 depends on how much the floor systems' guessing depends on the answer letter, which I do not know.
- Filter–clean pairs: above R1 and R2, carried by Llama-2-7b, which falls among the systems at the floor further than questions of the same difficulty and key take systems of its ability.
- K: lineage–clean pairs above R1 and R2 for the two Llama-2 lines and inside for Mistral-7B; lineage–lineage-or-filter pairs above.

**Limits.**

- Difficulty rests on 10 and 16 systems, those of fold 2 mostly Qwen (line-weight sensitivity). The matched subsets remain easier than the kept set: the panel's own outcomes carry information on difficulty that the estimators lack (regression towards the mean). The effect of that residual on the counts has no known direction.
- With 10 binary estimators difficulty takes at most 11 values, so the tie-break decides the stratum of some questions (tie diagnostics and tie-key sensitivity).
- Discrimination is not matched. Zheng & Yang add a control matched on discrimination and reliability as a separate diagnostic; nothing of the kind is done here.
- The estimators and strata were chosen from the matching check, which used the scored halves' levels.
- The 1,224 clean–clean pairs across halves are not evaluated.
- One split is primary; lines are read from names.
- R2 matches the answer-key mix, which is itself a product of the panel's choice; R1 keeps it. Neither separates content from other features of the questions.
- Exploratory, after the results of Sections 9 to 11, and prompted by a precedent read after them.
- As in Section 11: "kept" mixes the filter with the later review; archived runs in the original four-option format; one run per system.

**Outputs.** `scripts/mmlu_difficulty_check.py` (written and run for the check above); `scripts/mmlu_difficulty_reference_test.py`, to be written after approval and run first on a synthetic matrix, which checks that its halves equal the check's and that R0 on all of population A reproduces the reference of Section 11 (16 to 39 sign changes); `results/mmlu_reconstruction/summary_difficulty_reference.md`, `records_difficulty_reference.tsv`, `halves.tsv`. No manuscript file is changed.

**Revision 1** (2026-11-16, before any order statistic on matched subsets was computed). Justin asked for the changes below and then the run.

1. *No bound.* The first readings said that a fully matched reference was not expected to give fewer sign changes than the matched subsets, and that an excess above R1 or R2 was an upper bound on what the panel's choice adds. Both rested on fewer sign changes being expected on easier subsets, which nothing guarantees: the order of two systems can change more than once as the questions get harder (example in the readings). Both statements are withdrawn, and the residual level gap is reported as unresolved matching bias.
2. *Same strata for R1 and R2.* R1 had 16 strata and R2 8 strata crossed with the key, so the two differed in two respects. Both now have 8 strata; R1 at 16 strata is a sensitivity.
3. *Tighter readings.* Inside a reference means compatible with it at this precision, not that the matched features account for the change. A residual above R2 is not attributed to the panel: the error of the difficulty estimates, discrimination, content and the later review remain.
4. *Status.* The matching check used the scored halves' scores, and the estimators and strata were chosen from it. The analysis is tuned on evaluation-side levels and then fixed, blind to the order statistics only.
5. *Added:* difficulty with equal weight per line (11 of the 16 estimators of fold 2 are Qwen); ten further tie-break keys, since 10 binary estimators give at most 11 difficulty values and 8 or 16 strata then cut groups of tied questions; tie diagnostics. The sensitivity table covers the four pair types that are read, not only those of population A.
6. *Unchanged:* the cross-fit by halves, the clean–clean pairs across halves left out, the mean over folds for pairs scored in both, the test on a synthetic matrix first, the reproduction of R0.

**Synthetic checks** (2026-11-17, before the run). Two matrices with the real names, sets, tags and subjects and with synthetic outcomes (one-parameter model with guessing), kept sets and answer keys; the script of Section 11 was run on each first, with 400 subsets, and this script with 400 and 100.

- *Kept set independent of the outcomes:* every position inside, for every pair type, statistic and variant; R0 reproduced the synthetic records of Section 11 to six digits; every matched subset had the right composition.
- *Kept set chosen through the outcomes:* the questions on which at most two of the five synthetic filter models are right, so that the set is chosen through difficulty and through the filter models' own outcomes, with level gaps of about 3 points left after matching. Clean–clean pairs above R0 and inside R1 and R2 (20 against 13 to 26 and 13 to 25). Filter–clean pairs, scored on the outcomes that voted, above all three (144 against at most 10). Lineage–clean pairs, which carry no planted effect, inside R1 and R2 in the primary analysis but at their upper edge, and above in 5 of the 19 sensitivity references. So in this one model the residual matching bias did not move clean–clean pairs outside a matched reference, and a count at the edge of a reference can fall just outside without any effect: only clear excesses carry information.

**Run.** One run, 2026-11-17: `python -E -P -X utf8 scripts/mmlu_difficulty_reference_test.py results/mmlu_reconstruction results/mmlu_reconstruction/raw/cais_mmlu_all_test.parquet` (log `difficulty_reference.log`). Checks: the halves equal the matching check's; R0 on all 2,775 pairs of population A reproduces Section 11 to six digits (16 to 39 sign changes); every matched subset has the kept set's number of questions in every cell. No deviation.

**Results by the fixed readings** (`results/mmlu_reconstruction/summary_difficulty_reference.md`).

- **Sign changes between subset and all questions**, kept set against the central 95% of R0, R1 and R2:
  - clean–clean pairs: 113 against 6 to 19, 28 to 48 and 46 to 63;
  - filter–clean pairs: 28 against 0 to 4, 0 to 5 and 0 to 4;
  - lineage–clean pairs: 375 against 8 to 25, 24 to 42 and 58 to 73;
  - lineage–lineage and lineage–filter pairs: 273 against 47 to 94, 90 to 119.5 and 132 to 164.5.
  
  Each is above all three references. **Reading fixed in the design, for each of the four: a residual difference remains after matching on subject, estimated difficulty stratum and answer key; it is not attributed to the panel.**
- **Share of the excess over the R0 median that the R1 and R2 medians reach:** 25% and 43% for clean–clean pairs, 7% and 4% for filter–clean, 5% and 14% for lineage–clean, 18% and 39% for lineage–lineage and lineage–filter.
- Sign changes between subset and complement: above all three references for every pair type. Kendall's τ_b between kept and all questions: 0.818 in A and 0.849 in K, below all three references (R2: 0.915 to 0.938 and 0.935 to 0.945).
- **Sensitivity** (reported, not read): the same position for every pair type in every variant (R1 at 4 and 16 strata, R2 at 4, both other estimator sets, equal weight per line), under all ten further tie-break keys and in all five further splits.

**Descriptions.**

- *Floor status:* clean–clean pairs of two systems at or above the cut: 3 sign changes against 1 to 5 under R1 and R2, at each of the three cuts, so inside all three references. Pairs with at least one system below the cut: 110 against 25 to 45 (R1) and 44 to 60 (R2). Filter–clean pairs: 3 or 0 with both systems above the cut (inside), 25 to 28 with at least one below (above).
- *Lineage–clean pairs by line:* Llama-2-7b 315 against 34 to 44 (R2); Llama-2-13b 34 against 3 to 8; Mistral-7B 25, above R1 (9 to 22) and inside R2 (16 to 25); gemma-7b 1, inside.
- *Matching:* level gaps as in the matching check; 11 and 17 distinct difficulty values in the two folds; the stratum of 72.8% and 55.6% of the questions depends on the tie-break, and changing the tie-break key changes no position.

**Against what was stated in advance.**

- *Clean–clean: the excess shrinks against R1 and further against R2:* as stated (medians 12, 37 and 55). *Inside R2:* **wrong**, 113 against 46 to 63.
- *Filter–clean above R1 and R2, carried by Llama-2-7b:* as stated; the 28 sign changes are those of Section 11, 25 of them with Llama-2-7b.
- *Llama-2 lines above, lineage–lineage-or-filter above:* as stated. *Mistral-7B inside:* inside R2, above R1.

**Reading** (mine; exploratory; item 1 and the account of the synthetic check corrected below, 2026-11-17).

1. Subject, estimated difficulty in eight strata and the answer-key mix do not account for the excess of sign changes. Together they reduce it by about two fifths for clean–clean pairs and by less for pairs with a filter-line system. A residual difference remains. By the readings fixed in advance it is not attributed to the panel: errors of the difficulty estimates (the kept set stays 1 to 2 points harder for the scored systems), discrimination, content and the later expert review are not separated from the panel's choice.
2. The residual lies among pairs that involve a system at the guessing floor. Pairs of two systems above it are within all three references.
3. The synthetic check speaks to one of those sources only within its model: there a residual gap of about 3 points left clean–clean pairs inside the matched references. That is one model, not the data.
4. Among lineage systems the residual is that of the Llama-2-7b line (315 against at most 44), then Llama-2-13b. Mistral-7B, also a filter model's line, is within R2.
5. Wording for the paper: a residual difference remains after matching on subject, estimated difficulty and answer key, concentrated at the guessing floor and in the Llama-2 lines. Never that difficulty explains the change, and never that the panel causes the residual.

**Corrections of 2026-11-17** (at Justin's request, before any manuscript text; nothing was recomputed).

1. *The shares of the excess are descriptive.* The share reported under Descriptions is (median of R1 or R2 − median of R0) / (kept value − median of R0): a ratio of counts between references. It is not the share of the change explained by subject, estimated difficulty or answer key. In Reading item 1, "Together they reduce it by about two fifths" is withdrawn. The descriptive statement is that the R2 median lies about two fifths of the way from the R0 median to the kept value for clean–clean pairs. The main text does not carry these shares. If they are reported, they go to the supplement with this formula.
2. *The synthetic check is not a calibration under a pure-difficulty null.* In the second synthetic matrix the kept set was chosen from the answers of five synthetic filter models (as stated above). Filter–clean pairs therefore carry the same-output selection effect. For clean–clean and lineage–clean pairs, the kept set relates to the scored outcomes only through item difficulty, and only within the one-dimensional generating model. The check calibrates the matched references against that source, in that model. A summary in chat had called it a difficulty-only filter; that was wrong.
3. *Place in the evidence.* These results bound the order evidence. A count above R2 does not exclude difficulty: the error of the difficulty estimates and the residual matching gap of about 1.8 to 2.0 points remain. Nor is it attributed to the panel.
