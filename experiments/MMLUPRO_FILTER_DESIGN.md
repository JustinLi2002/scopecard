# Are MMLU-Pro's eight filter models penalised on the questions they filtered? — study design

**Status:** design fixed 2026-11-15, before any file of this study was downloaded and before any model output on MMLU-Pro was seen. Justin approved the download and the test on that day. Internal, dated document: the word used is "prespecified".

**What is known beforehand.** From the MMLU-Pro paper (arXiv 2406.01574, Section 3.2): eight models (Llama-2-7B, Llama-2-7B-Chat, Llama-2-13B, Llama-2-13B-Chat, Mistral-7B, Gemma-7B, Yi-6B, Yi-6B-Chat) were run on MMLU; a question answered correctly by more than four of them was removed (5,886 questions). Questions from a STEM website, TheoremQA and SciBench were then added, which the eight models did not filter; options were expanded from four to ten with GPT-4-Turbo; experts reviewed. From the repository listing (names and sizes only): `eval_results/` holds per-question outputs of 48 systems, among them six of the eight filter models. From our own simulation on BigCodeBench (`PANEL_FILTER_DESIGN.md`, Section 12): with eight builders and a 50% rule a builder loses 3 to 4 points on the kept tasks. No accuracy of any model on any part of MMLU-Pro has been looked at for this study; overall leaderboard scores of some models are public and partly known.

## 1. Question

Do the filter models score lower on the MMLU-origin questions of MMLU-Pro, which they helped to select, than non-filter models of the same ability, once ability is measured on the questions they did not select?

## 2. Data

- **Outputs:** `github.com/TIGER-AI-Lab/MMLU-Pro`, every file under `eval_results/` at the head commit of the default branch when downloaded (hash recorded). One system per file.
- **Questions:** Hugging Face `TIGER-Lab/MMLU-Pro`, test split (12,032 expected): `question_id`, `category`, `src`, `answer_index` and the number of options.
- **Source of a question:** M (MMLU-origin) if `src` starts with `ori_mmlu`, O (other) otherwise.
- **Correctness:** the system's recorded prediction equals the gold answer. An unanswered or unparsable prediction is wrong. A question missing from a system's file is missing; a system with fewer than 95% of the questions is dropped and listed.
- **Filter models F:** the systems whose names are Llama-2-7b-hf, Llama-2-13b-hf, Mistral-7B-v0.1, gemma-7b, Yi-6B and Yi-6b-Chat (the two Llama-2 chat models have no file). **Siblings:** Llama-2-70b-hf, Mistral-7B-v0.2-hf, Mistral-7B-Instruct-v0.1, Mistral-7B-Instruct-v0.2, Yi-34B. All other systems are non-filter models. If a zip holds several runs of one system, each run is a system and the list is reported.

## 3. Estimand and primary analysis

- **Eligible categories:** those with at least 100 questions from each source. Both source-specific accuracies of a system are standardised to the same category mix, with weights proportional to the number of questions of the category over both sources: a(M) and a(O).
- **Controls of a filter model m:** the non-filter systems, siblings included, whose a(O) is within 0.10 of a_m(O); if fewer than three, the three nearest.
- **Penalty of m:** P_m = [mean over controls of a(M) − a_m(M)] − [mean over controls of a(O) − a_m(O)]. Positive means m does worse on the questions it filtered than systems of equal ability on the questions it did not filter.
- **Primary estimand:** P = the mean of P_m over the filter models.

**Prediction, written in advance:** P > 0. The simulated 3 to 4 points are an upper guide only: the kept questions were rewritten to ten options and the public runs use another prompt than the filter runs, so a filter model's old errors need not recur. An estimate anywhere between 0 and 4 points is compatible with the mechanism.

## 4. Inference

- **Placebo test (primary).** Systems differ in how they do on M against O for reasons unrelated to filtering, and there are few systems, so the reference is model-level. Every non-filter system whose a(O) lies inside the range of the filter models' a(O), widened by 0.10 on each side, gets a placebo penalty computed exactly as above, with itself removed from the controls. The p-value is the share of all subsets of that pool with as many systems as F whose mean placebo penalty is at least P (exact if the number of subsets is at most 200,000, otherwise 200,000 random subsets).
- **Question bootstrap.** 1,000 replicates resampling questions within category and source; percentile 95% intervals for P and each P_m. It treats the systems as fixed and ignores model-level variation, so it is the weaker of the two.
- **Non-degeneracy gate, checked before P is computed:** if more than half of the filter models have a(O) below 0.15 (ten options, chance 0.10), ability cannot be matched on O and the primary analysis is declared uninformative; only descriptive tables are then reported.

## 5. Reading, fixed in advance

- **Confirmed:** P at least 1 point, placebo p below 0.05, and the bootstrap interval above zero.
- **Detectable but small:** the two tests pass and P is below 1 point.
- **Not detected:** otherwise. Not detected does not refute the mechanism, because the reformatting can hide it; it means this benchmark's public outputs do not show it at the precision available.

## 6. Secondary analyses (fixed)

1. **Siblings removed from the controls and from the placebo pool.** If family dependence exists, siblings carry part of the penalty and dilute P. The siblings' own placebo penalties are listed.
2. **Regression version.** Over non-filter systems, a(M) regressed on a(O) and its square; the filter models' residuals against the leave-one-out residuals of the non-filter systems in the same ability range.
3. **Dependence footprint.** The filter rule caps how many filter models may be right on a kept question, which should lower the correlation of their correctness on M. Statistic: mean pairwise phi coefficient among the filter models on M minus on O (eligible categories). Reference: the same statistic for 2,000 random sets of as many systems from the placebo pool.
4. **All categories,** without standardisation.
5. **By model,** with the family noted (two Llama-2, two Yi, one Mistral, one Gemma).

## 7. Limits stated in advance

- Six of eight filter models; all are small and old, so the ability range has few non-filter systems of other families.
- The filter ran on four-option MMLU with unknown prompts; the public outputs are on ten-option MMLU-Pro.
- O questions are mostly computational STEM items converted by GPT-4-Turbo, and M questions mostly knowledge items: systems can differ on that axis for reasons unrelated to filtering. The placebo test is the guard, and it is only as good as the pool.
- One benchmark, one set of public runs, no repeated runs.

## 8. Outputs

`experiments/scripts/mmlupro_fetch.py`, `experiments/scripts/mmlupro_filter_test.py`; `experiments/results/mmlupro_filter/` (raw/ with a manifest, acceptance.txt, summary.md, records.tsv, systems.tsv). No manuscript file is changed.

## 9. Deviations

None in the analysis. Two notes on the data, recorded 2026-11-15 after the run:

- **Gold answers.** The public outputs were produced on earlier versions of the dataset: about 170 to 210 recorded gold answers per system differ from the current table. As fixed in Section 2, correctness uses the gold answer recorded with each prediction; the current table supplies only category and source.
- **One misaligned file.** `Meta-Llama-3-70B-Instruct_5shots` has 12,187 question ids and 10,756 gold answers that differ from the table, so its ids do not correspond to the current questions. The design has no rule for this case and the file was not removed. It lies far outside the ability range of the filter models, so it is in no control set and no placebo pool; it enters only the regression fit of secondary analysis 2.

**Corrections of 2026-11-16.** Justin approved them on that day. They were written here before the script was run again.

1. **The misaligned file is excluded (deviation).** Keeping a file whose question ids are known not to correspond to the questions was a mistake. That the design had no rule for the case was not a reason to use it. Rule, fixed now: a run is excluded from every analysis if more than 5% of its recorded gold answers differ from the question table. The other 46 runs differ in 173 to 206 of 12,032 answers (at most 1.7%); `Meta-Llama-3-70B-Instruct_5shots` differs in 10,756. The run was in no control set and no placebo pool, so only the regression of secondary analysis 2 can change. Result of the new run (2026-11-16): the regression is fitted on 40 systems; the mean shortfall of the filter models goes from 1.40 to 1.13 points and its reference tail share from 0.089 to 0.135. Every other record is identical to the first run.
2. **The placebo share is not a calibrated p-value.** The share of subsets of the placebo pool whose mean is at least P says where P lies among those subsets. It controls no error rate: the filter models were not assigned at random and are not exchangeable with the pool systems, the pool systems are not independent of each other, and the controls are not matched again for each subset. Complete enumeration makes the share exact as a count and nothing more. The summary now calls it the reference tail share. The readings fixed in this design stand as the labels that the rule gives; they are not tests with a controlled error rate.
