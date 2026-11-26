# Measured c on released subsets — study design (SWE-bench Lite and Verified, BigCodeBench-Hard)

**Status:** design fixed 2026-11-14. Justin approved the design and the four defaults in Section 12 on that day. **The design was fixed before any data file was downloaded and before any comparison between a subset and its complement was computed.** Like `MEASURED_C_DESIGN.md`, this is an internal, dated document, not an entry in a public registry: the word used is "prespecified".

**What had been looked at before this file was written:** the published construction descriptions (Section 13), repository file listings, column names and file sizes, and the status counts of one submission (SWE-agent with GPT-4 on the full SWE-bench test split: 286 resolved). Split-level scores of early systems on Lite and on the full split are public and were partly known to the designer. No per-instance comparison between a subset and its complement had been made.

## 1. Question

Many current benchmarks ship a rule-filtered subset of a parent set ("Lite", "Verified", "Hard"), and their leaderboards publish per-instance outcomes of many systems on the whole parent. For such a pair the kept part O, the excluded part U and the excluded share W are all observed, so c can be measured without training or running any model.

1. **Magnitude.** How does each system's score differ between the kept and the excluded instances?
2. **Comparison.** How much of a gap between two systems measured on the subset carries over to the excluded part and to the parent? Does a comparison that the subset resolves change sign on the parent?
3. **Rule or size.** Is the difference between the subset's ranking and the parent's larger than a random subset of the same size would give?
4. **Rule diagnosis.** Does the stated construction rule reproduce the release?

## 2. Declared targets, releases and units

**Unit.** One task instance is one query. Exclusion is at the query level: an instance is kept or dropped whole, and its outcome does not depend on other instances. **Metric:** the mean over instances of the binary outcome y (resolved, or passed), which is each benchmark's own headline metric. For a region R, s_m(R) is the mean of y over R for system m. The identity is exact: s_m(V) = (1 − W)·s_m(O) + W·s_m(U), with W = |U| / |V|.

**Case A — SWE-bench Lite (primary; a scope rule).**

- V_A: the SWE-bench test split (2,294 instances expected). O_A: the Lite test instances (300 expected). U_A = V_A \ O_A.
- Stated rule: seven filters on the problem statement, the gold patch and the tests, then a random sample of 300 from the remaining candidates.
- E_A: the instances of V_A that pass the stated filters, recomputed from the published criteria (`swebench/collect/make_lite/criteria.py`). The criteria are re-implemented in our own script from a reading of that file; the downloaded code is not executed.
- Three regions follow: O_A (kept), E_A \ O_A (eligible, not sampled) and V_A \ E_A (excluded by the filters).
- If fewer than 95% of O_A fall inside the recomputed E_A, the re-implementation is judged unfaithful: the analyses that use E_A are reported as not available. The filters are not tuned to raise the containment.

**Case B — SWE-bench Verified (contrast; a validity filter).**

- V_B: the test instances that carry OpenAI's annotations (1,699 expected). O_B: the Verified instances (500 expected). U_B = V_B \ O_B, split into U_B,fail (fails the stated filter) and U_B,pass (passes, not selected).
- Stated rule: three annotators per sample, the highest severity kept; a sample is dropped if underspecification or test validity is 2 or above, or if another major issue is flagged; among the passing samples as many of the 1–4 hour and >4 hour ones as possible are kept and the rest drawn at random to reach 500.
- Test instances without annotations are outside V_B and appear only in a side table.
- Tasks are dropped here because the task or its tests are judged defective, so the complement is not a population anyone intends to rank on. The same quantities are computed as in A and C, and the text will not present U_B as a target population.

**Case C — BigCodeBench-Hard (primary; a difficulty rule).**

- V_C: BigCodeBench (1,140 tasks). O_C: BigCodeBench-Hard (148 tasks). U_C = V_C \ O_C (992 tasks, W = 0.870).
- Two prompt variants: Complete (primary; 153 models in the public table when the design was fixed) and Instruct (replication; 126 models).
- Stated rule: among 626 tasks matched to Stack Overflow questions (embedding similarity above 0.7), keep the tasks that use more than two libraries, whose canonical solution is longer than 426 tokens (the benchmark's mean), and whose solve rate over leaderboard models is below 50%.

## 3. Data, pinning and inclusion

- **SWE-bench outcomes:** `github.com/SWE-bench/experiments`, `evaluation/test/*/results/results.json` and `metadata.yaml`, at the commit that is the head of the default branch when downloaded (hash recorded). Every submission under `evaluation/test` is included. y = 1 if the instance id is in the `resolved` list, else 0: instances with no generation, no logs or errors count as unresolved, as on the leaderboard. A submission is dropped only if `results.json` is missing or has no `resolved` list. A resolved id outside V_A is listed and ignored.
- **SWE-bench instance tables:** Hugging Face `princeton-nlp/SWE-bench` (test), `princeton-nlp/SWE-bench_Lite` (test) and `princeton-nlp/SWE-bench_Verified` (test), at the revision current when downloaded (recorded).
- **Verified annotations:** `cdn.openai.com/introducing-swe-bench-verified/swe-bench-annotation-results.zip`. If the file holds the three raw annotations, the ensemble is recomputed as the maximum; if it holds ensembled labels, they are used as given.
- **BigCodeBench:** Hugging Face `bigcode/bigcodebench-perf` (splits complete and instruct: one row per model, one column per task), `bigcode/bigcodebench` and `bigcode/bigcodebench-hard` (task ids, libraries, canonical solutions; the latest version split), the three solve-rate tables, and `bigcode/bigcodebench-results` (model list with dates).
- **Cell coding of the performance table:** the mapping from cell values to y is fixed after listing the distinct cell values and before any score is computed. Acceptance check: with that mapping, a model's mean over V_C is compared with its published full-set score in `bigcodebench-results`. The check is on levels, not on regions. If fewer than 95% of the matched models agree within 0.5 points, the table is still used, since it is the only per-task source, and the disagreement is reported.
- **Models with missing cells** are dropped and listed.
- **Systems are never selected by performance.**
- Every downloaded file is listed in a manifest with its source URL, size and SHA-256.

## 4. Estimands

For a case with regions O, U, V and two systems:

- **Orientation:** A is the system with the higher score on O. A pair with Δ_O = 0 is "tied on O": it has no ρ and does not enter sign counts.
- Δ_R = s_A(R) − s_B(R) for R in {O, U, V}.
- **c = Δ_O − Δ_U** and **c\* = |Δ_O| / W**. Δ_V = Δ_O − W·c holds exactly.
- **ρ = c / Δ_O = 1 − Δ_U / Δ_O**, reported for pairs resolved on O. ρ = 0: the gap is the same on the excluded part. ρ = 1: there is no gap there. **ρ > 1/W: the parent ordering is the opposite of the subset ordering.**
- **Per system:** s(U) − s(O).
- **Ranking level:** Kendall's τ_b between the systems' scores on O and on V; whether the top system is the same; the number of pairs, among those not tied on O, whose Δ_O and Δ_V have opposite signs.

**Case A, three-region analyses.** The same quantities for (O_A against E_A \ O_A), the negative control in which exclusion is the stated random sampling, and for (E_A against V_A \ E_A), the filters alone, with W_rule = |V_A \ E_A| / |V_A|.

**Case A, mechanism check (secondary).** On the annotated instances only: c between Lite and the rest, raw and with the excluded instances reweighted to the kept instances' distribution over OpenAI's four difficulty labels. A label with no kept instance is dropped from both sides and reported.

**Case C, selection on outcomes (sensitivity).** The solve-rate filter used leaderboard models' own outcomes. Models are grouped by their date in `bigcodebench-results`: on or before the Hard release (2024-07-24), after it, and undated or unmatched. All quantities are also reported for pairs formed within the "after" group, which could not have entered the filter.

## 5. Decision language

As in `MEASURED_C_DESIGN.md`, Section 7:

- **Resolved** in a region: the 95% interval of Δ_R excludes zero.
- **Reversal:** Δ_O and Δ_V are both resolved, with opposite signs.
- **Newly resolved:** Δ_O unresolved and Δ_V resolved.

Added here:

- **Lost:** Δ_O resolved and Δ_V unresolved.
- **Attenuated / amplified:** Δ_O is resolved and the 95% interval of the oriented c lies entirely above / below zero.

Every label is reported under unadjusted per-pair intervals, as in the WN18RR study, and under simultaneous intervals (Section 6).

## 6. Uncertainty

- **Systems are fixed; instances are the sampling unit.**
- **Bootstrap:** paired, over the instances of V, 2,000 replicates. Each resampled instance carries its region label and every system's outcome, so the same replicates serve all systems, pairs and regions, and c comes from the same replicate as Δ_O and Δ_U. RNG streams are keyed by a string (`subset_c/<case>/<analysis>`), so reruns reproduce every interval. Intervals are percentile intervals.
- **ρ:** its interval uses the replicates with a positive Δ_O, and ρ is reported only for pairs resolved on O under the unadjusted interval.
- **Simultaneous intervals:** bootstrap maximum of |t| over the family of all pairs and both of Δ_O and Δ_V, with t = (replicate − estimate) / bootstrap SD. The interval is the estimate ± q × SD, with q the 95th percentile of the maximum. A statistic with zero SD is left out of the family and counted as unresolved. The oriented c gets its own family over pairs.
- **Sensitivity:** (i) the bootstrap stratified by region, which keeps W fixed; (ii) for SWE-bench, a cluster bootstrap by repository (12 clusters, so indicative only).
- **Not in the intervals:** run-to-run variation. Each system has one run.

## 7. Reference distributions

- **(a) Random subsets (rule or size).** 2,000 subsets of V with |O| instances, drawn uniformly without replacement with a keyed RNG. For each: τ_b between the ranking on the subset and on V; the number of pairs with opposite signs on the subset and on V; and the mean over pairs of |c|, with c taken between the subset and its complement. The released subset's values are reported with their percentile in this reference. For SWE-bench a second reference draws subsets with the same per-repository counts as O.
- **Reading, fixed in advance:** a value inside the central 95% of the reference means that, on that statistic, the released subset is not distinguishable from a random subset of its size.
- **(b) Case A negative control:** O_A against E_A \ O_A (Section 4).

## 8. Calibration of c\*

For each case and analysis, over the pairs resolved on O under unadjusted intervals: the distribution of ρ (median, quartiles, 5th and 95th percentiles, the share above 1 and the share above 1/W) and of |c| in score points. A benchmark whose excluded part has not been measured has its reversal threshold at ρ = 1/W, which can be set against these distributions. This is calibration by observed cases, in the sense used in sensitivity analysis. It is not a prediction for any unmeasured benchmark, and no pooled test is made across cases.

## 9. Rule diagnosis

- **A:** |E_A|; the containment |O_A ∩ E_A| / |O_A|; the number of instances that each filter alone excludes; W_rule.
- **B:** the pass/fail filter recomputed from the annotation file; whether every Verified instance passes; the number passing and the share of them selected.
- **C:** over V_C, the tasks that satisfy each computable filter (more than two libraries; solve rate below 50% in each solve-rate table; canonical solution longer than the mean, in characters and in whitespace-delimited tokens, since the tokenizer is not stated), the Jaccard similarity of their intersection with the release, and the share of Hard tasks that satisfy each filter. Whether membership of the 626-task Stack Overflow pool is public for tasks outside Hard is recorded.

## 10. Outputs and provenance

- **Raw downloads:** `experiments/results/subsets/raw/`, one directory per source, with `manifest.tsv` (path, bytes, SHA-256, URL, upstream revision, date).
- **Scripts:** `experiments/scripts/subsets_fetch.py` (downloads) and `experiments/scripts/subsets_analyze.py` (acceptance checks and every analysis here).
- **Results:** `experiments/results/subsets/acceptance.txt`, a keyed `records.tsv` (case / analysis / subject / quantity / value), one pair table per case and analysis, and `summary.md`.
- **No manuscript file is changed by this study.** Whether and how the results enter the paper is decided after Justin has seen them.

## 11. Threats and limits stated in advance

- **A:** 24 systems on the full split, from 2023 retrieval baselines to 2025 agents, with widely spaced scores. Few pairs are close, so few can reverse. Submissions are voluntary and are not a sample of systems.
- **A:** Lite is a random sample of the eligible set, so its difference from the parent mixes the rule and the sampling. The three-region analysis separates them.
- **B:** a validity filter that rests on annotators' judgments; the final 500 over-represent the harder passing samples by design.
- **C:** the solve-rate filter was computed from leaderboard models' outcomes, so models evaluated before the Hard release took part in their own filter. The before/after split addresses this.
- **C:** Hard tasks also had to match a Stack Overflow question, and the pool's membership may not be public.
- **All:** one run per system; possible training-data contamination; outcomes depend on scaffolds and harness versions. The results describe these systems on these benchmarks. They are not an estimate of how often rankings reverse in general.
- **All:** published split-level scores were partly known before the design was fixed.

## 12. Decisions (Justin, 2026-11-14, all at the proposed defaults)

1. Cases A and C are primary; case B is a contrast.
2. Case C: Complete is primary and Instruct is the replication.
3. Every listed system is included; none is selected by performance.
4. No manuscript change until Justin has seen the results.

## 13. Sources read before the design was fixed (2026-11-14)

- SWE-bench Lite: https://www.swebench.com/lite.html (filters, sampling, 11 of 12 repositories).
- SWE-bench Verified: https://openai.com/index/introducing-swe-bench-verified/ (13 August 2024; annotation procedure, filter, selection of the 500, released annotations).
- BigCodeBench-Hard: https://huggingface.co/blog/terryyz/bigcodebench-hard (24 July 2024; pool and three filters).
- Related work seen only as abstracts or summaries, to be read in full before any citation: arXiv 2609.17394 (separability of SWE-bench leaderboard entries within each split), arXiv 2603.03493 (ranking reversals across protocol axes in gene-regulatory-network benchmarking), arXiv 2202.03408 (sensitivity analysis for generalising experimental results).

## 14. Deviations from the fixed design

**1. Lite's issue-reference filter cannot be recomputed (recorded 2026-11-14, after the downloads and before any analysis was run).** On reading `criteria.py`: an instance whose problem statement has a bare `#n` is excluded only if an HTTP request for that issue returned status 200 when Lite was built. That request cannot be reproduced. Two readings bracket the true eligible set: **strict** (every `#n` excludes) gives a subset of it, and **loose** (no bare `#n` excludes) gives a superset. `unidiff`, which the criteria use to read patches, is replaced by a minimal reader written from its file classes (added, removed, modified).

- **Faithfulness (Section 2)** is judged on the loose reading: at least 95% of Lite must lie inside it.
- **Negative control:** Lite ∩ E_strict against E_strict \ Lite. Both sides then hold only instances that are eligible for certain, and the stated sampling does not depend on whether a statement has a `#n`.
- **Filters alone:** reported under both readings (E_strict, and E_loose together with any Lite instance outside it, each against the rest of the test split).

**2. Size of the downloads.** The approved estimate was about 20 MB; 40.3 MB were stored (69 files). This machine has no parquet reader, so the Hugging Face tables were read through the datasets-server rows API as JSON, with only the needed columns kept, and the annotation archive expands to 8.3 MB.

**3. Dates of BigCodeBench models (noted 2026-11-14, before any analysis).** `bigcodebench-results` has one `date` per model and does not say whether it is a release or an evaluation date. The grouping of Section 4 uses it as given.
