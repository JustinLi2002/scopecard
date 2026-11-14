# Measured c on WN18RR — experiment design

**Status:** design fixed 2026-10-18; the author took the four decisions in Section 12 at their proposed defaults the same day. Benchmark chosen by the author on 2026-10-18 under a criterion in the project plan, which is not part of this copy. **The design was fixed before any model was trained.** The target population below is built and passes its acceptance checks.

**Terminology (2026-10-28):** this document used to call the design and its analyses "pre-registered". It is an internal, dated document, not an entry in a public registry, so the wording is now "prespecified". No design content changed.

## 1. Question

WN18RR's entity set (inherited from WN18) excludes 63% of the WordNet 3.0 synsets that carry its relations, by a rule its paper states but that does not reproduce (`pilot_decidability.tsv`). The excluded region is fully observable in WordNet 3.0. The experiment measures what the exclusion costs:

1. **Magnitude.** How much worse does each model rank the benchmark's own test positives once the excluded synsets are admitted as candidates?
2. **Ranking.** Does any model comparison that the benchmark resolves change sign, or does an unresolved one become resolved, on the declared target?
3. **Mechanism check.** How much of any O–U difference is explained by the excluded synsets being rarer (degree) or of a different part of speech?

## 2. Declared target population

Built by `scripts/build_wn18rr_target.py` into `results/wn18rr_target/`; 18 acceptance checks in `scripts/accept_wn18rr_target.py` pass, including a byte-identical rebuild; table hashes in `target_files.sha256`.

- **Universe V.** Every WordNet 3.0 synset carrying at least one triple of WN18RR's 11 relation types, with all such triples (lexical pointers lifted to synsets, duplicates collapsed): 112,199 synsets, 218,367 triples.
- **Identity.** A synset is (POS, offset); satellite adjectives share `data.adj`'s offsets and key as `a`. WN18RR keys by bare offset, which merges synsets of different POS; the mapping goes through the triples, so the 161 merged ids become the synsets they actually denote. O therefore has 41,105 synsets covering exactly WN18RR's 40,943 ids.
- **Regions.** O = synsets occurring in a WN18RR triple (41,105); U = V \ O (71,094). **W_rule = 0.633642** (entity share of U).
- **Benchmark triples.** 92,998 of 93,003 WN18RR triples map onto the target; 5 do not (4 train, 1 test; `unmatched_wn18rr.tsv`) and are dropped.
- **Triple roles.** `bench_train` 86,831, `bench_valid` 3,034, `bench_test` 3,133 (mapped WN18RR splits); `u_valid` 3,034 and `u_test` 3,134 (held-out triples with at least one U end, sampled with seed 20261018 so that both ends keep a training triple); `extra` 119,201 (everything else).
- **E2 training set** = `bench_train` + `extra` = 206,032 triples. Every U synset has at least one E2 training triple. 182 O synsets have none: they occur only in WN18RR's valid/test, which is the benchmark's own unseen-entity issue, kept as is.
- **Filtering.** Known true answers are removed from the candidates using all 218,367 target triples. 2,055 of the 6,266 benchmark test queries have at least one true answer in U: 3,065 true tails and 28,921 true heads that the benchmark never shows as candidates.

**Composition** (`results/wn18rr_target/composition.txt`):

| | O | U |
|---|---|---|
| POS shares n / v / a / r | 0.782 / 0.189 / 0.029 / 0.000 | 0.703 / 0.083 / 0.213 / 0.001 |
| E2 training degree, median (90th pct) | 4 (12) | 2 (4) |
| Share with training degree 1 | 0.078 | 0.495 |

Restricting candidates to the POS seen in each relation slot moves the mean per-query U share only from 0.634 to 0.607. Degree is the larger difference and gets its own sensitivity analysis (S2).

## 3. Two experiments

| | E1: benchmark as published | E2: declared target |
|---|---|---|
| Training data | WN18RR train, original files | E2 training set (Section 2) |
| Positive reference | WN18RR test triples | the same 3,133 mapped test triples |
| Candidates scored | O only | O and U, enumerated (no sampling) |
| Answers | reproduction; benchmark ranking | magnitude, ranking on the target, c |

Embedding models trained in E1 have no vectors for U synsets, so E1 cannot score U. That is a fact about deploying a benchmark-trained transductive model, not a gap in the design. E2 trains on the target, as KG completion is deployed on a whole graph.

E1-O and E2-O differ in training data only. That difference is reported as the **training-side effect**, separately from the evaluation-side effect (E2-O vs E2-U). The two are never summed. *[Corrected 2026-10-21, after the results; see Section 14, item 1: "training data only" is wrong. Synset identity and the filter differ too, so this is a training/protocol effect.]*

## 4. Estimands and metrics

**Primary metric: per-query rank AUC.** For a query q (head or tail prediction of a test triple) with positive p and filtered candidate set C_q = O_q ∪ U_q:

- AUC_q(S) = share of candidates in S scored below p, ties counted 1/2.
- Exact decomposition: **AUC_q(C) = (1 − W_q)·AUC_q(O) + W_q·AUC_q(U)**, with W_q = |U_q| / |C_q|.
- Target score = mean over the 6,266 queries of AUC_q(C). Region scores are the means of AUC_q(O) and AUC_q(U).

This is [P2]'s linear identity applied per query. It is chosen because it carries the identification argument, not because it is the field's headline.

**Model pair (A, B).**

- δ_O,q = AUC^A_q(O) − AUC^B_q(O), and likewise δ_U,q.
- Δ_O = mean δ_O,q; Δ_U = mean δ_U,q.
- Δ_target = mean[(1 − W_q) δ_O,q + W_q δ_U,q], computed exactly.
- **c = Δ_O − Δ_U** (the regional difference); **c\* = |Δ_O| / W̄** (break-even), where W̄ is the mean W_q.
- Δ_target = Δ_O − W̄·c holds only up to the covariance of W_q and δ, so the exact Δ_target is the reported quantity. c and c\* are reported with the size of that approximation error.

**Secondary metrics (field convention, not linear in the mixture):**

- MRR and Hits@1/10 over O (the benchmark's number) and over C, using the realistic rank (ties averaged).
- **U intrusion:** the share of queries whose positive is outranked by at least one U candidate, and the mean number of U candidates in the top 10.

These are reported for every model and pair but carry no identification claim.

**Secondary estimand, query side (kept separate):** `u_test` queries (strata by head–tail region: U–U 950, U–O 1,414, O–U 770). They give E2 model metrics on queries about excluded synsets, compared with `bench_test`. This changes the positive reference, so it is never mixed into W or c.

## 5. Sensitivity analyses (prespecified)

- **S1 Type-restricted candidates.** Keep only candidates whose POS was seen in the relation slot in E2 training.
- **S2 Degree-matched U.** Within each query, reweight U candidates so that their E2 training-degree distribution matches O's. Bins: 1, 2, 3, 4–5, 6–10, 11+. This separates "excluded" from "rare". If the O–U difference vanishes under S2, it is stated as a rarity effect, not a property of the eligibility rule (compare the within-protein analysis in [P3]). *[Wording clarified 2026-10-21, after the results: Section 14, item 2.]*
- **S3 Benchmark identity.** Key synsets by bare offset as WN18RR does, merging the 161 ids.
- **S4 Trained O only.** Drop the 182 O synsets without an E2 training triple from the candidates.

## 6. Model ladder

| Tier | Model | Source of hyperparameters | Published WN18RR MRR |
|---|---|---|---|
| T0 | relation-slot popularity: score(e) = count of e in that slot of relation r in training | none | — |
| T1 | TransE | Ruffinelli et al. 2020 (ICLR), LibKGE config `wnrr-transe.yaml` | 0.228 |
| T1 | ConvE | same, `wnrr-conve.yaml` | 0.442 |
| T1 | DistMult | same, `wnrr-distmult.yaml` | 0.452 |
| T1 | RESCAL | same, `wnrr-rescal.yaml` | 0.467 |
| T1 | ComplEx | same, `wnrr-complex.yaml` | 0.475 |
| T2 | NBFNet | Zhu et al. 2021 (NeurIPS), official code and config | 0.551 |

- The T1 configs are downloaded (`pilot_data/kge_configs/`). They are used unmodified in E1 and E2: no tuning by us, so no researcher degrees of freedom in the hyperparameters. The published values are from the kge-iclr20 repository (T1) and the NBFNet paper's Table 3 (T2).
- **Close pairs** (published |ΔMRR| ≤ 0.025): ComplEx–RESCAL 0.008, DistMult–ConvE 0.010, RESCAL–DistMult 0.015, ComplEx–DistMult 0.023, RESCAL–ConvE 0.025. These are where a reversal is plausible; all 21 pairs among T0–T2 are reported anyway.
- **NBFNet** is conditional on a feasibility check: one E1 run within 24 h on one HPC L40S/A100 (the paper used 4 V100s). It is the only ladder model that can score U in E1, because it reads a graph rather than entity vectors. That allows one extra comparison: the benchmark-trained model deployed on the E2 graph.
- **Seeds** 0–4 for every learned model in every experiment.
- **Reproduction gate:** E1 seed-0 MRR on WN18RR is compared with the published value before any E2 run. A gap is reported (as [P3] reported MolTrans's 0.9363 vs 0.952) and not fixed by retuning.
- **Early stopping** uses the configs' own criterion: validation MRR on `bench_valid` over the benchmark's candidates O, which is what a benchmark user does. See Section 12, item 1.

## 7. Comparisons and decision language

- **Resolved:** a pair is resolved in a region when the 95% interval of its Δ excludes zero.
- **Reversal:** Δ_O and Δ_target are both resolved, with opposite signs.
- **Newly resolved:** Δ_O unresolved and Δ_target resolved (the BioT5 pattern in [P3]).
- **Reporting:** every pair, every seed and the seed mean. The paired statistic Δ_U − Δ_O comes from the same bootstrap replicate. *[Multiple comparisons, clarified 2026-10-21: Section 14, item 3.]*
- **Magnitude:** per model, AUC(U) − AUC(O) with an interval, plus the same for MRR(C) − MRR(O).

## 8. Uncertainty

- Candidates are enumerated, so the only sampling is over test queries and seeds.
- **Bootstrap:** 2,000 replicates over the 3,133 test triples (each triple carries its head and tail query), with keyed RNG streams so that reruns reproduce every interval.
- **Sensitivity:** clustering by the triple's head synset.
- **Seeds:** the seed spread is reported next to the bootstrap intervals, not folded into them.

## 9. Outputs and provenance

- **Per model × seed × query:** n_O, n_U, O_above, O_ties, U_above, U_ties, the positive's score, query_id and triple_id. These counts give AUC, rank and Hits exactly, in every region and under S1–S4 (S1/S2 also need per-candidate POS and degree, so those analyses compute their counts at scoring time).
- **Weights:** saved for every run.
- **Statistics:** go into a keyed `records.tsv` (design / estimand / subject / seed / quantity). Numbers in the text are tagged with their record keys and checked against the records by a script.

## 10. Compute and environment

- **E1:** 41k entities, 87k training triples, 1vsAll scoring.
- **E2:** 112k entities and 206k triples, so roughly 6× E1's cost per epoch.
- **Runs:** 5 T1 models × 5 seeds × 2 experiments = 50 runs, plus T0 (deterministic) and NBFNet if feasible.
- **Hardware:** the T1 runs are expected to fit the laptop RTX 4070 (to be measured by a timing-only smoke test before the runs); NBFNet goes to HPC.
- **Environment:** a new WSL env for this paper (LibKGE at a pinned commit, torch 2.4 cu124).
- **Scoring:** a custom scorer calls each model's score_sp / score_po over all target entities, so that the evaluation is the same code for every model and does not depend on LibKGE's evaluator.
- **Bit-reproducibility:** cuDNN nondeterminism applies, and the 5 seeds show its size.

## 11. Threats and limits stated in advance

- WN18's entity rule is not reproducible, so the experiment measures the effect of the benchmark's **entity set**, not of a stated threshold.
- U is rarer than O (Section 2). S2 is the guard against attributing rarity effects to exclusion.
- E2 changes training data. The training-side effect is reported separately (Section 3).
- **Filtering:** in the target, 2,055 test queries lose true U answers from their candidate lists. A benchmark-trained model never faced these synsets at all.
- **Scope:** one benchmark, one domain. The result is a measured c for WN18RR, not a prevalence estimate.

## 12. Decisions (the author, 2026-10-18, all at the proposed defaults)

1. **Model selection in E2:** on `bench_valid` over O (benchmark practice). Selection over C is not run.
2. **NBFNet:** included, conditional on the feasibility check in Section 6. *Scope reduced on 2026-10-19: see Section 13, item 1.*
3. **Primary metric:** per-query AUC; MRR and Hits@k co-reported.
4. **Query-side estimand:** the `u_test` analysis stays as a secondary result, reported separately.

## 13. Deviations from the fixed design

Each was decided before any result it could depend on existed.

1. **NBFNet scope reduced (the author, 2026-10-19).**
   - **Why:** the feasibility check (job 4268184; the run log is not part of this copy) passed: one E1 run takes about 9 h on an A100 with the rspmm kernel. But E2 has 2.4× the edges and training triples and 2.7× the nodes, which puts one E2 run at roughly 50–56 GPU-hours and five seeds at about 260.
   - **What is run instead:** NBFNet is trained in **E1 only** (5 seeds, official config). It is then used for a **deployment test**: each benchmark-trained model reads the E2 training graph at inference time and scores the `bench_test` queries against O and U candidates, filtered with every target triple. This is the "benchmark-trained model deployed on the E2 graph" comparison of Section 6. Its analysis is item 4.
   - **Consequences:**
     - NBFNet is **not** in the primary estimand (E2). The E2 ladder is T0–T1, and the 21 pairs of Section 6 become 15 (T0 and the five T1 models).
     - The deployment test is a separate estimand, reported apart from E2. No embedding model can do it, so it has no T1 comparison.
     - No NBFNet E2 result exists or was looked at when this was decided.
2. **U intrusion, top-10 part (2026-10-19, while writing the analysis).** "The mean number of U candidates in the top 10" (Section 4) cannot be computed from the stored counts: the scorer records how many U candidates rank above the positive, not their absolute ranks.
   - Reported instead, both exact from the counts: the share of queries whose positive is outranked by at least one U candidate (as designed), and the share of queries that are in the top 10 over O but drop out of it over C.
   - When this was decided, the only E2 numbers seen were the O-restricted test MRRs in the job status report; no O–U comparison had been computed.
3. **Post hoc analyses (2026-10-19). These are NOT prespecified.** They were added *after* seeing the first E2 results for ComplEx and DistMult, where `u_test` U→U queries had MRR about 0.85.
   - **What was found:** a test triple *leaks* when its reverse (t, r, h) is an E2 training triple. Through the symmetric WordNet relations (`derivationally_related_form`, `also_see`, `verb_group`, `similar_to`) such queries are answered by one training edge. This covers 2,172 of 6,266 `bench_test` queries and 3,568 of 6,268 `u_test` queries; leaking queries have MRR ≈ 1.
   - **The benchmark has this property itself.** Dettmers et al. 2018 note that WN18RR keeps symmetric relations, and the leak count above uses the E2 training set.
   - **Added**, each labelled `posthoc_*` in `records.tsv`:
     - (a) query-side metrics by relation group and leak flag, for `u_test` and `bench_test`;
     - (b) the primary estimand and its pairs restricted to the 4,094 non-leaking `bench_test` queries (`posthoc_noleak`).
   - **Use:** neither replaces a prespecified estimand. They are reported as robustness or explanation, with their post hoc status stated.
4. **Deployment-test analysis specified (2026-10-19, before any trained NBFNet model was scored).** Item 1 named the deployment test but not its analysis. The analysis was written into `analyze_wn18rr.py` and `summarize_wn18rr.py` while NBFNet array 4270011 was still pending. The only NBFNet scores that existed came from a random-init model used to test the adapter.
   - **Queries and candidates:** the 6,266 `bench_test` queries, with the 42 candidate cells and the target filter of E2. `u_test` is not scored in the deployment test.
   - **Quantities:** for each seed and the seed mean, the per-model quantities of Section 4:
     - W; AUC over O, U and C; AUC U − O;
     - MRR and Hits@1/10 over O and C; MRR C − O;
     - U intrusion and top-10 drop (item 2).
   - **Estimands:** the primary estimand, S1, S2 and S4. Also, labelled post hoc as in item 3, the non-leaking queries (`posthoc_noleak`) and the query-side leak groups.
   - **No pairs.** NBFNet is the only model deployed this way, so the deployment test has no Δ, c or c\*.
   - **Graph-side effect:** deploy-O minus E1-O for the same checkpoint, in AUC and MRR, per seed. It is reported like the training-side effect of Section 3 and never summed with the evaluation-side effect. It changes three things at once: the graph read at inference, the filter (benchmark triples → target triples) and the O keying (the 161 merged ids are split).
   - **Uncertainty:** as Section 8: 2,000 triple-bootstrap replicates with keyed RNG, and head-synset clustering for the primary estimand.
   - **Checks that must pass before any number is used:**
     - the deploy query set and per-cell candidate counts equal E2's;
     - deploy and E1 scores come from the same checkpoint (SHA-256 in `meta.json`);
     - our E1 MRR is within 2e-3 of run.py's own test MRR. run.py includes the one unmapped test triple and breaks ties pessimistically.
   - **Records:** design `deploy` with E2's estimand names, plus `graph_side`.

## 14. Corrections and clarifications added after the results (2026-10-21)

**Status of these items:**
- They follow the Codex review of 2026-10-19, whose report is not part of this copy. The author approved them on 2026-10-21.
- They were written after every result had been seen, including S3. They are therefore not part of the prespecified design.
- They change no analysis, estimand, decision rule or number. The original text above is kept, with a pointer to the relevant item here.

1. **Section 3, "E1-O and E2-O differ in training data only", is wrong.** Three things change together between E1-O and E2-O:
   - the training data (WN18RR train vs the E2 training set);
   - synset identity (WN18RR's bare offsets, with 161 merged ids, vs POS + offset);
   - the filter (benchmark triples vs every target triple).

   The difference is therefore called the **training/protocol effect**. Like the deployment test's graph-side effect (Section 13, item 4), it is not attributed to any one of the three, and it stays separate from the evaluation-side effect.
2. **Section 5, S2 wording.** S2 reweights U candidates to O's training-degree distribution, so it changes the comparison distribution. When an O–U difference, or a pair's reversal, vanishes under S2, the rule in Section 5 still applies: it is reported as a rarity effect and not as a property of the eligibility rule. The write-up says what S2 shows, which is that the result is sensitive to degree composition. It does not claim that rarity has been shown to be the mechanism.
3. **Section 7, multiple comparisons.** No correction was prespecified.
   - **What the intervals are:** per pair and per estimand, unadjusted across the 15 pairs and the estimands.
   - **What they cover:** test-triple resampling with the five trained seeds held fixed. Seed variation is reported separately (Section 8).
   - **In the text:** wherever a single pair's result is highlighted, the write-up states this.
