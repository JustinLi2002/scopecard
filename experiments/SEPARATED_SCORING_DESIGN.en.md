# Separating selection from scoring: a local check across prompt formats

> **English translation.** Made on 2026-11-17 for the reproduction package, from the design as fixed in Chinese on
> 2026-11-16 (SHA-256 `366a45ac1f292d076b38cc86949ab1e046dc85a1c3bf9ca21d817d711cefcab0`, the `design_sha256`
> recorded in `results/separated_scoring/run_manifest.json`). The original is not part of this package. A rerun
> from this package reads this translation, so the `design_sha256` it records is the SHA-256 of this file. Notes
> in square brackets are the translator's.

2026-11-16, America/Chicago. Status: fixed before the crossed scores, effect sizes and intervals of this experiment
were computed. Exploratory analysis; not a preregistration and not a validation on new, unseen data. Earlier studies
analysed the two matrices separately; this experiment is the first to fix the crossed use described below. The author
authorised the run.

## Question and data boundaries

An earlier round found that when binary item outcomes are used both to filter items by a threshold and to score, the
effect of a model's own participation is necessarily non-negative. This experiment filters items with the outcomes of
one prompt format only and scores the outcomes of the other prompt format, to test whether the effect carries over
across prompt formats.

It uses the 1,140 tasks of the existing BigCodeBench v0.1.4 and the **124 models whose names are identical** in the
Complete and Instruct tables. Models are not selected by score, direction, family or significance, and different
aliases are not matched. The two tables have the same upstream revision `249b3054d25ae61746e8fbe014782ebe4526f844`;
the hashes of the input files must match the existing manifest.

Complete and Instruct are outcomes under different prompt formats: the latter replaces the detailed docstring with a
shorter natural-language instruction. The two share tasks and many models; they are not statistically independent
data sets, nor repeated samples of one protocol. The public tables hold model names only, so the weight hashes or
generation seeds of the runs cannot be verified. This experiment's wording is "scored in another prompt format that
did not vote"; it does not say "newly run, independent model inference". [Original paper,
§2.4](https://arxiv.org/html/2406.15877v5)

## The fixed experiment

- Main analysis: **Complete filters → Instruct scores, p = 8**. 8 is a moderate panel size chosen in advance; it also
  matches the size of the filter panel that MMLU-Pro published. That rule and those models are not reproduced here.
- Supplementary: p ∈ {4, 16, 32, 64}, and the reverse direction, Instruct filters → Complete scores, at all five
  sizes. The reverse direction is not an independent replication.
- At each size, 100 panels drawn uniformly at random without replacement. For p ≤ 16 every member of a panel is
  evaluated; otherwise 16 members are drawn uniformly. Random seeds are fixed by string hashes; both directions use the
  same panels, members and replacement plan.
- For each focal member m, another panel member u and a replacement model r from outside the panel are drawn at
  random. Apart from the three being distinct, no family is excluded, so that an unverified family assignment does not
  enter the main analysis. This differs from the family exclusion of the earlier study, so the sizes of the numbers
  cannot be compared directly.
- Self set Hₛ: after replacing m by r, keep the items on which **strictly fewer than p/2** members are right.
- Reference set Hᵣ: after replacing u by the same r, keep items by the same threshold.
- Filtering reads only the source matrix. The target matrix cannot change the panels, members, replacements,
  threshold or item sets. No hyperparameter is chosen by target scores.
- For each comparison, another 10 models that never vote in either of the two panels are drawn and scored in the
  target format, as a control for the change in the overall difficulty of the sets. Models of the same family are not
  excluded.
- Panels are weighted equally, and members equally within a panel. All comparisons are reported; no "successful"
  panels are selected. An empty set, a non-binary input, a missing task or a hash mismatch stops the formal analysis;
  nothing is filled with zeros or redrawn to make up for it.

## Measures and readings

S_R(z) is the mean pass rate of the scoring vector z over the item set R.

1. `source`: S_Hₛ(y_m) − S_Hᵣ(y_m). A control that reproduces the quantity that is non-negative by mathematics; it is
   not used to show that an unknown mechanism exists.
2. `target`: S_Hₛ(z_m) − S_Hᵣ(z_m). z is the outcome in the other prompt format, which took no part in filtering; it
   can be negative.
3. `outside`: the mean of the same target difference over the 10 non-voting models. It uses exactly the same two item
   sets as m and is never mixed with the original panel's item sets.
4. **Primary measure `excess = target − outside`.** It measures whether the focal model is affected in the other
   prompt format, in the same direction, more than outside models are; it does not by itself mean a "bias" relative to
   a correct target.

Two-sided 95% interval of the main analysis's excess: a lower bound above 0 means that a positive carry-over across
prompt formats is detected; an upper bound below 0 means the opposite direction; otherwise it is unresolved. Support
for at least 1 percentage point is claimed only if the lower bound is at least 0.01; 1 point is a display threshold
agreed for this run, not a general deployment standard. Effect sizes and intervals are kept in full; the threshold
does not replace the continuous result.

Supplementary displays: the point ratio target/source (defined only when source > 0), each model's target effect, the
share of member events with a negative effect, and the target pass rates on the items that enter and leave. The
distribution of effects across panels and the Monte Carlo standard error of its mean describe random-panel variation;
they are not treated as confidence intervals for a new population of models.

Rankings describe only the point sign changes between Hᵣ and Hₛ under target, for member–outside model pairs and
outside–outside pairs; comparisons tied before do not enter the denominator. The share of reversals whose absolute
differences before and after are both at least 1 point / 2 points is also reported. Comparison events repeat the same
model pairs, so the share cannot be called a share of unique model pairs; no ordinary bootstrap significance judgement
is given for sign-change rates.

## Uncertainty and reliability checks

- 2,000 task-paired bootstrap replicates: each draws 1,140 times with replacement from the 1,140 task IDs; all
  models, both prompt formats and all panel sizes share the same resampling weights.
- Panels, rules and each item's source votes are frozen. Computing set means with the resampling weights is
  equivalent to copying each task and filtering it again by the same per-item rule; source and target tasks are not
  resampled separately.
- Percentile intervals are reported for each quantity; the main analysis is the one prespecified excess contrast
  above. The curves also get **an approximate max-|t| interval joint over 20 target/excess contrasts (2 directions × 5
  sizes × 2 measures)**: centred bootstrap differences are divided by each column's bootstrap standard deviation, and
  the 95% quantile of the largest absolute value is taken. It is not an exact finite-sample guarantee.
- These intervals are conditional on the current models, archived runs and random panels, and reflect task
  resampling only; they do not cover the uncertainty of new training runs, of a population of models or of historical
  filtering protocols.
- Checked on synthetic data: when target = source the identity holds; when target = 1 − source the effect flips; with
  a constant target it is zero; replacing target does not change the sets; the vectorised weighted results equal
  explicit resampling. A small batch of direct calculations on the real data is also checked, to avoid implementation
  errors.
- Figures and conclusions always keep the qualifiers "across prompt formats", "exploratory" and "fixed archived runs".

## Sources and outputs

Existing inputs are only read. Outputs go to `experiments/results/separated_scoring/`: hashes of the plan and code, data
checks, the panel plan, per-event and per-model results, summary intervals, bootstrap distributions and static curves.
Script `experiments/scripts/separated_scoring.py`; the conclusions are written to `experiments/SEPARATED_SCORING_FINDINGS.md`
[a working note, not part of this package].

The frozen manuscript is not changed, and the outputs of the earlier studies are not overwritten. The 24 existing protected files are
checked again [a list of the authors' working files, not part of this package; a rerun from this package checks
instead that no file listed in its `SHA256SUMS` changes during the run]. If no carry-over is detected in this
round, that is reported as it is; the main analysis is not replaced by switching direction or choosing models.
