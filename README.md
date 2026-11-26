# ScopeCard

ScopeCard asks which part of a declared target population a benchmark can evaluate, and what comparisons on the benchmark therefore license. It accompanies the paper *Which population does a benchmark ranking describe?* by Justin Li (2026). This repository holds:

- the audit tool;
- the card library;
- the code and stored outputs of the WN18RR measured-c experiment;
- the designs, code and stored outputs of the filter studies, which ask what a benchmark's filter on model outputs keeps: released subsets of SWE-bench and BigCodeBench, panel filters on BigCodeBench, the filter of MMLU-Pro and its reconstruction on MMLU.

The browser page runs an audit without installing anything, and the files you choose stay on your machine: https://justinli2002.github.io/scopecard/app/. The card library is at https://justinli2002.github.io/scopecard/.

The numbers reported in the paper come from the stored outputs in this copy, from the designs and from constants in the code. `reproduce.py` regenerates the stored outputs and compares them with the copies here; the tiers below say which. The validation runs V2 and V3 are the exception: their scripts (`validate_synthetic.py`, `validate_refusals.py`) are here and write their outputs to `experiments/results/validation/` again, but `reproduce.py` does not run them.

This is version 0.1.0 of the copy. It adds the filter studies to version 0.0.1 and renames the audit tool, which was called BenchScope up to version 0.0.1. Its version 0.0.2 is version 0.0.1 under the new name: the package, the command and the card schema's id are now `scopecard`, and every number the tool reports is unchanged. Records made before the rename keep the old name.

[P2] and [P3] in these files are two earlier studies by the author, cited in the paper:

- **[P2]** J. Li. Constructed negatives determine the population a benchmark can rank over. Research Square preprint, 2026. https://doi.org/10.21203/rs.3.rs-11252843/v2
- **[P3]** J. Li. What a low-bias filter leaves out: measured performance on the excluded negatives of a drug–target interaction benchmark. ChemRxiv preprint, 2026. https://doi.org/10.26434/chemrxiv.15010766/v1

## Contents

| Path | What it holds |
|---|---|
| `tool/` | **The `scopecard` package**: a command line with seven audit types, with its tests. Also here: the browser page (`web/`), the card library (`library/`; the built site is `site/`) and the example cards. See `tool/README.md`. |
| `experiments/MEASURED_C_DESIGN.md` | **The design of the WN18RR experiment.** Changes made after the design was fixed are dated in place. Clarifications written after the results are in Section 14. |
| `experiments/scripts/` | **Scripts:** declared-target construction and acceptance checks, the scorers, the prespecified analysis, and the figures. `make_fig_filters.py` draws the figure on filtering by model outputs from the stored records, and `check_fig_filters.py` checks every value it draws against its record. `hpc/` holds the GPU job templates. The pilot's first-pass scripts, the validation scripts and the scripts of the filter studies are here too. |
| `experiments/results/wn18rr_runs/` | **Per-query scorer outputs of the 87 runs**: E1 31, E2 26, S3 25, deployment 5. |
| `experiments/results/wn18rr_analysis/` | **`records.tsv`** (every reported value, keyed by design, estimand, subject, seed, quantity and cluster) and its formatted `summary.md`. |
| `experiments/results/wn18rr_target*/` | **Hashes and summaries of the two declared targets.** The targets themselves are rebuilt from WordNet 3.0, not shipped. |
| `experiments/results/compute/` | **GPU accounting** of the WN18RR runs and the GPU-hours computed from it. |
| `experiments/results/kg/`, `experiments/results/validation/` | **First-pass outputs of the pilot** (WN18, FB15k-237, YAGO3-10) and **the tool's validation** (V2 known rules, V3 refusals, V5 clean-environment timings). |
| `experiments/figures/` | **Figures 2–4, Tables 1–2 and Box 1, and the figure on filtering by model outputs** (`fig_filters.pdf`), with `fig_filters_values.tsv`, which lists every value that figure draws and the record it comes from. |
| `experiments/env/`, `experiments/configs/` | **Setup:** pinned LibKGE and NBFNet commits, their patches, the package lists of the GPU environments, the E1 training configs, and the package versions of the filter studies (`requirements-filters.txt`). |
| `experiments/pilot_decidability.tsv` | **The pilot coding** of the ten benchmarks. |
| `experiments/*_DESIGN.md` (filter studies) | **Designs of the filter studies**, each dated and fixed before the quantities it defines were computed, with later changes, deviations and corrections dated in place: `SUBSET_C` (released subsets), `PANEL_FILTER` (panel filters and the panel-size relation), `SEPARATED_SCORING` (selection and scoring across prompt formats; `SEPARATED_SCORING_DESIGN.en.md` is the English translation of the design as fixed in Chinese, which is not in this copy), `PANEL_COMPOSITION`, `MMLUPRO_FILTER` and `MMLU_RECONSTRUCTION`. |
| `experiments/results/subsets/`, `panel_filter/`, `separated_scoring/`, `panel_composition/`, `mmlupro_filter/`, `mmlu_reconstruction/` | **Stored outputs of the filter studies**: summaries, records and the logs of the original runs. Each `raw/` folder holds the manifest of the study's downloads and the other records of the fetch, not the downloaded files. `mmlu_reconstruction/correct_matrix.npz` is the per-question matrix of the 268 systems on MMLU. |
| `reproduce.py`, `inputs.json` | **One-command reproduction**, plus the public inputs of the tool and the WN18RR experiment with their SHA-256 hashes. |
| `replay_inputs.py`, `filter_inputs.json` | **Inputs of the filter studies:** the script that downloads them again and checks them, and what its checks need. |
| `COPY_MAP.tsv`, `SHA256SUMS` | **Records of this copy:** for each file, its source, the SHA-256 of the source and of the copy, and the lines a rewrite changed; and the SHA-256 of every file. |

Exploratory studies that the paper reports are included with their designs and their records of deviations. Exploratory analyses that the paper does not use are not included.

**Run records are historical.** The logs, `run_manifest.json`, `verification.json` and `SHA256SUMS.json` of a study record the original run, with the hashes of its files as they were then; some of them name working files that are not in this copy. This copy changes some files: paths and names; the words in Chinese that the designs quote with each approval; references to files and working notes that are not here; and the cross-prompt run's script, whose guard here checks this copy's files instead of a list of the authors' working files and which reads the English translation of its design. That design, the one written in Chinese, is replaced by its translation. `COPY_MAP.tsv` gives the SHA-256 of each original next to that of its copy.

## Manuscript figure mapping

The component files in `experiments/figures/` keep an earlier numbering. `figB1_benchmarks.pdf` and `fig2_audit.pdf` make up the figure of what the ten benchmarks exclude; `fig4_conve_distmult.pdf` is the ConvE − DistMult figure; `fig_filters.pdf` is the figure on filtering by model outputs. In Extended Data, `table1_benchmarks.tex` is the table of the ten benchmarks, `box1_wn18rr_card.md` is the WN18RR scope card, shown there abridged, and `fig3_wn18rr_models.pdf` and `table2_models.tex` are the per-model figure and table. The tables of the filter studies are typeset in the manuscript from the records in `experiments/results/`. The `paper` tier regenerates the components of the WN18RR experiment and their records, and the `filters-figure` tier the figure on filtering by model outputs. Neither compiles the manuscript, which is not part of this repository, or redraws the schematic of Fig. 1, which is drawn by hand.

## Reproduce

You need Python 3.9 or later. No GPU is needed. The `paper` and `filters-figure` tiers, and the last step of
`filters`, also need LaTeX with the `helvet`, `type1cm` and `underscore` packages, because the figures are typeset
with Matplotlib's usetex so that they embed only Type 1 fonts. PyMuPDF (`pip install pymupdf`) is optional; without it the figures are drawn as PDF only. The filter studies
also need scipy and pyarrow; `experiments/env/requirements-filters.txt` pins the versions they were run with.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install "./tool[test,repro]"     # numpy, matplotlib, pytest
python reproduce.py --tier quick
python reproduce.py --tier filters-figure
python reproduce.py --tier paper --fetch
python reproduce.py --tier cards --fetch
pip install -r experiments/env/requirements-filters.txt
python reproduce.py --tier filters --fetch
```

`--fetch` downloads any missing input from the public source listed in `inputs.json` into `experiments/pilot_data/`, or, for the filter studies, from the sources in their manifests (see below). Every input file is checked against its SHA-256 before use. Outputs go to `repro_out/`; nothing stored here is overwritten. Each step's time and result are written to `repro_out/timing.tsv`. A step fails when its output differs from the stored copy. Compressed files (`.gz`, `.zip`) are compared after decompression, because their bytes depend on the zlib build: Python 3.14 on Windows ships zlib-ng.

| Tier | What it does | Inputs | Time |
|---|---|---|---|
| `quick` | Runs the tests that need no data: synthetic cases with known answers, plus the registry, schema and site checks. | none | 1.5 s |
| `paper` | Rebuilds both declared targets from WordNet 3.0 and WN18RR, checks them against the recorded hashes, and runs the acceptance checks. It then reruns the prespecified analysis on the stored scorer outputs and compares every record. It redraws Figures 2–4, Tables 1–2 and Box 1, and compares the tables and Box 1 with the stored copies. Finally it recomputes the GPU-hours from the stored accounting records. | WordNet 3.0, WN18RR (41 MB) | 5 min |
| `cards` | Rebuilds every library scope card from the public sources, along with the example cards, the browser page and the library site, and compares each file with the stored copy. Then it runs the full test suite, including the golden tests, and the `memory` step. | all of `inputs.json` (3.9 GB) | 22 min + `memory` |
| `memory` | Runs the YAGO3-10 and Amazon-Book audits alone, each in a child process whose peak resident memory is measured, and compares their cards with the stored ones. | those two benchmarks' inputs (1.4 GB) | 17 min |
| `filters` | Checks the inputs of the filter studies (below), then reruns every study and compares every stored output: the released subsets, the panel filters with the panel-size relation, the cross-prompt check, the panel composition, the MMLU-Pro test, and the MMLU analyses from the stored per-question matrix. Finally it draws the figure on filtering by model outputs from the rerun's records and checks every value it draws against them. | 255 MB (121 files) | 4.5 h |
| `filters-upstream` | Runs `filters`, then rebuilds the MMLU per-question matrix from the leaderboard's records of the 268 systems and compares it with the stored matrix, field by field. | a further 6.3 GB (15,334 files) | `filters` + 5 min |
| `filters-figure` | Draws the figure on filtering by model outputs from the stored records of the filter studies, checks every value it draws against its record and each record file against its SHA-256, and compares the list of drawn values with the stored copy. | none | under a minute |
| `all` | Runs `paper`, `cards` and `filters`. | | |

**How outputs are compared.** Text files and tables must be byte-identical, with three exceptions, each listed in the step's report. A text file may differ in a printed number by at most one unit of its last printed digit. A JSON file is compared field by field, with numbers equal to a relative 1e-9. An `.npz` file is compared array by array, because its compressed bytes depend on the zlib build. Five fields of the cross-prompt run are left out of the comparison: its start time and run time, which are not checked; the SHA-256 of its script and of its design, each checked against the file in this copy that the rerun read; and the number of files its guard checked, checked against `SHA256SUMS`. Two files are checked in their own way: the checksum list of the cross-prompt run, whose hashes must match the files it names; and the acceptance log of the MMLU analyses, whose lines on the build of the matrix are not rerun when the analyses start from the stored matrix. Figures are drawn but not compared, because PDF files carry their creation time; the figure on filtering by model outputs is compared through its list of drawn values. The logs of the original runs are kept as they are and are not compared.

**Measured times.** All on one Windows 11 laptop with 22 logical CPUs and 15 GB of memory, each tier in a fresh virtual environment with its inputs already present.

- **Version 0.0.1** timed `quick`, `paper` and `cards`: Python 3.12.5, numpy 2.5.4, matplotlib 3.11.2, pytest 9.2.0, with TeX Live 2025 for the figures and without PyMuPDF. Creating the environment and running `pip install` took under a minute; `--fetch` adds 12 MB of downloads to the paper tier, a few seconds. Rebuilding the library cards took 11 minutes and the full test suite 11.
- **This version** ran with Python 3.14.3: numpy 2.5.5, matplotlib 3.11.2 and pytest 9.2.0, and for the filter tiers the packages in `experiments/env/requirements-filters.txt`. Other jobs were running on the laptop at the same time, so these times are upper bounds. `quick` took 7 s and `paper` 6 minutes. `memory` took 17 minutes; in version 0.0.1 the same two audits took 3 and 5 minutes.
- **`filters`** took 4 h 25 min. The MMLU analyses took 2 h 48 min, 2 h 8 min of it the panel-size curve. The panel filter took 55 minutes and the panel composition 38; in an earlier run under less load they took 35 and 2.5 minutes. `filters-upstream` added 2 minutes to check the 6.3 GB of records and 3 minutes to rebuild the matrix.
- **Downloads with `--fetch`:** about a minute for the 255 MB of the filter studies, and 44 minutes for the further 6.3 GB. Most of that time is pacing: the 6.3 GB come in 15,334 files, and `replay_inputs.py` sends at most six requests a second.
- **The rename.** All these times were measured before the tool was renamed from BenchScope to ScopeCard (2026-11-23). The rename changes no computation, and the times were not measured again. On the renamed copy, `quick`, `paper`, `cards` and `filters-figure` were run again (Python 3.14.3, numpy 2.5.1, not in a fresh environment) and every step passed.

**Not rerun: GPU training and scoring.** The paper tier starts from their stored per-query outputs. To rerun them:

1. Install LibKGE and NBFNet at the commits in `experiments/env/`, apply the patches there, and match the package lists in `env_freeze.txt` and `env_nbf_freeze.txt`.
2. **E1 (WN18RR as published):** train with the unmodified configs in `experiments/configs/e1/`.
3. **E2 and S3 (the declared targets):** build the training data with `export_libkge_e2.py` and `make_o_candidates.py`, and the configs with `make_e2_config.py`.
4. **Score each checkpoint:** use `score_wn18rr.py` (LibKGE models), `score_popularity.py` (the popularity baseline) or `score_nbfnet.py` (NBFNet).

The SLURM templates in `experiments/scripts/hpc/` show the exact commands; site paths in them are replaced by `/home/anon`.

## Inputs of the filter studies

The filter studies read files that their fetch scripts downloaded in November 2026. Each fetch script wrote a manifest next to the files (`raw/manifest.tsv`, and `raw/manifest_alt.tsv` for the earlier MMLU runs): one row per file, with its path, size, SHA-256, source, upstream revision and date. The downloaded files are not redistributed. `replay_inputs.py` downloads them again from the recorded sources and checks every file, and `filter_inputs.json` names the manifests and holds what the checks need. The fetch scripts are kept as the record of how the files were first collected; they are not run again.

```bash
python replay_inputs.py replay filter_inputs.json subsets --into DIR      # or mmlupro_filter, mmlu_questions,
python replay_inputs.py replay filter_inputs.json subsets --check-only    #    mmlu_details, mmlu_details_alt
```

`reproduce.py` puts the files under `experiments/pilot_data/filters/`, or under the folder named by the environment variable `SCOPECARD_FILTER_INPUTS`. On Windows, unless long paths are enabled, a path may have at most 259 characters: the per-question MMLU records need up to 166 characters below the inputs folder, and the studies' staged copy needs up to 135 below the `--out` folder. `reproduce.py` checks both before it copies or downloads anything into these folders, and if a path would be too long it stops and asks for a shorter folder, for example `C:\bsf` for the inputs or `C:\bso` for `--out`.

- **Pinned files**, a GitHub or Hugging Face file at a recorded commit, must have the recorded size and SHA-256.
- **Tables read through the Hugging Face rows API**: 12 tables of SWE-bench and BigCodeBench, and the MMLU-Pro question table. That API serves only the current revision and does not fix its serialization, so each table is rebuilt by a fixed converter from the dataset's own parquet file at the recorded revision. The parquet file must have the size and SHA-256 that the Hugging Face hub lists for it at that revision. The rebuilt table must have the recorded SHA-256 or, failing that, the same normalized content. To normalize, every line is parsed as JSON and written back with sorted keys, compact separators and non-ASCII characters kept, with the rows in their original order; the SHA-256 of the result must equal the value in `filter_inputs.json`. This check compares the fields, identifiers, answers and order of the rows, and ignores only key order, spacing and escaping. The rule and these hashes were fixed on 2026-11-17, from the archived copies, before any file was downloaded again. The replay report records both hashes and which check passed. The converter writes the columns in the order recorded from the archived table. It was revised to do so after the first replay, which wrote them in the parquet file's order: twelve tables then had the recorded SHA-256, and the Instruct table of BigCodeBench's per-model results matched under the normalized check only, because its parquet file orders the 1,141 columns differently. Two of the studies check the SHA-256 of their inputs themselves, so they need the recorded bytes. The checks were not changed.
- **GitHub directory listings** at a commit must have the recorded SHA-256 or, failing that, the same normalized JSON content.
- **The SWE-bench Verified annotation archive** comes from a URL without a revision. It must have the recorded SHA-256, and so must each file extracted from it.
- **The run.json files of the MMLU study** are not in its manifests. `replay_inputs.py run-json` writes them from the recorded URLs and `fetch_status.json`.

A file that fails its check fails the replay of its set, and every study that reads the set fails with it. `--keep-going` runs the steps that do not depend on the failed set, but the run still ends as failed. Running a study on files that differ from the manifests is a diagnosis of an input change, not a reproduction: `--diagnose-input-change` does so, labels those steps "input change diagnosis" and ends as failed.

## Use the tool

```bash
pip install "git+https://github.com/JustinLi2002/scopecard@v0.1.0#subdirectory=tool"
# or, inside a clone of the repository: pip install ./tool
scopecard kg-wordnet --release WN18RR/ --wordnet WordNet-3.0/dict --name WN18RR --out card.json --md card.md
```

`tool/README.md` describes all seven audit types, the card schema and the browser page. The browser page runs the same code in the browser and uploads nothing. It is served at https://justinli2002.github.io/scopecard/app/; to run it locally, serve `tool/web/dist/` with any static server, for example `python -m http.server --directory tool/web/dist`. The card library is served at https://justinli2002.github.io/scopecard/ and is built in `tool/site/`.

## Citation

To cite ScopeCard, use `CITATION.cff` (GitHub shows it as "Cite this repository"). A link to the paper will be added when its preprint is posted.

## Licences

- **Code:** MIT (`LICENSE`).
- **Cards, library entries, the pilot table and stored outputs:** CC BY 4.0.
- **WordNet 3.0:** files derived from WordNet 3.0 carry its licence.
- **Benchmarks, source datasets and the inputs of the filter studies:** not redistributed.

See `NOTICE` for the details of all four.
