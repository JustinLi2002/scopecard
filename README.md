# BenchScope

BenchScope asks which part of a declared target population a benchmark can evaluate, and what comparisons on the benchmark therefore license. It accompanies the paper *Which population does a benchmark ranking describe?* by Justin Li (2026). This repository holds:

- the audit tool;
- the card library;
- the code and stored outputs of the WN18RR measured-c experiment.

The browser page runs an audit without installing anything, and the files you choose stay on your machine: https://justinli2002.github.io/benchscope/app/. The card library is at https://justinli2002.github.io/benchscope/.

Every number reported in the paper comes from three sources, and `reproduce.py` regenerates all three and compares them with the copies stored here:

- `experiments/results/wn18rr_analysis/records.tsv`;
- the scope cards;
- the pilot table.

[P2] and [P3] in these files are two earlier studies by the author, cited in the paper:

- **[P2]** J. Li. Constructed negatives determine the population a benchmark can rank over. Research Square preprint, 2026. https://doi.org/10.21203/rs.3.rs-11252843/v2
- **[P3]** J. Li. What a low-bias filter leaves out: measured performance on the excluded negatives of a drug–target interaction benchmark. ChemRxiv preprint, 2026. https://doi.org/10.26434/chemrxiv.15010766/v1

## Contents

| Path | What it holds |
|---|---|
| `tool/` | **The `benchscope-audit` package**: a command line with seven audit types, with its tests. Also here: the browser page (`web/`), the card library (`library/`; the built site is `site/`) and the example cards. See `tool/README.md`. |
| `experiments/MEASURED_C_DESIGN.md` | **The design of the WN18RR experiment.** Changes made after the design was fixed are dated in place. Clarifications written after the results are in Section 14. |
| `experiments/scripts/` | **Scripts:** declared-target construction and acceptance checks, the scorers, the prespecified analysis, and the figures. `hpc/` holds the GPU job templates. The pilot's first-pass scripts are here too. |
| `experiments/results/wn18rr_runs/` | **Per-query scorer outputs of the 87 runs**: E1 31, E2 26, S3 25, deployment 5. |
| `experiments/results/wn18rr_analysis/` | **`records.tsv`** (every reported value, keyed by design, estimand, subject, seed, quantity and cluster) and its formatted `summary.md`. |
| `experiments/results/wn18rr_target*/` | **Hashes and summaries of the two declared targets.** The targets themselves are rebuilt from WordNet 3.0, not shipped. |
| `experiments/figures/` | **Figures 2–4, Tables 1–2 and Box 1.** |
| `experiments/env/`, `experiments/configs/` | **Training setup:** pinned LibKGE and NBFNet commits, their patches, the package lists of the GPU environments, and the E1 training configs. |
| `experiments/pilot_decidability.tsv` | **The pilot coding** of the ten benchmarks. |
| `reproduce.py`, `inputs.json` | **One-command reproduction**, plus the public inputs with their SHA-256 hashes. |

## Reproduce

You need Python 3.9 or later. No GPU is needed. The `paper` tier also needs LaTeX with the `helvet`, `type1cm` and
`underscore` packages, because the figures are typeset with Matplotlib's usetex so that they embed only Type 1
fonts. PyMuPDF (`pip install pymupdf`) is optional; without it the figures are drawn as PDF only.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install "./tool[test,repro]"     # numpy, matplotlib, pytest
python reproduce.py --tier quick
python reproduce.py --tier paper --fetch
python reproduce.py --tier cards --fetch
```

`--fetch` downloads any missing input from the public source listed in `inputs.json` into `experiments/pilot_data/`. Every input file is checked against its SHA-256 before use. Outputs go to `repro_out/`; nothing stored here is overwritten. Each step's time and result are written to `repro_out/timing.tsv`. A step fails when its output differs from the stored copy. Compressed files (`.gz`, `.zip`) are compared after decompression, because their bytes depend on the zlib build: Python 3.14 on Windows ships zlib-ng.

| Tier | What it does | Inputs | Time |
|---|---|---|---|
| `quick` | Runs the tests that need no data: synthetic cases with known answers, plus the registry, schema and site checks. | none | 1.5 s |
| `paper` | Rebuilds both declared targets from WordNet 3.0 and WN18RR, checks them against the recorded hashes, and runs the acceptance checks. It then reruns the prespecified analysis on the stored scorer outputs and compares every record. Finally it redraws Figures 2–4, Tables 1–2 and Box 1, and compares the tables and Box 1 with the stored copies. | WordNet 3.0, WN18RR (41 MB) | 5 min |
| `cards` | Rebuilds every library scope card from the public sources, along with the example cards, the browser page and the library site, and compares each file with the stored copy. Then it runs the full test suite, including the golden tests. | all of `inputs.json` (3.9 GB) | 22 min |

**Measured times.** The times above come from a fresh virtual environment on a Windows 11 machine with 22 logical CPUs: Python 3.12.5, numpy 2.5.4, matplotlib 3.11.2, pytest 9.2.0, with TeX Live 2025 for the figures and without PyMuPDF.

- **Setup:** creating the environment and running `pip install` took under a minute.
- **Paper tier:** ran with its inputs already present (`--fetch` adds 12 MB of downloads, a few seconds). Most of it is the analysis, 5 minutes.
- **Cards tier:** ran with the other inputs already present. Rebuilding the library cards takes 11 minutes and the full test suite 11.

A run of the previous version with Python 3.14 gave identical results.

**Not rerun: GPU training and scoring.** The paper tier starts from their stored per-query outputs. To rerun them:

1. Install LibKGE and NBFNet at the commits in `experiments/env/`, apply the patches there, and match the package lists in `env_freeze.txt` and `env_nbf_freeze.txt`.
2. **E1 (WN18RR as published):** train with the unmodified configs in `experiments/configs/e1/`.
3. **E2 and S3 (the declared targets):** build the training data with `export_libkge_e2.py` and `make_o_candidates.py`, and the configs with `make_e2_config.py`.
4. **Score each checkpoint:** use `score_wn18rr.py` (LibKGE models), `score_popularity.py` (the popularity baseline) or `score_nbfnet.py` (NBFNet).

The SLURM templates in `experiments/scripts/hpc/` show the exact commands; site paths in them are replaced by `/home/anon`.

## Use the tool

```bash
pip install "git+https://github.com/JustinLi2002/benchscope@v0.0.1#subdirectory=tool"
# or, inside a clone of the repository: pip install ./tool
benchscope-audit kg-wordnet --release WN18RR/ --wordnet WordNet-3.0/dict --name WN18RR --out card.json --md card.md
```

`tool/README.md` describes all seven audit types, the card schema and the browser page. The browser page runs the same code in the browser and uploads nothing. It is served at https://justinli2002.github.io/benchscope/app/; to run it locally, serve `tool/web/dist/` with any static server, for example `python -m http.server --directory tool/web/dist`. The card library is served at https://justinli2002.github.io/benchscope/ and is built in `tool/site/`.

## Citation

To cite BenchScope, use `CITATION.cff` (GitHub shows it as "Cite this repository"). A link to the paper will be added when its preprint is posted.

## Licences

- **Code:** MIT (`LICENSE`).
- **Cards, library entries, the pilot table and stored outputs:** CC BY 4.0.
- **WordNet 3.0:** files derived from WordNet 3.0 carry its licence.
- **Benchmarks and source datasets:** not redistributed.

See `NOTICE` for the details of all four.
