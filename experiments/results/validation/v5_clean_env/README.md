# V5 and V7: the clean-environment run (2026-10-28)

Copied from runs of `release/reproduce.py` in a new virtual environment (Python 3.12.5, numpy 2.5.4, matplotlib
3.11.2, pytest 9.2.0, packages from PyPI, no PyMuPDF) on the laptop (Windows 11, 15 GB RAM), with LaTeX from TeX
Live 2025 for the figures. The code is the frozen submission version, including the wording changes of
2026-10-28 ("prespecified"; the break-even "carries no guarantee"). `timing_<tier>.tsv` hold the rows the script
wrote for each tier; the lines below are copied from its console output and from `logs/make_entries.log` (local
paths left out). `--tier quick` and `--tier all` were run one after the other, and every step of both passed.

## Tier totals

- quick: 1.5 s
- paper: 5 min 20 s
- cards: 21 min 42 s

## Library cards: time per benchmark (V7)

- wn18rr: card in 5 s
- yago3-10: card in 182 s
- ogbl-citation2: card in 57 s
- gowalla: card in 81 s
- amazon-book: card in 313 s
- trec-dl-2019-passage: card in 0 s
- trec-dl-2020-passage: card in 0 s

Peak memory was not logged by the script. TOOL_DESIGN.md Section 10 records about 2.5 GB for YAGO3-10 and
Amazon-Book (measured when each audit was added).

Earlier runs in the same kind of environment: on 2026-10-21, before the io.py, per-query and later fixes, 2.0 s
(quick), 6 min 20 s (paper) and 21 min 49 s (cards); earlier on 2026-10-28, before the wording changes, every step
passed except the paper tier's figure step, which failed on an unconditional PyMuPDF import and passed after the
fix (quick 1.6 s, paper 5 min 13 s, cards 23 min 21 s).
