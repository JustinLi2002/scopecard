# V3: refusal and warning suite (browser page)

Checked 2026-10-25 in the built-in browser on `tool/site/app/index.html` (benchscope-audit 0.0.1 on Pyodide 0.27.7,
served by `python -m http.server`), after that day's fixes and a rebuild of `web/dist` and
`site/`. The page loaded the rebuilt package (zip sha256 5dab72099e8c2dbe...). The files are the V3 fixtures of the
same id, given to the page's file inputs through a script (DataTransfer), then Run clicked. The page calls the same
`main()` as the command line.

Rechecked 2026-10-27 on the rebuilt package with per-query shares (zip sha256 2ef86596c67ceb81...): the same
14 rows, same results. On that package the page's WN18RR card (with a model difference of -0.0142 AUC) is
byte-identical to the command line's `tool/examples/wn18rr_card.json` and `.md` (11 s in the page).

Rechecked 2026-10-27 again, after further fixes (zip sha256 5a50e1628265826b...): all 14 rows above
unchanged, plus R20 and R21 below. The page's WN18RR card with a model difference, which now carries the
left-out-test-triple warning, is byte-identical to the command line's (8.7 s). R21's wording was then made
independent of number; on that final package (zip sha256 dd641bf277c19095...) R21 and the WN18RR card were
checked again, with the same results.

Not rechecked in the browser after 2026-10-28: the R20 warning and the break-even statement were reworded
("approximate" became "carries no guarantee"), and `web/dist` and `site/` were rebuilt. The page runs the same
package code, the change touches wording only, and the command-line V3 passes 27 of 27 with it; the rows below keep
the wording observed on 2026-10-27.

Rechecked 2026-11-23, after the tool was renamed from BenchScope to ScopeCard (version 0.0.2; only names changed),
on the rebuilt `tool/site/app/index.html` (the page reports "scopecard 0.0.2 on Pyodide 0.27.7"; package zip sha256
c723de59b09910ce...), the same way: the fixtures given to the form's file inputs through a script (DataTransfer),
the fields set, Run clicked. All 16 rows behave as specified, with the messages in the table; R20 now shows the
wording of 2026-10-28 ("... for which the break-even carries no guarantee; recompute the difference on the 2
triples the shares use"). The page's cards are byte-identical, JSON and Markdown by SHA-256, to the command line's
for the same command: WN18RR without and with a model difference of -0.0142 AUC (12 s each), TREC DL 2019 passage
with its rule source, a pair-classification example (4 units, 3 listed pairs, every pair declared evaluated) and
Gowalla (130 s, in a background tab while other jobs ran). The WN18RR, TREC DL 2019 and Gowalla JSON equal the
library cards, and the WN18RR card with the model difference equals `tool/examples/wn18rr_card.json` and `.md`.

16 of 16 rows behave as specified.

| id | scenario | expected | observed in the page | result |
|---|---|---|---|---|
| R1 | Release does not match the universe (KG ids) | refused, no card | "Input problem: Only 0 of the 5 release entities (0.0%) occur in universe.tsv, …"; no card | pass |
| R4 | Malformed release line | refused, naming the file and the line | "Input problem: train.txt, line 2: expected 3 tab-separated fields (head, relation, tail), found 1: 'e1 r e2'" | pass |
| R5 | One empty split in a KG release (train.txt) | refused, naming the empty file | "Input problem: train.txt has no data lines." | pass |
| R5b | One empty split in a LightGCN release (train.txt) | refused, naming the empty file | "Input problem: train.txt has no data lines." | pass |
| R5c | All release files empty | refused, naming an empty file | "Input problem: train.txt has no data lines." | pass |
| R5d | Empty universe unit list | refused, naming the empty file | "Input problem: units.txt has no data lines." | pass |
| R5g | LightGCN split with user ids but no items | refused, naming the file | "Input problem: train.txt lists no user-item pairs: each of its 2 lines has a user id only." | pass |
| R10 | Declared candidate rule that the file contradicts | warning, break-even withheld | the "lists only 3 of the 6 pairs" warning and "No break-even is given: the declared candidate rule …" | pass |
| R15 | Nothing excluded (W = 0) with a model difference | no warning; nothing excluded can reverse it | "None of the checks run found a problem"; "nothing of the declared universe is excluded …" | pass |
| R16 | Two unit sets that share ids (bipartite, all pairs listed) | no warning; the declaration holds | "consistent with the file"; "None of the checks run found a problem"; 0.0% of the 4 unobserved pairs excluded | pass |
| R17 | Recommendation release counted with the pilot table's release positives | the headline is the share (80%) | "can never evaluate 80.0% of the 5 unobserved user-item pairs" | pass |
| R18 | Corpus size smaller than the documents judged for one topic | refused, naming the corpus size | "Input problem: --corpus-size 3 is smaller than the 4 distinct documents judged for one topic. …" | pass |
| R19 | Model difference that is not a finite difference in [-1, 1] (1.5 typed in the page) | refused as an input error | "Input problem: --delta-O must be a finite difference between -1 and 1" | pass |
| R20 | Model difference while some test triples fall outside the universe | warned: the break-even is approximate | "Left out of the per-query shares and the break-even: 1 of the 3 test triples (an entity outside the universe); … approximate" | pass |
| R21 | A test query with no kept candidate (W_q = 1) | break-even withheld | "No break-even is given: the benchmark difference is not defined on the 1 of the 2 test queries that keep no candidate of the release." | pass |
| stale | Settings or files changed after a card is shown | stale banner until the next run | shown after changing the model difference; hidden after the rerun; shown again after replacing the unit list | pass |

History:
- 2026-10-24, before the empty-file fix: R5 gave a card from valid.txt and test.txt alone (5 of 6 rows passed).
- 2026-10-25, first run after the fixes: R19 showed "The audit failed: SystemExit" with a traceback, because
  the range check used argparse's `error()`, which exits; the page cannot. The check now raises an input error,
  as every other input check does.
