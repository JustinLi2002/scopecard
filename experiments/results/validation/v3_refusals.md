# V3: refusal and warning suite (command line)

27 of 27 scenarios behave as specified. Every card written also passes validate_card; no scenario prints a traceback. Fixtures: v3_fixtures/<id>/.

| id | scenario | expected | observed | result |
|---|---|---|---|---|
| R1 | Release does not match the universe (KG ids) | refused: fewer than 50% of release ids occur in the universe | exit 2; benchscope-audit: input error: Only 0 of the 5 release entities (0.0%) occur in universe.tsv, so these files do not look like a release of this universe. Check  | pass |
| R2 | Release does not match the universe (query level) | refused: no judged topic is a released topic | exit 2; benchscope-audit: input error: Only 0 of the 3 judged topics (0.0%) occur in the topic list (topics.tsv), so these files do not look like a release of this univ | pass |
| R3 | Release partly outside the universe | warned: between 50% and 95% of release ids occur in the universe | exit 0; warnings 1; structural True; c* None | pass |
| R4 | Malformed release line | refused, naming the file and the line | exit 2; benchscope-audit: input error: train.txt, line 2: expected 3 tab-separated fields (head, relation, tail), found 1: 'e1 r e2' | pass |
| R5 | One empty split in a KG release (train.txt) | refused, naming the empty file | exit 2; benchscope-audit: input error: train.txt has no data lines. | pass |
| R5b | One empty split in a LightGCN release (train.txt) | refused, naming the empty file | exit 2; benchscope-audit: input error: train.txt has no data lines. | pass |
| R5c | All release files empty | refused, naming an empty file | exit 2; benchscope-audit: input error: train.txt has no data lines. | pass |
| R5d | Empty universe unit list | refused, naming the empty file | exit 2; benchscope-audit: input error: units.txt has no data lines. | pass |
| R5e | Pair file with a header only | refused, naming the empty file | exit 2; benchscope-audit: input error: pairs.csv has no data lines. | pass |
| R5f | Empty topic list | refused, naming the empty file | exit 2; benchscope-audit: input error: topics.tsv has no data lines. | pass |
| R5g | LightGCN split with user ids but no items | refused, naming the file | exit 2; benchscope-audit: input error: train.txt lists no user-item pairs: each of its 2 lines has a user id only. | pass |
| R6 | Raw file with lines of the wrong format | warned: skipped lines counted per file, first one named | exit 0; warnings 1; structural True; c* None | pass |
| R7 | Raw file with no line of the right format | refused: no line of the file fits its format | exit 2; benchscope-audit: input error: qrels.txt: none of its 2 lines has the expected format (4 whitespace-separated fields: topic, iteration, document, integer grade) | pass |
| R8 | Pair file with an unused label column | warned: a label column was not used | exit 0; warnings 1; structural False; c* None | pass |
| R9 | Sampled pairs with an unknown sampling rule | coverage only, and no break-even | exit 0; warnings 0; structural False; c* None | pass |
| R10 | Declared candidate rule that the file contradicts | warned, and the break-even withheld: the file lists fewer pairs than the declaration implies | exit 0; warnings 1; structural True; c* None | pass |
| R11 | Declared candidate rule that the file supports (control) | no warning; the exclusion is structural | exit 0; warnings 0; structural True; c* None | pass |
| R12 | Break-even asked for a metric that does not decompose (MRR) | withheld: MRR is not linear in the candidate mix | exit 0; warnings 0; structural True; c* None | pass |
| R13 | Model difference given without its metric | refused by the argument parser: --delta-O needs --metric | exit 2; benchscope-audit: error: --delta-O needs --metric (auc, mrr or hits) | pass |
| R14 | No unobserved pair in the universe (zero candidates) | W_pairs undefined with a note, no crash | exit 0; warnings 0; structural False; c* None | pass |
| R15 | Nothing excluded (W = 0) with a model difference | no crash; the comparison says nothing excluded can reverse it | exit 0; warnings 0; structural True; c* None | pass |
| R16 | Two unit sets that share ids (bipartite, all pairs listed) | no warning; coverage 100%; the declaration holds (equal ids on the two sides are different units) | exit 0; warnings 0; structural True; c* None | pass |
| R17 | Recommendation release counted with the pilot table's release positives | the headline is the share (80%); the pilot value (60%) is kept beside it, not reported as excluded | exit 0; warnings 0; structural True; c* None | pass |
| R18 | Corpus size smaller than the documents judged for one topic | refused, naming the corpus size (an unjudged share would be negative) | exit 2; benchscope-audit: input error: --corpus-size 3 is smaller than the 4 distinct documents judged for one topic. Give the number of documents in the collection the | pass |
| R19 | Model difference that is not a finite difference in [-1, 1] | refused as an input error (also in the page, which cannot exit) | exit 2; benchscope-audit: input error: --delta-O must be a finite difference between -1 and 1 | pass |
| R20 | Model difference while some test triples fall outside the universe | warned: the break-even uses fewer queries than the difference and carries no guarantee | exit 0; warnings 2; structural True; c* 0.029932 | pass |
| R21 | A test query with no kept candidate (W_q = 1) | withheld: the benchmark difference is undefined on that query | exit 0; warnings 0; structural True; c* None | pass |
