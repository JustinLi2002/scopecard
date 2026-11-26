# scopecard

The audit package of **ScopeCard**. Given a benchmark release and a declared target universe, it reports which part of the universe the benchmark can never evaluate, and what comparisons on the benchmark therefore license. It needs no labels, no models and no GPU.

The construction rule does not have to be known. The package compares the units that actually occur in the release with the universe. It then reports:

- **Mapping:** how release ids map onto universe units (unmapped, merged).
- **Exclusion:** **W**, the excluded share, by unit and by candidate pair, with strata.
- **Depth profile:** how exclusion relates to source depth.
- **Rule diagnosis:** whether a simple depth rule (threshold, both-ends threshold, seeds plus neighbours, k-core) reproduces the release, and whether the rule stated in the paper does.
- **Derived scope:** the depth threshold d_min at which W ≤ 0.5, with the whole trade-off curve.
- **Comparison:** for a stated AUC difference Δ_O on the benchmark, the two break-even thresholds: the excluded part's opposite-direction difference must exceed (1 − W)|Δ_O| / W, equivalently the between-region change c = Δ_O − Δ_U must exceed **c\*** = |Δ_O| / W. None is given for MRR or Hits@k, or when the release does not establish an excluded share.
- **Candidates:** whether the benchmark enumerates its candidates over the release's units (then W is an excluded share) or only lists drawn pairs (then W is coverage).
- **Hygiene (KG):** reverse-edge leakage and unseen test entities.
- **Universe fingerprint:** a SHA-256 of the universe's units and relations, so two cards can be checked against the same target.
- **Input checks:** the scope statement names what W counts, with its size ("63.4% of the 112,199 WordNet 3.0 synsets that carry at least one triple of its 11 relations"). The audit stops with a one-line message when a release file does not have its format (file and line named) or when fewer than 50% of the release ids occur in the universe (wrong files, audit type or universe). It warns when fewer than 95% do, when lines of a raw file were skipped (counted per file under `inputs`), and when a pair file has a label column that was not used. Warnings are listed on the card and repeated after the scope statement. On the command line an input error exits with code 2.

The result is a **scope card** (JSON plus Markdown, schema `scopecard/0.3`). It constrains claims; it does not repair data. The size of the difference on the excluded part, c, needs labels there and is not computed.

## Status

Version 0.0.2 (2026-11-23), the version named in the paper: version 0.0.1 (frozen 2026-10-28), which computed the paper's results, renamed. Up to version 0.0.1 the tool was called BenchScope. The package, the command and the card schema's id are now `scopecard`; every number a card reports is unchanged. Seven pipelines:

| Command | Task type | Benchmark | Universe |
|---|---|---|---|
| `kg-wordnet` | KG link prediction | WN18, WN18RR | WordNet 3.0 (Princeton `dict` folder) |
| `kg-triples` | KG link prediction | releases keyed by the universe's own ids (e.g. YAGO3-10) | a triple file: YAGO 3 Turtle theme (`yagoFacts.ttl`) or head/relation/tail TSV |
| `recsys-checkins` | recommendation | LightGCN/NGCF-format releases (e.g. Gowalla) | raw SNAP check-ins |
| `recsys-ratings` | recommendation | LightGCN/NGCF-format releases (e.g. Amazon-Book) | an Amazon 2014 ratings CSV (e.g. `ratings_Books.csv`) |
| `query-qrels` | query level | TREC-style qrels (e.g. TREC DL passage) | the released test topics |
| `ogbl-citation2` | query level | ogbl-citation2 (OGB folder) | the papers of the source year |
| `pair-classification` | pair classification | labelled pairs or a negatives-only file (CSV/TSV), e.g. protein–protein or drug–target | a declared unit list (one or two sides), optionally with observed pairs |

For query-level benchmarks the units are queries: topics for TREC, source papers for ogbl-citation2.

- **ogbl-citation2:** depth is a paper's references inside the subset, and the rule search runs the threshold family on that depth. The card also checks the sampled negatives: none later than the source, none cited by it, and uniform over nodes. OGB's split `.pt` files are read without torch, by an unpickler that accepts numpy arrays only and refuses any other code.
- **Pair classification:** the units are both ends of a pair: one set with unordered pairs (protein–protein) or two sets (drug–target, `--bipartite`). W is the share of declared units the release never uses; W_pairs is the share of the universe's unobserved pairs with an end outside the retained units. Observed pairs (known interactions), when given, give each unit its depth for the rule search and flag release negatives that are known interactions. A release lists its pairs but not how they were drawn, so unless the user declares that every pair among the release units is evaluated (`--candidate-rule all-pairs`), W, W_pairs and the pair coverage are reported as coverage, not exclusion, and no break-even is given.
- **TREC:** W is the share of released test topics that never get judgments. The card also reports, within each judged topic, the share of the corpus that is never judged and is therefore scored as non-relevant. A stated topic rule can be checked only on the released topics, because excluded topics have no published judgments.

## Use

```bash
pip install "git+https://github.com/JustinLi2002/scopecard@v0.1.0#subdirectory=tool"
# or, inside a clone of the repository: pip install -e tool
scopecard kg-wordnet --release WN18RR/ --wordnet WordNet-3.0/dict --name WN18RR --out card.json --md card.md
scopecard kg-triples --release YAGO3-10/ --universe yagoFacts.ttl --format yago-ttl --name YAGO3-10 \
    --stated-family both_ends --stated-T 10 --out card.json
scopecard recsys-checkins --release gowalla/ --checkins loc-gowalla_totalCheckins.txt.gz --out card.json
scopecard recsys-ratings --release amazon-book/ --ratings ratings_Books.csv --out card.json
scopecard query-qrels --topics msmarco-test2019-queries.tsv.gz --qrels 2019qrels-pass.txt \
    --corpus-size 8841823 --relevant-grade 2 --min-relevant 3 --max-ratio 0.6 --name "TREC DL 2019 passage" --out card.json
scopecard ogbl-citation2 --dataset citation-v2/ --out card.json
scopecard pair-classification --pairs negatives.csv --a-col 0 --b-col 1 --no-header \n    --universe-units proteins.txt --universe-pairs interactions.tsv --universe-name "human proteome" --out card.json
```

Release units that are absent from the universe can be selected by no rule over it, so they count against every rule in the Jaccard (YAGO3-10 has 2,044 such entities).

Example cards: [WN18RR](examples/wn18rr_card.md), [YAGO3-10](examples/yago3_10_card.md), [TREC DL 2019 passage](examples/trecdl2019_card.md), [ogbl-citation2](examples/ogbl_citation2_card.md), [Amazon-Book](examples/amazon_book_card.md).

## Browser page

`web/` holds a static page that runs the same package in the browser with [Pyodide](https://pyodide.org) 0.27.7. The files a user chooses are read inside the browser and never uploaded. The page downloads only the Python runtime (from cdn.jsdelivr.net) and its own files. Python runs in a Web Worker, so the page stays responsive during long audits.

```bash
python web/build.py WordNet-3.0/dict      # writes web/dist/: index.html, worker.js, scopecard.zip, wordnet30_wn18.tsv.gz, example card
python -m http.server 8931 --bind 127.0.0.1 --directory web/dist
```

- **While an audit runs** the page shows its stage (reading, mapping, rules) and a Cancel button, which stops the worker and reloads Python. A new run, an error or a change of audit type hides the previous card. Input problems are shown as one sentence naming the file and line; other failures keep the Python traceback under "Technical details".
- **Without files:** "Show an example card" displays the command line's WN18RR card, and "What the card's terms mean" explains W, W_pairs, Jaccard, d_min and c\*. Inside the card library (`site/app/`), the page links back to the library, "How to read a card" and the schema.
- **One code path.** The page calls the command line's own `main()` inside Pyodide, so a card made in the browser equals the CLI's card byte for byte. This was checked for WN18RR, TREC DL 2019, Gowalla and a pair-classification example.
- **Built-in WordNet 3.0.** WordNet ships as a compact table (`export_wordnet30_table`, 2.4 MB) in place of the dict folder; the golden test `test_wordnet_table_equals_dict` shows both give the same universe and the same card. The table carries WordNet's licence notice, as the licence requires.
- **Size limit.** YAGO 3, Amazon Books and ogbl-citation2 are too large for the browser; use the command line for them.
- **Hosting.** Any static host serves `web/dist/` as is (the card library's copy, `site/app/`, is served at https://justinli2002.github.io/scopecard/app/). Both archives are built without timestamps, so a rebuild gives the same bytes.

## Card library

`library/` builds a static site with one page per benchmark of the 2026-10 pilot.

```bash
python library/make_entries.py ../experiments/pilot_data ../experiments/pilot_decidability.tsv   # cards + entries (about 9 minutes)
python library/build_site.py                                                           # writes site/
```

- **`benchmarks.json`** is the registry: each benchmark's row in the pilot table, the command that makes its card, and the public source of every input. For benchmarks without a card it says what is missing (FB15k-237: Freebase 2013 is not distributed; ogbl-collab: needs an external author graph; Yelp2018: needs Yelp's terms).
- **`make_entries.py`** runs each command in its own process, writes `cards/<slug>.json`, and rejects any card that fails `validate_card`. It also copies the benchmark's pilot-table row into `entries/<slug>.json` as is.
- **`build_site.py`** writes `site/`:
  - `index.html`, a table of all 10 benchmarks;
  - `cards/<slug>.html` and `.json`;
  - `how-to-read.html`;
  - the card's JSON Schema;
  - `app/`, the browser auditor from `web/dist`.

  Every number comes from a card or the pilot table, and a rebuild gives the same bytes.
- **Contributing a card:** run the auditor on the release, add an entry to `benchmarks.json`, run the two scripts and the tests, and open a pull request. `schema/scopecard-0.3.schema.json` describes the card for other languages (the 0.2 schema, `schema/benchscope-card-0.2.schema.json`, is kept for cards written before 2026-10-21, when the tool was still called BenchScope). Biology benchmarks are added once the papers that audit them are published.

## Tests

```bash
python -m pytest -q tests
```

- **`test_synthetic.py`:**
  - the rule search recovers generating rules; W matches brute-force enumeration; merged ids are reported;
  - vectorised triple matching agrees with a set lookup; unmapped release units count against rules;
  - the query-level pipeline counts topics, unjudged shares and rule violations correctly;
  - the torch-free reader loads both torch file layouts and refuses a pickle that calls anything but numpy;
  - the threshold search on a supplied depth recovers its rule;
  - the streaming loader for large interaction files gives the same universe as the simple one;
  - pair-level W for unordered pairs matches enumeration, also inside the derived scope; the pair-classification pipeline counts units, pairs, duplicates, conflicting labels and negatives that are known interactions, for one and for two unit sets;
  - input checks, from the 2026-10-21 user test: the scope statement names what W counts; a release that does not match the universe is refused and a partly matching one is flagged; skipped raw lines are counted and reported; malformed release lines name the file and line; the CLI exits with code 2 and no traceback; an unused label column is flagged.
- **`test_golden.py`:** the package reproduces the numbers of the 2026-10 pilot and of the declared WN18RR target from the same public files:
  - WN18RR: W 0.633642 (synset identity) and 71,032 / 111,975 (the benchmark's own keying), 161 merged ids;
  - WN18's stated rule: Jaccard 0.150, not reproduced;
  - the browser's WordNet table gives the same universe as the dict folder (8 variants) and the same WN18RR card, and rebuilds byte-identically;
  - YAGO3-10 vs YAGO 3.0.2: W 0.9540; 2,044 release entities and 34,668 triples not in YAGO 3.0.2; both-ends T = 10 at Jaccard 0.967, found by the search unprompted (about 3 minutes);
  - Gowalla: W 0.991088, 10-core approximately reproduced;
  - Amazon-Book vs the 2014 Books ratings: pair-level W 0.999742; the stated 10-core contains the whole release but keeps three times its users, so it is not reproduced (several minutes);
  - TREC DL 2019 / 2020 passage: topic W 0.785 / 0.730; the 2019 topic rule holds for every released topic;
  - ogbl-citation2: W 0.2025 of 2019 papers; "at least 6 in-subset references" reproduces the source papers exactly; negatives never later than the source and uniform over nodes (about 1 minute).

  The golden tests read the source files from `../experiments/pilot_data` (override with `SCOPECARD_PILOT_DATA`) and are skipped when they are absent.
- **`test_library.py`:** the registry is consistent; entries copy the pilot table exactly; every library card validates; the validator catches inconsistent cards; the schema file matches the validator; the site rebuilds byte-identically; the quick cards equal a fresh run.
