# Scope card: TREC DL 2019 passage

> Comparisons on TREC DL 2019 passage license claims about the queries it can evaluate, which leave out 78.5% of the 200 TREC DL 2019 passage test topics. Within each evaluated query, 99.9934% to 99.9985% of the 8,841,823 documents are unjudged and count as non-relevant. The stated rule (>= 3 relevant (grade >= 2) and relevant/judged < 0.6) is consistent with every released unit; excluded units cannot be checked (no published judgments).

- **Task:** query_level
- **Universe:** TREC DL 2019 passage test topics msmarco-test2019-queries.tsv.gz
- **W is a share of:** the 200 TREC DL 2019 passage test topics
- **Definitions:** Universe = the released test topics; release = topics with judgments. Within an evaluated topic, unjudged documents are scored as non-relevant by trec_eval (implicit negatives).
- **Tool:** scopecard 0.0.2 (schema scopecard/0.3)
- **Universe fingerprint:** `sha256:2be40e59dab214f971e1ce8e911e7af197a8dd75c0c74098048591f21ca28e6c`
- **Candidates:** the judged topics; within a topic, every corpus document (excluded units cannot be evaluated). Only topics with judgments are scored, so a released test topic without judgments is never evaluated; inside a topic, unjudged documents count as non-relevant.

## Mapping

43 of 43 release ids mapped; 0 unmapped; 0 ids stand for more than one universe unit.

## Exclusion

Units: 157 of 200 excluded, **W = 0.7850**.
Within the 43 evaluated queries: 132 to 582 judged documents each (median 161); unjudged share of the 8,841,823 documents 0.999934 to 0.999985. Judgments by grade: 0: 5158, 1: 1601, 2: 1804, 3: 697.

## Stated rule check

Rule: >= 3 relevant (grade >= 2) and relevant/judged < 0.6 (stated in Craswell et al. 2020, Overview of the TREC 2019 deep learning track (arXiv 2003.07820)). Checked on 43 released units: 0 violations (observed minimum relevant 3, maximum relevant/judged 0.534). Verdict: consistent with every released unit; excluded units cannot be checked (no published judgments).

## Input files

- **2019qrels-pass.txt:** 9,260 lines read.
