"""Check TREC DL passage-task topic and judgment rules against the released qrels.

Craswell et al. 2020 (TREC DL 2019 overview, judging section): of 200 test
topics, 52 were judged (median MRR of submitted document runs under sparse MARCO
labels in (0, 0.5]); the evaluation set keeps topics with at least 3 relevant
and relevant/judged < 0.6. For passages, relevant = grade >= 2. The 2020
overview says only that NIST chose a subset for judging.

usage: python -I trecdl_scope.py <qrels> <test queries .tsv.gz> <corpus size> <name> <out dir>
"""
import collections
import gzip
import os
import sys


def main(qrels, queries, corpus_n, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    corpus_n = int(corpus_n)
    out = []

    def say(*a):
        print(*a)
        out.append(' '.join(str(x) for x in a))

    with gzip.open(queries, 'rt', encoding='utf-8') as f:
        topics = [line.split('\t')[0] for line in f if line.strip()]
    J = collections.Counter()
    R = collections.Counter()
    grades = collections.Counter()
    with open(qrels, encoding='utf-8') as f:
        for line in f:
            q, _, d, g = line.split()
            J[q] += 1
            grades[int(g)] += 1
            if int(g) >= 2:
                R[q] += 1
    ev = sorted(J, key=int)
    say(f'[{name}] test topics {len(topics)}, topics in qrels {len(ev)}, all in the test list: {set(ev) <= set(topics)}')
    say('  judgments', sum(J.values()), 'by grade', dict(sorted(grades.items())))
    ratio = {q: R[q] / J[q] for q in ev}
    say(f'  topics with < 3 relevant (grade >= 2): {sum(R[q] < 3 for q in ev)} | with relevant/judged >= 0.6: {sum(ratio[q] >= 0.6 for q in ev)}'
        f' | min relevant {min(R[q] for q in ev)}, max ratio {max(ratio.values()):.3f}')
    say(f'  judged per topic: min {min(J.values())}, median {sorted(J.values())[len(J) // 2]}, max {max(J.values())}')
    # topic-level exclusion: share of the 200 test topics without an evaluation set entry
    say(f'  topic-level excluded share: {1 - len(ev) / len(topics):.3f} ({len(topics) - len(ev)} of {len(topics)})')
    # passage level within evaluated topics: share of the corpus never judged (scored as non-relevant by trec_eval)
    unj = [1 - J[q] / corpus_n for q in ev]
    say(f'  per evaluated topic, unjudged share of the {corpus_n} passages: min {min(unj):.6f}, max {max(unj):.6f}')
    with open(os.path.join(out_dir, f'{name}_scope.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:6])
