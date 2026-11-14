"""Query-level benchmarks with relevance judgments (TREC style).

Two layers of exclusion:
  queries    the released test topics that never reach the evaluation set (unit-level W over topics)
  documents  within an evaluated topic, the corpus documents never judged; trec_eval scores them as
             non-relevant, so they act as implicit negatives (per-query unjudged share)
A stated topic rule (e.g. TREC DL 2019: >= 3 relevant and relevant/judged < 0.6) can only be checked
on the released topics: excluded topics have no published judgments.
"""
from typing import Dict, Optional

import numpy as np


def judgment_profile(qrels: Dict[str, Dict[str, int]], corpus_size: int, relevant_grade: int = 1) -> Dict:
    judged = np.array([len(d) for d in qrels.values()])
    if not len(judged) or corpus_size < judged.max():
        raise ValueError('corpus_size must be at least the number of distinct documents judged for any query')
    rel = np.array([sum(g >= relevant_grade for g in d.values()) for d in qrels.values()])
    grades: Dict[str, int] = {}
    for d in qrels.values():
        for g in d.values():
            grades[str(g)] = grades.get(str(g), 0) + 1
    unj = 1 - judged / corpus_size
    stats = lambda x: {'min': float(x.min()), 'median': float(np.median(x)), 'max': float(x.max())}
    return {'evaluated_queries': len(qrels), 'corpus_size': int(corpus_size), 'judgments': int(judged.sum()),
            'by_grade': dict(sorted(grades.items(), key=lambda kv: int(kv[0]))), 'relevant_grade': relevant_grade,
            'judged_per_query': stats(judged), 'relevant_per_query': stats(rel), 'unjudged_share': stats(unj)}


def check_topic_rule(qrels: Dict[str, Dict[str, int]], relevant_grade: int, min_relevant: Optional[int] = None,
                     max_ratio: Optional[float] = None, text: str = '', source: str = '') -> Dict:
    """Check a stated topic rule on the released topics: >= min_relevant relevant judgments and
    relevant/judged < max_ratio. `source` names where the rule is stated."""
    viol_min, viol_ratio, ratios, rels = [], [], [], []
    for q, d in sorted(qrels.items()):
        r = sum(g >= relevant_grade for g in d.values())
        ratio = r / len(d)
        rels.append(r)
        ratios.append(ratio)
        if min_relevant is not None and r < min_relevant:
            viol_min.append(q)
        if max_ratio is not None and ratio >= max_ratio:
            viol_ratio.append(q)
    bad = sorted(set(viol_min) | set(viol_ratio))
    return {'rule': text, 'source': source, 'relevant_grade': relevant_grade, 'min_relevant': min_relevant,
            'max_ratio': max_ratio,
            'released_units_checked': len(qrels), 'violations': len(bad), 'violating_examples': bad[:5],
            'below_min_relevant': len(viol_min), 'at_or_above_max_ratio': len(viol_ratio),
            'observed_min_relevant': int(min(rels)), 'observed_max_ratio': float(max(ratios)),
            'verdict': ('consistent with every released unit' if not bad else f'violated by {len(bad)} released units')
            + '; excluded units cannot be checked (no published judgments)'}
