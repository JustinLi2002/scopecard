"""Test-set hygiene for relational benchmarks.

reverse_leakage    share of test triples whose reverse (t, r, h) is a training triple: for
                   symmetric relations the answer is then one training edge away
                   (about a third of WN18RR's test triples, through e.g. derivationally_related_form)
unseen_test_units  test units that never occur in training
pair_release_checks  labelled pair releases: duplicates, conflicting labels, self pairs, and
                   negatives that the universe records as observed interactions
"""
from typing import Sequence, Tuple

Triple = Tuple[str, str, str]


def reverse_leakage(test: Sequence[Triple], train: Sequence[Triple]) -> dict:
    tr = set(train)
    leak = [(h, r, t) for h, r, t in test if (t, r, h) in tr]
    by_rel = {}
    for h, r, t in test:
        d = by_rel.setdefault(r, [0, 0])
        d[0] += 1
        d[1] += (t, r, h) in tr
    return {'test_triples': len(test), 'leaking': len(leak), 'share': len(leak) / len(test) if test else float('nan'),
            'by_relation': {r: {'test': n, 'leaking': k} for r, (n, k) in sorted(by_rel.items())}}


def pair_release_checks(rows: Sequence[Tuple[str, str, object]], observed: set, symmetric: bool,
                        positive_label: str = '1', labelled: bool = True) -> dict:
    """Checks on a labelled pair release. rows = (a, b, label); observed = the universe's observed pairs
    (as ordered tuples, both orders for symmetric). Negatives that are observed interactions contradict
    the release's own labels; positives absent from the observed set show how far the universe's
    interaction data covers the release."""
    key = (lambda a, b: (min(a, b), max(a, b))) if symmetric else (lambda a, b: (a, b))
    labels = {}
    for a, b, lab in rows:
        labels.setdefault(key(a, b), set()).add(lab)
    out = {'rows': len(rows), 'distinct_pairs': len(labels), 'duplicate_rows': len(rows) - len(labels),
           'self_pairs': sum(1 for a, b, _ in rows if a == b) if symmetric else 0}    # two unit sets: no self pairs
    if labelled:
        out['by_label'] = {str(k): v for k, v in sorted(_count(lab for _, _, lab in rows).items())}
        out['pairs_with_conflicting_labels'] = sum(1 for v in labels.values() if len(v) > 1)
    if observed:
        if labelled:
            neg = [p for p, v in labels.items() if positive_label not in v]
            pos = [p for p, v in labels.items() if positive_label in v]
            out['negatives_observed_as_interactions'] = sum(1 for p in neg if p in observed)
            out['positives_not_observed'] = sum(1 for p in pos if p not in observed)
            out['negatives'], out['positives'] = len(neg), len(pos)
        else:
            out['release_pairs_observed_as_interactions'] = sum(1 for p in labels if p in observed)
    return out


def _count(it):
    c = {}
    for x in it:
        c[x] = c.get(x, 0) + 1
    return c


def unseen_test_units(test: Sequence[Triple], train: Sequence[Triple]) -> dict:
    seen = {x for h, _, t in train for x in (h, t)}
    unseen = {x for h, _, t in test for x in (h, t)} - seen
    hit = sum(1 for h, _, t in test if h in unseen or t in unseen)
    return {'unseen_units': len(unseen), 'test_triples_affected': hit, 'test_triples': len(test)}
