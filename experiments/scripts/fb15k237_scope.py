"""Check FB15k-237's construction rules against FB15k and the release.

Toutanova & Chen 2015 (CVSC workshop, Sect. 4.3): starting from FB15k, keep the
401 most frequent relations, detect near-duplicate and inverse relations ("at
least 97% of the pairs are in the intersection"), keep one of each group (237
relations), and drop from validation and test any triple whose entity pair is
directly linked in training. FB15k itself (Bordes et al. 2013) kept Freebase
entities present in Wikilinks with at least 100 Freebase mentions; that part
needs Freebase and Wikilinks and is not checked here.

usage: python -I fb15k237_scope.py <FB15k dir> <FB15k-237 Release dir> <ConvE FB15k-237 dir> <out dir>
"""
import collections
import hashlib
import itertools
import os
import sys


def read(path):
    with open(path, encoding='utf-8') as f:
        return [tuple(line.rstrip('\n').split('\t')) for line in f if line.strip()]


def main(fb_dir, rel_dir, conve_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out = []

    def say(*a):
        print(*a)
        out.append(' '.join(str(x) for x in a))

    fb = {s: read(os.path.join(fb_dir, f'freebase_mtr100_mte100-{s}.txt')) for s in ('train', 'valid', 'test')}
    rr = {s: read(os.path.join(rel_dir, f'{s}.txt')) for s in ('train', 'valid', 'test')}
    for s in rr:
        a = hashlib.sha256(open(os.path.join(rel_dir, f'{s}.txt'), 'rb').read()).hexdigest()
        b = hashlib.sha256(open(os.path.join(conve_dir, f'{s}.txt'), 'rb').read()).hexdigest()
        same = a == b or sorted(rr[s]) == sorted(read(os.path.join(conve_dir, f'{s}.txt')))
        say(f'{s}: Microsoft release {len(rr[s])} triples; ConvE copy identical as a triple set: {same}')
    fb_all = set().union(*map(set, fb.values()))
    rr_all = set().union(*map(set, rr.values()))
    say('FB15k triples', len(fb_all), 'FB15k-237 triples', len(rr_all), '| FB15k-237 triples not in FB15k:', len(rr_all - fb_all))

    # --- relation rule
    freq = collections.Counter(r for _, r, _ in fb['train'])
    freq_all = collections.Counter(r for _, r, _ in fb_all)
    R237 = {r for _, r, _ in rr_all}
    for name, fr in (('train', freq), ('all splits', freq_all)):
        top = [r for r, _ in fr.most_common()]
        rank = {r: i + 1 for i, r in enumerate(top)}
        cut = fr[top[400]]
        say(f'relation frequency on FB15k {name}: 401st relation has {cut} triples; ties at that count: {sum(1 for r in fr if fr[r] == cut)};'
            f' 237 relations all within rank 401: {all(rank[r] <= 401 for r in R237)}; max rank of a kept relation {max(rank[r] for r in R237)}')
    top401 = {r for r, _ in freq.most_common(401)}
    pairs = collections.defaultdict(set)
    for h, r, t in fb_all:
        pairs[r].add((h, t))

    def overlap(a, b, inverse):
        A, B = pairs[a], pairs[b]
        if inverse:
            B = {(t, h) for h, t in B}
        inter = len(A & B)
        return inter / min(len(A), len(B)), inter / max(len(A), len(B))

    links = []
    for a, b in itertools.combinations(sorted(top401), 2):
        for inv in (False, True):
            lo, hi = overlap(a, b, inv)
            if lo >= 0.97:
                links.append((a, b, inv, lo, hi))
    say('top-401 relation pairs with >= 97% overlap (intersection / smaller set):', len(links),
        '| of these also >= 97% of the larger set:', sum(hi >= 0.97 for *_, hi in links))
    dropped = top401 - R237
    linked_to_kept = {r for a, b, *_ in links for r, o in ((a, b), (b, a)) if r in dropped and o in R237}
    linked_any = {r for a, b, *_ in links for r in (a, b) if r in dropped}
    say('dropped relations (top 401 minus 237):', len(dropped), '| with a >= 97% partner that was kept:', len(linked_to_kept),
        '| with any >= 97% partner:', len(linked_any))
    kept_pairs = [(a, b, inv, round(lo, 4), round(hi, 4)) for a, b, inv, lo, hi in links if a in R237 and b in R237]
    say('pairs of KEPT relations >= 97% of the smaller set (should be 0 under that reading):', len(kept_pairs),
        '| >= 97% of the larger set too:', sum(hi >= 0.97 for *_, hi in kept_pairs))
    with open(os.path.join(out_dir, 'fb15k237_kept_duplicate_pairs.tsv'), 'w', encoding='utf-8') as f:
        f.write('relation_a\trelation_b\tinverse\toverlap_of_smaller\toverlap_of_larger\n')
        for row in kept_pairs:
            f.write('\t'.join(map(str, row)) + '\n')
    dropped_strict = {r for a, b, inv, lo, hi in links if hi >= 0.97 for r, o in ((a, b), (b, a)) if r in top401 - R237 and o in R237}
    say('dropped relations explained under the larger-set reading:', len(dropped_strict), 'of', len(top401 - R237))
    unexplained = sorted(dropped - linked_any, key=lambda r: -freq[r])
    say('dropped relations with no >= 97% partner:', len(unexplained), [(r, freq[r]) for r in unexplained[:5]])

    # --- triple retention per split on the 237 relations
    for s in ('train', 'valid', 'test'):
        on237 = [x for x in fb[s] if x[1] in R237]
        say(f'FB15k {s} on the 237 relations: {len(on237)}; FB15k-237 {s}: {len(rr[s])}; release subset of FB15k {s}: {set(rr[s]) <= set(fb[s])}')

    # --- test-side pair rule
    train_pairs = {(h, t) for h, _, t in rr['train']}
    train_und = train_pairs | {(t, h) for h, t in train_pairs}
    for s in ('valid', 'test'):
        same_dir = sum((h, t) in train_pairs for h, _, t in rr[s])
        either = sum((h, t) in train_und for h, _, t in rr[s])
        say(f'FB15k-237 {s} triples whose entity pair is linked in train: same direction {same_dir}, either direction {either}')
    removed = {}
    for s in ('valid', 'test'):
        cand = [x for x in fb[s] if x[1] in R237]
        linked = [x for x in cand if (x[0], x[2]) in train_und]
        removed[s] = (len(cand), len(linked))
    say('FB15k valid/test triples on the 237 relations that the pair rule would remove (vs FB15k-237 train):',
        {s: f'{b} of {a} ({b / a:.3f})' for s, (a, b) in removed.items()})

    # --- entity side
    E15 = {e for h, _, t in fb_all for e in (h, t)}
    E237 = {e for h, _, t in rr_all for e in (h, t)}
    Etr = {e for h, _, t in rr['train'] for e in (h, t)}
    say('entities: FB15k', len(E15), 'FB15k-237', len(E237), 'lost from FB15k', len(E15 - E237),
        '| FB15k-237 valid/test entities absent from its train:', len(E237 - Etr))
    # entities removed although they still carry triples on the 237 relations (not stated in the paper)
    fb237 = [x for x in fb_all if x[1] in R237]
    d15 = collections.Counter()
    for h, _, t in fb237:
        d15[h] += 1
        d15[t] += 1
    lost = E15 - E237
    lost_with = {e for e in lost if d15[e] > 0}
    rr_train = set(rr['train'])
    miss_tr = [x for x in fb['train'] if x[1] in R237 and x not in rr_train]
    say('lost entities with 0 triples on the 237 relations:', len(lost) - len(lost_with), '| lost entities that still carry such triples:',
        len(lost_with), '| largest:', sorted(((d15[e], e) for e in lost_with), reverse=True)[:3])
    say('FB15k train triples on the 237 relations missing from FB15k-237 train:', len(miss_tr),
        '| of these touching a removed entity:', sum(h in lost or t in lost for h, _, t in miss_tr))
    d = collections.Counter()
    for h, _, t in rr_all:
        d[h] += 1
        d[t] += 1
    v = sorted(d.values())
    say('FB15k-237 entity degree: min', v[0], 'median', v[len(v) // 2], '| entities with degree < 10:', sum(x < 10 for x in v))

    with open(os.path.join(out_dir, 'fb15k237_scope.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:5])
