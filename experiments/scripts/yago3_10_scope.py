"""Reconstruct YAGO3-10's eligibility rule from YAGO 3.0.2 yagoFacts.

Dettmers et al. 2018 (ConvE): YAGO3-10 "is a subset of YAGO3 which consists of
entities which have a minimum of 10 relations each"; Rossi et al. 2021 read it
as entities with at least 10 different relations plus all facts involving them.
The YAGO3 version is not stated. This script checks the release against YAGO
3.0.2's yagoFacts (the only theme holding entity-entity facts) and tests the
readings of the rule.

usage: python -I yago3_10_scope.py <yagoFacts.ttl> <YAGO3-10 dir> <out dir>
"""
import collections
import os
import sys


def read_release(d):
    out = {}
    for s in ('train', 'valid', 'test'):
        with open(os.path.join(d, f'{s}.txt'), encoding='utf-8') as f:
            out[s] = [tuple(line.rstrip('\n').split('\t')) for line in f if line.strip()]
    return out


def read_facts(path):
    facts = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.startswith('<'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3 or not parts[2].startswith('<'):
                continue
            facts.add((parts[0][1:-1], parts[1][1:-1], parts[2][1:-1]))
    return facts


def main(facts_path, rel_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out = []

    def say(*a):
        print(*a)
        out.append(' '.join(str(x) for x in a))

    rel = read_release(rel_dir)
    R = set().union(*({x for x in v} for v in rel.values()))
    rels = {r for _, r, _ in R}
    E = {e for h, _, t in R for e in (h, t)}
    say('YAGO3-10 triples', len(R), 'relations', len(rels), 'entities', len(E))

    facts = read_facts(facts_path)
    say('YAGO 3.0.2 yagoFacts entity-entity facts', len(facts), 'relations', len({r for _, r, _ in facts}))
    f37 = {x for x in facts if x[1] in rels}
    say('  on the 37 YAGO3-10 relations', len(f37))
    say('YAGO3-10 triples found in YAGO 3.0.2:', len(R & f37), 'of', len(R),
        {s: sum(x in f37 for x in v) for s, v in rel.items()})
    say('  not found, by relation:', collections.Counter(r for _, r, _ in R - f37).most_common(6))
    E_y = {e for h, _, t in f37 for e in (h, t)}
    say('YAGO3-10 entities present in YAGO 3.0.2 (37 relations):', len(E & E_y), 'of', len(E))

    induced = {x for x in f37 if x[0] in E and x[2] in E}
    say('YAGO 3.0.2 facts induced on the YAGO3-10 entity set:', len(induced), '| in release', len(induced & R),
        '| induced but not in release', len(induced - R))

    # readings of "a minimum of 10 relations"
    deg = collections.Counter()
    types = collections.defaultdict(set)
    for h, r, t in f37:
        deg[h] += 1
        deg[t] += 1
        types[h].add(r)
        types[t].add(r)
    deg_all = collections.Counter()
    for h, r, t in facts:
        deg_all[h] += 1
        deg_all[t] += 1

    def report(name, seeds):
        closure = {x for x in f37 if x[0] in seeds or x[2] in seeds}
        ent_cl = {e for h, _, t in closure for e in (h, t)}
        both = {x for x in f37 if x[0] in seeds and x[2] in seeds}
        ent_both = {e for h, _, t in both for e in (h, t)}
        j = lambda A, B: len(A & B) / len(A | B) if A | B else 0
        say(f'{name}: seeds {len(seeds)} | seeds+neighbours: entities {len(ent_cl)} (Jaccard {j(ent_cl, E):.3f}), '
            f'triples {len(closure)} (release covered {len(R & closure)}/{len(R)}) | seeds only: entities {len(ent_both)} '
            f'(Jaccard {j(ent_both, E):.3f}), triples {len(both)} (release covered {len(R & both)})')

    report('>= 10 facts on the 37 relations', {e for e, c in deg.items() if c >= 10})
    report('>= 10 distinct relation types (37 relations)', {e for e, s in types.items() if len(s) >= 10})
    report('>= 10 facts on all yagoFacts relations', {e for e, c in deg_all.items() if c >= 10})

    # which release entities would be seeds under the leading reading, and how the rest attach
    seeds = {e for e, c in deg.items() if c >= 10}
    say('release entities that are seeds (>= 10 facts on the 37 relations):', len(E & seeds), '| non-seeds:', len(E - seeds),
        '| non-seeds with no seed neighbour in the release:',
        len({e for e in E - seeds} - {o for h, _, t in R for e, o in ((h, t), (t, h)) if e in seeds}))
    v = sorted(deg[e] for e in E)
    say('YAGO 3.0.2 degree (37 relations) of release entities: min', v[0], '| < 10:', sum(x < 10 for x in v))

    # W, entity side, against YAGO 3.0.2 entities that carry >= 1 fact of the 37 relations
    k = len(E & E_y)
    say(f'W_entity, universe = YAGO 3.0.2 entities with >= 1 fact on the 37 relations: {1 - k / len(E_y):.4f} '
        f'({len(E_y) - k} of {len(E_y)} excluded)')
    say('YAGO 3.0.2 facts on the 37 relations touching an excluded entity:', sum(1 for h, _, t in f37 if h not in E or t not in E), 'of', len(f37))

    with open(os.path.join(out_dir, 'yago3_10_scope.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:4])
