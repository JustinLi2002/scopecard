"""Reconstruct WN18 / WN18RR's eligibility rule from WordNet 3.0 and compute W.

Bordes et al. 2014 (Mach Learn 94:233, Sect. 5.1): synsets connected by 18
relation types were kept, "we filtered out the synsets appearing in less that
15 triplets", giving 40,943 synsets. WN18RR (Dettmers et al. 2018) keeps WN18's
entities and drops inverse relations. This script rebuilds the relation graph
over all WordNet 3.0 synsets, checks the release against it, tests readings of
the stated rule, and reports the excluded share of two declared universes.

usage: python -I wn18_scope.py <WordNet-3.0/dict> <wordnet-mlj12 dir> <WN18RR dir> <out dir>
"""
import collections
import os
import sys

# WN18 relation name -> WordNet 3.0 pointer symbol; triple read as (source, rel, target)
REL = {
    '_hypernym': '@', '_hyponym': '~',
    '_instance_hypernym': '@i', '_instance_hyponym': '~i',
    '_member_holonym': '#m', '_member_meronym': '%m',
    '_part_of': '#p', '_has_part': '%p',
    '_derivationally_related_form': '+',
    '_synset_domain_topic_of': ';c', '_member_of_domain_topic': '-c',
    '_synset_domain_region_of': ';r', '_member_of_domain_region': '-r',
    '_synset_domain_usage_of': ';u', '_member_of_domain_usage': '-u',
    '_also_see': '^', '_verb_group': '$', '_similar_to': '&',
}
SYM2REL = {v: k for k, v in REL.items()}
TAG2POS = {'NN': 'n', 'VB': 'v', 'JJ': 'a', 'RB': 'r'}


def read_wordnet(dict_dir):
    """Return {(pos, offset): ss_type} and every pointer as ((pos, off), symbol, (pos, off)).
    Satellite adjectives ('s') live in data.adj and share its offset space, so they key as 'a'."""
    synsets, ptrs = {}, []
    for pos in ('noun', 'verb', 'adj', 'adv'):
        with open(os.path.join(dict_dir, 'data.' + pos), encoding='latin-1') as f:
            for line in f:
                if line.startswith('  '):
                    continue
                tok = line.split(' | ')[0].split()
                off, sstype = tok[0], tok[2]
                src = ('a' if sstype == 's' else sstype, off)
                synsets[src] = sstype
                i = 4 + 2 * int(tok[3], 16)
                p_cnt = int(tok[i])
                i += 1
                for _ in range(p_cnt):
                    sym, toff, tpos = tok[i], tok[i + 1], tok[i + 2]
                    i += 4
                    ptrs.append((src, sym, ('a' if tpos == 's' else tpos, toff)))
    return synsets, ptrs


def read_release(paths):
    trip = []
    for p in paths:
        with open(p, encoding='utf-8') as f:
            for line in f:
                h, r, t = line.rstrip('\n').split('\t')
                trip.append((h, r, t))
    return trip


def degrees(triples):
    d = collections.Counter()
    for s, _, t in triples:
        d[s] += 1
        d[t] += 1
    return d


def main(dict_dir, wn18_dir, wn18rr_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out = []

    def say(*a):
        print(*a)
        out.append(' '.join(str(x) for x in a))

    syn_pos, ptrs_pos = read_wordnet(dict_dir)
    say('WordNet 3.0 synsets', len(syn_pos), dict(collections.Counter(p for p, _ in syn_pos)))

    # --- identity: WN18 ids are bare offsets, unique only within one POS file
    n_off = collections.Counter(off for _, off in syn_pos)
    shared = {o for o, c in n_off.items() if c > 1}
    say('offsets shared by synsets of different POS:', len(shared), '(synsets involved:', sum(n_off[o] for o in shared), ')')
    full_pos = {(s, SYM2REL[sym], t) for s, sym, t in ptrs_pos if sym in SYM2REL}
    full = {(s[1], r, t[1]) for s, r, t in full_pos}          # keyed the way WN18 keys it
    offsets = set(n_off)

    wn18 = read_release([os.path.join(wn18_dir, f'wordnet-mlj12-{s}.txt') for s in ('train', 'valid', 'test')])
    wn18_set = set(wn18)
    E = {h for h, _, _ in wn18} | {t for _, _, t in wn18}
    pos_tag = {}
    with open(os.path.join(wn18_dir, 'wordnet-mlj12-definitions.txt'), encoding='utf-8') as f:
        for line in f:
            sid, name = line.split('\t')[:2]
            pos_tag[sid] = TAG2POS[name.rsplit('_', 2)[-2]]
    say('WN18 triples', len(wn18), 'unique', len(wn18_set), 'entities', len(E), 'not a WordNet 3.0 offset', len(E - offsets))

    # --- the release against the rebuilt graph
    say('WN18 triples reproduced from WordNet 3.0 (bare-offset keys):', len(wn18_set & full), 'of', len(wn18_set))
    say('  not reproduced:', len(wn18_set - full), collections.Counter(r for _, r, _ in wn18_set - full).most_common())
    induced = {x for x in full if x[0] in E and x[2] in E}
    say('triples of the rebuilt graph induced on the WN18 entity set:', len(induced),
        '| in release', len(induced & wn18_set), '| induced but absent from release', len(induced - wn18_set))
    say('  => WN18 is (up to', len(induced - wn18_set), 'triples) the induced subgraph on its entity set: the entity set is the rule')

    # merged identities: WN18 entities whose released triples come from two different-POS synsets
    used_pos = collections.defaultdict(set)
    for s, r, t in full_pos:
        if (s[1], r, t[1]) in wn18_set:
            used_pos[s[1]].add(s[0])
            used_pos[t[1]].add(t[0])
    merged = {e for e, ps in used_pos.items() if len(ps) > 1}
    say('WN18 entities on a shared offset:', len(E & shared), '| of these, carrying triples of two different synsets (merged):', len(merged))
    say('  example:', sorted(merged)[:3], [sorted(used_pos[e]) for e in sorted(merged)[:3]],
        'named in definitions file as POS', [pos_tag[e] for e in sorted(merged)[:3]])

    # --- readings of the stated rule
    dfull = degrees(full)
    say('stated rule check: WN18 entities with < 15 triples in the rebuilt 18-relation graph:',
        sum(dfull[e] < 15 for e in E), 'of', len(E), '| min degree', min(dfull[e] for e in E))
    say('WN18 entities with < 15 triples in the release itself:', sum(c < 15 for c in degrees(wn18_set).values()))

    def single_pass(T):
        keep = {e for e, c in dfull.items() if c >= T}
        return {e for s, _, t in full if s in keep and t in keep for e in (s, t)}

    def iterated(T):
        cur = set(full)
        while True:
            d = degrees(cur)
            nxt = {x for x in cur if d[x[0]] >= T and d[x[2]] >= T}
            if nxt == cur:
                return {e for s, _, t in cur for e in (s, t)}
            cur = nxt

    def seeds_and_neighbours(T):
        seeds = {e for e, c in dfull.items() if c >= T}
        return {e for s, _, t in full if s in seeds or t in seeds for e in (s, t)}

    jac = lambda S: len(S & E) / len(S | E)
    for name, f in (('single pass', single_pass), ('iterated (15-core)', iterated), ('seeds + neighbours', seeds_and_neighbours)):
        S = f(15)
        say(f'reading "{name}", T=15: {len(S)} entities, overlap {len(S & E)}, Jaccard {jac(S):.3f}')
    best = max(((jac(single_pass(T)), T) for T in range(1, 41)))
    say(f'best single-pass threshold over T=1..40: T={best[1]}, Jaccard {best[0]:.3f}')
    best = max(((jac(seeds_and_neighbours(T)), T) for T in range(1, 41)))
    say(f'best seeds+neighbours threshold over T=1..40: T={best[1]}, Jaccard {best[0]:.3f}')

    with open(os.path.join(out_dir, 'wn18_by_full_degree.tsv'), 'w', encoding='utf-8') as f:
        f.write('full_degree_18rel(40=40+)\toffsets\tin_wn18\tshare\n')
        tab = collections.defaultdict(lambda: [0, 0])
        for e in offsets:
            b = min(dfull[e], 40)
            tab[b][0] += 1
            tab[b][1] += e in E
        for b in sorted(tab):
            n, k = tab[b]
            f.write(f'{b}\t{n}\t{k}\t{k / n:.4f}\n')

    # --- W, entity side, for WN18RR (its 11 relation types)
    rr_files = [os.path.join(wn18rr_dir, f'{s}.txt') for s in ('train', 'valid', 'test')]
    rr = read_release(rr_files)
    rr_rel = {r for _, r, _ in rr}
    E_rr = {h for h, _, _ in rr} | {t for _, _, t in rr}
    say('WN18RR triples', len(rr), 'relations', len(rr_rel), 'entities', len(E_rr), 'outside WN18 entities', len(E_rr - E))
    rr_full = {x for x in full if x[1] in rr_rel}
    say('WN18RR triples reproduced from WordNet 3.0:', len(set(rr) & rr_full), 'of', len(set(rr)))
    conn_rr = {e for s, _, t in rr_full for e in (s, t)}
    for name, U in (('all WordNet 3.0 offsets', offsets), ("offsets with >= 1 triple of WN18RR's 11 relations", conn_rr)):
        k = len(E_rr & U)
        say(f'W_entity, universe = {name}: {1 - k / len(U):.4f} ({len(U) - k} of {len(U)} excluded)')
    by_pos_all = collections.Counter(p for p, _ in syn_pos)
    by_pos_in = collections.Counter(pos_tag[e] for e in E_rr)
    say('excluded share by POS (definitions-file tag; offsets merged across POS counted once):',
        {p: round(1 - by_pos_in[p] / by_pos_all[p], 4) for p in 'nvar'})
    # triples of the 11 relations that the entity rule removes
    say('rebuilt WN18RR-relation triples', len(rr_full), '| with both ends retained', len({x for x in rr_full if x[0] in E_rr and x[2] in E_rr}),
        '| touching an excluded entity', len({x for x in rr_full if x[0] not in E_rr or x[2] not in E_rr}))
    train_ids = {e for h, _, t in read_release(rr_files[:1]) for e in (h, t)}
    rr_test = read_release(rr_files[2:])
    say('WN18RR test triples with an entity absent from train', sum(h not in train_ids or t not in train_ids for h, _, t in rr_test), 'of', len(rr_test))

    with open(os.path.join(out_dir, 'wn18_scope.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:5])
