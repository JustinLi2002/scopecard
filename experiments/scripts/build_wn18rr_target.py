"""Build the declared target population for the WN18RR measured-c experiment.

Target: every WordNet 3.0 synset that carries at least one triple of WN18RR's 11
relation types, with all such triples (lexical pointers lifted to synsets,
duplicates collapsed). Synsets are identified by (POS, offset), satellite
adjectives keyed as 'a' because they share data.adj's offset space; WN18RR's
bare-offset ids are mapped onto this identity through the triples they occur in.

Regions: O = synsets that occur in a WN18RR triple (the benchmark's entity set),
U = the rest of the target. Roles of triples:
  bench_train / bench_valid / bench_test  WN18RR triples, mapped
  u_valid / u_test                        held-out triples with >= 1 U end, sampled
  extra                                   all other target triples
E2 trains on bench_train + extra; bench_test is the fixed positive reference of
the primary estimand; u_test serves the secondary (query-side) estimand only.

usage: python -I build_wn18rr_target.py <WordNet-3.0/dict> <WN18RR dir> <out dir>
"""
import collections
import hashlib
import os
import random
import sys

SEED = 20261018
REL = {   # WN18RR relation -> WordNet 3.0 pointer symbol, triple read as (source, rel, target)
    '_hypernym': '@', '_instance_hypernym': '@i', '_member_meronym': '%m', '_has_part': '%p',
    '_derivationally_related_form': '+', '_synset_domain_topic_of': ';c',
    '_member_of_domain_region': '-r', '_member_of_domain_usage': '-u',
    '_also_see': '^', '_verb_group': '$', '_similar_to': '&',
}
SYM2REL = {v: k for k, v in REL.items()}


def read_wordnet(dict_dir):
    synsets, ptrs = {}, []
    for pos in ('noun', 'verb', 'adj', 'adv'):
        with open(os.path.join(dict_dir, 'data.' + pos), encoding='latin-1') as f:
            for line in f:
                if line.startswith('  '):
                    continue
                head, _, gloss = line.partition(' | ')
                tok = head.split()
                off, sstype = tok[0], tok[2]
                sid = ('a' if sstype == 's' else sstype) + off
                lemmas = [tok[4 + 2 * k] for k in range(int(tok[3], 16))]
                synsets[sid] = (sstype, lemmas[0], gloss.strip())
                i = 4 + 2 * int(tok[3], 16)
                p_cnt = int(tok[i])
                i += 1
                for _ in range(p_cnt):
                    sym, toff, tpos = tok[i], tok[i + 1], tok[i + 2]
                    i += 4
                    ptrs.append((sid, sym, ('a' if tpos == 's' else tpos) + toff))
    return synsets, ptrs


def read_release(path):
    with open(path, encoding='utf-8') as f:
        return [tuple(line.rstrip('\n').split('\t')) for line in f if line.strip()]


def main(dict_dir, rr_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log = []

    def say(*a):
        print(*a)
        log.append(' '.join(str(x) for x in a))

    synsets, ptrs = read_wordnet(dict_dir)
    G = sorted({(s, SYM2REL[sym], t) for s, sym, t in ptrs if sym in SYM2REL})
    Gset = set(G)
    say('target triples (11 relations, synset level)', len(G))

    # map WN18RR triples (bare offsets) onto (POS, offset) triples
    by_off = collections.defaultdict(list)
    for h, r, t in G:
        by_off[(h[1:], r, t[1:])].append((h, r, t))
    roles = {}
    unmatched, ambiguous = [], []
    for split in ('train', 'valid', 'test'):
        for h, r, t in read_release(os.path.join(rr_dir, f'{split}.txt')):
            cand = by_off.get((h, r, t), [])
            if not cand:
                unmatched.append((split, h, r, t))
                continue
            if len(cand) > 1:
                ambiguous.append((split, h, r, t, cand))
            for x in cand:
                roles.setdefault(x, 'bench_' + split)
    say('WN18RR triples mapped:', sum(1 for v in roles.values()), '| unmatched:', len(unmatched),
        '| ambiguous (mapped to every candidate):', len(ambiguous))
    O = {e for (h, _, t), role in roles.items() for e in (h, t)}
    V = {e for h, _, t in G for e in (h, t)}
    U = V - O
    say('universe V (synsets with >= 1 target triple):', len(V), '| O:', len(O), '| U:', len(U),
        f'| W_rule (entity share of U) = {len(U) / len(V):.6f}')

    # held-out triples touching U, sampled so that both ends keep >= 1 training triple
    rng = random.Random(SEED)
    pool = sorted(x for x in G if x not in roles and (x[0] in U or x[2] in U))
    rng.shuffle(pool)
    deg = collections.Counter()
    train_like = [x for x in G if roles.get(x) not in ('bench_valid', 'bench_test')]
    for h, _, t in train_like:
        deg[h] += 1
        deg[t] += 1
    want = {'u_valid': 3034, 'u_test': 3134}
    for role, n in want.items():
        got = 0
        for x in pool:
            if got == n:
                break
            if x in roles:
                continue
            h, _, t = x
            if deg[h] > 1 and deg[t] > 1:
                roles[x] = role
                deg[h] -= 1
                deg[t] -= 1
                got += 1
        say(f'{role}: sampled {got} of {n} (pool {len(pool)})')
    for x in G:
        roles.setdefault(x, 'extra')

    train = [x for x in G if roles[x] in ('bench_train', 'extra')]
    dtrain = collections.Counter(e for h, _, t in train for e in (h, t))
    say('E2 training triples', len(train), '| synsets with no E2 training triple:', sum(1 for e in V if dtrain[e] == 0),
        '(O:', sum(1 for e in O if dtrain[e] == 0), ', U:', sum(1 for e in U if dtrain[e] == 0), ')')
    say('roles', dict(collections.Counter(roles.values())))
    strata = collections.Counter(('U' if h in U else 'O') + '-' + ('U' if t in U else 'O')
                                 for x, role in roles.items() if role == 'u_test' for h, t in [(x[0], x[2])])
    say('u_test strata (head-tail region):', dict(strata))

    # hidden positives: true answers of benchmark test queries that lie in U
    tails = collections.defaultdict(set)
    heads = collections.defaultdict(set)
    for h, r, t in G:
        tails[(h, r)].add(t)
        heads[(r, t)].add(h)
    bt = [x for x, role in roles.items() if role == 'bench_test']
    hid_t = sum(len(tails[(h, r)] & U) for h, r, t in bt)
    hid_h = sum(len(heads[(r, t)] & U) for h, r, t in bt)
    q_with = sum(1 for h, r, t in bt if tails[(h, r)] & U) + sum(1 for h, r, t in bt if heads[(r, t)] & U)
    say('benchmark test queries (2 per triple):', 2 * len(bt), '| with >= 1 true answer in U (filtered in the target, absent in the benchmark):',
        q_with, '| such answers: tail', hid_t, 'head', hid_h)

    # tables
    ent_rows = []
    for e in sorted(V):
        sst, lemma, gloss = synsets[e]
        ent_rows.append((e, e[0], e[1:], sst, 'O' if e in O else 'U', str(dtrain[e]), lemma))
    with open(os.path.join(out_dir, 'entities.tsv'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('entity_id\tpos\toffset\tss_type\tregion\ttrain_degree\tfirst_lemma\n')
        f.writelines('\t'.join(r) + '\n' for r in ent_rows)
    with open(os.path.join(out_dir, 'triples.tsv'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('triple_id\thead\trelation\ttail\trole\n')
        for h, r, t in G:
            tid = hashlib.sha1(f'{h}|{r}|{t}'.encode()).hexdigest()[:16]
            f.write(f'{tid}\t{h}\t{r}\t{t}\t{roles[(h, r, t)]}\n')
    with open(os.path.join(out_dir, 'unmatched_wn18rr.tsv'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('split\thead\trelation\ttail\n')
        f.writelines('\t'.join(u) + '\n' for u in unmatched)
    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(log) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:4])
