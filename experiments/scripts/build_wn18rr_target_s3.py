"""Build the S3 target (MEASURED_C_DESIGN.md Sect. 5, S3 "Benchmark identity"): the primary declared target
with synsets keyed by bare offset, as WN18RR keys them, so synsets of different POS that share an offset merge.

It is a projection of the primary target (results/wn18rr_target), not a rebuild, so that S3 differs from the
primary estimand in identity only:
  entity     bare offset; members = the primary synsets with that offset; region O if any member is in O
  triple     (head offset, relation, tail offset), duplicates collapsed
  role       the role of its primary triple(s); when merged triples carry different roles, the first of
             bench_test, bench_valid, bench_train, u_test, u_valid, extra wins (benchmark splits are kept, and
             a held-out fact stays held out)
A u_valid / u_test triple whose ends would lose every training triple is not possible here (merging only adds
training triples), and the acceptance script checks it. train_degree is recomputed on the S3 training set.

usage: python -I build_wn18rr_target_s3.py <primary target dir> <out dir>
"""
import collections
import hashlib
import os
import sys

PRIORITY = ['bench_test', 'bench_valid', 'bench_train', 'u_test', 'u_valid', 'extra']


def read_tsv(path):
    with open(path, encoding='utf-8') as f:
        head = f.readline().rstrip('\n').split('\t')
        return [dict(zip(head, line.rstrip('\n').split('\t'))) for line in f]


def main(pdir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log = []

    def say(*a):
        print(*a)
        log.append(' '.join(str(x) for x in a))

    E = read_tsv(os.path.join(pdir, 'entities.tsv'))
    T = read_tsv(os.path.join(pdir, 'triples.tsv'))
    members = collections.defaultdict(list)
    for e in E:
        members[e['offset']].append(e)
    merged = {k: v for k, v in members.items() if len(v) > 1}
    say('primary synsets', len(E), '| S3 entities (bare offsets)', len(members), '| offsets shared by >1 synset', len(merged),
        '| of which touch O', sum(1 for v in merged.values() if any(e['region'] == 'O' for e in v)))

    roles = collections.defaultdict(set)
    for t in T:
        roles[(t['head'][1:], t['relation'], t['tail'][1:])].add(t['role'])
    role = {x: min(rs, key=PRIORITY.index) for x, rs in roles.items()}
    conflicts = collections.Counter(tuple(sorted(rs, key=PRIORITY.index)) for rs in roles.values() if len(rs) > 1)
    say('primary triples', len(T), '| S3 triples', len(role), '| collapsed', len(T) - len(role))
    say('merged triples with different primary roles (kept role first):', dict(conflicts) if conflicts else 'none')
    before = collections.Counter(t['role'] for t in T)
    after = collections.Counter(role.values())
    say('roles primary', dict(sorted(before.items())))
    say('roles S3     ', dict(sorted(after.items())))

    train = [x for x, r in role.items() if r in ('bench_train', 'extra')]
    dtrain = collections.Counter(e for h, _, t in train for e in (h, t))
    used = {e for h, _, t in role for e in (h, t)}
    assert used == set(members), 'every S3 entity carries a triple and every triple end is an S3 entity'
    O = {k for k, v in members.items() if any(e['region'] == 'O' for e in v)}
    U = set(members) - O
    say(f'O {len(O)} | U {len(U)} | W_rule (entity share of U) = {len(U) / len(members):.6f}')
    say('E2 training triples', len(train), '| entities with no training triple:', sum(1 for e in members if dtrain[e] == 0),
        '(O:', sum(1 for e in O if dtrain[e] == 0), ', U:', sum(1 for e in U if dtrain[e] == 0), ')')

    with open(os.path.join(out_dir, 'entities.tsv'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('entity_id\tpos\toffset\tss_type\tregion\ttrain_degree\tfirst_lemma\tmembers\n')
        for k in sorted(members):
            ms = sorted(members[k], key=lambda e: e['entity_id'])
            f.write('\t'.join([k, '/'.join(e['pos'] for e in ms), k, '/'.join(e['ss_type'] for e in ms),
                               'O' if k in O else 'U', str(dtrain[k]), ms[0]['first_lemma'],
                               ','.join(e['entity_id'] for e in ms)]) + '\n')
    with open(os.path.join(out_dir, 'triples.tsv'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('triple_id\thead\trelation\ttail\trole\n')
        for h, r, t in sorted(role):
            tid = hashlib.sha1(f'{h}|{r}|{t}'.encode()).hexdigest()[:16]
            f.write(f'{tid}\t{h}\t{r}\t{t}\t{role[(h, r, t)]}\n')
    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(log) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:3])
