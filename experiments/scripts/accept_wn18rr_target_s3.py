"""Acceptance checks for the S3 target (results/wn18rr_target_s3), the primary target keyed by bare offset.

Each check prints PASS/FAIL; the script exits non-zero on any failure. Run after build_wn18rr_target_s3.py
and before any training. The last check rebuilds the tables in a temporary folder and compares hashes.

usage: python -I accept_wn18rr_target_s3.py <S3 target dir> <primary target dir> <WN18RR dir> <build script>
"""
import collections
import hashlib
import os
import subprocess
import sys
import tempfile


def read_tsv(path):
    with open(path, encoding='utf-8') as f:
        head = f.readline().rstrip('\n').split('\t')
        return [dict(zip(head, line.rstrip('\n').split('\t'))) for line in f]


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def main(sdir, pdir, rr_dir, build):
    ok = True

    def check(n, cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f'{n:2d} {"PASS" if cond else "FAIL"}  {msg}')

    E, T = read_tsv(os.path.join(sdir, 'entities.tsv')), read_tsv(os.path.join(sdir, 'triples.tsv'))
    PE, PT = read_tsv(os.path.join(pdir, 'entities.tsv')), read_tsv(os.path.join(pdir, 'triples.tsv'))
    ent = {e['entity_id']: e for e in E}
    rr = {s: [tuple(l.rstrip('\n').split('\t')) for l in open(os.path.join(rr_dir, f'{s}.txt'), encoding='utf-8') if l.strip()]
          for s in ('train', 'valid', 'test')}
    unm = read_tsv(os.path.join(pdir, 'unmatched_wn18rr.tsv'))
    roles = collections.Counter(t['role'] for t in T)

    check(1, len(ent) == len(E), 'entity ids unique')
    check(2, len({t['triple_id'] for t in T}) == len(T) and len({(t['head'], t['relation'], t['tail']) for t in T}) == len(T),
          'triple ids and (h, r, t) unique')
    check(3, all(t['head'] in ent and t['tail'] in ent for t in T), 'every triple end is an S3 entity')
    check(4, {e['region'] for e in E} == {'O', 'U'}, 'regions are exactly O and U')
    check(5, {t['relation'] for t in T} == {r for s in rr.values() for _, r, _ in s}, 'the 11 WN18RR relations, no others')
    for k, s in enumerate(('train', 'valid', 'test')):
        n_un = sum(1 for u in unm if u['split'] == s)
        check(6 + k, roles['bench_' + s] + n_un == len(rr[s]), f'bench_{s}: {roles["bench_" + s]} + {n_un} unmatched = {len(rr[s])}')
    bench = {(t['head'], t['relation'], t['tail']) for t in T if t['role'].startswith('bench_')}
    rr_all = {x for s in rr.values() for x in s}
    check(9, bench == rr_all - {(u['head'], u['relation'], u['tail']) for u in unm},
          'benchmark triples equal the WN18RR triples themselves (minus the 5 unmatched): no id mapping is needed')
    O = {e['entity_id'] for e in E if e['region'] == 'O'}
    check(10, O == {x for h, _, t in bench for x in (h, t)}, f'O equals the ends of the benchmark triples ({len(O)})')
    rr_ids = {x for s in rr.values() for h, _, t in s for x in (h, t)}
    check(11, O == rr_ids, f'O is exactly the {len(rr_ids)} WN18RR ids')
    train = [t for t in T if t['role'] in ('bench_train', 'extra')]
    held = {(t['head'], t['relation'], t['tail']) for t in T if t['role'] in ('bench_valid', 'bench_test', 'u_valid', 'u_test')}
    check(12, not ({(t['head'], t['relation'], t['tail']) for t in train} & held), 'no held-out triple in E2 training')
    deg = collections.Counter(x for t in train for x in (t['head'], t['tail']))
    check(13, all(int(e['train_degree']) == deg[e['entity_id']] for e in E), 'train_degree matches the S3 training triples')
    uh = [t for t in T if t['role'] in ('u_valid', 'u_test')]
    check(14, all(ent[t['head']]['region'] == 'U' or ent[t['tail']]['region'] == 'U' for t in uh)
          and all(deg[t['head']] >= 1 and deg[t['tail']] >= 1 for t in uh), 'u_valid/u_test: >= 1 U end, both ends trained')
    check(15, all(deg[e['entity_id']] >= 1 for e in E if e['region'] == 'U'), 'every U candidate has >= 1 training triple')
    w = (len(E) - len(O)) / len(E)
    check(16, f'W_rule (entity share of U) = {w:.6f}' in open(os.path.join(sdir, 'summary.txt'), encoding='utf-8').read(),
          f'W_rule recomputed = {w:.6f} matches summary.txt')
    check(17, roles == collections.Counter(t['role'] for t in PT), f'role counts equal the primary target\'s {dict(roles)}')
    srole = {(t['head'], t['relation'], t['tail']): t['role'] for t in T}
    check(18, all(srole.get((t['head'][1:], t['relation'], t['tail'][1:])) == t['role'] for t in PT),
          'every primary triple projects to an S3 triple with the same role')
    mem = [m for e in E for m in e['members'].split(',')]
    check(19, sorted(mem) == sorted(e['entity_id'] for e in PE) and all(m[1:] == e['entity_id'] for e in E for m in e['members'].split(',')),
          'S3 entities partition the primary synsets by offset')
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, '-I', build, pdir, tmp], check=True, capture_output=True)
        same = all(sha(os.path.join(tmp, f)) == sha(os.path.join(sdir, f)) for f in ('entities.tsv', 'triples.tsv', 'summary.txt'))
    check(20, same, 'rebuild in a fresh folder is byte-identical')
    with open(os.path.join(sdir, 'target_files.sha256'), 'w', encoding='utf-8', newline='\n') as f:
        for fn in ('entities.tsv', 'triples.tsv', 'summary.txt'):
            f.write(f'{sha(os.path.join(sdir, fn))}  {fn}\n')
    print('ALL PASS' if ok else 'SOME CHECKS FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main(*sys.argv[1:5])
