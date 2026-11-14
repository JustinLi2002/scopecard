"""Acceptance checks for the WN18RR declared target (results/wn18rr_target).

Each check prints PASS/FAIL; the script exits non-zero on any failure. Run after
build_wn18rr_target.py and before any training. Check 14 rebuilds the tables
in a temporary folder and compares hashes (determinism).

usage: python -I accept_wn18rr_target.py <target dir> <WN18RR dir> <WordNet-3.0/dict> <build script>
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


def main(tdir, rr_dir, dict_dir, build):
    ok = True

    def check(n, cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f'{n:2d} {"PASS" if cond else "FAIL"}  {msg}')

    E = read_tsv(os.path.join(tdir, 'entities.tsv'))
    T = read_tsv(os.path.join(tdir, 'triples.tsv'))
    ent = {e['entity_id']: e for e in E}
    rr = {s: [tuple(l.rstrip('\n').split('\t')) for l in open(os.path.join(rr_dir, f'{s}.txt'), encoding='utf-8') if l.strip()]
          for s in ('train', 'valid', 'test')}
    roles = collections.Counter(t['role'] for t in T)
    unm = read_tsv(os.path.join(tdir, 'unmatched_wn18rr.tsv'))

    check(1, len(ent) == len(E), 'entity ids unique')
    check(2, len({t['triple_id'] for t in T}) == len(T) and len({(t['head'], t['relation'], t['tail']) for t in T}) == len(T),
          'triple ids and (h, r, t) unique')
    check(3, all(t['head'] in ent and t['tail'] in ent for t in T), 'every triple end is a target entity')
    check(4, all(int(e['train_degree']) >= 0 for e in E) and {e['region'] for e in E} == {'O', 'U'}, 'regions are exactly O and U')
    rels = {t['relation'] for t in T}
    check(5, rels == {r for s in rr.values() for _, r, _ in s} and len(rels) == 11, 'the 11 WN18RR relations, no others')
    for k, s in enumerate(('train', 'valid', 'test')):
        n_rel = len(rr[s])
        n_map = roles['bench_' + s]
        n_un = sum(1 for u in unm if u['split'] == s)
        check(6 + k, n_map + n_un == n_rel, f'bench_{s}: {n_map} mapped + {n_un} unmatched = {n_rel} WN18RR triples')
    bare = {(t['head'][1:], t['relation'], t['tail'][1:]) for t in T if t['role'].startswith('bench_')}
    rr_all = {x for s in rr.values() for x in s}
    unm_set = {(u['head'], u['relation'], u['tail']) for u in unm}
    check(9, bare == rr_all - unm_set, 'mapped benchmark triples equal WN18RR (bare offsets) minus the unmatched')
    O = {e['entity_id'] for e in E if e['region'] == 'O'}
    O_from_bench = {x for t in T if t['role'].startswith('bench_') for x in (t['head'], t['tail'])}
    check(10, O == O_from_bench, f'O equals the synsets of the mapped benchmark triples ({len(O)})')
    rr_ids = {x for s in rr.values() for h, _, t in s for x in (h, t)}
    check(11, {e[1:] for e in O} == rr_ids, f'O covers exactly the {len(rr_ids)} WN18RR ids (bare offsets)')
    train = [t for t in T if t['role'] in ('bench_train', 'extra')]
    held = {(t['head'], t['relation'], t['tail']) for t in T if t['role'] in ('bench_valid', 'bench_test', 'u_valid', 'u_test')}
    check(12, not ({(t['head'], t['relation'], t['tail']) for t in train} & held), 'no held-out triple in E2 training')
    deg = collections.Counter(x for t in train for x in (t['head'], t['tail']))
    check(13, all(int(e['train_degree']) == deg[e['entity_id']] for e in E), 'train_degree column matches E2 training triples')
    uh = [t for t in T if t['role'] in ('u_valid', 'u_test')]
    check(14, all((t['head'] in ent and ent[t['head']]['region'] == 'U') or ent[t['tail']]['region'] == 'U' for t in uh)
          and all(deg[t['head']] >= 1 and deg[t['tail']] >= 1 for t in uh), 'u_valid/u_test: >= 1 U end, both ends trained')
    U_untrained = [e for e in E if e['region'] == 'U' and deg[e['entity_id']] == 0]
    check(15, not U_untrained, 'every U candidate has >= 1 E2 training triple')
    nO, nU = len(O), len(E) - len(O)
    w = nU / len(E)
    summ = open(os.path.join(tdir, 'summary.txt'), encoding='utf-8').read()
    check(16, f'W_rule (entity share of U) = {w:.6f}' in summ, f'W_rule recomputed from entities.tsv = {w:.6f} matches summary.txt')
    check(17, roles['u_valid'] == 3034 and roles['u_test'] == 3134, 'u_valid 3034 and u_test 3134 sampled')
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, '-I', build, dict_dir, rr_dir, tmp], check=True, capture_output=True)
        same = all(sha(os.path.join(tmp, f)) == sha(os.path.join(tdir, f))
                   for f in ('entities.tsv', 'triples.tsv', 'unmatched_wn18rr.tsv', 'summary.txt'))
    check(18, same, 'rebuild in a fresh folder is byte-identical (deterministic sampling)')
    with open(os.path.join(tdir, 'target_files.sha256'), 'w', encoding='utf-8', newline='\n') as f:
        for fn in ('entities.tsv', 'triples.tsv', 'unmatched_wn18rr.tsv', 'summary.txt'):
            f.write(f'{sha(os.path.join(tdir, fn))}  {fn}\n')
    print('ALL PASS' if ok else 'SOME CHECKS FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main(*sys.argv[1:5])
