"""Format results/wn18rr_analysis/records.tsv into summary.md (no new numbers: every
value printed here is a record).

usage: python -E -P summarize_wn18rr.py <analysis dir>
"""
import collections
import os
import sys

ORDER = ['popularity', 'transe', 'conve', 'distmult', 'rescal', 'complex', 'nbfnet']


def main(d):
    R = {}
    with open(os.path.join(d, 'records.tsv'), encoding='utf-8') as f:
        head = f.readline().rstrip('\n').split('\t')
        for line in f:
            r = dict(zip(head, line.rstrip('\n').split('\t')))
            R[(r['design'], r['estimand'], r['subject'], r['seed'], r['quantity'], r['cluster'])] = r
    out = []
    fmt = lambda r: f"{float(r['value']):.4f} [{float(r['lo']):.4f}, {float(r['hi']):.4f}]"
    g = lambda *k: R.get(k)

    models = [m for m in ORDER if g('E2', 'primary', m, 'mean', 'auc_O', 'triple')]
    out.append('# WN18RR measured c: summary of records\n')
    out.append('Seed-mean values (per-query values averaged over seeds) with 95% triple-bootstrap intervals.\n')
    for est in ('primary', 'S1', 'S2', 'S3', 'S4', 'posthoc_noleak'):
        if not any(k[0] == 'E2' and k[1] == est for k in R):      # S3 appears once its runs exist
            continue
        out.append(f'\n## E2 {est}: per model\n')
        out.append('| model | W | AUC O | AUC U | AUC U − O | AUC C | MRR O | MRR C | MRR C − O | share outranked by U | top-10 drop |')
        out.append('|---|---|---|---|---|---|---|---|---|---|---|')
        for m in models:
            k = lambda q: g('E2', est, m, 'mean', q, 'triple')
            if not k('auc_O'):
                continue
            out.append(f"| {m} | {float(k('W')['value']):.4f} | {fmt(k('auc_O'))} | {fmt(k('auc_U'))} | {fmt(k('auc_U_minus_O'))} | "
                       f"{fmt(k('auc_C'))} | {fmt(k('rr_O'))} | {fmt(k('rr_C'))} | {fmt(k('mrr_C_minus_O'))} | "
                       f"{fmt(k('intrusion'))} | {fmt(k('top10_drop'))} |")
        out.append(f'\n## E2 {est}: pairs (A − B), every seed and the seed mean\n')
        out.append('| pair | seed | Δ_O | Δ_U | Δ_target | Δ_U − Δ_O | c | c* | ΔMRR_O | ΔMRR_C |')
        out.append('|---|---|---|---|---|---|---|---|---|---|')
        pairs = sorted({k[2] for k in R if k[0] == 'E2' and k[1] == est and '-' in k[2]},
                       key=lambda p: (ORDER.index(p.split('-')[0]), ORDER.index(p.split('-')[1])))
        for p in pairs:
            seeds = sorted({k[3] for k in R if k[0] == 'E2' and k[1] == est and k[2] == p}, key=lambda s: (s == 'mean', s))
            for s in seeds:
                k = lambda q: g('E2', est, p, s, q, 'triple')
                out.append(f"| {p} | {s} | {fmt(k('delta_O'))} | {fmt(k('delta_U'))} | {fmt(k('delta_target'))} | "
                           f"{fmt(k('delta_U_minus_O'))} | {fmt(k('c'))} | {fmt(k('c_star'))} | {fmt(k('delta_mrr_O'))} | {fmt(k('delta_mrr_C'))} |")

    out.append('\n## E1 benchmark (over O) and training-side effect (E2-O minus E1-O), seed by seed\n')
    out.append('| model | seed | AUC O (E1) | MRR O (E1) | E2−E1 AUC O | E2−E1 MRR O |')
    out.append('|---|---|---|---|---|---|')
    for m in ORDER:
        for s in sorted({k[3] for k in R if k[0] == 'E1' and k[2] == m}):
            a = g('E1', 'benchmark', m, s, 'auc_O', 'triple')
            b = g('E1', 'benchmark', m, s, 'rr_O', 'triple')
            c = g('E1', 'training_side', m, s, 'train_effect_auc_O', 'triple')
            e = g('E1', 'training_side', m, s, 'train_effect_mrr_O', 'triple')
            out.append(f"| {m} | {s} | {fmt(a)} | {fmt(b)} | {fmt(c) if c else ''} | {fmt(e) if e else ''} |")

    dep = sorted({(k[2], k[3]) for k in R if k[0] == 'deploy' and k[1] == 'primary'}, key=lambda x: (ORDER.index(x[0]), x[1] == 'mean', x[1]))
    if dep:
        out.append('\n## Deployment test: E1-trained NBFNet reading the E2 training graph (separate estimand, design Sect. 13 item 1)\n')
        out.append('Same queries, candidate cells and target filter as E2; every seed and the seed mean. No pairs: no other model can be deployed this way.\n')
        for est in ('primary', 'S1', 'S2', 'S4', 'posthoc_noleak'):
            out.append(f'\n### deploy {est}\n')
            out.append('| model | seed | W | AUC O | AUC U | AUC U − O | AUC C | MRR O | MRR C | MRR C − O | share outranked by U | top-10 drop |')
            out.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
            for m, s in dep:
                k = lambda q: g('deploy', est, m, s, q, 'triple')
                if not k('auc_O'):
                    continue
                out.append(f"| {m} | {s} | {float(k('W')['value']):.4f} | {fmt(k('auc_O'))} | {fmt(k('auc_U'))} | {fmt(k('auc_U_minus_O'))} | "
                           f"{fmt(k('auc_C'))} | {fmt(k('rr_O'))} | {fmt(k('rr_C'))} | {fmt(k('mrr_C_minus_O'))} | "
                           f"{fmt(k('intrusion'))} | {fmt(k('top10_drop'))} |")
        out.append('\n### Graph-side effect (deploy-O minus E1-O, same checkpoint), seed by seed\n')
        out.append('Changes the graph read at inference, the filter (benchmark → target triples) and the O keying (161 merged ids split).\n')
        out.append('| model | seed | deploy−E1 AUC O | deploy−E1 MRR O |')
        out.append('|---|---|---|---|')
        for m, s in dep:
            c, e = g('deploy', 'graph_side', m, s, 'graph_effect_auc_O', 'triple'), g('deploy', 'graph_side', m, s, 'graph_effect_mrr_O', 'triple')
            if c:
                out.append(f'| {m} | {s} | {fmt(c)} | {fmt(e)} |')

    out.append('\n## Query side (u_test, E2 models), seed by seed\n')
    out.append('| model | seed | stratum | n | AUC C | MRR C | Hits@10 C |')
    out.append('|---|---|---|---|---|---|---|')
    for k in sorted(k for k in R if k[0] == 'E2' and k[1].startswith('query_side') and k[4] == 'auc_C'):
        st = k[1].split(':', 1)[1]
        r1, r2, r3 = R[k], g('E2', k[1], k[2], k[3], 'rr_C', k[5]), g('E2', k[1], k[2], k[3], 'h10_C', k[5])
        out.append(f"| {k[2]} | {k[3]} | {st} | {r1['n_queries']} | {fmt(r1)} | {fmt(r2)} | {fmt(r3)} |")
    out.append('\n## POST HOC: query side by relation group and reverse-edge leakage (E2 models and deploy), seed by seed\n')
    out.append('Not prespecified (design Sect. 13 item 3). Compare rows with the same group across query sets.\n')
    out.append('| query set | group | model | n | AUC C | MRR C | Hits@10 C |')
    out.append('|---|---|---|---|---|---|---|')
    keys = sorted({(k[0], k[1], k[2]) for k in R if k[0] in ('E2', 'deploy') and k[1].startswith('posthoc_') and ':' in k[1]},
                  key=lambda x: (x[1], x[0] == 'deploy', x[2]))
    for design, est, m in keys:
        seeds = sorted({k[3] for k in R if k[0] == design and k[1] == est and k[2] == m})
        for s in seeds:
            r1 = g(design, est, m, s, 'auc_C', 'triple')
            if not r1:
                continue
            qset, grp = est[len('posthoc_'):].split(':', 1)
            who = f'{m} s{s}' + (' (deploy)' if design == 'deploy' else '')
            out.append(f"| {qset} | {grp} | {who} | {r1['n_queries']} | {fmt(r1)} | {fmt(g(design, est, m, s, 'rr_C', 'triple'))} | "
                       f"{fmt(g(design, est, m, s, 'h10_C', 'triple'))} |")
    with open(os.path.join(d, 'summary.md'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main(sys.argv[1])
