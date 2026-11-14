"""Write the LibKGE index file of O synsets for the E2 dataset (one index per line), used
as user.candidate_entities_file so that E2 validation ranks over O only
(MEASURED_C_DESIGN.md Sect. 12, decision 1).

usage: python make_o_candidates.py <target dir> <LibKGE wnrr_target dir>
"""
import os
import sys


def main(tdir, ddir):
    region = {}
    with open(os.path.join(tdir, 'entities.tsv'), encoding='utf-8') as f:
        head = f.readline().rstrip('\n').split('\t')
        for line in f:
            r = dict(zip(head, line.rstrip('\n').split('\t')))
            region[r['entity_id']] = r['region']
    idx = []
    with open(os.path.join(ddir, 'entity_ids.del'), encoding='utf-8') as f:
        for line in f:
            i, name = line.rstrip('\n').split('\t')
            if region[name] == 'O':
                idx.append(i)
    with open(os.path.join(ddir, 'o_entities.del'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(idx) + '\n')
    print('O candidates', len(idx), 'of', len(region))


if __name__ == '__main__':
    main(*sys.argv[1:3])
