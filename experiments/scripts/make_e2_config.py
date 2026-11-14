"""Write an E2 LibKGE config: the unmodified Ruffinelli et al. 2020 WN18RR config with two
changes only, the dataset (wnrr_target) and the O-only validation candidates
(user.candidate_entities_file, MEASURED_C_DESIGN.md Sect. 12 decision 1).

usage: python make_e2_config.py <wnrr-MODEL.yaml> <o_entities.del path> <out.yaml> [dataset name, default wnrr_target]
(S3 uses wnrr_target_s3, the same target keyed by bare offset.)
"""
import sys

import yaml


def main(src, cand, out, dataset='wnrr_target'):
    c = yaml.safe_load(open(src))
    c.setdefault('dataset', {})['name'] = dataset
    c.setdefault('user', {})['candidate_entities_file'] = cand
    with open(out, 'w') as f:
        yaml.safe_dump(c, f, sort_keys=True)
    print('wrote', out)


if __name__ == '__main__':
    main(*sys.argv[1:5])
