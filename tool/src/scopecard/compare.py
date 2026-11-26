"""Break-even for a comparison made on the benchmark ([P2]).

For a metric that is linear in the candidate mix (per-query or pooled AUC with a fixed positive reference,
excluded candidates weighted like the benchmark's own), the target-level difference is
    delta_target = (1 - W) delta_O + W delta_U = delta_O - W c,   with c = delta_O - delta_U,
where delta_U is the two models' difference on the excluded candidates. The benchmark ordering
(sign of delta_O) is reversed exactly when
    c has the sign of delta_O and |c| > c* = |delta_O| / W     (the between-region change), or equivalently
    delta_U is opposite in sign and |delta_U| > (1 - W) |delta_O| / W   (the excluded-region difference).
c and delta_U need labels on the excluded region; the card reports only the thresholds.

The thresholds are exact when every query's candidate set has excluded share W. A card's W is a unit- or
pair-level share; when per-query candidate shares vary, delta_target = delta_O - mean(W) c holds only up to
the covariance of the per-query shares and differences, and the thresholds carry no guarantee. With per-query
shares (scope.query_shares, scope.user_shares), W is their mean: c* is then exact for the c of the paper's
Corollary 1, and only the delta_U threshold takes that covariance as zero. MRR and
Hits@k are not linear in the candidate mix, so no threshold is given for them.
"""
from typing import Optional

LINEAR_METRICS = ('auc',)
METRICS = ('auc', 'mrr', 'hits')


def c_star(delta_O: float, W: Optional[float], metric: str = 'auc', W_source: str = '',
           per_query: Optional[dict] = None) -> dict:
    """Break-even values for a benchmark difference delta_O (first model minus second) on `metric`.
    W_source says which share W is (for example 'entity share'), so the card states what was assumed.
    per_query: the card's exclusion.per_query block, whose mean W_q is W. c* is then exact (Corollary 1) for a
    delta_O measured on the same queries under the same filter, and is withheld when a query keeps no candidate."""
    if metric not in METRICS:
        raise ValueError(f'metric must be one of {METRICS}')
    out = {'metric': metric, 'delta_O': delta_O, 'W': W, 'W_source': W_source, 'applicable': False,
           'c_star': None, 'delta_U_break_even': None}
    lead = 'first' if delta_O > 0 else 'second'
    if not (isinstance(delta_O, (int, float)) and -1 <= delta_O <= 1):      # NaN and infinities fail here too
        out['statement'] = 'No break-even is given: delta_O must be a finite difference in [-1, 1].'
        return out
    if metric not in LINEAR_METRICS:
        out['statement'] = (f'No break-even is given for {metric.upper()}: it is not linear in the candidate mix, so the '
                            f'benchmark difference does not decompose into regional differences.')
        return out
    if W is None or W != W or not 0 <= W <= 1:
        out['statement'] = 'No break-even is given: W is undefined for this release.'
        return out
    if per_query and per_query.get('queries_without_kept'):
        out['statement'] = (f"No break-even is given: the benchmark difference is not defined on the "
                            f"{per_query['queries_without_kept']:,} of the {per_query['queries']:,} test queries that keep no "
                            f"candidate of the release.")
        return out
    if delta_O == 0:
        out['statement'] = 'No break-even is given: the two models tie on the benchmark.'
        return out
    if W == 0:
        out.update(applicable=True)
        out['statement'] = (f'The {lead} model leads by {abs(delta_O):.4g} AUC on the benchmark, and nothing of the declared '
                            f'universe is excluded, so no difference on an excluded part can reverse it.')
        return out
    cs = abs(delta_O) / W
    du = (1 - W) * abs(delta_O) / W
    if per_query:
        n = per_query.get('queries', 0)
        out.update(applicable=True, c_star=cs, delta_U_break_even=du,
                   assumptions=['a fixed positive reference', 'excluded candidates weighted like the benchmark candidates',
                                f'delta_O is the mean per-query difference over the same {n:,} test queries, each ranked '
                                f'against its kept candidates under the filter of exclusion.per_query',
                                f'W = {W:.4g} is the mean of the per-query excluded shares W_q'
                                + (f' ({W_source})' if W_source else '')
                                + '; c* is then exact for c, the W_q-weighted mean of the per-query changes (Corollary 1, '
                                  'which writes it as delta_O - delta_U + Cov(W_q, c_q) / W); the delta_U threshold also '
                                  'takes that covariance as zero'])
        out['statement'] = (f'The {lead} model leads by {abs(delta_O):.4g} AUC on the benchmark. On the declared universe '
                            f'that ordering reverses exactly when the change between the two regions, c (the per-query '
                            f'changes delta_O,q - delta_U,q averaged with weights W_q), has the sign of delta_O and its '
                            f'magnitude exceeds c* = |delta_O| / mean(W_q) = {cs:.4g}. This holds for a delta_O measured on '
                            f'the same {n:,} test queries, each ranked against its kept candidates under the filter above; '
                            f'for a difference from another query set or from the benchmark\'s own filter it carries no '
                            f'guarantee, and delta_O has to be recomputed on these queries. '
                            f'If the per-query shares do not co-vary with the per-query changes, the ordering reverses when '
                            f'the models\' difference on the excluded part is in the opposite direction and larger than '
                            f'{du:.4g}.')
        return out
    out.update(applicable=True, c_star=cs, delta_U_break_even=du,
               assumptions=['a fixed positive reference', 'excluded candidates weighted like the benchmark candidates',
                            f'every query\'s candidate set has excluded share W = {W:.4g}'
                            + (f' (the card\'s {W_source})' if W_source else '')
                            + '; otherwise the thresholds carry no guarantee'])
    out['statement'] = (f'The {lead} model leads by {abs(delta_O):.4g} AUC on the benchmark. On the declared universe '
                        f'that ordering reverses only if the models\' difference on the excluded part is in the opposite '
                        f'direction and larger than {du:.4g}; equivalently, if the change between the two regions, '
                        f'c = delta_O - delta_U, has the sign of delta_O and its magnitude exceeds '
                        f'c* = |delta_O| / W = {cs:.4g}.')
    return out
