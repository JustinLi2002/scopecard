"""The scope card: one JSON record per (benchmark, universe) and a Markdown rendering."""
import json
import math
from typing import Dict, List, Optional

CARD_SCHEMA = 'benchscope-card/0.3'


def plural(n: int, word: str, many: str = '') -> str:
    return f"{n:,} {word if n == 1 else (many or word + 's')}"


def _clean(x):
    if isinstance(x, float):
        return None if math.isnan(x) else round(x, 6)
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def _no_false_bound(x: float, fmt) -> str:
    """Format with as many decimals as it takes not to show 0 or 1 (or 0% / 100%) for a value strictly between."""
    for d in range(fmt[1], fmt[1] + 6):
        s = fmt[0](x, d)
        v = float(s.rstrip('%')) / (100 if s.endswith('%') else 1)
        if not ((x < 1 and v >= 1) or (x > 0 and v <= 0)):
            return s
    return fmt[0](x, fmt[1] + 6)


def pct(w: float) -> str:
    """A share as a percentage that never rounds to 0% or 100% (Amazon-Book's 0.999742 must not read as 100.0%)."""
    return _no_false_bound(w, (lambda x, d: f'{x:.{d}%}', 1))


def jac(j: float) -> str:
    """A Jaccard that reads 1.000 only when the rule reproduces the release exactly (Gowalla's 0.9999 is not 1)."""
    return _no_false_bound(j, (lambda x, d: f'{x:.{d}f}', 3))


def build_card(benchmark: Dict, universe: Dict, task: str, mapping: Dict, exclusion: Dict,
               depth: Optional[Dict] = None, rules: Optional[Dict] = None, scope: Optional[Dict] = None,
               comparison: Optional[Dict] = None, hygiene: Optional[Dict] = None, notes: str = '',
               rule_check: Optional[Dict] = None, support: Optional[Dict] = None, counted: str = '',
               warnings: Optional[List[str]] = None, inputs: Optional[Dict] = None) -> Dict:
    """`rules` is a rule search over depth families; `rule_check` checks a stated rule that can only be
    evaluated on released units (query-level benchmarks with judgments). `support` says how the benchmark
    chooses its evaluation candidates: {'candidates': ..., 'structural': bool, 'basis': ...}. Only when the
    candidates are enumerated over the release's units (structural) is W an excluded share; otherwise the
    card reports coverage and says so.

    `counted` names what W is a share of, with its size ('the 112,199 WordNet 3.0 synsets that carry ...'), so
    the scope statement never reads as a share of the universe's name. `warnings` are problems the user must
    see (few ids mapped, skipped input lines, a likely misread file); each is repeated after the statement.
    `inputs` is the InputLog: lines read and skipped per input file."""
    from . import __version__
    support = support or {'candidates': 'not stated', 'structural': False,
                          'basis': 'The candidate mechanism was not declared, so W is reported as coverage.'}
    warnings = list(warnings or [])
    name, uname = benchmark.get('name', 'this benchmark'), universe.get('name', 'the declared universe')
    counted = counted or uname
    defined = lambda x: x is not None and x == x
    unit_word = {'kg_link_prediction': 'entities', 'recommendation': 'user-item pairs',
                 'pair_classification': 'candidate pairs', 'query_level': 'queries'}.get(task, 'units')
    if support.get('structural'):
        w = exclusion.get('W_pairs') if 'W_pairs' in exclusion else exclusion.get('W')
        statement = (f"Comparisons on {name} license claims about the {unit_word} it can evaluate, which leave out "
                     f"{pct(w)} of {counted}." if defined(w) else
                     f"The excluded share of {counted} is undefined for {name}: {exclusion.get('note', 'no candidates')}")
    else:
        parts = []
        if defined(exclusion.get('W')):
            parts.append(f"{name} uses {pct(1 - exclusion['W'])} of {counted}")
        if defined(exclusion.get('pair_coverage')):
            parts.append(f"{pct(exclusion['pair_coverage'])} of their pairs")
        statement = (' and '.join(parts) + '. ' if parts else '') + (
            'The release does not show how its pairs were drawn, so these are coverage figures: they do not show '
            'that the absent units or pairs could never enter the benchmark.')
    wq = exclusion.get('within_query')
    if wq:
        u = wq['unjudged_share']
        statement += (f" Within each evaluated query, {u['min']:.4%} to {u['max']:.4%} of the {wq['corpus_size']:,} "
                      f"documents are unjudged and count as non-relevant.")
    if rules:
        b = rules['best']
        statement += (f" The closest simple rule ({b['family']}, T = {b['T']}) is {b['verdict']} "
                      f"(Jaccard {jac(b['jaccard'])}).")
    if rule_check:
        statement += (f" The stated rule ({rule_check['rule']}) is {rule_check['verdict']}.")
    if warnings:
        statement += ' Caution: ' + ' '.join(warnings)
    card = {'schema': CARD_SCHEMA, 'tool_version': __version__, 'task': task, 'benchmark': benchmark,
            'universe': universe, 'mapping': mapping, 'exclusion': exclusion, 'depth_profile': depth,
            'rule_diagnosis': rules, 'rule_check': rule_check, 'derived_scope': scope, 'comparison': comparison,
            'hygiene': hygiene, 'support': support, 'scope_statement': statement, 'counted': counted,
            'warnings': warnings, 'inputs': inputs or {}, 'notes': notes}
    return _clean(card)


TASKS = ('kg_link_prediction', 'recommendation', 'pair_classification', 'query_level')
REQUIRED = ('schema', 'tool_version', 'task', 'benchmark', 'universe', 'mapping', 'exclusion', 'depth_profile',
            'rule_diagnosis', 'rule_check', 'derived_scope', 'comparison', 'hygiene', 'support', 'scope_statement',
            'counted', 'warnings', 'inputs', 'notes')
VERDICTS = ('reproduced', 'approximately reproduced', 'not reproduced')


def validate_card(card: Dict) -> List[str]:
    """Problems that make a card unfit for the library (empty list = valid). Checks the schema version,
    the required fields and their types, and that the numbers are internally consistent. It does not
    rerun the audit."""
    p: List[str] = []
    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
    share = lambda x: num(x) and 0 <= x <= 1
    missing = [k for k in REQUIRED if k not in card]
    if missing:
        return [f'missing fields: {", ".join(missing)}']
    if card['schema'] != CARD_SCHEMA:
        p.append(f"schema is {card['schema']!r}, expected {CARD_SCHEMA!r}")
    if card['task'] not in TASKS:
        p.append(f"unknown task {card['task']!r}")
    b, u, m, ex = card['benchmark'], card['universe'], card['mapping'], card['exclusion']
    if not (isinstance(b, dict) and isinstance(b.get('name'), str) and b['name']):
        p.append('benchmark.name missing')
    hashes = b.get('files_sha256') if isinstance(b, dict) else None
    if not (isinstance(hashes, dict) and hashes and all(isinstance(v, str) and len(v) == 64 for v in hashes.values())):
        p.append('benchmark.files_sha256 must map each release file to a SHA-256 hex digest')
    if not (isinstance(u, dict) and isinstance(u.get('name'), str) and isinstance(u.get('units'), int) and u['units'] > 0):
        p.append('universe needs a name and a positive unit count')
    if isinstance(m, dict) and all(isinstance(m.get(k), int) for k in ('release_ids', 'mapped_ids', 'unmapped_ids')):
        if m['mapped_ids'] + m['unmapped_ids'] > m['release_ids']:
            p.append('mapping: mapped + unmapped ids exceed release ids')
    else:
        p.append('mapping needs integer release_ids, mapped_ids and unmapped_ids')
    ws = [ex.get(k) for k in ('W', 'W_pairs') if isinstance(ex, dict) and ex.get(k) is not None]
    if not ws or not all(share(w) for w in ws):
        p.append('exclusion needs W or W_pairs, each in [0, 1] (W_pairs may be null when undefined)')
    for k in ('pair_coverage', 'W_pairs_release_convention'):
        if isinstance(ex, dict) and ex.get(k) is not None and not share(ex[k]):
            p.append(f'exclusion.{k} must be in [0, 1] or null')
    pq = ex.get('per_query') if isinstance(ex, dict) else None
    if pq is not None and not (isinstance(pq, dict) and isinstance(pq.get('queries'), int) and (
            pq['queries'] == 0 or (share(pq.get('W_bar')) and all(share(v) for v in (pq.get('W_q') or {}).values())))):
        p.append('exclusion.per_query needs a query count, and W_bar and W_q in [0, 1]')
    unj = ((ex.get('within_query') or {}).get('unjudged_share') if isinstance(ex, dict) else None) or {}
    if not all(share(v) for v in unj.values()):
        p.append('exclusion.within_query.unjudged_share must lie in [0, 1]')
    sp = card['support']
    if not (isinstance(sp, dict) and isinstance(sp.get('structural'), bool) and isinstance(sp.get('candidates'), str)):
        p.append('support needs candidates (text) and structural (true or false)')
    if isinstance(ex, dict) and all(isinstance(ex.get(k), int) for k in ('units', 'included', 'excluded')):
        if ex['included'] + ex['excluded'] != ex['units']:
            p.append('exclusion: included + excluded != units')
        elif share(ex.get('W')) and ex['units'] and abs(ex['W'] - ex['excluded'] / ex['units']) > 1e-6:
            p.append('exclusion: W does not equal excluded / units')
    rd = card['rule_diagnosis']
    if rd is not None:
        rows = (rd.get('families') or []) + ([rd['stated']] if rd.get('stated') else [])
        if not rd.get('families'):
            p.append('rule_diagnosis has no families')
        for r in rows:
            if not (isinstance(r.get('T'), int) and share(r.get('jaccard')) and r.get('verdict') in VERDICTS):
                p.append(f"rule_diagnosis row {r.get('family')!r} needs an integer T, a Jaccard in [0, 1] and a known verdict")
            elif verdict_of(r['jaccard']) != r['verdict']:
                p.append(f"rule_diagnosis row {r.get('family')!r}: verdict does not match its Jaccard")
    ds = card['derived_scope']
    if ds is not None and not (ds.get('d_min') is None or isinstance(ds.get('d_min'), int)):
        p.append('derived_scope.d_min must be an integer or null')
    cmp_ = card['comparison']
    if cmp_ is not None:
        if cmp_.get('metric') not in ('auc', 'mrr', 'hits'):
            p.append('comparison.metric must be auc, mrr or hits')
        d = cmp_.get('delta_O')
        if d is not None and not (num(d) and -1 <= d <= 1):                   # NaN and infinities fail too
            p.append('comparison.delta_O must be a finite difference in [-1, 1]')
        if cmp_.get('W') is not None and not share(cmp_['W']):
            p.append('comparison.W must be in [0, 1] or null')
        if cmp_.get('c_star') is not None and not (num(cmp_['c_star']) and cmp_['c_star'] >= 0):
            p.append('comparison.c_star must be a non-negative number or null')
        if cmp_.get('c_star') is not None and cmp_.get('metric') != 'auc':
            p.append('comparison: a break-even is defined only for AUC')
    if not (isinstance(card['scope_statement'], str) and card['scope_statement']):
        p.append('scope_statement is empty')
    if not (isinstance(card['counted'], str) and card['counted']):
        p.append('counted must name what W is a share of')
    if not (isinstance(card['warnings'], list) and all(isinstance(w, str) and w for w in card['warnings'])):
        p.append('warnings must be a list of non-empty strings')
    inp = card['inputs']
    if not (isinstance(inp, dict) and all(isinstance(e, dict) and isinstance(e.get('lines'), int)
                                          and isinstance(e.get('skipped'), int) and 0 <= e['skipped'] <= e['lines']
                                          for e in inp.values())):
        p.append('inputs must map each input file to its lines and skipped lines (0 <= skipped <= lines)')
    elif any(e['skipped'] for e in inp.values()) and not card['warnings']:
        p.append('inputs record skipped lines, but the card has no warning')
    return p


def verdict_of(j: float) -> str:
    """The rule verdict for a Jaccard (as rules.verdict; repeated here so cards validate without the search)."""
    return 'reproduced' if j == 1.0 else 'approximately reproduced' if j >= 0.95 else 'not reproduced'


def to_json(card: Dict) -> str:
    return json.dumps(card, indent=1, sort_keys=True, ensure_ascii=False) + '\n'


def to_markdown(card: Dict) -> str:
    b, u, ex = card['benchmark'], card['universe'], card['exclusion']
    lines = [f"# Scope card: {b.get('name')}", '', f"> {card['scope_statement']}", '']
    if card.get('warnings'):
        lines += ['## Warnings', ''] + [f'- {w}' for w in card['warnings']] + ['']
    lines += [f"- **Task:** {card['task']}", f"- **Universe:** {u.get('name')} {u.get('version', '')}".rstrip(),
              f"- **W is a share of:** {card.get('counted') or u.get('name')}"]
    if card.get('notes'):
        lines.append(f"- **Definitions:** {card['notes']}")
    lines.append(f"- **Tool:** benchscope-audit {card['tool_version']} (schema {card['schema']})")
    if u.get('fingerprint'):
        lines.append(f"- **Universe fingerprint:** `{u['fingerprint']}`")
    sp = card.get('support') or {}
    if sp:
        kind = 'excluded units cannot be evaluated' if sp.get('structural') else 'coverage only'
        lines.append(f"- **Candidates:** {sp.get('candidates')} ({kind}). {sp.get('basis', '')}".rstrip())
    lines.append('')
    m = card['mapping']
    txt = (f"{m.get('mapped_ids')} of {m.get('release_ids')} release ids mapped; {m.get('unmapped_ids')} unmapped; "
           f"{m.get('merged_ids', 0)} ids stand for more than one universe unit")
    if m.get('release_triples'):
        txt += f"; {m.get('unmatched_triples', 0)} of {m.get('release_triples')} release triples have no universe match"
    lines += ['## Mapping', '', txt + '.', '']
    structural = bool(sp.get('structural'))
    lines += ['## Exclusion' if structural else '## Coverage', '']
    f6 = lambda x: f'{x:.6f}' if x is not None else 'undefined'
    if 'W' in ex:
        lines.append(f"Units: {ex['excluded']} of {ex['units']} {'excluded' if structural else 'not in the release'}, "
                     f"**W = {ex['W']:.4f}**.")
    if 'W_pairs' in ex:
        where = (f"outside the retained rectangle ({ex['part0_retained']} of {ex['part0_units']} and {ex['part1_retained']} of "
                 f"{ex['part1_units']} units retained)" if 'part0_units' in ex else
                 f"with an end outside the retained units ({ex['units_retained']} of {ex['units']} retained)")
        lines.append(f"Unobserved pairs {where}: **W_pairs = {f6(ex['W_pairs'])}**.")
        if ex.get('note'):
            lines.append(ex['note'])
        if 'W_pairs_release_convention' in ex:
            lines.append(f"The 2026-10 pilot table counted the release's own positives instead: "
                         f"{f6(ex['W_pairs_release_convention'])}. {ex.get('release_convention_note', '')}")
    if 'pair_coverage' in ex:
        lines.append(f"Release pairs: {ex['release_pairs_in_universe']} of the universe's {ex['universe_pairs']} pairs "
                     f"(coverage {f6(ex['pair_coverage'])}).")
    for sname, d in (ex.get('by_strata') or {}).items():
        lines.append(f"By {sname}: " + ', '.join(f"{k} {v['W']:.4f}" for k, v in d.items()) + '.')
    pq = ex.get('per_query')
    if pq and pq.get('queries'):
        w = pq['W_q']
        lines.append(f"Per query, on the universe under the benchmark's protocol: mean W_q = **{pq['W_bar']:.4f}** over "
                     f"{pq['queries']:,} queries (from {w['min']:.4f} to {w['max']:.4f}, median {w['median']:.4f}); "
                     f"{pq['queries_without_excluded']:,} have no excluded candidate"
                     + (f", {pq['queries_without_kept']:,} keep no candidate of the release"
                        if pq.get('queries_without_kept') else '')
                     + (f"; {pq['queries_without_candidates']:,} more have no candidate at all and are left out"
                        if pq.get('queries_without_candidates') else '') + '.'
                     + (f" The release also drops {pct(pq['excluded_queries_share'])} of the universe's users, whose "
                        f"queries it never asks." if pq.get('excluded_queries_share') is not None else ''))
    if ex.get('within_query'):
        wq = ex['within_query']
        u, j = wq['unjudged_share'], wq['judged_per_query']
        lines.append(f"Within the {wq['evaluated_queries']} evaluated queries: {j['min']:.0f} to {j['max']:.0f} judged documents "
                     f"each (median {j['median']:g}); unjudged share of the {wq['corpus_size']:,} documents "
                     f"{u['min']:.6f} to {u['max']:.6f}. Judgments by grade: "
                     + ', '.join(f'{g}: {n}' for g, n in wq['by_grade'].items()) + '.')
    if card.get('rule_check'):
        c = card['rule_check']
        lines += ['', '## Stated rule check', '',
                  f"Rule: {c['rule']}" + (f" (stated in {c['source']})" if c.get('source') else '') +
                  f". Checked on {c['released_units_checked']} released units: {c['violations']} violations "
                  f"(observed minimum relevant {c['observed_min_relevant']}, maximum relevant/judged {c['observed_max_ratio']:.3f}). "
                  f"Verdict: {c['verdict']}."]
    if card.get('rule_diagnosis'):
        lines += ['', '## Rule diagnosis', '', '| family | best T | Jaccard | verdict |', '|---|---|---|---|']
        top = card['rule_diagnosis']['thresholds_searched'][1]
        for r in card['rule_diagnosis']['families']:
            edge = ' (top of the searched range; a higher T may fit better)' if r['T'] == top else ''
            lines.append(f"| {r['family']} | {r['T']}{edge} | {jac(r['jaccard'])} | {r['verdict']} |")
        if 'stated' in card['rule_diagnosis']:
            s = card['rule_diagnosis']['stated']
            lines.append(f"\nStated rule ({s['family']}, T = {s['T']}): Jaccard {jac(s['jaccard'])}, {s['verdict']}.")
    if card.get('derived_scope'):
        s = card['derived_scope']
        if s['d_min'] is not None:
            txt = f"d_min = {s['d_min']} (smallest depth threshold with W <= 0.5, {s['level']} level)."
        else:
            top = s['curve'][-1] if s['curve'] else None
            txt = (f"No depth threshold up to {top['d']} brings W to 0.5 or below ({s['level']} level; W = {top['W']:.6f} "
                   f"at d = {top['d']})." if top and top.get('W') is not None else 'No depth threshold brings W to 0.5 or below.')
        lines += ['', '## Derived scope', '', txt + ' This is a coverage threshold: it says where the benchmark covers at '
                                                'least half of the restricted universe, not that rankings hold there.']
    if card.get('hygiene'):
        h = card['hygiene']
        if 'reverse_leakage' in h:
            r = h['reverse_leakage']
            lines += ['', '## Hygiene', '', f"Reverse-edge leakage: {r['leaking']:,} of {r['test_triples']:,} test triples "
                                            f"({r['share']:.1%}) have their reverse (tail, relation, head) in training."]
            byrel = sorted(((k, v) for k, v in (r.get('by_relation') or {}).items() if v.get('leaking')),
                           key=lambda kv: -kv[1]['leaking'])
            if byrel:
                lines.append('By relation: ' + '; '.join(f"{k} {v['leaking']:,} of {v['test']:,} ({v['leaking'] / v['test']:.0%})"
                                                         for k, v in byrel) + '.')
        if 'unseen_test_units' in h:
            s = h['unseen_test_units']
            lines.append(f"Unseen in training: {plural(s['unseen_units'], 'test entity', 'test entities')} never occur in a "
                         f"training triple; they touch {plural(s['test_triples_affected'], 'test triple')} of "
                         f"{s['test_triples']:,}.")
        if 'pairs' in h:
            q = h['pairs']
            txt = (f"{plural(q['rows'], 'release row')}, {plural(q['distinct_pairs'], 'distinct pair')} "
                   f"({plural(q['duplicate_rows'], 'duplicate row')}, {plural(q['self_pairs'], 'self pair')}")
            if 'by_label' in q:
                txt += f"; labels {', '.join(f'{k}: {v}' for k, v in q['by_label'].items())}; " \
                       f"{plural(q['pairs_with_conflicting_labels'], 'pair')} with conflicting labels"
            txt += ').'
            if 'negatives_observed_as_interactions' in q:
                n, p = q['negatives_observed_as_interactions'], q['positives_not_observed']
                txt += (f" {n:,} of {q['negatives']:,} negatives {'is an observed interaction' if n == 1 else 'are observed interactions'} "
                        f"in the universe; {p:,} of {q['positives']:,} positives {'is' if p == 1 else 'are'} not.")
            if 'release_pairs_observed_as_interactions' in q:
                n = q['release_pairs_observed_as_interactions']
                txt += f" {plural(n, 'release pair')} {'is an observed interaction' if n == 1 else 'are observed interactions'} in the universe."
            lines += ['', '## Hygiene', '', txt]
        if 'negative_sampling' in h:
            lines += ['', '## Sampled negatives', '']
            order = {'train': 0, 'valid': 1, 'test': 2}          # JSON keys come back sorted; keep split order
            for s, d in sorted(h['negative_sampling'].items(), key=lambda kv: order.get(kv[0], 3)):
                lines.append(f"- **{s}:** {d['negatives_per_source']} per source for {d['sources']} sources. Later than the source: "
                             f"{d['later_than_source']}; same year: {d['same_year_as_source']}; cited by the source: "
                             f"{d['cited_by_source']}. Share in the lower-degree half of all nodes: "
                             f"{d['share_in_lower_degree_half']:.3f} (uniform sampling gives 0.5).")
    if card.get('comparison'):
        c = card['comparison']
        lines += ['', '## Comparison', '', c['statement']]
        if c.get('assumptions'):
            lines.append('Assumes ' + '; '.join(c['assumptions']) + '.')
    inp = card.get('inputs') or {}
    if inp:
        lines += ['', '## Input files', '']
        for fn, e in inp.items():
            txt = f"- **{fn}:** {plural(e['lines'], 'line')} read"
            if e['skipped']:
                fs = e['first_skipped']
                txt += (f"; {plural(e['skipped'], 'line')} skipped for not matching {e['expected']} "
                        f"(first at line {fs['line']}: `{fs['text']}`)")
            lines.append(txt + '.')
    return '\n'.join(lines) + '\n'
