"""Build the static card library (Layer 3) into tool/site/ from library/entries, library/cards and the card schema.

site/index.html              all benchmarks: excluded share, closest rule, pilot coding
site/how-to-read.html        what each part of a card means
site/cards/<slug>.html|json  one page per benchmark, with its card, its sources and how to reproduce it
site/schema/                 the card's JSON Schema
site/app/                    the browser auditor, copied from web/dist when it has been built

Every page is generated from the JSON files, never typed by hand, and the output has no timestamps,
so a rebuild from the same files gives the same bytes. Pages carry no author names (double-blind review).

usage: python library/build_site.py [output folder, default tool/site]
"""
import html
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(TOOL, 'src'))
from benchscope_audit import __version__  # noqa: E402
from benchscope_audit.card import CARD_SCHEMA, jac, pct, to_markdown, validate_card  # noqa: E402

SCHEMA_FILE = f"benchscope-card-{CARD_SCHEMA.split('/')[1]}.schema.json"

TASK_NAME = {'kg_link_prediction': 'KG link prediction', 'recommendation': 'recommendation',
             'query_level': 'query level', 'graph_link_prediction': 'graph link prediction',
             'pair_classification': 'pair classification'}
UNIT_NAME = {'kg_link_prediction': 'entities', 'query_level': 'queries'}


def esc(s):
    return html.escape(str(s), quote=True)


def linkify(s):
    """Escape, then make bare http(s) URLs clickable (pilot-table text cites its sources this way)."""
    out, last = [], 0
    for m in re.finditer(r'https?://[^\s;,)]+', s):
        url = m.group(0).rstrip('.')
        out.append(esc(s[last:m.start()]))
        out.append(f'<a href="{esc(url)}">{esc(url)}</a>')
        last = m.start() + len(url)
    out.append(esc(s[last:]))
    return ''.join(out)


def inline(s):
    s = esc(s)
    s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
    return re.sub(r'`([^`]+)`', r'<code>\1</code>', s)


def md_to_html(md):
    """The Markdown subset that card.to_markdown writes (the browser page renders it the same way)."""
    lines, out, i = md.split('\n'), [], 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        m = re.match(r'^(#{1,3}) (.*)$', line)
        if m:
            n = len(m.group(1))
            out.append(f'<h{n}>{inline(m.group(2))}</h{n}>')
            i += 1
        elif line.startswith('> '):
            out.append(f'<blockquote>{inline(line[2:])}</blockquote>')
            i += 1
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                rows.append([c.strip() for c in lines[i][1:-1].split('|')])
                i += 1
            head = ''.join(f'<th>{inline(c)}</th>' for c in rows[0])
            body = ''.join('<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in r) + '</tr>' for r in rows[2:])
            out.append(f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')
        elif line.startswith('- '):
            items = []
            while i < len(lines) and lines[i].startswith('- '):
                items.append(f'<li>{inline(lines[i][2:])}</li>')
                i += 1
            out.append('<ul>' + ''.join(items) + '</ul>')
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not re.match(r'^(#|> |\||- )', lines[i]):
                para.append(lines[i])
                i += 1
            out.append(f'<p>{inline(" ".join(para))}</p>')
    return '\n'.join(out)


def page(title, body, depth=0, description=''):
    up = '../' * depth
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(description)}">
<link rel="stylesheet" href="{up}style.css">
</head>
<body>
<main>
<nav class="top"><a href="{up}index.html">Card library</a><a href="{up}how-to-read.html">How to read a card</a><a href="{up}app/index.html">Audit your own benchmark</a><a href="{up}schema/{SCHEMA_FILE}">Card schema</a></nav>
{body}
<footer>
<p>Cards written by benchscope-audit {esc(__version__)} (schema {esc(CARD_SCHEMA)}). Every number on these pages is read from a card or from the 2026-10 pilot table.</p>
<p>Biology benchmarks are added once the papers that audit them are published.</p>
</footer>
</main>
</body>
</html>
'''


def w_cell(e, card):
    if card is None:
        return 'no card'
    ex = card['exclusion']
    if 'W_pairs' in ex:
        return f"{pct(ex['W_pairs'])} of unobserved user–item pairs"
    return f"{pct(ex['W'])} of {UNIT_NAME.get(card['task'], 'units')}"


def rule_cell(card):
    if card is None:
        return '—'
    rd, rc = card.get('rule_diagnosis'), card.get('rule_check')
    if rd:
        b = rd['best']
        cls = {'reproduced': 'good', 'approximately reproduced': 'mid'}.get(b['verdict'], 'bad')
        return f'{esc(b["family"].replace("_", " "))}, T = {b["T"]} <span class="tag {cls}">{esc(b["verdict"])}</span> <span class="muted">Jaccard {jac(b["jaccard"])}</span>'
    if rc:
        cls = 'good' if rc['violations'] == 0 else 'bad'
        return f'stated topic rule <span class="tag {cls}">{"holds on released topics" if rc["violations"] == 0 else "violated"}</span>'
    return 'no rule searched'


def index_page(entries, cards):
    rows = []
    for e in entries:
        c = cards.get(e['slug'])
        dec = e['pilot']['decidability']
        cls = {'verified': 'good', 'decidable, unverified': 'mid', 'partial': 'mid'}.get(dec, 'bad')
        rows.append(f'<tr><td><a href="cards/{e["slug"]}.html">{esc(e["name"])}</a></td><td>{esc(TASK_NAME.get(e["task"], e["task"]))}</td>'
                    f'<td class="num">{esc(w_cell(e, c))}</td><td>{rule_cell(c)}</td><td><span class="tag {cls}">{esc(dec)}</span></td></tr>')
    body = f'''<h1>BenchScope card library</h1>
<p class="lede">What each benchmark can never evaluate, measured against the universe it claims to cover. A card is computed from the public release and a public universe, without labels or models. The last column is how far the paper's own description lets a reader decide the eligibility rule (2026-10 pilot).</p>
<div class="panel tablewrap"><table>
<thead><tr><th>Benchmark</th><th>Task</th><th>Excluded share (W)</th><th>Closest simple rule</th><th>Rule decidable from the paper</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody></table></div>
<p class="lede">{sum(1 for e in entries if cards.get(e["slug"]))} of {len(entries)} benchmarks have a card; each page without one says what is missing. To add a benchmark, run the auditor on its release and open a pull request with the card (see the card schema).</p>'''
    return page('BenchScope card library', body, 0, 'Scope cards for benchmark releases: what each can never evaluate.')


def card_page(e, card):
    p = e['pilot']
    facts = [('Task', esc(TASK_NAME.get(e['task'], e['task']))), ('Domain', esc(p['domain'])),
             ('Rule stated in the paper', linkify(p['rule_stated_in_paper'])),
             ('What the release shows', linkify(p['rule_found_in_code_or_release'])),
             ('Rule decidable from the paper', f"{esc(p['decidability'])}. {linkify(p['verification'])}"),
             ('Negatives', linkify(p['negative_source'])), ('Evaluation', linkify(p['evaluation_protocol'])),
             ('Sources checked', linkify(p['sources_checked']))]
    if p['notes']:
        facts.append(('Notes', linkify(p['notes'])))
    dl = ''.join(f'<dt>{k}</dt><dd>{v}</dd>' for k, v in facts)
    parts = [f'<h1>{esc(e["name"])}</h1>']
    if card:
        parts.append(f'<p class="statement">{esc(card["scope_statement"])}</p>')
    else:
        parts.append(f'<p class="note"><strong>No card yet.</strong> {esc(e["why_no_card"])}</p>')
    if e.get('note'):
        parts.append(f'<p class="note">{esc(e["note"])}</p>')
    parts.append(f'<h2>From the 2026-10 pilot</h2><div class="panel"><dl class="facts">{dl}</dl></div>')
    if card:
        parts.append(f'<h2>Scope card</h2><div class="panel card">{md_to_html(to_markdown(card))}</div>')
        rows = ''.join(f'<tr><td><code>{esc(i["display"])}</code></td><td><a href="{esc(i["source"])}">{esc(i["source"])}</a></td></tr>'
                       for i in e['inputs'])
        cmd = ' '.join(f'"{a}"' if re.search(r'\s', a) else a for a in e['command'])
        parts.append(f'''<h2>Reproduce this card</h2><div class="panel">
<div class="tablewrap"><table><thead><tr><th>Input</th><th>Public source</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>With the benchscope-audit package installed:</p>
<pre>{esc(cmd)}</pre>
<p>The card records the SHA-256 of every release file it read, so a reader can check that the same files were audited. <a href="{esc(e["slug"])}.json">Download this card (JSON)</a>.</p></div>''')
    return page(f'{e["name"]}: BenchScope card', '\n'.join(parts), 1, f'Scope card for {e["name"]}.')


HOW_TO_READ = '''<h1>How to read a scope card</h1>
<p class="lede">A card compares a benchmark release with a declared universe: the population the benchmark is meant to stand for. It reports what the release can never evaluate. It needs no labels and no models, so it cannot say how models would score on the excluded part; it says how large that difference would have to be to matter.</p>
<div class="panel">
<h2>Scope statement</h2>
<p>One sentence: the share the benchmark leaves out, of a population the sentence names with its size (for WN18RR, the 112,199 WordNet 3.0 synsets that carry at least one triple of its 11 relations), and whether a simple rule explains the release. The card's <strong>W is a share of</strong> line repeats that population, and <strong>Definitions</strong> states how the universe and depth were built. Claims made on the benchmark are licensed for the part it covers.</p>
<h2>Warnings and input files</h2>
<p>A card lists, under <strong>Warnings</strong> and again after the scope statement, anything a reader must check before using it: only part of the release's ids occur in the universe (between 50% and 95%; below 50% the tool refuses to write a card, because the files do not look like a release of that universe), lines of a raw file that did not have the expected format and were skipped, or a pair file with a label column that was not used. <strong>Input files</strong> gives the lines read and skipped per file read leniently (raw universes, qrels, pair tables). Release files have a fixed format: a line that does not fit stops the audit with the file name and line number.</p>
<h2>Mapping</h2>
<p>How release ids land on universe units. <strong>Unmapped</strong> ids are absent from the universe. <strong>Merged</strong> ids stand for more than one unit: WN18RR keys synsets by bare WordNet offsets, which synsets of different parts of speech share, so 161 of its ids each stand for more than one synset. Unmapped release units count against every rule in the rule diagnosis.</p>
<h2>Candidates: exclusion or coverage</h2>
<p>A unit missing from a release is excluded only if the benchmark could never evaluate it. Each card says how the benchmark chooses its evaluation candidates. When candidates are enumerated over the release's units (every entity in link prediction, every item in all-item ranking, every judged topic), a unit absent from the release is never evaluated, and W is an <strong>excluded share</strong>. When the release only lists pairs drawn by an unstated rule (pair classification), an absent unit or pair may still have had a chance of being drawn; the card then reports <strong>coverage</strong> and says so, and gives no break-even.</p>
<h2>Exclusion (W)</h2>
<p><strong>W</strong> is the excluded share of the universe: units for knowledge graphs and query-level benchmarks, unobserved user–item pairs for recommendation. For query-level benchmarks with relevance judgments, the card also gives, inside each evaluated query, the share of the corpus that was never judged and is therefore scored as non-relevant.</p>
<h2>Depth profile and derived scope</h2>
<p>Depth is how much source evidence a unit has in the universe, for example the number of facts on an entity. The profile shows whether exclusion follows depth (a rarity cut) or not. The <strong>derived scope</strong> d_min is the smallest depth threshold whose restricted universe has W at or below 0.5. When no threshold gets there, the card says so.</p>
<h2>Rule diagnosis</h2>
<p>The rule is never an input. The tool searches simple rule families on depth (single threshold, both ends above a threshold, seeds plus their neighbours, iterated k-core) and reports the best threshold of each with its Jaccard overlap with the release. <span class="tag good">reproduced</span> means Jaccard 1, <span class="tag mid">approximately reproduced</span> at least 0.95, <span class="tag bad">not reproduced</span> below. A stated rule from the paper is checked the same way. A best threshold at the top of the searched range is marked, because a higher one may fit better.</p>
<h2>Stated rule check (query level)</h2>
<p>Some rules can be checked only on the released units, because excluded units have no published data (TREC topics that were never judged). The card then says whether every released unit satisfies the rule and that the excluded side cannot be checked.</p>
<h2>Comparison (break-even)</h2>
<p>For a difference Δ<sub>O</sub> between two models on the benchmark, the difference on the universe is (1 − W) Δ<sub>O</sub> + W Δ<sub>U</sub>, where Δ<sub>U</sub> is their difference on the excluded part. The ordering reverses exactly when Δ<sub>U</sub> has the opposite sign and exceeds (1 − W) |Δ<sub>O</sub>| / W, or, equivalently, when the change between the regions, c = Δ<sub>O</sub> − Δ<sub>U</sub>, has the sign of Δ<sub>O</sub> and its magnitude exceeds c* = |Δ<sub>O</sub>| / W. The card gives both thresholds. They hold for metrics that are linear in the candidate mix, such as per-query AUC with a fixed positive reference. With per-query shares, W is their mean, and c* is exact for a Δ<sub>O</sub> measured on the same test queries under the same filter; for a difference from another query set or filter, or a single W where the shares vary between queries, the thresholds carry no guarantee. MRR and Hits@k are not linear in the mix, so the card gives no threshold for them. Δ<sub>U</sub> and c need labels on the excluded part and are not computed.</p>
<h2>Hygiene</h2>
<p>Checks that do not depend on the universe: for knowledge graphs, test triples whose reverse is a training triple (answered by one training edge through a symmetric relation), broken down by relation, and test entities unseen in training; for sampled negatives, whether any negative is later than the query or cited by it, and whether sampling is uniform over nodes.</p>
</div>'''


def main(site=None):
    site = site or os.path.join(TOOL, 'site')
    if os.path.isdir(site):
        shutil.rmtree(site)
    os.makedirs(os.path.join(site, 'cards'))
    os.makedirs(os.path.join(site, 'schema'))
    reg = json.load(open(os.path.join(HERE, 'benchmarks.json'), encoding='utf-8'))['benchmarks']
    entries, cards = [], {}
    for b in reg:
        e = json.load(open(os.path.join(HERE, 'entries', b['slug'] + '.json'), encoding='utf-8'))
        entries.append(e)
        if e['card']:
            src = os.path.join(HERE, 'cards', e['card'])
            card = json.load(open(src, encoding='utf-8'))
            problems = validate_card(card)
            if problems:
                raise ValueError(f"{e['slug']}: {problems}")
            cards[e['slug']] = card
            shutil.copyfile(src, os.path.join(site, 'cards', e['card']))
    write = lambda rel, text: open(os.path.join(site, rel), 'w', encoding='utf-8', newline='\n').write(text)
    write('index.html', index_page(entries, cards))
    write('how-to-read.html', page('How to read a scope card', HOW_TO_READ, 0, 'What each part of a BenchScope card means.'))
    for e in entries:
        write(f'cards/{e["slug"]}.html', card_page(e, cards.get(e['slug'])))
    shutil.copyfile(os.path.join(HERE, 'assets', 'style.css'), os.path.join(site, 'style.css'))
    shutil.copyfile(os.path.join(HERE, 'schema', SCHEMA_FILE), os.path.join(site, 'schema', SCHEMA_FILE))
    app = os.path.join(TOOL, 'web', 'dist')
    if os.path.exists(os.path.join(app, 'index.html')):
        shutil.copytree(app, os.path.join(site, 'app'))
    n = sum(len(fs) for _, _, fs in os.walk(site))
    print(f'site: {len(entries)} benchmarks, {len(cards)} cards, {n} files' + ('' if os.path.isdir(os.path.join(site, 'app')) else
                                                                                  ' (no app/: run web/build.py first)'))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
