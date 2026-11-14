"""Figures 2-4, Tables 1-2 and Box 1 of the paper (OUTLINE.md), built only from stored results.

Sources, and nothing else:
  results/wn18rr_analysis/records.tsv   measured c; every value looked up by its full key
                                        (design, estimand, subject, seed, quantity, cluster)
  ../tool/library/cards/<slug>.json     the tool's scope cards (W, depth profile, rule diagnosis)
  ../tool/library/benchmarks.json       the card library's registry (why a benchmark has no card)
  pilot_decidability.tsv                the pilot coding (domain, decidability)

Nothing here computes a statistic. A value is plotted or printed as stored, so a figure cannot pair a
number with the wrong seed, estimand or clustering without the lookup failing. Intervals are the 95%
triple-bootstrap intervals in the records; seeds are plotted as separate points, never pooled.
Figure 1 is a schematic and is drawn separately. Sizes follow ICML's two-column page: Figures 2 and 4 span the
text width (6.75 in), Figure 3 one column (3.25 in). Table 1's LaTeX is a compact full-width table without the
construction findings, which stay in its Markdown version. Figure text is typeset by LaTeX (text.usetex), so the
PDFs embed only Type 1 fonts, as ICML asks. This needs a LaTeX installation with the helvet, type1cm and
underscore packages. The PNGs are rendered from the PDFs with PyMuPDF, and skipped if it is not installed.

Colours: one categorical pair, blue = the benchmark side (O, Delta_O), orange = the excluded or target side
(U, Delta_target), from the dataviz reference palette; validated with validate_palette.js --mode light
(all checks pass: CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on white). Grey marks carry no identity.

usage: python -E -P make_figures.py [OUT=figures]   (run from experiments/ or anywhere)
"""
import csv
import json
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

P4 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
TOOL = os.path.normpath(os.path.join(P4, '..', 'tool'))
RECORDS = os.path.join(P4, 'results', 'wn18rr_analysis', 'records.tsv')
CARDS = os.path.join(TOOL, 'library', 'cards')
REGISTRY = os.path.join(TOOL, 'library', 'benchmarks.json')
PILOT = os.path.join(P4, 'pilot_decidability.tsv')

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
COLUMN, FULL = 3.25, 6.75                      # ICML column and text width, inches (icml2026.sty)

MODELS = ('popularity', 'transe', 'conve', 'distmult', 'rescal', 'complex')
NAME = {'popularity': 'Popularity', 'transe': 'TransE', 'conve': 'ConvE', 'distmult': 'DistMult',
        'rescal': 'RESCAL', 'complex': 'ComplEx', 'nbfnet': 'NBFNet'}

# The ten benchmarks in the pilot's order. Unit and finding texts are descriptions (no numbers); the
# finding status is PRIOR_WORK.md's (summary table; section in brackets).
BENCH = (
    ('fb15k-237', 'entity', None, 'entities with triples on the kept relations removed, unstated', 'new [2]'),
    ('wn18rr', 'entity', 'W', "stated rule does not reproduce the entity set; ids merge synsets",
     'new / partly known [1]'),
    ('yago3-10', 'entity', 'W', "every description misstates the rule (facts, not relation types)", 'new [3]'),
    ('ogbl-collab', 'author pair', None, 'author subset rule not stated anywhere checked', '—'),
    ('ogbl-citation2', '2019 source paper', 'W', 'a source paper needs a minimum of in-subset references, '
     'unstated', 'new [4]'),
    ('gowalla', 'user–item pair', 'W_pairs', '10-core approximately reproduced', '—'),
    ('yelp2018', 'user–item pair', None, 'raw data not public; release-internal checks only', '—'),
    ('amazon-book', 'user–item pair', 'W_pairs', 'release is a strict subset of the stated 10-core; the extra '
     'user reduction is unexplained', 'partly known [5]'),
    ('trec-dl-2019-passage', 'topic', 'W', 'topic selection by judged-relevant counts and density', 'known [6]'),
    ('trec-dl-2020-passage', 'topic', 'W', 'the 2019 selection, reused by reference', 'known [6]'),
)
NO_CARD_SHORT = {'fb15k-237': 'universe (Freebase 2013) no longer distributed',
                 'ogbl-collab': 'universe needs an external author graph',
                 'yelp2018': 'raw data requires accepting Yelp’s terms'}

# ICML accepts Type 1 fonts only, so all figure text is typeset by LaTeX: Helvetica (URW Nimbus Sans, psnfss helvet)
# for text, Computer Modern for mathematics. Serif and monospace also name Helvetica only so that Matplotlib's preamble
# does not ask for cm-super; no figure text uses them. Tick labels arrive as $\mathdefault{...}$, which the preamble
# sets in the text font, so the numbers stay in Helvetica. The minus sign is Helvetica's (\textminus), never the math
# minus: Matplotlib 3.11.0 drops glyph 0 of every embedded Type 1 font (_type1font.subset sets encoding[0] to
# .notdef), and glyph 0 of Computer Modern's symbol font is the minus.
plt.rcParams.update({
    'text.usetex': True, 'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica'],
    'font.serif': ['Helvetica'], 'font.monospace': ['Helvetica'],
    'text.latex.preamble': r'\renewcommand{\mathdefault}[1]{\mbox{#1}}\DeclareUnicodeCharacter{2212}{\textminus}',
    'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5, 'legend.fontsize': 6.5, 'axes.linewidth': 0.6, 'axes.edgecolor': AXIS,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False,
})


class Records:
    def __init__(self, path):
        self.v = {}
        with open(path, newline='', encoding='utf-8') as fh:
            for r in csv.DictReader(fh, delimiter='\t'):
                f = lambda x: float(x) if x not in ('', 'nan') else None  # noqa: E731
                self.v[(r['design'], r['estimand'], r['subject'], r['seed'], r['quantity'], r['cluster'])] = \
                    (f(r['value']), f(r['lo']), f(r['hi']))

    def has(self, design, estimand, subject, seed, quantity, cluster='triple'):
        return (design, estimand, subject, str(seed), quantity, cluster) in self.v

    def __call__(self, design, estimand, subject, seed, quantity, cluster='triple'):
        return self.v[(design, estimand, subject, str(seed), quantity, cluster)][0]

    def interval(self, design, estimand, subject, seed, quantity, cluster='triple'):
        val = self.v[(design, estimand, subject, str(seed), quantity, cluster)]
        if None in val:
            raise KeyError(f'no interval stored for {design}/{estimand}/{subject}/{seed}/{quantity}/{cluster}')
        return val

    def seeds(self, design, estimand, subject, quantity):
        return sorted(int(k[3]) for k in self.v if k[:3] == (design, estimand, subject) and k[4] == quantity
                      and k[5] == 'triple' and k[3] != 'mean')


def card(slug):
    with open(os.path.join(CARDS, slug + '.json'), encoding='utf-8') as fh:
        return json.load(fh)


def pilot_rows():
    with open(PILOT, newline='', encoding='utf-8') as fh:
        return {r['benchmark']: r for r in csv.DictReader(fh, delimiter='\t')}


def registry():
    with open(REGISTRY, encoding='utf-8') as fh:
        return {b['slug']: b for b in json.load(fh)['benchmarks']}


def save(fig, out, name):
    """The PDF from Matplotlib; the PNG is rendered from that PDF with PyMuPDF, since usetex PNGs would need
    dvipng. Without PyMuPDF the PNG is skipped: the paper uses the PDFs, and reproduce.py compares neither."""
    os.makedirs(out, exist_ok=True)
    pdf = os.path.join(out, f'{name}.pdf')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    try:
        import pymupdf
    except ImportError:
        print(f'{name}.png skipped: PyMuPDF is not installed')
        return
    with pymupdf.open(pdf) as doc:
        doc[0].get_pixmap(dpi=600, alpha=False).save(os.path.join(out, f'{name}.png'))


def panel(ax, letter, dx=8):
    """Panel letter at the top left, dx points left of the axes (past the tick labels)."""
    ax.annotate(rf'\textbf{{{letter}}}', xy=(0, 1), xycoords='axes fraction', xytext=(-dx, 5),
                textcoords='offset points', fontsize=9, va='bottom', ha='left')   # usetex ignores fontweight


def hairline_x(ax):
    ax.grid(axis='x', color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def zero_line(ax, vertical=True):
    (ax.axvline if vertical else ax.axhline)(0, color=AXIS, lw=0.8, zorder=1)


def pair_name(p):
    a, b = p.split('-')
    return f'{NAME[a]} − {NAME[b]}'


def top_legend(fig, handles, y=1.0, ncol=None):
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, y), ncol=ncol or len(handles), frameon=False,
               handletextpad=0.3, columnspacing=1.4, borderaxespad=0.0)


def h(marker, label, **kw):
    return plt.Line2D([], [], marker=marker, ls='', label=label, **kw)


# ---------------------------------------------------------------- Figure 2: excluded share by depth
DEPTH_UNIT = {'wn18rr': 'WordNet triples per synset', 'yago3-10': 'YAGO facts per entity',
              'ogbl-citation2': 'references inside 2019, per paper', 'gowalla': 'check-in pairs per user or location',
              'amazon-book': 'ratings per user or item'}


def fig2(out):
    """Single column, 3 x 2 small multiples on one y scale. Each benchmark keeps its own depth unit, so the
    profiles are not overlaid (2026-10-28; W per benchmark and the rule diagnosis moved to Appendix Figure B1)."""
    reg = registry()
    fig, axes = plt.subplots(3, 2, figsize=(COLUMN - 0.2, 3.4), sharey=True,
                             gridspec_kw={'hspace': 1.25, 'wspace': 0.12})
    for ax, slug in zip(axes.flat, ('wn18rr', 'yago3-10', 'ogbl-citation2', 'gowalla', 'amazon-book')):
        bins = card(slug)['depth_profile']['by_depth']
        lab = [str(b['depth_from']) if b['depth_to'] == b['depth_from'] else
               (f"{b['depth_from']}+" if b['depth_to'] is None else f"{b['depth_from']}–{b['depth_to']}")
               for b in bins]
        ax.bar(range(len(bins)), [b['excluded_share'] for b in bins], width=0.75, color=ORANGE, lw=0)
        ax.set_xticks(range(len(bins)), lab, rotation=90, fontsize=5.5)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_title(reg[slug]['name'], fontsize=6.5, pad=3)
        ax.set_xlabel(DEPTH_UNIT[slug], fontsize=5.8, color=INK2, labelpad=2)
        ax.grid(axis='y', color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', length=0)
    for ax in axes[:, 0]:
        ax.set_ylabel('excluded share', fontsize=6.5)
    last = axes.flat[-1]
    last.axis('off')
    last.text(0.04, 0.55, 'Share of the units in each\ndepth bin that the release\nexcludes. A threshold rule\n'
              'is a step; the units of\neach panel differ.', transform=last.transAxes, fontsize=6, color=INK2,
              va='center', ha='left')
    save(fig, out, 'fig2_audit')


# ---------------------------------------------------------------- Appendix Figure B1: W and rule diagnosis
def fig_app_ten(out):
    reg = registry()
    fig = plt.figure(figsize=(FULL - 1.3, 2.5))   # the tight bbox adds the row labels
    top = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.62)

    # (a) W at the evaluated unit
    ax = fig.add_subplot(top[0])
    labels = []
    for i, (slug, unit, field, *_) in enumerate(BENCH[::-1]):
        labels.append(rf"{reg[slug]['name']}  $\cdot$ {unit}")
        if field is None:
            ax.text(0.01, i, f'no card: {NO_CARD_SHORT[slug]}', va='center', fontsize=5.8, color=MUTED)
            continue
        w = card(slug)['exclusion'][field]
        ax.barh(i, w, height=0.55, color=ORANGE, lw=0)
        ax.text(w + 0.015, i, f'{w:.4f}', va='center', fontsize=6, color=INK2)
    ax.set_yticks(range(len(labels)), labels, fontsize=6)
    ax.set_xlim(0, 1.13)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel('W, excluded share at the evaluated unit')
    ax.tick_params(axis='y', length=0)
    hairline_x(ax)
    panel(ax, 'a', dx=118)

    # (b) rule diagnosis: stated rule vs best rule found
    ax = fig.add_subplot(top[1])
    rows = [(s, card(s)['rule_diagnosis']) for s in ('wn18rr', 'yago3-10', 'ogbl-citation2', 'gowalla', 'amazon-book')]
    for i, (slug, rd) in enumerate(rows[::-1]):
        best = rd['best']
        if 'stated' in rd:
            st = rd['stated']
            ax.plot([st['jaccard'], best['jaccard']], [i, i], color=AXIS, lw=0.8, zorder=2)
            ax.plot([st['jaccard']], [i], 'o', ms=7.5, mfc='none', mec=BLUE, mew=0.9, zorder=4)
        ax.plot([best['jaccard']], [i], 'o', ms=4.2, color=BLUE, zorder=3)
        ax.text(1.04, i, best['verdict'].replace('approximately', 'approx.'), va='center', fontsize=5.8,
                color=INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(rows)), [reg[s]['name'] for s, _ in rows[::-1]], fontsize=6)
    ax.set_xlim(0, 1.04)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel('Jaccard with the release')
    ax.tick_params(axis='y', length=0)
    ax.legend(handles=[h('o', 'stated rule', ms=7.5, mfc='none', mec=BLUE, mew=0.9),
                       h('o', 'best rule found', ms=4.2, color=BLUE)],
              loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, handletextpad=0.2,
              columnspacing=0.8, borderaxespad=0.3)
    hairline_x(ax)
    panel(ax, 'b', dx=62)
    save(fig, out, 'figB1_benchmarks')


# ---------------------------------------------------------------- Figure 3: WN18RR, per model
def fig3(R, out):
    rows = [('E2', m) for m in MODELS] + [('deploy', 'nbfnet')]
    y = {r: len(rows) - 1 - i + (0 if r[0] == 'E2' else -0.4) for i, r in enumerate(rows)}
    fig, axes = plt.subplots(1, 3, figsize=(COLUMN - 0.2, 2.6), sharey=True, gridspec_kw={'wspace': 0.14})
    spec = (('auc_U_minus_O', 'AUC(U) − AUC(O)', True), ('mrr_C_minus_O', 'MRR(C) − MRR(O)', True),
            ('intrusion', 'outranked by U', False))
    for ax, (q, label, zero) in zip(axes, spec):
        for design, m in rows:
            yy = y[(design, m)]
            mk = 'o' if design == 'E2' else 'D'
            for s in R.seeds(design, 'primary', m, q):
                ax.plot([R(design, 'primary', m, s, q)], [yy + 0.18], '.', ms=2.6, color=MUTED, zorder=2)
            v, lo, hi = R.interval(design, 'primary', m, 'mean', q)
            ax.plot([lo, hi], [yy, yy], color=BLUE, lw=1.0, zorder=3)
            ax.plot([v], [yy], mk, ms=4.2, color=BLUE, zorder=4)
            if q == 'auc_U_minus_O':
                v2, lo2, hi2 = R.interval(design, 'S2', m, 'mean', q)
                ax.plot([lo2, hi2], [yy - 0.22, yy - 0.22], color=BLUE, lw=0.7, zorder=3)
                ax.plot([v2], [yy - 0.22], mk, ms=3.8, mfc='white', mec=BLUE, mew=0.8, zorder=4)
        if zero:
            zero_line(ax)
        ax.axhline(y[('deploy', 'nbfnet')] + 0.7, color=GRID, lw=0.6)
        ax.set_xlabel(label, fontsize=6.3)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.tick_params(axis='x', labelsize=5.8)
        hairline_x(ax)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks([y[r] for r in rows],
                       [NAME[m] if d == 'E2' else 'NBFNet\n(deployed)' for d, m in rows], fontsize=6.2)
    for ax, letter, dx in zip(axes, 'abc', (52, 6, 6)):
        panel(ax, letter, dx=dx)
    top_legend(fig, [h('o', r'primary: seed mean, 95\% CI', color=BLUE, ms=4),
                     h('o', 'S2, degree-matched U (a)', mfc='white', mec=BLUE, mew=0.8, ms=3.8),
                     h('.', 'single seeds', color=MUTED, ms=3.5)], y=0.99, ncol=2)
    save(fig, out, 'fig3_wn18rr_models')


# ---------------------------------------------------------------- Figure 4: the one reversal, and metrics
def fig4(R, out):
    pair = 'conve-distmult'
    rows = (('primary', 'triple', 'primary'), ('S1', 'triple', 'S1: POS-compatible'),
            ('S2', 'triple', 'S2: degree-matched U'), ('S3', 'triple', 'S3: benchmark identity'),
            ('S4', 'triple', 'S4: trained O only'), ('primary', 'head', 'primary, head clustering'),
            ('posthoc_noleak', 'triple', 'non-leaking queries\n(post hoc)'))
    fig = plt.figure(figsize=(FULL - 0.07, 3.8))     # 4.1 -> 3.8 in on 2026-10-28 for the page limit
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.62)

    def two(ax, yy, est, seed, cl, offs=0.14):
        for q, col, dy in (('delta_O', BLUE, offs), ('delta_target', ORANGE, -offs)):
            v, lo, hi = R.interval('E2', est, pair, seed, q, cl)
            ax.plot([lo, hi], [yy + dy, yy + dy], color=col, lw=1.0)
            ax.plot([v], [yy + dy], 'o', ms=4, color=col, mec='white', mew=0.6, zorder=3)

    ax = fig.add_subplot(gs[0, 0])
    for i, (est, cl, _) in enumerate(rows):
        two(ax, len(rows) - 1 - i, est, 'mean', cl)
    ax.set_yticks(range(len(rows)), [r[2] for r in rows][::-1])
    zero_line(ax)
    ax.set_xlabel(r'ConvE − DistMult, $\Delta$ in per-query AUC')
    hairline_x(ax)
    ax.tick_params(axis='y', length=0)
    panel(ax, 'a', dx=100)

    ax = fig.add_subplot(gs[0, 1])
    seeds = [str(s) for s in R.seeds('E2', 'primary', pair, 'delta_O')] + ['mean']
    for i, s in enumerate(seeds):
        two(ax, len(seeds) - 1 - i, 'primary', s, 'triple')
    ax.set_yticks(range(len(seeds)), [f'seed {s}' if s != 'mean' else 'seed mean' for s in seeds][::-1])
    zero_line(ax)
    ax.set_xlabel('ConvE − DistMult, primary estimand')
    hairline_x(ax)
    ax.tick_params(axis='y', length=0)
    panel(ax, 'b', dx=48)

    ax = fig.add_subplot(gs[1, 0])
    for i, (est, cl, _) in enumerate(rows):
        yy = len(rows) - 1 - i
        assert R('E2', est, pair, 'mean', 'delta_O', cl) < 0, 'c * sign(delta_O) = delta_U - delta_O needs delta_O < 0'
        v, lo, hi = R.interval('E2', est, pair, 'mean', 'delta_U_minus_O', cl)
        cs = R('E2', est, pair, 'mean', 'c_star', cl)
        ax.plot([lo, hi], [yy, yy], color=INK2, lw=1.0)
        ax.plot([v], [yy], 'o', ms=4, color=INK2, zorder=3)
        ax.plot([cs], [yy], '|', ms=8, mew=1.4, color=ORANGE, zorder=4)
    ax.set_yticks(range(len(rows)), [r[2] for r in rows][::-1])
    ax.set_xlim(left=0)
    ax.set_xlabel(r'$\bar{c}\cdot\mathrm{sign}(\Delta_O) = \Delta_U$ − $\Delta_O$; reversal right of $c^*$')
    ax.legend(handles=[h('o', r'$\bar{c}\cdot\mathrm{sign}(\Delta_O)$, 95\% CI', color=INK2, ms=4),
                       h('|', r'$c^* = |\Delta_O|\,/\,\bar{W}$', color=ORANGE, ms=8, mew=1.4)],
              loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, handletextpad=0.2,
              columnspacing=1.0, borderaxespad=0.3)
    hairline_x(ax)
    ax.tick_params(axis='y', length=0)
    panel(ax, 'c', dx=100)

    ax = fig.add_subplot(gs[1, 1])
    pairs = sorted({k[2] for k in R.v if k[0] == 'E2' and k[1] == 'primary' and '-' in k[2]})
    at = lambda p, q: R('E2', 'primary', p, 'mean', q)  # noqa: E731
    dis = [p for p in pairs if (at(p, 'delta_O') > 0) != (at(p, 'delta_mrr_O') > 0)]
    for p in pairs:
        ax.plot([at(p, 'delta_O')], [at(p, 'delta_mrr_O')], 'o', ms=4, color=ORANGE if p in dis else MUTED,
                mec='white', mew=0.6, zorder=3)
    # labels: the TransE pairs form one cluster, the two RESCAL pairs sit near the origin, and the grey pairs
    # all involve popularity
    transe = [p for p in dis if p.startswith('transe-')]
    near = sorted((p for p in dis if p not in transe), key=lambda p: -at(p, 'delta_mrr_O'))   # upper point, upper label
    ax.annotate('TransE − each other\nlearned model', (min(at(p, 'delta_O') for p in transe),
                sum(at(p, 'delta_mrr_O') for p in transe) / len(transe)), xytext=(-64, -20),
                textcoords='offset points', fontsize=5.6, color=INK2,
                arrowprops=dict(arrowstyle='-', color=AXIS, lw=0.6))
    for p, dy in zip(near, (-16, -30)):
        ax.annotate(pair_name(p), (at(p, 'delta_O'), at(p, 'delta_mrr_O')), xytext=(-86, dy),
                    textcoords='offset points', fontsize=5.6, color=INK2, va='center',
                    arrowprops=dict(arrowstyle='-', color=AXIS, lw=0.6))
    pop = [p for p in pairs if p.startswith('popularity-')]
    lowest = min(pop, key=lambda p: at(p, 'delta_mrr_O'))
    ax.annotate('popularity − each\nlearned model (agree)', (at(lowest, 'delta_O'), at(lowest, 'delta_mrr_O')),
                xytext=(16, 4), textcoords='offset points', fontsize=5.6, color=MUTED, va='center')
    zero_line(ax)
    zero_line(ax, vertical=False)
    ax.set_xlabel(r'$\Delta_O$ in per-query AUC')
    ax.set_ylabel(r'$\Delta_O$ in MRR')
    ax.set_title(f'AUC and MRR disagree on {len(dis)} of {len(pairs)} pairs', fontsize=6.3,
                 color=INK2, pad=4)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    panel(ax, 'd', dx=30)
    top_legend(fig, [h('o', r'$\Delta_O$, benchmark candidates O', color=BLUE, ms=4),
                     h('o', r'$\Delta_{\mathrm{target}}$, declared target O $\cup$ U', color=ORANGE, ms=4)], y=0.95)
    save(fig, out, 'fig4_conve_distmult')


# ---------------------------------------------------------------- tables
def num(v, d=4, signed=False):
    """v at d decimals, with more decimals where rounding would print a bound (0 or 1) that v is not."""
    while d < 8 and v not in (0, 1) and round(v, d) in (0, 1, -0.0):
        d += 1
    s = f'{v:+.{d}f}' if signed else f'{v:.{d}f}'
    return s.replace('-', '−')


FAMILY = {'both_ends': 'both-ends threshold', 'kcore': 'k-core', 'threshold': 'threshold'}


def ci(R, *key, d=4, signed=False):
    v, lo, hi = R.interval(*key)
    return f'{num(v, d, signed)} [{num(lo, d, signed)}, {num(hi, d, signed)}]'


def write_table(out, name, head, rows, note):
    md = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(r) + ' |' for r in rows]
    with open(os.path.join(out, name + '.md'), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(md) + '\n\n' + note + '\n')
    tex = lambda s: s.replace('−', '$-$').replace('–', '--').replace('%', r'\%').replace('&', r'\&')  # noqa: E731
    lines = [r'\begin{tabular}{' + 'l' * len(head) + '}', r'\toprule', ' & '.join(tex(h) for h in head) + r' \\',
             r'\midrule'] + [' & '.join(tex(c) for c in r) + r' \\' for r in rows] + [r'\bottomrule', r'\end{tabular}']
    with open(os.path.join(out, name + '.tex'), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('% generated by scripts/make_figures.py; ' + note.replace('\n', ' ') + '\n' + '\n'.join(lines) + '\n')


def regime(slug, field):
    """The paper's regime (Section 2.3): (iii) without a card; otherwise (i) when the card's support is
    structural, (ii) when it is coverage only."""
    if field is None:
        return '(iii)'
    return '(i)' if card(slug)['support']['structural'] else '(ii)'


def table1(out):
    reg, pilot = registry(), pilot_rows()
    rows = []
    for slug, unit, field, finding, status in BENCH:
        b = reg[slug]
        p = pilot[b['pilot_row']]
        if field is None:
            w, rule = 'not computable', '—'
        else:
            c = card(slug)
            w = num(c['exclusion'][field])
            rd, rc = c.get('rule_diagnosis') or {}, c.get('rule_check') or {}
            if rd:
                best = rd['best']
                rule = f"best {FAMILY[best['family']]}, T = {best['T']}: J = {num(best['jaccard'], 3)}"
                if 'stated' in rd and (rd['stated']['family'], rd['stated']['T']) != (best['family'], best['T']):
                    rule = f"stated: J = {num(rd['stated']['jaccard'], 3)}; " + rule
            elif rc:
                rule = f"stated topic rule: {rc['violations']} violations on {rc['released_units_checked']} judged topics"
            else:
                rule = '—'
        rows.append([b['name'], p['domain'], unit, regime(slug, field), p['decidability'], w, rule,
                     f'{finding} ({status})'])
    head = ['Benchmark', 'Domain', 'Unit', 'Regime', 'Decidability', 'W', 'Rule diagnosis',
            'Construction finding (prior work)']
    note = ('W and rule diagnosis from the tool cards (tool/library/cards); regime from the card (support) or the '
            'registry (no card); domain and decidability from pilot_decidability.tsv; prior-work status from '
            'PRIOR_WORK.md (section in brackets).')
    write_table(out, 'table1_benchmarks', head, rows, note)
    # LaTeX for ICML: a compact full-width table. The construction findings stay in the Markdown table (and in
    # the text of Section 4); the domain becomes a group row, and the rule diagnosis is shortened.
    tex = lambda s: s.replace('−', '$-$').replace('–', '--').replace('%', r'\%').replace('&', r'\&')  # noqa: E731
    short = {'both_ends': 'both-ends', 'kcore': '$k$-core', 'threshold': 'threshold'}
    cols = [('Benchmark', 1.25), ('Unit', 1.0), ('Regime', 0.42), ('Decidability', 0.8), ('$W$', 0.45),
            ('Rule diagnosis', 2.12)]      # widths set 2026-10-28 so that no name or unit wraps
    spec = ''.join(r'>{\raggedright\arraybackslash}p{' + f'{wd}in' + '}' for _, wd in cols)
    lines = [r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{' + spec + '}', r'\toprule',
             ' & '.join(name for name, _ in cols) + r' \\', r'\midrule']
    domain = None
    for (slug, unit, field, *_), r in zip(BENCH, rows):
        if r[1] != domain:
            if domain is not None:
                lines.append(r'\addlinespace')
            domain = r[1]
            lines.append(r'\multicolumn{' + str(len(cols)) + r'}{l}{\emph{' + tex(domain[0].upper() + domain[1:]) + r'}} \\')
        w, rule = '—', '—'
        if field is not None:
            c = card(slug)
            w = num(c['exclusion'][field])
            rd, rc = c.get('rule_diagnosis') or {}, c.get('rule_check') or {}
            if rd:
                best = rd['best']
                rule = f"best {short[best['family']]} $T={best['T']}$: $J={num(best['jaccard'], 3)}$"
                if 'stated' in rd and (rd['stated']['family'], rd['stated']['T']) != (best['family'], best['T']):
                    rule = f"stated $J={num(rd['stated']['jaccard'], 3)}$; " + rule
            elif rc:
                rule = f"stated rule: {rc['violations']} violations, {rc['released_units_checked']} judged topics"
        lines.append(' & '.join(tex(x) for x in (r[0], r[2], r[3], r[4], w, rule)) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    with open(os.path.join(out, 'table1_benchmarks.tex'), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('% generated by scripts/make_figures.py; ' + note + '\n' + '\n'.join(lines) + '\n')


def table2(R, out):
    rows = []
    for design, m in [('E2', m) for m in MODELS] + [('deploy', 'nbfnet')]:
        k = lambda q: (design, 'primary', m, 'mean', q)  # noqa: E731
        name = NAME[m] if design == 'E2' else 'NBFNet (deployment test)'
        rows.append([name, num(R(*k('W'))), ci(R, *k('auc_O')), ci(R, *k('auc_U_minus_O'), signed=True),
                     ci(R, *k('rr_O')), ci(R, *k('mrr_C_minus_O'), signed=True), ci(R, *k('intrusion'), d=3)])
    write_table(out, 'table2_models',
                ['Model', 'W', 'AUC(O)', 'AUC(U) − AUC(O)', 'MRR(O)', 'MRR(C) − MRR(O)', 'Outranked by U'],
                rows, 'Primary estimand, seed means with 95% triple-bootstrap intervals (records.tsv, design E2 or '
                      'deploy, estimand primary, seed mean, cluster triple). The deployment test is a separate '
                      'estimand and enters no comparison.')


def box1(out):
    c = card('wn18rr')
    ex, rd, mp, hy = c['exclusion'], c['rule_diagnosis'], c['mapping'], c['hygiene']
    pos = ex['by_strata']['pos']
    top_rel = max(hy['reverse_leakage']['by_relation'].items(), key=lambda kv: kv[1]['leaking'])
    lines = [
        '**Box 1. The WN18RR scope card, abridged** (`tool/library/cards/wn18rr.json`)', '',
        f"- **Scope statement:** {c['scope_statement']}",
        f"- **Mapping:** {mp['mapped_ids']:,} of {mp['release_ids']:,} ids mapped through {mp['matched_triples']:,} of "
        f"{mp['release_triples']:,} triples; {mp['merged_ids']} ids merge synsets of different POS.",
        f"- **Exclusion:** W = {ex['W']:.4f} ({ex['excluded']:,} of {ex['units']:,} synsets); by POS: "
        + ', '.join(f"{k} {v['W']:.3f}" for k, v in sorted(pos.items()))
        + f"; per filtered test query on the universe, mean W_q = {ex['per_query']['W_bar']:.4f} over "
          f"{ex['per_query']['queries']:,} queries.",
        f"- **Rule diagnosis:** stated ({FAMILY[rd['stated']['family']]}, T = {rd['stated']['T']}): Jaccard "
        f"{num(rd['stated']['jaccard'], 3)}, {rd['stated']['verdict']}; best found ({FAMILY[rd['best']['family']]}, "
        f"T = {rd['best']['T']}): {num(rd['best']['jaccard'], 3)}, {rd['best']['verdict']}.",
        f"- **Derived scope:** d_min = {c['derived_scope']['d_min']}.",
        f"- **Hygiene:** {hy['reverse_leakage']['leaking']:,} of {hy['reverse_leakage']['test_triples']:,} test "
        f"triples ({hy['reverse_leakage']['share']:.1%}) have their reverse in training, among them {top_rel[1]['leaking']:,} "
        f"of the {top_rel[1]['test']:,} {top_rel[0]} test triples; "
        f"{hy['unseen_test_units']['unseen_units']} test entities never occur in training "
        f"({hy['unseen_test_units']['test_triples_affected']} test triples).",
    ]
    with open(os.path.join(out, 'box1_wn18rr_card.md'), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(lines) + '\n')


def main(out):
    out = os.path.abspath(out)
    R = Records(RECORDS)
    fig2(out)
    fig_app_ten(out)
    fig3(R, out)
    fig4(R, out)
    table1(out)
    table2(R, out)
    box1(out)
    print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(P4, 'figures'))
