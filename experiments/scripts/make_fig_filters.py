"""The figure on filtering by model outputs (integrated version, Results block 3), built only from stored records.

usage: python -E -P make_fig_filters.py [--results DIR] [--out DIR]
       (defaults: results/ and figures/ in the folder above this script)

Writes fig_filters.pdf (and fig_filters.png if PyMuPDF is installed) and fig_filters_values.tsv, which lists every
value drawn with the record it comes from and the SHA-256 of each source file. check_fig_filters.py checks that list
against the records.

Sources, and nothing else (paths under --results):
  panel_filter/records_panel_size.tsv          a  BigCodeBench, own-vote penalty T_self by panel size, per format
  mmlu_reconstruction/records_panel_size.tsv   a  MMLU under MMLU-Pro's rule (analysis A), T_self by panel size
  separated_scoring/effects.tsv                b  votes from Complete outputs: the penalty on the outputs that voted
                                                  (metric source) and on the same models' Instruct outputs, which did
                                                  not vote (metric target)
  panel_composition/records.tsv                c  concentration cost: the pooled D(8) and the eight line contrasts

Nothing here computes a statistic. Each value is looked up by its full key, which must be unique in its file, and
drawn as stored, in points (100 times the stored share).

Intervals, as stored: a, percentile 95% intervals per panel size (100 bootstrap replicates); b, pointwise percentile
95% intervals (low, high) for both series, not the simultaneous ones; c, the percentile 95% interval of the pooled
D(8) and, for each line, the simultaneous 95% interval over the eight line contrasts (seven lines at 8 builders, Qwen
at 16). In a and b both axes are logarithmic and the lines only join the measured panel sizes; nothing is fitted.

Style as make_figures.py: all text typeset by LaTeX (text.usetex), so the PDF embeds only Type 1 fonts; this needs a
LaTeX installation with the helvet, type1cm and underscore packages. Blue and orange are the pair of the dataviz
reference palette that make_figures.py validated; grey marks carry no identity. The width is the text width of the
integrated version (letter paper, 2.5 cm margins).
"""
import argparse
import csv
import hashlib
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator  # noqa: E402

P4 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
NAME = 'fig_filters'
WIDTH, HEIGHT = 6.5, 2.45                     # inches; the text width is 6.53 in

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BAND = '#e9f1fb'                              # the pooled row of c: BLUE at about 10% on white

# as make_figures.py: Helvetica for all text; tick labels in the text font; the minus sign is \textminus
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

# file, key columns
SOURCES = {
    'panel_filter/records_panel_size.tsv': ('split', 'subject', 'quantity'),
    'mmlu_reconstruction/records_panel_size.tsv': ('analysis', 'subject', 'quantity'),
    'separated_scoring/effects.tsv': ('direction', 'p', 'metric'),
    'panel_composition/records.tsv': ('analysis', 'group', 'quantity'),
}
LINES = {'qwen': 'Qwen', 'llama': 'Llama', 'deepseek-coder': 'DeepSeek-Coder', 'gpt': 'GPT', 'mistral': 'Mistral',
         'codellama': 'Code Llama', 'qwen-coder': 'Qwen-Coder'}
P_BCB, P_MMLU, P_SEP = (4, 8, 16, 32, 64, 96), (4, 8, 16, 32, 64), (4, 8, 16, 32, 64)
LOG_TICKS = (0.1, 0.2, 0.5, 1, 2, 5, 10)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def points(raw):
    """A stored share as drawn and listed: 100 times the value, ten significant digits."""
    return f'{100 * float(raw):.10g}'


class Records:
    """Every row of the four sources by (file, key); a key that occurs twice is an error."""

    def __init__(self, results):
        self.results, self.rows = results, {}
        for rel, cols in SOURCES.items():
            with open(os.path.join(results, *rel.split('/')), newline='', encoding='utf-8') as fh:
                for r in csv.DictReader(fh, delimiter='\t'):
                    key = (rel, '|'.join(r[c] for c in cols))
                    if key in self.rows:
                        raise SystemExit(f'{rel}: key {key[1]} occurs twice')
                    self.rows[key] = r

    def ref(self, rel, key, column):
        """The stored string and its reference 'key#column'; a missing or empty field is an error."""
        r = self.rows.get((rel, key))
        if r is None or r.get(column, '') in ('', None):
            raise SystemExit(f'{rel}: no value for {key}#{column}')
        return r[column], f'{key}#{column}'


class Listing:
    """The values file: one line per element drawn."""

    HEADER = ('panel', 'element', 'x', 'estimate', 'lo', 'hi', 'interval', 'source', 'ref_estimate', 'ref_lo', 'ref_hi')

    def __init__(self):
        self.lines = []

    def add(self, panel, element, x, interval, rel, refs):
        (e, re_), (lo, rlo), (hi, rhi) = refs
        self.lines.append((panel, element, str(x), points(e), points(lo), points(hi), interval, rel, re_, rlo, rhi))
        return float(points(e)), float(points(lo)), float(points(hi))

    def write(self, path, results):
        with open(path, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('# Every value drawn in fig_filters.pdf, in points (100 times the stored value), with its record;\n'
                     '# written by make_fig_filters.py and checked by check_fig_filters.py. Source files and SHA-256:\n')
            for rel in SOURCES:
                fh.write(f'# sha256\t{rel}\t{sha256(os.path.join(results, *rel.split("/")))}\n')
            fh.write('\t'.join(self.HEADER) + '\n')
            for line in self.lines:
                fh.write('\t'.join(line) + '\n')


def log_axes(ax, xs):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in xs]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator(LOG_TICKS))
    ax.yaxis.set_major_formatter(FixedFormatter([f'{t:g}' for t in LOG_TICKS]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(0.05, 12)                     # room below the data for the legend
    ax.set_xlim(xs[0] / 1.25, xs[-1] * 1.25)
    ax.grid(axis='y', color=GRID, lw=0.5)
    ax.set_xlabel('builders on the panel (log scale)')


def series(ax, xs, vals, color, filled, ls):
    ys, los, his = zip(*vals)
    ax.plot(xs, ys, ls=ls, color=color, lw=1.0, zorder=2)          # joins the measured sizes; nothing is fitted
    ax.errorbar(xs, ys, yerr=[[y - lo for y, lo in zip(ys, los)], [hi - y for y, hi in zip(ys, his)]], fmt='none',
                ecolor=color, elinewidth=0.8, capsize=0, zorder=3)
    ax.plot(xs, ys, 'o', ms=4, color=color, mfc=color if filled else 'white', mew=1.0, ls='', zorder=4)


def handle(color, ls, filled, label):
    return Line2D([], [], color=color, ls=ls, lw=1.0, marker='o', ms=4, mew=1.0, mfc=color if filled else 'white',
                  label=label)


LEGEND = dict(loc='lower left', frameon=True, facecolor='white', edgecolor='none', framealpha=1.0, borderpad=0.2,
              handlelength=2.2, handletextpad=0.4, borderaxespad=0.1, labelspacing=0.45)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results', default=os.path.join(P4, 'results'))
    ap.add_argument('--out', default=os.path.join(P4, 'figures'))
    a = ap.parse_args()
    rec, out = Records(a.results), Listing()

    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, HEIGHT),
                             gridspec_kw={'width_ratios': [1, 1, 0.52, 1.1], 'wspace': 0.3})
    axes[2].set_axis_off()                    # room for the row labels of c
    ax_a, ax_b, ax_c = axes[0], axes[1], axes[3]

    # a: a member's own vote; the outputs that vote are the outputs scored
    rel = 'panel_filter/records_panel_size.tsv'
    pct100 = 'percentile 95%, 100 bootstrap replicates'
    for split, ls, element in (('complete', '-', 'BigCodeBench, Complete'), ('instruct', '--', 'BigCodeBench, Instruct')):
        vals = [out.add('a', element, p, pct100, rel,
                        [rec.ref(rel, f'{split}|p={p}|T_self{s}', 'value') for s in ('', '_lo', '_hi')])
                for p in P_BCB]
        series(ax_a, P_BCB, vals, BLUE, True, ls)
    rel = 'mmlu_reconstruction/records_panel_size.tsv'
    vals = [out.add('a', "MMLU, MMLU-Pro's rule", p, pct100, rel,
                    [rec.ref(rel, f'A|p={p}|T_self{s}', 'value') for s in ('', '_lo', '_hi')]) for p in P_MMLU]
    series(ax_a, P_MMLU, vals, ORANGE, True, '-')
    log_axes(ax_a, P_BCB)
    ax_a.set_ylabel('penalty on the kept set\n(points, log scale)')
    ax_a.set_title(r'\textbf{a}\enspace A member\textquoteright s own vote', loc='left')
    ax_a.legend(handles=[handle(BLUE, '-', True, 'BigCodeBench,\nComplete'),
                         handle(BLUE, '--', True, 'BigCodeBench,\nInstruct'),
                         handle(ORANGE, '-', True, "MMLU,\nMMLU-Pro's rule")], **LEGEND)

    # b: the same models' other prompt format, which did not vote (votes from Complete outputs)
    rel = 'separated_scoring/effects.tsv'
    pct2000 = 'pointwise percentile 95%, 2,000 bootstrap replicates'
    for metric, filled, element in (('source', True, 'Complete outputs (voted)'),
                                    ('target', False, 'Instruct outputs (did not vote)')):
        vals = [out.add('b', element, p, pct2000, rel,
                        [rec.ref(rel, f'complete_to_instruct|{p}|{metric}', c) for c in ('estimate', 'low', 'high')])
                for p in P_SEP]
        series(ax_b, P_SEP, vals, BLUE, filled, '-')
    log_axes(ax_b, P_SEP)
    ax_b.set_title(r'\textbf{b}\enspace Same model, other prompt format', loc='left')
    ax_b.legend(handles=[handle(BLUE, '-', True, 'Complete outputs\n(voted)'),
                         handle(BLUE, '-', False, 'Instruct outputs\n(did not vote)')], **LEGEND)

    # c: a concentrated against a spread panel, for the line's models that never voted
    rel = 'panel_composition/records.tsv'
    pooled = out.add('c', 'seven lines, pooled', 'pooled', 'percentile 95%, 1,000 bootstrap replicates', rel,
                     [rec.ref(rel, f'primary|T8|{q}', 'value') for q in ('estimate', 'lo', 'hi')])
    sim = 'simultaneous 95% over the eight line contrasts, 1,000 bootstrap replicates'
    by_line = {k: out.add('c', LINES[k], 'line, 8 builders', sim, rel,
                          [rec.ref(rel, f'line_8|{k}|{q}', 'value') for q in ('estimate', 'simultaneous_lo',
                                                                               'simultaneous_hi')])
               for k in LINES}
    supp = out.add('c', 'Qwen, 16 builders (supplementary)', 'line, 16 builders', sim, rel,
                   [rec.ref(rel, f'line_16|qwen|{q}', 'value') for q in ('estimate', 'simultaneous_lo',
                                                                        'simultaneous_hi')])
    order = sorted(LINES, key=lambda k: -by_line[k][0])
    y_pool, y_supp = len(order) + 1.5, -0.4
    ax_c.axhspan(y_pool - 0.62, y_pool + 0.62, color=BAND, lw=0, zorder=0)
    ax_c.axhline(len(order) + 0.45, color=GRID, lw=0.6, zorder=1)
    ax_c.axhline(0.35, color=GRID, lw=0.6, zorder=1)
    ax_c.axvline(0, color=AXIS, lw=0.8, zorder=1)
    ax_c.axvline(1, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
    v, lo, hi = pooled
    ax_c.errorbar([v], [y_pool], xerr=[[v - lo], [hi - v]], fmt='D', ms=5.2, color=BLUE, mfc=BLUE, mew=1.0,
                  elinewidth=2.0, capsize=2.6, capthick=1.2, zorder=4)
    label = f'{v:.2f} [{lo:.2f}, {hi:.2f}]'
    out.lines.append(('c', 'printed under the label of the pooled row', 'pooled', label, '', '',
                      'text: estimate [lo, hi] to 2 decimals', rel, 'primary|T8|estimate#value', 'primary|T8|lo#value',
                      'primary|T8|hi#value'))
    ticks, names = [y_pool], [r'\textbf{seven lines, pooled}' + '\n' + rf'\textbf{{{label}}}']
    for i, k in enumerate(order):
        y = len(order) - i
        v, lo, hi = by_line[k]
        ax_c.plot([lo, hi], [y, y], color=BLUE, lw=1.0, zorder=3)
        ax_c.plot([v], [y], 'o', ms=4.2, color=BLUE, mfc='white', mew=1.0, zorder=4)
        ticks.append(y)
        names.append(LINES[k])
    v, lo, hi = supp
    ax_c.plot([lo, hi], [y_supp, y_supp], color=MUTED, lw=1.0, zorder=3)
    ax_c.plot([v], [y_supp], 'o', ms=4.2, color=MUTED, mfc='white', mew=1.0, zorder=4)
    ticks.append(y_supp)
    names.append('Qwen, 16 builders\n(supplementary)')
    ax_c.set_yticks(ticks)
    ax_c.set_yticklabels(names, fontsize=6.3)
    ax_c.get_yticklabels()[0].set_color(INK)
    ax_c.get_yticklabels()[-1].set_color(MUTED)
    ax_c.tick_params(axis='y', length=0)
    ax_c.text(1.06, y_supp - 0.05, "1 point: prespecified\n``material''", fontsize=5.8, color=INK2, va='center',
              ha='left')
    ax_c.set_xlim(-0.6, 2.45)
    ax_c.set_ylim(y_supp - 0.9, y_pool + 0.75)
    ax_c.xaxis.set_major_locator(FixedLocator([0, 1, 2]))
    ax_c.grid(axis='x', color=GRID, lw=0.5)
    ax_c.set_xlabel('disadvantage of same-line models\nthat did not vote (points)')
    ax_c.set_title(r'\textbf{c}\enspace Concentrated vs spread panel', loc='left')

    os.makedirs(a.out, exist_ok=True)
    pdf = os.path.join(a.out, f'{NAME}.pdf')
    fig.savefig(pdf, bbox_inches='tight', metadata={'CreationDate': None})
    plt.close(fig)
    out.write(os.path.join(a.out, f'{NAME}_values.tsv'), a.results)
    try:
        import pymupdf
    except ImportError:
        print(f'{NAME}.png skipped: PyMuPDF is not installed')
    else:
        with pymupdf.open(pdf) as doc:
            doc[0].get_pixmap(dpi=600, alpha=False).save(os.path.join(a.out, f'{NAME}.png'))
    print(f'{NAME}: {len(out.lines)} elements listed; written to {a.out}')


if __name__ == '__main__':
    main()
