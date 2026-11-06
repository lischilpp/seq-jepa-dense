r"""Figure style of the paper, used by eval_umap.

Figures are built at their final printed size (the template's \textwidth) and
included without rescaling, so FONTSIZE is the printed font size. Text uses
matplotlib's bundled Times-like STIX fonts.
"""
import matplotlib
import matplotlib.pyplot as plt

TEXTWIDTH_IN = 312.98038 / 72.27  # \textwidth of the single-column template, in inches
FONTSIZE = 8.0      # printed pt, as the template's captions
INK = "#000000"

# Okabe-Ito: distinguishable under common forms of color blindness.
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
             "#D55E00", "#56B4E9", "#F0E442", "#000000"]

matplotlib.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": FONTSIZE,
    "text.color": INK,
    "axes.labelcolor": INK,
})


def palette(keys):
    """Stable color per category."""
    return {k: OKABE_ITO[i % len(OKABE_ITO)] for i, k in enumerate(keys)}


def renderer(fig):
    """Renderer for measuring text."""
    return fig.canvas.get_renderer()


def measure(fig, text, fontsize, weight="normal"):
    """Width in inches of the widest line of `text` at `fontsize`."""
    rend = renderer(fig)
    widest = 0.0
    for line in text.split("\n"):
        t = fig.text(0.5, 0.5, line, fontsize=fontsize, fontweight=weight)
        widest = max(widest, t.get_window_extent(rend).width / fig.dpi)
        t.remove()
    return widest


def probe_figure(width=4.0, height=2.0):
    """Throwaway figure for measuring text before the real one exists."""
    return plt.figure(figsize=(width, height))


def margins(fig_w, fig_h, left, right, top, bottom):
    """Margins in inches -> figure fractions for gridspec."""
    return dict(left=left / fig_w, right=1 - right / fig_w,
                top=1 - top / fig_h, bottom=bottom / fig_h)


def style(ax):
    """Bare image axes: no ticks, no spines."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save(fig, stem, formats=("pdf", "png"), dpi=300):
    """One file per format; no bbox_inches="tight", which would change the printed width."""
    for ext in formats:
        fig.savefig(f"{stem}.{ext}", dpi=dpi)
        print(f"  saved {stem}.{ext}")
    plt.close(fig)
