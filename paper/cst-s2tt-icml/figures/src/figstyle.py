"""House style for the CST-S2TT paper figures.

ICML: 10 pt Times text, 3.25 in column, 6.75 in page width; Okabe-Ito palette; TrueType embedding, never Type 3.
Extended with ink tokens, diagram helpers and a Hangul fallback font (AppleMyungjo, a serif Hangul face) so that
Korean text renders in the same serif look.
Note: a string that contains mathtext ($...$) cannot use the fallback; keep Hangul and math in separate strings.

Colour semantics (identical in every figure):
  language / speaker identity : German #0072B2 (blue), English #E69F00 (orange), Korean #009E73 (green)
  timing configuration        : L0 natural #009E73, L0 mediated #CC79A7 (always direct-labelled), L1 overlap #D55E00
  results                     : our model #D55E00 (vermillion accent); baselines in neutral inks + a fixed marker per
                                system; systems with oracle input (offline, Gold->LLM (MU2)) as filled black markers
                                with a white edge, direct-labelled; oracle wrapper = filled, realistic wrapper = hollow,
                                no wrapper (2xSeamlessStreaming) = half-filled
Text never wears a data colour; labels inside a coloured fill use white or ink by luminance.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

COL_W, PAGE_W = 3.25, 6.75
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", purple="#CC79A7",
         sky="#56B4E9", yellow="#F0E442", grey="#7F7F7F", black="#000000")
LANG_COLOR = {"German": C["blue"], "English": C["orange"], "Korean": C["green"]}
LANG_TEXT_ON = {"German": "white", "English": "#1a1a1a", "Korean": "white"}   # label ink inside a fill
LANG_TINT = {"German": "#cfe3f1", "English": "#fbe7bf", "Korean": "#cdeee3"}   # light washes behind ink text
CONFIG_COLOR = {"natural": C["green"], "mediated": C["purple"], "L1": C["red"]}

INK, INK2, MUTED, RULE, WASH = "#1a1a1a", "#4d4d4d", "#8c8c8c", "#bdbdbd", "#f2f2f2"
GRID = "#e6e6e6"
OURS = C["red"]
PLANNED_LS = (0, (3, 2))


def setup(base=7):
    plt.rcParams.update({
        "font.family": ["STIXGeneral", "AppleMyungjo"],     # serif; Hangul falls back to a serif Hangul face
        "mathtext.fontset": "stix",
        "font.size": base, "axes.titlesize": base, "axes.labelsize": base, "legend.fontsize": base - 1,
        "xtick.labelsize": base - 1, "ytick.labelsize": base - 1,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
        "lines.linewidth": 1.0, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,          # embed TrueType, never Type 3
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })


def save(fig, stem, outdir, png=True):
    """Vector PDF for the paper plus a 300 dpi PNG preview."""
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf")
    if png:
        fig.savefig(outdir / f"{stem}.png", dpi=300)
    plt.close(fig)
    print("wrote", outdir / f"{stem}.pdf")


def rbox(ax, x, y, w, h, fc="white", ec=INK2, lw=0.7, ls="-", r=0.04, z=2, aspect=1.0, clip_on=True):
    """Rounded box in data coordinates (use an axes whose units are inches for diagrams)."""
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", mutation_aspect=aspect,
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, clip_on=clip_on)
    ax.add_patch(p)
    return p


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=0.7, ms=6, z=3, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=ms, shrinkA=0, shrinkB=0,
                                ls=ls), zorder=z)


def canvas(w, h):
    """Figure with one axes in inch units, no ticks: for schematic diagrams."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def text_width(fig, s, size, weight="normal", style="normal"):
    """Rendered width of a one-line string in inches (for sizing diagram boxes so text never overflows)."""
    r = fig.canvas.get_renderer()
    t = fig.text(0, 0, s, fontsize=size, weight=weight, style=style)
    w = t.get_window_extent(r).width / fig.dpi
    t.remove()
    return w


def style_axes(ax, grid_axis=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=INK2, length=2, width=0.6, color=RULE)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, lw=0.5)
        ax.set_axisbelow(True)


def pending_note(ax, text="results pending", x=0.5, y=0.5, ha="center"):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va="center", fontsize=6.0, color=MUTED, style="italic")
