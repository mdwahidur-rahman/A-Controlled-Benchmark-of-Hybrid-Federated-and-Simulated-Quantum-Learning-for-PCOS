"""Shared publication style. Palette = validated reference palette (slots 1-3, light mode, surface #ffffff for print).
Large, high-contrast text: all axis/tick/data text in primary ink, bold axis labels, 12-14 pt."""
import matplotlib as mpl, matplotlib.pyplot as plt
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
PRETTY = {"mobilenetv2_100": "MobileNetV2", "efficientnet_b0": "EfficientNet-B0", "densenet121": "DenseNet-121",
          "resnet50": "ResNet-50", "resnext50_32x4d": "ResNeXt-50", "pvt_v2_b2": "PVT-v2-B2", "twins_pcpvt_small": "Twins-PCPVT-S",
          "swin_tiny_patch4_window7_224": "Swin-T", "crossvit_small_240": "CrossViT-S", "mobilevit_s": "MobileViT-S",
          "eca_botnext26ts_256": "ECA-BotNeXt26ts"}
FAMCOL = {"CNN": BLUE, "Transformer": ORANGE, "Hybrid": AQUA}
FAMMK = {"CNN": "o", "Transformer": "s", "Hybrid": "^"}
def setup():
    mpl.rcParams.update({
        "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 12, "text.color": INK,
        "axes.labelsize": 13, "axes.labelweight": "bold", "axes.labelcolor": INK, "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 10, "axes.edgecolor": AXIS, "axes.linewidth": 1.1,
        "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": "white", "figure.facecolor": "white",
        "xtick.labelsize": 12, "ytick.labelsize": 12, "xtick.color": INK, "ytick.color": INK, "xtick.major.width": 1.1,
        "ytick.major.width": 1.1, "xtick.major.size": 5, "ytick.major.size": 5, "xtick.major.pad": 4, "ytick.major.pad": 4,
        "grid.color": GRID, "grid.linewidth": 0.9, "grid.linestyle": "-", "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 12, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.pad_inches": 0.12})
def panel(ax, letter, dx=-0.0, dy=1.0):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=17, fontweight="bold", color=INK, va="bottom", ha="right",
            bbox=dict(fc="none", ec="none", pad=0), clip_on=False)
def save(fig, name, root):
    fig.savefig(f"{root}/figures/{name}.pdf"); fig.savefig(f"{root}/figures/{name}.png", dpi=600)
    fig.savefig(f"{root}/figures/preview/{name}.png", dpi=110); plt.close(fig)
def dotrow(ax, y, v, lo=None, hi=None, color=BLUE, marker="o", ms=10, label=None, z=3):
    if lo is not None: ax.plot([lo, hi], [y, y], color=color, lw=2.0, solid_capstyle="round", zorder=z - 1)
    ax.plot([v], [y], marker=marker, ms=ms, mfc=color, mec="white", mew=1.3, ls="none", zorder=z, label=label)

def heading(fig, ax, letter, title, gap=0.014):
    """Panel letter + title, left-aligned to the leftmost tick/axis label of this panel."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    xs = [t.get_window_extent(r).x0 for t in ax.get_yticklabels() if t.get_text()]
    if ax.get_ylabel(): xs.append(ax.yaxis.label.get_window_extent(r).x0)
    x0 = (min(xs) if xs else ax.get_window_extent(r).x0) / fig.bbox.width
    y = ax.get_position().y1 + gap
    fig.text(x0, y, letter, fontsize=17, fontweight="bold", ha="left", va="bottom", color=INK)
    fig.text(x0 + 0.034, y + 0.0015, title, fontsize=14, fontweight="bold", ha="left", va="bottom", color=INK)
