#!/usr/bin/env python3
"""Vector schematics for the manuscript: Fig. 1 (workflow) and Fig. 2 (architecture), drawn to the specification in figures/FIGURE_SPECS.md.
Only counts and design facts that appear in the manuscript tables or in results/stage0_audit.json / stage1_splits.json are printed. No result (accuracy, MCC) is drawn.
Every text block is wrapped by measured width and checked to lie inside its box (fit check); the script reports any violation.
Usage: PCOS_BENCH_ROOT=<repo root> OUT=<output dir> PREV=<preview dir> python3 31_schematics.py"""
import os, json
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

R = os.environ.get("PCOS_BENCH_ROOT", os.getcwd())
OUT = os.environ.get("OUT", "figures"); PREV = os.environ.get("PREV", "figures/preview")
os.makedirs(OUT, exist_ok=True); os.makedirs(PREV, exist_ok=True)
aud = json.load(open(f"{R}/results/stage0_audit.json")); spl = json.load(open(f"{R}/results/stage1_splits.json"))
INK, INK2, EDGE, FILL = "#0b0b0b", "#3d3c39", "#9c9b92", "#eeede8"
BLUE, ORANGE, PURPLE, GREEN = "#2a78d6", "#eb6834", "#7b5fc4", "#1baf7a"
TINT = {BLUE: "#e6effb", ORANGE: "#fdeee7", PURPLE: "#eeeaf8", GREEN: "#e3f5ee"}
mpl.rcParams.update({"mathtext.fontset": "dejavusans", "font.family": ["Liberation Sans", "DejaVu Sans"], "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white", "figure.facecolor": "white"})
FP = {w: FontProperties(family=["Liberation Sans", "DejaVu Sans"], weight=w) for w in ("normal", "bold")}
FP["italic"] = FontProperties(family=["Liberation Sans", "DejaVu Sans"], style="italic")
LS = 1.22                     # line spacing (x font size)
PROBLEMS = []

def tw(s, fs, w="normal"):
    if not s.strip(): return 0.0
    return TextPath((0, 0), s, size=fs, prop=FP[w]).get_extents().width / 72.0   # inches
def wrap(s, width_in, fs, w="normal"):
    out = []
    for para in s.split("\n"):
        line = ""
        for word in para.split(" "):
            t = (line + " " + word).strip()
            if tw(t, fs, w) <= width_in or not line: line = t
            else: out.append(line); line = word
        out.append(line)
    return out
def block_h(lines, fs): return len(lines) * fs * LS / 72.0

def put(ax, x, y, s, fs=6.6, w="normal", color=INK, ha="center", va="center", style=None, width=None, what=""):
    """Draw wrapped text centred on (x, y) [inches]; if width is given the text is wrapped to it. Returns (lines, height)."""
    lines = wrap(s, width, fs, w) if width else s.split("\n"); h = block_h(lines, fs)
    if width:
        for l in lines:
            if tw(l, fs, w) > width + 1e-3: PROBLEMS.append(f"word wider than box: {what or s[:30]!r}")
    y0 = y + h / 2 if va == "center" else (y if va == "top" else y + h)
    for i, l in enumerate(lines):
        ax.text(x if ha == "center" else x, y0 - (i + 0.5) * fs * LS / 72.0, l, fontsize=fs, fontweight=w if w != "italic" else "normal", fontstyle="italic" if w == "italic" else "normal",
                color=color, ha=ha, va="center", fontfamily=["Liberation Sans", "DejaVu Sans"])
    return lines, h

def box(ax, x, y, w, h, title=None, body=None, ec=EDGE, fc=FILL, lw=0.7, fs=6.6, tfs=7.0, pad=0.05, r=0.035, ls="-", tcolor=INK, bold_body=False):
    """Rounded box with optional bold title and body text, both wrapped to the inner width and vertically centred; checks the fit."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    iw = w - 2 * pad; tl = wrap(title, iw, tfs, "bold") if title else []; bl = wrap(body, iw, fs, "bold" if bold_body else "normal") if body else []
    ht = block_h(tl, tfs); hb = block_h(bl, fs); gap = 0.03 if (tl and bl) else 0; tot = ht + hb + gap
    if tot > h - 2 * 0.025 + 1e-6: PROBLEMS.append(f"text too tall in box {(title or body)[:28]!r}: {tot:.3f} > {h - 0.05:.3f} in")
    yy = y + h / 2 + tot / 2
    if tl: put(ax, x + w / 2, yy - ht / 2, title, tfs, "bold", tcolor, width=iw); yy -= ht + gap
    if bl: put(ax, x + w / 2, yy - hb / 2, body, fs, "bold" if bold_body else "normal", INK, width=iw)
    return (x, y, w, h)

def arrow(ax, p, q, color=INK2, lw=0.9, ls="-", rad=0.0, ms=6, z=3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z))
def line(ax, pts, color=INK2, lw=0.9, ls="-", z=3):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=z)

def canvas(W, H):
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off"); return fig, ax

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{PREV}/{name}.png", dpi=220); plt.close(fig)

# =============================================================== Fig. 1: workflow
def badge(ax, x, y, k):
    ax.add_patch(plt.Circle((x, y), 0.07, fc=INK, ec="white", lw=0.8, zorder=5)); ax.text(x, y, str(k), color="white", fontsize=6.8, fontweight="bold", ha="center", va="center", zorder=6)

def fig1():
    W = 7.0; m = 0.05; gap = 0.30; bw = (W - 2 * m - 3 * gap) / 4; r1h, r2h, vgap, bandh = 0.84, 1.08, 0.34, 0.20
    H = 0.07 + r1h + vgap + r2h + 0.10 + bandh + 0.03; fig, ax = canvas(W, H)
    y1 = H - 0.07 - r1h; y2 = y1 - vgap - r2h; xs = [m + i * (bw + gap) for i in range(4)]
    npos, nneg = aud["class_counts"]["infected"], aud["class_counts"]["noninfected"]
    row1 = [("Public archive", f"figshare, CC BY 4.0\n{aud['n']:,} ultrasound JPEG images\n{npos:,} PCOS-positive\n{nneg:,} PCOS-negative"),
            ("Quality control", f"MD5 and perceptual-hash audit; {spl['n_unique_md5']:,} byte-unique files; {aud['exact_dup_groups']:,} exact-duplicate groups hold {aud['exact_dup_images']:,} images"),
            ("Preprocessing", "three variants: plain, CLAHE, shortcut-suppressed"),
            ("Partition", f"duplicate groups g0 ({spl['n_groups_g0']:,}) are the unit; 0.64 / 0.16 / 0.20, seeds 0 to 4; image-level split kept only as a contrast")]
    for i, (t, b) in enumerate(row1):
        box(ax, xs[i], y1, bw, r1h, t, b, fs=6.6, tfs=7.2, pad=0.07); badge(ax, xs[i] + 0.02, y1 + r1h - 0.02, i + 1)
        if i < 3: arrow(ax, (xs[i] + bw + 0.015, y1 + r1h / 2), (xs[i + 1] - 0.015, y1 + r1h / 2))
    put(ax, xs[0] + bw / 2, y1 + 0.075, "no patient identifiers in the archive", 6.5, "italic", INK2, width=bw - 0.1)
    # row 2: stage 5, wide stage 6, stage 7
    box(ax, xs[0], y2, bw, r2h, "Frozen encoders", "11 ImageNet backbones (CNN, Transformer, hybrid); features cached once per byte-unique file", fs=6.6, tfs=7.2, pad=0.07); badge(ax, xs[0] + 0.02, y2 + r2h - 0.02, 5)
    x6 = xs[1]; w6 = 2 * bw + gap
    box(ax, x6, y2, w6, r2h, None, None, fc="white", lw=0.7); badge(ax, x6 + 0.02, y2 + r2h - 0.02, 6)
    put(ax, x6 + w6 / 2, y2 + r2h - 0.115, "Three tracks on the same frozen features", 7.2, "bold", width=w6 - 0.2)
    trk = [("Classical heads", "tuned heads on frozen features", BLUE), ("Federated", "non-IID clients, group-level DP", ORANGE), ("Simulated quantum slot", "noiseless statevector, matched controls", PURPLE)]
    sw = (w6 - 0.16 - 2 * 0.08) / 3
    for k, (t2, b2, c) in enumerate(trk):
        box(ax, x6 + 0.08 + k * (sw + 0.08), y2 + 0.08, sw, r2h - 0.34, t2, b2, ec=c, fc=TINT[c], lw=1.2, fs=6.5, tfs=6.9, pad=0.05)
    box(ax, xs[3], y2, bw, r2h, "Evaluation", "13 metrics at threshold 0.5; group-cluster bootstrap, B\u00a0=\u00a01000; Holm-adjusted contrasts; calibration", fs=6.6, tfs=7.2, pad=0.07); badge(ax, xs[3] + 0.02, y2 + r2h - 0.02, 7)
    arrow(ax, (xs[0] + bw + 0.015, y2 + r2h / 2), (x6 - 0.015, y2 + r2h / 2)); arrow(ax, (x6 + w6 + 0.015, y2 + r2h / 2), (xs[3] - 0.015, y2 + r2h / 2))
    # snake from stage 4 down to stage 5
    ymid = y1 - vgap / 2; x4c = xs[3] + bw / 2; x5c = xs[0] + bw / 2
    line(ax, [(x4c, y1 - 0.004), (x4c, ymid), (x5c, ymid)]); arrow(ax, (x5c, ymid), (x5c, y2 + r2h + 0.004))
    # explanation arm: green box hanging under the evaluation box
    ax.add_patch(FancyBboxPatch((m, 0.03), W - 2 * m, bandh, boxstyle="round,pad=0,rounding_size=0.03", fc="#f6f5f1", ec=EDGE, lw=0.6))
    ax.text(W / 2, 0.03 + bandh / 2, "Duplicate groups are the unit of partitioning, resampling and federated clients throughout.", fontsize=7.0, ha="center", va="center", color=INK, fontweight="bold")
    save(fig, "fig01_workflow")

# =============================================================== Fig. 2: architecture
def fig2():
    W, H = 7.0, 2.93; fig, ax = canvas(W, H)
    gut = 0.16; tx, tw_ = gut + 0.06, 1.22; top = H - 0.05
    th, tg = 0.44, 0.12
    labels = [("Ultrasound image", None), ("Preprocessing variant", "plain, CLAHE or shortcut-suppressed"), ("Frozen encoder E", "11 backbones; frozen, not trained"), ("Pooled feature", "$\\mathbf{f}\\in\\mathbb{R}^{D}$")]
    ys = []; y = top - th
    for t, b in labels:
        ys.append(y); box(ax, tx, y, tw_, th, t, b, fs=6.5, tfs=6.9, pad=0.04); y -= th + tg
    for i in range(3): arrow(ax, (tx + tw_ / 2, ys[i] - 0.005), (tx + tw_ / 2, ys[i + 1] + th + 0.005))
    px, pw = 2.0, 4.17; ph, pg, phd = 0.64, 0.12, 0.17
    pys = [top - ph - k * (ph + pg) for k in range(3)]
    bus_x, bus_w = 1.62, 0.20; bus_y0, bus_y1 = pys[2], pys[0] + ph
    ax.add_patch(FancyBboxPatch((bus_x, bus_y0), bus_w, bus_y1 - bus_y0, boxstyle="round,pad=0,rounding_size=0.03", fc=FILL, ec=EDGE, lw=0.7, zorder=2))
    ax.text(bus_x + bus_w / 2, (bus_y0 + bus_y1) / 2, "Standardization (training statistics)", rotation=90, fontsize=6.6, fontweight="bold", ha="center", va="center", color=INK, zorder=4)
    assert tw("Standardization (training statistics)", 6.6, "bold") < (bus_y1 - bus_y0) - 0.08
    arrow(ax, (tx + tw_ + 0.005, ys[3] + th / 2), (bus_x - 0.005, ys[3] + th / 2))
    panels = [("(a) Classical head", BLUE, [("Logistic regression or MLP head", 1.70), ("p(PCOS-positive)", 1.05)]),
              ("(b) Federated head", ORANGE, [("Clients = duplicate groups", 0.86), ("FedAvg / FedProx / SCAFFOLD, optional group-level DP-SGD", 1.16), ("Global head", 0.60), ("p(PCOS-positive)", 0.90)]),
              ("(c) Simulated quantum slot", PURPLE, [("tanh compressor to n angles", 0.80), ("Simulated circuit (statevector, noiseless)", 1.00), ("Read-out $\\langle Z_i\\rangle$", 0.54), ("Linear layer", 0.52), ("p(PCOS-positive)", 0.90)])]
    ends = []
    for k, (ttl, c, items) in enumerate(panels):
        py = pys[k]
        ax.add_patch(FancyBboxPatch((px, py), pw, ph, boxstyle="round,pad=0,rounding_size=0.04", fc=TINT[c], ec=c, lw=1.2, zorder=1))
        ax.text(px + 0.07, py + ph - phd / 2 - 0.005, ttl, fontsize=6.9, fontweight="bold", ha="left", va="center", color=INK, zorder=4)
        ax.text(px + pw - 0.07, py + ph - phd / 2 - 0.005, "trained", fontsize=6.5, fontstyle="italic", ha="right", va="center", color=INK2, zorder=4)
        inner_w = pw - 0.12; sumw = sum(w for _, w in items); g = (inner_w - sumw) / (len(items) - 1)
        if g < 0.07: PROBLEMS.append(f"panel {ttl}: arrow gap {g:.3f} in")
        bx = px + 0.06; byy = py + 0.045; bhh = ph - phd - 0.075
        for j, (txt, w) in enumerate(items):
            last = j == len(items) - 1
            box(ax, bx, byy, w, bhh, None, txt, fc="white", ec=c, lw=0.8, fs=6.5, pad=0.035, bold_body=last, r=0.03)
            if not last: arrow(ax, (bx + w + 0.003, byy + bhh / 2), (bx + w + g - 0.003, byy + bhh / 2), ms=5, lw=0.8)
            bx += w + g
        ends.append((px + pw - 0.0, py + 0.045 + bhh / 2)); arrow(ax, (bus_x + bus_w + 0.005, py + ph / 2), (px - 0.005, py + ph / 2))
    put(ax, px + pw / 2, pys[2] - 0.095, "setting A: fusion branch beside a linear branch; setting B: hybrid layer", 6.5, "italic", INK2, width=pw)
    dx = px + pw + 0.2; dw = W - dx - 0.04
    box(ax, dx, pys[2], dw, pys[0] + ph - pys[2], "Decision", "at threshold 0.5, same rule for all three tracks", fs=6.5, tfs=7.0, pad=0.05)
    for (ex, ey) in ends: arrow(ax, (ex + 0.003, ey), (dx - 0.004, ey), lw=0.8)
    ey0 = 0.02; eh = 0.46; cam_w = 3.55
    box(ax, tx, ey0, cam_w, eh, "CAM and attribution methods on the last feature map", "Grad-CAM, Grad-CAM++, Eigen-CAM, Score-CAM, Occlusion, Integrated Gradients", ec=GREEN, fc=TINT[GREEN], lw=1.1, fs=6.5, tfs=6.9, ls=(0, (4, 2)), pad=0.05)
    hm_x = tx + cam_w + 0.3; hm_w = 1.75
    box(ax, hm_x, ey0, hm_w, eh, "Heat map", "is not clinical evidence", ec=GREEN, fc=TINT[GREEN], lw=1.1, fs=6.5, tfs=6.9, ls=(0, (4, 2)), pad=0.05)
    arrow(ax, (tx + cam_w + 0.005, ey0 + eh / 2), (hm_x - 0.005, ey0 + eh / 2), color=GREEN, ls=(0, (4, 2)), lw=1.0)
    ex_mid = ys[2] + th / 2
    line(ax, [(tx - 0.004, ex_mid), (gut / 2 + 0.01, ex_mid), (gut / 2 + 0.01, ey0 + eh / 2)], color=GREEN, ls=(0, (4, 2)), lw=1.0)
    arrow(ax, (gut / 2 + 0.01, ey0 + eh / 2), (tx - 0.005, ey0 + eh / 2), color=GREEN, ls=(0, (4, 2)), lw=1.0)
    save(fig, "fig02_architecture")

if __name__ == "__main__":
    fig1(); fig2()
    print("fit problems:", len(PROBLEMS)); [print("  ", p) for p in PROBLEMS]
