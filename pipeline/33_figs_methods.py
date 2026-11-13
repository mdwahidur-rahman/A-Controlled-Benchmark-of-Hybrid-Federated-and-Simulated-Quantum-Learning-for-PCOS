#!/usr/bin/env python3
"""Manuscript figures for Sections 3 to 5 that are drawn from the raw images, the stored explanation maps and the literature table.
  fig_preproc      : (a) four ultrasound images through the three preprocessing variants with intensity histograms, (b) class-mean histograms
  fig_xai_classical / fig_xai_federated / fig_xai_quantum : explanation grids per track (4 images x 6 methods), written as JPEG at print size
  fig_lit_bars     : accuracies printed in earlier PCOS ultrasound reports next to this study (not comparable; no ranking implied)
Inputs: data/manifest.csv, the raw images, results/xai_arms/display_*.npz.  Outputs: OUT/*.pdf|jpg and results/tables/preproc_stats.csv
Usage: PCOS_BENCH_ROOT=<root> PCOS_RAW=<raw dir> OUT=<figure dir> python3 33_figs_methods.py"""
import os, io, glob, json, sys
import numpy as np, pandas as pd, cv2
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from PIL import Image

B = os.environ.get("PCOS_BENCH_ROOT", os.getcwd()); RAW = os.environ.get("PCOS_RAW", "data/raw/PCOS"); OUT = os.environ.get("OUT", "figures")
os.makedirs(OUT, exist_ok=True)
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#7b5fc4"
mpl.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 7, "text.color": INK, "axes.edgecolor": AXIS,
                     "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK, "ytick.color": INK,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.labelsize": 7, "axes.labelcolor": INK, "pdf.fonttype": 42,
                     "savefig.facecolor": "white", "figure.facecolor": "white", "legend.frameon": False, "legend.fontsize": 6.5})
man = pd.read_csv(f"{B}/data/manifest.csv"); S = 224

# ---------------------------------------------------------------- the three preprocessing variants (identical to 04_extract.py)
def v_plain(im, s=S): return np.asarray(im.convert("RGB").resize((s, s), Image.BICUBIC))
def v_clahe(im, s=S):
    g = np.asarray(im.convert("RGB").convert("L")); c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    c = cv2.resize(c, (s, s), interpolation=cv2.INTER_CUBIC); return np.stack([c] * 3, -1)
def v_neutral(im, s=S):
    g = im.convert("RGB").convert("L"); w, h = g.size; k = min(w, h); l, t = (w - k) // 2, (h - k) // 2
    g = g.crop((l, t, l + k, t + k)).resize((128, 128), Image.LANCZOS); buf = io.BytesIO(); g.save(buf, "JPEG", quality=75)
    g = Image.open(buf).convert("L"); return np.stack([np.asarray(g.resize((s, s), Image.BICUBIC))] * 3, -1)
def gray(a): return np.asarray(Image.fromarray(a).convert("L")) if a.ndim == 3 else a
def hist(a, bins=32):
    h, _ = np.histogram(a.ravel(), bins=bins, range=(0, 256)); return h / h.sum()
def save_pdf(fig, name): fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.04); plt.close(fig)
def save_jpg(fig, name, dpi=230, q=88):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.03); plt.close(fig); buf.seek(0)
    Image.open(buf).convert("RGB").save(f"{OUT}/{name}.jpg", quality=q, optimize=True)

# ================================================================ Figure: preprocessing grid and histograms
def fig_preproc():
    rows = [4918, 5637, 7111, 7159]; CL = {1: "PCOS-positive", 0: "PCOS-negative"}
    names = ["Original file", "Plain", "CLAHE", "Shortcut-suppressed"]; cols = [INK2, BLUE, ORANGE, AQUA]
    fig = plt.figure(figsize=(7.0, 7.1))
    gs = fig.add_gridspec(4, 4, left=0.085, right=0.700, top=0.955, bottom=0.385, wspace=0.06, hspace=0.20)
    gsh = fig.add_gridspec(4, 1, left=0.795, right=0.995, top=0.955, bottom=0.385, hspace=0.20)
    for i, r in enumerate(rows):
        im = Image.open(f"{RAW}/{man.path[r]}").convert("RGB"); raw = np.asarray(im)
        vs = [raw, v_plain(im), v_clahe(im), v_neutral(im)]
        for c in range(4):
            ax = fig.add_subplot(gs[i, c]); ax.imshow(vs[c], interpolation="bicubic"); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(True); s.set_color(AXIS)
            if i == 0: ax.set_title(names[c], fontsize=7.5, fontweight="bold", pad=3)
            if c == 0:
                ax.text(-0.07, 0.5, f"{CL[int(man.label[r])]}\n{os.path.basename(man.path[r])}", transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=6.5, fontweight="bold", linespacing=1.4)
                ax.set_xlabel(f"{raw.shape[1]}×{raw.shape[0]} px, {man.bytes[r] / 1000:.1f} KB", fontsize=6, labelpad=2)
            elif c == 3 and i == 0: pass
        ax = fig.add_subplot(gsh[i, 0])
        for c in range(4):
            g = gray(vs[c]); h = hist(g, 32); ax.step(np.arange(32) * 8, h, where="post", color=cols[c], lw=1.1 if c else 1.3, label=None)
        ax.set_yscale("log"); ax.set_xlim(0, 256); ax.set_ylim(1e-4, 1.0); ax.set_xticks([0, 64, 128, 192, 255])
        if i == 0: ax.set_title("Intensity histogram", fontsize=7.5, fontweight="bold", pad=3)
        if i < 3: ax.set_xticklabels([])
        else: ax.set_xlabel("Gray level (bins of 8)", fontsize=6.5)
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_ylabel("Pixel fraction" if i == 1 else "", fontsize=6.5)
    fig.text(0.004, 0.975, "(a)", fontsize=9, fontweight="bold", va="center")
    # ---- (b) class-mean histograms over a random sample
    rs = np.random.RandomState(2026); stats = []; H = {}
    for lab in (1, 0):
        idx = man.index[man.label == lab].values; pick = rs.choice(idx, 200, replace=False); acc = {k: [] for k in names}; m_ = {k: [] for k in names}; s_ = {k: [] for k in names}
        for r in pick:
            im = Image.open(f"{RAW}/{man.path[r]}").convert("RGB"); vs = [np.asarray(im), v_plain(im), v_clahe(im), v_neutral(im)]
            for n, a in zip(names, vs):
                g = gray(a); acc[n].append(hist(g, 32)); m_[n].append(float(g.mean())); s_[n].append(float(g.std()))
        H[lab] = {n: np.mean(acc[n], 0) for n in names}
        for n in names: stats.append(dict(cls=CL[lab], variant=n, n_images=200, mean_intensity=np.mean(m_[n]), mean_within_image_std=np.mean(s_[n]), share_dark_bin0=float(H[lab][n][0])))
    pd.DataFrame(stats).to_csv(f"{B}/results/tables/preproc_stats.csv", index=False)
    gsb = fig.add_gridspec(1, 2, left=0.085, right=0.995, top=0.285, bottom=0.075, wspace=0.22)
    for k, lab in enumerate((1, 0)):
        ax = fig.add_subplot(gsb[0, k])
        for c, n in enumerate(names): ax.step(np.arange(32) * 8, H[lab][n], where="post", color=cols[c], lw=1.4, label=n)
        ax.set_yscale("log"); ax.set_xlim(0, 256); ax.set_ylim(1e-4, 1.0); ax.set_xticks([0, 64, 128, 192, 255]); ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_xlabel("Gray level (bins of 8)", fontsize=7); ax.set_ylabel("Mean pixel fraction" if k == 0 else "", fontsize=7)
        ax.set_title(f"{CL[lab]} (200 random images)", fontsize=7.5, fontweight="bold", loc="left", pad=4)
        if k == 0: ax.legend(loc="upper right", fontsize=6.2, handlelength=1.4)
    fig.text(0.004, 0.32, "(b)", fontsize=9, fontweight="bold", va="center")
    save_jpg(fig, "fig_preproc", dpi=230, q=90); print("fig_preproc written")
    print(pd.DataFrame(stats).round(3).to_string(index=False))

# ================================================================ Figures: explanation grids per track (manuscript scale)
METH = [("gradcam", "Grad-CAM"), ("gradcampp", "Grad-CAM++"), ("eigencam", "Eigen-CAM"), ("scorecam", "Score-CAM"), ("occlusion", "Occlusion"), ("ig", "Integrated\nGradients")]
CLS = {1: "PCOS-positive", 0: "PCOS-negative"}
def overlay(img, cam, a=0.5):
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-12); hm = plt.get_cmap("jet")(cam)[..., :3]; return np.clip((1 - a) * img / 255.0 + a * hm, 0, 1)
def load_display():
    D = []
    for f in glob.glob(f"{B}/results/xai_arms/display_*.npz"):
        z = np.load(f, allow_pickle=True)
        D.append(dict(row=int(f.split("_")[-1][:-4]), label=int(z["label"]), img=z["img"], plive=json.loads(str(z["plive"])), maps={k: z[k].astype(np.float32) for k in z.files if "__" in k}))
    D.sort(key=lambda d: (-d["label"], d["row"])); return D
def xai_grid(D, arms, titles, name):
    n = len(D) * len(arms); cw = 0.86; lw_ = 0.55; W = lw_ + 7 * cw + 0.45; Hh = n * cw + 0.38
    fig, axs = plt.subplots(n, 7, figsize=(W, Hh)); fig.subplots_adjust(left=lw_ / W, right=(lw_ + 7 * cw) / W, top=1 - 0.30 / Hh, bottom=0.002, hspace=0.05, wspace=0.03)
    for ai, arm in enumerate(arms):
        for ii, d in enumerate(D):
            r = ai * len(D) + ii
            for c in range(7):
                ax = axs[r, c]; ax.axis("off")
                ax.imshow(d["img"] if c == 0 else overlay(d["img"], d["maps"][f"{arm}__{METH[c - 1][0]}"]), interpolation="bicubic")
                if r == 0: ax.set_title("Original" if c == 0 else METH[c - 1][1], fontsize=6.6, fontweight="bold", pad=2.5, linespacing=1.05)
            axs[r, 0].text(-0.07, 0.5, f"{CLS[d['label']]}\nP = {d['plive'][arm]:.3f}", transform=axs[r, 0].transAxes, rotation=90, ha="right", va="center", fontsize=5.6, fontweight="bold", linespacing=1.3)
        if len(arms) > 1:
            r0 = ai * len(D); y1 = axs[r0, 0].get_position().y1; y2 = axs[r0 + len(D) - 1, 0].get_position().y0
            fig.text(0.012, (y1 + y2) / 2, titles[ai], rotation=90, ha="center", va="center", fontsize=6.4, fontweight="bold")
    cax = fig.add_axes([(lw_ + 7 * cw + 0.12) / W, 0.3, 0.012, 0.4]); cb = fig.colorbar(cm.ScalarMappable(Normalize(0, 1), cmap="jet"), cax=cax)
    cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["low", "mid", "high"]); cb.ax.tick_params(labelsize=5.8, length=2); cb.outline.set_edgecolor(AXIS)
    cb.set_label("Relative class evidence", fontsize=6, fontweight="bold", labelpad=3); save_jpg(fig, name, dpi=230, q=88); print(name, "written", round(W, 2), "x", round(Hh, 2), "in")

# ================================================================ Figure: accuracies printed in earlier reports
def fig_lit():
    # Headline accuracies as printed in the sources (abstract level; see the literature table of the manuscript). Study-side values come from the stored result tables.
    lit = [("Suha and Islam 2022", 99.89, ""), ("Hosain et al. 2022", 98.12, ""), ("Kaur et al. 2023", 99.48, ""), ("Tiwari and Maheshwari 2023", 99.70, ""),
           ("Bhatnagar 2025", 84.00, "validation accuracy"), ("Kumari and Walgampaya 2025", 99.89, "5-fold CV"), ("Pandey et al. 2025, dataset 1", 98.25, ""),
           ("Pandey et al. 2025, dataset 2", 94.56, ""), ("Borna et al. 2025", 76.20, "3 classes"), ("Srikanth et al. 2026", 95.31, "")]
    mr = pd.read_csv(f"{B}/results/tables/main_results.csv"); bm = pd.read_csv(f"{B}/results/tables/baselines_metadata_summary.csv")
    p = mr[(mr.variant == "plain") & (mr.rule == "group_g0")]
    best, mean_ = 100 * p.acc.max(), 100 * p.acc.mean()
    lr = 100 * float(bm[(bm.model == "meta_size_only_LR") & (bm.rule == "group_g0")].acc_mean.iloc[0]); rf = 100 * float(bm[(bm.model == "meta_size_only_RF") & (bm.rule == "group_g0")].acc_mean.iloc[0])
    ours = [("This study: best encoder (plain, g0 split)", best, ""), ("This study: mean of 11 encoders", mean_, ""),
            ("No-pixel reference: width, height, aspect ratio, LR", lr, "no image content"), ("No-pixel reference: width, height, aspect ratio, RF", rf, "no image content")]
    rows = [(n, v, t, "lit") for n, v, t in lit] + [None] + [(n, v, t, "ours" if "This study" in n else "ref") for n, v, t in ours]
    LIT, REF = "#9db7d9", ORANGE
    fig, ax = plt.subplots(figsize=(7.0, 4.55)); fig.subplots_adjust(left=0.40, right=0.965, top=0.915, bottom=0.15)
    y = 0; ys = []; 
    for r in rows:
        if r is None: y += 0.55; continue
        n, v, t, k = r; c = {"lit": LIT, "ours": BLUE, "ref": REF}[k]
        ax.barh(y, v, height=0.68, color=c, ec="white", lw=0.6, hatch="////" if k == "ref" else None, zorder=3)
        ax.text(v + 0.8, y, f"{v:.2f}" + (f"  ({t})" if t else ""), va="center", ha="left", fontsize=6.4, color=INK, zorder=5, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))
        ys.append((y, n)); y += 1
    ax.set_yticks([a for a, _ in ys]); ax.set_yticklabels([b for _, b in ys], fontsize=6.6); ax.invert_yaxis()
    ax.set_xlim(0, 130); ax.set_xticks([0, 20, 40, 60, 80, 100]); ax.set_xlabel("Headline accuracy as printed or measured (%)", fontsize=7, fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0); ax.axvline(100, color=AXIS, lw=0.7, ls=(0, (3, 2)), zorder=2); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=LIT, label="Earlier PCOS ultrasound reports (printed headline)"), Patch(fc=BLUE, label="This study (duplicate-group split)"), Patch(fc=REF, hatch="////", ec="white", label="This study, no-pixel reference")],
              loc="upper center", bbox_to_anchor=(0.36, 1.085), ncol=3, fontsize=6.2, handlelength=1.3, columnspacing=1.2, frameon=False)
    fig.text(0.40 + (0.965 - 0.40) / 2, 0.012, "Studies differ in data, splits and metrics; the chart is not a ranking.", ha="center", va="bottom", fontsize=6.4, fontstyle="italic", color=INK2)
    save_jpg(fig, "fig_lit_bars", dpi=230, q=90); print("fig_lit_bars written", round(best, 2), round(mean_, 2), round(lr, 2), round(rf, 2))
if __name__ == "__main__":
    what = sys.argv[1:] or ["preproc", "xai", "lit"]
    if "preproc" in what: fig_preproc()
    if "xai" in what:
        D = load_display()
        xai_grid(D, ["classical"], [""], "fig_xai_classical"); xai_grid(D, ["fl", "fl_dp"], ["FedAvg (label skew 0.5)", "FedAvg + DP (σ = 1.2)"], "fig_xai_federated"); xai_grid(D, ["quantum"], [""], "fig_xai_quantum")
    if "lit" in what: fig_lit()
