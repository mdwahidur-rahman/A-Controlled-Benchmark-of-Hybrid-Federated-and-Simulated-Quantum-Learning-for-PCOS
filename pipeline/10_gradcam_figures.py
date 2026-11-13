#!/usr/bin/env python3
"""Figures 5-7 and results/tables/gradcam_faithfulness.csv, from results/gradcam/{display.npz,faith_items.json} only."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.cm as cm
from matplotlib.colors import Normalize
from figstyle import *
setup()
ROOT = os.environ.get("PCOS_BENCH_ROOT", os.getcwd()); G = f"{ROOT}/results/gradcam"
D = np.load(f"{G}/display.npz", allow_pickle=True); items = list(json.load(open(f"{G}/faith_items.json")).values())
imgs, labels, probs = D["imgs"], D["labels"], D["probs"]
def overlay(img, cam, a=0.5):
    hm = plt.get_cmap("jet")(cam)[..., :3]; return np.clip((1 - a) * img / 255.0 + a * hm, 0, 1)
CLS = {1: "PCOS-positive", 0: "PCOS-negative"}

# ---------- Fig 5: the paper's layout (original | Grad-CAM overlay; rows = four examples; letters (a)-(h))
fig, axs = plt.subplots(4, 2, figsize=(6.9, 12.2)); fig.subplots_adjust(hspace=0.42, wspace=0.05, left=0.02, right=0.88, top=0.98, bottom=0.02)
letters = "abcdefgh"
for r in range(4):
    for c in range(2):
        ax = axs[r, c]; ax.axis("off")
        ax.imshow(imgs[r] if c == 0 else overlay(imgs[r], D["gradcam"][r]), interpolation="bicubic")
        k = letters[2 * r + c]; cls = CLS[int(labels[r])]
        cap = f"({k}) {cls}" if c == 0 else f"({k}) Grad-CAM with {cls}"
        ax.text(0.5, -0.035, cap, transform=ax.transAxes, ha="center", va="top", fontsize=14, fontweight="bold", color=INK)
        if c == 1: ax.text(0.5, -0.125, f"head probability {probs[r]:.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=12.5, color=INK2)
cax = fig.add_axes([0.915, 0.30, 0.022, 0.40]); cb = fig.colorbar(cm.ScalarMappable(Normalize(0, 1), cmap="jet"), cax=cax)
cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["low", "mid", "high"]); cb.ax.tick_params(labelsize=13, length=4, width=1.1)
cb.outline.set_edgecolor(AXIS); cb.set_label("Relative class evidence", fontsize=13, fontweight="bold", labelpad=8)
save(fig, "fig5_gradcam_paper_format", ROOT)

# ---------- Fig 6: three CAM methods on the same four images
fig, axs = plt.subplots(4, 4, figsize=(11.2, 11.6)); fig.subplots_adjust(hspace=0.12, wspace=0.04, left=0.115, right=0.985, top=0.93, bottom=0.01)
cols = [("Original", None), ("Grad-CAM", "gradcam"), ("Grad-CAM++", "gradcampp"), ("Eigen-CAM", "eigencam")]
for r in range(4):
    for c, (nm, key) in enumerate(cols):
        ax = axs[r, c]; ax.axis("off"); ax.imshow(imgs[r] if key is None else overlay(imgs[r], D[key][r]), interpolation="bicubic")
        if r == 0: ax.set_title(nm, fontsize=15, fontweight="bold", color=INK, loc="center", pad=8)
    axs[r, 0].text(-0.05, 0.5, f"{CLS[int(labels[r])]}\nP = {probs[r]:.3f}", transform=axs[r, 0].transAxes, rotation=90, ha="right", va="center", fontsize=13.5, fontweight="bold", color=INK, linespacing=1.6)
save(fig, "fig6_cam_methods", ROOT)

# ---------- faithfulness table + Fig 7
rng = np.random.RandomState(0); B = 4000
def ci(v):
    v = np.asarray(v); bs = rng.choice(v, (B, len(v)), replace=True).mean(1); return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
M = [("gradcam", "Grad-CAM"), ("gradcampp", "Grad-CAM++"), ("eigencam", "Eigen-CAM"), ("random_cell", "Random 7×7 blocks"), ("random", "Random pixels")]
rows = []
for cl in (1, 0):
    it = [v for v in items if v["label"] == cl]
    for k, nm in M:
        for met in ("del", "ins"):
            v = [x[k][met] for x in it]; m, lo, hi = ci(v)
            pd_ = [x[k][met] - x["random_cell"][met] for x in it] if k != "random_cell" else [0.0] * len(it)
            dm, dlo, dhi = ci(pd_)
            rows.append(dict(cls=CLS[cl], method=nm, key=k, metric={"del": "deletion AUC (lower is better)", "ins": "insertion AUC (higher is better)"}[met], n=len(it), mean=m, lo=lo, hi=hi,
                             diff_vs_block_random=dm, diff_lo=dlo, diff_hi=dhi))
    for k, nm in M[:3]:
        v = [x[k]["border_mass"] for x in it]; m, lo, hi = ci(v)
        rows.append(dict(cls=CLS[cl], method=nm, key=k, metric="saliency mass in outer 10% border ring", n=len(it), mean=m, lo=lo, hi=hi))
    rows.append(dict(cls=CLS[cl], method="Constant (degenerate) maps", key="const", metric="count", n=len(it), mean=float(sum(x[k]["const"] for x in it for k in ("gradcam", "gradcampp", "eigencam")))))
tab = pd.DataFrame(rows); tab.to_csv(f"{ROOT}/results/tables/gradcam_faithfulness.csv", index=False)

COL = {"gradcam": BLUE, "gradcampp": ORANGE, "eigencam": AQUA, "random_cell": INK, "random": INK2}
MK = {"gradcam": "o", "gradcampp": "s", "eigencam": "^", "random_cell": "D", "random": "X"}
fig = plt.figure(figsize=(12.6, 6.9)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8], wspace=0.10, left=0.19, right=0.985, top=0.88, bottom=0.14)
rowspec = []; y = 0
for cl in (1, 0):
    for k, nm in M: rowspec.append((CLS[cl], k, nm, y)); y += 1
    y += 0.8
yt = [r[3] for r in rowspec]
def panel_dotplot(ax, metric, xl, ref=None):
    for cls, k, nm, y in rowspec:
        t = tab[(tab.cls == cls) & (tab.key == k) & (tab.metric == metric)].iloc[0]
        ax.plot([t.lo, t.hi], [y, y], color=COL[k], lw=2.2, solid_capstyle="round", zorder=2)
        ax.plot([t["mean"]], [y], marker=MK[k], ms=10, mfc=COL[k], mec="white", mew=1.2, ls="none", zorder=3)
        ax.text(t.hi + 0.012, y, f"{t['mean']:.2f}", va="center", ha="left", fontsize=12, fontweight="bold")
    ax.set_ylim(max(yt) + 0.7, -0.9); ax.grid(True, axis="x"); ax.set_xlabel(xl)
ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharey=ax1); ax3 = fig.add_subplot(gs[2], sharey=ax1)
panel_dotplot(ax1, "deletion AUC (lower is better)", "Deletion AUC  (lower = more faithful)")
panel_dotplot(ax2, "insertion AUC (higher is better)", "Insertion AUC  (higher = more faithful)")
ax1.set_yticks(yt); ax1.set_yticklabels([r[2] for r in rowspec]); ax2.tick_params(labelleft=False); ax3.tick_params(labelleft=False)
for ax in (ax1, ax2): ax.set_xlim(0.05, 1.14); ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
ring = 1 - (1 - 2 * 22 / 224) ** 2
for cls, k, nm, y in rowspec:
    t = tab[(tab.cls == cls) & (tab.key == k) & (tab.metric == "saliency mass in outer 10% border ring")] if k in ("gradcam", "gradcampp", "eigencam") else []
    if len(t):
        t = t.iloc[0]; ax3.plot([t.lo, t.hi], [y, y], color=COL[k], lw=2.2, solid_capstyle="round", zorder=2)
        ax3.plot([t["mean"]], [y], marker=MK[k], ms=10, mfc=COL[k], mec="white", mew=1.2, ls="none", zorder=3); ax3.text(max(t.hi, ring) + 0.012, y, f"{t['mean']:.2f}", va="center", ha="left", fontsize=12, fontweight="bold")
ax3.axvline(ring, color=INK, lw=1.8, zorder=1); ax3.text(ring + 0.01, -0.62, "uniform 0.35", fontsize=12, fontweight="bold", ha="left", va="center")
ax3.set_xlim(0.1, 0.56); ax3.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5]); ax3.grid(True, axis="x"); ax3.set_xlabel("Saliency in outer 10% ring")
ax3.set_ylim(max(yt) + 0.7, -0.9)
for ax in (ax1, ax2, ax3): ax.tick_params(axis="y", length=0)
for cl, y0 in ((1, -0.55), (0, 5.8 - 0.55)):
    ax1.text(-0.02, y0 - 0.0, "", transform=ax1.get_yaxis_transform())
fig.canvas.draw()
for cl, y0 in ((1, 0), (0, 5.8)):
    fig.text(0.02, ax1.transData.transform((0.3, y0 - 0.62))[1] / fig.bbox.height, CLS[cl], fontsize=14, fontweight="bold", ha="left", va="center", color=INK)
for ax, (L, t_) in zip((ax1, ax2, ax3), (("a", "Deletion"), ("b", "Insertion"), ("c", "Border mass"))):
    x0 = ax.get_position().x0 + (0 if ax is not ax1 else -0.0); fig.text(x0, ax.get_position().y1 + 0.02, L, fontsize=17, fontweight="bold", ha="left", va="bottom")
    fig.text(x0 + 0.022, ax.get_position().y1 + 0.0215, t_, fontsize=14, fontweight="bold", ha="left", va="bottom")
save(fig, "fig7_cam_faithfulness", ROOT)
pd.set_option("display.width", 250); print(tab.round(3).to_string())
