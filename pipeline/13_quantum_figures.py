#!/usr/bin/env python3
"""Figures 8-9 (quantum slot), generated only from results/tables/quantum_*.csv."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from figstyle import *
setup()
ROOT = os.environ.get("PCOS_BENCH_ROOT", os.getcwd()); T = f"{ROOT}/results/tables"
cond = pd.read_csv(f"{T}/quantum_conditions.csv"); pool = pd.read_csv(f"{T}/quantum_pooled.csv").set_index("condition"); con = pd.read_csv(f"{T}/quantum_contrasts.csv")
NAMES = {"mobilenetv2_100": "MobileNetV2", "swin_tiny_patch4_window7_224": "Swin-T", "resnet50": "ResNet-50"}
# ---------------- Fig 8: hybrid layer (Setting B)
SL = [("B_identity", "Identity (linear)", INK2, "o"), ("B_dense_w", "Dense, width-matched", INK2, "s"), ("B_dense_p", "Dense, parameter-matched", INK2, "^"),
      ("B_vqc_noent", "Closed-form twin (no entanglers)", AQUA, "D"), ("B_vqc_pairs", "VQC, nearest-neighbor CNOTs", BLUE, "o"), ("B_vqc", "VQC, ring CNOTs", ORANGE, "o")]
fig = plt.figure(figsize=(9.6, 11.4)); gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.9], hspace=0.40, left=0.30, right=0.97, top=0.94, bottom=0.075)
ax = fig.add_subplot(gs[0])
for i, (c, nm, col, mk) in enumerate(SL):
    r = pool.loc[c]; dotrow(ax, i, r.acc, r.lo, r.hi, col, mk, ms=11); ax.text(r.hi + 0.25, i, f"{r.acc:.2f}", va="center", ha="left", fontsize=12.5, fontweight="bold")
ax.set_yticks(range(len(SL))); ax.set_yticklabels([s[1] for s in SL]); ax.set_ylim(len(SL) - 0.4, -0.6); ax.set_xlim(86, 100); ax.grid(True, axis="x")
ax.set_xlabel("Test accuracy (%), mean of 6 encoder × preprocessing combos"); ax.tick_params(axis="y", length=0); heading(fig, ax, "a", "Pooled over all six combinations")
ax = fig.add_subplot(gs[1]); combos = [(e, v) for e in NAMES for v in ("plain", "neutral")]; yt = []; yl = []; step = len(SL) + 1.3
for k, (e, v) in enumerate(combos):
    for i, (c, nm, col, mk) in enumerate(SL):
        r = cond[(cond.encoder == e) & (cond.variant == v) & (cond.condition == c)].iloc[0]; y = k * step + i * 0.82
        dotrow(ax, y, r.acc, r.acc_lo, r.acc_hi, col, mk, ms=8.5)
    yt.append(k * step + 2.05); yl.append(f"{NAMES[e]}, {'plain' if v == 'plain' else 'shortcut-suppressed'}")
ax.set_yticks(yt); ax.set_yticklabels(yl); ax.set_ylim(len(combos) * step - 1.2, -1.0); ax.set_xlim(70, 100.5); ax.grid(True, axis="x"); ax.tick_params(axis="y", length=0)
ax.set_xlabel("Test accuracy (%), group-disjoint split; bars = 95% cluster CI")
from matplotlib.lines import Line2D
hs = [Line2D([], [], marker=mk, color=col, mfc=col, mec="white", mew=1.2, ms=10, ls="-", lw=2, label=nm) for c, nm, col, mk in SL]
ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=2, fontsize=12, columnspacing=1.6, handletextpad=0.5)
heading(fig, ax, "b", "Each combination separately"); save(fig, "fig8_quantum_hybrid_layer", ROOT)

# ---------------- Fig 9: fusion + permutation controls (Setting A)
RO = [("A_classical", "Classical branch only", INK2, "o", 1), ("A_dense_w", "+ dense branch, width-matched", INK2, "s", 1), ("A_dense_p", "+ dense branch, parameter-matched", INK2, "^", 1),
      ("A_vqc_pairs", "+ VQC (NN CNOTs), matched pairing", BLUE, "o", 1), ("A_vqc_pairs_lp", "   label-preserving partner", BLUE, "D", 1), ("A_vqc_pairs_shuf", "   shuffled partner", BLUE, "s", 1), ("A_vqc_pairs_derg", "   deranged partner", BLUE, "^", 1),
      ("A_vqc", "+ VQC (ring CNOTs), matched pairing", ORANGE, "o", 1), ("A_vqc_lp", "   label-preserving partner", ORANGE, "D", 1), ("A_vqc_shuf", "   shuffled partner", ORANGE, "s", 1), ("A_vqc_derg", "   deranged partner", ORANGE, "^", 1)]
fig = plt.figure(figsize=(9.8, 11.2)); gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.38, left=0.37, right=0.965, top=0.94, bottom=0.07)
ax = fig.add_subplot(gs[0])
for i, (c, nm, col, mk, f) in enumerate(RO):
    r = pool.loc[c]; dotrow(ax, i + (0.5 if i >= 3 else 0) + (0.5 if i >= 7 else 0), r.acc, r.lo, r.hi, col, mk, ms=10); yy = i + (0.5 if i >= 3 else 0) + (0.5 if i >= 7 else 0)
    ax.text(r.hi + 0.06, yy, f"{r.acc:.2f}", va="center", ha="left", fontsize=12.5, fontweight="bold")
ys = [i + (0.5 if i >= 3 else 0) + (0.5 if i >= 7 else 0) for i in range(len(RO))]
ax.set_yticks(ys); ax.set_yticklabels([r[1] for r in RO]); ax.set_ylim(max(ys) + 0.7, -0.7); ax.set_xlim(95.6, 99.35); ax.grid(True, axis="x"); ax.tick_params(axis="y", length=0)
ax.set_xlabel("Test accuracy (%), mean of 6 combos; bars = 95% cluster CI"); heading(fig, ax, "a", "Accuracy with each partner rule")
ax = fig.add_subplot(gs[1]); pc = con[(con.encoder == "pooled") & (con.family == "A")].set_index("contrast")
CR = [("A_vqc_pairs - A_classical", "vs classical branch only", "pairs"), ("A_vqc_pairs - A_dense_w", "vs dense, width-matched", "pairs"), ("A_vqc_pairs - A_dense_p", "vs dense, parameter-matched", "pairs"),
      ("A_vqc_pairs - A_vqc_pairs_shuf", "vs shuffled partner", "pairs"), ("A_vqc_pairs - A_vqc_pairs_derg", "vs deranged partner", "pairs"), ("A_vqc_pairs - A_vqc_pairs_lp", "vs label-preserving partner", "pairs"),
      ("A_vqc - A_classical", "vs classical branch only", "ring"), ("A_vqc - A_dense_w", "vs dense, width-matched", "ring"), ("A_vqc - A_dense_p", "vs dense, parameter-matched", "ring"),
      ("A_vqc - A_vqc_shuf", "vs shuffled partner", "ring"), ("A_vqc - A_vqc_derg", "vs deranged partner", "ring"), ("A_vqc - A_vqc_lp", "vs label-preserving partner", "ring")]
ax.axvspan(-0.5, 0.5, color=GRID, alpha=0.9, zorder=0, lw=0); ax.axvline(0, color=INK, lw=1.4, zorder=1)
yy = []
for i, (c, nm, d) in enumerate(CR):
    y = i + (0.9 if i >= 6 else 0); yy.append(y); r = pc.loc[c]; col = BLUE if d == "pairs" else ORANGE; dotrow(ax, y, r.diff_pts, r.lo, r.hi, col, "o", ms=10)
    ax.text(max(r.hi, 0.5) + 0.06, y, f"{r.diff_pts:+.2f}".replace("-", "\u2212"), va="center", ha="left", fontsize=12.5, fontweight="bold")
ax.set_yticks(yy); ax.set_yticklabels([c[1] for c in CR]); ax.set_ylim(max(yy) + 0.8, -1.8); ax.set_xlim(-2.4, 1.35); ax.tick_params(axis="y", length=0); ax.grid(True, axis="x")
ax.set_xlabel("VQC branch minus control, accuracy points (95% cluster CI)")
ax.text(0, -1.25, "equivalence band \u00b10.5", ha="center", va="center", fontsize=12, fontweight="bold")
tr = ax.get_yaxis_transform()
ax.text(-0.60, -1.25, "VQC with nearest-neighbor CNOTs", transform=tr, fontsize=12.5, fontweight="bold", color=BLUE, ha="left", va="center")
ax.text(-0.60, 5.95, "VQC with ring CNOTs", transform=tr, fontsize=12.5, fontweight="bold", color=ORANGE, ha="left", va="center")
heading(fig, ax, "b", "Contrasts against each control"); save(fig, "fig9_quantum_fusion_controls", ROOT)
print("quantum figures written")
