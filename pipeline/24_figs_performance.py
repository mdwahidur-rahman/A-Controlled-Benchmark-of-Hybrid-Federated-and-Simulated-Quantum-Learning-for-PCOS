#!/usr/bin/env python3
"""Figures 19-20 (performance analysis): confusion matrices and MCC / sensitivity-specificity forest, from results/tables/performance_*.csv only."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from figstyle import *
import perf_rows
setup()
ROOT = os.environ.get("PCOS_BENCH_ROOT", os.getcwd()); T = f"{ROOT}/results/tables"; os.makedirs(f"{ROOT}/figures/preview", exist_ok=True)
PURPLE = "#7b5fc4"; BLUES = LinearSegmentedColormap.from_list("b", ["#eaf2fc", BLUE, "#0f3f7a"])
m, G = perf_rows.build(T); cf = pd.read_csv(f"{T}/performance_confusion.csv"); cf = cf[cf.scope == "pooled"].set_index("row_id")
def txt_on(v, cmap=BLUES):
    r, g, b, _ = cmap(v / 100.0); return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.52 else INK
# ======================================================================= Fig 19: confusion matrices
PANELS = [("cls_main_best_plain", "Classical", "main head, best encoder"), ("cls_ens_transformer_plain", "Classical", "tuned 4-Transformer ensemble"),
          ("fl_fedavg_lab0.5", "Federated", "FedAvg, label skew 0.5"), ("fl_fedavg_lab0.1", "Federated", "FedAvg, label skew 0.1"),
          ("fl_fedavg_dp4.0_lab0.5", "Federated + DP", "FedAvg, σ = 4 (ε ≈ 3.7), skew 0.5"), ("q_A_classical", "Quantum A", "classical-only head"),
          ("q_A_vqc_pairs", "Quantum A", "+ VQC branch"), ("q_B_vqc", "Quantum B", "VQC slot, ring CNOTs")]
fig, axs = plt.subplots(2, 4, figsize=(18.5, 10.2)); plt.subplots_adjust(left=0.085, right=0.985, top=0.84, bottom=0.09, hspace=0.78, wspace=0.42)
cls = ["PCOS-\nnegative", "PCOS-\npositive"]
for ax, (rid, trk, sub) in zip(axs.ravel(), PANELS):
    c = cf.loc[rid]; neg = c.TN + c.FP; pos = c.TP + c.FN; rate = np.array([[c.TN / neg, c.FP / neg], [c.FN / pos, c.TP / pos]]) * 100; cnt = np.array([[c.TN, c.FP], [c.FN, c.TP]]); tot = np.array([[neg, neg], [pos, pos]])
    ax.imshow(rate, cmap=BLUES, vmin=0, vmax=100, aspect="equal")
    for i in range(2):
        for j in range(2):
            col = txt_on(rate[i, j]); fmt = (lambda v: f"{v:,.1f}" if v < 100 else f"{v:,.0f}")
            ax.text(j, i - 0.08, f"{rate[i, j]:.1f}%", ha="center", va="center", fontsize=23, fontweight="bold", color=col)
            ax.text(j, i + 0.27, f"{fmt(cnt[i, j])} of {tot[i, j]:,.0f}", ha="center", va="center", fontsize=12.5, color=col)
    ax.set_xticks([0, 1]); ax.set_xticklabels(cls, fontsize=13); ax.set_yticks([0, 1]); ax.set_yticklabels(cls, fontsize=13, rotation=90, va="center")
    ax.set_xlabel("Predicted class", fontsize=13.5); ax.set_ylabel("True class", fontsize=13.5)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0); ax.set_xticks(np.arange(-.5, 2, 1), minor=True); ax.set_yticks(np.arange(-.5, 2, 1), minor=True); ax.grid(which="minor", color="white", lw=3); ax.grid(which="major", visible=False); ax.tick_params(which="minor", length=0)
    ax.text(0.0, 1.20, trk, transform=ax.transAxes, fontsize=15, fontweight="bold", color=INK, va="bottom"); ax.text(0.0, 1.11, sub, transform=ax.transAxes, fontsize=13, color=INK, va="bottom")
    ax.text(0.0, 1.025, f"MCC {m.loc[rid, 'mcc']:.3f}   accuracy {m.loc[rid, 'acc'] * 100:.2f}%", transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK2, va="bottom")
fig.text(0.085, 0.975, "Rows are true classes (each row sums to 100%). Counts are mean test-set counts per held-out split, pooled over seeds (and encoders for federated and quantum rows).", fontsize=13, color=INK, ha="left")
save(fig, "fig19_confusion_matrices", ROOT)
# ======================================================================= Fig 20: MCC + sensitivity/specificity forest
rows = [(t, rid, lab) for t, rs in G for rid, lab in rs]; n = len(rows); ypos = []; y = 0; last = None; heads = []
for t, rid, lab in rows:
    if t != last:
        y += 0.9 if last is not None else 0; heads.append((y, t)); y += 1.0; last = t
    ypos.append(y); y += 1.0
H = y; TRK = {"Classical": BLUE, "Federated": ORANGE, "Quantum": AQUA}
fig = plt.figure(figsize=(19.5, 15.8)); gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], left=0.315, right=0.925, top=0.945, bottom=0.07, wspace=0.12)
axa = fig.add_subplot(gs[0]); axb = fig.add_subplot(gs[1], sharey=axa)
for (t, rid, lab), yy in zip(rows, ypos):
    r = m.loc[rid]; col = TRK[t.split(",")[0].split(" ")[0]]; dp = "DP" in t or "+ DP" in t
    dotrow(axa, yy, r.mcc, r.mcc_lo, r.mcc_hi, color=col, marker="o", ms=11); 
    if dp: axa.plot([r.mcc], [yy], marker="o", ms=11, mfc="white", mec=col, mew=2.4, ls="none", zorder=5)
    axa.text(0.9935, yy, f"{r.mcc:.3f}", ha="left", va="center", fontsize=12.5, fontweight="bold", color=INK, transform=axa.get_yaxis_transform() if False else axa.transData)
    dotrow(axb, yy, r.rec * 100, r.rec_lo * 100, r.rec_hi * 100, color=INK2, marker="o", ms=10)
    dotrow(axb, yy, r.spec * 100, r.spec_lo * 100, r.spec_hi * 100, color=PURPLE, marker="D", ms=9)
axa.set_yticks(ypos); axa.set_yticklabels([lab for _, _, lab in rows], fontsize=13); axa.set_ylim(H - 0.3, -0.8)
for yy, t in heads:
    for ax in (axa,): ax.axhspan(yy - 0.5, yy + 0.5, color="#f0efe9", zorder=0, lw=0)
    axa.text(0.82, yy, t, ha="left", va="center", fontsize=13.5, fontweight="bold", color=INK)
axa.set_xlim(0.82, 1.0); axa.set_xticks([0.84, 0.88, 0.92, 0.96, 1.0]); axa.set_xlabel("Matthews correlation coefficient (95% interval)"); axa.grid(axis="x")
axb.set_xlim(85, 100); axb.set_xticks([85, 90, 95, 100]); axb.set_xlabel("Sensitivity and specificity, % (95% interval)"); axb.grid(axis="x"); plt.setp(axb.get_yticklabels(), visible=False); axb.tick_params(axis="y", length=0)
for yy, t in heads: axb.axhspan(yy - 0.5, yy + 0.5, color="#f0efe9", zorder=0, lw=0)
axa.tick_params(axis="y", length=0)
for yy, (t, rid, lab) in zip(ypos, rows):
    r = m.loc[rid]; axb.text(100.4, yy, f"{r.rec * 100:.1f}", ha="left", va="center", fontsize=12.5, color=INK2, fontweight="bold", clip_on=False); axb.text(102.9, yy, f"{r.spec * 100:.1f}", ha="left", va="center", fontsize=12.5, color=PURPLE, fontweight="bold", clip_on=False)
axb.text(100.4, -0.55, "Sens.", fontsize=12.5, color=INK2, fontweight="bold", ha="left", va="bottom", clip_on=False); axb.text(102.9, -0.55, "Spec.", fontsize=12.5, color=PURPLE, fontweight="bold", ha="left", va="bottom", clip_on=False)
axa.text(0.9935, -0.55, "MCC", fontsize=12.5, fontweight="bold", ha="left", va="bottom", color=INK, clip_on=False)
handles = [Line2D([], [], marker="o", color=BLUE, ls="none", ms=11, label="Classical"), Line2D([], [], marker="o", color=ORANGE, ls="none", ms=11, label="Federated"), Line2D([], [], marker="o", mfc="white", mec=ORANGE, mew=2.4, ls="none", ms=11, label="Federated + DP (hollow)"),
           Line2D([], [], marker="o", color=AQUA, ls="none", ms=11, label="Quantum"), Line2D([], [], marker="o", color=INK2, ls="none", ms=10, label="Sensitivity (recall of PCOS-positive)"), Line2D([], [], marker="D", color=PURPLE, ls="none", ms=9, label="Specificity (recall of PCOS-negative)")]
fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.62, 0.995), fontsize=12.5, handletextpad=0.3, columnspacing=1.4)
fig.text(0.012, 0.947, "a", fontsize=17, fontweight="bold", ha="left", va="center", color=INK); fig.text(axb.get_position().x0 - 0.004, 0.947, "b", fontsize=17, fontweight="bold", ha="left", va="center", color=INK)
save(fig, "fig20_mcc_forest", ROOT)
print("ok")
