#!/usr/bin/env python3
"""Figures 15-18: multi-method XAI per arm (classical, federated, federated+DP, quantum) in the paper's overlay format, and the faithfulness /
agreement summary. Inputs: results/xai_arms/display_*.npz (four items, two per class) and results/tables/xai_*.csv."""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D
from figstyle import *
setup()
ROOT = os.environ.get("PCOS_BENCH_ROOT", os.getcwd()); T = f"{ROOT}/results/tables"; os.makedirs(f"{ROOT}/figures/preview", exist_ok=True)
PURPLE = "#7b5fc4"; GRAY = "#8a8981"; BLUES = LinearSegmentedColormap.from_list("b", ["#eaf2fc", BLUE, "#0f3f7a"])
D = []
for f in glob.glob(f"{ROOT}/results/xai_arms/display_*.npz"):
    z = np.load(f, allow_pickle=True); D.append(dict(row=int(f.split("_")[-1][:-4]), label=int(z["label"]), img=z["img"], plive=json.loads(str(z["plive"])), maps={k: z[k].astype(np.float32) for k in z.files if "__" in k}))
D.sort(key=lambda d: (-d["label"], d["row"])); CLS = {1: "PCOS-positive", 0: "PCOS-negative"}
METH = [("gradcam", "Grad-CAM"), ("gradcampp", "Grad-CAM++"), ("eigencam", "Eigen-CAM"), ("scorecam", "Score-CAM"), ("occlusion", "Occlusion"), ("ig", "Integrated\nGradients")]
ARMN = {"classical": "Classical (tuned MLP head)", "fl": "Federated (FedAvg, label skew α=0.5)", "fl_dp": "Federated + DP (σ=1.2)", "quantum": "Quantum (tuned VQC layer)"}
def overlay(img, cam, a=0.5):
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-12); hm = plt.get_cmap("jet")(cam)[..., :3]; return np.clip((1 - a) * img / 255.0 + a * hm, 0, 1)
def grid(arms, name, suptitle_note):
    nr = len(arms) * len(D); cell = 1.9; fig, axs = plt.subplots(nr, 7, figsize=(7 * cell + 2.0, nr * cell + 0.9)); fig.subplots_adjust(hspace=0.06, wspace=0.03, left=0.145, right=0.935, top=1 - 0.55 / (nr * cell + 0.9), bottom=0.005)
    for ai, arm in enumerate(arms):
        for ii, d in enumerate(D):
            r = ai * len(D) + ii
            for c in range(7):
                ax = axs[r, c]; ax.axis("off")
                if c == 0: ax.imshow(d["img"], interpolation="bicubic")
                else: ax.imshow(overlay(d["img"], d["maps"][f"{arm}__{METH[c - 1][0]}"]), interpolation="bicubic")
                if r == 0: ax.set_title("Original" if c == 0 else METH[c - 1][1], fontsize=14, fontweight="bold", color=INK, pad=6)
            axs[r, 0].text(-0.06, 0.5, f"{CLS[d['label']]}\nP = {d['plive'][arm]:.3f}", transform=axs[r, 0].transAxes, rotation=90, ha="right", va="center", fontsize=12, fontweight="bold", color=INK, linespacing=1.5)
        if len(arms) > 1:
            r0 = ai * len(D); y1 = axs[r0, 0].get_position().y1; y2 = axs[r0 + len(D) - 1, 0].get_position().y0
            fig.text(0.012, (y1 + y2) / 2, ARMN[arm], rotation=90, ha="center", va="center", fontsize=14, fontweight="bold", color=INK)
    cax = fig.add_axes([0.955, 0.3, 0.014, 0.4]); cb = fig.colorbar(cm.ScalarMappable(Normalize(0, 1), cmap="jet"), cax=cax); cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["low", "mid", "high"]); cb.ax.tick_params(labelsize=12, length=3)
    cb.outline.set_edgecolor(AXIS); cb.set_label("Relative class evidence", fontsize=12.5, fontweight="bold", labelpad=6); save(fig, name, ROOT)
grid(["classical"], "fig15_xai_classical", ""); grid(["fl", "fl_dp"], "fig16_xai_federated", ""); grid(["quantum"], "fig17_xai_quantum", "")

# ---------------- Fig 18: faithfulness + agreement
F = pd.read_csv(f"{T}/xai_faithfulness.csv"); A = pd.read_csv(f"{T}/xai_agreement.csv"); ARMS = ["classical", "fl", "fl_dp", "quantum"]; ACOL = {"classical": BLUE, "fl": ORANGE, "fl_dp": AQUA, "quantum": INK}; AMK = {"classical": "o", "fl": "s", "fl_dp": "^", "quantum": "D"}
SHORT = {"classical": "Classical", "fl": "Federated", "fl_dp": "Federated + DP", "quantum": "Quantum"}
fig = plt.figure(figsize=(17.5, 7.4)); gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.05], wspace=0.55, left=0.12, right=0.975, top=0.87, bottom=0.14)
for k, (col, ttl, xl) in enumerate((("ins_minus_del", "Faithfulness: insertion − deletion", "Insertion AUC − deletion AUC (higher = more faithful)"), ("ins_vs_random", "Insertion above random order", "Insertion AUC − random-pixel insertion AUC"))):
    ax = fig.add_subplot(gs[0, k]); ys = np.arange(len(METH))[::-1]; ax.axvline(0, color=AXIS, lw=1.4)
    for j, a in enumerate(ARMS):
        for y_, (m, nm) in zip(ys, METH):
            r = F[(F.arm == a) & (F.method == m)].iloc[0]; lo, hi = (r.ins_minus_del_lo, r.ins_minus_del_hi) if col == "ins_minus_del" else (r.ins_vs_random_lo, r.ins_vs_random_hi)
            dotrow(ax, y_ + (1.5 - j) * 0.17, r[col], lo, hi, color=ACOL[a], marker=AMK[a], ms=8.5)
    ax.set_yticks(ys); ax.set_yticklabels([nm.replace("\n", " ") for _, nm in METH], fontsize=12.5) if k == 0 else (ax.set_yticks(ys), ax.set_yticklabels([]))
    ax.set_xlabel(xl, fontsize=12); ax.grid(axis="x"); ax.set_ylim(-0.6, len(METH) - 0.4); heading(fig, ax, "ab"[k], ttl)
    if k == 0: ax.legend(handles=[Line2D([], [], marker=AMK[a], ls="none", ms=9, mfc=ACOL[a], mec="white", label=SHORT[a]) for a in ARMS], loc="lower right", fontsize=11.5, handletextpad=0.2, labelspacing=0.3)
ax = fig.add_subplot(gs[0, 2]); pairs = [("classical", "fl"), ("classical", "fl_dp"), ("classical", "quantum"), ("fl", "fl_dp"), ("fl", "quantum"), ("fl_dp", "quantum")]; ML = ["gradcam", "gradcampp", "eigencam", "scorecam", "occlusion", "ig"]
M = np.array([[A[(A.method == m) & (A.arm_a == a) & (A.arm_b == b)].spearman.iloc[0] for a, b in pairs] for m in ML]); ax.imshow(M, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]): ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=13, fontweight="bold", color="white" if M[i, j] > 0.5 else INK)
ab = {"classical": "Cl", "fl": "FL", "fl_dp": "DP", "quantum": "Q"}; ax.set_xticks(range(6)); ax.set_xticklabels([f"{ab[a]}–{ab[b]}" for a, b in pairs], fontsize=12.5); ax.set_yticks(range(6)); ax.set_yticklabels([nm.replace("\n", " ") for _, nm in METH], fontsize=12.5); ax.tick_params(length=0)
for sp_ in ax.spines.values(): sp_.set_visible(False)
heading(fig, ax, "c", "Do arms look at the same pixels? (Spearman)"); save(fig, "fig18_xai_faithfulness", ROOT); print("figures 15-18 saved")
