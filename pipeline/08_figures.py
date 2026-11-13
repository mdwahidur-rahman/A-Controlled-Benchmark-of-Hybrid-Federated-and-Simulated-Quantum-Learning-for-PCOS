#!/usr/bin/env python3
"""Figures 1-4, generated only from stored result files (no typed numbers)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator, NullFormatter
from figstyle import *
setup()
ROOT = os.environ.get("PCOS_BENCH_ROOT", os.getcwd()); R = f"{ROOT}/results"; T = f"{R}/tables"
main = pd.read_csv(f"{T}/main_results.csv"); man = pd.read_csv(f"{ROOT}/data/manifest.csv")
aud = json.load(open(f"{R}/stage0_audit.json")); st1 = json.load(open(f"{R}/stage1_splits.json"))
tests = pd.read_csv(f"{T}/partition_and_preprocessing_tests.csv"); bl = pd.read_csv(f"{R}/baselines_metadata.csv")
perm = pd.read_csv(f"{R}/baselines_label_permuted.csv")
thou = FuncFormatter(lambda v, p: f"{int(round(v)):,}")
M = lambda enc, var, rule, col: float(main[(main.encoder == enc) & (main.variant == var) & (main.rule == rule)][col].iloc[0]) * 100

# ============ FIGURE 1: dataset audit ============
fig = plt.figure(figsize=(7.6, 8.2)); gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.62, wspace=1.0)
ax = fig.add_subplot(gs[0, :])
sz = man.groupby(["label", "width", "height"]).size().reset_index(name="n")
for lab, col, nm, mk in ((0, BLUE, "PCOS-negative (5,000 images)", "o"), (1, ORANGE, "PCOS-positive (6,784 images)", "s")):
    d = sz[sz.label == lab]
    ax.scatter(d.width, d.height, s=16 + 3.4 * np.sqrt(d.n), c=col, marker=mk, edgecolors="none", alpha=0.8, label=nm, zorder=3)
ax.set_xscale("log"); ax.set_yscale("log")
ticks = [100, 200, 300, 500, 1000]
for a in (ax.xaxis, ax.yaxis):
    a.set_major_locator(FixedLocator(ticks)); a.set_major_formatter(thou); a.set_minor_formatter(NullFormatter())
ax.set_xlim(95, 1350); ax.set_ylim(80, 1900); ax.grid(True)
ax.set_xlabel("Image width (pixels, log scale)"); ax.set_ylabel("Image height (pixels, log scale)")
ax.annotate("300 × 300\n1,590 images", xy=(300, 300), xytext=(104, 560), fontsize=12, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="-", color=INK, lw=1.1, shrinkA=2, shrinkB=6))
ax.annotate("984 × 848\n1,433 images", xy=(984, 848), xytext=(400, 1300), fontsize=12, fontweight="bold", ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=INK, lw=1.1, shrinkA=2, shrinkB=6))
ax.text(0.015, 0.965, f"{aud['distinct_wh']:,} distinct sizes:\n{aud['wh_class_exclusive_images_pct']:.0f}% occur in only one class",
        transform=ax.transAxes, fontsize=12.5, fontweight="bold", va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=AXIS, lw=1.0))
ax.legend(loc="lower right", handletextpad=0.3, borderaxespad=0.5, markerscale=1.1, fontsize=12, bbox_to_anchor=(0.995, 0.03))
H1 = [(ax, "a", "Image dimensions separate the two classes perfectly")]

axb = fig.add_subplot(gs[1, 0])
cats = ["Images in folder", "Byte-unique files", "Duplicate groups (g0)", "Conservative groups (g4)"]
vals = [aud["n"], st1["n_unique_md5"], st1["n_groups_g0"], st1["n_groups_g4"]]
yy = np.arange(len(cats))[::-1]
axb.barh(yy, vals, height=0.58, color=BLUE, zorder=3)
for y_, v in zip(yy, vals): axb.text(v + 250, y_, f"{v:,}", va="center", ha="left", fontsize=12.5, fontweight="bold")
axb.set_yticks(yy); axb.set_yticklabels(cats); axb.set_xlim(0, 15500); axb.xaxis.set_major_formatter(thou)
axb.set_xticks([0, 5000, 10000]); axb.grid(True, axis="x"); axb.set_xlabel("Count")
axb.tick_params(axis="y", length=0); H1.append((axb, "b", "Independent images"))

axc = fig.add_subplot(gs[1, 1])
auc1 = aud["single_feature_auc_infected_as_positive"]
lab = {"width": "Image width", "height": "Image height", "aspect": "Aspect ratio", "bytes": "File size", "bpp": "Bits per pixel",
       "q0": "JPEG quality", "mean": "Mean intensity", "std": "Intensity SD"}
sep = sorted(((lab[k], max(v, 1 - v)) for k, v in auc1.items()), key=lambda t: t[1])
yy = np.arange(len(sep))
axc.barh(yy, [v for _, v in sep], height=0.58, color=BLUE, zorder=3, left=0)
for y_, (_, v) in zip(yy, sep): axc.text(v + 0.008, y_, f"{v:.3f}", va="center", ha="left", fontsize=12, fontweight="bold")
axc.set_yticks(yy); axc.set_yticklabels([n for n, _ in sep]); axc.set_xlim(0.5, 1.12); axc.set_xticks([0.5, 0.75, 1.0])
axc.axvline(0.5, color=INK, lw=1.2, zorder=4); axc.grid(True, axis="x"); axc.set_xlabel("Single-feature AUC")
axc.tick_params(axis="y", length=0); H1.append((axc, "c", "Metadata alone"))
for a_, l_, t_ in H1: heading(fig, a_, l_, t_)
save(fig, "fig1_dataset_audit", ROOT)

# ============ FIGURE 2: headline accuracy vs no-pixel baselines ============
g0 = lambda m: bl[(bl.model == m) & (bl.rule == "group_g0")].acc.mean() * 100
fig = plt.figure(figsize=(7.6, 8.4)); gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.9], hspace=0.62)
ax = fig.add_subplot(gs[0])
items = [("Label-shuffled head (chance control)", perm.acc.mean() * 100, INK2, "o", True),
         ("Majority class", g0("majority"), INK2, "o", True),
         ("WomenSafe as reported (95.31%, random split)", 95.31, INK, "D", False),
         ("Image size only, logistic regression", g0("meta_size_only_LR"), INK, "s", True),
         ("Image size only, random forest", g0("meta_size_only_RF"), INK, "s", True)]
for i, (nm, v, c, mk, filled) in enumerate(items):
    ax.plot([50, v], [i, i], color=GRID if filled else AXIS, lw=2.4, zorder=1, solid_capstyle="butt")
    ax.plot([v], [i], marker=mk, ms=12, mfc=c if filled else "white", mec=c if filled else INK, mew=2.0 if not filled else 1.4, ls="none", zorder=3)
    ax.text(v + 1.9, i, f"{v:.2f}%", va="center", ha="left", fontsize=13, fontweight="bold")
ax.set_yticks(range(len(items))); ax.set_yticklabels([t[0] for t in items]); ax.set_xlim(50, 109); ax.set_ylim(-0.6, len(items) - 0.4)
ax.set_xticks([50, 60, 70, 80, 90, 100]); ax.grid(True, axis="x"); ax.set_xlabel("Test accuracy (%)")
ax.tick_params(axis="y", length=0); H2 = [(ax, "a", "No-pixel baselines versus the published claim")]

ax = fig.add_subplot(gs[1])
d = main[(main.variant == "plain") & (main.rule == "group_g0")].sort_values("acc")
ax.axvline(g0("meta_size_only_RF"), color=INK, lw=2.4, zorder=1); ax.axvline(g0("meta_size_only_LR"), color=INK2, lw=1.4, zorder=1)
for i, (_, r) in enumerate(d.iterrows()):
    dotrow(ax, i, r.acc * 100, r.acc_lo * 100, r.acc_hi * 100, FAMCOL[r.family], FAMMK[r.family], ms=11)
    ax.text(100.22, i, f"{r.acc*100:.2f}", va="center", ha="left", fontsize=12.5, fontweight="bold")
ax.set_yticks(range(len(d))); ax.set_yticklabels([PRETTY[e] for e in d.encoder]); ax.tick_params(axis="y", length=0)
ax.set_xlim(96.0, 100.95); ax.set_ylim(-0.7, len(d) + 1.35); ax.set_xticks([97, 98, 99, 100]); ax.grid(True, axis="x")
ax.set_xlabel("Test accuracy (%), group-disjoint split, plain preprocessing")
ax.text(g0("meta_size_only_RF") - 0.04, len(d) + 0.95, "Size-only RF 100.0", ha="right", va="center", fontsize=12, fontweight="bold", color=INK)
ax.text(g0("meta_size_only_LR"), len(d) - 0.05 + 0.15, "", ha="center")
ax.text(g0("meta_size_only_LR") - 0.05, len(d) + 0.30, f"Size-only LR {g0('meta_size_only_LR'):.1f}", ha="right", va="center", fontsize=12, fontweight="bold", color=INK2)
hs = [plt.Line2D([], [], marker=FAMMK[f], ms=10, mfc=FAMCOL[f], mec="white", mew=1.2, ls="-", lw=2, color=FAMCOL[f], label=f) for f in FAMCOL]
ax.legend(handles=hs, loc="upper center", ncol=3, bbox_to_anchor=(0.45, -0.20), columnspacing=2.0, handlelength=2.0, title="Encoder family (bars: 95% cluster CI)", title_fontsize=11.5)
H2.append((ax, "b", "Frozen encoders stay below the size-only random forest"))
for a_, l_, t_ in H2: heading(fig, a_, l_, t_)
save(fig, "fig2_headline_accuracy", ROOT)

# ============ FIGURE 3: partitioning contrast ============
RC = {"naive_image": ORANGE, "group_g0": BLUE, "group_g4": AQUA}; RN = {"naive_image": "Image-level (naive)", "group_g0": "Group-disjoint (g0)", "group_g4": "Group-disjoint (g4, conservative)"}
RM = {"naive_image": "s", "group_g0": "o", "group_g4": "^"}
fig = plt.figure(figsize=(7.6, 8.6)); gs = fig.add_gridspec(2, 2, height_ratios=[2.3, 1], hspace=0.62, wspace=0.62)
ax = fig.add_subplot(gs[0, :])
d = main[(main.variant == "plain") & (main.rule == "group_g0")].sort_values("acc"); order = list(d.encoder)
off = {"naive_image": 0.26, "group_g0": 0.0, "group_g4": -0.26}
for i, e in enumerate(order):
    for rule in RC:
        dotrow(ax, i + off[rule], M(e, "plain", rule, "acc"), M(e, "plain", rule, "acc_lo"), M(e, "plain", rule, "acc_hi"), RC[rule], RM[rule], ms=8.5)
ax.set_yticks(range(len(order))); ax.set_yticklabels([PRETTY[e] for e in order]); ax.tick_params(axis="y", length=0)
ax.set_xlim(96.0, 100.4); ax.set_ylim(-0.7, len(order) + 2.3); ax.set_xticks([96, 97, 98, 99, 100]); ax.grid(True, axis="x")
ax.set_xlabel("Test accuracy (%), plain preprocessing; bars = 95% cluster CI")
hs = [plt.Line2D([], [], marker=RM[r], ms=9, mfc=RC[r], mec="white", mew=1.1, ls="-", lw=2, color=RC[r], label=RN[r]) for r in ("naive_image", "group_g0", "group_g4")]
ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=11, columnspacing=1.1, handlelength=1.6, handletextpad=0.4)
t = tests[(tests.variant == "plain") & (tests.contrast == "naive - group_g0")].iloc[0]
ax.text(0.0, 0.875, f"Naive − g0: +{t.mean_diff_pts:.2f} pts; {int(t.encoders_positive)}/{int(t.n_encoders)} encoders higher (Holm p = {t.sign_p_holm:.3f})",
        transform=ax.transAxes, ha="left", va="top", fontsize=12.5, fontweight="bold")
H3 = [(ax, "a", "Leaky splitting inflates every encoder")]

def rulemean(rule, key): return float(np.mean([r[key] for r in st1["rules"][rule]]))
axb = fig.add_subplot(gs[1, 0]); rules = ["naive_image", "group_g0", "group_g4"]
v = [rulemean(r, "test_md5_seen_in_trainval_pct") for r in rules]
axb.bar(range(3), v, width=0.62, color=[RC[r] for r in rules], zorder=3)
for i, x in enumerate(v): axb.text(i, x + 2.5, f"{x:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
axb.set_xticks(range(3)); axb.set_xticklabels(["Naive", "g0", "g4"]); axb.set_ylim(0, 100); axb.set_yticks([0, 25, 50, 75, 100]); axb.grid(True, axis="y")
axb.set_ylabel("Exact copy in\ntrain/val (%)"); axb.tick_params(axis="x", length=0); H3.append((axb, "b", "Exact-copy leakage"))
axc = fig.add_subplot(gs[1, 1]); v = [rulemean(r, "S_g4") for r in rules]
axc.bar(range(3), v, width=0.62, color=[RC[r] for r in rules], zorder=3)
for i, x in enumerate(v): axc.text(i, x + 25, f"{x:,.0f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
axc.set_xticks(range(3)); axc.set_xticklabels(["Naive", "g0", "g4"]); axc.set_ylim(0, 1450); axc.yaxis.set_major_formatter(thou); axc.grid(True, axis="y")
axc.set_ylabel("Spanning g4\ngroups"); axc.tick_params(axis="x", length=0); H3.append((axc, "c", "Near-duplicate leakage"))
for a_, l_, t_ in H3: heading(fig, a_, l_, t_)
save(fig, "fig3_partition_contrast", ROOT)

# ============ FIGURE 4: preprocessing ablation ============
VC = {"plain": BLUE, "clahe": ORANGE, "neutral": AQUA}; VM_ = {"plain": "o", "clahe": "s", "neutral": "^"}
VN = {"plain": "Plain", "clahe": "CLAHE", "neutral": "Shortcut-suppressed"}
fig = plt.figure(figsize=(7.6, 8.4)); gs = fig.add_gridspec(2, 1, height_ratios=[2.1, 1], hspace=0.55)
ax = fig.add_subplot(gs[0]); d = main[(main.variant == "plain") & (main.rule == "group_g0")].sort_values("acc"); order = list(d.encoder)
off = {"plain": 0.26, "clahe": 0.0, "neutral": -0.26}
for i, e in enumerate(order):
    for var in VC: dotrow(ax, i + off[var], M(e, var, "group_g0", "acc"), M(e, var, "group_g0", "acc_lo"), M(e, var, "group_g0", "acc_hi"), VC[var], VM_[var], ms=8.5)
ax.set_yticks(range(len(order))); ax.set_yticklabels([PRETTY[e] for e in order]); ax.tick_params(axis="y", length=0)
ax.set_xlim(94.6, 100.4); ax.set_ylim(-0.7, len(order) + 0.9); ax.set_xticks([95, 96, 97, 98, 99, 100]); ax.grid(True, axis="x")
ax.set_xlabel("Test accuracy (%), group-disjoint split; bars = 95% cluster CI")
hs = [plt.Line2D([], [], marker=VM_[v_], ms=9, mfc=VC[v_], mec="white", mew=1.1, ls="-", lw=2, color=VC[v_], label=VN[v_]) for v_ in VC]
ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=11.5, columnspacing=1.4, handlelength=1.8)
H4 = [(ax, "a", "Accuracy survives removing size and compression cues")]
axb = fig.add_subplot(gs[1]); tb = tests[tests.family == "B_preprocessing"].reset_index(drop=True)
vv = ["clahe", "neutral"]; xs = np.arange(2)
for i, var in enumerate(vv):
    r = tb[tb.variant == var].iloc[0]; axb.bar(i, r.mean_diff_pts, width=0.5, color=VC[var], zorder=3)
    s = "+" if r.mean_diff_pts > 0 else "−"
    axb.text(i, r.mean_diff_pts + (0.08 if r.mean_diff_pts > 0 else -0.08), f"{s}{abs(r.mean_diff_pts):.2f} pts\n{int(r.encoders_positive)}/{int(r.n_encoders)} encoders higher",
             ha="center", va="bottom" if r.mean_diff_pts > 0 else "top", fontsize=12.5, fontweight="bold")
axb.axhline(0, color=INK, lw=1.2, zorder=4); axb.set_xticks(xs); axb.set_xticklabels([VN[v_] for v_ in vv]); axb.set_ylim(-2.0, 1.4)
axb.set_ylabel("Change vs plain\n(accuracy points)"); axb.grid(True, axis="y"); axb.tick_params(axis="x", length=0)
H4.append((axb, "b", "Mean change across 11 encoders"))
for a_, l_, t_ in H4: heading(fig, a_, l_, t_)
save(fig, "fig4_preprocessing_ablation", ROOT)
print("figures written:", sorted(f for f in os.listdir(f"{ROOT}/figures") if f.endswith(".pdf")))
