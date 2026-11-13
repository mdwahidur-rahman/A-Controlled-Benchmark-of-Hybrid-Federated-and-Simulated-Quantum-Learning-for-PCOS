#!/usr/bin/env python3
"""Journal-scale figures for the accompanying manuscript. Every value is read from results/tables/*.csv (or the raw images);
nothing is typed in by hand. Figures are drawn at their final printed width (7.16 in for two columns) with 6.5-8 pt text, and
every figure is checked for overlapping text before it is saved (check_overlap).
Usage: PCOS_BENCH_ROOT=<repo root> OUT=<output dir> python3 pipeline/26_manuscript_figures.py [name ...]"""
import os, sys, json, glob, itertools
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from sklearn.metrics import roc_curve, precision_recall_curve

R = os.environ.get("PCOS_BENCH_ROOT", os.getcwd()); T = f"{R}/results/tables"; RAW = os.environ.get("PCOS_RAW", "data/raw/PCOS")
OUT = os.environ.get("OUT", "figures"); os.makedirs(OUT, exist_ok=True)
PREV = os.environ.get("PREV", "figures/preview"); os.makedirs(PREV, exist_ok=True)
INK, INK2, GRID, AXIS = "#0b0b0b", "#3d3c39", "#e1e0d9", "#9c9b92"
BLUE, ORANGE, AQUA, PURPLE, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#7b5fc4", "#6f6e68"
W2 = 7.0; W1 = 3.36
PRETTY = {"mobilenetv2_100": "MobileNetV2", "efficientnet_b0": "EfficientNet-B0", "densenet121": "DenseNet-121", "resnet50": "ResNet-50",
          "resnext50_32x4d": "ResNeXt-50", "pvt_v2_b2": "PVT-v2-B2", "twins_pcpvt_small": "Twins-PCPVT-S", "swin_tiny_patch4_window7_224": "Swin-T",
          "crossvit_small_240": "CrossViT-S", "mobilevit_s": "MobileViT-S", "eca_botnext26ts_256": "ECA-BotNeXt26ts"}
FAM = {"mobilenetv2_100": "CNN", "efficientnet_b0": "CNN", "densenet121": "CNN", "resnet50": "CNN", "resnext50_32x4d": "CNN", "pvt_v2_b2": "Transformer",
       "twins_pcpvt_small": "Transformer", "swin_tiny_patch4_window7_224": "Transformer", "crossvit_small_240": "Transformer", "mobilevit_s": "Hybrid", "eca_botnext26ts_256": "Hybrid"}
FCOL = {"CNN": BLUE, "Transformer": ORANGE, "Hybrid": AQUA}
mpl.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 7.5, "text.color": INK, "axes.labelsize": 7.5, "axes.labelweight": "bold",
    "axes.titlesize": 8, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 4, "axes.edgecolor": AXIS, "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.labelsize": 7, "ytick.labelsize": 7, "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 3, "ytick.major.size": 3, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white", "figure.facecolor": "white"})

def _visible_texts(fig):
    """All text artists that are actually drawn: tick labels only for ticks inside the view interval and axes that are switched on."""
    skip = set(); keep = []
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            labs = list(axis.get_majorticklabels()) + list(axis.get_minorticklabels()); skip.update(id(t) for t in labs)
            if not ax.axison: continue
            lo, hi = sorted(axis.get_view_interval()); span = hi - lo
            for tick, loc in zip(axis.get_major_ticks(), axis.get_majorticklocs()):
                if lo - 1e-9 * span <= loc <= hi + 1e-9 * span:
                    for t in (tick.label1, tick.label2):
                        if t.get_visible() and t.get_text().strip(): keep.append(t)
    for t in fig.findobj(mpl.text.Text):
        if id(t) in skip or not t.get_visible() or not t.get_text().strip(): continue
        keep.append(t)
    return keep

def check_overlap(fig, name, tol=0.5):
    """Report any two drawn text items whose rendered boxes intersect (more than tol pixels in both directions), and text outside the canvas."""
    fig.canvas.draw(); rd = fig.canvas.get_renderer(); items = []
    for t in _visible_texts(fig):
        try: bb = t.get_window_extent(rd)
        except Exception: continue
        if bb.width < 1 or bb.height < 1: continue
        items.append((t, bb))
    bad = []
    for (a, ba), (b, bb) in itertools.combinations(items, 2):
        ox = min(ba.x1, bb.x1) - max(ba.x0, bb.x0); oy = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
        if ox > tol and oy > tol: bad.append((a.get_text()[:28].replace("\n", " "), b.get_text()[:28].replace("\n", " "), round(ox, 1), round(oy, 1)))
    W, H = fig.canvas.get_width_height(); out = [t.get_text()[:28] for t, bb in items if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1]
    print(f"[overlap check] {name}: {len(items)} text items, {len(bad)} overlaps, {len(out)} outside canvas")
    for x in bad[:12]: print("   OVERLAP", x)
    for x in out[:8]: print("   OUTSIDE", x)
    return len(bad) + len(out)

def save(fig, name):
    n = check_overlap(fig, name); fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{PREV}/{name}.png", dpi=170); plt.close(fig); return n

def ptitle(ax, letter, text): ax.set_title(f"({letter}) {text}", loc="left")

# ---------------------------------------------------------------- sample grid
def fig_samples():
    man = pd.read_csv(f"{R}/data/manifest.csv"); fig, axs = plt.subplots(2, 4, figsize=(W2, 3.3), gridspec_kw=dict(hspace=0.5, wspace=0.05, left=0.005, right=0.995, top=0.995, bottom=0.085))
    for r, (cls, lab) in enumerate([("infected", "PCOS-positive"), ("noninfected", "PCOS-negative")]):
        d = man[(man.cls == cls) & (man.family == "plain")].drop_duplicates("g4").sample(4, random_state=2026)
        for c, (_, row) in enumerate(d.iterrows()):
            ax = axs[r, c]; im = Image.open(f"{RAW}/{row.path}").convert("L"); ax.imshow(np.asarray(im), cmap="gray", aspect="equal", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
            ax.text(0.5, -0.04, f"{lab}\n{cls}/{row.file}: {row.width} x {row.height} px", transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color=INK, linespacing=1.3)
    return save(fig, "fig_samples")

# ---------------------------------------------------------------- dataset audit
def fig_audit():
    a = json.load(open(f"{R}/results/stage0_audit.json")); s = json.load(open(f"{R}/results/stage1_splits.json")); man = pd.read_csv(f"{R}/data/manifest.csv")
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.3), layout="constrained", gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.1]))
    ax = axs[0]; lab = ["Images", "Byte-unique\nfiles", "Duplicate groups\n(g0)", "Near-duplicate\ngroups (g4)"]; val = [s["n_images"], s["n_unique_md5"], s["n_groups_g0"], s["n_groups_g4"]]
    y = np.arange(4)[::-1]; ax.barh(y, val, color=[GRAY, BLUE, ORANGE, AQUA], height=0.62)
    for yy, v in zip(y, val): ax.text(v + 250, yy, f"{v:,}", va="center", fontsize=7, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(lab); ax.set_xlim(0, 15500); ax.set_xticks([0, 5000, 10000]); ax.set_xticklabels(["0", "5k", "10k"]); ax.set_xlabel("Count"); ptitle(ax, "a", "Redundancy"); ax.grid(axis="x")
    ax = axs[1]
    for cls, col, nm in [("infected", BLUE, "PCOS-positive"), ("noninfected", ORANGE, "PCOS-negative")]:
        w = np.sort(man[man.cls == cls].width.values); ax.step(w, np.arange(1, len(w) + 1) / len(w), color=col, lw=1.6, label=nm, where="post")
    ax.set_xlabel("Image width (px)"); ax.set_ylabel("Cumulative share"); ax.set_xlim(100, 1000); ax.grid(True); ptitle(ax, "b", "Width by class"); ax.legend(loc="lower right", handlelength=1.4)
    ax = axs[2]; f = a["single_feature_auc_infected_as_positive"]; nm = {"width": "Width", "bpp": "Bits per pixel", "height": "Height", "bytes": "File size", "mean": "Mean intensity", "aspect": "Aspect ratio", "std": "Intensity SD", "q0": "JPEG quant."}
    it = sorted([(max(v, 1 - v), nm[k]) for k, v in f.items()])
    ax.barh(np.arange(len(it)), [i[0] for i in it], color=BLUE, height=0.62); ax.set_yticks(np.arange(len(it))); ax.set_yticklabels([i[1] for i in it])
    for yy, (v, _) in enumerate(it): ax.text(v + 0.006, yy, f"{v:.3f}", va="center", fontsize=6.5, fontweight="bold")
    ax.set_xlim(0.5, 1.08); ax.set_xticks([0.5, 0.75, 1.0]); ax.set_xlabel("Single-feature AUC"); ptitle(ax, "c", "Class separability"); ax.grid(axis="x")
    return save(fig, "fig_audit")

# ---------------------------------------------------------------- partition rule effect + no-pixel references
def fig_protocol():
    m = pd.read_csv(f"{T}/main_results.csv"); mb = pd.read_csv(f"{T}/baselines_metadata_summary.csv"); lp = pd.read_csv(f"{T}/baselines_label_permuted_summary.csv")
    g0 = m[(m.rule == "group_g0") & (m.variant == "plain")].set_index("encoder").acc * 100; nv = m[(m.rule == "naive_image") & (m.variant == "plain")].set_index("encoder").acc * 100
    order = g0.sort_values().index.tolist()
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.85), layout="constrained", gridspec_kw=dict(width_ratios=[1.05, 1.0]))
    ax = axs[0]; y = np.arange(len(order))
    for i, e in enumerate(order): ax.plot([g0[e], nv[e]], [i, i], color=AXIS, lw=1.0, zorder=1)
    ax.scatter([nv[e] for e in order], y, s=26, color=ORANGE, zorder=3, label="Image-level split"); ax.scatter([g0[e] for e in order], y, s=26, color=BLUE, zorder=3, label="Group-disjoint split")
    ax.set_yticks(y); ax.set_yticklabels([PRETTY[e] for e in order]); ax.set_xlabel("Test accuracy (%)"); ax.set_xlim(96.8, 100.2); ax.grid(axis="x"); ptitle(ax, "a", "Same images, two splits")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handletextpad=0.2, columnspacing=1.0)
    ax = axs[1]; gmb = mb[mb.rule == "group_g0"].set_index("model").acc_mean * 100
    best = g0.idxmax(); items = [("Label-permuted heads", lp.acc.mean() * 100, GRAY), ("Majority class", gmb["majority"], GRAY), (f"Best encoder\n({PRETTY[best]})", g0.max(), BLUE), ("MobileNetV2", g0["mobilenetv2_100"], BLUE),
                                 ("Size-only logistic\nregression", gmb["meta_size_only_LR"], ORANGE), ("Size-only random\nforest", gmb["meta_size_only_RF"], ORANGE)]
    yy = np.arange(len(items)); ax.barh(yy, [i[1] for i in items], color=[i[2] for i in items], height=0.62)
    for k, it in enumerate(items): ax.text(it[1] + 0.8, k, f"{it[1]:.2f}", va="center", fontsize=7, fontweight="bold")
    ax.set_yticks(yy); ax.set_yticklabels([i[0] for i in items]); ax.set_xlim(40, 112); ax.set_xticks([50, 75, 100]); ax.set_xlabel("Test accuracy (%)"); ax.grid(axis="x"); ptitle(ax, "b", "No-pixel references")
    return save(fig, "fig_protocol")

# ---------------------------------------------------------------- classical tuning
def fig_classical():
    t = pd.read_csv(f"{T}/tuning_results.csv"); m = pd.read_csv(f"{T}/main_results.csv")
    order = (m[(m.rule == "group_g0") & (m.variant == "plain")].set_index("encoder").acc).sort_values().index.tolist()
    fig, axs = plt.subplots(1, 2, figsize=(W2, 3.0), sharey=True, layout="constrained")
    for ax, var, ttl in [(axs[0], "plain", "Plain preprocessing"), (axs[1], "neutral", "Shortcut-suppressed preprocessing")]:
        d = t[t.variant == var].set_index("encoder"); y = np.arange(len(order))
        for i, e in enumerate(order):
            c = FCOL[FAM[e]]; ax.plot([d.tuned_lo[e], d.tuned_hi[e]], [i, i], color=c, lw=1.3, zorder=2); ax.scatter(d.tuned_acc[e], i, s=24, color=c, zorder=3)
            ax.scatter(d.fixedC_acc[e], i, s=24, facecolor="white", edgecolor=c, linewidth=1.2, zorder=4)
        ax.set_xlabel("Test accuracy (%)"); ax.grid(axis="x"); ptitle(ax, "a" if var == "plain" else "b", ttl); ax.set_xlim(95.4, 99.4); ax.set_xticks([96, 97, 98, 99])
    axs[0].set_yticks(y); axs[0].set_yticklabels([PRETTY[e] for e in order])
    h = [Line2D([], [], marker="o", ls="", color=GRAY, ms=5, label="Validation-tuned head (95% CI)"), Line2D([], [], marker="o", ls="", mfc="white", mec=GRAY, ms=5, label="Fixed 3-value C grid")] + \
        [Line2D([], [], marker="o", ls="", color=c, ms=5, label=k) for k, c in FCOL.items()]
    fig.legend(handles=h, loc="outside lower center", ncol=5, columnspacing=1.2, handletextpad=0.2)
    return save(fig, "fig_classical")

# ---------------------------------------------------------------- confusion matrices + MCC forest
FOREST = [("Classical", "cls_main_best_plain", "Best head, plain"), ("Classical", "cls_tuned_best_plain", "Tuned head, plain"), ("Classical", "cls_ens_all11_plain", "Soft-vote ensemble (11), plain"),
          ("Classical", "cls_main_best_neutral", "Best head, suppressed"), ("Classical", "cls_ens_all11_neutral", "Ensemble (11), suppressed"),
          ("Federated", "fl_central_all_iid", "Central, all data"), ("Federated", "fl_fedavg_iid", "FedAvg, IID"), ("Federated", "fl_fedavg_lab0.5", "FedAvg, label skew 0.5"),
          ("Federated", "fl_fedavg_comb0.5", "FedAvg, combined skew"), ("Federated", "fl_fedavg_lab0.1", "FedAvg, label skew 0.1"), ("Federated", "fl_fedavg_dp0.8_lab0.5", "FedAvg + DP, σ = 0.8 (label 0.5)"),
          ("Federated", "fl_fedavg_dp4.0_lab0.5", "FedAvg + DP, σ = 4 (label 0.5)"), ("Quantum", "q_A_classical", "Fusion, classical only"), ("Quantum", "q_A_vqc_pairs", "Fusion, VQC pairs"),
          ("Quantum", "q_B_identity", "Hybrid layer, identity"), ("Quantum", "q_B_vqc_pairs", "Hybrid layer, VQC pairs"), ("Quantum", "q_B_vqc", "Hybrid layer, VQC ring")]
def fig_performance():
    c = pd.read_csv(f"{T}/performance_confusion.csv"); c = c[c.scope == "pooled"].set_index("row_id"); pm = pd.read_csv(f"{T}/performance_metrics.csv").set_index("row_id"); mb = pd.read_csv(f"{T}/baselines_metadata_summary.csv")
    mats = [("cls_main_best_plain", "Classical\nbest head"), ("fl_fedavg_iid", "FedAvg\nIID"), ("fl_fedavg_dp4.0_iid", "FedAvg + DP\nσ = 4, IID"), ("q_B_identity", "Quantum slot:\nidentity"), ("q_B_vqc_pairs", "Quantum slot:\nVQC pairs"), ("q_B_vqc", "Quantum slot:\nVQC ring")]
    fig = plt.figure(figsize=(W2, 5.35), layout="constrained"); sfa, sfb = fig.subfigures(2, 1, height_ratios=[1.0, 1.85], hspace=0.01)
    sfa.suptitle("(a) Row-normalized confusion matrices (mean counts over five test splits in parentheses)", x=0.003, ha="left", fontsize=8, fontweight="bold")
    axs = sfa.subplots(1, 6); cm = LinearSegmentedColormap.from_list("b", ["#ffffff", "#2a78d6"])
    for k, (rid, ttl) in enumerate(mats):
        ax = axs[k]; r = c.loc[rid]; M = np.array([[r.TN, r.FP], [r.FN, r.TP]]); rate = M / M.sum(1, keepdims=True)
        ax.imshow(rate, cmap=cm, vmin=0, vmax=1, aspect="equal")
        for i in range(2):
            for j in range(2): ax.text(j, i, f"{rate[i, j] * 100:.1f}%\n({M[i, j]:.0f})", ha="center", va="center", fontsize=6, color="white" if rate[i, j] > 0.55 else INK, fontweight="bold", linespacing=1.15)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Neg", "Pos"], fontsize=6.5); ax.set_yticks([0, 1]); ax.set_yticklabels(["Neg", "Pos"] if k == 0 else ["", ""], fontsize=6.5)
        ax.set_xlabel("Predicted", fontsize=6.5, labelpad=2); ax.set_ylabel("True" if k == 0 else "", fontsize=6.5, labelpad=2)
        ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]; ax.set_title(ttl, fontsize=6.8, loc="center", pad=3)
    sfb.suptitle("(b) MCC by track and setting", x=0.003, ha="left", fontsize=8, fontweight="bold")
    ax = sfb.subplots(); y = np.arange(len(FOREST))[::-1]; cols = {"Classical": BLUE, "Federated": ORANGE, "Quantum": PURPLE}
    for yy, (tr, rid, lab) in zip(y, FOREST):
        r = pm.loc[rid]; ax.plot([r.mcc_lo, r.mcc_hi], [yy, yy], color=cols[tr], lw=1.6, solid_capstyle="round"); ax.scatter(r.mcc, yy, s=24, color=cols[tr], zorder=3, edgecolor="white", linewidth=0.6)
        ax.text(1.012, yy, f"{r.mcc:.3f}", va="center", fontsize=6.8, fontweight="bold", transform=ax.get_yaxis_transform())
    ax.set_yticks(y); ax.set_yticklabels([f[2] for f in FOREST]); sz = mb[(mb.model == "meta_size_only_LR") & (mb.rule == "group_g0")].mcc_mean.iloc[0]
    ax.axvline(sz, color=GRAY, ls="--", lw=0.9); ax.text(sz - 0.003, -0.55, f"size-only LR {sz:.3f}", fontsize=6.5, color=GRAY, ha="right", va="center", fontweight="bold")
    ax.set_xlim(0.80, 1.0); ax.set_xlabel("Matthews correlation coefficient (95% group-cluster bootstrap interval)"); ax.grid(axis="x"); ax.set_ylim(-1.0, len(FOREST) - 0.4)
    sfb.legend(handles=[Line2D([], [], marker="o", ls="-", color=v, ms=4, lw=1.4, label=k) for k, v in cols.items()], loc="outside lower center", ncol=3, handletextpad=0.3, columnspacing=1.2)
    return save(fig, "fig_performance")

# ---------------------------------------------------------------- federated: heterogeneity + privacy-utility
PNAME = {"iid": "IID", "lab10": "Label\n(α=10)", "lab0.5": "Label\n(α=0.5)", "lab0.1": "Label\n(α=0.1)", "qty0.3": "Quantity", "feat": "Feature", "comb0.5": "Combined"}
def fig_fl():
    p = pd.read_csv(f"{T}/fl_pooled.csv"); pv = pd.read_csv(f"{T}/fl_privacy.csv"); eps = pv.groupby("sigma").epsilon.mean(); sg = (4.0, 2.0, 1.2, 0.8)
    fig, axs = plt.subplots(1, 2, figsize=(W2, 3.05), layout="constrained", gridspec_kw=dict(width_ratios=[1.3, 1.0]))
    ax = axs[0]; parts = list(PNAME); x = np.arange(len(parts)); off = {"central_ret": -0.27, "fedavg": -0.09, "fedprox": 0.09, "scaffold": 0.27}; col = {"central_ret": GRAY, "fedavg": BLUE, "fedprox": ORANGE, "scaffold": AQUA}
    lab = {"central_ret": "Matched central", "fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD"}
    for run in off:
        d = p[p.run == run].set_index("partition").loc[parts]; xs = x + off[run]
        ax.errorbar(xs, d.acc, yerr=[d.acc - d.lo, d.hi - d.acc], fmt="o" if run != "central_ret" else "s", ms=3.6, color=col[run], lw=0.9, capsize=0, label=lab[run])
    ax.set_xticks(x); ax.set_xticklabels([PNAME[k] for k in parts], fontsize=6.5); ax.set_ylabel("Test accuracy (%)"); ax.grid(axis="y"); ptitle(ax, "a", "Non-IID partitions, no privacy noise")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, handletextpad=0.2, columnspacing=0.9)
    ax = axs[1]; pc = {"iid": BLUE, "lab0.5": ORANGE, "lab0.1": PURPLE, "comb0.5": AQUA}; xe = [eps[s] for s in sg]
    for part in pc:
        d = p[p.partition == part].set_index("run"); ys = [d.loc[f"fedavg_dp{s}"].acc for s in sg]
        ax.plot(xe, ys, "-o", ms=3.6, lw=1.3, color=pc[part], label=PNAME[part].replace("\n", " ")); ax.scatter([110], [d.loc["fedavg"].acc], marker="D", s=22, color=pc[part], zorder=3)
    ax.set_xscale("log"); ax.set_xticks(xe + [110]); ax.set_xticklabels([f"{e:.1f}\nσ={s:g}" for e, s in zip(xe, sg)] + ["no DP"], fontsize=6.3); ax.minorticks_off(); ax.set_xlim(3.0, 150)
    ax.set_xlabel("Privacy budget ε (δ = 1e-5)\nσ = noise multiplier"); ax.set_ylabel("Test accuracy (%)"); ax.grid(True); ptitle(ax, "b", "Privacy-utility trade-off")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, handletextpad=0.3, columnspacing=1.0)
    return save(fig, "fig_fl")

# ---------------------------------------------------------------- quantum
def fig_quantum():
    q = pd.read_csv(f"{T}/quantum_final_pooled.csv").set_index("condition")
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3), layout="constrained")
    sets = [(axs[0], "a", "Slot as hybrid layer (setting B)", [("B_identity", "Identity (no slot)"), ("B_dense_w", "Dense, width-matched"), ("B_dense_p", "Dense, parameter-matched"), ("B_vqc_pairs", "VQC, brick entanglers"), ("B_vqc", "VQC, ring entanglers")]),
            (axs[1], "b", "Slot as fusion branch (setting A)", [("A_classical", "Classical branch only"), ("A_dense_w", "+ dense, width-matched"), ("A_dense_p", "+ dense, parameter-matched"), ("A_vqc_pairs", "+ VQC, matched partner"), ("A_vqc_pairs_derg", "+ VQC, deranged partner")])]
    for ax, l, ttl, rows in sets:
        y = np.arange(len(rows))[::-1]
        for yy, (k, nm) in zip(y, rows):
            col = PURPLE if "vqc" in k else (GRAY if k in ("B_identity", "A_classical") else BLUE); r = q.loc[k]
            ax.plot([r.lo, r.hi], [yy, yy], color=col, lw=1.5, solid_capstyle="round"); ax.scatter(r.acc, yy, s=26, color=col, zorder=3); ax.text(r.hi + 0.25, yy, f"{r.acc:.2f}", va="center", fontsize=6.8, fontweight="bold")
        ax.set_yticks(y); ax.set_yticklabels([n for _, n in rows]); ax.set_xlabel("Test accuracy (%, 95% CI)"); ax.grid(axis="x"); ptitle(ax, l, ttl)
    axs[0].set_xlim(90.5, 99.6); axs[1].set_xlim(95.5, 99.6)
    return save(fig, "fig_quantum")

# ---------------------------------------------------------------- XAI grid and faithfulness
METH = [("gradcam", "Grad-CAM"), ("gradcampp", "Grad-CAM++"), ("eigencam", "Eigen-CAM"), ("scorecam", "Score-CAM"), ("occlusion", "Occlusion"), ("ig", "Integr.\nGradients")]
ARMS = [("classical", "Classical"), ("fl", "Federated"), ("fl_dp", "Federated\n+ DP"), ("quantum", "Quantum")]
def overlay(img, cam, a=0.5):
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-12); hm = plt.get_cmap("jet")(cam)[..., :3]; return np.clip((1 - a) * img / 255.0 + a * hm, 0, 1)
def fig_xai():
    D = []
    for f in glob.glob(f"{R}/results/xai_arms/display_*.npz"):
        z = np.load(f, allow_pickle=True); D.append(dict(row=int(f.split("_")[-1][:-4]), label=int(z["label"]), img=z["img"], plive=json.loads(str(z["plive"])), maps={k: z[k].astype(np.float32) for k in z.files if "__" in k}))
    D.sort(key=lambda d: (-d["label"], d["row"])); d = D[0]; assert d["label"] == 1
    print("XAI display item: manifest row", d["row"], "plive", d["plive"])
    fig, axs = plt.subplots(4, 7, figsize=(W2, 4.5), gridspec_kw=dict(hspace=0.05, wspace=0.03, left=0.115, right=0.995, top=0.925, bottom=0.075))
    for r, (arm, an) in enumerate(ARMS):
        for c in range(7):
            ax = axs[r, c]; ax.axis("off"); ax.imshow(d["img"] if c == 0 else overlay(d["img"], d["maps"][f"{arm}__{METH[c - 1][0]}"]), interpolation="bicubic")
            if r == 0: ax.set_title("Original" if c == 0 else METH[c - 1][1], fontsize=7, fontweight="bold", pad=2, loc="center")
        axs[r, 0].text(-0.06, 0.5, f"{an}\nP = {d['plive'][arm]:.3f}", transform=axs[r, 0].transAxes, ha="right", va="center", fontsize=6.8, fontweight="bold", linespacing=1.35)
    cb = fig.add_axes([0.40, 0.034, 0.2, 0.014]); cb.imshow(np.linspace(0, 1, 256)[None, :], cmap="jet", aspect="auto", alpha=0.85); cb.set_xticks([]); cb.set_yticks([]); [s.set_visible(False) for s in cb.spines.values()]
    fig.text(0.385, 0.041, "Low relevance", ha="right", va="center", fontsize=6.5); fig.text(0.615, 0.041, "High relevance", ha="left", va="center", fontsize=6.5)
    return save(fig, "fig_xai")

def fig_faith():
    F = pd.read_csv(f"{T}/xai_faithfulness.csv"); A = pd.read_csv(f"{T}/xai_agreement.csv"); meths = [m for m, _ in METH]; names = {m: n.replace("\n", " ") for m, n in METH}
    cols = {"classical": BLUE, "fl": ORANGE, "fl_dp": AQUA, "quantum": INK}; mk = {"classical": "o", "fl": "s", "fl_dp": "^", "quantum": "D"}
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.75), layout="constrained", gridspec_kw=dict(width_ratios=[1.45, 1.0]))
    ax = axs[0]; y = np.arange(len(meths))[::-1]
    for k, (arm, an) in enumerate(ARMS):
        dd = F[(F.arm == arm)].set_index("method")
        for yy, m in zip(y, meths):
            r = dd.loc[m]; dy = (k - 1.5) * 0.15; ax.plot([r.ins_minus_del_lo, r.ins_minus_del_hi], [yy + dy, yy + dy], color=cols[arm], lw=0.9); ax.scatter(r.ins_minus_del, yy + dy, s=14, color=cols[arm], marker=mk[arm], zorder=3)
    ax.axvline(0, color=AXIS, lw=0.8); ax.set_yticks(y); ax.set_yticklabels([names[m] for m in meths]); ax.set_xlabel("Insertion AUC minus deletion AUC (95% CI)"); ax.grid(axis="x"); ptitle(ax, "a", "Faithfulness (higher is better)")
    ax = axs[1]
    for yy, m in zip(y, meths):
        d = A[A.method == m]; wq = d[(d.arm_b == "quantum") | (d.arm_a == "quantum")]; nq = d[(d.arm_a != "quantum") & (d.arm_b != "quantum")]
        ax.scatter(nq.spearman.mean(), yy, s=22, color=BLUE, zorder=3); ax.scatter(wq.spearman.mean(), yy, s=22, facecolor="white", edgecolor=INK, linewidth=1.2, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.set_xlim(-0.05, 1.08); ax.set_xlabel("Mean Spearman ρ between arms"); ax.grid(axis="x"); ptitle(ax, "b", "Agreement of maps")
    h = [Line2D([], [], marker=mk[a], ls="", color=cols[a], ms=4.5, label=n.replace("\n", " ")) for a, n in ARMS] + [Line2D([], [], marker="o", ls="", color=BLUE, ms=4.5, label="ρ: non-quantum pairs"), Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=4.5, label="ρ: pairs with quantum")]
    fig.legend(handles=h, loc="outside lower center", ncol=3, columnspacing=1.0, handletextpad=0.2)
    return save(fig, "fig_faith")

# ---------------------------------------------------------------- calibration: reliability diagrams (10 equal-width bins) with count strips
def fig_calibration():
    b = pd.read_csv(f"{T}/calibration_bins.csv"); e = pd.read_csv(f"{T}/calibration_ece.csv").set_index("row_id")
    panels = [("cls_main_best_plain", "Classical, best head", BLUE), ("fl_fedavg_iid", "FedAvg, IID", ORANGE), ("fl_fedavg_dp4.0_iid", "FedAvg + DP, \u03c3 = 4, IID", ORANGE),
              ("q_B_identity", "Quantum slot: identity", GRAY), ("q_B_vqc_pairs", "Quantum slot: VQC pairs", PURPLE), ("q_B_vqc", "Quantum slot: VQC ring", PURPLE)]
    fig, axs = plt.subplots(4, 3, figsize=(W2, 5.6), layout="constrained", gridspec_kw=dict(height_ratios=[0.8, 3, 0.8, 3]))
    for k, (rid, ttl, col) in enumerate(panels):
        r0 = 2 * (k // 3); c0 = k % 3; ast, am = axs[r0, c0], axs[r0 + 1, c0]; d = b[b.row_id == rid].sort_values("bin"); ctr = (d.lo + d.hi) / 2
        ast.bar(ctr, d.n.clip(lower=0.8), width=0.085, color=col, alpha=0.85, linewidth=0); ast.set_yscale("log"); ast.set_ylim(0.8, 6e4); ast.set_yticks([1, 100, 10000]); ast.set_yticklabels(["1", "10\u00b2", "10\u2074"])
        ast.set_xlim(0, 1); ast.set_xticks([]); ast.set_title(f"({'abcdef'[k]}) {ttl}", loc="left"); ast.spines["bottom"].set_visible(False); ast.tick_params(axis="x", length=0)
        if c0 == 0: ast.set_ylabel("count", fontsize=6.5, labelpad=1)
        ok = d.n > 0; am.plot([0, 1], [0, 1], color=AXIS, lw=0.9, ls="--"); am.plot(d.mean_p[ok], d.frac_pos[ok], color=col, lw=1.2); am.scatter(d.mean_p[ok], d.frac_pos[ok], s=20, color=col, zorder=3, edgecolor="white", linewidth=0.5)
        am.set_xlim(-0.02, 1.02); am.set_ylim(-0.02, 1.02); am.set_xticks([0, 0.25, 0.5, 0.75, 1]); am.set_xticklabels(["0", "", "0.5", "", "1"]); am.set_yticks([0, 0.25, 0.5, 0.75, 1]); am.set_yticklabels(["0", "", "0.5", "", "1"]); am.grid(True)
        am.set_aspect("equal", adjustable="box"); am.text(0.97, 0.05, f"ECE {e.loc[rid].ece:.3f}", ha="right", va="bottom", fontsize=7, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=GRID, lw=0.6))
        if c0 == 0: am.set_ylabel("Observed fraction positive", fontsize=6.8)
        if r0 == 2: am.set_xlabel("Mean predicted probability", fontsize=6.8)
    return save(fig, "fig_calibration")

# ---------------------------------------------------------------- ROC / PR / operating points (stored per-image probabilities of the main heads)
def _heads_curves():
    sp = pd.read_csv(f"{R}/data/splits.csv"); y = sp.label.values; out = {}
    grid = np.concatenate([np.logspace(-4, 0, 300), [1.0]])
    for e in PRETTY:
        P = np.load(f"{R}/results/heads/{e}__plain__group_g0.npz")["probs"]; tp, pr, sens = [], [], {}
        for k in range(5):
            t = (sp[f"grp_s{k}"] == "test").values; yt, pt = y[t], P[k][t]
            fpr, tpr, _ = roc_curve(yt, pt); tp.append(np.interp(grid, fpr, tpr))
            prec, rec, _ = precision_recall_curve(yt, pt); o = np.argsort(rec); rg = np.linspace(0.0, 1.0, 401)
            pr.append(np.interp(rg, rec[o], np.maximum.accumulate(prec[o][::-1])[::-1]))
            for sp_ in (0.95, 0.99): sens.setdefault(sp_, []).append(tpr[np.searchsorted(fpr, 1 - sp_, side="right") - 1])
        out[e] = dict(grid=grid, tpr=np.mean(tp, 0), rg=rg, prec=np.mean(pr, 0), sens95=np.mean(sens[0.95]), sens99=np.mean(sens[0.99]))
    return out
def fig_roc():
    C = _heads_curves(); m = pd.read_csv(f"{T}/main_results.csv"); g = m[(m.rule == "group_g0") & (m.variant == "plain")].set_index("encoder")
    order = g.sort_values("auc", ascending=False).index.tolist()
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.75), layout="constrained", gridspec_kw=dict(width_ratios=[1.05, 1.05, 1.0]))
    for e in order:
        c = FCOL[FAM[e]]; ax = axs[0]; ax.plot(C[e]["grid"], C[e]["tpr"], color=c, lw=0.9, alpha=0.9)
        axs[1].plot(C[e]["rg"], C[e]["prec"], color=c, lw=0.9, alpha=0.9)
    axs[0].set_xscale("log"); axs[0].set_xlim(1e-3, 1); axs[0].set_ylim(0.85, 1.003); axs[0].set_xlabel("False-positive rate (log scale)"); axs[0].set_ylabel("True-positive rate"); axs[0].grid(True)
    ptitle(axs[0], "a", "ROC curves"); axs[1].set_xlim(0.85, 1.0); axs[1].set_ylim(0.95, 1.003); axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision"); axs[1].grid(True); ptitle(axs[1], "b", "Precision--recall")
    y = np.arange(len(order))[::-1]; ax = axs[2]
    for i, e in enumerate(order):
        c = FCOL[FAM[e]]; ax.scatter(100 * C[e]["sens95"], y[i] + 0.16, s=15, color=c, zorder=3); ax.scatter(100 * C[e]["sens99"], y[i] - 0.16, s=15, facecolor="white", edgecolor=c, linewidth=1.0, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([PRETTY[e] for e in order], fontsize=6.3); ax.set_xlabel("Sensitivity (%) at fixed specificity"); ax.grid(axis="x"); ptitle(ax, "c", "Operating points")
    ax.set_xlim(92, 100.2)
    h = [Line2D([], [], marker="o", ls="", color=GRAY, ms=4, label="at 95% specificity"), Line2D([], [], marker="o", ls="", mfc="white", mec=GRAY, ms=4, label="at 99% specificity")] + [Line2D([], [], color=c, lw=1.4, label=k) for k, c in FCOL.items()]
    fig.legend(handles=h, loc="outside lower center", ncol=5, columnspacing=1.2, handletextpad=0.3)
    return save(fig, "fig_roc")

# ---------------------------------------------------------------- hyperparameter surfaces of the tuned heads (validation AUC; test accuracy of every train-only trial)
def fig_tuning():
    s = pd.read_csv(f"{T}/tuning_surface.csv"); s["hpd"] = s.hp.map(json.loads)
    fig, axs = plt.subplots(1, 4, figsize=(W2, 2.55), layout="constrained", gridspec_kw=dict(width_ratios=[1, 1, 1, 0.9]))
    for ax, fam, ttl, letter in [(axs[0], "lr_std", "Logistic regression", "a"), (axs[1], "lr_pca", "PCA(256) + logistic", "b")]:
        d = s[s.family == fam].copy(); d["C"] = d.hpd.map(lambda h: h["C"])
        for e, dd in d.groupby("encoder"):
            dd = dd.sort_values("C"); ax.plot(np.log10(dd.C), 100 * dd.test_acc, color=FCOL[FAM[e]], lw=0.8, alpha=0.9)
        ax.set_xlabel("log$_{10}$ C"); ax.grid(True); ptitle(ax, letter, ttl)
    axs[0].set_ylabel("Test accuracy (%), train-only fit")
    d = s[s.family == "mlp"].copy(); d["h"] = d.hpd.map(lambda h: h["h"]); d["a"] = d.hpd.map(lambda h: h["alpha"]); d["x"] = d.groupby(["h", "a"]).ngroup()
    lab = {(h, a): f"{h}/{a:g}" for h, a in sorted(set(zip(d.h, d.a)))}
    for e, dd in d.groupby("encoder"):
        dd = dd.sort_values(["h", "a"]); axs[2].plot(range(len(dd)), 100 * dd.test_acc, color=FCOL[FAM[e]], lw=0.8, alpha=0.9, marker="o", ms=1.8)
    axs[2].set_xticks(range(6)); axs[2].set_xticklabels([f"{a:g}" for h in (64, 256) for a in (0.1, 1.0, 10.0)]); axs[2].set_xlabel("$\\alpha$ (64 hidden units | 256)"); axs[2].grid(True); ptitle(axs[2], "c", "One-hidden-layer MLP")
    for ax in axs[:3]: ax.set_ylim(88, 99.5)
    sens = pd.read_csv(f"{T}/tuning_sensitivity.csv"); sp_ = sens[sens.variant == "plain"].set_index("encoder"); order = sp_.sort_values("median").index.tolist(); y = np.arange(len(order))
    for i, e in enumerate(order):
        c = FCOL[FAM[e]]; axs[3].plot([100 * sp_["min"][e], 100 * sp_["max"][e]], [i, i], color=c, lw=1.3); axs[3].scatter(100 * sp_["median"][e], i, s=14, color=c, zorder=3)
        axs[3].scatter(100 * sp_.selected_trial_test_acc[e], i, s=16, marker="D", facecolor="white", edgecolor=c, linewidth=1.0, zorder=4)
    axs[3].set_yticks(y); axs[3].set_yticklabels([PRETTY[e] for e in order], fontsize=6.0); axs[3].set_xlabel("Test accuracy (%)"); axs[3].grid(axis="x"); ptitle(axs[3], "d", "Range per encoder")
    h = [Line2D([], [], color=c, lw=1.4, label=k) for k, c in FCOL.items()] + [Line2D([], [], marker="o", ls="", color=GRAY, ms=4, label="median trial"), Line2D([], [], marker="D", ls="", mfc="white", mec=GRAY, ms=4, label="validation-selected trial")]
    fig.legend(handles=h, loc="outside lower center", ncol=5, columnspacing=1.2, handletextpad=0.3)
    return save(fig, "fig_tuning")

# ---------------------------------------------------------------- quantum tuning grid: validation AUC over qubits, layers, learning rates
def fig_qgrid():
    g = pd.read_csv(f"{T}/quantum_tuning_grid.csv")
    fig, axs = plt.subplots(1, 4, figsize=(W2, 2.5), layout="constrained", gridspec_kw=dict(width_ratios=[1, 1, 0.95, 0.95]))
    cmap = LinearSegmentedColormap.from_list("q", ["#f3effb", "#b7a6e3", "#7b5fc4"])
    for ax, cond, ttl, letter in [(axs[0], "B_vqc_pairs", "Hybrid layer (B)", "a"), (axs[1], "A_vqc_pairs", "Fusion branch (A)", "b")]:
        d = g[g.cond == cond]; best = d.groupby(["encoder", "NL", "NQ"]).val_auc.max().reset_index()          # best slot learning rate per encoder
        M = best.groupby(["NL", "NQ"]).val_auc.mean().unstack().reindex(index=[2, 3, 4], columns=[4, 6, 8])
        im = ax.imshow(M.values, cmap=cmap, aspect="auto", vmin=np.nanmin(M.values) - 0.0004, vmax=np.nanmax(M.values) + 0.0012, origin="lower")
        ax.set_xticks(range(3)); ax.set_xticklabels(M.columns); ax.set_yticks(range(3)); ax.set_yticklabels(M.index); ax.set_xlabel("Qubits"); ax.spines[:].set_visible(False); ax.tick_params(length=0)
        for (i, j), v in np.ndenumerate(M.values):
            if np.isfinite(v): ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=6.2, color=INK)
            else: ax.text(j, i, "not run", ha="center", va="center", fontsize=6.0, color=INK2, style="italic")
        ptitle(ax, letter, ttl)
    axs[0].set_ylabel("Layers"); fig.text(0.0, 0.0, "", fontsize=1)
    for ax, cond, ttl, letter in [(axs[2], "B_vqc_pairs", "Hybrid layer (B)", "c"), (axs[3], "A_vqc_pairs", "Fusion branch (A)", "d")]:
        d = g[g.cond == cond]
        for nq, col, off in zip([4, 6, 8], [GRAY, BLUE, PURPLE], [-0.2, 0, 0.2]):
            for k, lrs in enumerate(sorted(d.LRS.unique())):
                v = d[(d.NQ == nq) & (d.LRS == lrs)].val_auc.values; ax.scatter(np.full(len(v), k) + off, v, s=8, color=col, alpha=0.8, linewidths=0)
        ax.set_xticks(range(3)); ax.set_xticklabels(["0.01", "0.03", "0.1"]); ax.set_xlabel("slot learning rate"); ax.grid(axis="y"); ptitle(ax, letter, ttl)
    axs[2].set_ylabel("Validation AUC")
    h = [Line2D([], [], marker="o", ls="", color=c, ms=4, label=f"{q} qubits") for q, c in zip([4, 6, 8], [GRAY, BLUE, PURPLE])]
    fig.legend(handles=h, loc="outside lower center", ncol=3, columnspacing=1.2, handletextpad=0.3)
    return save(fig, "fig_qgrid")

ALL = dict(samples=fig_samples, audit=fig_audit, protocol=fig_protocol, classical=fig_classical, performance=fig_performance, fl=fig_fl, quantum=fig_quantum, calibration=fig_calibration, roc=fig_roc, tuning=fig_tuning, qgrid=fig_qgrid, xai=fig_xai, faith=fig_faith)
if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL); tot = 0
    for n in names: tot += ALL[n]()
    print("TOTAL overlap/outside flags:", tot)
