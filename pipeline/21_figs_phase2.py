#!/usr/bin/env python3
"""Figures 10-14 (classical tuning, federated non-IID, differential privacy, quantum tuning and tuned finals) from results/tables only."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from figstyle import *
setup()
ROOT = os.environ.get("PCOS_BENCH_ROOT", os.getcwd()); T = f"{ROOT}/results/tables"; os.makedirs(f"{ROOT}/figures/preview", exist_ok=True)
GRAY = "#8a8981"; PURPLE = "#7b5fc4"
BLUES = LinearSegmentedColormap.from_list("b", ["#eaf2fc", BLUE, "#0f3f7a"]); ORANGES = LinearSegmentedColormap.from_list("o", ["#fdeee6", ORANGE, "#8a2f0b"])
PN = {"iid": "IID (Dirichlet α=100)", "lab10": "Label skew α=10", "lab0.5": "Label skew α=0.5", "lab0.1": "Label skew α=0.1", "qty0.3": "Quantity skew", "feat": "Feature skew", "comb0.5": "Label + feature + quantity"}
def txt_on(v, lo, hi, cmap):
    r, g, b, _ = cmap((v - lo) / (hi - lo + 1e-12)); return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.52 else INK

# ======================================================================= Fig 10: classical tuning
tr = pd.read_csv(f"{T}/tuning_results.csv"); en = pd.read_csv(f"{T}/tuning_ensembles.csv"); sf = pd.read_csv(f"{T}/tuning_surface.csv")
fig = plt.figure(figsize=(17.5, 9.4)); gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 0.8, 1.05], height_ratios=[1.0, 0.7], hspace=0.6, wspace=0.62, left=0.17, right=0.965, top=0.93, bottom=0.2)
axa = fig.add_subplot(gs[:, 0]); axb = fig.add_subplot(gs[0, 1]); axd = fig.add_subplot(gs[1, 1]); axc = fig.add_subplot(gs[:, 2])
p = tr[tr.variant == "plain"].sort_values("tuned_acc"); ys = np.arange(len(p))
for y_, (_, r) in zip(ys, p.iterrows()):
    c = FAMCOL[r.family]; dotrow(axa, y_ + 0.14, r.tuned_acc, r.tuned_lo, r.tuned_hi, color=c, marker=FAMMK[r.family], ms=11)
    axa.plot([r.fixedC_acc], [y_ - 0.17], marker=FAMMK[r.family], ms=10, mfc="white", mec=c, mew=2.0, ls="none"); axa.text(99.42, y_ + 0.14, f"{r.tuned_acc:.2f}", ha="left", va="center", fontsize=11.5, color=INK, fontweight="bold")
axa.set_yticks(ys); axa.set_yticklabels([PRETTY[e] for e in p.encoder], fontsize=12.5); axa.set_xlim(96.2, 99.45); axa.set_xlabel("Test accuracy (%), group split, plain"); axa.grid(axis="x"); axa.set_ylim(-0.7, len(p) - 0.2)
axa.legend(handles=[Line2D([], [], marker="o", ls="none", ms=10, mfc=GRAY, mec="white", label="Validation-tuned (filled, 95% CI)"), Line2D([], [], marker="o", ls="none", ms=10, mfc="white", mec=GRAY, mew=2, label="Fixed C, 3 values (hollow)")] +
           [Line2D([], [], marker=FAMMK[f], ls="none", ms=10, mfc=FAMCOL[f], mec="white", label=f) for f in FAMCOL], loc="upper center", fontsize=11.5, bbox_to_anchor=(0.45, -0.085), ncol=2, handletextpad=0.3, columnspacing=1.2)
heading(fig, axa, "a", "Tuning does not beat a fixed head")
e = en[en.variant == "plain"].set_index("ensemble").loc[["all 11", "CNN", "Transformer", "Hybrid"]]; yb = np.arange(len(e))[::-1]
for y_, (nm, r) in zip(yb, e.iterrows()):
    c = {"all 11": INK, "CNN": BLUE, "Transformer": ORANGE, "Hybrid": AQUA}[nm]; dotrow(axb, y_ + 0.12, r.acc, r.acc_lo, r.acc_hi, color=c, marker="D", ms=10)
    axb.plot([r.mean_member_acc], [y_ - 0.16], marker="D", ms=9, mfc="white", mec=c, mew=2.0, ls="none"); axb.text(99.95, y_ + 0.12, f"{r.diff_pts:+.2f}", ha="left", va="center", fontsize=12, fontweight="bold", color=INK)
axb.set_yticks(yb); axb.set_yticklabels([f"{n} ({int(m)})" if n != "all 11" else "All 11 encoders" for n, m in zip(e.index, e.members)], fontsize=12.5); axb.set_xlim(96.8, 100.5); axb.set_xlabel("Soft-vote accuracy (%)"); axb.grid(axis="x"); axb.set_ylim(-0.7, 3.6)
axb.text(99.95, 3.62, "vs mean\nmember", ha="left", va="bottom", fontsize=11.5, color=INK2, fontweight="bold"); heading(fig, axb, "b", "Ensembles")
# (d) selected head families
sel = pd.read_csv(f"{T}/tuning_selected.csv"); cnt = sel.groupby(["variant", "head"]).size().unstack().fillna(0).reindex(columns=["lr_std", "lr_pca", "mlp"]); x = np.arange(3); w = 0.36
for i, (v, c) in enumerate((("plain", BLUE), ("neutral", ORANGE))):
    axd.bar(x + (i - 0.5) * w, cnt.loc[v].values, w * 0.92, color=c, label=v); [axd.text(x[k] + (i - 0.5) * w, cnt.loc[v].values[k] + 0.7, f"{int(cnt.loc[v].values[k])}", ha="center", fontsize=11.5, fontweight="bold") for k in range(3)]
axd.set_xticks(x); axd.set_xticklabels(["LR\nstandardized", "LR\n+PCA-256", "MLP"], fontsize=12); axd.set_ylabel("Selections (of 55)"); axd.grid(axis="y"); axd.set_ylim(0, 40); axd.legend(loc="upper left", fontsize=11.5); heading(fig, axd, "c", "Selected head")
s = sf.copy(); s["lab"] = s.apply(lambda r: (lambda h: f"LR  C=1e{np.log10(h['C']):.1f}".replace(".0", "") if r.family == "lr_std" else f"LR+PCA  C=1e{np.log10(h['C']):.1f}".replace(".0", "") if r.family == "lr_pca" else f"MLP  h={h['h']}  α={h['alpha']:g}")(json.loads(r.hp)), axis=1)
order = list(dict.fromkeys(s.sort_values(["family", "hp"]).lab)); piv = s.pivot_table(index="lab", columns="encoder", values="val_auc").reindex(order); cols = list(p.encoder[::-1]); piv = piv[cols]
vals = piv.values; lo = np.nanpercentile(vals, 3); im = axc.imshow(np.clip(vals, lo, None), aspect="auto", cmap=BLUES, vmin=lo, vmax=np.nanmax(vals))
axc.set_xticks(range(len(cols))); axc.set_xticklabels([PRETTY[c] for c in cols], rotation=55, ha="right", fontsize=11.5); axc.set_yticks(range(len(order))); axc.set_yticklabels(order, fontsize=10.5); axc.tick_params(length=0)
for j in range(vals.shape[1]): i_ = int(np.nanargmax(vals[:, j])); axc.plot([j], [i_], marker="*", ms=13, mfc="white", mec=INK, mew=1.0)
for sp_ in axc.spines.values(): sp_.set_visible(False)
cb = fig.colorbar(im, ax=axc, fraction=0.046, pad=0.03); cb.set_label("Validation AUC (train-only fits, mean of 5 seeds)", fontsize=12, fontweight="bold"); cb.ax.tick_params(labelsize=11.5)
axc.set_title("d   Tuning surface (★ = best per encoder)", fontsize=14, fontweight="bold", loc="left", pad=10)
save(fig, "fig10_classical_tuning", ROOT)

# ======================================================================= Fig 11: federated non-IID
pt = pd.read_csv(f"{T}/fl_partitions.csv"); po = pd.read_csv(f"{T}/fl_pooled.csv"); co = pd.read_csv(f"{T}/fl_contrasts.csv"); PARTS = list(PN)
fig, axs = plt.subplots(1, 3, figsize=(18, 6.8), gridspec_kw=dict(width_ratios=[0.85, 1.5, 1.0], wspace=0.5)); fig.subplots_adjust(left=0.205, right=0.985, top=0.86, bottom=0.14)
ax = axs[0]; ys = np.arange(len(PARTS))[::-1]; m = pt.groupby("partition").agg(c=("clients_retained", "mean"), mn=("min_clients", "min"), gr=("groups_retained", "mean")).reindex(PARTS)
for y_, pp in zip(ys, PARTS):
    ax.barh(y_, m.loc[pp, "c"], 0.62, color=AQUA if m.loc[pp, "c"] > 9.95 else ORANGE); ax.text(m.loc[pp, "c"] + 0.2, y_, f"{m.loc[pp, 'c']:.1f}  (min {int(m.loc[pp, 'mn'])})", va="center", fontsize=12, fontweight="bold", color=INK)
ax.set_yticks(ys); ax.set_yticklabels([PN[p_] for p_ in PARTS], fontsize=12.5); ax.set_xlim(0, 14.2); ax.set_xticks([0, 5, 10]); ax.set_xlabel("Clients retained (of 10)"); ax.grid(axis="x"); heading(fig, ax, "a", "Who can train")
ax = axs[1]; ser = [("central_all", "Central, all data", INK, "D", False), ("central_ret", "Central, retained clients", INK, "s", True), ("local", "No collaboration (mean)", GRAY, "v", False), ("fedavg", "FedAvg", BLUE, "o", False), ("fedprox", "FedProx", ORANGE, "s", False), ("scaffold", "SCAFFOLD", AQUA, "^", False)]
offs = np.linspace(0.33, -0.33, len(ser)); P_ = po.set_index(["partition", "run"])
for k, (run, nm, c, mk, hollow) in enumerate(ser):
    for y_, pp in zip(ys, PARTS):
        r = P_.loc[(pp, run)]; yy = y_ + offs[k]
        if run == "local" and r.acc < 92.6:
            ax.plot([92.35], [yy], marker="v", ms=10, mfc=GRAY, mec="white", mew=1.0, ls="none", zorder=4); ax.text(92.6, yy, f"{r.acc:.1f}", va="center", fontsize=11.5, fontweight="bold", color=INK2); continue
        if run in ("fedavg", "fedprox", "scaffold"): dotrow(ax, yy, r.acc, r.lo, r.hi, color=c, marker=mk, ms=9)
        elif hollow: ax.plot([r.acc], [yy], marker=mk, ms=9, mfc="white", mec=c, mew=2.0, ls="none")
        else: ax.plot([r.acc], [yy], marker=mk, ms=9, mfc=c, mec="white", mew=1.0, ls="none")
ax.set_yticks(ys); ax.set_yticklabels([]); ax.set_xlim(92, 99.3); ax.set_xlabel("Test accuracy (%), mean of 3 encoders"); ax.grid(axis="x"); ax.set_ylim(-0.6, len(PARTS) - 0.4)
ax.legend(handles=[Line2D([], [], marker=mk, ls="none", ms=9, mfc="white" if h else c, mec=c, mew=2 if h else 1, label=nm) for _, nm, c, mk, h in ser], loc="lower left", fontsize=11.5, ncol=2, bbox_to_anchor=(0.0, -0.36), columnspacing=1.0, handletextpad=0.3)
heading(fig, ax, "b", "Accuracy by heterogeneity")
ax = axs[2]; ax.axvspan(-0.5, 0.5, color="#e8f3ee", zorder=0); ax.axvline(0, color=AXIS, lw=1.3, zorder=1)
for k, (run, nm, c, mk) in enumerate([("fedavg", "FedAvg", BLUE, "o"), ("fedprox", "FedProx", ORANGE, "s"), ("scaffold", "SCAFFOLD", AQUA, "^")]):
    for y_, pp in zip(ys, PARTS):
        r = co[(co.family == "gap_vs_matched_central") & (co.partition == pp) & (co.a == run) & (co.encoder == "pooled")].iloc[0]; dotrow(ax, y_ + (1 - k) * 0.24, r.diff_pts, r.lo, r.hi, color=c, marker=mk, ms=9)
ax.set_yticks(ys); ax.set_yticklabels([]); ax.set_xlim(-1.6, 1.6); ax.set_xlabel("Federated − matched central (points)"); ax.grid(axis="x"); ax.set_ylim(-1.15, len(PARTS) - 0.4)
ax.text(0.0, -0.78, "±0.5 equivalence band", ha="center", va="top", fontsize=11.5, color=INK2, fontweight="bold"); heading(fig, ax, "c", "Federation gap")
save(fig, "fig11_fl_noniid", ROOT)

# ======================================================================= Fig 12: differential privacy
pv = pd.read_csv(f"{T}/fl_privacy.csv"); eps = pv.groupby("sigma").epsilon.median(); SG = [0.8, 1.2, 2.0, 4.0]; DPP = ["iid", "lab0.5", "lab0.1", "comb0.5"]
fig, axs = plt.subplots(1, 2, figsize=(15, 6.9), gridspec_kw=dict(width_ratios=[1.25, 1.0], wspace=0.28)); fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.15)
ax = axs[0]; cols = {"iid": BLUE, "lab0.5": ORANGE, "lab0.1": AQUA, "comb0.5": PURPLE}; mks = {"iid": "o", "lab0.5": "s", "lab0.1": "^", "comb0.5": "D"}
for pp in DPP:
    d = po[po.partition == pp].set_index("run"); xs = np.array([eps[s] for s in SG]); ysd = np.array([d.loc[f"fedavg_dp{s}", "acc"] for s in SG]); lo = np.array([d.loc[f"fedavg_dp{s}", "lo"] for s in SG]); hi = np.array([d.loc[f"fedavg_dp{s}", "hi"] for s in SG])
    ax.fill_between(xs, lo, hi, color=cols[pp], alpha=0.14, lw=0); ax.plot(xs, ysd, color=cols[pp], lw=2.4, marker=mks[pp], ms=10, mfc=cols[pp], mec="white", mew=1.2, label=PN[pp]); ax.axhline(d.loc["fedavg", "acc"], color=cols[pp], lw=1.5, ls=(0, (5, 3)))
ax.set_xscale("log"); ax.set_xticks([3.7, 8.9, 19.8, 46]); ax.set_xticklabels(["3.7\n(σ=4)", "8.9\n(σ=2)", "19.8\n(σ=1.2)", "46\n(σ=0.8)"]); ax.minorticks_off(); ax.set_xlabel("Privacy budget ε (δ=1e-5, group-level) — smaller is stronger"); ax.set_ylabel("Test accuracy (%)"); ax.grid(True)
ax.set_ylim(86, 101.5); ax.legend(loc="upper center", fontsize=11.5, ncol=2, title="Solid = FedAvg with DP;  dashed = same partition without DP", title_fontsize=11.5, columnspacing=1.2); heading(fig, ax, "a", "Privacy–utility")
ax = axs[1]; M = np.array([[co[(co.family == "privacy_cost") & (co.partition == pp) & (co.a == f"fedavg_dp{s}") & (co.encoder == "pooled")].iloc[0].diff_pts for s in SG] for pp in DPP]); lo_, hi_ = 0, np.abs(M).max()
ax.imshow(-M, cmap=ORANGES, vmin=lo_, vmax=hi_, aspect="auto")
for i in range(len(DPP)):
    for j in range(len(SG)): ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=15, fontweight="bold", color=txt_on(-M[i, j], lo_, hi_, ORANGES))
ax.set_xticks(range(4)); ax.set_xticklabels([f"σ={s:g}\nε≈{eps[s]:.1f}" for s in SG], fontsize=12); ax.set_yticks(range(4)); ax.set_yticklabels([PN[p_] for p_ in DPP], fontsize=12.5); ax.tick_params(length=0)
for sp_ in ax.spines.values(): sp_.set_visible(False)
heading(fig, ax, "b", "Accuracy cost of DP (points)"); save(fig, "fig12_fl_privacy", ROOT)

# ======================================================================= Fig 13: quantum tuning
g = pd.read_csv(f"{T}/quantum_tuning_grid.csv"); sl = pd.read_csv(f"{T}/quantum_tuning_selected.csv"); ECOL = {"mobilenetv2_100": BLUE, "swin_tiny_patch4_window7_224": ORANGE, "resnet50": AQUA}; EMK = {"mobilenetv2_100": "o", "swin_tiny_patch4_window7_224": "s", "resnet50": "^"}
fig = plt.figure(figsize=(16, 10.4)); gs = fig.add_gridspec(2, 6, height_ratios=[1, 0.95], hspace=0.5, wspace=1.15, left=0.12, right=0.985, top=0.935, bottom=0.07)
rng = np.random.RandomState(3)
for k, (setting, conds, ttl) in enumerate((("B", ["B_identity", "B_dense_w", "B_dense_p", "B_vqc_pairs"], "Hybrid layer (setting B)"), ("A", ["A_dense_w", "A_dense_p", "A_vqc_pairs"], "Two-branch fusion (setting A)"))):
    ax = fig.add_subplot(gs[0, 3 * k:3 * k + 3]); ys = np.arange(len(conds))[::-1]
    for y_, c in zip(ys, conds):
        for e_ in ECOL:
            d = g[(g.cond == c) & (g.encoder == e_)]; ax.plot(d.val_auc, y_ + rng.uniform(-0.28, 0.28, len(d)) + {"mobilenetv2_100": 0.0, "swin_tiny_patch4_window7_224": 0.0, "resnet50": 0.0}[e_], ls="none", marker=EMK[e_], ms=6.5, mfc=ECOL[e_], mec="white", mew=0.5, alpha=0.75)
            b = sl[(sl.cond == c) & (sl.encoder == e_)].iloc[0]; ax.plot([b.val_auc], [y_], marker="*", ms=17, mfc=ECOL[e_], mec=INK, mew=1.1, ls="none", zorder=5)
    if setting == "A":
        for e_ in ECOL: ax.axvline(sl[(sl.cond == "A_classical") & (sl.encoder == e_)].val_auc.iloc[0], color=ECOL[e_], lw=1.4, ls=(0, (4, 3)))
    ax.set_yticks(ys); ax.set_yticklabels([c[2:].replace("vqc_pairs", "VQC (pairs)").replace("dense_w", "dense (width)").replace("dense_p", "dense (params)").replace("identity", "identity") + f"\n{int(g[g.cond == c].groupby('encoder').size().iloc[0])} trials" for c in conds], fontsize=12); ax.set_xlabel("Validation AUC (3 seeds)"); ax.grid(axis="x"); ax.set_xlim(0.980, 0.9992)
    heading(fig, ax, "ab"[k], ttl)
    if k == 1: ax.legend(handles=[Line2D([], [], marker=EMK[e_], ls="none", ms=9, mfc=ECOL[e_], mec="white", label=PRETTY[e_]) for e_ in ECOL] + [Line2D([], [], marker="*", ls="none", ms=14, mfc="white", mec=INK, label="Selected (validation AUC)"), Line2D([], [], color=GRAY, lw=1.6, ls=(0, (4, 3)), label="Classical branch only")], loc="center left", fontsize=11.5, handletextpad=0.3, labelspacing=0.4)
for k, e_ in enumerate(ECOL):
    ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2]); d = g[(g.encoder == e_) & (g.cond == "B_vqc_pairs")]; piv = d.groupby(["NL", "NQ"]).val_auc.max().unstack(); piv = piv.reindex(index=[2, 3, 4], columns=[4, 6, 8]); v = piv.values; lo_, hi_ = np.nanmin(v) - 0.002, np.nanmax(v)
    ax.imshow(np.where(np.isnan(v), np.nan, v), cmap=BLUES, vmin=lo_, vmax=hi_, aspect="auto")
    for i in range(3):
        for j in range(3): ax.text(j, i, "—" if np.isnan(v[i, j]) else f"{v[i, j]:.4f}", ha="center", va="center", fontsize=12.5, fontweight="bold", color=INK if np.isnan(v[i, j]) else txt_on(v[i, j], lo_, hi_, BLUES))
    ax.set_xticks(range(3)); ax.set_xticklabels([4, 6, 8]); ax.set_yticks(range(3)); ax.set_yticklabels([2, 3, 4]); ax.set_xlabel("Qubits"); ax.set_ylabel("Layers"); ax.tick_params(length=0)
    for sp_ in ax.spines.values(): sp_.set_visible(False)
    heading(fig, ax, "cde"[k], f"{PRETTY[e_]}: VQC grid");
save(fig, "fig13_quantum_tuning", ROOT)

# ======================================================================= Fig 14: tuned quantum finals
qp = pd.read_csv(f"{T}/quantum_final_pooled.csv").set_index("condition"); qc = pd.read_csv(f"{T}/quantum_final_contrasts.csv"); qc = qc[qc.encoder == "pooled"]
fig, axs = plt.subplots(1, 3, figsize=(20, 6.9), gridspec_kw=dict(width_ratios=[1.0, 0.9, 0.95], wspace=1.0)); fig.subplots_adjust(left=0.15, right=0.985, top=0.88, bottom=0.14)
NM = {"B_identity": "No slot (identity)", "B_dense_w": "Dense, matched width", "B_dense_p": "Dense, matched params", "B_vqc": "VQC ring (not tuned)", "B_vqc_pairs": "VQC pairs (tuned)", "A_classical": "Classical branch only", "A_dense_w": "+ dense branch (width)", "A_dense_p": "+ dense branch (params)", "A_vqc_pairs": "+ VQC branch (tuned)", "A_vqc_pairs_derg": "+ VQC, partner swapped"}
conds = ["B_identity", "B_dense_w", "B_dense_p", "B_vqc", "B_vqc_pairs", "A_classical", "A_dense_w", "A_dense_p", "A_vqc_pairs", "A_vqc_pairs_derg"]; ys = np.arange(len(conds))[::-1] - np.r_[np.zeros(5), np.ones(5) * 0.7]
ax = axs[0]
for y_, c in zip(ys, conds):
    r = qp.loc[c]; col = ORANGE if "vqc" in c else BLUE; dotrow(ax, y_, r.acc, r.lo, r.hi, color=col, marker="D" if "vqc" in c else "o", ms=10); ax.text(99.7, y_, f"{r.acc:.2f}", va="center", fontsize=11.5, fontweight="bold", color=INK)
ax.set_yticks(ys); ax.set_yticklabels([NM[c] for c in conds], fontsize=12); ax.set_xlim(91.0, 100.8); ax.set_xticks([92, 94, 96, 98]); ax.set_xlabel("Test accuracy (%), pooled over 3 encoders"); ax.grid(axis="x"); ax.axhline(4.35, color=AXIS, lw=1.2); heading(fig, ax, "a", "Tuned slots (B above, A below)")
ax = axs[1]; ax.axvspan(-0.5, 0.5, color="#e8f3ee", zorder=0); ax.axvline(0, color=AXIS, lw=1.3)
rowsB = ["B_vqc_pairs - B_identity", "B_vqc_pairs - B_dense_w", "B_vqc_pairs - B_dense_p", "B_vqc_pairs - B_vqc", "A_vqc_pairs - A_classical", "A_vqc_pairs - A_dense_w", "A_vqc_pairs - A_dense_p", "A_vqc_pairs - A_vqc_pairs_derg"]
lab = ["B: VQC − identity", "B: VQC − dense (width)", "B: VQC − dense (params)", "B: pairs − ring", "A: VQC − classical", "A: VQC − dense (width)", "A: VQC − dense (params)", "A: VQC − swapped"]; yy = np.arange(8)[::-1] - np.r_[np.zeros(4), np.ones(4) * 0.6]
for y_, rname in zip(yy, rowsB):
    r = qc[qc.contrast == rname].iloc[0]; dotrow(ax, y_, r.diff_pts, r.lo, r.hi, color=ORANGE if r.diff_pts < 0 else BLUE, marker="D", ms=10)
ax.set_yticks(yy); ax.set_yticklabels(lab, fontsize=12); ax.set_xlim(-2, 6); ax.set_xlabel("Difference in accuracy (points)"); ax.grid(axis="x"); heading(fig, ax, "b", "Does the VQC help?")
ax = axs[2]; ax.axvspan(-0.5, 0.5, color="#e8f3ee", zorder=0); ax.axvline(0, color=AXIS, lw=1.3); tun = qc[qc.family == "tuning"].copy(); tun["c"] = tun.contrast.str.replace(" tuned - .* earlier", "", regex=True); tun = tun.set_index("c").reindex(conds).dropna(subset=["diff_pts"]); yt = np.arange(len(tun))[::-1]
for y_, (c, r) in zip(yt, tun.iterrows()): dotrow(ax, y_, r.diff_pts, r.lo, r.hi, color=ORANGE if "vqc" in c else BLUE, marker="D" if "vqc" in c else "o", ms=10)
ax.set_yticks(yt); ax.set_yticklabels([NM[c] for c in tun.index], fontsize=12); ax.set_xlim(-1.6, 1.0); ax.set_xlabel("Tuned − earlier fixed settings (points)"); ax.grid(axis="x"); heading(fig, ax, "c", "Effect of tuning")
save(fig, "fig14_quantum_final", ROOT); print("figures 10-14 saved")
