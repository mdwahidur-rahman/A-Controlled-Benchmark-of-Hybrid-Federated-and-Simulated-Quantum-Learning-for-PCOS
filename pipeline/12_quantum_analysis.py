#!/usr/bin/env python3
"""Stage 9 for the quantum slot: group-cluster bootstrap (resampling g0 groups, B=1000, shared draws across all conditions), paired contrasts,
equivalence call at +/-0.5 accuracy points, Holm within family. Tables are written from the stored per-item predictions only."""
import os, json, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
B_ = os.environ.get("PCOS_BENCH_ROOT", os.getcwd()); R = f"{B_}/results"; Q = f"{R}/quantum"; T = f"{R}/tables"
man = pd.read_csv(f"{B_}/data/manifest.csv"); y = man.label.values; g = pd.factorize(man.g0)[0]; G = g.max() + 1; N = len(man)
SEEDS = [0, 1, 2, 3, 4]; BOOT = 1000; MARGIN = 0.5
W = np.random.RandomState(2026).multinomial(G, np.full(G, 1.0 / G), size=BOOT).astype(np.float64)
ENC = ["mobilenetv2_100", "swin_tiny_patch4_window7_224", "resnet50"]; VAR = ["plain", "neutral"]
PRETTY = {"mobilenetv2_100": "MobileNetV2", "swin_tiny_patch4_window7_224": "Swin-T", "resnet50": "ResNet-50"}
BC = ["B_identity", "B_dense_w", "B_dense_p", "B_vqc_noent", "B_vqc", "B_vqc_pairs"]
AC = ["A_classical", "A_dense_w", "A_dense_p", "A_vqc", "A_vqc_lp", "A_vqc_shuf", "A_vqc_derg", "A_vqc_pairs", "A_vqc_pairs_lp", "A_vqc_pairs_shuf", "A_vqc_pairs_derg"]
def boot(P):
    acc = np.zeros(BOOT); auc = np.zeros(BOOT); pa = []; pu = []
    from sklearn.metrics import roc_auc_score
    for si in range(len(SEEDS)):
        te = np.nonzero(~np.isnan(P[si]))[0]; p = P[si, te]; yt = y[te]; Wi = W[:, g[te]]; pred = (p >= 0.5).astype(int)
        acc += (Wi @ (pred == yt)) / Wi.sum(1)
        o = np.argsort(p, kind="mergesort"); ys = yt[o]; Ws = Wi[:, o]; pos = Ws * ys; neg = Ws * (1 - ys); cn = np.cumsum(neg, axis=1) - neg
        auc += (pos * cn).sum(1) / (pos.sum(1) * neg.sum(1)); pa.append((pred == yt).mean()); pu.append(roc_auc_score(yt, p))
    return acc / len(SEEDS) * 100, auc / len(SEEDS), np.mean(pa) * 100, np.mean(pu), pa
def load(enc, var, c):
    fs = sorted(glob.glob(f"{Q}/{enc}__{var}__{c}__r*.npz")); ds = [np.load(f) for f in fs]; bs = [boot(d["probs"]) for d in ds]
    seed_acc = np.mean([np.array(b[4]) * 100 for b in bs], axis=0)                      # per-seed accuracy averaged over repeats
    rep_acc = [b[2] for b in bs]
    return dict(acc_d=np.mean([b[0] for b in bs], 0), auc_d=np.mean([b[1] for b in bs], 0), acc=np.mean(rep_acc), auc=np.mean([b[3] for b in bs]),
                seed_sd=float(np.std(seed_acc, ddof=1)), rep_sd=float(np.std(rep_acc, ddof=1)) if len(rep_acc) > 1 else np.nan, nrep=len(fs),
                slot_params=int(ds[0]["slot_params"]), total_params=int(ds[0]["total_params"]), val_auc=float(np.mean([d["val_auc"].mean() for d in ds])),
                best_ep=float(np.mean([d["best_ep"].mean() for d in ds])), secs=float(np.sum([d["secs"].sum() for d in ds])))
ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj
res = {}
for e in ENC:
    for v in VAR:
        for c in BC + AC: res[(e, v, c)] = load(e, v, c)
rows = []
for (e, v, c), r in res.items():
    lo, hi = ci(r["acc_d"]); alo, ahi = ci(r["auc_d"])
    rows.append(dict(encoder=e, variant=v, setting=c[0], condition=c, acc=r["acc"], acc_lo=lo, acc_hi=hi, auc=r["auc"], auc_lo=alo, auc_hi=ahi, seed_sd_acc=r["seed_sd"],
                     repeat_sd_acc=r["rep_sd"], n_repeats=r["nrep"], slot_params=r["slot_params"], total_params=r["total_params"], mean_val_auc=r["val_auc"], mean_best_epoch=r["best_ep"], train_secs=r["secs"]))
pd.DataFrame(rows).to_csv(f"{T}/quantum_conditions.csv", index=False)
# pooled over the 6 encoder x variant combos (same group draws -> dependence preserved)
pool = []
for c in BC + AC:
    d = np.mean([res[(e, v, c)]["acc_d"] for e in ENC for v in VAR], 0); lo, hi = ci(d)
    pool.append(dict(condition=c, acc=np.mean([res[(e, v, c)]["acc"] for e in ENC for v in VAR]), lo=lo, hi=hi))
pd.DataFrame(pool).to_csv(f"{T}/quantum_pooled.csv", index=False); pooled_d = {c: np.mean([res[(e, v, c)]["acc_d"] for e in ENC for v in VAR], 0) for c in BC + AC}
CONTR = [("B", "B_vqc", x) for x in ("B_identity", "B_dense_w", "B_dense_p", "B_vqc_noent")] + [("B", "B_vqc_pairs", x) for x in ("B_identity", "B_dense_w", "B_dense_p", "B_vqc_noent", "B_vqc")] + [("B", "B_vqc_noent", "B_dense_w")] + \
        [("A", "A_vqc", x) for x in ("A_classical", "A_dense_w", "A_dense_p", "A_vqc_lp", "A_vqc_shuf", "A_vqc_derg")] + [("A", "A_vqc_lp", "A_vqc_shuf")] + \
        [("A", "A_vqc_pairs", x) for x in ("A_classical", "A_dense_w", "A_dense_p", "A_vqc_pairs_lp", "A_vqc_pairs_shuf", "A_vqc_pairs_derg")] + [("A", "A_vqc_pairs_lp", "A_vqc_pairs_shuf")]
out = []
for fam, a, b in CONTR:
    for e, v in [(e, v) for e in ENC for v in VAR] + [("pooled", "pooled")]:
        if e == "pooled": da, db, pa_, pb_ = pooled_d[a], pooled_d[b], np.mean([res[(x, w, a)]["acc"] for x in ENC for w in VAR]), np.mean([res[(x, w, b)]["acc"] for x in ENC for w in VAR])
        else: da, db, pa_, pb_ = res[(e, v, a)]["acc_d"], res[(e, v, b)]["acc_d"], res[(e, v, a)]["acc"], res[(e, v, b)]["acc"]
        d = da - db; lo, hi = ci(d); p = min(1.0, 2 * (1 + min((d <= 0).sum(), (d >= 0).sum())) / (BOOT + 1))
        call = "better" if lo > 0 else "worse" if hi < 0 else ("equivalent within +/-0.5 pts" if (lo > -MARGIN and hi < MARGIN) else "inconclusive")
        out.append(dict(family=fam, contrast=f"{a} - {b}", encoder=e, variant=v, diff_pts=pa_ - pb_, lo=lo, hi=hi, boot_p=p, call=call))
cd = pd.DataFrame(out); cd["holm_p"] = np.nan
for (fam, con), idx in cd[cd.encoder != "pooled"].groupby(["family", "contrast"]).groups.items(): pass
for fam in ("B", "A"):                                                       # Holm across all per-combo contrasts in a family
    m = (cd.family == fam) & (cd.encoder != "pooled"); cd.loc[m, "holm_p"] = holm(cd.loc[m, "boot_p"].values)
    m = (cd.family == fam) & (cd.encoder == "pooled"); cd.loc[m, "holm_p"] = holm(cd.loc[m, "boot_p"].values)
cd.to_csv(f"{T}/quantum_contrasts.csv", index=False)
# schedule tuning audit trail
tune = []
for d in sorted(glob.glob(f"{R}/quantum_tune/*")) + [f"{R}/quantum_failed_lr0.01"]:
    fs = glob.glob(f"{d}/*__r0.npz")
    if not fs: continue
    v = {os.path.basename(f).split("__")[2]: float(np.load(f)["val_auc"].mean()) for f in fs if "A_vqc_" not in f}
    tune.append(dict(setting_dir=os.path.basename(d), n_conditions=len(v), mean_val_auc=np.mean(list(v.values())), vqc_val_auc=v.get("B_vqc", v.get("A_vqc", np.nan)), **{k: round(x, 4) for k, x in v.items()}))
pd.DataFrame(tune).to_csv(f"{T}/quantum_schedule_tuning.csv", index=False)
# architecture / initialization sensitivity (MobileNetV2 plain, ring entangler, seeds 0-4, test accuracy; no CI, descriptive)
sens = []
for d in sorted(glob.glob(f"{R}/quantum_sens/*")):
    row = dict(config=os.path.basename(d))
    for f in glob.glob(f"{d}/*__r0.npz"):
        c = os.path.basename(f).split("__")[2]; P = np.load(f)["probs"]; row[c] = np.mean([((P[i, ~np.isnan(P[i])] >= 0.5).astype(int) == y[~np.isnan(P[i])]).mean() for i in range(5)]) * 100
        row["slot_params_" + c] = int(np.load(f)["slot_params"])
    sens.append(row)
pd.DataFrame(sens).to_csv(f"{T}/quantum_sensitivity.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(pd.DataFrame(sens).round(2).iloc[:, :6].to_string()); print(pd.DataFrame(pool).round(2).to_string()); print(cd[cd.encoder == "pooled"].round(3).to_string()); print(pd.DataFrame(tune).round(4).iloc[:, :4].to_string())
