#!/usr/bin/env python3
"""Quantum subsection, tuned final runs: group-cluster bootstrap (resample g0 groups, B=1000, shared draws), paired contrasts, Holm within family,
equivalence margin +/-0.5 points. Also the tuning effect (tuned minus the earlier fixed-hyperparameter run of the same condition).
Reads results/quantum_final (tuned) and results/quantum (earlier). Writes results/tables/quantum_final_*.csv"""
import os, glob, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
B_ = os.environ.get("PCOS_BENCH_ROOT", os.getcwd()); R = f"{B_}/results"; QF = f"{R}/quantum_final"; QO = f"{R}/quantum"; T = f"{R}/tables"
man = pd.read_csv(f"{B_}/data/manifest.csv"); y = man.label.values; g = pd.factorize(man.g0)[0]; G = g.max() + 1
BOOT = 1000; MARGIN = 0.5; W = np.random.RandomState(2026).multinomial(G, np.full(G, 1.0 / G), size=BOOT).astype(np.float64)
ENC = ["mobilenetv2_100", "swin_tiny_patch4_window7_224", "resnet50"]
BC = ["B_identity", "B_dense_w", "B_dense_p", "B_vqc", "B_vqc_pairs"]; AC = ["A_classical", "A_dense_w", "A_dense_p", "A_vqc_pairs", "A_vqc_pairs_derg"]
ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj
def boot(P):
    acc = np.zeros(BOOT); auc = np.zeros(BOOT); pa = []; pu = []
    for si in range(P.shape[0]):
        te = np.nonzero(~np.isnan(P[si]))[0]; p = P[si, te]; yt = y[te]; Wi = W[:, g[te]]; pred = (p >= 0.5).astype(int)
        acc += (Wi @ (pred == yt)) / Wi.sum(1)
        o = np.argsort(p, kind="mergesort"); ys = yt[o]; Ws = Wi[:, o]; pos = Ws * ys; neg = Ws * (1 - ys); cn = np.cumsum(neg, axis=1) - neg
        auc += (pos * cn).sum(1) / (pos.sum(1) * neg.sum(1)); pa.append((pred == yt).mean()); pu.append(roc_auc_score(yt, p))
    return acc / P.shape[0] * 100, auc / P.shape[0], np.mean(pa) * 100, np.mean(pu), pa
def load(root, enc, var, c):
    fs = sorted(glob.glob(f"{root}/{enc}__{var}__{c}__r*.npz"))
    if not fs: return None
    ds = [np.load(f) for f in fs]; bs = [boot(d["probs"]) for d in ds]; seed_acc = np.mean([np.array(b[4]) * 100 for b in bs], axis=0)
    return dict(acc_d=np.mean([b[0] for b in bs], 0), auc_d=np.mean([b[1] for b in bs], 0), acc=np.mean([b[2] for b in bs]), auc=np.mean([b[3] for b in bs]), seed_sd=float(np.std(seed_acc, ddof=1)), nrep=len(fs),
                slot_params=int(ds[0]["slot_params"]), total_params=int(ds[0]["total_params"]), val_auc=float(np.mean([d["val_auc"].mean() for d in ds])), best_ep=float(np.mean([d["best_ep"].mean() for d in ds])), secs=float(np.sum([d["secs"].sum() for d in ds])))
ALL = BC + AC
VARS = [v for v in ("plain", "neutral") if all(load(QF, e, v, c) is not None for e in ENC for c in ALL if not c.endswith("derg") and c != "B_vqc")]
print("variants complete:", VARS)
res = {(e, v, c): load(QF, e, v, c) for e in ENC for v in VARS for c in ALL}; old = {(e, v, c): load(QO, e, v, c) for e in ENC for v in VARS for c in ALL}
combos = [(e, v) for e in ENC for v in VARS]
rows = []
for (e, v, c), r in res.items():
    if r is None: continue
    lo, hi = ci(r["acc_d"]); alo, ahi = ci(r["auc_d"]); rows.append(dict(encoder=e, variant=v, setting=c[0], condition=c, acc=r["acc"], acc_lo=lo, acc_hi=hi, auc=r["auc"], auc_lo=alo, auc_hi=ahi, seed_sd_acc=r["seed_sd"], n_repeats=r["nrep"], slot_params=r["slot_params"], total_params=r["total_params"], mean_val_auc=r["val_auc"], mean_best_epoch=r["best_ep"], train_secs=r["secs"]))
pd.DataFrame(rows).to_csv(f"{T}/quantum_final_conditions.csv", index=False)
pool = {c: dict(acc_d=np.mean([res[(e, v, c)]["acc_d"] for e, v in combos if res[(e, v, c)]], 0), acc=np.mean([res[(e, v, c)]["acc"] for e, v in combos if res[(e, v, c)]])) for c in ALL}
pr = []
for c in ALL:
    lo, hi = ci(pool[c]["acc_d"]); pr.append(dict(condition=c, acc=pool[c]["acc"], lo=lo, hi=hi))
pd.DataFrame(pr).to_csv(f"{T}/quantum_final_pooled.csv", index=False)
CONTR = [("B", "B_vqc_pairs", x) for x in ("B_identity", "B_dense_w", "B_dense_p", "B_vqc")] + [("A", "A_vqc_pairs", x) for x in ("A_classical", "A_dense_w", "A_dense_p", "A_vqc_pairs_derg")]
out = []
def add(fam, a, b, da, db, pa_, pb_, e, v):
    d = da - db; lo, hi = ci(d); p = min(1.0, 2 * (1 + min((d <= 0).sum(), (d >= 0).sum())) / (BOOT + 1))
    out.append(dict(family=fam, contrast=f"{a} - {b}", encoder=e, variant=v, diff_pts=pa_ - pb_, lo=lo, hi=hi, boot_p=p, call="better" if lo > 0 else "worse" if hi < 0 else ("equivalent within +/-0.5 pts" if (lo > -MARGIN and hi < MARGIN) else "inconclusive")))
for fam, a, b in CONTR:
    for e, v in combos + [("pooled", "pooled")]:
        if e == "pooled": add(fam, a, b, pool[a]["acc_d"], pool[b]["acc_d"], pool[a]["acc"], pool[b]["acc"], e, v); continue
        ra, rb = res[(e, v, a)], res[(e, v, b)]
        if ra and rb: add(fam, a, b, ra["acc_d"], rb["acc_d"], ra["acc"], rb["acc"], e, v)
for c in ALL:                                                              # tuning effect: tuned - earlier run of the same condition
    cm = [(e, v) for e, v in combos if res[(e, v, c)] and old[(e, v, c)]]
    if not cm: continue
    dn = np.mean([res[k + (c,)]["acc_d"] for k in cm], 0); do = np.mean([old[k + (c,)]["acc_d"] for k in cm], 0)
    add("tuning", f"{c} tuned", f"{c} earlier", dn, do, np.mean([res[k + (c,)]["acc"] for k in cm]), np.mean([old[k + (c,)]["acc"] for k in cm]), "pooled", "pooled")
cd = pd.DataFrame(out); cd["holm_p"] = np.nan
for fam in cd.family.unique():
    for sel in (cd.encoder != "pooled", cd.encoder == "pooled"):
        m = (cd.family == fam) & sel
        if m.any(): cd.loc[m, "holm_p"] = holm(cd.loc[m, "boot_p"].values)
cd.to_csv(f"{T}/quantum_final_contrasts.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(pd.DataFrame(pr).round(2).to_string()); print(cd[cd.encoder == "pooled"].round(3).to_string())
