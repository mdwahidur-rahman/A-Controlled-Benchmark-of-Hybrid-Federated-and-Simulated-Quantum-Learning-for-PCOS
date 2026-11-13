#!/usr/bin/env python3
"""Classical subsection statistics: validation-tuned heads (14_tuning.py) vs the fixed 3-value-C heads (05_heads.py), group-cluster bootstrap
(resample g0 groups, B=1000, shared draws), Holm within family, equivalence margin +/-0.5 points; family contrasts; soft-vote ensembles;
selected-configuration frequencies; sensitivity surface of the 26 trials (test metrics there are descriptive, never used for selection)."""
import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
B_ = os.environ.get("PCOS_BENCH_ROOT", os.environ.get("PCOS_BENCH_ROOT", os.getcwd())); R = f"{B_}/results"; T = f"{R}/tables"
man = pd.read_csv(f"{B_}/data/manifest.csv"); y = man.label.values; g = pd.factorize(man.g0)[0]; G = g.max() + 1; N = len(man)
SEEDS = [0, 1, 2, 3, 4]; BOOT = 1000; MARGIN = 0.5
W = np.random.RandomState(2026).multinomial(G, np.full(G, 1.0 / G), size=BOOT).astype(np.float64)
mr = pd.read_csv(f"{T}/main_results.csv"); FAM = dict(zip(mr.encoder, mr.family)); ENCS = list(dict.fromkeys(mr.encoder))
ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj
def auc_boot(p, yt, Wi):
    o = np.argsort(p, kind="mergesort"); ys = yt[o]; Ws = Wi[:, o]; pos = Ws * ys; neg = Ws * (1 - ys); cn = np.cumsum(neg, axis=1) - neg
    return (pos * cn).sum(1) / (pos.sum(1) * neg.sum(1))
def stats(P):
    """P (5,N) with NaN outside each seed's test set -> bootstrap draws and point estimates (mean over seeds)"""
    from sklearn.metrics import roc_auc_score
    acc = np.zeros(BOOT); auc = np.zeros(BOOT); pa = []; pu = []
    for s in range(len(SEEDS)):
        m = ~np.isnan(P[s]); p = P[s][m]; yt = y[m]; Wi = W[:, g[m]]
        acc += (Wi @ ((p >= .5).astype(int) == yt)) / Wi.sum(1); auc += auc_boot(p, yt, Wi); pa.append(((p >= .5) == yt).mean()); pu.append(roc_auc_score(yt, p))
    return dict(acc_d=acc / len(SEEDS) * 100, auc_d=auc / len(SEEDS), acc=float(np.mean(pa) * 100), auc=float(np.mean(pu)))
def contrast(a, b):
    d = a["acc_d"] - b["acc_d"]; lo, hi = ci(d); p = min(1.0, 2 * (1 + min((d <= 0).sum(), (d >= 0).sum())) / (BOOT + 1))
    return dict(diff_pts=a["acc"] - b["acc"], lo=lo, hi=hi, boot_p=p, call="better" if lo > 0 else "worse" if hi < 0 else ("equivalent within +/-0.5 pts" if (lo > -MARGIN and hi < MARGIN) else "inconclusive"))
tuned, orig = {}, {}
for e in ENCS:
    for v in ("plain", "neutral"):
        ft, fo = f"{R}/tuning/{e}__{v}.npz", f"{R}/heads/{e}__{v}__group_g0.npz"
        if os.path.exists(ft): tuned[(e, v)] = np.load(ft, allow_pickle=True)["probs"]
        if os.path.exists(fo): orig[(e, v)] = np.load(fo, allow_pickle=True)["probs"]
st_t = {k: stats(P) for k, P in tuned.items()}; st_o = {k: stats(P) for k, P in orig.items() if k in tuned}
rows = []
for v in ("plain", "neutral"):
    blk = []
    for e in ENCS:
        if (e, v) not in st_t or (e, v) not in st_o: continue
        a, b = st_t[(e, v)], st_o[(e, v)]; c = contrast(a, b)
        blk.append(dict(encoder=e, family=FAM[e], variant=v, tuned_acc=a["acc"], tuned_lo=ci(a["acc_d"])[0], tuned_hi=ci(a["acc_d"])[1], tuned_auc=a["auc"],
                        fixedC_acc=b["acc"], fixedC_lo=ci(b["acc_d"])[0], fixedC_hi=ci(b["acc_d"])[1], fixedC_auc=b["auc"], **c))
    h = holm([r["boot_p"] for r in blk])
    for r, hp in zip(blk, h): r["holm_p"] = hp
    rows += blk
pd.DataFrame(rows).to_csv(f"{T}/tuning_results.csv", index=False)
# pooled over encoders + family contrasts + ensembles
fam_rows = []
for v in ("plain", "neutral"):
    es = [e for e in ENCS if (e, v) in st_t]
    pool = lambda S, sel: dict(acc_d=np.mean([S[(e, v)]["acc_d"] for e in sel], 0), acc=float(np.mean([S[(e, v)]["acc"] for e in sel])))
    fam = {f: [e for e in es if FAM[e] == f] for f in ("CNN", "Transformer", "Hybrid")}
    pooled_t, pooled_o = pool(st_t, es), pool(st_o, es)
    c = contrast(pooled_t, pooled_o); fam_rows.append(dict(variant=v, comparison="tuned - fixed-C (all 11 encoders)", **c))
    for f, sel in fam.items():
        if sel: pt = pool(st_t, sel); lo, hi = ci(pt["acc_d"]); fam_rows.append(dict(variant=v, comparison=f"tuned {f} mean", diff_pts=pt["acc"], lo=lo, hi=hi, boot_p=np.nan, call=""))
    for a_, b_ in (("Transformer", "CNN"), ("Hybrid", "CNN"), ("Transformer", "Hybrid")):
        if fam[a_] and fam[b_]: fam_rows.append(dict(variant=v, comparison=f"tuned {a_} - {b_}", **contrast(pool(st_t, fam[a_]), pool(st_t, fam[b_]))))
pd.DataFrame(fam_rows).to_csv(f"{T}/tuning_family.csv", index=False)
ens = []
for v in ("plain", "neutral"):
    es = [e for e in ENCS if (e, v) in tuned]; singles = {e: st_t[(e, v)] for e in es}
    mean_single = dict(acc_d=np.mean([s["acc_d"] for s in singles.values()], 0), acc=float(np.mean([s["acc"] for s in singles.values()])))
    for name, sel in [("all 11", es)] + [(f, [e for e in es if FAM[e] == f]) for f in ("CNN", "Transformer", "Hybrid")]:
        if len(sel) < 2: continue
        P = np.mean([tuned[(e, v)] for e in sel], 0); s_ = stats(P); lo, hi = ci(s_["acc_d"]); ms = dict(acc_d=np.mean([singles[e]["acc_d"] for e in sel], 0), acc=float(np.mean([singles[e]["acc"] for e in sel])))
        ens.append(dict(variant=v, ensemble=name, members=len(sel), acc=s_["acc"], acc_lo=lo, acc_hi=hi, auc=s_["auc"], mean_member_acc=ms["acc"], **contrast(s_, ms)))
pd.DataFrame(ens).to_csv(f"{T}/tuning_ensembles.csv", index=False)
# selected configurations + sensitivity
sel_rows, sens_rows, surf = [], [], []
for e in ENCS:
    for v in ("plain", "neutral"):
        f = f"{R}/tuning/{e}__{v}.npz"
        if not os.path.exists(f): continue
        for s in json.loads(str(np.load(f, allow_pickle=True)["selected"])): sel_rows.append(dict(encoder=e, variant=v, seed=s["seed"], head=s["family"], hp=json.dumps(s["hp"])))
        tr = pd.read_csv(f"{R}/tuning/{e}__{v}__trials.csv")
        for sd, d in tr.groupby("seed"):
            i = d.assign(k=(d.val_auc / 5e-4).round(), nl=-d.val_ll).sort_values(["k", "nl"], ascending=False).iloc[0]
            sens_rows.append(dict(encoder=e, variant=v, seed=sd, selected_trial_test_acc=i.test_acc, median=d.test_acc.median(), min=d.test_acc.min(), max=d.test_acc.max(), val_auc_selected=i.val_auc))
        if v == "plain":
            for (fam_, hp), d in tr.groupby(["family", "hp"]): surf.append(dict(encoder=e, family=fam_, hp=hp, val_auc=d.val_auc.mean(), val_acc=d.val_acc.mean(), test_acc=d.test_acc.mean(), test_auc=d.test_auc.mean()))
pd.DataFrame(sel_rows).to_csv(f"{T}/tuning_selected.csv", index=False)
se = pd.DataFrame(sens_rows); se.groupby(["encoder", "variant"]).mean(numeric_only=True).drop(columns="seed").to_csv(f"{T}/tuning_sensitivity.csv")
pd.DataFrame(surf).to_csv(f"{T}/tuning_surface.csv", index=False)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows)[["encoder", "variant", "tuned_acc", "fixedC_acc", "diff_pts", "lo", "hi", "holm_p", "call"]].round(3).to_string())
print(pd.DataFrame(fam_rows).round(3).to_string()); print(pd.DataFrame(ens).round(3).to_string())
sl = pd.DataFrame(sel_rows); print(sl.groupby(["variant", "head"]).size().unstack().fillna(0))
