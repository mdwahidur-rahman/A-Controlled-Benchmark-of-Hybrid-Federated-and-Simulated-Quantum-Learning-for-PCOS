#!/usr/bin/env python3
"""Federated subsection statistics: group-cluster bootstrap (resampling g0 groups, B=1000, draws shared across all
conditions and encoders), paired contrasts, Holm within family, equivalence margin +/-0.5 accuracy points.
Reads results/fl/runs/{enc}__s{seed}.npz written by 15_fl.py; writes results/tables/fl_*.csv"""
import os, json, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
B_ = os.environ.get("PCOS_BENCH_ROOT", os.environ.get("PCOS_BENCH_ROOT", os.getcwd())); R = f"{B_}/results"; T = f"{R}/tables"; os.makedirs(T, exist_ok=True)
man = pd.read_csv(f"{B_}/data/manifest.csv"); g = pd.factorize(man.g0)[0]; G = g.max() + 1
SEEDS = [0, 1, 2, 3, 4]; BOOT = 1000; MARGIN = 0.5
W = np.random.RandomState(2026).multinomial(G, np.full(G, 1.0 / G), size=BOOT).astype(np.float64)
ENCS = ["mobilenetv2_100", "swin_tiny_patch4_window7_224", "mobilevit_s"]
PARTS = ["iid", "lab10", "lab0.5", "lab0.1", "qty0.3", "feat", "comb0.5"]; SIGMAS = [0.8, 1.2, 2.0, 4.0]; STRATS = ["fedavg", "fedprox", "scaffold"]
runs = {e: {s: dict(np.load(f"{R}/fl/runs/{e}__s{s}.npz", allow_pickle=True)) for s in SEEDS if os.path.exists(f"{R}/fl/runs/{e}__s{s}.npz")} for e in ENCS}
ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj
def auc_boot(p, yt, Wi):
    o = np.argsort(p, kind="mergesort"); ys = yt[o]; Ws = Wi[:, o]; pos = Ws * ys; neg = Ws * (1 - ys); cn = np.cumsum(neg, axis=1) - neg
    return (pos * cn).sum(1) / (pos.sum(1) * neg.sum(1))
def boot(enc, key):
    acc = np.zeros(BOOT); auc = np.zeros(BOOT); pa = []; pu = []; n = 0
    for s, d in runs[enc].items():
        if key not in d: continue
        te = d["test_rows"]; yt = d["test_y"]; P = d[key]; P = P[None] if P.ndim == 1 else P; Wi = W[:, g[te]]
        acc += np.mean([(Wi @ ((p >= .5).astype(int) == yt)) / Wi.sum(1) for p in P], 0); auc += np.mean([auc_boot(p, yt, Wi) for p in P], 0)
        pa.append(np.mean([((p >= .5).astype(int) == yt).mean() for p in P])); pu.append(np.mean([roc_auc_score(yt, p) for p in P])); n += 1
    if n == 0: return None
    return dict(acc_d=acc / n * 100, auc_d=auc / n, acc=float(np.mean(pa) * 100), auc=float(np.mean(pu)), n=n)
NAMES = ["central_all", "central_ret", "local"] + STRATS + [f"{k}_dp{s}" for s in SIGMAS for k in STRATS]
res = {}
for e in ENCS:
    for sp in PARTS:
        for nm in NAMES:
            r = boot(e, f"{sp}|{nm}")
            if r is not None: res[(e, sp, nm)] = r
rows = []
for (e, sp, nm), r in res.items():
    lo, hi = ci(r["acc_d"]); alo, ahi = ci(r["auc_d"]); rows.append(dict(encoder=e, partition=sp, run=nm, n_seeds=r["n"], acc=r["acc"], acc_lo=lo, acc_hi=hi, auc=r["auc"], auc_lo=alo, auc_hi=ahi))
pd.DataFrame(rows).to_csv(f"{T}/fl_results.csv", index=False)
def pooled(sp, nm):
    ds = [res[(e, sp, nm)] for e in ENCS if (e, sp, nm) in res]
    if len(ds) < len(ENCS): return None
    return dict(acc_d=np.mean([d["acc_d"] for d in ds], 0), acc=float(np.mean([d["acc"] for d in ds])), auc=float(np.mean([d["auc"] for d in ds])), auc_d=np.mean([d["auc_d"] for d in ds], 0))
pr = []
for sp in PARTS:
    for nm in NAMES:
        p = pooled(sp, nm)
        if p is None: continue
        lo, hi = ci(p["acc_d"]); alo, ahi = ci(p["auc_d"]); pr.append(dict(partition=sp, run=nm, acc=p["acc"], lo=lo, hi=hi, auc=p["auc"], auc_lo=alo, auc_hi=ahi))
pd.DataFrame(pr).to_csv(f"{T}/fl_pooled.csv", index=False)
CON = []
for sp in PARTS:
    for s_ in STRATS:
        CON += [("gap_vs_matched_central", sp, s_, "central_ret"), ("gap_vs_all_data_central", sp, s_, "central_all"), ("collaboration_gain", sp, s_, "local")]
    CON += [("retention_cost", sp, "central_ret", "central_all"), ("fedprox_vs_fedavg", sp, "fedprox", "fedavg"), ("scaffold_vs_fedavg", sp, "scaffold", "fedavg")]
    for s in SIGMAS:
        for s_ in STRATS: CON.append(("privacy_cost", sp, f"{s_}_dp{s}", s_))
out = []
for fam, sp, a, b in CON:
    for e in ENCS + ["pooled"]:
        if e == "pooled": pa_, pb_ = pooled(sp, a), pooled(sp, b)
        else: pa_, pb_ = res.get((e, sp, a)), res.get((e, sp, b))
        if pa_ is None or pb_ is None: continue
        d = pa_["acc_d"] - pb_["acc_d"]; lo, hi = ci(d); p = min(1.0, 2 * (1 + min((d <= 0).sum(), (d >= 0).sum())) / (BOOT + 1))
        call = "better" if lo > 0 else "worse" if hi < 0 else ("equivalent within +/-0.5 pts" if (lo > -MARGIN and hi < MARGIN) else "inconclusive")
        out.append(dict(family=fam, partition=sp, a=a, b=b, encoder=e, diff_pts=pa_["acc"] - pb_["acc"], lo=lo, hi=hi, boot_p=p, call=call))
cd = pd.DataFrame(out); cd["holm_p"] = np.nan
for (fam, e), idx in cd.groupby(["family", "encoder"]).groups.items(): cd.loc[idx, "holm_p"] = holm(cd.loc[idx, "boot_p"].values)
cd.to_csv(f"{T}/fl_contrasts.csv", index=False)
# partition characterization (seed-averaged)
pc = []
for e in ENCS:
    for sp in PARTS:
        ms = [json.loads(str(d["meta"])).get(f"{sp}|central_all") or json.loads(str(d["meta"])).get(f"{sp}|fedavg") for d in runs[e].values() if "meta" in d]
        ms = [m for m in ms if m]
        if not ms: continue
        nr = [m["n_clients_retained"] for m in ms]; gs = [np.array(m["retained_groups"], float) for m in ms if len(m["retained_groups"])]
        pc.append(dict(encoder=e, partition=sp, seeds=len(ms), clients_retained=float(np.mean(nr)), min_clients=int(min(nr)), groups_retained=float(np.mean([x.sum() for x in gs])) if gs else 0.0,
                       size_cv=float(np.mean([x.std() / x.mean() for x in gs])) if gs else np.nan))
pd.DataFrame(pc).to_csv(f"{T}/fl_partitions.csv", index=False)
# privacy accounting: epsilon recorded by the run (delta=1e-5, group-level, Poisson rate q=0.1) + tuned config
eps = []
for e in ENCS:
    sel = json.load(open(f"{R}/fl/tuned_{e}.json"))
    for s in SIGMAS:
        c = sel[f"dp{s}"]; par = c["par"]; ev = []
        for d in runs[e].values():
            m = json.loads(str(d["meta"]))
            for k, v in m.items():
                for st in STRATS:
                    if f"{st}_dp{s}_eps" in v: ev.append(v[f"{st}_dp{s}_eps"])
        eps.append(dict(encoder=e, sigma=s, clip=par["C"], lr=par["lr"], q=par["q"], rounds=c["R"], total_steps=c["R"] * int(round(par["E"] / par["q"])), val_auc=c["val_auc"], epsilon=float(np.median(ev)) if ev else np.nan))
ep = pd.DataFrame(eps); ep.to_csv(f"{T}/fl_privacy.csv", index=False)
# tuning audit trail (deduplicated by config id)
tr = []
for e in ENCS:
    seen = {}
    for l in open(f"{R}/fl/tune2/{e}.jsonl"):                       # non-DP: stable grid, 6 runs, worst-case selection
        r = json.loads(l); seen[r["cid"]] = r
    for r in seen.values():
        A, L = np.array(r["auc"]), np.array(r["ll"]); mn, ml = A.min(0), L.mean(0); i = max(range(A.shape[1]), key=lambda j: (round(mn[j] / 5e-4), -ml[j]))
        tr.append(dict(encoder=e, strategy=r["strat"], **r["par"], best_round=i + 1, val_auc_mean=float(A[:, i].mean()), val_auc_min=float(mn[i]), val_ll=float(ml[i]), stage="stable grid (non-DP)"))
    seen = {}
    for l in open(f"{R}/fl/tune/{e}.jsonl"):                        # DP slots: original grid, mean over 4 runs (non-DP rows of this file are superseded)
        r = json.loads(l)
        if r["strat"] == "dpfedavg": seen[r["cid"]] = r
    for r in seen.values():
        auc = np.array(r["auc"]); ll = np.array(r["ll"]); i = max(range(len(auc)), key=lambda j: (round(auc[j] / 5e-4), -ll[j]))
        tr.append(dict(encoder=e, strategy=r["strat"], **r["par"], best_round=i + 1, val_auc_mean=float(auc[i]), val_auc_min=np.nan, val_ll=float(ll[i]), stage="DP grid"))
pd.DataFrame(tr).to_csv(f"{T}/fl_tuning_trials.csv", index=False)
pd.set_option("display.width", 250)
print(pd.DataFrame(pr).round(2).to_string()); print(ep.round(3).to_string()); print(pd.DataFrame(pc).round(2).to_string())
print(cd[(cd.encoder == "pooled") & (cd.family.isin(["gap_vs_matched_central", "collaboration_gain", "retention_cost", "fedprox_vs_fedavg", "scaffold_vs_fedavg"]))].round(3).to_string())
