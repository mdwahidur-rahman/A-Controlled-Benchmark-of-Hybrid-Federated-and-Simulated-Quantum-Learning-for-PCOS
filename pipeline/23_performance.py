#!/usr/bin/env python3
"""Performance analysis matrices: 13 metrics per row (track x setting) with 95% g0 group-cluster bootstrap intervals (B=1000, shared draws
RandomState(2026).multinomial), pooled over encoders / seeds / repeats exactly as in the accuracy tables (metric computed per test set, then averaged),
mean confusion counts per test set, and paired contrasts on MCC and accuracy (Holm within family).
Usage: python3 pipeline/23_performance.py selftest | run [budget_s] | assemble
Metrics (threshold 0.5, positive class = infected = PCOS-positive, label 1): acc, bal_acc, prec, rec (sensitivity), spec, npv, f1, mcc, kappa (Cohen),
auc, ap (average precision), brier, logloss (probabilities clipped to [1e-15, 1-1e-15])."""
import os, sys, glob, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
B_ = os.environ.get("PCOS_BENCH_ROOT", os.getcwd()); R = f"{B_}/results"; T = f"{R}/tables"; P = f"{R}/performance"; os.makedirs(f"{P}/rows", exist_ok=True)
man = pd.read_csv(f"{B_}/data/manifest.csv"); y_all = man.label.values.astype(int); g = pd.factorize(man.g0)[0]; G = g.max() + 1
BOOT = 1000; W = np.random.RandomState(2026).multinomial(G, np.full(G, 1.0 / G), size=BOOT).astype(np.float64)
M = ["acc", "bal_acc", "prec", "rec", "spec", "npv", "f1", "mcc", "kappa", "auc", "ap", "brier", "logloss"]
ENC_FL = ["mobilenetv2_100", "swin_tiny_patch4_window7_224", "mobilevit_s"]; ENC_Q = ["mobilenetv2_100", "swin_tiny_patch4_window7_224", "resnet50"]
PARTS = ["iid", "lab10", "lab0.5", "lab0.1", "qty0.3", "feat", "comb0.5"]; DPPARTS = ["iid", "lab0.5", "lab0.1", "comb0.5"]; SIGMAS = ["0.8", "1.2", "2.0", "4.0"]
PN = {"iid": "IID", "lab10": "label skew a=10", "lab0.5": "label skew a=0.5", "lab0.1": "label skew a=0.1", "qty0.3": "quantity skew", "feat": "feature skew", "comb0.5": "combined skew"}
QC = ["A_classical", "A_dense_w", "A_dense_p", "A_vqc_pairs", "A_vqc_pairs_derg", "B_identity", "B_dense_w", "B_dense_p", "B_vqc", "B_vqc_pairs"]
_wc = {}
def Wsub(te):
    k = hash(te.tobytes())
    if k not in _wc: _wc[k] = W[:, g[te]]
    return _wc[k]
def metrics(p, yt, Wi):
    """Weighted metrics. p float64 (n,), yt int (n,), Wi (K, n) non-negative weights -> (K, 13)."""
    pred = p >= 0.5; pos = yt == 1; f = lambda m: Wi @ m.astype(np.float64)
    TP, FP, TN, FN = f(pred & pos), f(pred & ~pos), f(~pred & ~pos), f(~pred & pos); N = TP + FP + TN + FN
    with np.errstate(all="ignore"):
        acc = (TP + TN) / N; rec = TP / (TP + FN); spec = TN / (TN + FP); prec = TP / (TP + FP); npv = TN / (TN + FN)
        f1 = 2 * TP / (2 * TP + FP + FN); mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        pe = ((TP + FP) * (TP + FN) + (TN + FN) * (TN + FP)) / N ** 2; kap = (acc - pe) / (1 - pe)
        pc = np.clip(p, 1e-15, 1 - 1e-15); ll = -(yt * np.log(pc) + (1 - yt) * np.log(1 - pc)); ws = Wi.sum(1)
        brier = Wi @ ((p - yt) ** 2) / ws; logl = Wi @ ll / ws
        o = np.argsort(-p, kind="mergesort"); ps = p[o]; ys = yt[o]; st = np.r_[0, np.nonzero(ps[1:] != ps[:-1])[0] + 1]; Ws = Wi[:, o]
        cP = np.cumsum(np.add.reduceat(Ws * ys, st, axis=1), 1); cN = np.cumsum(np.add.reduceat(Ws * (1 - ys), st, axis=1), 1)
        tpr = cP / cP[:, -1:]; fpr = cN / cN[:, -1:]; z = np.zeros((Wi.shape[0], 1)); tpr0 = np.hstack([z, tpr[:, :-1]]); fpr0 = np.hstack([z, fpr[:, :-1]])
        auc = ((fpr - fpr0) * (tpr + tpr0) / 2).sum(1); prc = np.where(cP + cN > 0, cP / (cP + cN), 0.0); ap = ((tpr - tpr0) * prc).sum(1)
    out = np.stack([acc, (rec + spec) / 2, prec, rec, spec, npv, f1, mcc, kap, auc, ap, brier, logl], 1)
    return out, np.stack([TP, FP, TN, FN], 1)
def unit_stats(p, te):
    """-> point (13,), boot (B, 13), counts (4,) for one test set."""
    p = p.astype(np.float64); yt = y_all[te]; pt, ct = metrics(p, yt, np.ones((1, len(te)))); bt, _ = metrics(p, yt, Wsub(te))
    return pt[0], bt, ct[0]
# ---------------------------------------------------------------- unit loaders
def _sparse(probs):
    for s in range(probs.shape[0]):
        te = np.nonzero(~np.isnan(probs[s]))[0]; yield s, te, probs[s][te]
def u_heads(enc, var, rule="group_g0"):
    return [(enc, s, te, p) for s, te, p in _sparse(np.load(f"{R}/heads/{enc}__{var}__{rule}.npz")["probs"])]
def u_tuned(enc, var):
    return [(enc, s, te, p) for s, te, p in _sparse(np.load(f"{R}/tuning/{enc}__{var}.npz")["probs"])]
def u_ens(encs, var):
    pr = np.mean([np.load(f"{R}/tuning/{e}__{var}.npz")["probs"].astype(np.float64) for e in encs], 0)
    return [("ensemble", s, te, p) for s, te, p in _sparse(pr)]
def u_quantum(cond):
    out = []
    for e in ENC_Q:
        for r, f in enumerate(sorted(glob.glob(f"{R}/quantum_final/{e}__plain__{cond}__r*.npz"))):
            out += [(e, f"r{r}s{s}", te, p) for s, te, p in _sparse(np.load(f)["probs"])]
    return out
_flc = {}
def u_fl(part, run):
    out = []
    for e in ENC_FL:
        for s in range(5):
            f = f"{R}/fl/runs/{e}__s{s}.npz"
            if f not in _flc: _flc[f] = dict(np.load(f, allow_pickle=True))
            z = _flc[f]; te = z["test_rows"].astype(int); assert (y_all[te] == z["test_y"]).all()
            out.append((e, s, te, z[f"{part}|{run}"]))
    return out
# ---------------------------------------------------------------- row plan
def plan():
    mr = pd.read_csv(f"{T}/main_results.csv"); tr = pd.read_csv(f"{T}/tuning_results.csv"); fam = dict(zip(mr.encoder, mr.family)); rows = []
    best = lambda var: mr[(mr.variant == var) & (mr.rule == "group_g0")].sort_values("acc", ascending=False).encoder.iloc[0]
    tbest = lambda var: tr[tr.variant == var].sort_values("tuned_acc", ascending=False).encoder.iloc[0]
    A = lambda rid, track, label, setting, part, fn: rows.append(dict(id=rid, track=track, label=label, setting=setting, partition=part, fn=fn))
    for var, vn in (("plain", "plain"), ("neutral", "shortcut-suppressed")):
        e = best(var); A(f"cls_main_best_{var}", "Classical", f"Main analysis, best head ({e})", f"{vn}, coarse C grid", "", lambda e=e, var=var: u_heads(e, var))
    A("cls_main_mnv2_plain", "Classical", "Main analysis, MobileNetV2 head", "plain, coarse C grid", "", lambda: u_heads("mobilenetv2_100", "plain"))
    for var, vn in (("plain", "plain"), ("neutral", "shortcut-suppressed")):
        e = tbest(var); A(f"cls_tuned_best_{var}", "Classical", f"Tuned head, best encoder ({e})", f"{vn}, tuned", "", lambda e=e, var=var: u_tuned(e, var))
    A("cls_tuned_mnv2_plain", "Classical", "Tuned head, MobileNetV2", "plain, tuned", "", lambda: u_tuned("mobilenetv2_100", "plain"))
    tf = [e for e in mr.encoder.unique() if fam[e] == "Transformer"]
    A("cls_ens_transformer_plain", "Classical", "Tuned soft-vote ensemble, 4 Transformers", "plain, tuned", "", lambda: u_ens(tf, "plain"))
    A("cls_ens_all11_plain", "Classical", "Tuned soft-vote ensemble, all 11", "plain, tuned", "", lambda: u_ens(list(mr.encoder.unique()), "plain"))
    A("cls_ens_all11_neutral", "Classical", "Tuned soft-vote ensemble, all 11", "shortcut-suppressed, tuned", "", lambda: u_ens(list(mr.encoder.unique()), "neutral"))
    A("fl_central_all_iid", "Federated", "Central, all data", "IID", "iid", lambda: u_fl("iid", "central_all"))
    for pt in PARTS:
        A(f"fl_central_ret_{pt}", "Federated", "Central, retained clients (matched)", PN[pt], pt, lambda pt=pt: u_fl(pt, "central_ret"))
        A(f"fl_fedavg_{pt}", "Federated", "FedAvg", PN[pt], pt, lambda pt=pt: u_fl(pt, "fedavg"))
        for s in (SIGMAS if pt in DPPARTS else []): A(f"fl_fedavg_dp{s}_{pt}", "Federated + DP", f"FedAvg + DP (sigma={s})", PN[pt], pt, lambda pt=pt, s=s: u_fl(pt, f"fedavg_dp{s}"))
    for c in QC: A(f"q_{c}", "Quantum", c, "tuned, plain", "", lambda c=c: u_quantum(c))
    return rows
def run_row(r):
    us = r["fn"](); encs = sorted(set(u[0] for u in us)); st = [unit_stats(p, te) for _, _, te, p in us]
    pt = np.array([s[0] for s in st]); bt = np.array([s[1] for s in st]); ct = np.array([s[2] for s in st]); ue = np.array([u[0] for u in us])
    pool_b = bt.mean(0); sc = {"pooled": np.ones(len(us), bool), **{e: ue == e for e in encs}}
    d = dict(encs=np.array(list(sc)), n_units=np.array([m.sum() for m in sc.values()]), pool_boot=pool_b.astype(np.float32))
    d["point"] = np.array([pt[m].mean(0) for m in sc.values()]); d["lo"] = np.array([np.percentile(bt[m].mean(0), 2.5, axis=0) for m in sc.values()])
    d["hi"] = np.array([np.percentile(bt[m].mean(0), 97.5, axis=0) for m in sc.values()]); d["counts"] = np.array([ct[m].mean(0) for m in sc.values()])
    np.savez_compressed(f"{P}/rows/{r['id']}.npz", **d)
# ---------------------------------------------------------------- commands
def selftest():
    from sklearn import metrics as sm
    rows = {r["id"]: r for r in plan()}
    for rid in ("cls_main_best_plain", "q_B_vqc", "fl_fedavg_lab0.5"):
        e, s, te, p = rows[rid]["fn"]()[3]; p = p.astype(np.float64); yt = y_all[te]; pr = (p >= 0.5).astype(int); t0 = time.time()
        pt, ct = metrics(p, yt, np.ones((1, len(te)))); pt = pt[0]; tn, fp, fn, tp = sm.confusion_matrix(yt, pr).ravel()
        ref = [sm.accuracy_score(yt, pr), sm.balanced_accuracy_score(yt, pr), sm.precision_score(yt, pr), sm.recall_score(yt, pr), tn / (tn + fp), tn / (tn + fn), sm.f1_score(yt, pr),
               sm.matthews_corrcoef(yt, pr), sm.cohen_kappa_score(yt, pr), sm.roc_auc_score(yt, p), sm.average_precision_score(yt, p), sm.brier_score_loss(yt, p), sm.log_loss(yt, np.clip(p, 1e-15, 1 - 1e-15))]
        print(rid, "max |ours - sklearn| =", float(np.abs(pt - np.array(ref)).max()), "ties:", len(p) - len(np.unique(p)), "counts", ct[0], (tp, fp, tn, fn))
        # weights = integer duplicates equals repeating rows
        w = np.random.RandomState(1).randint(0, 3, size=len(te)); b, _ = metrics(p, yt, w[None].astype(float)); rep = np.repeat(np.arange(len(te)), w)
        pr2 = pr[rep]; y2 = yt[rep]; p2 = p[rep]
        ref2 = [sm.matthews_corrcoef(y2, pr2), sm.roc_auc_score(y2, p2), sm.average_precision_score(y2, p2), sm.cohen_kappa_score(y2, pr2)]
        print("   weighted vs row-repeat:", np.abs(b[0][[7, 9, 10, 8]] - np.array(ref2)).max())
        t0 = time.time(); unit_stats(p, te); print("   one unit bootstrap s:", round(time.time() - t0, 2))
def run(budget=140):
    t0 = time.time(); rs = plan(); todo = [r for r in rs if not os.path.exists(f"{P}/rows/{r['id']}.npz")]; print(f"{len(rs)} rows, {len(todo)} to do", flush=True)
    for r in todo:
        if time.time() - t0 > budget: print("budget reached"); break
        run_row(r); print("done", r["id"], round(time.time() - t0, 1), flush=True)
    print("remaining", len([r for r in rs if not os.path.exists(f"{P}/rows/{r['id']}.npz")]))
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run_ = 0
    for k, i in enumerate(o): run_ = max(run_, (m - k) * p[i]); adj[i] = min(1.0, run_)
    return adj
def assemble():
    rs = plan(); meta = {r["id"]: r for r in rs}; mrows, erows, crows = [], [], []; boots = {}
    for r in rs:
        f = f"{P}/rows/{r['id']}.npz"
        if not os.path.exists(f): print("missing", r["id"]); continue
        d = np.load(f, allow_pickle=True); boots[r["id"]] = (d["pool_boot"].astype(np.float64), d["point"][0]); encs = list(d["encs"])
        for i, e in enumerate(encs):
            base = dict(track=r["track"], row_id=r["id"], label=r["label"], setting=r["setting"], partition=r["partition"], scope=e, n_test_sets=int(d["n_units"][i]))
            row = dict(base); [row.update({m: d["point"][i][k], m + "_lo": d["lo"][i][k], m + "_hi": d["hi"][i][k]}) for k, m in enumerate(M)]
            (mrows if e == "pooled" else erows).append(row)
            cr = dict(base, TP=d["counts"][i][0], FP=d["counts"][i][1], TN=d["counts"][i][2], FN=d["counts"][i][3]); crows.append(cr)
    pd.DataFrame(mrows).to_csv(f"{T}/performance_metrics.csv", index=False); pd.DataFrame(erows).to_csv(f"{T}/performance_per_encoder.csv", index=False)
    cf = pd.DataFrame(crows); cf["n"] = cf[["TP", "FP", "TN", "FN"]].sum(1); cf["sens"] = cf.TP / (cf.TP + cf.FN); cf["spec"] = cf.TN / (cf.TN + cf.FP)
    cf["fp_rate"] = cf.FP / (cf.FP + cf.TN); cf["fn_rate"] = cf.FN / (cf.FN + cf.TP); cf.to_csv(f"{T}/performance_confusion.csv", index=False)
    C = [("tuning", "cls_tuned_best_plain", "cls_main_best_plain"), ("tuning", "cls_tuned_mnv2_plain", "cls_main_mnv2_plain"), ("tuning", "cls_ens_all11_plain", "cls_main_best_plain"),
         ("federation", "fl_fedavg_iid", "fl_central_all_iid"), ("federation", "fl_fedavg_lab0.5", "fl_central_all_iid"), ("federation", "fl_fedavg_lab0.1", "fl_central_all_iid"), ("federation", "fl_fedavg_comb0.5", "fl_central_all_iid")] + \
        [("privacy", f"fl_fedavg_dp{s}_lab0.5", "fl_fedavg_lab0.5") for s in SIGMAS] + \
        [("quantum", "q_B_vqc_pairs", "q_B_identity"), ("quantum", "q_B_vqc_pairs", "q_B_vqc"), ("quantum", "q_A_vqc_pairs", "q_A_classical"), ("quantum", "q_A_vqc_pairs", "q_A_vqc_pairs_derg")]
    out = []; MARG = {"mcc": 0.01, "acc": 0.005, "auc": 0.002}
    for fam_, a, b in C:
        if a not in boots or b not in boots: continue
        for m in MARG:
            k = M.index(m); da = boots[a][0][:, k] - boots[b][0][:, k]; dp = boots[a][1][k] - boots[b][1][k]; lo, hi = np.percentile(da, [2.5, 97.5])
            p = min(1.0, 2 * (1 + min((da <= 0).sum(), (da >= 0).sum())) / (BOOT + 1)); mg = MARG[m]
            out.append(dict(family=fam_, metric=m, a=a, b=b, diff=dp, lo=lo, hi=hi, boot_p=p, margin=mg, call="better" if lo > 0 else "worse" if hi < 0 else ("equivalent" if (lo > -mg and hi < mg) else "inconclusive")))
    cd = pd.DataFrame(out); cd["holm_p"] = np.nan
    for (fam_, m), idx in cd.groupby(["family", "metric"]).groups.items(): cd.loc[idx, "holm_p"] = holm(cd.loc[idx, "boot_p"].values)
    cd.to_csv(f"{T}/performance_contrasts.csv", index=False)
    print(len(mrows), "pooled rows,", len(erows), "per-encoder rows,", len(cd), "contrast rows")
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"selftest": selftest, "run": lambda: run(float(sys.argv[2]) if len(sys.argv) > 2 else 140), "assemble": assemble}[cmd]()
