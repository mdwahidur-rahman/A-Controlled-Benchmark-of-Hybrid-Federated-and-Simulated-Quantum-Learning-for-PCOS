#!/usr/bin/env python3
"""Calibration add-on: expected calibration error (ECE, 10 equal-width probability bins, positive-class probability) per representative row, with the same
group-cluster bootstrap (B=1000, shared draws) and the same pooling convention as 23_performance.py (metric per test set, then averaged over test sets).
Also writes pooled reliability-bin tables for the six rows shown in the figure. Reuses the loaders of 23_performance.py unchanged.
Usage: PCOS_BENCH_ROOT=<repo root> python3 pipeline/28_calibration.py   (needs the per-image probability files; writes results/tables/calibration_*.csv)"""
import os, sys, importlib.util, numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location("perf", os.path.join(os.environ.get("PCOS_BENCH_ROOT", os.getcwd()), "pipeline", "23_performance.py"))
perf = importlib.util.module_from_spec(spec); spec.loader.exec_module(perf)
NB = 10
SEL = ["cls_main_best_plain", "cls_tuned_best_plain", "cls_ens_all11_plain", "cls_main_best_neutral", "fl_central_all_iid", "fl_fedavg_iid", "fl_fedavg_lab0.5", "fl_fedavg_lab0.1",
       "fl_fedavg_dp4.0_lab0.5", "q_A_classical", "q_A_vqc_pairs", "q_B_identity", "q_B_vqc_pairs", "q_B_vqc"]
FIG = ["cls_main_best_plain", "fl_fedavg_iid", "fl_fedavg_dp4.0_iid", "q_B_identity", "q_B_vqc_pairs", "q_B_vqc"]
def ece_w(p, yt, Wi):
    """Weighted ECE for K weight vectors Wi (K, n): sum_b |sum_w(y - p) in bin b| / sum_w."""
    b = np.minimum((p * NB).astype(int), NB - 1); tot = np.zeros(Wi.shape[0])
    for k in range(NB):
        m = (b == k); tot += np.abs(Wi @ ((yt - p) * m))
    return tot / Wi.sum(1)
def stats(us):
    pt, bt = [], []
    for _, _, te, p in us:
        p = p.astype(np.float64); yt = perf.y_all[te].astype(np.float64)
        pt.append(ece_w(p, yt, np.ones((1, len(te))))[0]); bt.append(ece_w(p, yt, perf.Wsub(te)))
    pt = np.array(pt); bt = np.array(bt); pool = bt.mean(0)
    return pt.mean(), np.percentile(pool, 2.5), np.percentile(pool, 97.5), len(us), pool
def bins(us):
    P = np.concatenate([u[3].astype(np.float64) for u in us]); Y = np.concatenate([perf.y_all[u[2]].astype(np.float64) for u in us]); b = np.minimum((P * NB).astype(int), NB - 1); out = []
    for k in range(NB):
        m = b == k; out.append(dict(bin=k, lo=k / NB, hi=(k + 1) / NB, n=int(m.sum()), mean_p=float(P[m].mean()) if m.any() else np.nan, frac_pos=float(Y[m].mean()) if m.any() else np.nan))
    return out
if __name__ == "__main__":
    rows = {r["id"]: r for r in perf.plan()}; res = []; bl = []; pools = {}
    for rid in sorted(set(SEL) | set(FIG)):
        us = rows[rid]["fn"](); e, lo, hi, n, pool = stats(us); pools[rid] = (pool, e); res.append(dict(row_id=rid, label=rows[rid]["label"], n_test_sets=n, ece=e, ece_lo=lo, ece_hi=hi)); print(rid, round(e, 4), round(lo, 4), round(hi, 4), n, flush=True)
        if rid in FIG: bl += [dict(row_id=rid, **d) for d in bins(us)]
    C = [("tuning", "cls_tuned_best_plain", "cls_main_best_plain"), ("federation", "fl_fedavg_iid", "fl_central_all_iid"), ("federation", "fl_fedavg_lab0.5", "fl_central_all_iid"), ("federation", "fl_fedavg_lab0.1", "fl_central_all_iid"),
         ("privacy", "fl_fedavg_dp4.0_lab0.5", "fl_fedavg_lab0.5"), ("quantum", "q_B_vqc_pairs", "q_B_identity"), ("quantum", "q_B_vqc_pairs", "q_B_vqc"), ("quantum", "q_A_vqc_pairs", "q_A_classical")]; cr = []
    for fam, a, b in C:
        d = pools[a][0] - pools[b][0]; lo, hi = np.percentile(d, [2.5, 97.5]); cr.append(dict(family=fam, a=a, b=b, diff=pools[a][1] - pools[b][1], lo=lo, hi=hi, call="higher" if lo > 0 else "lower" if hi < 0 else "inconclusive"))
    T = f"{perf.R}/tables"; pd.DataFrame(cr).to_csv(f"{T}/calibration_contrasts.csv", index=False); print(pd.DataFrame(cr).round(4).to_string()); pd.DataFrame(res).to_csv(f"{T}/calibration_ece.csv", index=False); pd.DataFrame(bl).to_csv(f"{T}/calibration_bins.csv", index=False)
    # self-test: unweighted ECE vs an independent loop implementation on one row
    us = rows["cls_main_best_plain"]["fn"](); _, _, te, p = us[0]; p = p.astype(float); y = perf.y_all[te]; tot = 0.0
    for k in range(NB):
        m = (p >= k / NB) & (p < (k + 1) / NB) if k < NB - 1 else (p >= k / NB)
        if m.any(): tot += m.sum() * abs(y[m].mean() - p[m].mean())
    print("selftest ECE loop vs vectorized:", tot / len(p), ece_w(p, y.astype(float), np.ones((1, len(te))))[0])
