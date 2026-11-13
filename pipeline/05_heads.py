#!/usr/bin/env python3
"""Stage 4: L2 logistic-regression heads on frozen features, under each partitioning rule.
Protocol per (encoder, variant, rule, seed): scaler fit on train only; C chosen on validation
AUC; scaler+head refit on train+val; metrics on test. Byte-identical copies are collapsed to one
row with an integer sample weight (an exactly equivalent training objective, ~3x cheaper).
Usage: 05_heads.py BUDGET_SECONDS [--rules r1,r2] [--variants v1,v2] [--encoders e1,e2]"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, balanced_accuracy_score
warnings.filterwarnings("ignore")
B = os.environ.get("PCOS_BENCH_ROOT", os.getcwd()); F = f"{B}/features"; R = f"{B}/results/heads"
BUDGET = float(sys.argv[1]); T0 = time.time()
def arg(k, default):
    return sys.argv[sys.argv.index(k) + 1].split(",") if k in sys.argv else default
ENCS = arg("--encoders", ["mobilenetv2_100", "efficientnet_b0", "densenet121", "resnet50", "pvt_v2_b2", "twins_pcpvt_small",
        "swin_tiny_patch4_window7_224", "mobilevit_s", "eca_botnext26ts_256", "crossvit_small_240", "resnext50_32x4d"])
VARS = arg("--variants", ["plain", "clahe", "neutral"]); RULES = arg("--rules", ["group_g0", "naive_image", "group_g4"])
PRE = {"group_g0": "grp", "group_g4": "g4", "naive_image": "img"}
CS = [0.003, 0.03, 0.3]; SEEDS = [0, 1, 2, 3, 4]

man = pd.read_csv(f"{B}/data/manifest.csv"); spl = pd.read_csv(f"{B}/data/splits.csv")
uidx = pd.read_csv(f"{F}/unique_index.csv"); pos = {m: i for i, m in enumerate(uidx.md5)}
row2u = man.md5.map(pos).values; y = man.label.values; N = len(man)

def metrics(yt, p):
    yh = (p >= 0.5).astype(int)
    return dict(acc=accuracy_score(yt, yh), auc=roc_auc_score(yt, p), f1=f1_score(yt, yh), prec=precision_score(yt, yh),
                rec=recall_score(yt, yh), mcc=matthews_corrcoef(yt, yh), bal_acc=balanced_accuracy_score(yt, yh))

def fit_pred(Xu, tr, va, te, seed_C=None):
    """Xu: features per unique file; tr/va/te: row indices. Collapse duplicates -> weighted unique rows."""
    def collapse(rows):
        u, inv = np.unique(row2u[rows], return_inverse=True)
        w = np.bincount(inv); lab = np.zeros(len(u), int); lab[inv] = y[rows]
        return u, w, lab
    utr, wtr, ltr = collapse(tr)
    sc = StandardScaler().fit(Xu[utr], ) if False else None
    # weighted scaler = scaler over the actual training rows (copies included)
    sc = StandardScaler().fit(Xu[row2u[tr]])
    Xtr = sc.transform(Xu[utr]); Xva = sc.transform(Xu[row2u[va]])
    best = (-1, None)
    for C in CS:
        m = LogisticRegression(C=C, max_iter=200, tol=1e-3).fit(Xtr, ltr, sample_weight=wtr)
        a = roc_auc_score(y[va], m.predict_proba(Xva)[:, 1])
        if a > best[0]: best = (a, C)
    trva = np.concatenate([tr, va]); utv, wtv, ltv = collapse(trva)
    sc2 = StandardScaler().fit(Xu[row2u[trva]])
    m = LogisticRegression(C=best[1], max_iter=200, tol=1e-3).fit(sc2.transform(Xu[utv]), ltv, sample_weight=wtv)
    return m.predict_proba(sc2.transform(Xu[row2u[te]]))[:, 1], best[1], best[0]

done_jobs = 0
for enc in ENCS:
    for var in VARS:
        fp_ = f"{F}/{enc}__{var}.npz"
        if not os.path.exists(fp_): continue
        d = np.load(fp_)
        if not d["done"].all(): continue
        Xu = d["feat"].astype(np.float32)
        for rule in RULES:
            out = f"{R}/{enc}__{var}__{rule}.npz"
            if os.path.exists(out): continue
            if time.time() - T0 > BUDGET - 25: break
            probs = np.full((len(SEEDS), N), np.nan, np.float32); Csel = []; vauc = []; met = []
            for si, s in enumerate(SEEDS):
                part = spl[f"{PRE[rule]}_s{s}"].values
                tr, va, te = (np.nonzero(part == k)[0] for k in ("train", "val", "test"))
                p, C, a = fit_pred(Xu, tr, va, te)
                probs[si, te] = p; Csel.append(C); vauc.append(a); met.append(metrics(y[te], p))
            np.savez(out, probs=probs, C=np.array(Csel), val_auc=np.array(vauc), metrics=json.dumps(met))
            done_jobs += 1
            m = pd.DataFrame(met).mean()
            print(f"{enc[:18]:18s} {var:8s} {rule:11s} acc={m.acc:.4f} auc={m.auc:.4f} f1={m.f1:.4f}  ({time.time()-T0:.0f}s)")
total = len(ENCS) * len(VARS) * len(RULES)
have = len([f for f in os.listdir(R) if f.endswith(".npz")])
print(f"heads files {have}/{total}  new this call {done_jobs}  elapsed {time.time()-T0:.0f}s")
