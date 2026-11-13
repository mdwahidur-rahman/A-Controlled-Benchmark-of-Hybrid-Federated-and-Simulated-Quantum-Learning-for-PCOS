#!/usr/bin/env python3
"""Classical subsection: rigorous hyperparameter tuning of the head on frozen encoder features.
Per (encoder, variant, seed) on the group-disjoint g0 split: 26 trials spanning three head families
  LR-std   : standardize -> L2 logistic regression, C in 10^{-4..1} step 0.5          (11 trials)
  LR-pca   : standardize -> PCA(256) -> L2 logistic regression, C in 10^{-3..1}       ( 9 trials)
  MLP      : standardize -> 1 hidden layer {64, 256}, L2 alpha in {0.1, 1, 10}         ( 6 trials)
Each trial is fit on TRAIN ONLY (scaler/PCA on train) and scored on VALIDATION. Selection = highest validation
AUC, ties (<5e-4) broken by lowest validation log-loss. The selected configuration is refit on train+val and
scored on test. Test metrics of every train-only trial are stored as a sensitivity surface; they never select.
Duplicates are collapsed with integer sample weights. Usage: 14_tuning.py WORKERS [--variants plain,neutral]"""
import os, sys, json, time, warnings, itertools
import numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, f1_score, matthews_corrcoef, balanced_accuracy_score
B = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); F = f"{B}/features"; OUT = f"{B}/results/tuning"
ENCS = ["mobilenetv2_100", "efficientnet_b0", "densenet121", "resnet50", "pvt_v2_b2", "twins_pcpvt_small",
        "swin_tiny_patch4_window7_224", "mobilevit_s", "eca_botnext26ts_256", "crossvit_small_240", "resnext50_32x4d"]
SEEDS = [0, 1, 2, 3, 4]
man = pd.read_csv(f"{B}/data/manifest.csv"); spl = pd.read_csv(f"{B}/data/splits.csv")
uidx = pd.read_csv(f"{F}/unique_index.csv"); pos = {m: i for i, m in enumerate(uidx.md5)}
row2u = man.md5.map(pos).values; y = man.label.values; N = len(man)
TRIALS = ([("lr_std", "std", dict(C=float(10 ** e))) for e in np.arange(-4, 1.01, 0.5)] +
          [("lr_pca", "pca256", dict(C=float(10 ** e))) for e in np.arange(-3, 1.01, 0.5)] +
          [("mlp", "std", dict(h=h, alpha=a)) for h in (64, 256) for a in (0.1, 1.0, 10.0)])

def collapse(rows):
    u, inv = np.unique(row2u[rows], return_inverse=True)
    w = np.bincount(inv); lab = np.zeros(len(u), int); lab[inv] = y[rows]
    return u, w, lab

def prep_fit(prep, Xu, tr):
    sc = StandardScaler().fit(Xu[row2u[tr]])
    pca = None
    if prep == "pca256":
        utr, _, _ = collapse(tr); pca = PCA(256, random_state=0).fit(sc.transform(Xu[utr]))
    return lambda idx: (pca.transform(sc.transform(Xu[idx])) if pca is not None else sc.transform(Xu[idx]))

def make(fam, hp, seed):
    if fam.startswith("lr"): return LogisticRegression(C=hp["C"], max_iter=300, tol=1e-3)
    return MLPClassifier(hidden_layer_sizes=(hp["h"],), alpha=hp["alpha"], max_iter=100, random_state=seed, tol=1e-4)

def fit_predict(fam, prep, hp, Xu, fit_rows, pred_rows_list, seed):
    tf = prep_fit(prep, Xu, fit_rows); u, w, lab = collapse(fit_rows)
    m = make(fam, hp, seed)
    try: m.fit(tf(u), lab, sample_weight=w)
    except TypeError: m.fit(tf(u), lab)
    return [m.predict_proba(tf(row2u[r]))[:, 1] for r in pred_rows_list]

def metrics(yt, p):
    yh = (p >= 0.5).astype(int)
    return dict(acc=accuracy_score(yt, yh), auc=roc_auc_score(yt, p), f1=f1_score(yt, yh), mcc=matthews_corrcoef(yt, yh), bal_acc=balanced_accuracy_score(yt, yh))

def task(args):
    enc, var = args
    out = f"{OUT}/{enc}__{var}.npz"
    if os.path.exists(out): return enc, var, "skip", 0.0
    t0 = time.time(); d = np.load(f"{F}/{enc}__{var}.npz"); Xu = d["feat"].astype(np.float32)
    probs = np.full((len(SEEDS), N), np.nan, np.float32); sel = []; trials = []; met = []
    for si, s in enumerate(SEEDS):
        part = spl[f"grp_s{s}"].values; tr, va, te = (np.nonzero(part == k)[0] for k in ("train", "val", "test"))
        best = None
        for ti, (fam, prep, hp) in enumerate(TRIALS):
            pv, pt = fit_predict(fam, prep, hp, Xu, tr, [va, te], s)
            pv = np.clip(pv, 1e-6, 1 - 1e-6)
            rec = dict(enc=enc, var=var, seed=s, trial=ti, family=fam, prep=prep, hp=json.dumps(hp), val_auc=roc_auc_score(y[va], pv),
                       val_ll=log_loss(y[va], pv), val_acc=accuracy_score(y[va], (pv >= .5).astype(int)),
                       test_auc=roc_auc_score(y[te], pt), test_acc=accuracy_score(y[te], (pt >= .5).astype(int)))
            trials.append(rec)
            key = (round(rec["val_auc"] / 5e-4), -rec["val_ll"])
            if best is None or key > best[0]: best = (key, ti)
        fam, prep, hp = TRIALS[best[1]]
        trva = np.concatenate([tr, va]); pt, = fit_predict(fam, prep, hp, Xu, trva, [te], s)
        probs[si, te] = pt; sel.append(dict(seed=s, trial=best[1], family=fam, prep=prep, hp=hp)); met.append(metrics(y[te], pt))
    pd.DataFrame(trials).to_csv(f"{OUT}/{enc}__{var}__trials.csv", index=False)
    np.savez(out, probs=probs, selected=json.dumps(sel), metrics=json.dumps(met))
    return enc, var, f"acc={np.mean([m['acc'] for m in met]):.4f} auc={np.mean([m['auc'] for m in met]):.4f} sel={[s['family'] for s in sel]}", time.time() - t0

if __name__ == "__main__":
    nw = int(sys.argv[1]); vs = sys.argv[sys.argv.index("--variants") + 1].split(",") if "--variants" in sys.argv else ["plain"]
    jobs = [(e, v) for v in vs for e in ENCS]
    with Pool(nw) as p:
        for r in p.imap_unordered(task, jobs):
            print(f"{r[0]:30s} {r[1]:8s} {r[2]}  ({r[3]:.0f}s)", flush=True)
    print("ALL DONE", flush=True)
