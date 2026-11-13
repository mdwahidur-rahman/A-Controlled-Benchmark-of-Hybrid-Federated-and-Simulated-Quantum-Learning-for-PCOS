#!/usr/bin/env python3
"""Stage 4 baselines measured under the identical protocol (same splits, 5 seeds):
 (1) majority-class prior, (2) metadata-only heads (image size / compression / intensity statistics),
 (3) label-permuted encoder heads (labels shuffled across GROUPS in train+val; chance control).
Metadata features use no pixel content beyond mean/std intensity."""
import os, json, time, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
warnings.filterwarnings("ignore")
B = os.environ.get("PCOS_BENCH_ROOT", os.getcwd()); F = f"{B}/features"; R = f"{B}/results"
man = pd.read_csv(f"{B}/data/manifest.csv"); spl = pd.read_csv(f"{B}/data/splits.csv")
y = man.label.values; N = len(man); SEEDS = [0, 1, 2, 3, 4]
PRE = {"group_g0": "grp", "group_g4": "g4", "naive_image": "img"}
man["aspect"] = man.width / man.height; man["bpp"] = man.bytes / (man.width * man.height); man["lbytes"] = np.log(man.bytes)
SETS = {"meta_size_only": ["width", "height", "aspect"],
        "meta_all": ["width", "height", "aspect", "lbytes", "bpp", "q0", "mean", "std"]}
def met(yt, p):
    yh = (p >= 0.5).astype(int)
    return dict(acc=accuracy_score(yt, yh), auc=roc_auc_score(yt, p), f1=f1_score(yt, yh), mcc=matthews_corrcoef(yt, yh))
rows = []
for rule, pre in PRE.items():
    for s in SEEDS:
        part = spl[f"{pre}_s{s}"].values
        tr, va, te = (np.nonzero(part == k)[0] for k in ("train", "val", "test")); trva = np.concatenate([tr, va])
        # majority prior
        pm = np.full(len(te), y[trva].mean()); m0 = met(y[te], np.clip(pm + 1e-9 * np.arange(len(te)), 0, 1))
        m0["auc"] = 0.5; rows.append(dict(model="majority", rule=rule, seed=s, **m0))
        for name, cols in SETS.items():
            X = man[cols].values.astype(float)
            sc = StandardScaler().fit(X[trva])
            lr = LogisticRegression(C=1.0, max_iter=500).fit(sc.transform(X[trva]), y[trva])
            rows.append(dict(model=f"{name}_LR", rule=rule, seed=s, **met(y[te], lr.predict_proba(sc.transform(X[te]))[:, 1])))
            rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=3, random_state=s).fit(X[trva], y[trva])
            rows.append(dict(model=f"{name}_RF", rule=rule, seed=s, **met(y[te], rf.predict_proba(X[te])[:, 1])))
    print(rule, "done", flush=True)
meta = pd.DataFrame(rows); meta.to_csv(f"{R}/baselines_metadata.csv", index=False)

# label-permuted heads (group-level shuffle of train+val labels), plain variant, group_g0 rule
uidx = pd.read_csv(f"{F}/unique_index.csv"); pos = {m: i for i, m in enumerate(uidx.md5)}; r2u = man.md5.map(pos).values
ENCS = ["mobilenetv2_100", "efficientnet_b0", "densenet121", "resnet50", "pvt_v2_b2", "twins_pcpvt_small",
        "swin_tiny_patch4_window7_224", "mobilevit_s", "eca_botnext26ts_256", "crossvit_small_240", "resnext50_32x4d"]
prow = []
for enc in ENCS:
    Xu = np.load(f"{F}/{enc}__plain.npz")["feat"].astype(np.float32)
    for s in SEEDS:
        part = spl[f"grp_s{s}"].values; g = man.g0.values
        tr, va, te = (np.nonzero(part == k)[0] for k in ("train", "val", "test")); trva = np.concatenate([tr, va])
        rng = np.random.RandomState(100 + s); ug = np.unique(g[trva]); lab_g = {k: y[trva][g[trva] == k][0] for k in ug}
        shuf = dict(zip(ug, rng.permutation([lab_g[k] for k in ug])))
        yp = np.array([shuf[k] for k in g[trva]])
        X = Xu[r2u]; sc = StandardScaler().fit(X[trva])
        lr = LogisticRegression(C=0.03, max_iter=200, tol=1e-3).fit(sc.transform(X[trva]), yp)
        prow.append(dict(encoder=enc, seed=s, **met(y[te], lr.predict_proba(sc.transform(X[te]))[:, 1])))
    print(enc, "perm done", flush=True)
pd.DataFrame(prow).to_csv(f"{R}/baselines_label_permuted.csv", index=False)
t = meta.groupby(["model", "rule"])[["acc", "auc", "mcc"]].mean().round(4)
print(t.to_string())
print("label-permuted (group_g0, plain) mean acc/auc:", pd.DataFrame(prow)[["acc", "auc"]].mean().round(4).to_dict())
