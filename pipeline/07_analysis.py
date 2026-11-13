#!/usr/bin/env python3
"""Stage 9: statistical inference from stored per-item predictions.
Primary: group-cluster bootstrap (resample whole g0 groups with all their items across all seeds).
Partition contrast: encoder is the unit (exact sign test + Wilcoxon signed-rank), Holm within family.
Every table is written from the stored result files; nothing is typed by hand."""
import os, json, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
B_ = os.environ.get("PCOS_BENCH_ROOT", os.getcwd()); R = f"{B_}/results"; H = f"{R}/heads"; T = f"{R}/tables"
man = pd.read_csv(f"{B_}/data/manifest.csv"); spl = pd.read_csv(f"{B_}/data/splits.csv")
y = man.label.values; g = pd.factorize(man.g0)[0]; G = g.max() + 1; N = len(man)
SEEDS = [0, 1, 2, 3, 4]; PRE = {"group_g0": "grp", "group_g4": "g4", "naive_image": "img"}
FAM = {"mobilenetv2_100": "CNN", "efficientnet_b0": "CNN", "densenet121": "CNN", "resnet50": "CNN", "resnext50_32x4d": "CNN",
       "pvt_v2_b2": "Transformer", "twins_pcpvt_small": "Transformer", "swin_tiny_patch4_window7_224": "Transformer",
       "crossvit_small_240": "Transformer", "mobilevit_s": "Hybrid", "eca_botnext26ts_256": "Hybrid"}
VARS = ["plain", "clahe", "neutral"]; RULES = list(PRE)
BOOT = 1000; rng = np.random.RandomState(2026)
W = rng.multinomial(G, np.full(G, 1.0 / G), size=BOOT).astype(np.float64)       # (BOOT, G) group multiplicities

def mcc(tp, tn, fp, fn):
    d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)); return np.where(d > 0, (tp * tn - fp * fn) / np.where(d > 0, d, 1), 0.0)
def boot_metrics(probs, rule):
    """Per-draw metrics averaged over seeds. probs: (5, N) with NaN outside test."""
    acc = np.zeros(BOOT); auc = np.zeros(BOOT); mc = np.zeros(BOOT); f1 = np.zeros(BOOT)
    for si, s in enumerate(SEEDS):
        te = np.nonzero(~np.isnan(probs[si]))[0]; p = probs[si, te]; yt = y[te]; Wi = W[:, g[te]]
        pred = (p >= 0.5).astype(int)
        tp = Wi @ ((pred == 1) & (yt == 1)); tn = Wi @ ((pred == 0) & (yt == 0)); fp = Wi @ ((pred == 1) & (yt == 0)); fn = Wi @ ((pred == 0) & (yt == 1))
        acc += (tp + tn) / (tp + tn + fp + fn); mc += mcc(tp, tn, fp, fn); f1 += 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
        o = np.argsort(p, kind="mergesort"); ys = yt[o]; Ws = Wi[:, o]
        pos = Ws * ys; neg = Ws * (1 - ys); cn = np.cumsum(neg, axis=1) - neg
        auc += (pos * cn).sum(1) / (pos.sum(1) * neg.sum(1))
    n = len(SEEDS); return acc / n, auc / n, mc / n, f1 / n

ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
rows = []; boots = {}
for enc, fam in FAM.items():
    for var in VARS:
        for rule in RULES:
            d = np.load(f"{H}/{enc}__{var}__{rule}.npz"); met = pd.DataFrame(json.loads(str(d["metrics"])))
            ba, bu, bm, bf = boot_metrics(d["probs"], rule); boots[(enc, var, rule)] = (ba, bu)
            rows.append(dict(encoder=enc, family=fam, variant=var, rule=rule,
                acc=met.acc.mean(), acc_seed_sd=met.acc.std(ddof=1), acc_lo=ci(ba)[0], acc_hi=ci(ba)[1],
                auc=met.auc.mean(), auc_lo=ci(bu)[0], auc_hi=ci(bu)[1], f1=met.f1.mean(), mcc=met.mcc.mean(),
                mcc_lo=ci(bm)[0], mcc_hi=ci(bm)[1], prec=met.prec.mean(), rec=met.rec.mean(), bal_acc=met.bal_acc.mean(),
                C_median=float(np.median(d["C"]))))
main = pd.DataFrame(rows); main.to_csv(f"{T}/main_results.csv", index=False)

# ---- baselines
bl = pd.read_csv(f"{R}/baselines_metadata.csv"); blm = bl.groupby(["model", "rule"])[["acc", "auc", "f1", "mcc"]].agg(["mean", "std"]).round(4)
blm.columns = ["_".join(c) for c in blm.columns]; blm.reset_index().to_csv(f"{T}/baselines_metadata_summary.csv", index=False)
perm = pd.read_csv(f"{R}/baselines_label_permuted.csv"); perm.groupby("encoder")[["acc", "auc"]].mean().round(4).to_csv(f"{T}/baselines_label_permuted_summary.csv")

# ---- partition contrast: encoder as the unit
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj
famA = []
for var in VARS:
    for a, b, name in (("naive_image", "group_g0", "naive - group_g0"), ("group_g0", "group_g4", "group_g0 - group_g4")):
        da = main[(main.variant == var) & (main.rule == a)].set_index("encoder").acc
        db = main[(main.variant == var) & (main.rule == b)].set_index("encoder").acc.reindex(da.index)
        dif = (da - db); npos = int((dif > 0).sum()); n = len(dif)
        sign_p = stats.binomtest(npos, n, 0.5).pvalue
        wil_p = stats.wilcoxon(dif.values).pvalue
        # within-unit detectability: encoders whose paired cluster-bootstrap CI excludes 0
        det = 0
        for enc in dif.index:
            bd = boots[(enc, var, a)][0] - boots[(enc, var, b)][0]
            det += int(np.percentile(bd, 2.5) > 0 or np.percentile(bd, 97.5) < 0)
        famA.append(dict(family="A_partition", variant=var, contrast=name, n_encoders=n, mean_diff_pts=100 * dif.mean(),
                         encoders_positive=npos, sign_p=sign_p, wilcoxon_p=wil_p, encoders_ci_excludes_zero=det))
famB = []
for var, ref in (("clahe", "plain"), ("neutral", "plain")):
    da = main[(main.variant == var) & (main.rule == "group_g0")].set_index("encoder").acc
    db = main[(main.variant == ref) & (main.rule == "group_g0")].set_index("encoder").acc.reindex(da.index)
    dif = da - db; npos = int((dif > 0).sum()); n = len(dif)
    det = sum(int(np.percentile(boots[(e, var, "group_g0")][0] - boots[(e, ref, "group_g0")][0], 2.5) > 0 or
                  np.percentile(boots[(e, var, "group_g0")][0] - boots[(e, ref, "group_g0")][0], 97.5) < 0) for e in dif.index)
    famB.append(dict(family="B_preprocessing", variant=var, contrast=f"{var} - {ref} (group_g0)", n_encoders=n, mean_diff_pts=100 * dif.mean(),
                     encoders_positive=npos, sign_p=stats.binomtest(npos, n, 0.5).pvalue, wilcoxon_p=stats.wilcoxon(dif.values).pvalue,
                     encoders_ci_excludes_zero=det))
tests = pd.DataFrame(famA + famB)
for f in tests.family.unique():
    m = tests.family == f
    tests.loc[m, "sign_p_holm"] = holm(tests.loc[m, "sign_p"]); tests.loc[m, "wilcoxon_p_holm"] = holm(tests.loc[m, "wilcoxon_p"])
tests.round(6).to_csv(f"{T}/partition_and_preprocessing_tests.csv", index=False)

# ---- gap to the no-pixel baseline
rf = bl[(bl.model == "meta_size_only_RF")].groupby("rule").acc.mean()
gap = main[main.variant == "plain"].assign(gap_to_size_only_RF_pts=lambda d: 100 * (d.acc - d.rule.map(rf)))
gap[["encoder", "rule", "acc", "gap_to_size_only_RF_pts"]].round(4).to_csv(f"{T}/gap_to_size_only_baseline.csv", index=False)

# ---- confusion matrices (group_g0, plain, pooled over seeds), best encoder by AUC
best = main[(main.rule == "group_g0") & (main.variant == "plain")].sort_values("auc", ascending=False).iloc[0].encoder
d = np.load(f"{H}/{best}__plain__group_g0.npz")["probs"]
cm = np.zeros((2, 2), int)
for si in range(5):
    te = np.nonzero(~np.isnan(d[si]))[0]; pr = (d[si, te] >= 0.5).astype(int)
    for t_, p_ in zip(y[te], pr): cm[t_, p_] += 1

pd.set_option("display.width", 200)
print("== best encoder (group_g0, plain, by AUC):", best, "pooled confusion [[TN FP],[FN TP]] =", cm.tolist())
print("\n== accuracy % (mean over 5 seeds) by encoder x variant, group_g0 | naive")
piv = main.pivot_table(index="encoder", columns=["rule", "variant"], values="acc") * 100
print(piv.round(2).to_string())
print("\n== partition & preprocessing tests"); print(tests[["family", "variant", "contrast", "mean_diff_pts", "encoders_positive", "sign_p_holm", "wilcoxon_p_holm", "encoders_ci_excludes_zero"]].round(4).to_string(index=False))
print("\n== baselines (acc mean)"); print(blm[["acc_mean", "acc_std", "auc_mean"]].to_string())
print("\nlabel-permuted mean acc/auc:", perm[["acc", "auc"]].mean().round(4).to_dict())
json.dump({"best_encoder_group_g0_plain": best, "pooled_confusion": cm.tolist(), "boot": BOOT, "n_groups": int(G)}, open(f"{R}/stage9_meta.json", "w"))
