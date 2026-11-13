#!/usr/bin/env python3
"""Two small post-hoc summaries computed from stored outputs and written to results/tables:
  encoder_rank_stability.csv : Spearman rank correlation of the 11 encoder accuracies (g0 split, mean over seeds) between preprocessing variants;
  roc_operating_points.csv   : per encoder (plain, g0, main heads), mean over seeds of sensitivity at 95% and 99% specificity and of average precision, from stored per-image probabilities.
Usage: PCOS_BENCH_ROOT=<repo root> python3 32_extra_stats.py"""
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, average_precision_score
R = os.environ.get("PCOS_BENCH_ROOT", os.getcwd()); T = f"{R}/results/tables"
m = pd.read_csv(f"{T}/main_results.csv"); g = m[m.rule == "group_g0"]; a = {v: g[g.variant == v].set_index("encoder").acc for v in ("plain", "clahe", "neutral")}
rows = []
for x, y in (("plain", "clahe"), ("plain", "neutral"), ("clahe", "neutral")):
    r = spearmanr(a[x], a[y]); rows.append(dict(variant_a=x, variant_b=y, n_encoders=len(a[x]), spearman=r.statistic, p_value=r.pvalue))
pd.DataFrame(rows).to_csv(f"{T}/encoder_rank_stability.csv", index=False)
sp = pd.read_csv(f"{R}/data/splits.csv"); y = sp.label.values; out = []
for e in a["plain"].index:
    P = np.load(f"{R}/results/heads/{e}__plain__group_g0.npz")["probs"]; s95, s99, ap = [], [], []
    for k in range(5):
        t = (sp[f"grp_s{k}"] == "test").values; fpr, tpr, _ = roc_curve(y[t], P[k][t])
        s95.append(tpr[np.searchsorted(fpr, 0.05, side="right") - 1]); s99.append(tpr[np.searchsorted(fpr, 0.01, side="right") - 1]); ap.append(average_precision_score(y[t], P[k][t]))
    out.append(dict(encoder=e, sens_at_spec95=np.mean(s95), sens_at_spec99=np.mean(s99), avg_precision=np.mean(ap)))
pd.DataFrame(out).to_csv(f"{T}/roc_operating_points.csv", index=False); print(pd.DataFrame(rows).round(3).to_string()); print(pd.DataFrame(out).round(4).to_string())
