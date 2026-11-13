#!/usr/bin/env python3
"""Summaries for the XAI arms: per (arm, method) deletion/insertion AUC vs random pixel-order and random 7x7-block-order baselines (paired over items,
bootstrap over items B=2000), constant-map rate, border mass, and cross-arm map agreement (Spearman on 28x28 pooled maps)."""
import os, json, numpy as np, pandas as pd
B_ = os.environ.get("PCOS_BENCH_ROOT", os.getcwd()); R = f"{B_}/results"; T = f"{R}/tables"
D = json.load(open(f"{R}/xai_arms/faith_items.json")); items = list(D.values()); n = len(items); rs = np.random.RandomState(7); idx = rs.randint(0, n, size=(2000, n))
ARMS = ["classical", "fl", "fl_dp", "quantum"]; METH = ["gradcam", "gradcampp", "eigencam", "scorecam", "occlusion", "ig"]
ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
rows = []
for a in ARMS:
    rd = np.array([it["random"][a]["d"] for it in items]); ri = np.array([it["random"][a]["i"] for it in items]); cd = np.array([it["random"][a]["cd"] for it in items]); cim = np.array([it["random"][a]["ci"] for it in items])
    rows.append(dict(arm=a, method="random pixels", n=n, deletion=rd.mean(), insertion=ri.mean(), ins_minus_del=(ri - rd).mean(), del_vs_random=0.0, ins_vs_random=0.0, const_rate=0.0, border_mass=np.nan))
    rows.append(dict(arm=a, method="random 7x7 blocks", n=n, deletion=cd.mean(), insertion=cim.mean(), ins_minus_del=(cim - cd).mean(), del_vs_random=0.0, ins_vs_random=0.0, const_rate=0.0, border_mass=np.nan))
    for m in METH:
        d = np.array([it["faith"][a][m]["del"] for it in items]); i = np.array([it["faith"][a][m]["ins"] for it in items])
        dd = d - rd; ii = i - ri; bd = ci(dd[idx].mean(1)); bi = ci(ii[idx].mean(1)); bs = ci((i - d)[idx].mean(1))
        rows.append(dict(arm=a, method=m, n=n, deletion=d.mean(), insertion=i.mean(), ins_minus_del=(i - d).mean(), ins_minus_del_lo=bs[0], ins_minus_del_hi=bs[1],
                         del_vs_random=dd.mean(), del_vs_random_lo=bd[0], del_vs_random_hi=bd[1], ins_vs_random=ii.mean(), ins_vs_random_lo=bi[0], ins_vs_random_hi=bi[1],
                         ins_vs_blockrandom=(i - cim).mean(), del_vs_blockrandom=(d - cd).mean(),
                         const_rate=float(np.mean([it["faith"][a][m]["const"] for it in items])), border_mass=float(np.mean([it["faith"][a][m]["border"] for it in items]))))
F = pd.DataFrame(rows); F.to_csv(f"{T}/xai_faithfulness.csv", index=False)
ag = []
keys = sorted({k for it in items for k in it["agree"]})
for k in keys:
    v = [it["agree"][k] for it in items if it["agree"].get(k) is not None]; m, a, b = k.split("|"); lo, hi = ci(np.array(v)[rs.randint(0, len(v), size=(2000, len(v)))].mean(1)) if v else (np.nan, np.nan)
    ag.append(dict(method=m, arm_a=a, arm_b=b, spearman=float(np.mean(v)) if v else np.nan, lo=lo, hi=hi, n=len(v)))
pd.DataFrame(ag).to_csv(f"{T}/xai_agreement.csv", index=False)
pl = pd.DataFrame([dict(arm=a, mean_live_target_prob=np.mean([it["plive"][a] for it in items])) for a in ARMS]); pl.to_csv(f"{T}/xai_live_probs.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(F[["arm", "method", "deletion", "insertion", "ins_minus_del", "del_vs_random", "ins_vs_random", "const_rate", "border_mass"]].round(3).to_string()); print(pd.DataFrame(ag).round(2).to_string()); print(pl.round(3).to_string())
