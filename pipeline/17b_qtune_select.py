#!/usr/bin/env python3
"""Select quantum-subsection hyperparameters from the tuning grid by mean VALIDATION AUC (3 seeds), per encoder, per condition.
Ties (5e-4) -> fewer slot parameters, then fewer qubits. Writes results/tables/quantum_tuning_{grid,selected}.csv and results/quantum_final_plan.json."""
import os, re, json, glob
import numpy as np, pandas as pd
B_ = os.environ.get("PCOS_BENCH_ROOT", os.getcwd()); R = f"{B_}/results"; T = f"{R}/tables"
ENCS = ["mobilenetv2_100", "swin_tiny_patch4_window7_224", "resnet50"]
def parse(tag):
    m = re.fullmatch(r"q(\d+)l(\d+)s([\d.]+)", tag)
    if m: return dict(NQ=int(m[1]), NL=int(m[2]), LRS=float(m[3]))
    m = re.fullmatch(r"c(\d+)s([\d.]+)", tag)
    if m: return dict(NQ=int(m[1]), NL=3, LRS=float(m[2]))
    m = re.fullmatch(r"x(\d+)s([\d.]+)c([\d.]+)", tag)
    if m: return dict(NQ=int(m[1]), NL=3, LRS=float(m[2]), LRC=float(m[3]))
    m = re.fullmatch(r"h(\d+)s([\d.]+)p(\d+)", tag)
    if m: return dict(NQ=int(m[1]), NL=3, LRS=float(m[2]), HP=int(m[3]))
    return dict(NQ=6, NL=3, LRS=0.03)
rows = []
for e in ENCS:
    for f in sorted(glob.glob(f"{R}/quantum_tune/{e}/*/{e}__plain__*__r0.npz")):
        tag = f.split("/")[-2]; cond = f.split("__")[-2]; z = np.load(f); p = parse(tag)
        rows.append(dict(encoder=e, cond=cond, tag=tag, **{"NQ": p["NQ"], "NL": p["NL"], "LRS": p["LRS"], "LRC": p.get("LRC", 3e-4), "HP": p.get("HP", 4)}, val_auc=float(z["val_auc"].mean()), val_auc_min=float(z["val_auc"].min()), slot_params=int(z["slot_params"])))
grid = pd.DataFrame(rows); grid.to_csv(f"{T}/quantum_tuning_grid.csv", index=False)
sel, plan = [], {}
for (e, c), d in grid.groupby(["encoder", "cond"]):
    d = d.assign(k=(d.val_auc / 5e-4).round()).sort_values(["k", "slot_params", "NQ"], ascending=[False, True, True]); b = d.iloc[0]
    sel.append(dict(encoder=e, cond=c, tag=b.tag, NQ=b.NQ, NL=b.NL, LRS=b.LRS, LRC=b.LRC, HP=b.HP, val_auc=b.val_auc, trials=len(d), worst_trial_val_auc=d.val_auc.min(), median_trial_val_auc=d.val_auc.median(),
                    edge=("NQ at edge " if b.NQ in (4, 8) and c.endswith(("vqc_pairs", "dense_w", "dense_p", "identity")) else "") + ("lr at edge" if b.LRS in (0.01, 0.1) else "")))
    plan.setdefault(e, {})[c] = dict(NQ=int(b.NQ), NL=int(b.NL), LRS=float(b.LRS), LRC=float(b.LRC), HP=int(b.HP))
for e in plan:      # ring-entangler VQC is not tuned separately: it borrows the tuned VQC-pairs configuration of the same setting
    if "B_vqc_pairs" in plan[e]: plan[e]["B_vqc"] = dict(plan[e]["B_vqc_pairs"])
    if "A_vqc_pairs" in plan[e]: plan[e]["A_vqc_pairs_derg"] = dict(plan[e]["A_vqc_pairs"])
pd.DataFrame(sel).to_csv(f"{T}/quantum_tuning_selected.csv", index=False); json.dump(plan, open(f"{R}/quantum_final_plan.json", "w"), indent=1)
pd.set_option("display.width", 250); print(pd.DataFrame(sel).round(4).to_string())
