#!/usr/bin/env python3
"""Tuned final quantum runs: every condition uses its own validation-selected hyperparameters (results/quantum_final_plan.json),
5 seeds, group_g0 splits, outputs in results/quantum_final/. Usage: 18_qfinal.py BUDGET [--vars plain,neutral] [--conds c1,c2]"""
import os, sys, json, time, subprocess
B = os.environ.get("PCOS_BENCH_ROOT", os.getcwd()); budget = float(sys.argv[1]); T0 = time.time()
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1].split(",") if k in sys.argv else d
plan = json.load(open(f"{B}/results/quantum_final_plan.json")); OUT = f"{B}/results/quantum_final"; os.makedirs(OUT, exist_ok=True)
vars_ = arg("--vars", ["plain", "neutral"]); order = arg("--conds", ["A_classical", "B_identity", "A_dense_w", "B_dense_w", "A_dense_p", "B_dense_p", "A_vqc_pairs", "B_vqc_pairs", "B_vqc", "A_vqc_pairs_derg"])
left = 0
for var in vars_:
    for enc in plan:
        for c in order:
            cfg = plan[enc].get(c) or plan[enc].get("A_classical")
            if cfg is None: continue
            reps = 3 if c.endswith("derg") else 1
            if all(os.path.exists(f"{OUT}/{enc}__{var}__{c}__r{r}.npz") for r in range(reps)): continue
            rem = budget - (time.time() - T0)
            if rem < (100 if (cfg["NQ"] == 8 and "vqc" in c) else 30): left += 1; continue
            e = dict(os.environ, QOUT=OUT, SEEDS="0,1,2,3,4", OMP_NUM_THREADS="1", **{k: str(v) for k, v in cfg.items()})
            subprocess.run(["python3", f"{B}/pipeline/11_quantum.py", "run", str(rem), "--enc", enc, "--var", var, "--conds", c], env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not all(os.path.exists(f"{OUT}/{enc}__{var}__{c}__r{r}.npz") for r in range(reps)): left += 1
print(f"remaining groups: {left}  elapsed {time.time()-T0:.0f}s", flush=True)
