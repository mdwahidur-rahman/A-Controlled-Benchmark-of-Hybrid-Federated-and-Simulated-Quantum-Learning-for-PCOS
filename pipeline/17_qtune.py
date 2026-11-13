#!/usr/bin/env python3
"""Quantum subsection: equal-budget hyperparameter tuning on VALIDATION AUC only (train split fits, validation split selects;
test predictions are produced by the unchanged runner but never read for selection). Seeds 0-2, plain features, one encoder at a time.
  VQC-pairs slot (settings B and A): n_qubits {4,6,8} x layers {2,3,4} x slot/readout lr {0.01,0.03,0.1}; 8 qubits limited to <=3 layers (24 configs)
  dense_w, dense_p, identity controls (settings B/A): bottleneck width = n_qubits {4,6,8} x lr {0.01,0.03,0.1} (9 configs each)
  A_classical: fusion baseline (no slot).
Usage: 17_qtune.py BUDGET_SECONDS ENCODER"""
import os, sys, time, subprocess
B = os.environ.get("PCOS_BENCH_ROOT", os.getcwd()); budget = float(sys.argv[1]); enc = sys.argv[2]; T0 = time.time()
cfgs = []
for nq in (4, 6, 8):
    for s in (0.01, 0.03, 0.1): cfgs.append((f"c{nq}s{s}", dict(NQ=nq, NL=3, LRS=s), ["B_identity", "B_dense_w", "B_dense_p", "A_dense_w", "A_dense_p"]))
cfgs.append(("classical", dict(NQ=6, NL=3, LRS=0.03), ["A_classical"]))
if os.environ.get("EXT", "0") == "1":     # control extension so that every slot has about the same number of tuning trials (dense_w / identity: + compressor lr; dense_p: + hidden width)
    for nq in (4, 6, 8):
        for s in (0.01, 0.03, 0.1):
            for lrc in (0.001, 0.003): cfgs.append((f"x{nq}s{s}c{lrc}", dict(NQ=nq, NL=3, LRS=s, LRC=lrc), ["B_identity", "B_dense_w", "A_dense_w"]))
            for hp in (2, 8): cfgs.append((f"h{nq}s{s}p{hp}", dict(NQ=nq, NL=3, LRS=s, HP=hp), ["B_dense_p", "A_dense_p"]))
for nq in (4, 6, 8):
    for nl in (2, 3, 4):
        if nq == 8 and nl == 4: continue          # compute cap: 8 qubits x 4 layers (~4 min per config) is excluded from the grid
        for s in (0.01, 0.03, 0.1): cfgs.append((f"q{nq}l{nl}s{s}", dict(NQ=nq, NL=nl, LRS=s), ["B_vqc_pairs", "A_vqc_pairs"]))
left = 0
for tag, env, conds in cfgs:
    out = f"{B}/results/quantum_tune/{enc}/{tag}"
    if all(os.path.exists(f"{out}/{enc}__plain__{c}__r0.npz") for c in conds): continue
    rem = budget - (time.time() - T0)
    if rem < (110 if env['NQ'] == 8 and any('vqc' in c for c in conds) else 25): left += 1; continue
    e = dict(os.environ, QOUT=out, SEEDS="0,1,2", OMP_NUM_THREADS="1", **{k: str(v) for k, v in env.items()})
    subprocess.run(["python3", f"{B}/pipeline/11_quantum.py", "run", str(rem), "--enc", enc, "--var", "plain", "--conds", ",".join(conds)], env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not all(os.path.exists(f"{out}/{enc}__plain__{c}__r0.npz") for c in conds): left += 1
done = sum(all(os.path.exists(f"{B}/results/quantum_tune/{enc}/{t}/{enc}__plain__{c}__r0.npz") for c in cs) for t, _, cs in cfgs)
print(f"{enc}: {done}/{len(cfgs)} config groups complete, {time.time()-T0:.0f}s", flush=True)
