#!/usr/bin/env python3
"""Stage 0-1: manifest, group recovery, and both partitioning rules over 5 seeds.
Group (primary, g0) = same class AND identical dihedral-canonical 64-bit dHash
(this subsumes byte-identical files). Sensitivity group (g4) = same class AND
Hamming distance <= 4 (single linkage). Neither is verified to equal the patient:
the dataset ships no patient identifiers, so both are conservative PROXIES."""
import os, re, json
import numpy as np, pandas as pd

B = os.environ.get("PCOS_BENCH_ROOT", os.getcwd())
fp = pd.read_csv(f"{B}/data/fingerprints.csv")

def family(f):
    s = f[:-4]
    if "Rotate_Rotate_Right" in s: return "rot_right"
    if "Rotate" in s: return "rotate"
    if "resize" in s: return "resize"
    if re.search(r"(?i)seta", s): return "setA"
    if re.search(r"(?i)setb", s): return "setB"
    if s.startswith("image"): return "lc_image"
    return "plain"
fp["family"] = fp.file.map(family)
fp = fp.sort_values(["cls", "file"]).reset_index(drop=True)

# ---- primary group g0
fp["g0"] = pd.factorize(fp.cls + "|" + fp.dhash_canon)[0]

# ---- sensitivity group g4 (single linkage, Hamming <= 4, within class)
tab = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
comp = {}
for cls, g in fp.groupby("cls"):
    uh = sorted(g.dhash_canon.unique())
    H = np.array([int(s, 2) for s in uh], dtype=np.uint64)
    n = len(H); par = list(range(n))
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    d = tab[(H[:, None] ^ H[None, :]).view(np.uint8)].reshape(n, n, 8).sum(axis=2)
    for i, j in zip(*np.nonzero(np.triu(d <= 4, 1))):
        ri, rj = find(i), find(j)
        if ri != rj: par[rj] = ri
    for s, k in zip(uh, range(n)):
        comp[(cls, s)] = f"{cls}:{find(k)}"
fp["g4"] = pd.factorize(pd.Series([comp[(c, s)] for c, s in zip(fp.cls, fp.dhash_canon)]))[0]

# ---- splitting rules
NAMES = ["train", "val", "test"]; PROPS = np.array([0.64, 0.16, 0.20])
def split_groups(df, gcol, seed):
    rng = np.random.RandomState(seed)
    out = pd.Series("", index=df.index, dtype=object)
    for lab in (0, 1):
        sub = df[df.label == lab]
        sizes = sub.groupby(gcol).size()
        order = rng.permutation(sizes.index.values)
        target = PROPS * len(sub); cur = np.zeros(3); gmap = {}
        for g in order:
            k = int(np.argmax((target - cur) / target))
            gmap[g] = NAMES[k]; cur[k] += sizes[g]
        out[sub.index] = sub[gcol].map(gmap)
    return out
def split_images(df, seed):
    rng = np.random.RandomState(seed)
    out = pd.Series("", index=df.index, dtype=object)
    for lab in (0, 1):
        idx = df.index[df.label == lab].values.copy(); rng.shuffle(idx)
        n = len(idx); a = int(round(PROPS[0] * n)); b = a + int(round(PROPS[1] * n))
        out[idx[:a]] = "train"; out[idx[a:b]] = "val"; out[idx[b:]] = "test"
    return out

SEEDS = [0, 1, 2, 3, 4]
for s in SEEDS:
    fp[f"grp_s{s}"] = split_groups(fp, "g0", s)       # group-disjoint, primary groups
    fp[f"g4_s{s}"] = split_groups(fp, "g4", s)        # group-disjoint, conservative groups
    fp[f"img_s{s}"] = split_images(fp, s)             # naive image-level

def spanning(df, gcol, scol):
    return int((df.groupby(gcol)[scol].nunique() > 1).sum())
def md5_leak(df, scol):
    tr = set(df.md5[df[scol].isin(["train", "val"])]); te = df[df[scol] == "test"]
    return round(100 * float(te.md5.isin(tr).mean()), 2)
def g0_leak(df, scol):
    tr = set(df.g0[df[scol].isin(["train", "val"])]); te = df[df[scol] == "test"]
    return round(100 * float(te.g0.isin(tr).mean()), 2)

res = {"n_images": int(len(fp)), "n_unique_md5": int(fp.md5.nunique()),
       "n_groups_g0": int(fp.g0.nunique()), "n_groups_g4": int(fp.g4.nunique()),
       "largest_g0": int(fp.groupby("g0").size().max()), "largest_g4": int(fp.groupby("g4").size().max()),
       "singleton_g0_pct": round(100 * float((fp.groupby("g0").size() == 1).mean()), 1),
       "g0_cross_label_groups": int((fp.groupby("g0").label.nunique() > 1).sum()),
       "rules": {}}
for rule, pre in (("group_g0", "grp"), ("group_g4", "g4"), ("naive_image", "img")):
    rows = []
    for s in SEEDS:
        c = f"{pre}_s{s}"
        sz = fp[c].value_counts().to_dict()
        te = fp[fp[c] == "test"]
        rows.append({"seed": s, "train": int(sz["train"]), "val": int(sz["val"]), "test": int(sz["test"]),
                     "test_pos_frac": round(float(te.label.mean()), 4),
                     "S_g0": spanning(fp, "g0", c), "S_g4": spanning(fp, "g4", c),
                     "test_md5_seen_in_trainval_pct": md5_leak(fp, c),
                     "test_g0_seen_in_trainval_pct": g0_leak(fp, c)})
    res["rules"][rule] = rows
# integrity assertions (the study's spine)
assert all(r["S_g0"] == 0 for r in res["rules"]["group_g0"]), "group rule violated"
assert all(r["S_g4"] == 0 for r in res["rules"]["group_g4"]), "g4 group rule violated"
assert all(r["S_g0"] > 0 for r in res["rules"]["naive_image"]), "naive rule should span groups"

man = fp[["file", "cls", "label", "family", "width", "height", "bytes", "q0", "md5", "mean", "std", "g0", "g4"]].copy()
man.insert(0, "path", man.cls + "/" + man.file)
man.to_csv(f"{B}/data/manifest.csv", index=False)
fp[["path"] if "path" in fp else ["file"]].shape
spl = fp[["cls", "file", "label", "g0", "g4"] + [f"{p}_s{s}" for p in ("grp", "g4", "img") for s in SEEDS]]
spl.to_csv(f"{B}/data/splits.csv", index=False)
json.dump(res, open(f"{B}/results/stage1_splits.json", "w"), indent=1)

print(f"images {res['n_images']}  byte-unique {res['n_unique_md5']}  groups g0 {res['n_groups_g0']}  g4 {res['n_groups_g4']}")
print(f"largest group: g0 {res['largest_g0']}  g4 {res['largest_g4']}   singleton g0 groups {res['singleton_g0_pct']}%   cross-label g0 groups {res['g0_cross_label_groups']}")
for rule, rows in res["rules"].items():
    r = pd.DataFrame(rows)
    print(f"\n[{rule}]  train/val/test (seed0) {rows[0]['train']}/{rows[0]['val']}/{rows[0]['test']}")
    print(r[["seed", "test_pos_frac", "S_g0", "S_g4", "test_md5_seen_in_trainval_pct", "test_g0_seen_in_trainval_pct"]].to_string(index=False))
