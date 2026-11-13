#!/usr/bin/env python3
"""Stage 0 audit: filename families, size/format shortcuts, exact + near duplicates."""
import os, re, json, collections
import numpy as np, pandas as pd

B = os.environ.get("PCOS_BENCH_ROOT", os.getcwd())
df = pd.read_csv(f"{B}/data/fingerprints.csv")

def family(f):
    s = f[:-4]
    if "Rotate_Rotate_Right" in s: return "rot_right"
    if "Rotate" in s: return "rotate"
    if "resize" in s: return "resize"
    if re.search(r"(?i)seta", s): return "setA"
    if re.search(r"(?i)setb", s): return "setB"
    if s.startswith("image"): return "lc_image"
    return "plain"
def index(f):
    m = re.findall(r"\d+", f[:-4]); return int(m[0]) if m else -1
df["family"] = df.file.map(family); df["idx"] = df.file.map(index)
df["aspect"] = df.width / df.height; df["bpp"] = df.bytes / (df.width * df.height)
out = {}

out["n"] = int(len(df)); out["class_counts"] = df.cls.value_counts().to_dict()
out["family_by_class"] = {c: g.family.value_counts().to_dict() for c, g in df.groupby("cls")}

# --- size shortcut
df["wh"] = df.width.astype(str) + "x" + df.height.astype(str)
cls_per_wh = df.groupby("wh").label.nunique()
excl = df.wh.map(cls_per_wh) == 1
out["distinct_wh"] = int(df.wh.nunique())
out["wh_class_exclusive_images_pct"] = round(float(excl.mean() * 100), 2)
out["top_wh_by_class"] = {c: g.wh.value_counts().head(6).to_dict() for c, g in df.groupby("cls")}

def auc(x, y):
    r = pd.Series(np.asarray(x, dtype=float)).rank().values
    n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
y = df.label.values
out["single_feature_auc_infected_as_positive"] = {
    k: round(float(auc(df[k], y)), 4) for k in
    ["width", "height", "aspect", "bytes", "bpp", "q0", "mean", "std"]}

# --- exact duplicates
g = df.groupby("md5")
dup = g.size()[g.size() > 1]
out["exact_dup_groups"] = int(len(dup)); out["exact_dup_images"] = int(dup.sum())
out["exact_dup_groups_cross_class"] = int((g.label.nunique() > 1).sum())

# --- near duplicates (canonical dHash, Hamming)
H = np.array([int(s, 2) for s in df.dhash_canon], dtype=np.uint64)
N = len(H); tab = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
par = {t: list(range(N)) for t in (0, 4, 8)}
def find(p, a):
    while p[a] != a:
        p[a] = p[p[a]]; a = p[a]
    return a
def union(p, a, b):
    ra, rb = find(p, a), find(p, b)
    if ra != rb: p[rb] = ra
npairs = {0: 0, 4: 0, 8: 0}
BL = 250
for s in range(0, N, BL):
    x = (H[s:s + BL, None] ^ H[None, :]).view(np.uint8)
    d = tab[x].reshape(-1, N, 8).sum(axis=2)
    ii, jj = np.nonzero(d <= 8)
    for i, j in zip(ii, jj):
        i = i + s
        if j <= i: continue
        dd = d[i - s, j]
        union(par[8], i, j); npairs[8] += 1
        if dd <= 4: union(par[4], i, j); npairs[4] += 1
        if dd == 0: union(par[0], i, j); npairs[0] += 1
for t in (0, 4, 8):
    df[f"cl{t}"] = [find(par[t], i) for i in range(N)]
    sz = df.groupby(f"cl{t}").size()
    multi = sz[sz > 1]
    cross = (df.groupby(f"cl{t}").label.nunique() > 1).sum()
    famspan = (df.groupby(f"cl{t}").family.nunique() > 1).sum()
    out[f"dHash_T{t}"] = {"clusters": int(len(sz)), "multi_image_clusters": int(len(multi)),
        "images_in_multi": int(multi.sum()), "largest": int(sz.max()),
        "cross_class_clusters": int(cross), "cross_family_clusters": int(famspan),
        "pairs": int(npairs[t])}

# --- index-pair test inside each class: plain vs setA vs setB with same idx
pair = {}
for c, gcls in df.groupby("cls"):
    fam = {f: gcls[gcls.family == f].set_index("idx") for f in ("plain", "setA", "setB")}
    for a, b in (("plain", "setA"), ("plain", "setB"), ("setA", "setB")):
        common = fam[a].index.intersection(fam[b].index)
        if len(common) == 0: continue
        same8 = (fam[a].loc[common, "cl8"].values == fam[b].loc[common, "cl8"].values).mean()
        same_md5 = (fam[a].loc[common, "md5"].values == fam[b].loc[common, "md5"].values).mean()
        pair[f"{c}:{a}~{b}"] = {"shared_idx": int(len(common)),
            "same_cluster_T8_pct": round(float(same8 * 100), 1), "same_md5_pct": round(float(same_md5 * 100), 1)}
out["index_pair_test"] = pair

# --- per-family shortcut view
out["family_size_summary"] = {f"{c}|{f}": {"n": int(len(g2)), "median_w": float(g2.width.median()),
    "median_h": float(g2.height.median()), "median_bytes": float(g2.bytes.median())}
    for (c, f), g2 in df.groupby(["cls", "family"])}

df.drop(columns=["dhash_canon", "ahash_canon"]).to_csv(f"{B}/data/manifest_draft.csv", index=False)
json.dump(out, open(f"{B}/results/stage0_audit.json", "w"), indent=1)
print(json.dumps(out, indent=1))
