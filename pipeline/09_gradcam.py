#!/usr/bin/env python3
"""Stage 8: Grad-CAM / Grad-CAM++ / Eigen-CAM on the frozen MobileNetV2 encoder + L2 logistic head
(group-disjoint split, seed 0, plain preprocessing). Faithfulness = deletion/insertion curves (T=25, trapezoid)
against a random-order baseline under the identical masking protocol; constant maps are counted and rescored
with randomized ties (np.argsort of a constant array is a scan order, not saliency). Score-CAM is not run
(~1,280 forward passes per image on 3 CPU cores).  Usage: 09_gradcam.py prep | faith BUDGET"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd, torch, timm
import torch.nn.functional as Fn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore"); torch.set_num_threads(3)
B = os.environ.get("PCOS_BENCH_ROOT", os.getcwd()); W = os.environ.get("PCOS_WEIGHTS", "weights"); RAW = os.environ.get("PCOS_RAW", "data/raw/PCOS")
ENC = "mobilenetv2_100"; OUT = f"{B}/results/gradcam"; S = 224; T = 25; RDRAWS = 4
man = pd.read_csv(f"{B}/data/manifest.csv"); spl = pd.read_csv(f"{B}/data/splits.csv"); uidx = pd.read_csv(f"{B}/features/unique_index.csv")
pos = {m: i for i, m in enumerate(uidx.md5)}; r2u = man.md5.map(pos).values; y = man.label.values

import timm.models.helpers as H
M = json.load(open(f"{B}/data/weights_manifest.json"))
H.load_state_dict_from_url = lambda url, **kw: torch.load(f"{W}/{url.split('/')[-1]}", map_location="cpu", weights_only=True)
model = timm.create_model(ENC, pretrained=True, num_classes=0).eval()
cfg = model.default_cfg; mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1); std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)

# ---- head (seed 0, group-disjoint, plain): refit on train+val exactly as in Stage 4
Xu = np.load(f"{B}/features/{ENC}__plain.npz")["feat"].astype(np.float32); X = Xu[r2u]
part = spl["grp_s0"].values; tr, va, te = (np.nonzero(part == k)[0] for k in ("train", "val", "test")); trva = np.concatenate([tr, va])
C = float(np.load(f"{B}/results/heads/{ENC}__plain__group_g0.npz")["C"][0])
sc = StandardScaler().fit(X[trva]); lr = LogisticRegression(C=C, max_iter=500).fit(sc.transform(X[trva]), y[trva])
mu = torch.tensor(sc.mean_, dtype=torch.float32); sd = torch.tensor(sc.scale_, dtype=torch.float32)
w = torch.tensor(lr.coef_[0], dtype=torch.float32); b0 = float(lr.intercept_[0])

def load_x(path):
    im = Image.open(f"{RAW}/{path}").convert("RGB").resize((S, S), Image.BICUBIC)
    a = np.asarray(im).astype(np.float32) / 255
    return im, torch.from_numpy(((a - np.array(cfg["mean"])) / np.array(cfg["std"])).astype(np.float32)).permute(2, 0, 1)[None]
def logit_from_pooled(f): return ((f - mu) / sd) @ w + b0
@torch.no_grad()
def p_pos(x):
    return torch.sigmoid(logit_from_pooled(model(x))).numpy()

def cams(x, cls):
    """returns dict of 7x7 maps for target class cls (1=positive, 0=negative)."""
    x = x.clone().requires_grad_(False)
    A = model.forward_features(x).detach().requires_grad_(True)             # (1,1280,7,7)
    f = A.mean((2, 3)); z = logit_from_pooled(f)[0]; score = z if cls == 1 else -z
    g, = torch.autograd.grad(score, A)
    out = {}
    out["gradcam"] = torch.relu((g.mean((2, 3), keepdim=True) * A).sum(1))[0].detach().numpy()
    g2, g3 = g ** 2, g ** 3; den = 2 * g2 + A.sum((2, 3), keepdim=True) * g3; den = torch.where(den != 0, den, torch.ones_like(den))
    a_ = g2 / den; wts = (a_ * torch.relu(g)).sum((2, 3), keepdim=True)
    out["gradcampp"] = torch.relu((wts * A).sum(1))[0].detach().numpy()
    Am = A.detach()[0].reshape(A.shape[1], -1).T.numpy(); Ac = Am - Am.mean(0)
    _, _, VT = np.linalg.svd(Ac, full_matrices=False); proj = Ac @ VT[0]
    if np.corrcoef(proj, Am.mean(1))[0, 1] < 0: proj = -proj
    out["eigencam"] = proj.reshape(A.shape[2], A.shape[3])
    return out, float(torch.sigmoid(z))
def upsample(c):
    t = torch.from_numpy(np.ascontiguousarray(c, dtype=np.float32))[None, None]
    u = Fn.interpolate(t, size=(S, S), mode="bilinear", align_corners=False)[0, 0].numpy()
    lo, hi = u.min(), u.max(); return (u - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(u)

# ---- common class-balanced sample of CORRECTLY classified test images, distinct groups, seeded
pred = lr.predict_proba(sc.transform(X[te]))[:, 1]; ok = (pred >= 0.5).astype(int) == y[te]
cand = pd.DataFrame({"row": te[ok], "label": y[te][ok], "g0": man.g0.values[te[ok]], "path": man.path.values[te[ok]]}).drop_duplicates("g0")
rng = np.random.RandomState(0); sel = {k: cand[cand.label == k].sample(30, random_state=rng).row.tolist() for k in (1, 0)}

if sys.argv[1] == "prep":
    rows = []; imgs = []; maps = {k: [] for k in ("gradcam", "gradcampp", "eigencam")}; probs = []
    disp = sel[1][:2] + sel[0][:2]
    for r in disp:
        im, x = load_x(man.path[r]); cl = int(y[r]); cm, pr = cams(x, cl)
        chk = float(logit_from_pooled(model(x))[0]) - float(lr.decision_function(sc.transform(X[[r]]))[0])
        assert abs(chk) < 0.35, f"torch/sklearn logit mismatch {chk}"
        imgs.append(np.asarray(im)); probs.append(pr if cl == 1 else 1 - pr); rows.append(r)
        for k in maps: maps[k].append(upsample(cm[k]))
    np.savez(f"{OUT}/display.npz", imgs=np.stack(imgs), probs=np.array(probs), labels=y[rows], rows=np.array(rows),
             paths=man.path.values[rows], **{k: np.stack(v) for k, v in maps.items()})
    print("display examples:", [(man.path[r], int(y[r]), round(p, 4)) for r, p in zip(rows, probs)])
    print("logit check ok; head C =", C)
    sys.exit()

# ---- faithfulness
BUDGET = float(sys.argv[2]); T0 = time.time(); ck = f"{OUT}/faith_items.json"
done = json.load(open(ck)) if os.path.exists(ck) else {}
order = [(k, r) for k in (1, 0) for r in sel[k]]
fr = np.linspace(0, 1, T); K = np.round(fr * S * S).astype(int)
def curves(x, ordr, cl, mode):
    flat = x.reshape(3, -1); out = np.empty((T, 3, S * S), np.float32)
    for t, k in enumerate(K):
        base = np.zeros_like(flat) if mode == "ins" else flat.copy()
        sel_px = ordr[:k]
        if mode == "ins": base[:, sel_px] = flat[:, sel_px]
        else: base[:, sel_px] = 0.0
        out[t] = base
    pr = p_pos(torch.from_numpy(out.reshape(T, 3, S, S))); pr = pr if cl == 1 else 1 - pr
    return float(np.trapz(pr, fr)), pr
def cell_baseline(xs, cl, r):
    rs = np.random.RandomState(5000 + int(r)); rd, ri = [], []
    for _ in range(RDRAWS):
        cellrank = rs.permutation(49).reshape(7, 7)
        u = np.kron(cellrank, np.ones((32, 32)))                      # each 7x7 cell = 32x32 px block
        ordr = np.argsort(-u.flatten(), kind="stable")
        rd.append(curves(xs, ordr, cl, "del")[0]); ri.append(curves(xs, ordr, cl, "ins")[0])
    return {"del": float(np.mean(rd)), "ins": float(np.mean(ri)), "del_sd": float(np.std(rd)), "ins_sd": float(np.std(ri))}
for cl, r in order:
    if str(r) in done and "random_cell" in done[str(r)]: continue
    if time.time() - T0 > BUDGET - 12: break
    if str(r) in done:
        im, x = load_x(man.path[r]); done[str(r)]["random_cell"] = cell_baseline(x[0].numpy(), cl, r); json.dump(done, open(ck, "w")); continue
    im, x = load_x(man.path[r]); cm, pr0 = cams(x, cl); xs = x[0].numpy(); res = {"label": cl, "row": int(r), "p": pr0 if cl == 1 else 1 - pr0}
    rs = np.random.RandomState(1000 + int(r))
    for k in ("gradcam", "gradcampp", "eigencam"):
        u = upsample(cm[k]); const = bool(np.ptp(u) < 1e-9)
        ordr = np.argsort(-(u.flatten() + (1e-6 * rs.rand(S * S) if const else 0)), kind="stable")
        d, _ = curves(xs, ordr, cl, "del"); i_, _ = curves(xs, ordr, cl, "ins")
        ring = np.zeros((S, S), bool); m = S // 10; ring[:m] = ring[-m:] = True; ring[:, :m] = ring[:, -m:] = True
        res[k] = {"del": d, "ins": i_, "const": const, "border_mass": float(u[ring].sum() / max(u.sum(), 1e-12))}
    rd, ri = [], []
    for _ in range(RDRAWS):
        ordr = rs.permutation(S * S); rd.append(curves(xs, ordr, cl, "del")[0]); ri.append(curves(xs, ordr, cl, "ins")[0])
    res["random_cell"] = cell_baseline(xs, cl, r)
    res["random"] = {"del": float(np.mean(rd)), "ins": float(np.mean(ri)), "del_sd": float(np.std(rd)), "ins_sd": float(np.std(ri))}
    done[str(r)] = res; json.dump(done, open(ck, "w"))
print(f"faithfulness items {len(done)}/60  elapsed {time.time()-T0:.0f}s")
