#!/usr/bin/env python3
"""Stage 8 (extended): multi-method XAI for the three arms on ONE frozen encoder (MobileNetV2, plain, group split seed 0).
Arms (all differentiable heads on the pooled encoder feature, so every attribution reaches the pixels):
  classical : validation-tuned head from 14_tuning.py (refit on train+val, exported to torch and checked against sklearn)
  fl        : FedAvg global model, tuned config, label-skew Dirichlet(0.5) clients (non-DP)       [rebuilt with 15_fl.py, checked against stored predictions]
  fl_dp     : FedAvg with group-level DP-SGD, sigma=1.2 tuned config, same clients
  quantum   : tuned VQC-pairs hybrid layer (setting B), retrained seed 0 via 11_quantum.run_fit
Methods: Grad-CAM, Grad-CAM++, Eigen-CAM (encoder-only), Score-CAM (top-64 channels), occlusion (7x7 blocks), Integrated Gradients (12 steps, box-smoothed).
Faithfulness: deletion / insertion curves (T=21, trapezoid) vs a random pixel order and a random 7x7-block order, constant-map check, a common
class-balanced sample of test items (distinct groups) that every arm classifies correctly. Usage: 19_xai_arms.py check | run BUDGET"""
import os, sys, json, time, warnings, importlib.util
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn, timm
from PIL import Image
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
B = os.environ.get("PCOS_BENCH_ROOT", os.getcwd()); WT = os.environ.get("PCOS_WEIGHTS", "weights"); RAW = os.environ.get("PCOS_RAW", "data/raw/PCOS")
ENC = "mobilenetv2_100"; OUT = f"{B}/results/xai_arms"; os.makedirs(OUT, exist_ok=True); S = 224; T = 21; RD = 2; NPER = int(os.environ.get("NPER", "12")); STEPS = 12; TOPK = 64
torch.set_num_threads(int(os.environ.get("QT", "1")))
plan = json.load(open(f"{B}/results/quantum_final_plan.json"))[ENC]["B_vqc_pairs"]
os.environ.update(NQ=str(plan["NQ"]), NL=str(plan["NL"]), LRS=str(plan["LRS"]), LRC=str(plan["LRC"]), HP=str(plan["HP"]), SEEDS="0", QOUT=f"{OUT}/_q")
def load_mod(name, rel):
    sp = importlib.util.spec_from_file_location(name, f"{B}/{rel}"); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
Q = load_mod("qmod", "pipeline/11_quantum.py"); FL = load_mod("flmod", "pipeline/15_fl.py"); TN = load_mod("tnmod", "pipeline/14_tuning.py")
man = pd.read_csv(f"{B}/data/manifest.csv"); spl = pd.read_csv(f"{B}/data/splits.csv"); uidx = pd.read_csv(f"{B}/features/unique_index.csv")
pos = {m: i for i, m in enumerate(uidx.md5)}; r2u = man.md5.map(pos).values; y = man.label.values
import timm.models.helpers as H
H.load_state_dict_from_url = lambda url, **kw: torch.load(f"{WT}/{url.split('/')[-1]}", map_location="cpu", weights_only=True)
model = timm.create_model(ENC, pretrained=True, num_classes=0).eval(); cfg = model.default_cfg
for p in model.parameters(): p.requires_grad_(False)
Xu = np.load(f"{B}/features/{ENC}__plain.npz")["feat"].astype(np.float32)
part = spl["grp_s0"].values; tr, va, te = (np.nonzero(part == k)[0] for k in ("train", "val", "test")); trva = np.concatenate([tr, va])
tt = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)

# ---------------------------------------------------------------- arm heads (torch, on pooled features f)
class ClassicalHead(nn.Module):
    def __init__(self, fam, prep, hp):
        super().__init__(); sc = TN.StandardScaler().fit(Xu[r2u[trva]]); u, w, lab = TN.collapse(trva); pca = None
        if prep == "pca256": pca = TN.PCA(256, random_state=0).fit(sc.transform(Xu[u]))
        tf = lambda idx: (pca.transform(sc.transform(Xu[idx])) if pca is not None else sc.transform(Xu[idx]))
        m = TN.make(fam, hp, 0)
        try: m.fit(tf(u), lab, sample_weight=w)
        except TypeError: m.fit(tf(u), lab)
        self.m, self.tf, self.kind = m, tf, fam
        self.mu, self.sd = tt(sc.mean_), tt(sc.scale_); self.pm, self.pc = (tt(pca.mean_), tt(pca.components_)) if pca is not None else (None, None)
        if fam.startswith("lr"): self.W, self.b = tt(m.coef_[0]), float(m.intercept_[0])
        else: self.Ws, self.bs = [tt(c) for c in m.coefs_], [tt(b) for b in m.intercepts_]
    def forward(self, f):
        z = (f - self.mu) / self.sd
        if self.pc is not None: z = (z - self.pm) @ self.pc.T
        if self.kind.startswith("lr"): return z @ self.W + self.b
        for i, (W_, b_) in enumerate(zip(self.Ws, self.bs)):
            z = z @ W_ + b_
            if i < len(self.Ws) - 1: z = torch.relu(z)
        return z[:, 0]
class FLHead(nn.Module):
    def __init__(self, st, th):
        super().__init__(); P = st.pairs; w = P.w.values.astype(np.float64); U = P.u.values
        mean = (Xu[U] * w[:, None]).sum(0) / w.sum(); std = np.sqrt((((Xu[U] - mean) ** 2) * w[:, None]).sum(0) / w.sum()) + 1e-6
        self.mu, self.sd, self.th = tt(mean), tt(std), tt(th)
    def forward(self, f):
        z = (f - self.mu) / self.sd; z = z * FL.RN / (z.norm(dim=1, keepdim=True) + 1e-9); return z @ self.th[:-1] + self.th[-1]
class QuantumHead(nn.Module):
    def __init__(self, net, mu, sd): super().__init__(); self.net, self.mu, self.sd = net.eval(), tt(mu), tt(sd)
    def forward(self, f): return self.net((f - self.mu) / self.sd)
def build_heads():
    heads = {}; rep = {}
    sel = json.loads(str(np.load(f"{B}/results/tuning/{ENC}__plain.npz", allow_pickle=True)["selected"]))[0]; hc = ClassicalHead(sel["family"], sel["prep"], sel["hp"]); heads["classical"] = hc
    ref = TN.fit_predict(sel["family"], sel["prep"], sel["hp"], Xu, trva, [te], 0)[0]; Fte = tt(Xu[r2u[te]])
    with torch.no_grad(): pc = torch.sigmoid(hc(Fte)).numpy()
    rep["classical_cfg"] = sel; rep["classical_maxerr"] = float(np.abs(pc - ref).max())
    st = FL.Stage(Xu, 0, True); sf = json.load(open(f"{B}/results/fl/tuned_{ENC}.json")); cl = FL.assign("lab0.5", st, np.random.RandomState(500)); cr = FL.build_clients(st, cl, retain=True)
    a = sf["fedavg"]; th, _, _ = FL.run_fl(cr, "fedavg", a["par"]["lr"], a["par"]["E"], a["R"], 0); heads["fl"] = FLHead(st, th)
    d = sf["dp1.2"]; p_ = d["par"]; thd, _, _ = FL.run_fl(cr, "fedavg", p_["lr"], p_["E"], d["R"], 0, dp=(p_["C"], p_["sigma"], p_["q"])); heads["fl_dp"] = FLHead(st, thd)
    stored = np.load(f"{B}/results/fl/runs/{ENC}__s0.npz", allow_pickle=True)
    for k, th_, key in (("fl", th, "lab0.5|fedavg"), ("fl_dp", thd, "lab0.5|fedavg_dp1.2")):
        with torch.no_grad(): pk = torch.sigmoid(heads[k](Fte)).numpy()
        rep[f"{k}_maxerr_vs_stored"] = float(np.abs(FL.test_probs(st, th_) - stored[key]).max()); rep[f"{k}_head_vs_stage_maxerr"] = float(np.abs(pk - FL.test_probs(st, th_)).max())
    te_, pt, best, bep, nsl, ntot = Q.run_fit(Xu, 0, "vqc_pairs", False, None); L = Q.LAST; hq = QuantumHead(L["net"], L["mu"], L["sd"]); heads["quantum"] = hq
    with torch.no_grad(): pq = torch.sigmoid(hq(Fte)).numpy()
    rep["quantum_maxerr"] = float(np.abs(pq - pt).max()); rep["quantum_cfg"] = plan; rep["quantum_slot_params"] = int(nsl)
    return heads, rep

def load_x(path):
    im = Image.open(f"{RAW}/{path}").convert("RGB").resize((S, S), Image.BICUBIC); a = np.asarray(im).astype(np.float32) / 255
    return np.asarray(im), torch.from_numpy(((a - np.array(cfg["mean"])) / np.array(cfg["std"])).astype(np.float32)).permute(2, 0, 1)[None]
def up(c):
    t = torch.from_numpy(np.ascontiguousarray(c, dtype=np.float32))[None, None]; u = Fn.interpolate(t, size=(S, S), mode="bilinear", align_corners=False)[0, 0].numpy()
    lo, hi = u.min(), u.max(); return (u - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(u)
@torch.no_grad()
def pooled(xb, bs=64): return torch.cat([model(xb[i:i + bs]) for i in range(0, len(xb), bs)])
def tprob(h, f, cl):
    with torch.no_grad(): p = torch.sigmoid(h(f))
    return (p if cl == 1 else 1 - p).numpy()

def attributions(x, cl, heads):
    """returns maps[arm][method] (7x7 or SxS arrays, un-normalized) and live target-class probability per arm"""
    A = model.forward_features(x).detach().requires_grad_(True); f = A.mean((2, 3)); maps = {a: {} for a in heads}; plive = {}
    Ad = A.detach()[0]; Am = Ad.reshape(Ad.shape[0], -1).T.numpy(); Ac = Am - Am.mean(0); _, _, VT = np.linalg.svd(Ac, full_matrices=False); proj = Ac @ VT[0]
    if np.corrcoef(proj, Am.mean(1))[0, 1] < 0: proj = -proj
    eig = proj.reshape(7, 7)
    top = Ad.flatten(1).max(1).values.topk(TOPK).indices; mk = torch.stack([torch.from_numpy(up(Ad[k].numpy())) for k in top]); xs = x * mk[:, None]            # (K,3,S,S)
    fk = pooled(xs); f0 = pooled(torch.zeros_like(x)); occ = []
    for r in range(7):
        for c in range(7): o = x.clone(); o[:, :, r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = 0; occ.append(o[0])
    fo = pooled(torch.stack(occ))
    for arm, h in heads.items():
        z = h(f)[0]; score = z if cl == 1 else -z; g, = torch.autograd.grad(score, A, retain_graph=True); plive[arm] = float(torch.sigmoid(z) if cl == 1 else 1 - torch.sigmoid(z))
        maps[arm]["gradcam"] = torch.relu((g.mean((2, 3), keepdim=True) * A).sum(1))[0].detach().numpy()
        g2, g3 = g ** 2, g ** 3; den = 2 * g2 + A.sum((2, 3), keepdim=True) * g3; den = torch.where(den != 0, den, torch.ones_like(den)); wts = ((g2 / den) * torch.relu(g)).sum((2, 3), keepdim=True)
        maps[arm]["gradcampp"] = torch.relu((wts * A).sum(1))[0].detach().numpy(); maps[arm]["eigencam"] = eig
        cic = tprob(h, fk, cl) - float(tprob(h, f0, cl)[0]); wk = np.clip(cic, 0, None); wk = wk / wk.sum() if wk.sum() > 0 else np.full(len(cic), 1 / len(cic))
        maps[arm]["scorecam"] = np.clip((wk[:, None, None] * Ad[top].numpy()).sum(0), 0, None)
        maps[arm]["occlusion"] = np.clip(float(tprob(h, f.detach(), cl)[0]) - tprob(h, fo, cl), 0, None).reshape(7, 7)
        al = (torch.arange(STEPS, dtype=torch.float32) + 0.5) / STEPS; xb = (al[:, None, None, None] * x).requires_grad_(True); zz = h(model(xb)); sc_ = zz.sum() if cl == 1 else -zz.sum()
        gg, = torch.autograd.grad(sc_, xb); ig = (gg.mean(0) * x[0]).sum(0); ig = torch.relu(ig)[None, None]; maps[arm]["ig"] = Fn.avg_pool2d(ig, 9, 1, 4)[0, 0].numpy()
    return maps, plive
METHODS = ["gradcam", "gradcampp", "eigencam", "scorecam", "occlusion", "ig"]
def order_from(mp, rs):
    u = up(mp) if mp.shape[0] != S else (lambda a: (a - a.min()) / (a.max() - a.min()) if a.max() - a.min() > 1e-12 else np.zeros_like(a))(mp)
    const = bool(np.ptp(u) < 1e-9); return u, const, np.argsort(-(u.flatten() + (1e-6 * rs.rand(S * S) if const else 0)), kind="stable")
fr = np.linspace(0, 1, T); K = np.round(fr * S * S).astype(int)
def masked(xs, ordr, mode):
    flat = xs.reshape(3, -1); out = np.empty((T, 3, S * S), np.float32)
    for t, k in enumerate(K):
        base = np.zeros_like(flat) if mode == "ins" else flat.copy(); sp = ordr[:k]
        if mode == "ins": base[:, sp] = flat[:, sp]
        else: base[:, sp] = 0.0
        out[t] = base
    return torch.from_numpy(out.reshape(T, 3, S, S))
def auc_of(h, f, cl): return float(np.trapz(tprob(h, f, cl), fr))
def border_mass(u):
    ring = np.zeros((S, S), bool); m = S // 10; ring[:m] = ring[-m:] = True; ring[:, :m] = ring[:, -m:] = True; return float(u[ring].sum() / max(u.sum(), 1e-12))

def select_items(heads):
    Fte = tt(Xu[r2u[te]]); ok = np.ones(len(te), bool)
    with torch.no_grad():
        for h in heads.values(): z = h(Fte).numpy(); ok &= ((z > 0).astype(int) == y[te]) & (np.abs(z) > 0.5)
    cand = pd.DataFrame({"row": te[ok], "label": y[te][ok], "g0": man.g0.values[te[ok]], "path": man.path.values[te[ok]]}).drop_duplicates("g0"); rng = np.random.RandomState(0)
    return {k: cand[cand.label == k].sample(NPER, random_state=rng).row.tolist() for k in (1, 0)}, len(cand)

if __name__ == "__main__":
    heads, rep = build_heads(); print(json.dumps(rep, indent=1), flush=True); json.dump(rep, open(f"{OUT}/verification.json", "w"), indent=1)
    assert rep["classical_maxerr"] < 2e-3 and rep["quantum_maxerr"] < 2e-3 and rep["fl_maxerr_vs_stored"] < 2e-3 and rep["fl_dp_maxerr_vs_stored"] < 2e-3, "arm reconstruction mismatch"
    if sys.argv[1] == "check": sys.exit()
    BUDGET = float(sys.argv[2]); T0 = time.time(); ck = f"{OUT}/faith_items.json"; done = json.load(open(ck)) if os.path.exists(ck) and os.path.getsize(ck) > 2 else {}
    sel, ncand = select_items(heads); order = [(k, r) for i in range(NPER) for k in (1, 0) for r in [sel[k][i]]]; print("common-correct candidates (distinct groups):", ncand, flush=True)
    disp = {}
    for cl, r in order:
        need = (cl, r) in order[:4] and not os.path.exists(f"{OUT}/display_{r}.npz")
        if str(r) in done and not need: continue
        if time.time() - T0 > BUDGET - 55: break
        im, x = load_x(man.path[r]); maps, plive = attributions(x, cl, heads); xs = x[0].numpy()
        if need:
            np.savez(f"{OUT}/display_{r}.npz", img=im, label=cl, plive=json.dumps(plive), **{f"{a}__{m}": up(maps[a][m]).astype(np.float16) for a in heads for m in METHODS})
            if str(r) in done: continue
        rs = np.random.RandomState(1000 + int(r)); res = dict(label=cl, row=int(r), plive=plive, faith={}, agree={})
        for arm, h in heads.items():
            res["faith"][arm] = {}
            for mth in METHODS:
                if arm != "classical" and mth == "eigencam": res["faith"][arm][mth] = dict(res["faith"]["classical"][mth]) if False else None
                u, const, ordr = order_from(maps[arm][mth], rs)
                dl = auc_of(h, pooled(masked(xs, ordr, "del")), cl); ins = auc_of(h, pooled(masked(xs, ordr, "ins")), cl)
                res["faith"][arm][mth] = dict(**{"del": dl, "ins": ins}, const=const, border=border_mass(u))
        rnd = {a: dict(d=[], i=[], cd=[], ci=[]) for a in heads}
        for _ in range(RD):
            o1 = rs.permutation(S * S); o2 = np.argsort(-np.kron(rs.permutation(49).reshape(7, 7), np.ones((32, 32))).flatten(), kind="stable")
            fd1, fi1, fd2, fi2 = pooled(masked(xs, o1, "del")), pooled(masked(xs, o1, "ins")), pooled(masked(xs, o2, "del")), pooled(masked(xs, o2, "ins"))
            for a, h in heads.items(): rnd[a]["d"].append(auc_of(h, fd1, cl)); rnd[a]["i"].append(auc_of(h, fi1, cl)); rnd[a]["cd"].append(auc_of(h, fd2, cl)); rnd[a]["ci"].append(auc_of(h, fi2, cl))
        res["random"] = {a: {k: float(np.mean(v)) for k, v in d.items()} for a, d in rnd.items()}
        pool7 = lambda m: Fn.adaptive_avg_pool2d(torch.from_numpy(up(m))[None, None], 28)[0, 0].numpy().flatten()
        arms = list(heads); 
        for mth in METHODS:
            for i in range(len(arms)):
                for j in range(i + 1, len(arms)):
                    a, b = pool7(maps[arms[i]][mth]), pool7(maps[arms[j]][mth]); res["agree"][f"{mth}|{arms[i]}|{arms[j]}"] = float(spearmanr(a, b)[0]) if a.std() > 0 and b.std() > 0 else None
        done[str(r)] = res; json.dump(done, open(ck, "w"))
        print(f"item {len(done)} done  t={time.time()-T0:.0f}s", flush=True)
    print(f"xai items {len(done)}/{2*NPER}  elapsed {time.time()-T0:.0f}s", flush=True)
