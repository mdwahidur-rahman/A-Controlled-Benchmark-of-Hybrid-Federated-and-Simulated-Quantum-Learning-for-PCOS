#!/usr/bin/env python3
"""Federated subsection (Stage 7, extended): non-IID cross-silo simulation on frozen features with group-level
DP-SGD. Clients receive whole g0 GROUPS. Heterogeneity: label skew (Dirichlet alpha), quantity skew, feature skew
(groups ordered by the first principal component of the encoder features), and a combined setting. Strategies:
FedAvg, FedProx (analytic proximal gradient), SCAFFOLD. References: central on all data, central on exactly the
retained clients' groups, and no-collaboration (each retained client alone). DP: per-group clipping at norm C,
Gaussian noise sigma*C, Poisson group sampling rate q; RDP accountant (sampled Gaussian, integer orders) with the
improved RDP->(eps, delta) conversion. Tuning uses the validation partition only; final runs refit on train+val.
Usage: 15_fl.py selftest | tune ENC BUDGET | select ENC | final ENC SEEDS BUDGET   (SEEDS e.g. 0,1,2,3,4)"""
import os, sys, json, time, math, warnings
import numpy as np, pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
B = os.environ.get("PCOS_BENCH_ROOT", os.environ.get("PCOS_BENCH_ROOT", os.getcwd())); F = f"{B}/features"; OUT = f"{B}/results/fl"
os.makedirs(f"{OUT}/tune", exist_ok=True); os.makedirs(f"{OUT}/runs", exist_ok=True)
man = pd.read_csv(f"{B}/data/manifest.csv"); spl = pd.read_csv(f"{B}/data/splits.csv")
uidx = pd.read_csv(f"{F}/unique_index.csv"); pos = {m: i for i, m in enumerate(uidx.md5)}
r2u = man.md5.map(pos).values; y_row = man.label.values; g_row = man.g0.values
K = 10; RN = 10.0; LAM = 1e-3; DELTA = 1e-5; ROUNDS = 100
PARTS = {"iid": ("lab", 100.0, False), "lab10": ("lab", 10.0, False), "lab0.5": ("lab", 0.5, False), "lab0.1": ("lab", 0.1, False),
         "qty0.3": ("qty", 0.3, False), "feat": ("feat", 0.0, False), "comb0.5": ("lab", 0.5, True)}
DP_PARTS = ["iid", "lab0.5", "lab0.1", "comb0.5"]; SIGMAS = [0.8, 1.2, 2.0, 4.0]; Q = 0.1
TUNE_PARTS = ["lab0.5", "comb0.5"]; MIN_G, MIN_CLASS = 8, 1
sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

# ------------------------------------------------------------------ privacy accounting (sampled Gaussian, RDP)
ORDERS = list(range(2, 64)) + [128, 256, 512]
def _log_a(q, s, a):
    return logsumexp([math.lgamma(a + 1) - math.lgamma(k + 1) - math.lgamma(a - k + 1) + (a - k) * math.log1p(-q) + k * math.log(q)
                      + (k * k - k) / (2 * s * s) for k in range(a + 1)])
def epsilon(q, sigma, steps, delta=DELTA):
    rdp = np.array([steps * _log_a(q, sigma, a) / (a - 1) for a in ORDERS]); o = np.array(ORDERS, float)
    return float(np.min(rdp - (np.log(delta) + np.log(o)) / (o - 1) + np.log((o - 1) / o)))

# ------------------------------------------------------------------ data staging
def pairs_for(rows):
    df = pd.DataFrame({"g": g_row[rows], "u": r2u[rows], "y": y_row[rows]})
    return df.groupby(["g", "u", "y"], as_index=False).size().rename(columns={"size": "w"}).sort_values(["g", "u"]).reset_index(drop=True)

class Stage:
    def __init__(self, Xu, seed, final):
        part = spl[f"grp_s{seed}"].values; tr, va, te = (np.nonzero(part == k)[0] for k in ("train", "val", "test"))
        self.pairs = P = pairs_for(np.concatenate([tr, va]) if final else tr); w = P.w.values.astype(np.float64); U = P.u.values
        mean = (Xu[U] * w[:, None]).sum(0) / w.sum(); std = np.sqrt((((Xu[U] - mean) ** 2) * w[:, None]).sum(0) / w.sum()) + 1e-6
        Z = (Xu - mean) / std; Z *= RN / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        self.Zu = np.hstack([Z, np.ones((len(Z), 1), np.float32)]).astype(np.float32); self.d1 = self.Zu.shape[1]
        zc = self.Zu[U, :-1] - (self.Zu[U, :-1] * w[:, None]).sum(0) / w.sum(); v = np.random.RandomState(0).randn(zc.shape[1])
        for _ in range(30): v = zc.T @ (w * (zc @ v)); v /= np.linalg.norm(v)
        sc = zc @ v
        self.garr = np.unique(P.g.values); self.gi = np.searchsorted(self.garr, P.g.values)
        self.gy = P.groupby("g").y.first().reindex(self.garr).values
        self.gscore = pd.Series(sc * w).groupby(P.g.values).sum().reindex(self.garr).values / pd.Series(w).groupby(P.g.values).sum().reindex(self.garr).values
        if not final:
            vp = pairs_for(va); self.val = (self.Zu[vp.u.values], vp.y.values, vp.w.values.astype(np.float64))
        self.te = te; self.te_u = r2u[te]; self.te_y = y_row[te]

def assign(spec, st, rng):
    kind, par, ordered = PARTS[spec]; n = len(st.garr); cl = np.zeros(n, int)
    if kind == "qty":
        p = rng.dirichlet([par] * K); return rng.choice(K, size=n, p=p)
    for c in (0, 1):
        idx = np.nonzero(st.gy == c)[0]
        if kind == "feat":
            pool = idx[np.argsort(st.gscore[idx])]; cl[pool] = np.minimum((np.arange(len(pool)) * K) // len(pool), K - 1)
            sw = rng.rand(len(pool)) < 0.15; cl[pool[sw]] = rng.randint(0, K, sw.sum()); continue
        pool = idx[np.argsort(st.gscore[idx])] if ordered else rng.permutation(idx)
        cuts = (np.cumsum(rng.dirichlet([par] * K)) * len(pool)).astype(int); prev = 0
        for k, cut in enumerate(cuts): cl[pool[prev:cut]] = k; prev = cut
        cl[pool[prev:]] = K - 1
    return cl

class Client:
    def __init__(self, Xa, y, w, gi):
        self.Xa, self.y, self.w = Xa, y.astype(np.float32), w.astype(np.float32)
        _, self.first, self.gidx = np.unique(gi, return_index=True, return_inverse=True); self.ng = len(self.first)
        self.gsum = np.add.reduceat(self.w, self.first).astype(np.float32); self.a = (self.w / self.gsum[self.gidx]) / self.ng
        self.coef = self.w / self.gsum[self.gidx]
    def full_grad(self, th): return self.Xa.T @ (self.a * (sig(self.Xa @ th) - self.y))
    def dp_grad(self, th, rng, C, s, q):
        m = rng.random_sample(self.ng) < q; d1 = len(th); noise = rng.normal(0, s * C, d1)
        if not m.any(): return noise / (q * self.ng)
        fm = m[self.gidx]; Xs = self.Xa[fm]; gs = self.gidx[fm]
        r = self.coef[fm] * (sig(Xs @ th) - self.y[fm]); st = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
        G = np.add.reduceat(r[:, None] * Xs, st, axis=0); nr = np.linalg.norm(G, axis=1); G *= np.minimum(1.0, C / np.maximum(nr, 1e-12))[:, None]
        return (G.sum(0) + noise) / (q * self.ng)

def build_clients(st, cl, retain=True):
    out = {}
    for k in range(K):
        sel = np.nonzero(cl[st.gi] == k)[0]; gk = np.nonzero(cl == k)[0]
        if retain and (len(gk) < MIN_G or min((st.gy[gk] == 0).sum(), (st.gy[gk] == 1).sum()) < MIN_CLASS): continue
        out[k] = Client(st.Zu[st.pairs.u.values[sel]], st.pairs.y.values[sel], st.pairs.w.values[sel], st.gi[sel])
    return out

REG = None
def local_update(c, th0, steps, lr, rng, dp=None, mu=0.0, ci=None, cg=None):
    th = th0.copy()
    for _ in range(steps):
        g = c.dp_grad(th, rng, *dp) if dp is not None else c.full_grad(th)
        g = g + LAM * REG * th
        if mu > 0: g = g + mu * (th - th0)
        if ci is not None: g = g - ci + cg
        th = th - lr * g
    return th

def run_fl(clients, strat, lr, E, rounds, seed, mu=0.0, dp=None, st=None, record_val=False):
    """strat in fedavg|fedprox|scaffold. dp=(C, sigma, q): E is then the number of local EPOCHS, steps = round(E/q)."""
    global REG
    d1 = next(iter(clients.values())).Xa.shape[1]; REG = np.r_[np.ones(d1 - 1), 0.0].astype(np.float32)
    rng = np.random.RandomState(seed); th = np.zeros(d1, np.float32); steps = int(round(E / dp[2])) if dp else int(E)
    tot = np.array([c.ng for c in clients.values()], np.float64); wts = tot / tot.sum()
    c_loc = {k: np.zeros(d1, np.float32) for k in clients}; c_glob = np.zeros(d1, np.float32); hist = []; ll = []
    for r in range(rounds):
        news, new_c = [], {}
        for k, c in clients.items():
            if strat == "scaffold":
                t = local_update(c, th, steps, lr, rng, dp, ci=c_loc[k], cg=c_glob); new_c[k] = c_loc[k] - c_glob + (th - t) / (steps * lr)
            elif strat == "fedprox": t = local_update(c, th, steps, lr, rng, dp, mu=mu)
            else: t = local_update(c, th, steps, lr, rng, dp)
            news.append(t)
        th = np.nan_to_num(sum(w_ * t for w_, t in zip(wts, news)).astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
        if strat == "scaffold": c_loc = new_c; c_glob = np.mean(list(new_c.values()), 0)
        if record_val:
            Xv, yv, wv = st.val; z = np.nan_to_num(Xv @ th, nan=0.0, posinf=30.0, neginf=-30.0); hist.append(roc_auc_score(yv, z, sample_weight=wv) if np.ptp(z) > 0 else 0.5)
            p = np.clip(sig(z), 1e-6, 1 - 1e-6); ll.append(float(-(wv * (yv * np.log(p) + (1 - yv) * np.log(1 - p))).sum() / wv.sum()))
    return th, hist, ll

def test_probs(st, th): return sig(st.Zu[st.te_u] @ th)
def cmetrics(y, p):
    yh = (p >= 0.5).astype(int); return dict(acc=float((yh == y).mean()), auc=float(roc_auc_score(y, p)))

# ------------------------------------------------------------------ self test
def selftest():
    ok = True
    e1 = epsilon(0.1, 1.2, 600); e2 = epsilon(0.01, 1.1, 10000)
    print("epsilon (q=.1,s=1.2,T=600) =", round(e1, 4), "ref 14.8615 | (q=.01,s=1.1,T=1e4) =", round(e2, 4), "ref 5.6543")
    ok &= abs(e1 - 14.8615) < 0.15 and abs(e2 - 5.6543) < 0.06
    Xu = np.load(f"{F}/mobilenetv2_100__plain.npz")["feat"].astype(np.float32); st = Stage(Xu, 0, False); cl = assign("lab0.5", st, np.random.RandomState(0))
    cls = build_clients(st, cl); d1 = st.d1
    # (1) FedProx proximal gradient: zero at the first local step, nonzero afterwards, and it changes the trajectory
    global REG; REG = np.r_[np.ones(d1 - 1), 0.0].astype(np.float32); c = next(iter(cls.values())); th0 = np.zeros(d1, np.float32)
    rng = np.random.RandomState(0); t1 = local_update(c, th0, 1, 1.0, rng, mu=1.0); t3 = local_update(c, th0, 3, 1.0, rng, mu=1.0); t3n = local_update(c, th0, 3, 1.0, rng, mu=0.0)
    prox1 = np.linalg.norm(1.0 * (t1 - th0)); prox3 = np.linalg.norm(1.0 * (t3 - th0)); print("prox term norm after 1 step", prox1, "after 3", prox3, "| trajectory differs from FedAvg:", np.linalg.norm(t3 - t3n))
    ok &= np.linalg.norm(t3 - t3n) > 1e-6
    # (2) central GD vs sklearn LR
    from sklearn.linear_model import LogisticRegression
    cen = build_clients(st, np.zeros(len(st.garr), int) , retain=False); th, h, _ = run_fl({0: cen[0]}, "fedavg", 1.0, 5, 60, 0, st=st, record_val=True)
    p = test_probs(st, th); m = cmetrics(st.te_y, p); print("central GD  acc/auc", m, "val auc last", h[-1])
    lr_ = LogisticRegression(C=100, max_iter=500).fit(st.Zu[st.pairs.u.values][:, :-1], st.pairs.y.values, sample_weight=st.pairs.w.values)
    ps = lr_.predict_proba(st.Zu[st.te_u][:, :-1])[:, 1]; print("sklearn LR  acc/auc", cmetrics(st.te_y, ps))
    # (3) a DP run executes and eps is reported
    th, h, _ = run_fl(cls, "fedavg", 1.0, 1, 5, 0, dp=(2.0, 1.2, Q), st=st, record_val=True); print("DP 5 rounds val auc", h[-1], "eps(5 rounds)", epsilon(Q, 1.2, 5 * 10))
    print("clients retained", len(cls), "of", K, "groups/client", [c.ng for c in cls.values()]); print("SELFTEST", "PASS" if ok else "FAIL")

# ------------------------------------------------------------------ tuning
def grids():
    g = []
    for lr in (0.3, 1.0, 3.0, 10.0, 30.0):
        for E in (1, 5):
            g.append(("fedavg", dict(lr=lr, E=E))); g.append(("scaffold", dict(lr=lr, E=E)))
            for mu in (0.01, 0.1, 0.5, 2.0): g.append(("fedprox", dict(lr=lr, E=E, mu=mu)))
    for s in SIGMAS:
        for C in (0.25, 0.5, 1.0, 2.0, 5.0):
            for lr in (0.1, 0.3, 1.0): g.append(("dpfedavg", dict(sigma=s, C=C, lr=lr, q=Q, E=1)))
    return g
def tune(enc, budget):
    t0 = time.time(); Xu = np.load(f"{F}/{enc}__plain.npz")["feat"].astype(np.float32); fn = f"{OUT}/tune/{enc}.jsonl"
    done = {json.loads(l)["cid"] for l in open(fn)} if os.path.exists(fn) else set(); sts = {s: Stage(Xu, s, False) for s in (0, 1)}
    cls = {(sp, s): build_clients(sts[s], assign(sp, sts[s], np.random.RandomState(100 + s))) for sp in TUNE_PARTS for s in (0, 1)}
    for strat, par in grids():
        cid = strat + "|" + json.dumps(par, sort_keys=True)
        if cid in done: continue
        if time.time() - t0 > budget - 15: break
        H, L = [], []
        for (sp, s), c in cls.items():
            if strat == "dpfedavg": _, h, l = run_fl(c, "fedavg", par["lr"], par["E"], ROUNDS, s, dp=(par["C"], par["sigma"], par["q"]), st=sts[s], record_val=True)
            else: _, h, l = run_fl(c, strat, par["lr"], par["E"], ROUNDS, s, mu=par.get("mu", 0.0), st=sts[s], record_val=True)
            H.append(h); L.append(l)
        with open(fn, "a") as f: f.write(json.dumps(dict(cid=cid, strat=strat, par=par, auc=np.mean(H, 0).tolist(), ll=np.mean(L, 0).tolist())) + "\n")
    n = len(grids()); print(f"tune {enc}: {len(open(fn).readlines()) if os.path.exists(fn) else 0}/{n} configs  {time.time()-t0:.0f}s")
def select(enc):
    rows = [json.loads(l) for l in open(f"{OUT}/tune/{enc}.jsonl")]; best = {}
    for r in rows:
        auc, ll = np.array(r["auc"]), np.array(r["ll"]); key = lambda i: (round(auc[i] / 5e-4), -ll[i]); R = max(range(len(auc)), key=key)
        key_ = (key(R), r["strat"] if r["strat"] != "dpfedavg" else "dp%s" % r["par"]["sigma"])
        slot = key_[1]
        if slot not in best or key_[0] > best[slot][0]: best[slot] = (key_[0], dict(par=r["par"], R=R + 1, val_auc=float(auc[R]), val_ll=float(ll[R])))
    sel = {k: v[1] for k, v in best.items()}; json.dump(sel, open(f"{OUT}/tuned_{enc}.json", "w"), indent=1)
    for k, v in sel.items(): print(enc, k, v)

# ------------------------------------------------------------------ final runs
def final(enc, seeds, budget):
    t0 = time.time(); sel = json.load(open(f"{OUT}/tuned_{enc}.json")); Xu = np.load(f"{F}/{enc}__plain.npz")["feat"].astype(np.float32)
    for s in seeds:
        fn = f"{OUT}/runs/{enc}__s{s}.npz"; store = dict(np.load(fn, allow_pickle=True)) if os.path.exists(fn) else {}
        meta = json.loads(str(store["meta"])) if "meta" in store else {}; st = Stage(Xu, s, True); store["test_rows"] = st.te; store["test_y"] = st.te_y
        def save(): store["meta"] = json.dumps(meta); np.savez_compressed(fn, **store)
        for sp in PARTS:
            cl = assign(sp, st, np.random.RandomState(500 + s)); cl_all = build_clients(st, cl, retain=False); cl_ret = build_clients(st, cl, retain=True)
            info = dict(n_clients_retained=len(cl_ret), groups=[int(c.ng) for c in cl_all.values()], retained_groups=[int(c.ng) for c in cl_ret.values()])
            fa = sel["fedavg"]; base = f"{sp}|"
            if len(cl_ret) < 2: meta[base + "skipped"] = info; continue
            todo = [("central_all", None), ("central_ret", None), ("local", None)] + [(k, None) for k in ("fedavg", "fedprox", "scaffold")]
            if sp in DP_PARTS: todo += [(f"{k}_dp{sg}", sg) for sg in SIGMAS for k in ("fedavg", "fedprox", "scaffold")]
            for name, sg in todo:
                key = base + name
                if key in store: continue
                if time.time() - t0 > budget - 25: save(); print(f"final {enc} budget hit at seed {s} part {sp}"); return
                if name == "central_all":
                    allg = build_clients(st, np.zeros(len(st.garr), int), retain=False); th, _, _ = run_fl({0: allg[0]}, "fedavg", fa["par"]["lr"], fa["par"]["E"], fa["R"], s); p = test_probs(st, th)
                elif name == "central_ret":
                    mask = np.isin(cl, list(cl_ret.keys())); cm_ = np.where(mask, 0, 1); one = build_clients(st, cm_, retain=False)[0]
                    th, _, _ = run_fl({0: one}, "fedavg", fa["par"]["lr"], fa["par"]["E"], fa["R"], s); p = test_probs(st, th)
                elif name == "local":
                    ps = []
                    for k, c in cl_ret.items(): th, _, _ = run_fl({k: c}, "fedavg", fa["par"]["lr"], fa["par"]["E"], fa["R"], s); ps.append(test_probs(st, th))
                    p = np.stack(ps)
                elif sg is None:
                    cfg = sel[name]; th, _, _ = run_fl(cl_ret, name, cfg["par"]["lr"], cfg["par"]["E"], cfg["R"], s, mu=cfg["par"].get("mu", 0.0)); p = test_probs(st, th)
                else:
                    strat = name.split("_dp")[0]; cfg = sel[f"dp{sg}"]; par = cfg["par"]; mu = sel["fedprox"]["par"].get("mu", 0.0) if strat == "fedprox" else 0.0
                    th, _, _ = run_fl(cl_ret, strat, par["lr"], par["E"], cfg["R"], s, mu=mu, dp=(par["C"], par["sigma"], par["q"]))
                    p = test_probs(st, th); info[name + "_eps"] = epsilon(par["q"], par["sigma"], cfg["R"] * int(round(par["E"] / par["q"])))
                store[key] = p.astype(np.float32); meta[key] = dict(info) if name == "central_all" else {k_: v for k_, v in info.items() if k_.startswith(name) or k_ in ("n_clients_retained", "groups", "retained_groups")}
            save()
        print(f"final {enc} seed {s} done  {time.time()-t0:.0f}s", flush=True)


# ------------------------------------------------------------------ v2 non-DP tuning (stability-aware)
# v1 selected lr=10 by mean validation AUC over 2 seeds; on other seeds the full-batch GD iterate diverged (central AUC down to 0.78).
# v2: stable learning-rate grid, 3 tuning seeds x 2 heterogeneous partitions = 6 runs, up to ROUNDS2 rounds, and selection by the
# WORST-CASE (minimum over the 6 runs) validation AUC at the chosen round, ties (5e-4) broken by mean validation log-loss.
ROUNDS2 = 200; TS = (0, 1, 2)
def grids2():
    g = []
    for lr in (0.3, 1.0, 3.0):
        for E in (1, 5):
            g.append(("fedavg", dict(lr=lr, E=E))); g.append(("scaffold", dict(lr=lr, E=E)))
            for mu in (0.01, 0.1, 0.5, 2.0): g.append(("fedprox", dict(lr=lr, E=E, mu=mu)))
    return g
def tune2(enc, budget):
    t0 = time.time(); Xu = np.load(f"{F}/{enc}__plain.npz")["feat"].astype(np.float32); os.makedirs(f"{OUT}/tune2", exist_ok=True); fn = f"{OUT}/tune2/{enc}.jsonl"
    done = {json.loads(l)["cid"] for l in open(fn)} if os.path.exists(fn) else set(); sts = {s: Stage(Xu, s, False) for s in TS}
    cls = {(sp, s): build_clients(sts[s], assign(sp, sts[s], np.random.RandomState(100 + s))) for sp in TUNE_PARTS for s in TS}
    for strat, par in grids2():
        cid = strat + "|" + json.dumps(par, sort_keys=True)
        if cid in done: continue
        if time.time() - t0 > budget - 20: break
        H, L = [], []
        for (sp, s), c in cls.items():
            _, h, l = run_fl(c, strat, par["lr"], par["E"], ROUNDS2, s, mu=par.get("mu", 0.0), st=sts[s], record_val=True); H.append(h); L.append(l)
        with open(fn, "a") as f: f.write(json.dumps(dict(cid=cid, strat=strat, par=par, auc=H, ll=L)) + "\n")
    print(f"tune2 {enc}: {len(open(fn).readlines()) if os.path.exists(fn) else 0}/{len(grids2())} configs  {time.time()-t0:.0f}s", flush=True)
def select2(enc):
    rows = {}
    for l in open(f"{OUT}/tune2/{enc}.jsonl"): r = json.loads(l); rows[r["cid"]] = r
    old = f"{OUT}/tuned_{enc}_v1.json"
    if not os.path.exists(old): os.replace(f"{OUT}/tuned_{enc}.json", old)
    sel = json.load(open(old)); best = {}
    for r in rows.values():
        A, L = np.array(r["auc"]), np.array(r["ll"]); mn, mean_ll = A.min(0), L.mean(0); Rb = max(range(A.shape[1]), key=lambda i: (round(mn[i] / 5e-4), -mean_ll[i]))
        key = (round(mn[Rb] / 5e-4), -mean_ll[Rb])
        if r["strat"] not in best or key > best[r["strat"]][0]: best[r["strat"]] = (key, dict(par=r["par"], R=Rb + 1, val_auc=float(A[:, Rb].mean()), val_auc_min=float(mn[Rb]), val_ll=float(mean_ll[Rb])))
    for k, v in best.items(): sel[k] = v[1]; print(enc, k, v[1])
    if abs(sel["fedprox"]["par"]["mu"] - 0.01) > 1e-9: print("WARNING: tuned mu != 0.01 (stored DP FedProx runs used mu=0.01)")
    json.dump(sel, open(f"{OUT}/tuned_{enc}.json", "w"), indent=1)
def purge(enc):
    """drop stored NON-DP run results so that final() recomputes them with the v2 configuration (DP runs are kept)"""
    base = ("central_all", "central_ret", "local", "fedavg", "fedprox", "scaffold")
    for fn in sorted(os.listdir(f"{OUT}/runs")):
        if not fn.startswith(enc + "__"): continue
        d = dict(np.load(f"{OUT}/runs/{fn}", allow_pickle=True)); meta = json.loads(str(d["meta"])) if "meta" in d else {}
        drop = [k for k in d if "|" in k and k.split("|")[1] in base]; [d.pop(k) for k in drop]
        for k in [k for k in meta if k.split("|")[1] in base + ("skipped",)]: meta.pop(k)
        d["meta"] = json.dumps(meta); np.savez_compressed(f"{OUT}/runs/{fn}", **d); print(fn, "purged", len(drop))

if __name__ == "__main__":
    m = sys.argv[1]
    if m == "selftest": selftest()
    elif m == "tune": tune(sys.argv[2], float(sys.argv[3]))
    elif m == "select": select(sys.argv[2])
    elif m == "tune2": tune2(sys.argv[2], float(sys.argv[3]))
    elif m == "select2": select2(sys.argv[2])
    elif m == "purge": purge(sys.argv[2])
    elif m == "final": final(sys.argv[2], [int(x) for x in sys.argv[3].split(",")], float(sys.argv[4]))
