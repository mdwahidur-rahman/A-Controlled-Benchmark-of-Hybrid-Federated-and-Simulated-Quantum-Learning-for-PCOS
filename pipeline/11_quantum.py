#!/usr/bin/env python3
"""Stages 5-6: the transform slot and its control bank, on frozen features (noiseless statevector simulation, pure torch).
Slot under test = a variational quantum circuit (VQC). Everything outside the slot (encoder features, scaler, compressor
init, readout init, optimizer, schedule, batch order, seeds) is identical within a contrast; only the slot differs.
  Setting B (hybrid layer):  x -> Linear(D,n) -> tanh -> SLOT -> Linear(n,1)
  Setting A (two-branch fusion):  logit = Linear(D,1)(x) + Linear(n,1)(SLOT(tanh(Linear(D,n)(x'))))   x' = x (matched) or a permuted partner
Slots: identity | dense_w (hidden width = n) | dense_p (parameter count ~ VQC) | vqc | vqc_noent (closed-form twin, no CNOTs).
Usage: 11_quantum.py selftest | run BUDGET --enc e --var v [--conds c1,c2]"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); torch.set_num_threads(int(os.environ.get("QT", "1")))
B_ = os.environ.get("PCOS_BENCH_ROOT", os.getcwd()); F = f"{B_}/features"; OUT = os.environ.get("QOUT", f"{B_}/results/quantum"); os.makedirs(OUT, exist_ok=True)
NQ, NL, HP = int(os.environ.get("NQ", "6")), int(os.environ.get("NL", "3")), int(os.environ.get("HP", "4")); PHISTD = float(os.environ.get("PHISTD", "0.5"))                     # qubits, variational layers, hidden width of the parameter-matched dense control
EPOCHS, PATIENCE, BS, WD = int(os.environ.get("EPOCHS", "80")), int(os.environ.get("PATIENCE", "15")), 128, 1e-3
LRC = float(os.environ.get("LRC", "0.0003")); LRS = float(os.environ.get("LRS", "0.03"))   # lr for the feature compressor / classical branch vs slot + readout (same for every slot)
LRL = float(os.environ.get("LRL", "0.003"))                                                       # lr for the classical branch (Setting A only)
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]; RREP = 3

# ------------------------------------------------------------------ simulator
def bits(n):
    idx = np.arange(2 ** n); return np.stack([(idx >> (n - 1 - q)) & 1 for q in range(n)])          # (n, 2^n), qubit 0 = most significant
def cnot_matrix(n, c, t):
    idx = np.arange(2 ** n); b = bits(n); j = idx ^ (b[c] << (n - 1 - t)); M = np.zeros((2 ** n, 2 ** n)); M[j, idx] = 1.0; return M
def cnot_layers(n, mode, L):
    """CNOT list per variational layer. ring: sequential (c -> c+1 mod n); pairs: brick pattern of nearest-neighbor CNOTs; none: no entangler."""
    if mode == "none": return [[] for _ in range(L)]
    if mode == "ring": return [[(c, (c + 1) % n) for c in range(n)] for _ in range(L)]
    return [[(i, i + 1) for i in range(l % 2, n - 1, 2)] for l in range(L)]
def layer_matrix(n, pairs):
    M = np.eye(2 ** n)
    for c, t in pairs: M = cnot_matrix(n, c, t) @ M
    return M
def rot_mats(phi):
    """phi (...,3) -> (...,2,2) complex U = RZ(c) RY(b) RZ(a). Gate order: RZ(a) first."""
    a, b, c = phi.unbind(-1); cb, sb = torch.cos(b / 2), torch.sin(b / 2)
    em = lambda x: torch.complex(torch.cos(x / 2), -torch.sin(x / 2)); ep = lambda x: torch.complex(torch.cos(x / 2), torch.sin(x / 2))
    U00 = em(c) * cb * em(a); U01 = -em(c) * sb * ep(a); U10 = ep(c) * sb * em(a); U11 = ep(c) * cb * ep(a)
    return torch.stack([torch.stack([U00, U01], -1), torch.stack([U10, U11], -1)], -2)
class VQC(nn.Module):
    def __init__(self, n=NQ, L=NL, entangle="ring", dtype=torch.complex64):
        super().__init__(); entangle = {True: "ring", False: "none"}.get(entangle, entangle)
        self.n, self.L, self.mode, self.cd = n, L, entangle, dtype
        self.phi = nn.Parameter(PHISTD * torch.randn(L, n, 3))
        rd = torch.float32 if dtype == torch.complex64 else torch.float64
        self.register_buffer("P", torch.tensor(np.stack([layer_matrix(n, pl) for pl in cnot_layers(n, entangle, L)]), dtype=rd).to(dtype))
        self.register_buffer("zs", torch.tensor(1.0 - 2.0 * bits(n), dtype=rd))
    def unitary(self):
        U = rot_mats(self.phi.to(torch.float32 if self.cd == torch.complex64 else torch.float64)); W = None
        for l in range(self.L):
            K = U[l, 0]
            for q in range(1, self.n): K = torch.kron(K, U[l, q])
            M = self.P[l] @ K; W = M if W is None else M @ W
        return W
    def encode(self, u):
        th = np.pi * u; v = torch.stack([torch.cos(th / 2), torch.sin(th / 2)], -1)                   # (B,n,2): RY(pi u)|0>
        st = v[:, 0]
        for q in range(1, self.n): st = (st[:, :, None] * v[:, q][:, None, :]).reshape(u.shape[0], -1)
        return st.to(self.cd)
    def forward(self, u):
        st = self.encode(u) @ self.unitary().T; pr = st.real ** 2 + st.imag ** 2
        return pr @ self.zs.T                                                                           # <Z_i>, (B,n)
def ref_expz(u, phi, mode, n):
    """Slow reference: gate-by-gate einsum on complex128 (independent of the kron path)."""
    mode = {True: "ring", False: "none"}.get(mode, mode); B = u.shape[0]; st = torch.zeros(B, 2 ** n, dtype=torch.complex128); st[:, 0] = 1
    def ap(st, U, q):
        s = st.reshape(B, 2 ** q, 2, 2 ** (n - q - 1)); return (torch.einsum("ij,bxjy->bxiy", U, s) if U.dim() == 2 else torch.einsum("bij,bxjy->bxiy", U, s)).reshape(B, -1)
    for q in range(n):
        th = np.pi * u[:, q]; c, s_ = torch.cos(th / 2), torch.sin(th / 2)
        RY = torch.stack([torch.stack([c, -s_], -1), torch.stack([s_, c], -1)], -2).to(torch.complex128); st = ap(st, RY, q)
    Um = rot_mats(phi.to(torch.float64)); lay = cnot_layers(n, mode, phi.shape[0])
    for l in range(phi.shape[0]):
        for q in range(n): st = ap(st, Um[l, q], q)
        for c, t in lay[l]:
            j = np.arange(2 ** n) ^ (bits(n)[c] << (n - 1 - t)); st = st[:, torch.tensor(j)]          # CNOT is an involution: new[i] = old[perm[i]]
    pr = st.real ** 2 + st.imag ** 2; return pr @ torch.tensor(1.0 - 2.0 * bits(n), dtype=torch.float64).T
def closed_form(u, phi):
    """Entangler-free twin: each qubit is a Bloch-vector rotation; <Z_i> = z-component of R_i (sin a, 0, cos a), a = pi u_i."""
    L, n, _ = phi.shape; out = np.zeros(u.shape)
    Rz = lambda t: np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    Ry = lambda t: np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    for q in range(n):
        R = np.eye(3)
        for l in range(L): a, b, c = phi[l, q]; R = (Rz(c) @ Ry(b) @ Rz(a)) @ R
        v = np.stack([np.sin(np.pi * u[:, q]), np.zeros(len(u)), np.cos(np.pi * u[:, q])]); out[:, q] = (R @ v)[2]
    return out

def selftest():
    torch.manual_seed(0); rep = {}; u = torch.rand(7, NQ, dtype=torch.float64) * 2 - 1
    for ent in ("ring", "pairs", "none"):
        m = VQC(NQ, NL, ent, torch.complex128).double(); fast = m(u.double()).detach(); slow = ref_expz(u, m.phi.detach(), ent, NQ)
        W = m.unitary().detach(); rep[f"kron_vs_gate_by_gate_maxerr_{ent}"] = float((fast - slow).abs().max())
        rep[f"unitarity_maxerr_{ent}"] = float((W.conj().T @ W - torch.eye(2 ** NQ, dtype=torch.complex128)).abs().max())
        if ent == "none": rep["closed_form_vs_simulator_maxerr"] = float(np.abs(closed_form(u.numpy(), m.phi.detach().numpy()) - fast.numpy()).max())
    # Bell test: RY(pi/2) on qubit 0 then CNOT(0,1) -> (|00>+|11>)/sqrt2 : <Z0>=0, <Z0 Z1>=1
    n = 2; st = torch.zeros(1, 4, dtype=torch.complex128); st[0, 0] = 1
    c = np.cos(np.pi / 4); s_ = np.sin(np.pi / 4); st = torch.tensor([[c, 0, s_, 0]], dtype=torch.complex128)    # RY(pi/2)|0> (x) |0>
    st = st[:, torch.tensor(np.arange(4) ^ (bits(2)[0] << 0))]; pr = (st.real ** 2 + st.imag ** 2)[0].numpy(); zz = (pr * (1 - 2 * bits(2)[0]) * (1 - 2 * bits(2)[1])).sum()
    rep["bell_Z0Z1"] = float(zz); rep["bell_Z0"] = float((pr * (1 - 2 * bits(2)[0])).sum())
    m = VQC(NQ, NL, "ring"); ma = VQC(NQ, NL, "none"); mp = VQC(NQ, NL, "pairs"); dp = nn.Sequential(nn.Linear(NQ, HP), nn.Tanh(), nn.Linear(HP, NQ)); dw = nn.Linear(NQ, NQ)
    cnt = lambda x: sum(p.numel() for p in x.parameters())
    rep["slot_params"] = dict(vqc=cnt(m), vqc_pairs=cnt(mp), vqc_noent=cnt(ma), dense_w=cnt(dw), dense_p=cnt(dp), identity=0)
    rep["n_qubits"], rep["layers"] = NQ, NL; json.dump(rep, open(f"{OUT}/selftest.json", "w"), indent=1); print(json.dumps(rep, indent=1))
    assert rep["closed_form_vs_simulator_maxerr"] < 1e-9 and abs(rep["bell_Z0Z1"] - 1) < 1e-12 and abs(rep["bell_Z0"]) < 1e-12

# ------------------------------------------------------------------ model
class Slot(nn.Module):
    def __init__(self, kind):
        super().__init__(); self.kind = kind
        if kind == "dense_w": self.a = nn.Linear(NQ, NQ)
        elif kind == "dense_p": self.a = nn.Linear(NQ, HP); self.b = nn.Linear(HP, NQ)
        elif kind == "vqc": self.q = VQC(NQ, NL, "ring")
        elif kind == "vqc_pairs": self.q = VQC(NQ, NL, "pairs")
        elif kind == "vqc_noent": self.q = VQC(NQ, NL, "none")
    def forward(self, u):
        k = self.kind
        if k == "identity": return u
        if k == "dense_w": return torch.tanh(self.a(u))
        if k == "dense_p": return torch.tanh(self.b(torch.tanh(self.a(u))))
        return self.q(u)
class Net(nn.Module):
    def __init__(self, D, kind, seed, fusion):
        super().__init__(); torch.manual_seed(seed); self.kind, self.fusion = kind, fusion
        if fusion or kind == "classical": self.lin = nn.Linear(D, 1)
        if kind != "classical":
            self.comp = nn.Linear(D, NQ); self.read = nn.Linear(NQ, 1); torch.manual_seed(seed + 777); self.slot = Slot(kind)
    def forward(self, x, x2=None):
        if self.kind == "classical": return self.lin(x).squeeze(-1)
        z = self.read(self.slot(torch.tanh(self.comp(x2 if (self.fusion and x2 is not None) else x)))).squeeze(-1)
        return z + self.lin(x).squeeze(-1) if self.fusion else z

def derange(groups, rng):
    n = len(groups); p = rng.permutation(n)
    for _ in range(2000):
        bad = np.nonzero(groups[p] == groups)[0]
        if len(bad) == 0: break
        for i in bad:
            j = rng.randint(n)
            if groups[p[j]] != groups[i] and groups[p[i]] != groups[j]: p[i], p[j] = p[j], p[i]
    assert (groups[p] != groups).all(); return p
def pairing(kind, groups, labels, rng):
    n = len(groups)
    if kind == "shuf": return rng.permutation(n)
    if kind == "derg": return derange(groups, rng)
    out = np.arange(n)                                                     # label-preserving: another group, same label
    for lab in np.unique(labels):
        ids = np.nonzero(labels == lab)[0]; out[ids] = ids[derange(groups[ids], rng)]
    assert (labels[out] == labels).all() and (groups[out] != groups).all(); return out

# ------------------------------------------------------------------ data + fit
man = pd.read_csv(f"{B_}/data/manifest.csv"); spl = pd.read_csv(f"{B_}/data/splits.csv"); uidx = pd.read_csv(f"{F}/unique_index.csv")
pos = {m: i for i, m in enumerate(uidx.md5)}; row2u = man.md5.map(pos).values; Y = man.label.values; N = len(man); G0 = pd.factorize(man.g0)[0]
def collapse(rows):
    u, inv = np.unique(row2u[rows], return_inverse=True); w = np.bincount(inv).astype(np.float32)
    lab = np.zeros(len(u), int); lab[inv] = Y[rows]; grp = np.zeros(len(u), int); grp[inv] = G0[rows]; return u, inv, w, lab, grp
def run_fit(Xu, seed, kind, fusion, pair):
    part = spl[f"grp_s{seed}"].values; tr, va, te = (np.nonzero(part == k)[0] for k in ("train", "val", "test"))
    mu = Xu[row2u[tr]].mean(0); sd = Xu[row2u[tr]].std(0) + 1e-6
    sets = {}
    for nm, rows in (("tr", tr), ("va", va), ("te", te)):
        u, inv, w, lab, grp = collapse(rows); X = torch.tensor((Xu[u] - mu) / sd, dtype=torch.float32)
        if pair is None: X2 = None
        else:
            r = np.random.RandomState(1000 * seed + 17 * pair[1] + {"shuf": 1, "derg": 2, "lp": 3}[pair[0]] + 7 * {"tr": 0, "va": 1, "te": 2}[nm])
            X2 = X[torch.tensor(pairing(pair[0], grp, lab, r))]
        sets[nm] = dict(X=X, X2=X2, y=torch.tensor(lab, dtype=torch.float32), w=torch.tensor(w), lab=lab, wn=w, inv=inv)
    D = Xu.shape[1]; net = Net(D, kind, seed, fusion); grp = lambda pre: [p for nme, p in net.named_parameters() if nme.startswith(pre)]
    cp, lp_ = grp("comp"), grp("lin"); sp = [p for nme, p in net.named_parameters() if not nme.startswith(("comp", "lin"))]
    opt = torch.optim.AdamW([dict(params=g_, lr=l_) for g_, l_ in ((cp, LRC), (lp_, LRL), (sp, LRS)) if g_], weight_decay=WD)
    g = torch.Generator().manual_seed(seed); tr_, va_, te_ = sets["tr"], sets["va"], sets["te"]; n = len(tr_["y"]); best, state, bad, bep = -1, None, 0, 0
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(n, generator=g)
        for i in range(0, n, BS):
            idx = perm[i:i + BS]; lg = net(tr_["X"][idx], None if tr_["X2"] is None else tr_["X2"][idx])
            loss = (bce(lg, tr_["y"][idx]) * tr_["w"][idx]).sum() / tr_["w"][idx].sum(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): pv = torch.sigmoid(net(va_["X"], va_["X2"])).numpy()
        a = roc_auc_score(va_["lab"], pv, sample_weight=va_["wn"])
        if a > best + 1e-5: best, bad, bep = a, 0, ep; state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: break
    net.load_state_dict(state)
    with torch.no_grad(): pt = torch.sigmoid(net(te_["X"], te_["X2"])).numpy()
    globals()["LAST"] = dict(net=net, mu=mu, sd=sd)      # exposed for the XAI script (no effect on the runner)
    nsl = sum(p.numel() for p in net.slot.parameters()) if hasattr(net, "slot") else 0
    return te, pt[te_["inv"]], best, bep, nsl, sum(p.numel() for p in net.parameters())

CONDS = {  # name -> (kind, fusion, pairing kind)
    "B_identity": ("identity", False, None), "B_dense_w": ("dense_w", False, None), "B_dense_p": ("dense_p", False, None),
    "B_vqc": ("vqc", False, None), "B_vqc_pairs": ("vqc_pairs", False, None), "B_vqc_noent": ("vqc_noent", False, None),
    "A_classical": ("classical", True, None), "A_vqc": ("vqc", True, None), "A_dense_w": ("dense_w", True, None), "A_dense_p": ("dense_p", True, None),
    "A_vqc_shuf": ("vqc", True, "shuf"), "A_vqc_derg": ("vqc", True, "derg"), "A_vqc_lp": ("vqc", True, "lp"),
    "A_vqc_pairs": ("vqc_pairs", True, None), "A_vqc_pairs_shuf": ("vqc_pairs", True, "shuf"), "A_vqc_pairs_derg": ("vqc_pairs", True, "derg"), "A_vqc_pairs_lp": ("vqc_pairs", True, "lp")}
def jobs(enc, var, conds):
    for c in conds:
        for r in (range(RREP) if CONDS[c][2] else [0]): yield (c, r)

if __name__ == "__main__":
    if sys.argv[1] == "selftest": selftest(); sys.exit()
    BUDGET = float(sys.argv[2]); T0 = time.time()
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1].split(",") if k in sys.argv else d
    enc = arg("--enc", ["mobilenetv2_100"])[0]; var = arg("--var", ["plain"])[0]; conds = arg("--conds", list(CONDS))
    Xu = np.load(f"{F}/{enc}__{var}.npz")["feat"].astype(np.float32); est = 0.0; done = 0; todo = 0
    for c, r in jobs(enc, var, conds):
        out = f"{OUT}/{enc}__{var}__{c}__r{r}.npz"
        if os.path.exists(out): continue
        todo += 1
        if time.time() - T0 + est > BUDGET - 8: continue
        t1 = time.time(); kind, fusion, pk = CONDS[c]; P = np.full((len(SEEDS), N), np.nan, np.float32); va_, bep_, nsl_, ntot_, tm_ = [], [], [], [], []
        for si, s in enumerate(SEEDS):
            t2 = time.time(); te, p, va, bep, nsl, ntot = run_fit(Xu, s, kind, fusion, None if pk is None else (pk, r))
            P[si, te] = p; va_.append(va); bep_.append(bep); nsl_.append(nsl); ntot_.append(ntot); tm_.append(time.time() - t2)
        np.savez(out, probs=P, val_auc=np.array(va_), best_ep=np.array(bep_), slot_params=nsl_[0], total_params=ntot_[0], secs=np.array(tm_))
        est = max(est, time.time() - t1); done += 1
        acc = np.mean([((P[i, ~np.isnan(P[i])] >= 0.5).astype(int) == Y[~np.isnan(P[i])]).mean() for i in range(len(SEEDS))])
        print(f"{enc[:12]} {var:7s} {c:12s} r{r} acc={acc:.4f} valAUC={np.mean(va_):.4f} slot_params={nsl_[0]} ep={np.mean(bep_):.0f} ({time.time()-t1:.0f}s)", flush=True)
    print(f"{enc} {var}: new {done}, remaining {todo-done}, elapsed {time.time()-T0:.0f}s", flush=True)
