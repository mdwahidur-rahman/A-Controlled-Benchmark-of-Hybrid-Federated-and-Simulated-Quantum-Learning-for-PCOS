#!/usr/bin/env python3
"""Stage 2-3: frozen ImageNet-pretrained encoders -> pooled features, cached per (encoder, variant).
Resumable: each call processes pending images until its time budget, then checkpoints.
Features are computed once per BYTE-UNIQUE file and broadcast to all rows via md5.
Usage: 04_extract.py BUDGET_SECONDS [encoder ...]"""
import os, sys, io, json, time, warnings
import numpy as np, pandas as pd, torch, timm, cv2
from PIL import Image
warnings.filterwarnings("ignore")
torch.set_num_threads(3)

B = os.environ.get("PCOS_BENCH_ROOT", os.getcwd()); W = os.environ.get("PCOS_WEIGHTS", "weights")
RAW = os.environ.get("PCOS_RAW", "data/raw/PCOS"); F = f"{B}/features"
M = json.load(open(f"{B}/data/weights_manifest.json"))
ORDER = ["mobilenetv2_100", "efficientnet_b0", "densenet121", "resnet50", "pvt_v2_b2", "twins_pcpvt_small",
         "swin_tiny_patch4_window7_224", "mobilevit_s", "eca_botnext26ts_256", "crossvit_small_240", "resnext50_32x4d"]
VARIANTS = ["plain", "clahe", "neutral"]
BUDGET = float(sys.argv[1]); ONLY = sys.argv[2:] or ORDER
T0 = time.time()

man = pd.read_csv(f"{B}/data/manifest.csv")
uniq = man.drop_duplicates("md5").reset_index(drop=True)
if not os.path.exists(f"{F}/unique_index.csv"):
    uniq[["path", "md5", "label"]].to_csv(f"{F}/unique_index.csv", index=False)
N = len(uniq)

# --- local, weights-only loader that plugs into timm's own state-dict filtering
import timm.models.helpers as H
LOADED = []
def local_loader(url, map_location="cpu", progress=False, check_hash=False, **kw):
    p = f"{W}/{url.split('/')[-1]}"
    LOADED.append(p)
    return torch.load(p, map_location="cpu", weights_only=True)
H.load_state_dict_from_url = local_loader

def get_model(name):
    LOADED.clear()
    m = timm.create_model(name, pretrained=True, num_classes=0).eval()
    assert LOADED, f"{name}: pretrained weights were NOT loaded"
    return m

def to_rgb(path): return Image.open(path).convert("RGB")
def v_plain(im, s): return np.asarray(im.resize((s, s), Image.BICUBIC))
def v_clahe(im, s):
    g = np.asarray(im.convert("L")); c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    c = cv2.resize(c, (s, s), interpolation=cv2.INTER_CUBIC); return np.stack([c] * 3, -1)
def v_neutral(im, s):
    g = im.convert("L"); w, h = g.size; k = min(w, h); l, t = (w - k) // 2, (h - k) // 2
    g = g.crop((l, t, l + k, t + k)).resize((128, 128), Image.LANCZOS)
    buf = io.BytesIO(); g.save(buf, "JPEG", quality=75); g = Image.open(buf).convert("L")
    return np.stack([np.asarray(g.resize((s, s), Image.BICUBIC))] * 3, -1)
PREP = {"plain": v_plain, "clahe": v_clahe, "neutral": v_neutral}

def save(path, feat, done):
    tmp = path + ".tmp.npz"
    np.savez(tmp, feat=feat, done=done); os.replace(tmp, path)

report = []
for enc in [e for e in ORDER if e in ONLY]:
    if not os.path.exists(f"{W}/{M[enc]['file']}"):
        report.append((enc, "no weights yet")); continue
    pend = [v for v in VARIANTS if not (os.path.exists(f"{F}/{enc}__{v}.npz") and np.load(f"{F}/{enc}__{v}.npz")["done"].all())]
    if not pend:
        report.append((enc, "complete")); continue
    if time.time() - T0 > BUDGET - 8:
        report.append((enc, "pending")); continue
    model = get_model(enc)
    cfg = model.default_cfg; S = cfg["input_size"][1]
    mean = np.array(cfg["mean"], dtype=np.float32); std = np.array(cfg["std"], dtype=np.float32)
    for v in pend:
        path = f"{F}/{enc}__{v}.npz"
        if os.path.exists(path):
            d = np.load(path); feat, done = d["feat"], d["done"]
        else:
            feat, done = None, np.zeros(N, bool)
        todo = np.nonzero(~done)[0]; last_save = time.time(); BS = 16
        for i in range(0, len(todo), BS):
            if time.time() - T0 > BUDGET: break
            idx = todo[i:i + BS]
            x = np.stack([(PREP[v](to_rgb(f"{RAW}/{uniq.path[j]}"), S).astype(np.float32) / 255 - mean) / std for j in idx])
            with torch.inference_mode():
                out = model(torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()).float().numpy()
            assert np.isfinite(out).all(), f"non-finite features {enc}/{v}"
            if feat is None: feat = np.zeros((N, out.shape[1]), np.float16)
            feat[idx] = out.astype(np.float16); done[idx] = True
            if time.time() - last_save > 40: save(path, feat, done); last_save = time.time()
        if feat is not None: save(path, feat, done)
        report.append((f"{enc}/{v}", f"{int(done.sum())}/{N}" + (f" dim={feat.shape[1]} res={S}" if feat is not None else "")))
        if time.time() - T0 > BUDGET: break
    del model
    if time.time() - T0 > BUDGET: break

# overall completeness
tot = 0; fin = 0
for enc in ORDER:
    for v in VARIANTS:
        tot += 1
        p = f"{F}/{enc}__{v}.npz"
        if os.path.exists(p) and np.load(p)["done"].all(): fin += 1
print(" | ".join(f"{a}: {b}" for a, b in report))
print(f"COMPLETE {fin}/{tot} encoder-variants   elapsed {time.time()-T0:.0f}s")
