#!/usr/bin/env python3
"""Stage 0 raw fingerprinting (resumable).
Per image: byte size, width, height, JPEG quant-table[0][0], md5, dihedral-canonical
dHash/aHash, mean and std intensity. Used to (a) recover and verify acquisition groups,
(b) audit exact/near duplicates, (c) build a metadata-only shortcut baseline."""
import os, sys, csv, hashlib, time
from multiprocessing import Pool
import numpy as np
from PIL import Image

ROOT = os.environ.get("PCOS_RAW", "data/raw/PCOS")
OUT = os.path.join(os.environ.get("PCOS_BENCH_ROOT", os.getcwd()), "data", "fingerprints.csv")
LABEL = {"noninfected": 0, "infected": 1}
FIELDS = ["cls", "label", "file", "bytes", "width", "height", "q0", "md5",
          "dhash_canon", "ahash_canon", "mean", "std"]
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0

def bits(a):
    return "".join("1" if x else "0" for x in a.flatten())

def dihedral(a):
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out += [r, np.fliplr(r)]
    return out

def fp(args):
    cls, f = args
    p = os.path.join(ROOT, cls, f)
    raw = open(p, "rb").read()
    im0 = Image.open(p)
    try:
        q0 = int(im0.quantization[0][0])
    except Exception:
        q0 = -1
    im = im0.convert("L")
    w, h = im.size
    g9 = np.asarray(im.resize((9, 9), Image.BILINEAR), dtype=np.float32)
    g8 = np.asarray(im.resize((8, 8), Image.BILINEAR), dtype=np.float32)
    dh = min(bits(t[:8, 1:] > t[:8, :-1]) for t in dihedral(g9))
    ah = min(bits(t > t.mean()) for t in dihedral(g8))
    arr = np.asarray(im, dtype=np.float32)
    return (cls, LABEL[cls], f, len(raw), w, h, q0, hashlib.md5(raw).hexdigest(),
            dh, ah, round(float(arr.mean()), 3), round(float(arr.std()), 3))

if __name__ == "__main__":
    t0 = time.time()
    done = set()
    new = (not os.path.exists(OUT)) or os.path.getsize(OUT) == 0
    if not new:
        with open(OUT) as fh:
            for r in csv.DictReader(fh):
                done.add((r["cls"], r["file"]))
    todo = [(c, f) for c in LABEL for f in sorted(os.listdir(os.path.join(ROOT, c)))
            if (c, f) not in done]
    total = len(done) + len(todo)
    with open(OUT, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(FIELDS)
        with Pool(3) as pool:
            for i, row in enumerate(pool.imap_unordered(fp, todo, chunksize=16)):
                w.writerow(row)
                if i % 200 == 0:
                    fh.flush()
                if time.time() - t0 > BUDGET:
                    pool.terminate()
                    break
        fh.flush()
    n = sum(1 for _ in open(OUT)) - 1
    print(f"fingerprinted {n}/{total} in {time.time()-t0:.0f}s")
