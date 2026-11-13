#!/usr/bin/env python3
"""Resumable, chunked (HTTP Range) download of pretrained weights from GitHub releases.
Weights land in ~/weights (sandbox scratch, not the user's folders). Integrity: timm
file names embed the first 8 hex chars of the SHA-256; we verify it when present."""
import os, sys, re, json, time, hashlib, subprocess
B = os.environ.get("PCOS_BENCH_ROOT", os.getcwd())
W = os.environ.get("PCOS_WEIGHTS", "weights"); os.makedirs(W, exist_ok=True)
M = json.load(open(f"{B}/data/weights_manifest.json"))
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
CH = 8 * 1024 * 1024
t0 = time.time()

def rng(url, a, b, out):
    r = subprocess.run(["curl", "-sS", "-L", "-m", "70", "-r", f"{a}-{b}", "-D", "-", "-o", out, url],
                       capture_output=True, text=True)
    hdr = r.stdout
    codes = re.findall(r"^HTTP/\S+ (\d+)", hdr, re.M)
    tot = re.findall(r"^content-range:.*?/(\d+)", hdr, re.M | re.I)
    return (codes[-1] if codes else "000"), (int(tot[-1]) if tot else None)

def sha_ok(path, fname):
    m = re.search(r"-([0-9a-f]{8})\.pth$", fname)
    if not m: return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""): h.update(blk)
    return h.hexdigest()[:8] == m.group(1)

status = {}
def one_pass():
    moved = 0
    for name, v in M.items():
        final = f"{W}/{v['file']}"; part = final + ".part"
        if os.path.exists(final):
            status[name] = "done"; continue
        if time.time() - t0 > BUDGET: status[name] = "pending"; continue
        off = os.path.getsize(part) if os.path.exists(part) else 0
        total = None; fails = 0
        while time.time() - t0 < BUDGET:
            code, tot = rng(v["url"], off, off + CH - 1, "/tmp/chunk.bin")
            if code in ("206", "200") and os.path.exists("/tmp/chunk.bin") and os.path.getsize("/tmp/chunk.bin") > 0 and code == "206":
                total = tot or total
                with open(part, "ab") as f: f.write(open("/tmp/chunk.bin", "rb").read())
                off = os.path.getsize(part); fails = 0
                if total and off >= total: break
            elif code == "416":          # past end: complete
                break
            else:
                fails += 1
                if fails >= 3: status[name] = f"stuck(http {code})"; break
                time.sleep(1.0)
        if status.get(name, "").startswith("stuck"): continue
        if total and off >= total:
            ok = sha_ok(part, v["file"])
            if ok is False:
                os.remove(part); status[name] = "SHA MISMATCH (deleted)"; continue
            os.rename(part, final)
            status[name] = f"done {off/1048576:.1f} MB sha={'ok' if ok else 'n/a'}"
        else:
            status[name] = f"partial {off/1048576:.1f}/{(total or 0)/1048576:.1f} MB"
    return moved

while time.time() - t0 < BUDGET:
    status.clear()
    one_pass()
    if all(v.startswith("done") for v in status.values()): break
    time.sleep(3)
for n, s in status.items(): print(f"{n:30s} {s}")
print(f"elapsed {time.time()-t0:.0f}s")
