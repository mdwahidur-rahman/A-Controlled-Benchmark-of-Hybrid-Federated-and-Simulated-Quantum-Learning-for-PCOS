import os, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("PCOS_BENCH_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))
