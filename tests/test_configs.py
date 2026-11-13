"""The YAML record and the scripts must agree on the fixed settings."""
import pathlib, yaml
ROOT = pathlib.Path(__file__).resolve().parents[1]; cfg = yaml.safe_load((ROOT / "configs" / "benchmark.yaml").read_text())
src = lambda f: (ROOT / "pipeline" / f).read_text()

def test_splits_and_seeds():
    s = src("02_manifest_splits.py"); assert "PROPS = np.array([0.64, 0.16, 0.20])" in s and "SEEDS = [0, 1, 2, 3, 4]" in s
    assert cfg["splits"]["proportions"] == [0.64, 0.16, 0.20] and cfg["splits"]["seeds"] == [0, 1, 2, 3, 4]

def test_encoders_variants_and_grids():
    t = src("14_tuning.py");
    for e in cfg["encoders"]["all"]: assert f'"{e}"' in t
    assert 'VARIANTS = ["plain", "clahe", "neutral"]' in src("04_extract.py") and cfg["preprocessing_variants"] == ["plain", "clahe", "neutral"]
    assert "CS = [0.003, 0.03, 0.3]" in src("05_heads.py")
    assert sum(cfg["classical"]["tuning_trials"].values()) == 11 + 9 + 6 == 26

def test_federated_constants():
    f = src("15_fl.py"); fl = cfg["federated"]
    assert f"K = {fl['clients']};" in f and f"ROUNDS = {fl['rounds']}" in f and "LAM = 1e-3" in f and "DELTA = 1e-5" in f
    assert "SIGMAS = [0.8, 1.2, 2.0, 4.0]" in f and "Q = 0.1" in f and 'DP_PARTS = ["iid", "lab0.5", "lab0.1", "comb0.5"]' in f

def test_statistics_constants(): 
    p = src("23_performance.py"); st = cfg["statistics"]
    assert f"BOOT = {st['bootstrap_draws']}" in p and f"RandomState({st['bootstrap_seed']})" in p
    assert '{"mcc": 0.01, "acc": 0.005, "auc": 0.002}' in p and st["equivalence_margins"] == {"accuracy": 0.005, "mcc": 0.01, "auc": 0.002}
    assert f"NB = {st['ece_bins']}" in src("28_calibration.py")

def test_quantum_constants():
    q = src("11_quantum.py"); qc = cfg["quantum"]
    assert '"EPOCHS", "80"' in q and '"PATIENCE", "15"' in q and "BS, WD = " in q and '"LRC", "0.0003"' in q and '"LRS", "0.03"' in q and '"LRL", "0.003"' in q
    assert qc["max_epochs"] == 80 and qc["patience"] == 15 and qc["batch_size"] == 128
