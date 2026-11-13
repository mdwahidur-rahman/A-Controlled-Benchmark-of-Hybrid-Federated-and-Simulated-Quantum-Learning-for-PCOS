"""Unit tests of the statistics code (pipeline/23_performance.py and pipeline/28_calibration.py) on synthetic data and the shipped manifest."""
import importlib.util, os, pathlib, numpy as np
import pytest
from sklearn import metrics as sm
ROOT = pathlib.Path(__file__).resolve().parents[1]

def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, ROOT / "pipeline" / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

@pytest.fixture(scope="module")
def perf(): return load("perf", "23_performance.py")

@pytest.fixture(scope="module")
def cal(perf): return load("cal", "28_calibration.py")

def synth(n=600, seed=0):
    r = np.random.RandomState(seed); y = (r.rand(n) < 0.55).astype(int); p = np.clip(0.5 + (y - 0.5) * 0.7 + r.randn(n) * 0.25, 0.001, 0.999); p[::7] = np.round(p[::7], 1)  # includes ties
    return p, y

def test_metrics_match_sklearn(perf):
    p, y = synth(); pr = (p >= 0.5).astype(int); out, ct = perf.metrics(p, y, np.ones((1, len(y)))); out = out[0]; tn, fp, fn, tp = sm.confusion_matrix(y, pr).ravel()
    ref = [sm.accuracy_score(y, pr), sm.balanced_accuracy_score(y, pr), sm.precision_score(y, pr), sm.recall_score(y, pr), tn / (tn + fp), tn / (tn + fn), sm.f1_score(y, pr),
           sm.matthews_corrcoef(y, pr), sm.cohen_kappa_score(y, pr), sm.roc_auc_score(y, p), sm.average_precision_score(y, p), sm.brier_score_loss(y, p), sm.log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))]
    assert np.abs(out - np.array(ref)).max() < 1e-9
    assert tuple(ct[0]) == (tp, fp, tn, fn)

def test_integer_weights_equal_repeated_rows(perf):
    p, y = synth(300, 1); w = np.random.RandomState(2).randint(0, 3, size=len(y)); b, _ = perf.metrics(p, y, w[None].astype(float)); rep = np.repeat(np.arange(len(y)), w)
    ref = [sm.matthews_corrcoef(y[rep], (p[rep] >= 0.5).astype(int)), sm.roc_auc_score(y[rep], p[rep]), sm.average_precision_score(y[rep], p[rep]), sm.cohen_kappa_score(y[rep], (p[rep] >= 0.5).astype(int))]
    assert np.abs(b[0][[7, 9, 10, 8]] - np.array(ref)).max() < 1e-9

def test_bootstrap_weights(perf):
    assert perf.W.shape == (perf.BOOT, perf.G) and perf.BOOT == 1000 and perf.G == 3255
    assert np.all(perf.W.sum(1) == perf.G)
    again = np.random.RandomState(2026).multinomial(perf.G, np.full(perf.G, 1.0 / perf.G), size=perf.BOOT).astype(float)
    assert np.array_equal(again, perf.W)

def test_holm_matches_manual(perf):
    p = np.array([0.01, 0.04, 0.03, 0.005]); adj = perf.holm(p); o = np.argsort(p); m = len(p); run = 0; ref = np.empty(m)
    for k, i in enumerate(o): run = max(run, (m - k) * p[i]); ref[i] = min(1, run)
    assert np.allclose(adj, ref) and np.all(adj >= p)

def test_ece_matches_loop_and_perfect_calibration(cal):
    p, y = synth(2000, 3); yy = y.astype(float); fast = cal.ece_w(p, yy, np.ones((1, len(y))))[0]; tot = 0.0; NB = cal.NB
    for k in range(NB):
        m = (p >= k / NB) & (p < (k + 1) / NB) if k < NB - 1 else (p >= k / NB)
        if m.any(): tot += m.sum() * abs(yy[m].mean() - p[m].mean())
    assert abs(fast - tot / len(p)) < 1e-12
    r = np.random.RandomState(4); pp = r.rand(200000); yy2 = (r.rand(200000) < pp).astype(float); assert cal.ece_w(pp, yy2, np.ones((1, len(pp))))[0] < 0.01
