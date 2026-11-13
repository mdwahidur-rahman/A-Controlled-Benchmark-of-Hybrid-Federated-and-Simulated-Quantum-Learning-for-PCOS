"""Internal-consistency checks on the shipped result tables."""
import pathlib, numpy as np, pandas as pd
T = pathlib.Path(__file__).resolve().parents[1] / "results" / "tables"

def test_main_results_shape_and_best_encoder():
    d = pd.read_csv(T / "main_results.csv"); g = d[(d.variant == "plain") & (d.rule == "group_g0")]
    assert len(g) == 11 and g.encoder.nunique() == 11
    top = g.sort_values("acc", ascending=False).iloc[0]; assert top.encoder == "eca_botnext26ts_256" and round(100 * top.acc, 2) == 98.68 and round(top.mcc, 3) == 0.973
    assert (g.acc_lo <= g.acc).all() and (g.acc <= g.acc_hi).all()

def test_confusion_rates_consistent():
    c = pd.read_csv(T / "performance_confusion.csv"); assert np.allclose(c.n, c[["TP", "FP", "TN", "FN"]].sum(axis=1))
    assert np.allclose(c.sens, c.TP / (c.TP + c.FN)) and np.allclose(c.spec, c.TN / (c.TN + c.FP)) and np.allclose(c.fp_rate, 1 - c.spec) and np.allclose(c.fn_rate, 1 - c.sens)

def test_performance_metrics_bounds():
    m = pd.read_csv(T / "performance_metrics.csv")
    for k in ("acc", "auc", "ap", "mcc"): assert (m[k + "_lo"] <= m[k + "_hi"]).all() and m[k].between(0, 1).all()

def test_contrast_call_rule_and_holm():
    c = pd.read_csv(T / "performance_contrasts.csv")
    def rule(r): return "better" if r.lo > 0 else "worse" if r.hi < 0 else ("equivalent" if (r.lo > -r.margin and r.hi < r.margin) else "inconclusive")
    assert (c.apply(rule, axis=1) == c.call).all()
    assert (c.holm_p >= c.boot_p - 1e-12).all() and (c.holm_p <= 1).all()

def test_size_only_reference_is_close_to_best_encoder():
    b = pd.read_csv(T / "baselines_metadata_summary.csv"); r = b[(b.model == "meta_size_only_LR") & (b.rule == "group_g0")].iloc[0]
    assert round(100 * r.acc_mean, 2) == 98.59 and round(r.mcc_mean, 3) == 0.971

def test_calibration_tables():
    e = pd.read_csv(T / "calibration_ece.csv"); assert e.ece.between(0, 0.2).all() and (e.ece_lo <= e.ece_hi).all() and len(e) == 15
    b = pd.read_csv(T / "calibration_bins.csv"); assert (b.groupby("row_id").bin.count() == 10).all()
    c = pd.read_csv(T / "calibration_contrasts.csv"); assert (c.lo <= c.hi).all()

def test_encoder_rank_stability_recomputes_from_main_results():
    from scipy.stats import spearmanr
    m = pd.read_csv(T / "main_results.csv"); g = m[m.rule == "group_g0"]; a = {v: g[g.variant == v].set_index("encoder").acc for v in ("plain", "clahe", "neutral")}
    s = pd.read_csv(T / "encoder_rank_stability.csv"); assert len(s) == 3 and (s.n_encoders == 11).all() and s.p_value.between(0, 1).all()
    for _, r in s.iterrows():
        x = spearmanr(a[r.variant_a], a[r.variant_b]); assert abs(x.statistic - r.spearman) < 1e-9 and abs(x.pvalue - r.p_value) < 1e-9

def test_roc_operating_points_consistent():
    r = pd.read_csv(T / "roc_operating_points.csv"); assert len(r) == 11
    assert (r[["sens_at_spec95", "sens_at_spec99", "avg_precision"]].apply(lambda c: c.between(0, 1)).all()).all()
    assert (r.sens_at_spec99 <= r.sens_at_spec95 + 1e-12).all()

def test_federated_tuned_selection_files():
    import json
    for enc in ("mobilenetv2_100", "mobilevit_s", "swin_tiny_patch4_window7_224"):
        d = json.load(open(T.parent / "fl" / f"tuned_{enc}.json"))
        assert set(d) == {"fedavg", "fedprox", "scaffold", "dp0.8", "dp1.2", "dp2.0", "dp4.0"}
        for k in ("fedavg", "fedprox", "scaffold"):
            assert d[k]["par"]["E"] in (1, 5) and d[k]["par"]["lr"] in (0.3, 1.0, 3.0) and 1 <= d[k]["R"] <= 200
        assert all(d[f"dp{s}"]["par"]["C"] in (0.25, 0.5, 1.0, 2.0, 5.0) and d[f"dp{s}"]["par"]["lr"] in (0.1, 0.3, 1.0) and d[f"dp{s}"]["par"]["q"] == 0.1 for s in ("0.8", "1.2", "2.0", "4.0"))
