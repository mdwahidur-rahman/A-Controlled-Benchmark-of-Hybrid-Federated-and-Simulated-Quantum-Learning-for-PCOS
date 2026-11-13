# Duplicate-aware benchmark on a public PCOS ultrasound archive: code, configurations, tests and aggregate results

This repository accompanies the manuscript "A Controlled Benchmark of Hybrid, Federated and Simulated Quantum Learning for Polycystic Ovary Syndrome Ultrasound". It is meant for reviewers who want to inspect the pipeline, re-run the checks that do not need the images, or regenerate the results from the public archive.

**Scope and honesty notes**

* This is a controlled benchmark and audit, not a new model and not a diagnostic tool. Nothing here supports a claim of clinical readiness.
* The archive has no patient identifiers. Splits are disjoint at the level of duplicate groups, not patients.
* The quantum track is a noiseless statevector simulation. It says nothing about physical hardware.
* The federated track is simulated on one machine, and its privacy accounting is at the level of duplicate groups.
* No images and no per-image predictions are shipped. Only metadata and aggregate result tables are.

## Layout

```
configs/benchmark.yaml   fixed settings (checked against the code by tests/test_configs.py)
data/                    manifest.csv, splits.csv, weights_manifest.json  (metadata only; see data/README.md)
pipeline/                the scripts that produced every reported number, in run order
results/tables/*.csv     aggregate results that the manuscript tables are generated from
results/*.json           dataset audit, split summary, selected quantum configurations, simulator self-tests
results/fl/tuned_*.json  validation-selected federated and group-level-DP settings per encoder
tests/                   24 tests that run without the images
requirements.txt, environment.yml, LICENSE.md, .gitignore
```

## Quick check without the images (about 2 seconds)

```
pip install -r requirements.txt
PCOS_BENCH_ROOT=$PWD python -m pytest -q tests
```

The tests verify that (i) the metric code matches scikit-learn on synthetic data, including ties and integer sample weights; (ii) the bootstrap weights are reproducible from seed 2026; (iii) the Holm adjustment and the expected-calibration-error code behave as specified; (iv) the shipped manifest has 11,784 images, 6,784 positives, 3,996 byte-unique files and 3,255 class-pure duplicate groups, and every group-based split is disjoint while the image-level contrast split is not; (v) the result tables are internally consistent (confusion rates, interval ordering, the better / worse / equivalent / inconclusive call rule, Holm values); (vi) the YAML record agrees with the constants in the scripts; (vii) the encoder rank correlations recompute from `main_results.csv`, the ROC operating points are internally consistent, and the stored federated selections lie on the searched grids.

## Re-running the pipeline from the public archive

Environment variables (all paths are relative to where you run the scripts unless set):

| Variable | Meaning | Default |
|---|---|---|
| `PCOS_BENCH_ROOT` | repository root | current directory |
| `PCOS_RAW` | folder with `infected/` and `noninfected/` from the figshare archive | `data/raw/PCOS` |
| `PCOS_WEIGHTS` | folder for the pretrained ImageNet weights | `weights` |

The runs reported in the paper used Python 3.10.12 on CPU only (versions in `requirements.txt`). Run from the repository root with `PCOS_BENCH_ROOT=$PWD`.

| Step | Script | What it does |
|---|---|---|
| 0 | `00_fingerprint.py`, `01_audit.py`, `02_manifest_splits.py` | Byte and perceptual-hash fingerprints, dataset audit, duplicate groups g0 and g4, five-seed splits under three rules (`manifest.csv` and `splits.csv` are shipped in `data/`) |
| 1 | `03_download_weights.py`, `04_extract.py` | Pretrained weights, frozen-encoder features for three preprocessing variants, cached per byte-unique file |
| 2 | `05_heads.py`, `06_baselines.py`, `07_analysis.py` | Classical heads on the coarse C grid, no-pixel baselines (size-only, metadata, label-permuted), statistics |
| 3 | `14_tuning.py`, `14b_tuning_analysis.py` | Validation-tuned heads and ensembles |
| 4 | `15_fl.py`, `16_fl_analysis.py` | Non-IID federated simulation (FedAvg, FedProx, SCAFFOLD) with group-level DP and the RDP accountant |
| 5 | `11_quantum.py`, `17_qtune.py`, `17b_qtune_select.py`, `18_qfinal.py`, `12_quantum_analysis.py`, `20_quantum_final_analysis.py`, `q_driver.sh` | Noiseless simulated circuit, matched classical controls, equal-budget tuning, final runs, statistics |
| 6 | `09_gradcam.py`, `19_xai_arms.py`, `19b_xai_summary.py` | Six explanation methods on the classical, federated, federated + DP and quantum arms, faithfulness against random controls |
| 7 | `23_performance.py` (`selftest`, `run`, `assemble`) | 13 metrics with group-cluster bootstrap intervals and paired contrasts |
| 8 | `28_calibration.py` | Expected calibration error, reliability bins, paired ECE contrasts |
| 9 | `08_figures.py`, `10_gradcam_figures.py`, `13_quantum_figures.py`, `21_figs_phase2.py`, `22_figs_xai.py`, `24_figs_performance.py`, `26_manuscript_figures.py` | Figures, drawn only from stored result files |
| 10 | `32_extra_stats.py`, `31_schematics.py` | Encoder rank stability across preprocessing variants and ROC operating points (`results/tables/encoder_rank_stability.csv`, `roc_operating_points.csv`); code-drawn workflow and architecture figures |
| 11 | `33_figs_methods.py` (`preproc`, `xai`, `lit`), `34_fig_fl_qml.py` | Methods and Discussion figures of the Scientific Reports version: preprocessing grid with intensity histograms (writes `results/tables/preproc_stats.csv`), the three explanation grids (classical, federated, quantum; read `results/xai_arms/display_*.npz`), the literature bar chart, and the schematic of the federated track and the quantum slot (Fig. 3). Usage: `PCOS_BENCH_ROOT=$PWD OUT=<figure dir> python pipeline/33_figs_methods.py preproc` (and `xai`, `lit`); `34_fig_fl_qml.py` also needs `PREV=<preview dir>` |

Several steps are resumable and time-budgeted (the original runs were split into short sessions); see each script's docstring for its arguments. Neural-network training is seeded, but results can differ in the last digits across library versions and CPU types.

## Statistical protocol in one paragraph

The unit of partition, resampling and federated clients is the duplicate group g0 (same class and identical dihedral-canonical 64-bit dHash). Splits are 0.64 / 0.16 / 0.20 with seeds 0 to 4. Hyperparameters are chosen on validation data only; classical and federated models are refit on train plus validation, the quantum track is not. Metrics use a fixed threshold of 0.5 (positive class = PCOS-positive) and are computed per test partition and averaged. Intervals are 95 percent percentile intervals from 1,000 cluster-bootstrap draws that resample g0 groups, with the same draws shared by every row so that contrasts are paired. Contrasts are called better or worse when the interval excludes zero, equivalent when it includes zero and lies inside the margin (0.5 accuracy points, 0.010 MCC, 0.002 AUC), and inconclusive otherwise. Holm adjustment is applied within families.

## Important Notes

* A model that sees only image width and height reaches accuracy 98.59 percent and MCC 0.971 under the same duplicate-aware protocol, so the benchmark cannot show that any track learned ovarian morphology on this archive.
* Encoders are frozen and ImageNet-pretrained; no fine-tuned or from-scratch model was trained.
* Explanations come from one encoder, one seed and 20 images and were not reviewed by a clinician.
* Calibration is reported without recalibration; the bootstrap intervals of the expected calibration error are shifted upward by resampling.
* The compute environment was not logged per run.
* `26_manuscript_figures.py` (ROC and some other panels) and `32_extra_stats.py` read the stored per-image probabilities in `results/heads/`, which are not shipped; they run after step 2 has been re-run. The CSV files in `results/tables/` are the source of the manuscript tables.
* `33_figs_methods.py` reads the raw images (`PCOS_RAW`) and, for the explanation grids, `results/xai_arms/display_*.npz`, which `19_xai_arms.py` writes and which contain image pixels; they are therefore not shipped. `34_fig_fl_qml.py` and `31_schematics.py` need no data.
* No license has been chosen yet (`LICENSE.md`), and no archive DOI exists yet.

## Citation

Will be added.
