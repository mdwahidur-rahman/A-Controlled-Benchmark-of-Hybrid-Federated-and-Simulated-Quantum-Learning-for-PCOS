"""Row selection and display labels for the report-level performance tables and figures (all rows remain in results/tables/performance_*.csv)."""
import pandas as pd
from figstyle import PRETTY
def build(T):
    m = pd.read_csv(f"{T}/performance_metrics.csv").set_index("row_id")
    enc = lambda rid: PRETTY[m.loc[rid, "label"].split("(")[-1].rstrip(")")]
    G = []
    G.append(("Classical, plain images", [
        ("cls_main_best_plain", f"Main head, {enc('cls_main_best_plain')} (best)"), ("cls_main_mnv2_plain", "Main head, MobileNetV2"),
        ("cls_tuned_best_plain", f"Tuned head, {enc('cls_tuned_best_plain')} (best)"), ("cls_tuned_mnv2_plain", "Tuned head, MobileNetV2"),
        ("cls_ens_transformer_plain", "Tuned soft-vote ensemble, 4 Transformers"), ("cls_ens_all11_plain", "Tuned soft-vote ensemble, all 11")]))
    G.append(("Classical, shortcut-suppressed images", [
        ("cls_main_best_neutral", f"Main head, {enc('cls_main_best_neutral')} (best)"), ("cls_tuned_best_neutral", f"Tuned head, {enc('cls_tuned_best_neutral')} (best)"),
        ("cls_ens_all11_neutral", "Tuned soft-vote ensemble, all 11")]))
    G.append(("Federated, no privacy noise", [
        ("fl_central_all_iid", "Central, all data"), ("fl_fedavg_iid", "FedAvg, IID"), ("fl_fedavg_lab0.5", "FedAvg, label skew 0.5"),
        ("fl_central_ret_lab0.1", "Central, retained clients, label skew 0.1"), ("fl_fedavg_lab0.1", "FedAvg, label skew 0.1"), ("fl_fedavg_comb0.5", "FedAvg, combined skew")]))
    G.append(("Federated + DP, label skew 0.5", [
        ("fl_fedavg_dp0.8_lab0.5", "FedAvg + DP, σ = 0.8 (ε ≈ 40–47)"), ("fl_fedavg_dp1.2_lab0.5", "FedAvg + DP, σ = 1.2 (ε ≈ 18–20)"),
        ("fl_fedavg_dp2.0_lab0.5", "FedAvg + DP, σ = 2 (ε ≈ 9)"), ("fl_fedavg_dp4.0_lab0.5", "FedAvg + DP, σ = 4 (ε ≈ 3.7)")]))
    G.append(("Quantum, hybrid layer (setting B)", [
        ("q_B_identity", "Identity slot"), ("q_B_dense_w", "Dense slot, width-matched"), ("q_B_dense_p", "Dense slot, parameter-matched"),
        ("q_B_vqc_pairs", "VQC slot, nearest-neighbor CNOTs"), ("q_B_vqc", "VQC slot, ring CNOTs (borrowed config)")]))
    G.append(("Quantum, fusion branch (setting A)", [
        ("q_A_classical", "Classical-only head"), ("q_A_dense_w", "+ dense branch, width-matched"), ("q_A_dense_p", "+ dense branch, parameter-matched"),
        ("q_A_vqc_pairs", "+ VQC branch, nearest-neighbor CNOTs"), ("q_A_vqc_pairs_derg", "+ VQC branch, deranged partners")]))
    return m, G
