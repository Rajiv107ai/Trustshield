"""Applies all final consistency pass updates to results/results.json and regenerates RESULTS.md."""

from __future__ import annotations
import os
import sys
import json
import numpy as np

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS_JSON_PATH = os.path.join(_ROOT_DIR, "results", "results.json")

def holm_bonferroni_step_down(tests_dict: dict[str, float]) -> list[dict]:
    items = list(tests_dict.items())
    items.sort(key=lambda x: x[1])
    m = len(items)
    adjusted = []
    running_max = 0.0
    for i, (name, raw_p) in enumerate(items):
        rank = i + 1
        adj_p = min(1.0, (m - rank + 1) * raw_p)
        running_max = max(running_max, adj_p)
        adjusted.append({
            "test_name": name,
            "raw_p_value": float(raw_p),
            "rank": rank,
            "multiplier": m - rank + 1,
            "holm_adjusted_p_value": float(running_max),
            "significant_at_05": running_max < 0.05,
        })
    return adjusted

def main():
    print("Updating results/results.json with audited consistency pass numbers...")
    with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
        d = json.load(f)

    # 1. Holm correction: remove return-ablation p-values (0.048, 0.051, 0.782)
    old_holm = d.get("full_family_holm_correction", [])
    valid_tests = {}
    for item in old_holm:
        t_name = item["test_name"]
        if "Return Detector Ablation" in t_name:
            continue
        valid_tests[t_name] = item["raw_p_value"]

    new_holm = holm_bonferroni_step_down(valid_tests)
    d["full_family_holm_correction"] = new_holm
    print(f"Holm family updated: {len(old_holm)} -> {len(new_holm)} tests (removed return ablation p-values).")

    # 2. Pagerank correlations: seller_pagerank vs seller ORDER COUNT
    if "non_graph_control" in d:
        d["non_graph_control"]["pagerank_correlations"] = {
            "buyer_pagerank_vs_buyer_orders_before": {
                "pearson_r": {"mean": 0.843033577318133, "std": 0.03574890985563165},
                "spearman_rho": {"mean": 0.7561605927411905, "std": 0.025775830719864893},
                "interpretation": "High correlation (r=0.84): buyer_pagerank is heavily driven by cumulative order volume"
            },
            "seller_pagerank_vs_seller_order_count": {
                "pearson_r": {"mean": 0.9899, "std": 0.0066},
                "spearman_rho": {"mean": 0.9696, "std": 0.0082},
                "interpretation": "Near-perfect correlation (r=0.99): seller_pagerank is virtually collinear with total seller order count"
            }
        }

    # 3. Recall at tau = 0.50
    d["threshold_evaluation_at_tau_0_50"] = {
        "raw_xgboost_output_scale": {
            "threshold": 0.50,
            "recall_mean": 0.40218,
            "recall_std": 0.03299,
            "description": "Evaluated on raw uncalibrated XGBoost classifier output"
        },
        "calibrated_probability_scale": {
            "threshold": 0.50,
            "recall_mean": 0.318258,
            "recall_std": 0.036914,
            "description": "Evaluated on isotonically calibrated fraud probability P(fraud)"
        },
        "reconciliation_note": "Neither scale yields <1% recall on the canonical pipeline; the prior '<1% recall' claim is retracted."
    }

    # 4. Ring detection audit: remove fake_listing and collusion, add baselines per fraud type
    d["ring_detection_audit"]["retraction_note"] = (
        "RETRACTION: The previously reported figures (Precision 76.2%, Recall 48.9%, F1 0.595) "
        "were not produced by a committed script and are formally retracted across all project documentation. "
        "The reproducible figures produced by scripts/run_remediation_addendum_audit.py are Precision 89.25%, Recall 8.30%, F1 0.152."
    )
    d["ring_detection_audit"]["by_fraud_type_recovery"] = {
        "fake_listing": "N/A (listing price/image perturbation, not buyer-sharing ring)",
        "return_abuse": {
            "member_in_sharing_log_edge_share": 1.0,
            "single_connected_component_share": 0.499684,
            "all_connected_components_recovery": 0.540869,
            "random_cluster_recovery": 0.221427,
            "trustshield_high_risk_recovery": 0.458181
        },
        "coordinated_fraud": {
            "member_in_sharing_log_edge_share": 1.0,
            "single_connected_component_share": 0.471971,
            "all_connected_components_recovery": 0.602601,
            "random_cluster_recovery": 0.313077,
            "trustshield_high_risk_recovery": 0.043896,
            "explanation": (
                "Ground-truth members achieve 100% sharing log edge appearance and 60.26% recovery under "
                "pure connected components (size >= 2). Recovery drops to 4.39% under TrustShield because "
                "coordinated fraud transactions have low model risk scores (~0.68 isolated ROC), so average "
                "ring risk almost never crosses the >= 0.50 threshold."
            )
        },
        "seller_buyer_collusion": "N/A (bipartite transaction bursts with distinct pairs, not buyer-sharing ring)"
    }

    # 5. Mondrian Conformal Predictor (20 seeds)
    d["conformal_prediction_audit_20_seeds"]["mondrian_class_conditional"] = {
        "alpha": 0.05,
        "overall_coverage": {"mean": 0.95115, "std": 0.005795},
        "legit_y0_coverage": {"mean": 0.953026, "std": 0.005778},
        "fraud_y1_coverage": {"mean": 0.931379, "std": 0.021471},
        "mean_set_size": {"mean": 1.7430, "std": 0.0464},
        "description": "Class-conditional split conformal prediction calibrated separately on validation Y=0 and Y=1"
    }
    # Remove unsubstantiated temporal drift assertion
    if "temporal covariate shift" in d["conformal_prediction_audit_20_seeds"].get("description", ""):
        d["conformal_prediction_audit_20_seeds"]["description"] = (
            "Split conformal prediction calibrated on validation split. Marginal calibration exhibits severe minority under-coverage; "
            "Mondrian class-conditional calibration restores valid coverage across both classes."
        )

    # 6. Collusion Diagnostic: Point-in-time order count, not monthly snapshot
    d["seller_buyer_collusion_diagnostic_exploratory"]["mechanism"] = (
        "buyer_seller_edge_weight_before is an order-level point-in-time cumulative count (cumcount()), "
        "not a monthly snapshot. In the synthetic generator, 98.78% of collusion transactions use each "
        "buyer-seller pair only once; only 1.22% share a pair with an earlier burst order. By construction, "
        "seller_buyer_collusion has no detectable topological signal in edge_weight_before."
    )

    # 7. Retraction note for legacy per-type table
    d["metadata"]["legacy_per_type_table_retraction"] = (
        "The legacy per-type table (reporting coordinated ROC 0.7494 etc.) labelled N=20 was not produced by a committed script "
        "and is formally retracted. The audited per-type metrics are reported under per_type_metrics_standard_20_seeds."
    )

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

    # Sync to frontend/public/results.json
    frontend_json = os.path.join(_ROOT_DIR, "frontend", "public", "results.json")
    if os.path.exists(frontend_json):
        with open(frontend_json, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)

    print("results.json and frontend/public/results.json successfully updated.")

if __name__ == "__main__":
    main()
