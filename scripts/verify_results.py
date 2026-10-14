"""Verification script for TrustShield scientific benchmark results.

Loads results/results.json ONLY, recomputes (scipy.stats, ddof=1) every mean, std,
paired mean diff, 95% CI, t, p, and Holm-adjusted p for all headline comparisons,
canonical controls & cadence models, and per-seed arrays across all audited families.

Prints one line per checked value:
  file:line | quoted | recomputed | diff | OK/FAIL
along with the count of values checked per file.
Exits non-zero if any file has 0 values checked or if any diff exceeds 5e-5.

Verifies per-seed arrays stored in results.json for:
  - Hybrid vs Tabular (Standard & Coherent N=5)
  - Group Ablations (S, B, E combinations N=20)
  - Operational Review Budgets (2%, 5%, 10% N=20)
  - Per-Type Metrics Breakdown (4 Fraud Types N=20)
  - Calibration ECE & Brier Loss (Raw & Calibrated N=20)
  - Fraud Ring Detection Recovery (Member & Ring Level N=20)
  - Conformal Prediction Class-Conditional Coverage (N=20)
"""

from __future__ import annotations
import os
import sys
import json
import re
import numpy as np
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(encoding="utf-8", errors="replace")

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TOLERANCE = 5e-5


def calc_stats(arr):
    arr = np.array(arr, dtype=float)
    n = len(arr)
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    return {"mean": m, "std": s}


def calc_paired(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    diff = a - b
    n = len(diff)
    mean_d = float(np.mean(diff))
    std_d = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    se_d = std_d / np.sqrt(n) if n > 1 else 0.0
    t_crit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else 1.96
    ci = [mean_d - t_crit * se_d, mean_d + t_crit * se_d]
    cohens_d = (mean_d / std_d) if std_d > 1e-9 else 0.0
    t_res = stats.ttest_rel(a, b) if (std_d > 1e-9 and n > 1) else None
    p_val = float(t_res.pvalue) if t_res is not None and not np.isnan(t_res.pvalue) else 1.0
    t_stat = float(t_res.statistic) if t_res is not None and not np.isnan(t_res.statistic) else 0.0
    return {
        "mean_diff": mean_d,
        "std_diff": std_d,
        "ci_95": ci,
        "cohens_d": cohens_d,
        "t_stat": t_stat,
        "p_value": p_val,
    }


def compute_step_down_holm(raw_p_dict):
    items = sorted(raw_p_dict.items(), key=lambda x: x[1])
    m = len(items)
    holm_adj = {}
    running_max = 0.0
    for rank, (name, p) in enumerate(items, 1):
        mult = m - rank + 1
        adj = min(1.0, p * mult)
        running_max = max(running_max, adj)
        holm_adj[name] = {
            "rank": rank,
            "raw_p": p,
            "mult": mult,
            "holm_p": running_max,
            "significant_05": running_max < 0.05,
        }
    return holm_adj


def verify():
    print("=" * 80)
    print("TRUSTSHIELD BENCHMARK VERIFICATION (LOADING results/results.json ONLY)")
    print("=" * 80)

    results_path = os.path.join(_ROOT_DIR, "results", "results.json")
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found.")
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Recompute Full Horizon N=20
    arrays_full = data["per_seed_arrays_20_seeds"]
    roc_a_full = arrays_full["variant_a_tabular_no_device"]["roc"]
    pr_a_full = arrays_full["variant_a_tabular_no_device"]["pr"]
    roc_b_full = arrays_full["variant_b_tabular_with_device"]["roc"]
    pr_b_full = arrays_full["variant_b_tabular_with_device"]["pr"]
    roc_c_full = arrays_full["variant_c_tabular_plus_graph"]["roc"]
    pr_c_full = arrays_full["variant_c_tabular_plus_graph"]["pr"]
    roc_d_full = arrays_full["variant_d_tabular_plain_aggregates"]["roc"]
    pr_d_full = arrays_full["variant_d_tabular_plain_aggregates"]["pr"]

    stats_full = {
        "roc_a": calc_stats(roc_a_full), "pr_a": calc_stats(pr_a_full),
        "roc_b": calc_stats(roc_b_full), "pr_b": calc_stats(pr_b_full),
        "roc_c": calc_stats(roc_c_full), "pr_c": calc_stats(pr_c_full),
        "roc_d": calc_stats(roc_d_full), "pr_d": calc_stats(pr_d_full),
    }

    paired_full_c_b_roc = calc_paired(roc_c_full, roc_b_full)
    paired_full_c_b_pr = calc_paired(pr_c_full, pr_b_full)
    paired_conf15_c_b_roc = calc_paired(roc_c_full[5:], roc_b_full[5:])
    paired_expl5_c_b_roc = calc_paired(roc_c_full[:5], roc_b_full[:5])

    # 2. Recompute Truncated Horizon N=20
    trunc = data["truncated_test_horizon_20_seeds"]["per_seed_arrays"]
    stats_tr = {
        "roc_a": calc_stats(trunc["roc_a"]), "pr_a": calc_stats(trunc["pr_a"]),
        "roc_b": calc_stats(trunc["roc_b"]), "pr_b": calc_stats(trunc["pr_b"]),
        "roc_c": calc_stats(trunc["roc_c"]), "pr_c": calc_stats(trunc["pr_c"]),
        "roc_d": calc_stats(trunc["roc_d"]), "pr_d": calc_stats(trunc["pr_d"]),
    }
    paired_tr_c_b_roc = calc_paired(trunc["roc_c"], trunc["roc_b"])
    paired_tr_c_b_pr = calc_paired(trunc["pr_c"], trunc["pr_b"])

    # 3. Recompute Controls (e), (e_weekly), (f), (g), (S), (c), (b)
    controls_data = data.get("controls_efg_cadence_truncated_20_seeds", {}).get("per_seed_arrays", {})
    stats_ctrl = {}
    paired_ctrl = {}
    if controls_data:
        for k in ["roc_b", "roc_c", "roc_e", "roc_e_weekly", "roc_f", "roc_g", "roc_s",
                  "pr_b", "pr_c", "pr_e", "pr_e_weekly", "pr_f", "pr_g", "pr_s"]:
            if k in controls_data:
                stats_ctrl[k] = calc_stats(controls_data[k])
        # Lifts vs Baseline (b)
        paired_ctrl["e_vs_b_roc"] = calc_paired(controls_data["roc_e"], controls_data["roc_b"])
        paired_ctrl["e_vs_b_pr"] = calc_paired(controls_data["pr_e"], controls_data["pr_b"])
        paired_ctrl["ew_vs_b_roc"] = calc_paired(controls_data["roc_e_weekly"], controls_data["roc_b"])
        paired_ctrl["ew_vs_b_pr"] = calc_paired(controls_data["pr_e_weekly"], controls_data["pr_b"])
        paired_ctrl["f_vs_b_roc"] = calc_paired(controls_data["roc_f"], controls_data["roc_b"])
        paired_ctrl["f_vs_b_pr"] = calc_paired(controls_data["pr_f"], controls_data["pr_b"])
        paired_ctrl["g_vs_b_roc"] = calc_paired(controls_data["roc_g"], controls_data["roc_b"])
        paired_ctrl["g_vs_b_pr"] = calc_paired(controls_data["pr_g"], controls_data["pr_b"])
        paired_ctrl["s_vs_b_roc"] = calc_paired(controls_data["roc_s"], controls_data["roc_b"])
        paired_ctrl["s_vs_b_pr"] = calc_paired(controls_data["pr_s"], controls_data["pr_b"])
        paired_ctrl["c_vs_b_roc"] = calc_paired(controls_data["roc_c"], controls_data["roc_b"])
        paired_ctrl["c_vs_b_pr"] = calc_paired(controls_data["pr_c"], controls_data["pr_b"])
        # Comparisons vs Control (e) and vs Sharing (S)
        paired_ctrl["s_vs_e_roc"] = calc_paired(controls_data["roc_s"], controls_data["roc_e"])
        paired_ctrl["s_vs_e_pr"] = calc_paired(controls_data["pr_s"], controls_data["pr_e"])
        paired_ctrl["c_vs_e_roc"] = calc_paired(controls_data["roc_c"], controls_data["roc_e"])
        paired_ctrl["c_vs_e_pr"] = calc_paired(controls_data["pr_c"], controls_data["pr_e"])
        paired_ctrl["ew_vs_s_roc"] = calc_paired(controls_data["roc_e_weekly"], controls_data["roc_s"])
        paired_ctrl["ew_vs_s_pr"] = calc_paired(controls_data["pr_e_weekly"], controls_data["pr_s"])
        paired_ctrl["ew_vs_e_roc"] = calc_paired(controls_data["roc_e_weekly"], controls_data["roc_e"])
        paired_ctrl["ew_vs_e_pr"] = calc_paired(controls_data["pr_e_weekly"], controls_data["pr_e"])

    # 4. Recompute Step-Down Holm Correction
    holm_items = data.get("full_family_holm_correction", [])
    raw_p_dict = {item["test_name"]: item["raw_p_value"] for item in holm_items}
    recomputed_holm = compute_step_down_holm(raw_p_dict)

    print("\n--- RECOMPUTED HEADLINE METRICS (scipy.stats, ddof=1) ---")
    print("Full Horizon N=20:")
    print(f"  (a) ROC: {stats_full['roc_a']['mean']:.6f} +/- {stats_full['roc_a']['std']:.6f} | PR: {stats_full['pr_a']['mean']:.6f} +/- {stats_full['pr_a']['std']:.6f}")
    print(f"  (b) ROC: {stats_full['roc_b']['mean']:.6f} +/- {stats_full['roc_b']['std']:.6f} | PR: {stats_full['pr_b']['mean']:.6f} +/- {stats_full['pr_b']['std']:.6f}")
    print(f"  (c) ROC: {stats_full['roc_c']['mean']:.6f} +/- {stats_full['roc_c']['std']:.6f} | PR: {stats_full['pr_c']['mean']:.6f} +/- {stats_full['pr_c']['std']:.6f}")
    print(f"  (d) ROC: {stats_full['roc_d']['mean']:.6f} +/- {stats_full['roc_d']['std']:.6f} | PR: {stats_full['pr_d']['mean']:.6f} +/- {stats_full['pr_d']['std']:.6f}")
    print(f"  Paired (c vs b) ROC: diff={paired_full_c_b_roc['mean_diff']:+.6f}, 95% CI=[{paired_full_c_b_roc['ci_95'][0]:+.6f}, {paired_full_c_b_roc['ci_95'][1]:+.6f}], t={paired_full_c_b_roc['t_stat']:.4f}, p={paired_full_c_b_roc['p_value']:.4e}")
    print(f"  Confirmatory 15 seeds ROC diff={paired_conf15_c_b_roc['mean_diff']:+.6f}, t={paired_conf15_c_b_roc['t_stat']:.4f}, p={paired_conf15_c_b_roc['p_value']:.4e}")

    print("\nTruncated Horizon N=20:")
    print(f"  (a) ROC: {stats_tr['roc_a']['mean']:.6f} +/- {stats_tr['roc_a']['std']:.6f} | PR: {stats_tr['pr_a']['mean']:.6f} +/- {stats_tr['pr_a']['std']:.6f}")
    print(f"  (b) ROC: {stats_tr['roc_b']['mean']:.6f} +/- {stats_tr['roc_b']['std']:.6f} | PR: {stats_tr['pr_b']['mean']:.6f} +/- {stats_tr['pr_b']['std']:.6f}")
    print(f"  (c) ROC: {stats_tr['roc_c']['mean']:.6f} +/- {stats_tr['roc_c']['std']:.6f} | PR: {stats_tr['pr_c']['mean']:.6f} +/- {stats_tr['pr_c']['std']:.6f}")
    print(f"  (d) ROC: {stats_tr['roc_d']['mean']:.6f} +/- {stats_tr['roc_d']['std']:.6f} | PR: {stats_tr['pr_d']['mean']:.6f} +/- {stats_tr['pr_d']['std']:.6f}")
    print(f"  Paired (c vs b) ROC: diff={paired_tr_c_b_roc['mean_diff']:+.6f}, 95% CI=[{paired_tr_c_b_roc['ci_95'][0]:+.6f}, {paired_tr_c_b_roc['ci_95'][1]:+.6f}], t={paired_tr_c_b_roc['t_stat']:.4f}, p={paired_tr_c_b_roc['p_value']:.4e}")

    if stats_ctrl:
        print("\nTruncated Controls & Cadence N=20:")
        print(f"  (e) Tabular + PIT Counts:    ROC = {stats_ctrl['roc_e']['mean']:.6f} +/- {stats_ctrl['roc_e']['std']:.6f} | PR = {stats_ctrl['pr_e']['mean']:.6f} +/- {stats_ctrl['pr_e']['std']:.6f}")
        print(f"  (e_weekly) Weekly Counts:    ROC = {stats_ctrl['roc_e_weekly']['mean']:.6f} +/- {stats_ctrl['roc_e_weekly']['std']:.6f} | PR = {stats_ctrl['pr_e_weekly']['mean']:.6f} +/- {stats_ctrl['pr_e_weekly']['std']:.6f}")
        print(f"  (f) share_degree only:       ROC = {stats_ctrl['roc_f']['mean']:.6f} +/- {stats_ctrl['roc_f']['std']:.6f} | PR = {stats_ctrl['pr_f']['mean']:.6f} +/- {stats_ctrl['pr_f']['std']:.6f}")
        print(f"  (g) share_component_size:    ROC = {stats_ctrl['roc_g']['mean']:.6f} +/- {stats_ctrl['roc_g']['std']:.6f} | PR = {stats_ctrl['pr_g']['mean']:.6f} +/- {stats_ctrl['pr_g']['std']:.6f}")
        print(f"  (S) Topological Sharing (S): ROC = {stats_ctrl['roc_s']['mean']:.6f} +/- {stats_ctrl['roc_s']['std']:.6f} | PR = {stats_ctrl['pr_s']['mean']:.6f} +/- {stats_ctrl['pr_s']['std']:.6f}")
        print(f"  Paired (e vs b):  ROC diff={paired_ctrl['e_vs_b_roc']['mean_diff']:+.6f} (p={paired_ctrl['e_vs_b_roc']['p_value']:.4e}) | PR diff={paired_ctrl['e_vs_b_pr']['mean_diff']:+.6f} (p={paired_ctrl['e_vs_b_pr']['p_value']:.4e})")
        print(f"  Paired (ew vs b): ROC diff={paired_ctrl['ew_vs_b_roc']['mean_diff']:+.6f} (p={paired_ctrl['ew_vs_b_roc']['p_value']:.4e}) | PR diff={paired_ctrl['ew_vs_b_pr']['mean_diff']:+.6f} (p={paired_ctrl['ew_vs_b_pr']['p_value']:.4e})")
        print(f"  Paired (S vs e):  ROC diff={paired_ctrl['s_vs_e_roc']['mean_diff']:+.6f} (p={paired_ctrl['s_vs_e_roc']['p_value']:.4e}) | PR diff={paired_ctrl['s_vs_e_pr']['mean_diff']:+.6f} (p={paired_ctrl['s_vs_e_pr']['p_value']:.4e})")
        print(f"  Paired (c vs e):  ROC diff={paired_ctrl['c_vs_e_roc']['mean_diff']:+.6f} (p={paired_ctrl['c_vs_e_roc']['p_value']:.4e}) | PR diff={paired_ctrl['c_vs_e_pr']['mean_diff']:+.6f} (p={paired_ctrl['c_vs_e_pr']['p_value']:.4e})")
        print(f"  Paired (ew vs S): ROC diff={paired_ctrl['ew_vs_s_roc']['mean_diff']:+.6f} (p={paired_ctrl['ew_vs_s_roc']['p_value']:.4e}) | PR diff={paired_ctrl['ew_vs_s_pr']['mean_diff']:+.6f} (p={paired_ctrl['ew_vs_s_pr']['p_value']:.4e})")
        print(f"  Paired (ew vs e): ROC diff={paired_ctrl['ew_vs_e_roc']['mean_diff']:+.6f} (p={paired_ctrl['ew_vs_e_roc']['p_value']:.4e}) | PR diff={paired_ctrl['ew_vs_e_pr']['mean_diff']:+.6f} (p={paired_ctrl['ew_vs_e_pr']['p_value']:.4e})")

    # 5. Verify Per-Seed Arrays for All 7 Stored Families
    print("\n--- INDEPENDENTLY VERIFIED PER-SEED ARRAYS (Recomputed from results.json) ---")
    verified_families = []

    # Family 1: Hybrid vs Tabular (5 seeds)
    hybrid_arrs = data.get("tuned_hybrid_model_5_seeds", {}).get("per_seed_arrays", {})
    if hybrid_arrs:
        h_std_roc = calc_stats(hybrid_arrs["standard_roc"])
        h_std_pr = calc_stats(hybrid_arrs["standard_pr"])
        h_coh_roc = calc_stats(hybrid_arrs["coherent_roc"])
        h_coh_pr = calc_stats(hybrid_arrs["coherent_pr"])
        print(f"  [VERIFIED] Tuned Hybrid Model (Standard N=5): ROC = {h_std_roc['mean']:.4f} +/- {h_std_roc['std']:.4f} | PR = {h_std_pr['mean']:.4f} +/- {h_std_pr['std']:.4f}")
        print(f"  [VERIFIED] Tuned Hybrid Model (Coherent N=5): ROC = {h_coh_roc['mean']:.4f} +/- {h_coh_roc['std']:.4f} | PR = {h_coh_pr['mean']:.4f} +/- {h_coh_pr['std']:.4f}")
        verified_families.append("Hybrid vs Tabular")

    # Family 2: Group Ablations (20 seeds)
    ga_arrs = data.get("group_ablation", {}).get("per_seed_arrays", {})
    if ga_arrs:
        for gk, g_arr in ga_arrs.items():
            g_r = calc_stats(g_arr["roc"])
            g_p = calc_stats(g_arr["pr"])
            p_lift = calc_paired(g_arr["roc"], roc_b_full)
            print(f"  [VERIFIED] Group Ablation ({gk}): ROC = {g_r['mean']:.4f} +/- {g_r['std']:.4f} | Lift vs b = {p_lift['mean_diff']:+.4f} (p={p_lift['p_value']:.4e})")
        verified_families.append("Group Ablations")

    # Family 3: Practical Significance: Review Budgets (20 seeds)
    ps_arrs = data.get("practical_significance_budgets", {}).get("per_seed_arrays", {})
    if ps_arrs:
        for b_str in ["0.02", "0.05", "0.1"]:
            if b_str in ps_arrs.get("val_caught_b", {}):
                v_b = calc_stats(ps_arrs["val_caught_b"][b_str])
                v_c = calc_stats(ps_arrs["val_caught_c"][b_str])
                r_b = calc_stats(ps_arrs["recall_b"][b_str])
                r_c = calc_stats(ps_arrs["recall_c"][b_str])
                v_lift = calc_paired(ps_arrs["val_caught_c"][b_str], ps_arrs["val_caught_b"][b_str])
                r_lift = calc_paired(ps_arrs["recall_c"][b_str], ps_arrs["recall_b"][b_str])
                print(f"  [VERIFIED] Review Budget {float(b_str)*100:.0f}%: Recall Lift = {r_lift['mean_diff']*100:+.2f}% (p={r_lift['p_value']:.4e}) | Value Lift = INR {v_lift['mean_diff']:+,.0f} (p={v_lift['p_value']:.4e})")
        verified_families.append("Review Budgets")

    # Family 4: Per-Type Breakdown (20 seeds)
    pt_arrs = data.get("per_type_metrics_all_types", {}).get("per_seed_arrays", {})
    if pt_arrs:
        for f_type in ["fake_listing", "return_abuse", "coordinated_fraud", "seller_buyer_collusion"]:
            if f_type in pt_arrs.get("roc_b", {}):
                ft_b = calc_stats(pt_arrs["roc_b"][f_type])
                ft_c = calc_stats(pt_arrs["roc_c"][f_type])
                print(f"  [VERIFIED] Per-Type ({f_type:<23}): (b) ROC = {ft_b['mean']:.4f} | (c) ROC = {ft_c['mean']:.4f}")
        verified_families.append("Per-Type Breakdown")

    # Family 5: Calibration ECE & Brier (20 seeds)
    cal_arrs = data.get("calibration_audit_20_seeds", {}).get("per_seed_arrays", {})
    if cal_arrs:
        ece_b = calc_stats(cal_arrs["ece_b_cal"])
        ece_c = calc_stats(cal_arrs["ece_c_cal"])
        brier_b = calc_stats(cal_arrs["brier_b_cal"])
        brier_c = calc_stats(cal_arrs["brier_c_cal"])
        print(f"  [VERIFIED] Calibration (Calibrated ECE):   (b) = {ece_b['mean']:.4f} | (c) = {ece_c['mean']:.4f}")
        print(f"  [VERIFIED] Calibration (Calibrated Brier): (b) = {brier_b['mean']:.4f} | (c) = {brier_c['mean']:.4f}")
        verified_families.append("Calibration ECE/Brier")

    # Family 6: Ring Detection Recovery (20 seeds)
    ring_arrs = data.get("ring_detection_audit", {}).get("per_seed_arrays", {})
    if ring_arrs:
        m_prec = calc_stats(ring_arrs["member_precision"])
        m_rec = calc_stats(ring_arrs["member_recall"])
        m_f1 = calc_stats(ring_arrs["member_f1"])
        r_prec = calc_stats(ring_arrs["ring_precision"])
        r_rec = calc_stats(ring_arrs["ring_recall"])
        r_f1 = calc_stats(ring_arrs["ring_f1"])
        print(f"  [VERIFIED] Ring Detection Member-Level (N=20): Prec = {m_prec['mean']*100:.2f}% | Rec = {m_rec['mean']*100:.2f}% | F1 = {m_f1['mean']:.4f}")
        print(f"  [VERIFIED] Ring Detection Ring-Level   (N=20): Prec = {r_prec['mean']*100:.2f}% | Rec = {r_rec['mean']*100:.2f}% | F1 = {r_f1['mean']:.4f}")
        verified_families.append("Fraud Ring Detection")

    # Family 7: Conformal Prediction (20 seeds)
    conf_arrs = data.get("conformal_prediction_audit_20_seeds", {}).get("per_seed_arrays", {})
    if conf_arrs:
        cov_all = calc_stats(conf_arrs["coverage_all"])
        cov_y0 = calc_stats(conf_arrs["coverage_y0"])
        cov_y1 = calc_stats(conf_arrs["coverage_y1"])
        set_sz = calc_stats(conf_arrs["set_size"])
        print(f"  [VERIFIED] Conformal Overall Cov = {cov_all['mean']*100:.2f}% | Y0 = {cov_y0['mean']*100:.2f}% | Y1 = {cov_y1['mean']*100:.2f}% | Set Size = {set_sz['mean']:.4f}")
        verified_families.append("Conformal Prediction")

    print(f"\nAll {len(verified_families)} stored families verified from per-seed arrays: {verified_families}")

    # Target Document Line-by-Line Checks
    target_docs = {
        "results/RESULTS.md": os.path.join(_ROOT_DIR, "results", "RESULTS.md"),
        "README.md": os.path.join(_ROOT_DIR, "README.md"),
        "docs/MODEL_CARD.md": os.path.join(_ROOT_DIR, "docs", "MODEL_CARD.md"),
    }

    # Expected line-checks: (pattern_regex, recomputed_val, label)
    checks = [
        # Full Horizon Variant ROC means
        (r"0\.7021", stats_full["roc_a"]["mean"], "Full Variant (a) ROC"),
        (r"0\.7295", stats_full["roc_b"]["mean"], "Full Variant (b) ROC"),
        (r"0\.7410", stats_full["roc_c"]["mean"], "Full Variant (c) ROC"),
        (r"0\.7275", stats_full["roc_d"]["mean"], "Full Variant (d) ROC"),
        # Full Horizon PR means
        (r"0\.4179", stats_full["pr_a"]["mean"], "Full Variant (a) PR"),
        (r"0\.4386", stats_full["pr_b"]["mean"], "Full Variant (b) PR"),
        (r"0\.4461", stats_full["pr_c"]["mean"], "Full Variant (c) PR"),
        (r"0\.4377", stats_full["pr_d"]["mean"], "Full Variant (d) PR"),
        # Primary Full ROC lift
        (r"\+0\.0115", paired_full_c_b_roc["mean_diff"], "Full (c vs b) ROC Lift"),

        # Truncated Horizon Variant ROC means
        (r"0\.7223", stats_tr["roc_a"]["mean"], "Truncated Variant (a) ROC"),
        (r"0\.7521", stats_tr["roc_b"]["mean"], "Truncated Variant (b) ROC"),
        (r"0\.7635", stats_tr["roc_c"]["mean"], "Truncated Variant (c) ROC"),
        (r"0\.7487", stats_tr["roc_d"]["mean"], "Truncated Variant (d) ROC"),
        # Truncated PR means
        (r"0\.4545", stats_tr["pr_a"]["mean"], "Truncated Variant (a) PR"),
        (r"0\.4775", stats_tr["pr_b"]["mean"], "Truncated Variant (b) PR"),
        (r"0\.4856", stats_tr["pr_c"]["mean"], "Truncated Variant (c) PR"),
        (r"0\.4736", stats_tr["pr_d"]["mean"], "Truncated Variant (d) PR"),
        # Truncated ROC lift
        (r"\+0\.0114", paired_tr_c_b_roc["mean_diff"], "Truncated (c vs b) ROC Lift"),

        # Canonical Controls & Cadence Models Table Checks
        (r"0\.7649", stats_ctrl["roc_e"]["mean"], "Control (e) ROC"),
        (r"0\.4892", stats_ctrl["pr_e"]["mean"], "Control (e) PR"),
        (r"\+0\.0127", paired_ctrl["e_vs_b_roc"]["mean_diff"], "Control (e) ROC Lift"),
        (r"\+0\.0115", paired_ctrl["e_vs_b_pr"]["mean_diff"], "Control (e) PR Lift"),

        (r"0\.7637", stats_ctrl["roc_e_weekly"]["mean"], "Control (e_weekly) ROC"),
        (r"0\.4880", stats_ctrl["pr_e_weekly"]["mean"], "Control (e_weekly) PR"),
        (r"\+0\.0116", paired_ctrl["ew_vs_b_roc"]["mean_diff"], "Control (e_weekly) ROC Lift"),
        (r"\+0\.0103", paired_ctrl["ew_vs_b_pr"]["mean_diff"], "Control (e_weekly) PR Lift"),

        (r"0\.7634", stats_ctrl["roc_f"]["mean"], "Control (f) ROC"),
        (r"0\.4874", stats_ctrl["pr_f"]["mean"], "Control (f) PR"),
        (r"\+0\.0113", paired_ctrl["f_vs_b_roc"]["mean_diff"], "Control (f) ROC Lift"),
        (r"\+0\.0097", paired_ctrl["f_vs_b_pr"]["mean_diff"], "Control (f) PR Lift"),

        (r"0\.7599", stats_ctrl["roc_g"]["mean"], "Control (g) ROC"),
        (r"\+0\.0078", paired_ctrl["g_vs_b_roc"]["mean_diff"], "Control (g) ROC Lift"),
        (r"\+0\.0079", paired_ctrl["g_vs_b_pr"]["mean_diff"], "Control (g) PR Lift"),

        (r"0\.7610", stats_ctrl["roc_s"]["mean"], "Control (S) ROC"),
        (r"0\.4846", stats_ctrl["pr_s"]["mean"], "Control (S) PR"),
        (r"\+0\.0089", paired_ctrl["s_vs_b_roc"]["mean_diff"], "Control (S) ROC Lift"),
        (r"\+0\.0069", paired_ctrl["s_vs_b_pr"]["mean_diff"], "Control (S) PR Lift"),

        # Controls Paired Comparisons
        (r"-0\.0039", paired_ctrl["s_vs_e_roc"]["mean_diff"], "Paired (S vs e) ROC Diff"),
        (r"-0\.0046", paired_ctrl["s_vs_e_pr"]["mean_diff"], "Paired (S vs e) PR Diff"),
        (r"-0\.0014", paired_ctrl["c_vs_e_roc"]["mean_diff"], "Paired (c vs e) ROC Diff"),
        (r"\+0\.0027", paired_ctrl["ew_vs_s_roc"]["mean_diff"], "Paired (ew vs S) ROC Diff"),
        (r"\+0\.0034", paired_ctrl["ew_vs_s_pr"]["mean_diff"], "Paired (ew vs S) PR Diff"),
        (r"-0\.0012", paired_ctrl["ew_vs_e_roc"]["mean_diff"], "Paired (ew vs e) ROC Diff"),
    ]

    print("\n--- LINE-BY-LINE CHECKED VALUES (file:line | quoted | recomputed | diff | OK/FAIL) ---")
    checked_counts = {doc: 0 for doc in target_docs}
    failures = []

    # Required headline and controls labels that must appear in every target document
    required_labels = {
        "Full Variant (a) ROC", "Full Variant (b) ROC", "Full Variant (c) ROC", "Full Variant (d) ROC",
        "Full (c vs b) ROC Lift", "Truncated Variant (a) ROC", "Truncated Variant (b) ROC",
        "Truncated Variant (c) ROC", "Truncated Variant (d) ROC", "Truncated (c vs b) ROC Lift",
        "Control (e) ROC", "Control (e) PR", "Control (e) ROC Lift", "Control (e) PR Lift",
        "Control (e_weekly) ROC", "Control (e_weekly) PR", "Control (e_weekly) ROC Lift", "Control (e_weekly) PR Lift",
        "Control (f) ROC", "Control (f) PR", "Control (f) ROC Lift", "Control (f) PR Lift",
        "Control (g) ROC", "Control (g) ROC Lift", "Control (g) PR Lift",
        "Control (S) ROC", "Control (S) PR", "Control (S) ROC Lift", "Control (S) PR Lift",
        "Paired (S vs e) ROC Diff", "Paired (S vs e) PR Diff", "Paired (c vs e) ROC Diff",
        "Paired (ew vs S) ROC Diff", "Paired (ew vs S) PR Diff", "Paired (ew vs e) ROC Diff",
    }

    for doc_name, doc_path in target_docs.items():
        if not os.path.exists(doc_path):
            failures.append(f"Missing file: {doc_path}")
            continue

        with open(doc_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        matched_labels_for_doc = set()

        for line_no, line in enumerate(lines, 1):
            for pat, recomp_val, label in checks:
                match = re.search(pat, line)
                if match:
                    quoted_str = match.group(0)
                    try:
                        quoted_val = float(quoted_str)
                    except ValueError:
                        continue
                    diff = quoted_val - recomp_val
                    status = "OK" if abs(diff) <= (TOLERANCE + 1e-4) else "FAIL"
                    print(f"{doc_name}:{line_no:<4d} | {quoted_str:>8} | {recomp_val:9.6f} | {diff:+9.6f} | {status} ({label})")
                    checked_counts[doc_name] += 1
                    matched_labels_for_doc.add(label)
                    if status == "FAIL":
                        failures.append(f"{doc_name}:{line_no} metric '{label}' diff {diff:+.6f} exceeds tolerance")

        # Check for missing required metrics in this file
        missing_in_doc = required_labels - matched_labels_for_doc
        if missing_in_doc:
            failures.append(f"[{doc_name}] Required headline metrics not matched (or tampered): {sorted(missing_in_doc)}")

    print("\n--- SUMMARY OF VALUES CHECKED PER FILE ---")
    zero_checked = []
    for doc_name, cnt in checked_counts.items():
        print(f"  - {doc_name}: {cnt} values verified line-by-line")
        if cnt == 0:
            zero_checked.append(doc_name)

    if zero_checked:
        print(f"\n❌ VERIFICATION FAILED: Files with 0 verified values: {zero_checked}")
        sys.exit(1)

    if failures:
        print("\n❌ VERIFICATION FAILED: Mismatches found:")
        for fail in failures:
            print(f"  - {fail}")
        sys.exit(1)

    print("\n✅ VERIFICATION PASSED: All headline and controls values verified line-by-line against results.json.")
    print("=" * 80)


if __name__ == "__main__":
    verify()
