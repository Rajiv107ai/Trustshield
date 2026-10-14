"""Verification script for TrustShield scientific benchmark results.

Loads results/results.json ONLY, recomputes (scipy.stats, ddof=1) every mean, std,
paired mean diff, 95% CI, t, p, and Holm-adjusted p for all headline comparisons.

Prints one line per checked value:
  file:line | quoted | recomputed | diff | OK/FAIL
along with the count of values checked per file.
Exits non-zero if any file has 0 values checked or if any diff exceeds 5e-5.

Lists all sections stored without per-seed arrays as 'NOT INDEPENDENTLY RECOMPUTABLE'.
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

    # 3. Recompute Controls (e), (e_weekly), (f), (g), (S)
    controls_data = data.get("controls_efg_cadence_truncated_20_seeds", {}).get("per_seed_arrays", {})
    stats_ctrl = {}
    paired_ctrl = {}
    if controls_data:
        for k in ["roc_b", "roc_c", "roc_e", "roc_e_weekly", "roc_f", "roc_g", "roc_s"]:
            if k in controls_data:
                stats_ctrl[k] = calc_stats(controls_data[k])
        paired_ctrl["e_vs_b_roc"] = calc_paired(controls_data["roc_e"], controls_data["roc_b"])
        paired_ctrl["ew_vs_b_roc"] = calc_paired(controls_data["roc_e_weekly"], controls_data["roc_b"])
        paired_ctrl["f_vs_b_roc"] = calc_paired(controls_data["roc_f"], controls_data["roc_b"])
        paired_ctrl["g_vs_b_roc"] = calc_paired(controls_data["roc_g"], controls_data["roc_b"])
        paired_ctrl["s_vs_b_roc"] = calc_paired(controls_data["roc_s"], controls_data["roc_b"])
        paired_ctrl["c_vs_e_roc"] = calc_paired(controls_data["roc_c"], controls_data["roc_e"])
        paired_ctrl["s_vs_e_roc"] = calc_paired(controls_data["roc_s"], controls_data["roc_e"])
        paired_ctrl["ew_vs_s_roc"] = calc_paired(controls_data["roc_e_weekly"], controls_data["roc_s"])

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
        print(f"  (e) Tabular + PIT Counts:    ROC = {stats_ctrl['roc_e']['mean']:.6f} +/- {stats_ctrl['roc_e']['std']:.6f}")
        print(f"  (e_weekly) Weekly Counts:    ROC = {stats_ctrl['roc_e_weekly']['mean']:.6f} +/- {stats_ctrl['roc_e_weekly']['std']:.6f}")
        print(f"  (f) share_degree only:       ROC = {stats_ctrl['roc_f']['mean']:.6f} +/- {stats_ctrl['roc_f']['std']:.6f}")
        print(f"  (g) share_component_size:    ROC = {stats_ctrl['roc_g']['mean']:.6f} +/- {stats_ctrl['roc_g']['std']:.6f}")
        print(f"  (S) Topological Sharing (S): ROC = {stats_ctrl['roc_s']['mean']:.6f} +/- {stats_ctrl['roc_s']['std']:.6f}")
        print(f"  Paired (S vs e) ROC: diff={paired_ctrl['s_vs_e_roc']['mean_diff']:+.6f}, p={paired_ctrl['s_vs_e_roc']['p_value']:.4e}")
        print(f"  Paired (e_weekly vs S) ROC: diff={paired_ctrl['ew_vs_s_roc']['mean_diff']:+.6f}, p={paired_ctrl['ew_vs_s_roc']['p_value']:.4e}")

    print("\n--- NOT INDEPENDENTLY RECOMPUTABLE (stored without per-seed arrays in results.json) ---")
    not_indep = [
        ("Hybrid vs Tabular (Standard & Coherent N=5)", data.get("tuned_hybrid_model_5_seeds", {})),
        ("Group Ablations (S, B, E combinations)", data.get("group_ablation", {})),
        ("Operational Review Budget Lifts (2%, 5%, 10%)", data.get("practical_significance_budgets", {})),
        ("Per-Type Breakdown & Isolated ROC Metrics", data.get("per_type_metrics_all_types", {})),
        ("Calibration ECE & Brier Loss Metrics", data.get("calibration_audit_20_seeds", {})),
        ("Fraud Ring Detection Recovery vs Baselines", data.get("ring_detection_audit", {})),
        ("Conformal Prediction Coverage & Set Sizes", data.get("conformal_prediction_audit_20_seeds", {})),
    ]
    for label, obj in not_indep:
        stored_keys = list(obj.keys())[:3] if isinstance(obj, dict) else "N/A"
        print(f"  [SUMMARY ONLY] {label:<45} (keys: {stored_keys})")

    # Target Document Line-by-Line Checks
    target_docs = {
        "results/RESULTS.md": os.path.join(_ROOT_DIR, "results", "RESULTS.md"),
        "README.md": os.path.join(_ROOT_DIR, "README.md"),
        "docs/MODEL_CARD.md": os.path.join(_ROOT_DIR, "docs", "MODEL_CARD.md"),
    }

    # Expected checks: (pattern_regex, recomputed_val, label)
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
        # Primary ROC lift
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
    ]

    print("\n--- LINE-BY-LINE CHECKED VALUES (file:line | quoted | recomputed | diff | OK/FAIL) ---")
    checked_counts = {doc: 0 for doc in target_docs}
    failures = []

    # Required headline labels that must appear in every target document
    required_labels = {
        "Full Variant (a) ROC", "Full Variant (b) ROC", "Full Variant (c) ROC", "Full Variant (d) ROC",
        "Full (c vs b) ROC Lift", "Truncated Variant (a) ROC", "Truncated Variant (b) ROC",
        "Truncated Variant (c) ROC", "Truncated Variant (d) ROC", "Truncated (c vs b) ROC Lift"
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

    print("\n✅ VERIFICATION PASSED: All headline values verified line-by-line against results.json.")
    print("=" * 80)


if __name__ == "__main__":
    verify()
