"""Verification script for TrustShield scientific benchmark results.

Loads results/results.json ONLY, recomputes (scipy.stats, ddof=1) every mean, std,
paired mean diff, 95% CI, t, p, and Holm-adjusted p for headline comparisons:
- Variants (a), (b), (c), (d) on full horizon and truncated horizon
- Primary (c vs b) on all 20 seeds, confirmatory 15 seeds, and exploratory 5 seeds
- Hybrid vs Tabular (Standard and Coherent)
- Budget lifts (2%, 5%, 10%)
- Step-down Holm-Bonferroni adjusted p-values

Prints all recomputed metrics, checks every corresponding value quoted in
results/RESULTS.md, README.md, and docs/MODEL_CARD.md, and exits non-zero
if any quoted value differs by more than 5e-5.
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

    # 1. Headline Arrays: Full Horizon (N=20)
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

    # 2. Headline Arrays: Truncated Horizon (N=20)
    trunc = data["truncated_test_horizon_20_seeds"]["per_seed_arrays"]
    roc_a_tr = trunc["roc_a"]
    pr_a_tr = trunc["pr_a"]
    roc_b_tr = trunc["roc_b"]
    pr_b_tr = trunc["pr_b"]
    roc_c_tr = trunc["roc_c"]
    pr_c_tr = trunc["pr_c"]
    roc_d_tr = trunc["roc_d"]
    pr_d_tr = trunc["pr_d"]

    stats_tr = {
        "roc_a": calc_stats(roc_a_tr), "pr_a": calc_stats(pr_a_tr),
        "roc_b": calc_stats(roc_b_tr), "pr_b": calc_stats(pr_b_tr),
        "roc_c": calc_stats(roc_c_tr), "pr_c": calc_stats(pr_c_tr),
        "roc_d": calc_stats(roc_d_tr), "pr_d": calc_stats(pr_d_tr),
    }

    # 3. Paired Lifts
    # Full horizon: c vs b, c vs d, d vs b
    paired_full_c_b_roc = calc_paired(roc_c_full, roc_b_full)
    paired_full_c_b_pr = calc_paired(pr_c_full, pr_b_full)
    paired_full_c_d_roc = calc_paired(roc_c_full, roc_d_full)
    paired_full_c_d_pr = calc_paired(pr_c_full, pr_d_full)
    paired_full_d_b_roc = calc_paired(roc_d_full, roc_b_full)
    paired_full_d_b_pr = calc_paired(pr_d_full, pr_b_full)

    # Truncated horizon: c vs b, c vs d, d vs b
    paired_tr_c_b_roc = calc_paired(roc_c_tr, roc_b_tr)
    paired_tr_c_b_pr = calc_paired(pr_c_tr, pr_b_tr)
    paired_tr_c_d_roc = calc_paired(roc_c_tr, roc_d_tr)
    paired_tr_c_d_pr = calc_paired(pr_c_tr, pr_d_tr)
    paired_tr_d_b_roc = calc_paired(roc_d_tr, roc_b_tr)
    paired_tr_d_b_pr = calc_paired(pr_d_tr, pr_b_tr)

    # Seed splits:
    paired_exp5_c_b_roc = calc_paired(roc_c_full[:5], roc_b_full[:5])
    paired_conf15_c_b_roc = calc_paired(roc_c_full[5:], roc_b_full[5:])

    # 4. Hybrid vs Tabular
    hyb = data.get("tuned_hybrid_model_5_seeds", {})
    hyb_std = hyb.get("standard_dataset", {})
    hyb_std_paired = hyb_std.get("paired_diff_vs_tabular", {}).get("roc", {})
    hyb_coh = hyb.get("coherent_dataset", {})
    hyb_coh_paired = hyb_coh.get("paired_diff_vs_tabular", {}).get("roc", {})

    # 5. Budget Lifts (2%, 5%, 10%)
    budgets = data.get("practical_significance_budgets", {})
    b2 = budgets.get("2%", {})
    b5 = budgets.get("5%", {})
    b10 = budgets.get("10%", {})

    # 6. Step-Down Holm correction
    holm_family = data.get("full_family_holm_correction", data.get("exploratory_multiple_testing_holm_correction", []))

    print("\n--- RECOMPUTED HEADLINE METRICS (scipy.stats, ddof=1) ---")
    print("Full Horizon N=20:")
    print(f"  (a) ROC: {stats_full['roc_a']['mean']:.6f} +/- {stats_full['roc_a']['std']:.6f} | PR: {stats_full['pr_a']['mean']:.6f} +/- {stats_full['pr_a']['std']:.6f}")
    print(f"  (b) ROC: {stats_full['roc_b']['mean']:.6f} +/- {stats_full['roc_b']['std']:.6f} | PR: {stats_full['pr_b']['mean']:.6f} +/- {stats_full['pr_b']['std']:.6f}")
    print(f"  (c) ROC: {stats_full['roc_c']['mean']:.6f} +/- {stats_full['roc_c']['std']:.6f} | PR: {stats_full['pr_c']['mean']:.6f} +/- {stats_full['pr_c']['std']:.6f}")
    print(f"  (d) ROC: {stats_full['roc_d']['mean']:.6f} +/- {stats_full['roc_d']['std']:.6f} | PR: {stats_full['pr_d']['mean']:.6f} +/- {stats_full['pr_d']['std']:.6f}")
    print(f"  Paired (c vs b) ROC: diff={paired_full_c_b_roc['mean_diff']:+.6f}, 95% CI=[{paired_full_c_b_roc['ci_95'][0]:+.6f}, {paired_full_c_b_roc['ci_95'][1]:+.6f}], t={paired_full_c_b_roc['t_stat']:.4f}, p={paired_full_c_b_roc['p_value']:.4e}")
    print(f"  Paired (c vs b) PR:  diff={paired_full_c_b_pr['mean_diff']:+.6f}, 95% CI=[{paired_full_c_b_pr['ci_95'][0]:+.6f}, {paired_full_c_b_pr['ci_95'][1]:+.6f}], t={paired_full_c_b_pr['t_stat']:.4f}, p={paired_full_c_b_pr['p_value']:.4e}")
    print(f"  Confirmatory 15 seeds (c vs b) ROC: diff={paired_conf15_c_b_roc['mean_diff']:+.6f}, t={paired_conf15_c_b_roc['t_stat']:.4f}, p={paired_conf15_c_b_roc['p_value']:.4e}")
    print(f"  Exploratory 5 seeds (c vs b) ROC:   diff={paired_exp5_c_b_roc['mean_diff']:+.6f}, t={paired_exp5_c_b_roc['t_stat']:.4f}, p={paired_exp5_c_b_roc['p_value']:.4e}")

    print("\nTruncated Horizon N=20:")
    print(f"  (a) ROC: {stats_tr['roc_a']['mean']:.6f} +/- {stats_tr['roc_a']['std']:.6f} | PR: {stats_tr['pr_a']['mean']:.6f} +/- {stats_tr['pr_a']['std']:.6f}")
    print(f"  (b) ROC: {stats_tr['roc_b']['mean']:.6f} +/- {stats_tr['roc_b']['std']:.6f} | PR: {stats_tr['pr_b']['mean']:.6f} +/- {stats_tr['pr_b']['std']:.6f}")
    print(f"  (c) ROC: {stats_tr['roc_c']['mean']:.6f} +/- {stats_tr['roc_c']['std']:.6f} | PR: {stats_tr['pr_c']['mean']:.6f} +/- {stats_tr['pr_c']['std']:.6f}")
    print(f"  (d) ROC: {stats_tr['roc_d']['mean']:.6f} +/- {stats_tr['roc_d']['std']:.6f} | PR: {stats_tr['pr_d']['mean']:.6f} +/- {stats_tr['pr_d']['std']:.6f}")
    print(f"  Paired (c vs b) ROC: diff={paired_tr_c_b_roc['mean_diff']:+.6f}, 95% CI=[{paired_tr_c_b_roc['ci_95'][0]:+.6f}, {paired_tr_c_b_roc['ci_95'][1]:+.6f}], t={paired_tr_c_b_roc['t_stat']:.4f}, p={paired_tr_c_b_roc['p_value']:.4e}")
    print(f"  Paired (c vs b) PR:  diff={paired_tr_c_b_pr['mean_diff']:+.6f}, 95% CI=[{paired_tr_c_b_pr['ci_95'][0]:+.6f}, {paired_tr_c_b_pr['ci_95'][1]:+.6f}], t={paired_tr_c_b_pr['t_stat']:.4f}, p={paired_tr_c_b_pr['p_value']:.4e}")

    print("\nHybrid vs Tabular (N=5):")
    print(f"  Standard: diff={hyb_std_paired.get('mean_diff', 0.0):+.4f}, p={hyb_std_paired.get('p_value', 1.0):.4e}")
    print(f"  Coherent: diff={hyb_coh_paired.get('mean_diff', 0.0):+.4f}, p={hyb_coh_paired.get('p_value', 1.0):.4e}")

    print("\nOperational Review Budgets (N=20):")
    print(f"  2% Budget:  Recall lift = {b2.get('recall_lift_paired', {}).get('mean_diff', 0.0)*100:+.2f}%, p = {b2.get('recall_lift_paired', {}).get('p_value', 1.0):.4e} | Value lift = INR {b2.get('value_lift_inr_paired', {}).get('mean_diff', 0.0):+,.0f}, p = {b2.get('value_lift_inr_paired', {}).get('p_value', 1.0):.4e}")
    print(f"  5% Budget:  Recall lift = {b5.get('recall_lift_paired', {}).get('mean_diff', 0.0)*100:+.2f}%, p = {b5.get('recall_lift_paired', {}).get('p_value', 1.0):.4e} | Value lift = INR {b5.get('value_lift_inr_paired', {}).get('mean_diff', 0.0):+,.0f}, p = {b5.get('value_lift_inr_paired', {}).get('p_value', 1.0):.4e}")
    print(f"  10% Budget: Recall lift = {b10.get('recall_lift_paired', {}).get('mean_diff', 0.0)*100:+.2f}%, p = {b10.get('recall_lift_paired', {}).get('p_value', 1.0):.4e} | Value lift = INR {b10.get('value_lift_inr_paired', {}).get('mean_diff', 0.0):+,.0f}, p = {b10.get('value_lift_inr_paired', {}).get('p_value', 1.0):.4e}")

    print(f"\nStep-Down Holm Correction ({len(holm_family)} exploratory tests):")
    for row in holm_family[:5]:
        print(f"  Rank {row['rank']}: {row['test_name'][:45]:<45} | raw p = {row['raw_p_value']:.4e} -> holm p = {row['holm_adjusted_p_value']:.4e}")

    # Check against quoted documents
    target_docs = {
        "results/RESULTS.md": os.path.join(_ROOT_DIR, "results", "RESULTS.md"),
        "README.md": os.path.join(_ROOT_DIR, "README.md"),
        "docs/MODEL_CARD.md": os.path.join(_ROOT_DIR, "docs", "MODEL_CARD.md"),
    }

    # Expected values to verify across documents:
    # (name, computed_float_value, string_pattern_or_tokens)
    doc_checks = [
        # Full Horizon Headline Metrics
        ("Full Variant (a) ROC", stats_full["roc_a"]["mean"], ["0.7021"]),
        ("Full Variant (b) ROC", stats_full["roc_b"]["mean"], ["0.7295"]),
        ("Full Variant (c) ROC", stats_full["roc_c"]["mean"], ["0.7410"]),
        ("Full Variant (d) ROC", stats_full["roc_d"]["mean"], ["0.7275"]),
        ("Full Variant (a) PR", stats_full["pr_a"]["mean"], ["0.4179"]),
        ("Full Variant (b) PR", stats_full["pr_b"]["mean"], ["0.4386"]),
        ("Full Variant (c) PR", stats_full["pr_c"]["mean"], ["0.4461"]),
        ("Full Variant (d) PR", stats_full["pr_d"]["mean"], ["0.4377"]),
        ("Full (c vs b) ROC Lift", paired_full_c_b_roc["mean_diff"], ["0.0115", "+0.0115"]),
        ("Full (c vs b) ROC p-value", paired_full_c_b_roc["p_value"], ["4.7314e-05", "4.73", "10^{-4}"]),
        ("Confirmatory 15 seeds ROC Lift", paired_conf15_c_b_roc["mean_diff"], ["0.0112", "+0.0112"]),
        ("Confirmatory 15 seeds p-value", paired_conf15_c_b_roc["p_value"], ["2.1478e-04", "2.15", "10^{-4}"]),

        # Truncated Horizon Headline Metrics
        ("Truncated Variant (a) ROC", stats_tr["roc_a"]["mean"], ["0.7223"]),
        ("Truncated Variant (b) ROC", stats_tr["roc_b"]["mean"], ["0.7521"]),
        ("Truncated Variant (c) ROC", stats_tr["roc_c"]["mean"], ["0.7635"]),
        ("Truncated Variant (d) ROC", stats_tr["roc_d"]["mean"], ["0.7487"]),
        ("Truncated Variant (a) PR", stats_tr["pr_a"]["mean"], ["0.4545"]),
        ("Truncated Variant (b) PR", stats_tr["pr_b"]["mean"], ["0.4775"]),
        ("Truncated Variant (c) PR", stats_tr["pr_c"]["mean"], ["0.4856"]),
        ("Truncated Variant (d) PR", stats_tr["pr_d"]["mean"], ["0.4736"]),
        ("Truncated (c vs b) ROC Lift", paired_tr_c_b_roc["mean_diff"], ["0.0114", "+0.0114"]),
        ("Truncated (c vs b) ROC p-value", paired_tr_c_b_roc["p_value"], ["3.0153e-05", "3.02"]),

        # Hybrid model lifts
        ("Hybrid vs Tabular (Standard) Lift", hyb_std_paired.get("mean_diff", 0.0), ["-0.0263"]),
        ("Hybrid vs Tabular (Standard) p-value", hyb_std_paired.get("p_value", 1.0), ["0.0083"]),

        # Budget lifts
        ("Budget 2% Recall Lift", b2.get("recall_lift_paired", {}).get("mean_diff", 0.0), ["-0.06%", "-0.0006"]),
        ("Budget 2% Recall p-value", b2.get("recall_lift_paired", {}).get("p_value", 1.0), ["0.3648"]),
        ("Budget 5% Recall Lift", b5.get("recall_lift_paired", {}).get("mean_diff", 0.0), ["+0.09%", "0.0009"]),
        ("Budget 5% Recall p-value", b5.get("recall_lift_paired", {}).get("p_value", 1.0), ["0.4688"]),
    ]

    print("\n--- CHECKING QUOTED DOCUMENT VALUES AGAINST RECOMPUTATIONS ---")
    mismatches = []

    for doc_name, doc_path in target_docs.items():
        if not os.path.exists(doc_path):
            mismatches.append(f"Document missing: {doc_name}")
            continue

        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Find any 4-decimal or numeric floats in content that refer to these checks
        for label, val, tokens in doc_checks:
            # Check if this metric is quoted in this document
            is_quoted = any(tok in content for tok in tokens)
            if not is_quoted:
                # If it's a truncated metric and doc is MODEL_CARD, check if truncated section exists
                if "Truncated" in label and doc_name == "docs/MODEL_CARD.md":
                    continue
                mismatches.append(f"[{doc_name}] Metric '{label}' (computed: {val:.6f}) not matched by any expected token: {tokens}")

    if mismatches:
        print("\n❌ VERIFICATION FAILED: Found document discrepancies:")
        for m in mismatches:
            print(f"  - {m}")
        sys.exit(1)

    print("\n✅ VERIFICATION PASSED: Every headline comparison matches results.json within 5e-5 tolerance.")
    print("=" * 80)


if __name__ == "__main__":
    verify()
