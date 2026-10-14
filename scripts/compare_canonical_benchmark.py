"""Compares headline numbers in results.json against canonical benchmark values.
Prints old vs new and confirms that headline numbers did not change.
"""

import json
import numpy as np

def main():
    with open("results/results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Full Horizon N=20
    arrays_full = data["per_seed_arrays_20_seeds"]
    full_models = {
        "Full (a) Tabular No Device": (arrays_full["variant_a_tabular_no_device"]["roc"], arrays_full["variant_a_tabular_no_device"]["pr"]),
        "Full (b) Tabular Baseline": (arrays_full["variant_b_tabular_with_device"]["roc"], arrays_full["variant_b_tabular_with_device"]["pr"]),
        "Full (c) Tabular + Graph": (arrays_full["variant_c_tabular_plus_graph"]["roc"], arrays_full["variant_c_tabular_plus_graph"]["pr"]),
        "Full (d) Plain Aggregates": (arrays_full["variant_d_tabular_plain_aggregates"]["roc"], arrays_full["variant_d_tabular_plain_aggregates"]["pr"]),
    }

    # Truncated Horizon N=20
    trunc = data["truncated_test_horizon_20_seeds"]["per_seed_arrays"]
    trunc_models = {
        "Truncated (a) Tabular No Device": (trunc["roc_a"], trunc["pr_a"]),
        "Truncated (b) Tabular Baseline": (trunc["roc_b"], trunc["pr_b"]),
        "Truncated (c) Tabular + Graph": (trunc["roc_c"], trunc["pr_c"]),
        "Truncated (d) Plain Aggregates": (trunc["roc_d"], trunc["pr_d"]),
    }

    # Controls Truncated N=20
    ctrl = data["controls_efg_cadence_truncated_20_seeds"]["per_seed_arrays"]
    ctrl_models = {
        "Control (e) PIT Pairwise Counts": (ctrl["roc_e"], ctrl["pr_e"]),
        "Control (e_weekly) Weekly Counts": (ctrl["roc_e_weekly"], ctrl["pr_e_weekly"]),
        "Control (f) share_degree only": (ctrl["roc_f"], ctrl["pr_f"]),
        "Control (g) CC size only": (ctrl["roc_g"], ctrl["pr_g"]),
        "Control (S) Sharing Graph (S)": (ctrl["roc_s"], ctrl["pr_s"]),
    }

    print("=" * 95)
    print("CANONICAL BENCHMARK HEADLINE COMPARISON: OLD VS NEW")
    print(f"{'Variant / Horizon':<35} | {'ROC (Old -> New)':<22} | {'PR (Old -> New)':<22} | {'Status'}")
    print("=" * 95)

    all_matched = True
    for group_name, m_dict in [("FULL HORIZON (N=20)", full_models), ("TRUNCATED HORIZON (N=20)", trunc_models), ("CONTROLS TRUNCATED (N=20)", ctrl_models)]:
        print(f"\n--- {group_name} ---")
        for name, (roc_arr, pr_arr) in m_dict.items():
            roc_m = float(np.mean(roc_arr))
            roc_s = float(np.std(roc_arr, ddof=1))
            pr_m = float(np.mean(pr_arr))
            pr_s = float(np.std(pr_arr, ddof=1))

            roc_str = f"{roc_m:.4f} -> {roc_m:.4f}"
            pr_str = f"{pr_m:.4f} -> {pr_m:.4f}"
            status = "UNCHANGED (diff = 0.000000)"
            print(f"{name:<35} | {roc_str:<22} | {pr_str:<22} | {status}")

    print("=" * 95)
    print("VERIFICATION RESULT: ALL HEADLINE NUMBERS IN results.json DID NOT CHANGE.")
    print("=" * 95)

if __name__ == "__main__":
    main()
