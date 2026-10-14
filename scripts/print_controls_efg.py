import json

with open('results/controls_efg_results.json', 'r') as f:
    d = json.load(f)

print("=== CONTROLS (e), (f), (g) MODELS SUMMARY ===")
for k, v in d['models_summary'].items():
    print(f"{k}: ROC mean={v['roc_mean']:.6f} +/- {v['roc_std']:.6f} | PR mean={v['pr_mean']:.6f} +/- {v['pr_std']:.6f}")

print("\n=== PAIRED COMPARISONS ===")
for comp_name, comp in d['paired_comparisons'].items():
    print(f"{comp_name}:")
    print(f"  ROC: diff={comp['roc']['mean_diff']:+.6f}, 95% CI=[{comp['roc']['ci_95'][0]:+.6f}, {comp['roc']['ci_95'][1]:+.6f}], t={comp['roc']['t_stat']:+.4f}, p={comp['roc']['p_value']:.6e}")
    print(f"  PR:  diff={comp['pr']['mean_diff']:+.6f}, 95% CI=[{comp['pr']['ci_95'][0]:+.6f}, {comp['pr']['ci_95'][1]:+.6f}], t={comp['pr']['t_stat']:+.4f}, p={comp['pr']['p_value']:.6e}")
