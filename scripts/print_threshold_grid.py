import json

with open('results/threshold_grid_results.json', 'r') as f:
    d = json.load(f)

taus = d['taus']

print("=== SEED 42 THRESHOLD GRID ===")
s42 = d['seed_42']
print(f"{'tau':>6} | {'Raw Vol%':>10} | {'Raw Rec%':>10} | {'Cal Vol%':>10} | {'Cal Rec%':>10}")
print("-" * 56)
for tau in taus:
    k = str(tau)
    r_vol = s42['raw_volume'][k] * 100
    r_rec = s42['raw_recall'][k] * 100
    c_vol = s42['calibrated_volume'][k] * 100
    c_rec = s42['calibrated_recall'][k] * 100
    print(f"{tau:6.3f} | {r_vol:9.2f}% | {r_rec:9.2f}% | {c_vol:9.2f}% | {c_rec:9.2f}%")

print("\n=== 20-SEED MEAN THRESHOLD GRID ===")
mean_grid = d['mean_20_seeds']
print(f"{'tau':>6} | {'Raw Vol% (mean +/- std)':>25} | {'Raw Rec% (mean +/- std)':>25} | {'Cal Vol% (mean +/- std)':>25} | {'Cal Rec% (mean +/- std)':>25}")
print("-" * 115)
for tau in taus:
    k = str(tau)
    r_vm, r_vs = mean_grid['raw_volume'][k]['mean'] * 100, mean_grid['raw_volume'][k]['std'] * 100
    r_rm, r_rs = mean_grid['raw_recall'][k]['mean'] * 100, mean_grid['raw_recall'][k]['std'] * 100
    c_vm, c_vs = mean_grid['calibrated_volume'][k]['mean'] * 100, mean_grid['calibrated_volume'][k]['std'] * 100
    c_rm, c_rs = mean_grid['calibrated_recall'][k]['mean'] * 100, mean_grid['calibrated_recall'][k]['std'] * 100
    print(f"{tau:6.3f} | {r_vm:6.2f}% +/- {r_vs:5.2f}% | {r_rm:6.2f}% +/- {r_rs:5.2f}% | {c_vm:6.2f}% +/- {c_vs:5.2f}% | {c_rm:6.2f}% +/- {c_rs:5.2f}%")
