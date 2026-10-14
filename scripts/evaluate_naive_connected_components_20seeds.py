"""Evaluates naive connected components (size >= 2) across 20 canonical seeds.

Exact Definition:
1. Multi-relational sharing graph is constructed from base['address_sharing_log'] and
   base['device_sharing_log'] with edges restricted to events occurring strictly before
   the validation cutoff (order_date <= VAL_END: 2025-10-31), preventing future temporal leakage.
2. The graph is converted to an undirected graph.
3. Connected components of size >= 2 are identified.
4. Member-level evaluation:
   - Predicted ring members = union of all buyer nodes in components of size >= 2.
   - Ground-truth ring members = unique buyer entity IDs in res['fraud_ground_truth'] (entity_type == 'buyer').
   - Member Precision = |Predicted & GT| / |Predicted|
   - Member Recall = |Predicted & GT| / |GT|
   - Member F1 = 2 * Precision * Recall / (Precision + Recall)
5. Ring-level evaluation:
   - For each ground-truth ring cluster in fraud_ground_truth:
     A ground-truth ring is considered recovered (hit) if at least 50% of its member nodes
     are contained in the predicted member set.
   - Ring Recall = number of recovered ground-truth rings / total ground-truth rings.
   - For each predicted component of size >= 2:
     A predicted ring is considered valid if at least 50% of its members belong to ground-truth rings.
   - Ring Precision = number of valid predicted rings / total predicted rings.
   - Ring F1 = 2 * Ring Precision * Ring Recall / (Ring Precision + Ring Recall)
"""

from __future__ import annotations
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import networkx as nx

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
_PROJECT_DIR = os.path.join(_ROOT_DIR, "trustshield_project")
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from entity_generator import generate_full_pipeline
from graph_features import build_relationship_graph
from baseline_model import VAL_END

SEEDS_20 = [
    42, 101, 202, 303, 404, 505, 606, 707, 808, 909,
    1001, 1102, 1203, 1304, 1405, 1506, 1607, 1708, 1809, 1910
]


def evaluate_ring_recovery_20_seeds():
    print("=" * 80)
    print("NAIVE CONNECTED COMPONENTS (SIZE >= 2) EVALUATION ACROSS 20 CANONICAL SEEDS")
    print(f"Script: scripts/evaluate_naive_connected_components_20seeds.py")
    print(f"Graph Cutoff: VAL_END ({VAL_END.strftime('%Y-%m-%d')})")
    print("=" * 80)

    mem_p_list, mem_r_list, mem_f1_list = [], [], []
    ring_p_list, ring_r_list, ring_f1_list = [], [], []

    per_seed_data = []

    for idx, seed in enumerate(SEEDS_20, 1):
        t0 = time.perf_counter()
        pipe = generate_full_pipeline(seed=seed, ring_coherent=False)
        res = pipe["result"]
        base = pipe["base"]

        # Build graph with cutoff
        rel_graph = build_relationship_graph(
            base["address_sharing_log"],
            base["device_sharing_log"],
            cutoff_date=VAL_END
        )
        undirected = rel_graph.to_undirected()
        all_ccs = [set(c) for c in nx.connected_components(undirected) if len(c) >= 2]

        gt_df = res["fraud_ground_truth"]
        gt_buyers_df = gt_df[gt_df["entity_type"] == "buyer"] if "entity_type" in gt_df.columns else gt_df
        gt_id_col = "entity_id" if "entity_id" in gt_buyers_df.columns else "buyer_id"
        gt_buyers = set(gt_buyers_df[gt_id_col].unique())

        gt_rings = {}
        if "fraud_ring_id" in gt_buyers_df.columns:
            for rid, grp in gt_buyers_df.groupby("fraud_ring_id"):
                gt_rings[rid] = set(grp[gt_id_col].unique())

        # Predicted members
        detected_members = set().union(*all_ccs) if all_ccs else set()

        # Member-level
        p_mem = len(detected_members & gt_buyers) / max(len(detected_members), 1)
        r_mem = len(detected_members & gt_buyers) / max(len(gt_buyers), 1)
        f1_mem = 2 * p_mem * r_mem / max(p_mem + r_mem, 1e-9)

        # Ring-level
        if gt_rings:
            # Ring Recall: % of GT rings having >= 50% members detected
            rec_rings = sum(1 for r_nodes in gt_rings.values() if len(r_nodes & detected_members) / len(r_nodes) >= 0.5) / len(gt_rings)
        else:
            rec_rings = 0.0

        if all_ccs:
            # Ring Precision: % of predicted components having >= 50% members in GT
            prec_rings = sum(1 for c_nodes in all_ccs if len(c_nodes & gt_buyers) / len(c_nodes) >= 0.5) / len(all_ccs)
        else:
            prec_rings = 0.0

        f1_ring = 2 * prec_rings * rec_rings / max(prec_rings + rec_rings, 1e-9)

        mem_p_list.append(p_mem)
        mem_r_list.append(r_mem)
        mem_f1_list.append(f1_mem)
        ring_p_list.append(prec_rings)
        ring_r_list.append(rec_rings)
        ring_f1_list.append(f1_ring)

        dt = time.perf_counter() - t0
        print(f"[{idx:02d}/20] Seed {seed:4d} ({dt:.1f}s) | Mem P: {p_mem*100:5.2f}% R: {r_mem*100:5.2f}% F1: {f1_mem:.4f} | Ring P: {prec_rings*100:5.2f}% R: {rec_rings*100:5.2f}% F1: {f1_ring:.4f}")

        per_seed_data.append({
            "seed": seed,
            "member_precision": p_mem,
            "member_recall": r_mem,
            "member_f1": f1_mem,
            "ring_precision": prec_rings,
            "ring_recall": rec_rings,
            "ring_f1": f1_ring,
        })

    print("-" * 80)
    print("CANONICAL 20-SEED SUMMARY (ddof=1):")
    print(f"Member-Level Precision: {np.mean(mem_p_list)*100:6.2f}% +/- {np.std(mem_p_list, ddof=1)*100:5.2f}% (raw: {np.mean(mem_p_list):.6f})")
    print(f"Member-Level Recall:    {np.mean(mem_r_list)*100:6.2f}% +/- {np.std(mem_r_list, ddof=1)*100:5.2f}% (raw: {np.mean(mem_r_list):.6f})")
    print(f"Member-Level F1 Score:  {np.mean(mem_f1_list):8.4f} +/- {np.std(mem_f1_list, ddof=1):7.4f} (raw: {np.mean(mem_f1_list):.6f})")
    print(f"Ring-Level Precision:   {np.mean(ring_p_list)*100:6.2f}% +/- {np.std(ring_p_list, ddof=1)*100:5.2f}% (raw: {np.mean(ring_p_list):.6f})")
    print(f"Ring-Level Recall:      {np.mean(ring_r_list)*100:6.2f}% +/- {np.std(ring_r_list, ddof=1)*100:5.2f}% (raw: {np.mean(ring_r_list):.6f})")
    print(f"Ring-Level F1 Score:    {np.mean(ring_f1_list):8.4f} +/- {np.std(ring_f1_list, ddof=1):7.4f} (raw: {np.mean(ring_f1_list):.6f})")
    print("=" * 80)

    # Save to json artifact
    out_path = os.path.join(_ROOT_DIR, "results", "ring_detection_20seeds.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "script": "scripts/evaluate_naive_connected_components_20seeds.py",
            "definition": {
                "graph": "base['address_sharing_log'] + base['device_sharing_log'] with event_date <= VAL_END",
                "naive_baseline": "connected_components(size >= 2)",
                "ring_recall_criterion": ">= 50% members detected in ground-truth ring cluster",
                "ring_precision_criterion": ">= 50% component members belong to ground-truth ring",
            },
            "summary": {
                "member_precision": {"mean": float(np.mean(mem_p_list)), "std": float(np.std(mem_p_list, ddof=1))},
                "member_recall": {"mean": float(np.mean(mem_r_list)), "std": float(np.std(mem_r_list, ddof=1))},
                "member_f1": {"mean": float(np.mean(mem_f1_list)), "std": float(np.std(mem_f1_list, ddof=1))},
                "ring_precision": {"mean": float(np.mean(ring_p_list)), "std": float(np.std(ring_p_list, ddof=1))},
                "ring_recall": {"mean": float(np.mean(ring_r_list)), "std": float(np.std(ring_r_list, ddof=1))},
                "ring_f1": {"mean": float(np.mean(ring_f1_list)), "std": float(np.std(ring_f1_list, ddof=1))},
            },
            "per_seed_runs": per_seed_data,
        }, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    evaluate_ring_recovery_20_seeds()
