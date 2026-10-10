"""CI Guard: Ensures all reported metrics are strictly grounded in results/results.json

Fails CI if:
1. results/results.json is missing or corrupted.
2. Unverified or fabricated metrics (e.g., historical untraceable 0.792) are claimed as verified test performance.
3. Frontend or docs contain ungrounded hardcoded benchmark tables.
"""

from __future__ import annotations
import os
import sys
import json
import re

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

if hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(encoding="utf-8", errors="replace")


def check_no_hardcoded_metrics():
    print("=" * 70)
    print("TRUSTSHIELD CI GUARD: AUDITED METRIC GROUNDING CHECK")
    print("=" * 70)

    errors = []

    # 1. Verify results/results.json exists and contains complete audited schema
    results_path = os.path.join(_ROOT_DIR, "results", "results.json")
    if not os.path.exists(results_path):
        errors.append(f"Missing canonical benchmark file: {results_path}")
        print("\n".join(errors))
        sys.exit(1)

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        errors.append(f"Failed to parse results/results.json: {exc}")
        print("\n".join(errors))
        sys.exit(1)

    required_keys = [
        "pre_registered_primary_analysis_20_seeds",
        "graph_free_baseline_comparisons_20_seeds",
        "coherent_variant_sensitivity_analysis_5_seeds",
        "tuned_hybrid_model_5_seeds",
        "real_gnn_graph_topology_stats",
        "design_rule_ablations",
        "per_type_metrics_standard_20_seeds",
        "exploratory_multiple_testing_holm_correction",
    ]
    for k in required_keys:
        if k not in data:
            errors.append(f"Missing required section in results.json: '{k}'")

    # 2. Check that frontend evaluation and model registry pages read from results.json
    frontend_eval = os.path.join(_ROOT_DIR, "frontend", "src", "app", "evaluation", "page.tsx")
    if os.path.exists(frontend_eval):
        with open(frontend_eval, "r", encoding="utf-8") as f:
            content = f.read()
        if 'fetch("/results.json")' not in content:
            errors.append("frontend/src/app/evaluation/page.tsx must dynamically fetch /results.json")
        if '0.792' in content:
            errors.append("frontend/src/app/evaluation/page.tsx contains hardcoded 0.792 metric")

    frontend_models = os.path.join(_ROOT_DIR, "frontend", "src", "app", "models", "page.tsx")
    if os.path.exists(frontend_models):
        with open(frontend_models, "r", encoding="utf-8") as f:
            content = f.read()
        if 'fetch("/results.json")' not in content:
            errors.append("frontend/src/app/models/page.tsx must dynamically fetch /results.json")

    # 3. Check docs/MODEL_CARD.md and README.md for historical 0.792 assertion
    model_card = os.path.join(_ROOT_DIR, "docs", "MODEL_CARD.md")
    if os.path.exists(model_card):
        with open(model_card, "r", encoding="utf-8") as f:
            content = f.read()
        # If 0.792 is present, it must be explicitly disclaimed as "cannot be traced to any printed output"
        if "0.792" in content and "cannot be traced to any printed output" not in content:
            errors.append("docs/MODEL_CARD.md references 0.792 without disclaiming 'cannot be traced to any printed output'")

    readme = os.path.join(_ROOT_DIR, "README.md")
    if os.path.exists(readme):
        with open(readme, "r", encoding="utf-8") as f:
            content = f.read()
        if "| **Stacking Trust Engine (Final Ensemble)** | **0.792**" in content:
            errors.append("README.md contains legacy unverified 0.792 benchmark row")

    if errors:
        print("\n❌ CI GUARD FAILED with the following violations:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print("\n✅ CI GUARD PASSED: All metrics strictly grounded in results/results.json.")
    print("=" * 70)


if __name__ == "__main__":
    check_no_hardcoded_metrics()
