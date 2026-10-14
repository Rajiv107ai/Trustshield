"""CI Guard: Enforces allowlist grounding for all metrics and statistics.

Fails CI if:
1. results/results.json is missing, corrupted, or missing required schema keys.
2. frontend/public/results.json differs from results/results.json.
3. In README.md, docs/*.md, results/RESULTS.md, or frontend/src/**/*.tsx, any decimal
   with >= 2 fractional digits or any percentage appears outside a block marked:
   <!-- BEGIN GENERATED --> ... <!-- END GENERATED --> (or TS/TSX equivalents or generated TS file),
   unless covered by explicit regex exemptions (version numbers, dates, line refs, network ports).
"""

from __future__ import annotations
import os
import sys
import json
import glob
import re
import hashlib

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

if hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(encoding="utf-8", errors="replace")


GENERATED_BLOCK_PATTERNS = [
    r'<!--\s*BEGIN GENERATED\s*-->.*?<!--\s*END GENERATED\s*-->',
    r'\{\s*/\*\s*BEGIN GENERATED\s*\*/\s*\}.*?\{\s*/\*\s*END GENERATED\s*\*/\s*\}',
    r'/\*\s*BEGIN GENERATED\s*\*/.*?/\*\s*END GENERATED\s*\*/',
    r'//\s*BEGIN GENERATED.*?//\s*END GENERATED',
]

# Explicit regex exemptions:
EXEMPTION_PATTERNS = [
    # ISO / formatted dates (e.g. 2025-12-31, 2026-10-14T16:21:25Z, 2025/12/31)
    r'\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\b',
    r'\b\d{4}/\d{2}/\d{2}\b',
    # Semantic versions / package version numbers (e.g. v1.0.0, 14.1.0, 3.10+, >=0.115.0)
    r'\bv?\d+\.\d+\.\d+(?:[-+][a-zA-Z0-9_.-]+)?\b',
    r'\b(?:python|py|fastapi|next|nextjs|react|torch|pyg|sklearn|scikit-learn|networkx|redis|neo4j|turbopack|pydantic|v|ver|version|v\.)\s*[-_]?\s*v?\d+\.\d+(?:\.\d+)?\+?\b',
    r'\b>=\s*\d+\.\d+(?:\.\d+)?\b',
    # Line references & code symbols
    r'#?L\d+(?:-L?\d+)?\b',
    r'\blines?\s*\d+(?:-\d+)?\b',
    # Localhost, IPv4, network ports
    r'\b127\.0\.0\.1(?::\d{2,5})?\b',
    r':\d{2,5}\b',
    r'\blocalhost:\d+\b',
]

DECIMAL_REGEX = re.compile(r'(?<![\w./])([+-]?\d+\.\d{2,})(?![\w./])')
PERCENTAGE_REGEX = re.compile(r'(?<![\w./])(\d+(?:\.\d+)?\s*%)')


def is_generated_ts_file(filepath: str, content: str) -> bool:
    norm = filepath.replace("\\", "/")
    if "/generated/" in norm or norm.endswith("demo_telemetry.ts"):
        return True
    if content.startswith("// GENERATED FILE") or content.startswith("/* GENERATED FILE */"):
        return True
    return False


def scan_file_content(content: str, filepath: str = "snippet") -> list[str]:
    """Scans content for decimals (>= 2 fractional digits) and percentages outside generated blocks."""
    if is_generated_ts_file(filepath, content):
        return []

    # Strip blocks marked BEGIN GENERATED ... END GENERATED
    stripped = content
    for p in GENERATED_BLOCK_PATTERNS:
        stripped = re.sub(p, '', stripped, flags=re.DOTALL)

    # Strip explicit regex exemptions
    for ep in EXEMPTION_PATTERNS:
        stripped = re.sub(ep, '', stripped, flags=re.IGNORECASE)

    violations = []
    for m in DECIMAL_REGEX.finditer(stripped):
        val = m.group(1)
        violations.append(f"Unapproved decimal '{val}' in {filepath} outside BEGIN GENERATED block")

    for m in PERCENTAGE_REGEX.finditer(stripped):
        val = m.group(1).replace(" ", "")
        violations.append(f"Unapproved percentage '{val}' in {filepath} outside BEGIN GENERATED block")

    return violations


def check_no_hardcoded_metrics():
    print("=" * 80)
    print("TRUSTSHIELD CI GUARD: ALLOWLIST METRIC GROUNDING CHECK")
    print("=" * 80)

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

    # 2. Verify frontend/public/results.json matches results/results.json
    public_results_path = os.path.join(_ROOT_DIR, "frontend", "public", "results.json")
    if not os.path.exists(public_results_path):
        errors.append(f"Missing frontend results file: {public_results_path}")
    else:
        with open(results_path, "rb") as f1, open(public_results_path, "rb") as f2:
            h1 = hashlib.sha256(f1.read()).hexdigest()
            h2 = hashlib.sha256(f2.read()).hexdigest()
        if h1 != h2:
            errors.append("frontend/public/results.json differs from results/results.json")

    # 3. Allowlist scan of targets: README.md, docs/*.md, results/RESULTS.md, frontend/src/**/*.tsx
    targets = [
        os.path.join(_ROOT_DIR, "README.md"),
        os.path.join(_ROOT_DIR, "results", "RESULTS.md"),
    ]
    docs_dir = os.path.join(_ROOT_DIR, "docs")
    if os.path.exists(docs_dir):
        for fname in sorted(os.listdir(docs_dir)):
            if fname.endswith(".md"):
                targets.append(os.path.join(docs_dir, fname))

    frontend_src = os.path.join(_ROOT_DIR, "frontend", "src")
    if os.path.exists(frontend_src):
        for root, _, files in os.walk(frontend_src):
            for f in files:
                if f.endswith(".tsx"):
                    targets.append(os.path.join(root, f))

    for target_path in targets:
        if not os.path.exists(target_path):
            continue
        with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        rel_path = os.path.relpath(target_path, _ROOT_DIR).replace("\\", "/")
        file_errors = scan_file_content(content, rel_path)
        errors.extend(file_errors)

    if errors:
        print("\n❌ CI GUARD FAILED with the following violations:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print(f"\n✅ CI GUARD PASSED: All {len(targets)} files strictly adhere to allowlist grounding.")
    print("=" * 80)


if __name__ == "__main__":
    check_no_hardcoded_metrics()
