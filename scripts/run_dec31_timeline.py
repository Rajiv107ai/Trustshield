"""Dec-31 Timeline: Evaluates Seed 42 total and fraud orders on Dec 31 across 5 key git commits."""

from __future__ import annotations
import subprocess
import os
import shutil
import tempfile
import sys

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PYTHON_EXE = os.path.abspath(os.path.join(_ROOT_DIR, ".venv", "Scripts", "python.exe"))

COMMITS = [
    ("original_commit", "498315abca"),
    ("commit_before_af919e8", "d0465b335c"),
    ("af919e8", "af919e85e8"),
    ("first_commit_after_af919e8_order_return", "7eca046e3d"),
    ("HEAD", "7684c4e5ed"),
]

SUBPROCESS_SNIPPET = """
import sys, os, pandas as pd
p = os.path.join(os.getcwd(), "trustshield_project")
if p not in sys.path:
    sys.path.insert(0, p)

try:
    from entity_generator import generate_full_pipeline
    pipe = generate_full_pipeline(seed=42)
    orders = pipe["result"]["orders"]
except ImportError:
    from entity_generator import build_base_entities
    from product_listing_generator import build_catalog_and_listings
    from order_return_generator import build_orders_and_returns
    from fraud_injection import inject_all_fraud
    base = build_base_entities()
    catalog = build_catalog_and_listings(base["sellers"])
    txn = build_orders_and_returns(base["buyers"], catalog["sellers"], catalog["listings"], base["device_mapping"])
    result = inject_all_fraud(
        catalog["listings"], txn["orders"], txn["returns"], txn["buyers"], catalog["products"],
        base["address_sharing_log"], base["device_sharing_log"],
    )
    orders = result["orders"]

orders["order_date"] = pd.to_datetime(orders["order_date"])
dec31 = orders[orders["order_date"].dt.strftime("%Y-%m-%d") == "2025-12-31"]
tot = len(dec31)
fraud = int(dec31["is_fraudulent"].astype(bool).sum()) if "is_fraudulent" in dec31.columns else 0
print(f"DEC31_TOTAL={tot} DEC31_FRAUD={fraud}")
"""

def main():
    print("=" * 80)
    print("DEC-31 SEED 42 ORDER COUNTS ACROSS GIT TIMELINE")
    print("=" * 80)

    for label, commit in COMMITS:
        tmp_dir = os.path.join(_ROOT_DIR, "scratch", f"wt_{commit[:10]}")
        if os.path.exists(tmp_dir):
            subprocess.run(["git", "worktree", "remove", "--force", tmp_dir], capture_output=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)

        res = subprocess.run(["git", "worktree", "add", "--detach", tmp_dir, commit], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Failed to create worktree for {label} ({commit}): {res.stderr}")
            continue

        try:
            run_res = subprocess.run(
                [PYTHON_EXE, "-c", SUBPROCESS_SNIPPET],
                cwd=tmp_dir,
                capture_output=True,
                text=True,
            )
            out_lines = [l for l in run_res.stdout.splitlines() if "DEC31_" in l]
            output_str = out_lines[-1] if out_lines else run_res.stdout.strip()
            print(f"[{label.upper()}] Commit: {commit[:10]}")
            print(f"  Result: {output_str}")
            if run_res.returncode != 0:
                print(f"  Error (exit {run_res.returncode}): {run_res.stderr.strip()[:200]}")
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", tmp_dir], capture_output=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
