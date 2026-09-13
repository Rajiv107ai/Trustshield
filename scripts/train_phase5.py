"""
TrustShield AI — Phase 5 Training Script

Runs the full Hybrid GNN + XGBoost pipeline and saves four artifacts
to ../models/:

    hybrid_model.joblib          — fitted XGBoost/RF
                                   (Phase-3 tabular+graph features + GNN embeddings)
    buyer_embeddings.joblib      — dict {buyer_id  → np.array(16,) float32}
    seller_embeddings.joblib     — dict {seller_id → np.array(16,) float32}
    phase5_feature_meta.joblib   — feature column lists + metadata dict

Prerequisites
-------------
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install torch_geometric
    pip install xgboost           (recommended; RF used as fallback)

Run from the project root:
    python scripts/train_phase5.py
    python scripts/train_phase5.py --gnn-epochs 100

After completion, the FastAPI server automatically detects the Phase 5
artifacts on the next startup and serves hybrid predictions.  If the
artifacts are absent, the server continues serving Phase 3 predictions
unchanged — no configuration change required.
"""

import sys
import os
import argparse

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..", "trustshield_project")
sys.path.insert(0, PROJECT_DIR)

import joblib

MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

_PHASE5_ARTIFACTS = [
    "hybrid_model.joblib",
    "buyer_embeddings.joblib",
    "seller_embeddings.joblib",
    "phase5_feature_meta.joblib",
]


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main(gnn_epochs: int = 50) -> None:
    # Deferred import so the argparse --help flag works even without torch.
    try:
        from phase5_hybrid_model import run_phase5  # type: ignore
    except ImportError:
        from trustshield_project.phase5_hybrid_model import run_phase5  # type: ignore

    print("=" * 65)
    print("TrustShield -- Phase 5: Hybrid GNN + XGBoost Training")
    print("=" * 65)
    print(f"GNN epochs : {gnn_epochs}")
    print(f"Models dir : {os.path.abspath(MODELS_DIR)}")
    print()

    result = run_phase5(gnn_epochs=gnn_epochs)

    print("\n[Saving Phase 5 artifacts to models/]")

    joblib.dump(
        result["hybrid_model"],
        os.path.join(MODELS_DIR, "hybrid_model.joblib"),
    )
    print("  [SAVED]  hybrid_model.joblib")

    joblib.dump(
        result["buyer_embeddings"],
        os.path.join(MODELS_DIR, "buyer_embeddings.joblib"),
    )
    print(f"  [SAVED]  buyer_embeddings.joblib   "
          f"({len(result['buyer_embeddings']):,} buyers)")

    joblib.dump(
        result["seller_embeddings"],
        os.path.join(MODELS_DIR, "seller_embeddings.joblib"),
    )
    print(f"  [SAVED]  seller_embeddings.joblib  "
          f"({len(result['seller_embeddings']):,} sellers)")

    joblib.dump(
        result["phase5_meta"],
        os.path.join(MODELS_DIR, "phase5_feature_meta.joblib"),
    )
    meta = result["phase5_meta"]
    print(f"  [SAVED]  phase5_feature_meta.joblib  "
          f"({len(meta['hybrid_feature_cols'])} features: "
          f"{len(meta['phase3_feature_cols'])} Phase-3 + "
          f"{len(meta['gnn_embedding_cols'])} GNN)")

    print(f"\n{'=' * 65}")
    print(f"Phase 5 training complete.")
    print(f"  Val  ROC-AUC : {result['val_auc']:.4f}")
    print(f"  Test ROC-AUC : {result['test_auc']:.4f}")
    print(f"  Classifier   : {meta['classifier']}")
    print(f"{'=' * 65}")
    print("\nRestart the FastAPI server to activate Phase 5 hybrid scoring:")
    print("    cd backend && uvicorn main:app --reload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TrustShield Phase 5 Hybrid GNN + XGBoost model"
    )
    parser.add_argument(
        "--gnn-epochs",
        type=int,
        default=50,
        help="Number of GraphSAGE training epochs (default: 50). "
             "Increase to 100-200 for better embeddings at the cost of runtime.",
    )
    args = parser.parse_args()
    main(gnn_epochs=args.gnn_epochs)
