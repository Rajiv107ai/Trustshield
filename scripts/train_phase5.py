"""Trains hybrid GraphSAGE + XGBoost model and serializes artifacts."""

import sys
import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..", "trustshield_project")
sys.path.insert(0, PROJECT_DIR)

import joblib
from phase5_hybrid_model import run_phase5

MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def main(gnn_epochs: int = 50) -> None:
    print(f"Starting Hybrid GNN + XGBoost Training (epochs: {gnn_epochs})...")
    result = run_phase5(gnn_epochs=gnn_epochs)

    joblib.dump(result["hybrid_model"], os.path.join(MODELS_DIR, "hybrid_model.joblib"))
    joblib.dump(result["buyer_embeddings"], os.path.join(MODELS_DIR, "buyer_embeddings.joblib"))
    joblib.dump(result["seller_embeddings"], os.path.join(MODELS_DIR, "seller_embeddings.joblib"))
    joblib.dump(result["phase5_meta"], os.path.join(MODELS_DIR, "phase5_feature_meta.joblib"))
    if "calibrator" in result:
        joblib.dump(result["calibrator"], os.path.join(MODELS_DIR, "phase5_calibrator.joblib"))

    print(f"\nArtifacts saved to {MODELS_DIR}/")
    print(f"Val ROC-AUC: {result['val_auc']:.4f} | Test ROC-AUC: {result['test_auc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid GNN + XGBoost")
    parser.add_argument("--gnn-epochs", type=int, default=50, help="GraphSAGE training epochs")
    args = parser.parse_args()
    main(gnn_epochs=args.gnn_epochs)
