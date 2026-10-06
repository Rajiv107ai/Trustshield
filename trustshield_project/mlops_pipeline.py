"""Lightweight MLOps Experiment Tracking & Model Registry for TrustShield.

Provides structured lineage tracking and artifact versioning:
- Experiment Run Logging: Parameters, hyperparameters, temporal cutoffs, random seeds
- Metric Tracking: ROC-AUC, PR-AUC, F1, Expected Calibration Error, Latency
- Lineage Tracking: Artifact SHA-256 checksums, dataset hashes, schema versions
- MLflow-compatible metadata export without requiring external heavyweight daemons.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from trustshield_project.versioning import sha256_file, FEATURE_SCHEMA_VERSION, MODEL_METADATA_VERSION


@dataclass
class ExperimentRun:
    run_id: str
    experiment_name: str
    timestamp: float
    model_name: str
    model_version: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    artifact_paths: Dict[str, str]
    artifact_hashes: Dict[str, str]
    schema_version: str
    dataset_cutoff_date: str


class ExperimentTracker:
    """Experiment tracker and local model registry."""

    def __init__(self, registry_dir: str | Path = "model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.registry_dir / "experiment_runs.jsonl"

    def log_run(
        self,
        experiment_name: str,
        model_name: str,
        model_version: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[Dict[str, str | Path]] = None,
        dataset_cutoff: str = "2024-02-15",
    ) -> ExperimentRun:
        """Record an experiment run with cryptographically hashed artifact lineage."""
        run_id = f"run_{model_name}_{int(time.time())}"
        art_hashes = {}
        art_paths = {}

        if artifacts:
            for name, path in artifacts.items():
                p = Path(path)
                art_paths[name] = str(p)
                if p.exists() and p.is_file():
                    art_hashes[name] = sha256_file(p)

        run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            timestamp=time.time(),
            model_name=model_name,
            model_version=model_version,
            parameters=parameters,
            metrics={k: round(float(v), 5) for k, v in metrics.items()},
            artifact_paths=art_paths,
            artifact_hashes=art_hashes,
            schema_version=FEATURE_SCHEMA_VERSION,
            dataset_cutoff_date=dataset_cutoff,
        )

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(run)) + "\n")

        return run

    def list_runs(self) -> List[Dict[str, Any]]:
        """Read all logged experiment runs from the registry."""
        if not self.log_file.exists():
            return []
        runs = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    runs.append(json.loads(line.strip()))
        return runs
