"""TrustShield AI — Candidate Model Loader for Shadow-Scoring (Phase 4.2B).

Loads and caches the isolated 37-feature candidate model from:
models/experiments/phase41_causal/

Guarantees:
- Active only when TRUSTSHIELD_SHADOW_SCORING=true (or when explicitly requested).
- Validates artifact checksums and schema integrity at load time.
- Zero mutation of production Phase 3 model artifacts or serving defaults.
- Graceful exception handling if candidate artifacts are absent or corrupted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

logger = logging.getLogger(__name__)

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_EXP_DIR = os.path.join(_ROOT_DIR, "models", "experiments", "phase41_causal")

# Authoritative SHA-256 Checksums from Phase 4.1 Verification
EXPECTED_HASHES = {
    "candidate_model.joblib": "4e06e286018ca621f355139da527170b08e42ae1077666678dea06c5c2962020",
    "candidate_calibrator.joblib": "6c52a540c218c3d737acbdba0ee545bd86f9d01df11f6f6b0bcb85880f03096e",
    "candidate_feature_meta.json": "284066d1abaa74ce1a00c63606bb5bc53e07861040eb162f0aee1bcd05e60db9",
}


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class CandidateModelNotLoadedError(RuntimeError):
    """Raised when candidate model scoring is attempted but artifacts are unavailable."""
    pass


class CandidateModelLoader:
    """Singleton loader for candidate causal XGBoost model and probability calibrator."""

    def __init__(self, exp_dir: Optional[str] = None):
        self.exp_dir = exp_dir or _EXP_DIR
        self.model: Any = None
        self.calibrator: Any = None
        self.feature_meta: Optional[Dict[str, Any]] = None
        self.feature_names: List[str] = []
        self.optimal_threshold: float = 0.11
        self._is_loaded: bool = False
        self._load_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self.model is not None and self.calibrator is not None

    def load(self, verify_checksums: bool = True) -> bool:
        """Loads candidate model artifacts if available on disk."""
        model_path = os.path.join(self.exp_dir, "candidate_model.joblib")
        calib_path = os.path.join(self.exp_dir, "candidate_calibrator.joblib")
        meta_path = os.path.join(self.exp_dir, "candidate_feature_meta.json")

        for fpath, fname in [(model_path, "candidate_model.joblib"), (calib_path, "candidate_calibrator.joblib"), (meta_path, "candidate_feature_meta.json")]:
            if not os.path.isfile(fpath):
                self._load_error = f"Missing candidate artifact: {fname} at {fpath}"
                logger.warning(self._load_error)
                self._is_loaded = False
                return False

        if verify_checksums:
            for fpath, fname in [(model_path, "candidate_model.joblib"), (calib_path, "candidate_calibrator.joblib"), (meta_path, "candidate_feature_meta.json")]:
                digest = sha256_file(fpath)
                expected = EXPECTED_HASHES.get(fname)
                if expected and digest != expected:
                    self._load_error = f"Checksum mismatch for {fname}: expected {expected}, got {digest}"
                    logger.error(self._load_error)
                    self._is_loaded = False
                    return False

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                self.feature_meta = json.load(f)
            self.feature_names = self.feature_meta.get("candidate_feature_cols", [])
            self.optimal_threshold = float(self.feature_meta.get("optimal_threshold_validation", 0.11))

            if len(self.feature_names) != 37:
                raise ValueError(f"Candidate feature count mismatch: expected 37, found {len(self.feature_names)}")

            self.model = joblib.load(model_path)
            self.calibrator = joblib.load(calib_path)

            self._is_loaded = True
            self._load_error = None
            logger.info("CandidateModelLoader successfully loaded 37-feature causal candidate from %s", self.exp_dir)
            return True
        except Exception as exc:
            self._load_error = f"Candidate artifact load exception: {exc}"
            logger.error(self._load_error)
            self.model = None
            self.calibrator = None
            self._is_loaded = False
            return False

    def predict_proba(self, feature_row: np.ndarray) -> Tuple[float, float]:
        """Predicts raw and calibrated fraud probability for a single 37-feature vector.

        Returns:
            Tuple of (raw_probability, calibrated_probability)
        """
        if not self.is_loaded:
            raise CandidateModelNotLoadedError(f"Candidate model is not loaded: {self._load_error}")

        X = np.asarray(feature_row, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"Feature vector shape mismatch: expected {len(self.feature_names)} features, got {X.shape[1]}")

        raw_prob = float(self.model.predict_proba(X)[0, 1])
        cal_prob = float(self.calibrator.predict_proba(np.array([raw_prob]))[0])
        return raw_prob, cal_prob


# Global default instance (lazy-loaded upon first shadow invocation)
candidate_loader = CandidateModelLoader()
