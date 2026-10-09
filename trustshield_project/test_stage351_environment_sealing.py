"""
TrustShield Stage 3.5.1 — Environment Sealing & Security Enforcement Test Suite.

Verifies:
1. Environment definition and evaluator lockfile presence.
2. Candidate Tabular-Only XGBoost model and calibrator artifact integrity and schema.
3. Executable evaluation gate contract validation (counts, IDs, duplicates, NaNs, bounds).
4. Cryptographic prediction commitment generation and tamper sensitivity.
5. Fail-closed authorization gate and security audit logging.
6. Persistent single-shot evaluation ledger enforcement against repeated queries.
7. Verification that Holdout v2.2 generator does not exist and holdout data is not generated.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from typing import Any, Dict, List

import numpy as np
import pytest

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_MODELS_STAGE34_DIR = os.path.join(_ROOT_DIR, "models", "stage34")
_DATA_DIR = os.path.join(_ROOT_DIR, "data", "synthetic_v2_1")
_SCRIPTS_DIR = os.path.join(_ROOT_DIR, "scripts")
_DOCKER_DIR = os.path.join(_ROOT_DIR, "docker")
_PROJECT_DIR = os.path.join(_ROOT_DIR, "trustshield_project")

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from evaluation_gate import (
    create_prediction_commitment,
    enforce_single_shot_evaluation,
    validate_prediction_contract,
    verify_holdout_authorization,
)


class TestEnvironmentSealingAndArtifacts:
    """Tests environment pinning files and candidate model artifact integrity."""

    def test_evaluator_dockerfile_and_lockfile_exist(self):
        """Dockerfile.evaluator and pinned requirements lockfile must be present."""
        df_path = os.path.join(_DOCKER_DIR, "Dockerfile.evaluator")
        lock_path = os.path.join(_DOCKER_DIR, "requirements-evaluator.lock")

        assert os.path.exists(df_path), f"Missing Dockerfile: {df_path}"
        assert os.path.exists(lock_path), f"Missing lockfile: {lock_path}"

        with open(lock_path, "r", encoding="utf-8") as f:
            lock_content = f.read()
        # Verify scikit-learn is pinned to 1.9.0 to eliminate unpickling defects
        assert "scikit-learn==1.9.0" in lock_content
        assert "xgboost==3.4.1" in lock_content

    def test_candidate_tabular_model_and_calibrator_hashes(self):
        """Verifies exact SHA-256 hashes of candidate model and calibrator."""
        model_path = os.path.join(_MODELS_STAGE34_DIR, "stage34_tabular_model.joblib")
        cal_path = os.path.join(_MODELS_STAGE34_DIR, "stage34_tabular_calibrator.joblib")

        assert os.path.exists(model_path)
        assert os.path.exists(cal_path)

        with open(model_path, "rb") as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()
        with open(cal_path, "rb") as f:
            cal_hash = hashlib.sha256(f.read()).hexdigest()

        assert model_hash == "1310b43c3c8d23a1d4054200a5321f815de113b1a07793bb0d17f508c6cac22a"
        assert cal_hash == "152f287e8f4d478285eeac13d380f4312ceb453bed85b3ff1f99d799424148a8"

    def test_candidate_model_feature_schema(self):
        """Tabular model artifact expects exactly 10 tabular features."""
        import joblib
        model_path = os.path.join(_MODELS_STAGE34_DIR, "stage34_tabular_model.joblib")
        m = joblib.load(model_path)
        assert hasattr(m, "feature_names_in_")
        assert len(m.feature_names_in_) == 10
        assert m.n_features_in_ == 10


class TestExecutableSecurityGate:
    """Verifies executable gate logic in trustshield_project/evaluation_gate.py."""

    def test_contract_validation_success(self):
        """Valid inputs pass contract validation."""
        order_ids = [f"ORD_{i:04d}" for i in range(100)]
        preds = np.linspace(0.01, 0.99, 100)
        assert validate_prediction_contract(order_ids, preds, order_ids) is True

    def test_contract_validation_fails_on_length_mismatch(self):
        """Length mismatch between predictions and order IDs raises ValueError."""
        order_ids = [f"ORD_{i:04d}" for i in range(10)]
        preds = np.array([0.1] * 9)  # 9 vs 10
        with pytest.raises(ValueError, match="Row count mismatch"):
            validate_prediction_contract(order_ids[:9], preds, order_ids)

    def test_contract_validation_fails_on_duplicate_ids(self):
        """Duplicate order IDs raise ValueError."""
        order_ids = [f"ORD_{i}" for i in range(5)]
        dupe_ids = ["ORD_0", "ORD_0", "ORD_2", "ORD_3", "ORD_4"]
        preds = np.array([0.1] * 5)
        with pytest.raises(ValueError, match="Duplicate order IDs"):
            validate_prediction_contract(dupe_ids, preds, order_ids)

    def test_contract_validation_fails_on_id_reordering(self):
        """Reordered order IDs raise ValueError."""
        expected_ids = ["ORD_1", "ORD_2", "ORD_3"]
        submitted_ids = ["ORD_3", "ORD_1", "ORD_2"]
        preds = np.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="exact sequence"):
            validate_prediction_contract(submitted_ids, preds, expected_ids)

    def test_contract_validation_fails_on_nan_and_inf(self):
        """NaN or Inf predictions raise ValueError."""
        ids = ["ORD_1", "ORD_2"]
        preds_nan = np.array([0.5, np.nan])
        with pytest.raises(ValueError, match="non-finite"):
            validate_prediction_contract(ids, preds_nan, ids)

        preds_inf = np.array([np.inf, 0.5])
        with pytest.raises(ValueError, match="non-finite"):
            validate_prediction_contract(ids, preds_inf, ids)

    def test_contract_validation_fails_on_out_of_bounds(self):
        """Predictions outside [0, 1] raise ValueError."""
        ids = ["ORD_1", "ORD_2"]
        preds_neg = np.array([-0.01, 0.5])
        with pytest.raises(ValueError, match="outside valid probability range"):
            validate_prediction_contract(ids, preds_neg, ids)

        preds_high = np.array([1.01, 0.5])
        with pytest.raises(ValueError, match="outside valid probability range"):
            validate_prediction_contract(ids, preds_high, ids)

    def test_prediction_commitment_generation_and_tamper_sensitivity(self):
        """Commitment captures hash of prediction bytes; any score change modifies the digest."""
        preds = np.array([0.123, 0.456, 0.789], dtype=np.float64)
        c1 = create_prediction_commitment(
            preds,
            model_artifact_hash="model_hash_123",
            source_commit_hash="commit_hash_456",
            input_manifest_hash="input_hash_789",
            environment_digest="sha256:env_digest",
        )
        assert "prediction_sha256" in c1
        assert c1["prediction_record_count"] == 3

        # Tamper by 1e-12 in one score
        tampered_preds = preds.copy()
        tampered_preds[0] += 1e-12
        c2 = create_prediction_commitment(
            tampered_preds,
            model_artifact_hash="model_hash_123",
            source_commit_hash="commit_hash_456",
            input_manifest_hash="input_hash_789",
            environment_digest="sha256:env_digest",
        )
        assert c1["prediction_sha256"] != c2["prediction_sha256"], "Commitment must detect infinitesimal score tampering!"

    def test_holdout_authorization_fail_closed_and_audit_logging(self):
        """Missing or wrong token raises PermissionError and writes an audit event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_log = os.path.join(tmpdir, "security_audit.jsonl")
            AUTHORIZED_HASH = hashlib.sha256(b"TRUSTSHIELD_AUTH_TOKEN_TEST").hexdigest()

            # Missing token
            with pytest.raises(PermissionError, match="missing"):
                verify_holdout_authorization(None, AUTHORIZED_HASH, audit_log_path=audit_log)

            # Invalid token
            with pytest.raises(PermissionError, match="denied"):
                verify_holdout_authorization("WRONG_TOKEN", AUTHORIZED_HASH, audit_log_path=audit_log)

            # Valid token succeeds
            assert verify_holdout_authorization("TRUSTSHIELD_AUTH_TOKEN_TEST", AUTHORIZED_HASH, audit_log_path=audit_log) is True

            # Verify audit log recorded events
            assert os.path.exists(audit_log)
            with open(audit_log, "r", encoding="utf-8") as f:
                lines = [json.loads(l) for l in f if l.strip()]
            assert len(lines) == 3
            assert lines[0]["event"] == "AUTHORIZATION_FAILURE"
            assert lines[0]["reason"] == "MISSING_TOKEN"
            assert lines[1]["event"] == "AUTHORIZATION_FAILURE"
            assert lines[1]["reason"] == "INVALID_TOKEN_HASH"
            assert lines[2]["event"] == "AUTHORIZATION_SUCCESS"

    def test_single_shot_evaluation_ledger(self):
        """Repeated evaluation run IDs fail closed with RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_file = os.path.join(tmpdir, "execution_ledger.json")

            # First run succeeds
            enforce_single_shot_evaluation(ledger_file, "EVAL_RUN_001")
            assert os.path.exists(ledger_file)

            # Second attempt with same run ID raises RuntimeError
            with pytest.raises(RuntimeError, match="Repeated holdout evaluation"):
                enforce_single_shot_evaluation(ledger_file, "EVAL_RUN_001")

            # Distinct run ID succeeds
            enforce_single_shot_evaluation(ledger_file, "EVAL_RUN_002")


class TestHoldoutProtectionGuards:
    """Guards ensuring Synthetic Dataset v2.2 has not been generated or exposed."""

    def test_v2_2_generator_and_data_files_do_not_exist(self):
        """Ensures generate_realistic_synthetic_data_v2_2.py and data/synthetic_v2_2/ do not exist."""
        v2_2_script = os.path.join(_SCRIPTS_DIR, "generate_realistic_synthetic_data_v2_2.py")
        v2_2_data = os.path.join(_ROOT_DIR, "data", "synthetic_v2_2")

        assert not os.path.exists(v2_2_script), "Violation: generate_realistic_synthetic_data_v2_2.py must not exist!"
        assert not os.path.exists(v2_2_data), "Violation: data/synthetic_v2_2/ must not exist!"

    def test_dataset_manifest_v2_1_unmutated(self):
        """Verifies synthetic_v2_1 manifest SHA-256 remains intact."""
        manifest_path = os.path.join(_DATA_DIR, "dataset_manifest.json")
        with open(manifest_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        assert h == "b0aba240f1439c1d83c1c9773977334ddf39321feed0e95f53ef30b1c138e74b"
