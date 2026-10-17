"""
TrustShield Stage 3.5.4 — Independent Evaluator Protocol Reconciliation & Execution-Boundary Adversarial Test Suite.

Adversarial Tests Covering:
1. Incomplete or substituted 10-artifact cryptographic commitments.
2. Mismatched prediction serialization (<f8 little-endian float64 determinism, shape, bounds, NaN/Inf, row ordering).
3. Invalid, missing, or expired authorization tokens.
4. Input manifest tampering, duplicate PKs, path traversal, symlink escapes, missing files, closed-dataset violations, and TOCTOU file replacement.
5. Simultaneous run claims and server-side optimistic concurrency control (CAS).
6. Process restart and remote-ledger unavailability (timeouts, drops, HTTP 500).
7. Timeout after remote claim or evaluation start.
8. Scoring exceptions after label access (fail-closed, labels_accessed=True, terminal FAILED state, no automatic retry).
9. AES-256-GCM authentication failure using test-only keys (tampered ciphertext, tampered tag, truncated blob, wrong key, fail-closed).
10. Spy function proving that invalid preconditions never invoke the label-scoring boundary.
11. Capacity metric floor integer rounding (floor(b * N)) and deterministic secondary-key tie-breaking.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_PROJECT_DIR = os.path.join(_ROOT_DIR, "trustshield_project")
_DOCKER_DIR = os.path.join(_ROOT_DIR, "docker")
_MODELS_STAGE34_DIR = os.path.join(_ROOT_DIR, "models", "stage34")

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from evaluation_gate import (
    DurableFileLedger,
    EvaluationRunState,
    MockRemoteLedgerService,
    RemoteLedgerAdapter,
    create_prediction_commitment,
    decrypt_holdout_labels_envelope,
    encrypt_holdout_labels_envelope,
    execute_holdout_evaluation_gate,
    load_and_verify_dataset_file,
    validate_dataset_files_against_manifest,
    validate_prediction_contract,
    verify_holdout_authorization,
    verify_input_manifest,
    verify_prediction_commitment,
)


# ===========================================================================
# Fixtures & Test Constants
# ===========================================================================

@pytest.fixture
def ten_artifact_fixtures():
    """Generates a valid, complete 10-artifact commitment and approved baseline manifest."""
    predictions = np.array([0.15, 0.42, 0.88, 0.05, 0.73], dtype=np.float64)
    order_ids = [f"ORD_{i:04d}" for i in range(len(predictions))]

    pred_bytes = np.ascontiguousarray(predictions, dtype=np.dtype("<f8")).tobytes()
    pred_sha256 = hashlib.sha256(pred_bytes).hexdigest()

    run_id = "run-20261009-reconciliation-001"
    model_sha256 = "1310b43c3c8ddb664be8a8461f8a8489839462c82084df58d6aa92fa94cf8928"
    calibrator_sha256 = "152f287e8f4da13854890c2a714ee6b647c28eb5837fc9081e60aa88383a1523"
    source_commit = "4615a53602d0a4ffe447d4d2fd14bd69a70bf740"
    evaluation_code_sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    feature_schema_hash = "f4c8996fb92427ae41e4649b934ca495991b7852b855e3b0c44298fc1c149afb"
    input_manifest_sha256 = "b0aba240f143cf71a25b2f2115144b62db9c6d328906bd0f6125026df1265886"
    environment_digest = "sha256:7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069"
    evaluation_policy_sha256 = "a1b2c3d4e5f67890123456789abcdef0123456789abcdef0123456789abcdef0"

    approved_manifest = {
        "model_sha256": model_sha256,
        "calibrator_sha256": calibrator_sha256,
        "source_commit": source_commit,
        "evaluation_code_sha256": evaluation_code_sha256,
        "feature_schema_hash": feature_schema_hash,
        "input_manifest_sha256": input_manifest_sha256,
        "environment_digest": environment_digest,
        "evaluation_policy_sha256": evaluation_policy_sha256,
        "run_id": run_id,
    }

    commitment = create_prediction_commitment(
        predictions=predictions,
        model_artifact_hash=model_sha256,
        source_commit_hash=source_commit,
        input_manifest_hash=input_manifest_sha256,
        environment_digest=environment_digest,
        calibrator_artifact_hash=calibrator_sha256,
        evaluation_code_hash=evaluation_code_sha256,
        feature_schema_hash=feature_schema_hash,
        evaluation_policy_hash=evaluation_policy_sha256,
        run_id=run_id,
    )

    return {
        "predictions": predictions,
        "order_ids": order_ids,
        "pred_sha256": pred_sha256,
        "approved_manifest": approved_manifest,
        "commitment": commitment,
        "run_id": run_id,
    }


# ===========================================================================
# 1. Ten-Artifact Commitment Verification
# ===========================================================================

class TestTenArtifactCommitmentVerification:
    """Verifies complete 10-artifact cryptographic binding and substitution rejection."""

    def test_complete_valid_ten_artifact_commitment_passes(self, ten_artifact_fixtures):
        """All 10 required identity fields match approved baseline."""
        fix = ten_artifact_fixtures
        assert verify_prediction_commitment(
            commitment=fix["commitment"],
            approved_manifest=fix["approved_manifest"],
            predictions=fix["predictions"],
            require_all_fields=True,
        ) is True

    @pytest.mark.parametrize("missing_field", [
        "prediction_sha256",
        "bound_model_sha256",
        "bound_calibrator_sha256",
        "bound_source_commit",
        "bound_evaluation_code_sha256",
        "bound_feature_schema_hash",
        "bound_input_manifest_sha256",
        "bound_environment_digest",
        "bound_evaluation_policy_sha256",
        "bound_run_id",
    ])
    def test_incomplete_ten_artifact_commitment_rejected(self, ten_artifact_fixtures, missing_field):
        """Dropping any single artifact field from the commitment fails closed."""
        fix = ten_artifact_fixtures
        incomplete_commitment = copy.deepcopy(fix["commitment"])
        del incomplete_commitment[missing_field]

        with pytest.raises(ValueError, match="Cryptographic commitment violation: Missing mandatory field"):
            verify_prediction_commitment(
                commitment=incomplete_commitment,
                approved_manifest=fix["approved_manifest"],
                predictions=fix["predictions"],
                require_all_fields=True,
            )

    @pytest.mark.parametrize("tampered_field,substituted_val", [
        ("bound_model_sha256", "0000000000000000000000000000000000000000000000000000000000000000"),
        ("bound_calibrator_sha256", "1111111111111111111111111111111111111111111111111111111111111111"),
        ("bound_source_commit", "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"),
        ("bound_evaluation_code_sha256", "2222222222222222222222222222222222222222222222222222222222222222"),
        ("bound_feature_schema_hash", "3333333333333333333333333333333333333333333333333333333333333333"),
        ("bound_input_manifest_sha256", "4444444444444444444444444444444444444444444444444444444444444444"),
        ("bound_environment_digest", "sha256:0000000000000000000000000000000000000000000000000000000000000000"),
        ("bound_evaluation_policy_sha256", "5555555555555555555555555555555555555555555555555555555555555555"),
        ("bound_run_id", "run-unauthorized-substitute-999"),
    ])
    def test_substituted_artifact_hash_rejected(self, ten_artifact_fixtures, tampered_field, substituted_val):
        """Substituting any bound artifact with a foreign hash fails closed."""
        fix = ten_artifact_fixtures
        tampered_commitment = copy.deepcopy(fix["commitment"])
        tampered_commitment[tampered_field] = substituted_val

        with pytest.raises(ValueError, match="Cryptographic commitment violation: .* mismatch"):
            verify_prediction_commitment(
                commitment=tampered_commitment,
                approved_manifest=fix["approved_manifest"],
                predictions=fix["predictions"],
                require_all_fields=True,
            )

    def test_missing_field_in_approved_manifest_rejected_when_all_fields_required(self, ten_artifact_fixtures):
        """If the governance approved manifest omits a required field, evaluation fails closed."""
        fix = ten_artifact_fixtures
        incomplete_manifest = copy.deepcopy(fix["approved_manifest"])
        del incomplete_manifest["evaluation_policy_sha256"]

        with pytest.raises(ValueError, match="Approved manifest missing required field"):
            verify_prediction_commitment(
                commitment=fix["commitment"],
                approved_manifest=incomplete_manifest,
                predictions=fix["predictions"],
                require_all_fields=True,
            )


# ===========================================================================
# 2. Prediction Serialization, Byte Ordering, Shape, and Bounds
# ===========================================================================

class TestPredictionSerializationAndContract:
    """Verifies canonical <f8 float64 little-endian serialization and strict contract."""

    def test_canonical_little_endian_float64_hash_determinism(self):
        """Ensures numpy dtype('<f8') little-endian representation is invariant across platforms."""
        scores = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
        canonical_bytes = np.ascontiguousarray(scores, dtype=np.dtype("<f8")).tobytes()
        expected_hash = hashlib.sha256(canonical_bytes).hexdigest()

        # Commitment generator must produce exact same SHA-256
        commitment = create_prediction_commitment(
            predictions=scores,
            model_artifact_hash="m_hash",
            source_commit_hash="c_hash",
            input_manifest_hash="i_hash",
            environment_digest="env_digest",
        )
        assert commitment["prediction_sha256"] == expected_hash

    def test_endianness_mismatch_fails_hash_verification(self, ten_artifact_fixtures):
        """Big-endian serialized bytes differ from canonical little-endian bytes."""
        fix = ten_artifact_fixtures
        scores = fix["predictions"]
        big_endian_bytes = scores.astype(">f8").tobytes()
        big_endian_hash = hashlib.sha256(big_endian_bytes).hexdigest()

        # If a commitment used big-endian hash, verifying against native array must fail
        tampered_commitment = copy.deepcopy(fix["commitment"])
        tampered_commitment["prediction_sha256"] = big_endian_hash

        with pytest.raises(ValueError, match="Submitted predictions hash .* does not match committed digest"):
            verify_prediction_commitment(
                commitment=tampered_commitment,
                approved_manifest=fix["approved_manifest"],
                predictions=scores,
                require_all_fields=True,
            )

    @pytest.mark.parametrize("invalid_shape_array", [
        np.zeros((5, 2), dtype=np.float64),
        np.zeros((2, 2, 2), dtype=np.float64),
    ])
    def test_non_1d_prediction_array_rejected(self, invalid_shape_array):
        """2D or multidimensional arrays are strictly rejected."""
        with pytest.raises(ValueError, match="Predictions must be a 1D array"):
            create_prediction_commitment(
                predictions=invalid_shape_array,
                model_artifact_hash="m",
                source_commit_hash="c",
                input_manifest_hash="i",
                environment_digest="e",
            )

    @pytest.mark.parametrize("invalid_val", [np.nan, np.inf, -np.inf])
    def test_non_finite_predictions_rejected(self, invalid_val):
        """NaN or Inf predictions violate numerical contracts."""
        scores = np.array([0.1, invalid_val, 0.9], dtype=np.float64)
        with pytest.raises(ValueError, match="non-finite values"):
            create_prediction_commitment(
                predictions=scores,
                model_artifact_hash="m",
                source_commit_hash="c",
                input_manifest_hash="i",
                environment_digest="e",
            )

    @pytest.mark.parametrize("out_of_bounds_val", [-0.0001, 1.0001, -1.0, 2.5])
    def test_out_of_bounds_probabilities_rejected(self, out_of_bounds_val):
        """Probability values strictly bounded in [0.0, 1.0]."""
        scores = np.array([0.1, out_of_bounds_val, 0.9], dtype=np.float64)
        with pytest.raises(ValueError, match="valid probability range"):
            create_prediction_commitment(
                predictions=scores,
                model_artifact_hash="m",
                source_commit_hash="c",
                input_manifest_hash="i",
                environment_digest="e",
            )

    def test_row_reordering_violates_prediction_contract_and_hash(self, ten_artifact_fixtures):
        """Permuting row order changes byte SHA-256 and fails contract validation."""
        fix = ten_artifact_fixtures
        scores = fix["predictions"]
        order_ids = fix["order_ids"]

        # Permuted order
        permuted_scores = scores[::-1]
        permuted_ids = order_ids[::-1]

        # 1. Prediction contract fails if order IDs don't match expected holdout sequence
        with pytest.raises(ValueError, match="Contract violation: Submitted order IDs do not match"):
            validate_prediction_contract(
                submitted_order_ids=permuted_ids,
                predictions=scores,
                expected_order_ids=order_ids,
            )

        # 2. Commitment verification fails if array order changed relative to committed digest
        with pytest.raises(ValueError, match="does not match committed digest"):
            verify_prediction_commitment(
                commitment=fix["commitment"],
                approved_manifest=fix["approved_manifest"],
                predictions=permuted_scores,
                require_all_fields=True,
            )


# ===========================================================================
# 3. Invalid Authorization & Expiry
# ===========================================================================

class TestAuthorizationAndTokenExpiry:
    """Verifies fail-closed authorization semantics and token lifecycle."""

    TOKEN = "evaluator_ephemeral_token_2026"
    TOKEN_HASH = hashlib.sha256(b"evaluator_ephemeral_token_2026").hexdigest()

    def test_missing_auth_token_fails_closed(self, tmp_path):
        """Missing authorization token raises PermissionError."""
        with pytest.raises(PermissionError, match="Holdout authorization token is missing"):
            verify_holdout_authorization(None, hashlib.sha256(b"secret").hexdigest(), audit_log_path=str(tmp_path / "a.jsonl"))

    def test_invalid_auth_token_hash_fails_closed(self, tmp_path):
        """Token with mismatching SHA-256 digest fails closed."""
        with pytest.raises(PermissionError, match="Invalid authorization token hash"):
            verify_holdout_authorization("unauthorized_key", hashlib.sha256(b"auth_sec").hexdigest(), audit_log_path=str(tmp_path / "a.jsonl"))

    def test_expired_auth_token_fails_closed(self, tmp_path):
        """Token with past expiration timestamp fails closed."""
        expired = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        with pytest.raises(PermissionError, match="Holdout authorization token has expired"):
            verify_holdout_authorization(self.TOKEN, self.TOKEN_HASH, audit_log_path=str(tmp_path / "a.jsonl"), expires_at_utc=expired)

    def test_valid_token_within_expiry_passes(self, tmp_path):
        """Valid token with future expiration succeeds."""
        future = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        assert verify_holdout_authorization(self.TOKEN, self.TOKEN_HASH, audit_log_path=str(tmp_path / "a.jsonl"), expires_at_utc=future) is True


# ===========================================================================
# 4. Manifest Tampering, Duplicate PKs, Path Traversal, and TOCTOU
# ===========================================================================

class TestManifestTamperingAndDataIntegrity:
    """Verifies input manifest integrity, path containment, and TOCTOU elimination."""

    def test_manifest_hash_mismatch_fails_closed(self):
        """Manifest dictionary whose canonical hash mismatches expected value fails."""
        manifest_dict = {
            "dataset_version": "2.2",
            "schema_version": "1.0",
            "files": {"orders.csv": "abc"},
            "row_counts": {"orders.csv": 10},
        }
        with pytest.raises(ValueError, match="Manifest digest mismatch"):
            verify_input_manifest(
                manifest_data=manifest_dict,
                expected_manifest_sha256="0000000000000000000000000000000000000000000000000000000000000000",
            )

    @pytest.mark.parametrize("traversal_path", [
        "../escape.csv",
        "../../etc/shadow",
        "/etc/passwd",
        "\\windows\\system32\\cmd.exe",
    ])
    def test_path_traversal_in_manifest_filenames_rejected(self, tmp_path, traversal_path):
        """Path traversal patterns in manifest keys are rejected before file access."""
        manifest_data = {
            "dataset_version": "2.2",
            "files": {traversal_path: "d41d8cd98f00b204e9800998ecf8427e"},
            "row_counts": {traversal_path: 0},
        }
        with pytest.raises(ValueError, match="Path traversal|Security violation"):
            validate_dataset_files_against_manifest(manifest_data, str(tmp_path))

    def test_closed_dataset_policy_rejects_extra_unmanifested_files(self, tmp_path):
        """Extra unmanifested files in the dataset folder violate closed-dataset policy."""
        # Create manifest with only orders.csv
        orders_file = tmp_path / "orders.csv"
        orders_content = b"order_id,score\nORD_01,0.5\n"
        orders_file.write_bytes(orders_content)
        orders_hash = hashlib.sha256(orders_content).hexdigest()

        # Create unmanifested rogue file
        rogue_file = tmp_path / "rogue_injected_labels.csv"
        rogue_file.write_bytes(b"cheat_sheet\n")

        manifest_data = {
            "dataset_version": "2.2",
            "files": {"orders.csv": orders_hash},
            "row_counts": {"orders.csv": 1},
        }

        with pytest.raises(ValueError, match="Closed dataset policy violation"):
            validate_dataset_files_against_manifest(
                manifest_data,
                str(tmp_path),
                allow_extra_files=False,
            )

    def test_duplicate_primary_keys_in_orders_csv_rejected(self, tmp_path):
        """Duplicate primary keys in orders table violate tabular dataset integrity."""
        orders_file = tmp_path / "orders.csv"
        content = b"order_id,amount\nORD_001,10.0\nORD_001,20.0\n"
        orders_file.write_bytes(content)
        h = hashlib.sha256(content).hexdigest()

        manifest_data = {
            "dataset_version": "2.2",
            "files": {"orders.csv": h},
            "row_counts": {"orders.csv": 2},
        }

        with pytest.raises(ValueError, match="contains duplicate 'order_id'"):
            validate_dataset_files_against_manifest(manifest_data, str(tmp_path))

    def test_null_primary_keys_in_orders_csv_rejected(self, tmp_path):
        """Null primary keys in orders table violate tabular dataset integrity."""
        orders_file = tmp_path / "orders.csv"
        content = b"order_id,amount\nORD_001,10.0\n,20.0\n"
        orders_file.write_bytes(content)
        h = hashlib.sha256(content).hexdigest()

        manifest_data = {
            "dataset_version": "2.2",
            "files": {"orders.csv": h},
            "row_counts": {"orders.csv": 2},
        }

        with pytest.raises(ValueError, match="contains null 'order_id'"):
            validate_dataset_files_against_manifest(manifest_data, str(tmp_path))

    def test_toctou_file_replacement_detected_at_load_time(self, tmp_path):
        """If a dataset file is modified between manifest verification and use, loader fails closed."""
        file_path = tmp_path / "orders.csv"
        initial_bytes = b"order_id,user_id\nORD_1,U_1\n"
        file_path.write_bytes(initial_bytes)
        approved_hash = hashlib.sha256(initial_bytes).hexdigest()

        # File is verified initially
        df = load_and_verify_dataset_file(str(file_path), approved_hash)
        assert len(df) == 1

        # File is replaced on disk (TOCTOU attack)
        tampered_bytes = b"order_id,user_id\nORD_1,U_ROGUE\n"
        file_path.write_bytes(tampered_bytes)

        # Loader must reject the replaced file
        with pytest.raises(ValueError, match="TOCTOU violation: File 'orders.csv' hash mismatch"):
            load_and_verify_dataset_file(str(file_path), approved_hash)

    def test_disallowed_historical_v21_dataset_version_rejected(self):
        """Attempting to use historical v2.1 dataset manifest for a v2.2 evaluation is rejected."""
        manifest = {
            "dataset_version": "2.1",
            "schema_version": "1.0.0",
            "files": {"orders.csv": "a" * 64},
            "row_counts": {"orders.csv": 100},
        }
        canonical_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
        digest = hashlib.sha256(canonical_bytes).hexdigest()

        with pytest.raises(ValueError, match="explicitly disallowed"):
            verify_input_manifest(
                manifest_data=manifest,
                expected_manifest_sha256=digest,
                disallowed_dataset_versions=["2.1", "synthetic_v2_1"],
            )


# ===========================================================================
# 5. Simultaneous Run Claims & Optimistic Concurrency Control (CAS)
# ===========================================================================

class TestSimultaneousRunClaimsAndCAS:
    """Verifies that race conditions on run claims fail closed with optimistic concurrency."""

    def test_concurrent_threads_claiming_same_run_id(self):
        """Two concurrent evaluators attempting to claim the same run ID: exactly one succeeds."""
        api_key = "test_evaluator_cas_key"
        mock_service = MockRemoteLedgerService(valid_api_key=api_key)
        run_id = "run-20261009-simultaneous-claim-001"

        client_init = RemoteLedgerAdapter("https://ledger", api_key, transport=mock_service.handle_request, active=True)
        client_init.register_authorized_run(run_id)

        results: List[Tuple[int, bool, Optional[str]]] = []
        lock = threading.Lock()

        def claim_worker(worker_id: int):
            client = RemoteLedgerAdapter("https://ledger", api_key, transport=mock_service.handle_request, active=True)
            try:
                client.claim_run(run_id)
                with lock:
                    results.append((worker_id, True, None))
            except Exception as e:
                with lock:
                    results.append((worker_id, False, str(e)))

        threads = [threading.Thread(target=claim_worker, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        successes = [r for r in results if r[1] is True]
        conflicts = [r for r in results if r[1] is False]

        assert len(successes) == 1, "Exactly one thread must succeed in claiming the run ID"
        assert len(conflicts) == 1, "The competing thread must receive a conflict (409)"
        err_msg = conflicts[0][2] or ""
        assert "already active/claimed" in err_msg or "concurrency" in err_msg

    def test_terminal_state_enforcement_blocks_reclaim(self):
        """Runs in COMPLETED or FAILED terminal states cannot be re-claimed or reset."""
        api_key = "test_evaluator_terminal_key"
        mock_service = MockRemoteLedgerService(valid_api_key=api_key)
        client = RemoteLedgerAdapter("https://ledger", api_key, transport=mock_service.handle_request, active=True)
        run_id = "run-20261009-terminal-test-002"

        client.register_authorized_run(run_id)
        client.claim_run(run_id)
        client.commit_predictions(run_id, "pred_sha", {"bound": "data"})
        client.start_evaluation(run_id)
        client.complete_evaluation(run_id, {"status": "SUCCESS"})

        entry = client.get_run_entry(run_id)
        assert entry["state"] == EvaluationRunState.COMPLETED

        # Attempt to claim completed run must fail
        with pytest.raises(RuntimeError, match="already been consumed"):
            client.claim_run(run_id)


# ===========================================================================
# 6. Process Restart and Remote-Ledger Unavailability
# ===========================================================================

class TestProcessRestartAndLedgerUnavailability:
    """Verifies fail-closed behavior across process restarts and network partitions."""

    def test_process_restart_cannot_replay_in_progress_or_consumed_run(self):
        """Simulates process crash and restart: new instance detects active run and aborts."""
        api_key = "evaluator_restart_key"
        mock_service = MockRemoteLedgerService(valid_api_key=api_key)
        run_id = "run-20261009-restart-003"

        # Host 1 starts evaluation
        client1 = RemoteLedgerAdapter("https://ledger", api_key, transport=mock_service.handle_request, active=True)
        client1.register_authorized_run(run_id)
        client1.claim_run(run_id)

        # Host 1 crashes. Host 2 boots up with a fresh in-memory state
        client2 = RemoteLedgerAdapter("https://ledger", api_key, transport=mock_service.handle_request, active=True)

        with pytest.raises(RuntimeError, match="already active/claimed"):
            client2.claim_run(run_id)

    @pytest.mark.parametrize("fault_attr", [
        "simulate_timeout",
        "simulate_connection_drop",
        "simulate_500",
    ])
    def test_remote_ledger_network_faults_fail_closed(self, fault_attr):
        """Network timeouts, connection resets, and 500s fail closed immediately."""
        api_key = "evaluator_fault_key"
        mock_service = MockRemoteLedgerService(valid_api_key=api_key)
        setattr(mock_service, fault_attr, True)

        client = RemoteLedgerAdapter("https://ledger", api_key, transport=mock_service.handle_request, active=True)
        with pytest.raises(RuntimeError, match="Authoritative remote ledger communication failed or ambiguous|Remote ledger internal server error"):
            client.claim_run("run-fault-test")

    def test_authenticated_lifecycle_transitions_and_versioning(self):
        """Clean execution through all lifecycle states increments version monotonically."""
        api_key = "evaluator_lifecycle_key"
        mock_service = MockRemoteLedgerService(valid_api_key=api_key)
        client = RemoteLedgerAdapter("https://ledger", api_key, transport=mock_service.handle_request, active=True)
        run_id = "run-20261009-remote-001"

        client.register_authorized_run(run_id)
        entry = client.get_run_entry(run_id)
        assert entry["state"] == EvaluationRunState.AUTHORIZED
        assert entry["version"] == 1

        client.claim_run(run_id)
        entry = client.get_run_entry(run_id)
        assert entry["state"] == EvaluationRunState.CLAIMED
        assert entry["version"] == 2

        client.commit_predictions(run_id, "pred_sha_123", {"bound": "data"})
        entry = client.get_run_entry(run_id)
        assert entry["state"] == EvaluationRunState.PREDICTIONS_COMMITTED
        assert entry["version"] == 3

        client.start_evaluation(run_id)
        entry = client.get_run_entry(run_id)
        assert entry["state"] == EvaluationRunState.EVALUATING
        assert entry["labels_accessed"] is True
        assert entry["version"] == 4

    def test_server_side_terminal_states_cannot_be_reset(self):
        """Terminal states (COMPLETED and FAILED) cannot be re-claimed or transitioned."""
        api_key = "evaluator_term_key"
        mock_service = MockRemoteLedgerService(valid_api_key=api_key)
        client = RemoteLedgerAdapter("https://ledger", api_key, transport=mock_service.handle_request, active=True)
        run_failed = "run-failed-terminal"
        client.register_authorized_run(run_failed)
        client.claim_run(run_failed)
        client.fail_evaluation(run_failed, "Fatal crash", "EVALUATION_STEP")

        with pytest.raises(RuntimeError, match="already been consumed"):
            client.claim_run(run_failed)

        with pytest.raises(RuntimeError, match="already in terminal state"):
            client.fail_evaluation(run_failed, "Secondary failure", "STEP_2")

    def test_corrupted_and_persistent_durable_file_ledger(self, tmp_path):
        """Corrupted JSON fails closed, and file persistence preserves completed state across re-instantiation."""
        ledger_path = str(tmp_path / "ledger.json")
        ledger = DurableFileLedger(ledger_path)
        run_id = "run-persist-001"
        ledger.claim_run(run_id)
        ledger.commit_predictions(run_id, "pred_sha", {"info": "test"})
        ledger.start_evaluation(run_id)
        ledger.complete_evaluation(run_id, {"cost": 1000})

        # Re-instantiate ledger on same file
        reloaded = DurableFileLedger(ledger_path)
        entry = reloaded.get_run_entry(run_id)
        assert entry["state"] == EvaluationRunState.COMPLETED
        assert entry["labels_accessed"] is True

        with pytest.raises(RuntimeError, match="already been consumed"):
            reloaded.claim_run(run_id)

        # Corrupt file
        with open(ledger_path, "w", encoding="utf-8") as f:
            f.write("{ invalid json corrupted")
        with pytest.raises(RuntimeError, match="Corrupted evaluation ledger detected"):
            ledger.claim_run("run-corrupt-new")


# ===========================================================================
# 7. Scoring Exceptions After Label Access (Fail-Closed & No Retry)
# ===========================================================================

class TestScoringExceptionAndNoAutomaticRetry:
    """Verifies that exceptions after label access transition to FAILED with no retry."""

    def test_scoring_exception_marks_failed_and_prevents_retry(self, ten_artifact_fixtures, tmp_path):
        """Exception during scoring transitions run to FAILED with labels_accessed=True."""
        fix = ten_artifact_fixtures
        api_key = "evaluator_scoring_ex_key"
        mock_service = MockRemoteLedgerService(valid_api_key=api_key)
        client = RemoteLedgerAdapter("https://ledger", api_key, transport=mock_service.handle_request, active=True)
        run_id = fix["run_id"]
        client.register_authorized_run(run_id)

        auth_token = "valid_governance_token"
        auth_hash = hashlib.sha256(auth_token.encode("utf-8")).hexdigest()
        audit_log = str(tmp_path / "audit.jsonl")

        manifest_data = {
            "dataset_version": "2.2",
            "schema_version": "1.0",
            "files": {"orders.csv": "abc"},
            "row_counts": {"orders.csv": 5},
        }
        manifest_bytes = json.dumps(manifest_data, sort_keys=True, indent=2).encode("utf-8")
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        fix["approved_manifest"]["input_manifest_sha256"] = manifest_hash
        fix["commitment"]["bound_input_manifest_sha256"] = manifest_hash

        def failing_scorer():
            raise ZeroDivisionError("Uncaught scoring calculation exception inside evaluator boundary")

        # Gate execution must fail
        with pytest.raises(ZeroDivisionError, match="Uncaught scoring calculation exception"):
            execute_holdout_evaluation_gate(
                run_id=run_id,
                auth_token=auth_token,
                authorized_token_hash=auth_hash,
                predictions=fix["predictions"],
                submitted_order_ids=fix["order_ids"],
                expected_order_ids=fix["order_ids"],
                commitment=fix["commitment"],
                approved_manifest=fix["approved_manifest"],
                input_manifest_data=manifest_data,
                ledger=client,
                label_scoring_fn=failing_scorer,
                audit_log_path=audit_log,
            )

        # Ledger state must be FAILED, with labels_accessed=True
        entry = client.get_run_entry(run_id)
        assert entry["state"] == EvaluationRunState.FAILED
        assert entry["labels_accessed"] is True
        assert entry["error"]["stage"] == "LABEL_SCORING_EXECUTION"

        # Subsequent attempts to execute with the same run ID MUST BE BLOCKED
        with pytest.raises(RuntimeError, match="already been consumed"):
            client.claim_run(run_id)


# ===========================================================================
# 8. AES-256-GCM Envelope Encryption & Authentication Failure
# ===========================================================================

class TestAES256GCMEnvelopeSecurity:
    """Verifies authenticated encryption envelope, nonce integrity, and tamper rejection."""

    KEY = b"k" * 32
    PLAINTEXT = b"sensitive_holdout_labels_json_blob"

    def test_valid_aes_gcm_roundtrip(self):
        """Valid 32-byte key produces authentic envelope that decrypts cleanly."""
        envelope = encrypt_holdout_labels_envelope(self.PLAINTEXT, self.KEY)
        assert len(envelope) >= 28  # 12 nonce + ciphertext + 16 tag
        assert decrypt_holdout_labels_envelope(envelope, self.KEY) == self.PLAINTEXT

    def test_corrupted_ciphertext_fails_tag_verification(self):
        """Single-bit corruption in ciphertext causes fail-closed authentication failure."""
        envelope = bytearray(encrypt_holdout_labels_envelope(self.PLAINTEXT, self.KEY))
        envelope[15] ^= 0x01
        with pytest.raises(PermissionError, match="AES-256-GCM .* failed"):
            decrypt_holdout_labels_envelope(bytes(envelope), self.KEY)

    def test_corrupted_auth_tag_fails_tag_verification(self):
        """Tampering with trailing 16-byte authentication tag causes fail-closed failure."""
        envelope = bytearray(encrypt_holdout_labels_envelope(self.PLAINTEXT, self.KEY))
        envelope[-1] ^= 0x01
        with pytest.raises(PermissionError, match="AES-256-GCM .* failed"):
            decrypt_holdout_labels_envelope(bytes(envelope), self.KEY)

    def test_wrong_key_fails_authentication(self):
        """Decrypting envelope with incorrect 32-byte key fails closed."""
        envelope = encrypt_holdout_labels_envelope(self.PLAINTEXT, self.KEY)
        with pytest.raises(PermissionError, match="AES-256-GCM .* failed"):
            decrypt_holdout_labels_envelope(envelope, b"w" * 32)

    def test_truncated_envelope_fails_closed(self):
        """Payload smaller than 28 bytes violates envelope specification."""
        with pytest.raises(ValueError, match="Payload must be at least 28 bytes"):
            decrypt_holdout_labels_envelope(b"too_short_blob", self.KEY)

    def test_invalid_key_length_rejected(self):
        """Keys that are not exactly 32 bytes are rejected."""
        with pytest.raises(ValueError, match="AES-256-GCM key must be exactly 32 bytes"):
            encrypt_holdout_labels_envelope(b"data", b"short_key")


# ===========================================================================
# 9. Spy Function Proving Preconditions Never Invoke Scoring Boundary
# ===========================================================================

def _setup_gate_spy(fix: Dict[str, Any], tmp_path: Any, **kwargs: Any) -> Tuple[Dict[str, Any], MagicMock]:
    ledger = DurableFileLedger(str(tmp_path / "ledger.json"))
    spy_scorer = MagicMock()
    manifest_data = {
        "dataset_version": "2.2",
        "schema_version": "1.0",
        "files": {"orders.csv": "abc"},
        "row_counts": {"orders.csv": 5},
    }
    m_bytes = json.dumps(manifest_data, sort_keys=True, indent=2).encode("utf-8")
    m_hash = hashlib.sha256(m_bytes).hexdigest()
    fix["approved_manifest"]["input_manifest_sha256"] = m_hash
    fix["commitment"]["bound_input_manifest_sha256"] = m_hash

    auth_token = kwargs.get("auth_token", "valid_token")
    auth_hash = kwargs.get("authorized_token_hash", hashlib.sha256(auth_token.encode("utf-8")).hexdigest())

    gate_args: Dict[str, Any] = dict(
        run_id=fix["run_id"],
        auth_token=auth_token,
        authorized_token_hash=auth_hash,
        predictions=kwargs.get("predictions", fix["predictions"]),
        submitted_order_ids=fix["order_ids"],
        expected_order_ids=fix["order_ids"],
        commitment=kwargs.get("commitment", fix["commitment"]),
        approved_manifest=fix["approved_manifest"],
        input_manifest_data=kwargs.get("input_manifest_data", manifest_data),
        ledger=ledger,
        label_scoring_fn=spy_scorer,
    )
    return gate_args, spy_scorer


class TestSpyScoringFunctionIsolation:
    """Proves that under ANY invalid precondition, label scoring is NEVER invoked."""

    def test_spy_scorer_never_invoked_on_auth_failure(self, ten_artifact_fixtures, tmp_path):
        """Invalid auth token halts execution before label scorer is reached."""
        fix = ten_artifact_fixtures
        args, spy = _setup_gate_spy(
            fix, tmp_path, auth_token="wrong_token",
            authorized_token_hash=hashlib.sha256(b"correct_token").hexdigest(),
            input_manifest_data={"dummy": "data"},
        )
        with pytest.raises(PermissionError):
            execute_holdout_evaluation_gate(**args)
        assert spy.call_count == 0

    def test_spy_scorer_never_invoked_on_commitment_mismatch(self, ten_artifact_fixtures, tmp_path):
        """Tampered model hash in commitment halts execution before label scorer is reached."""
        fix = ten_artifact_fixtures
        tampered = copy.deepcopy(fix["commitment"])
        tampered["bound_model_sha256"] = "tampered_hash_00000000000000000000000"
        args, spy = _setup_gate_spy(fix, tmp_path, commitment=tampered)
        with pytest.raises(ValueError):
            execute_holdout_evaluation_gate(**args)
        assert spy.call_count == 0

    def test_spy_scorer_never_invoked_on_prediction_contract_violation(self, ten_artifact_fixtures, tmp_path):
        """Non-finite predictions halt execution before label scorer is reached."""
        fix = ten_artifact_fixtures
        args, spy = _setup_gate_spy(fix, tmp_path, predictions=np.array([0.1, np.nan, 0.3, 0.4, 0.5]))
        with pytest.raises(ValueError, match="non-finite values"):
            execute_holdout_evaluation_gate(**args)
        assert spy.call_count == 0


# ===========================================================================
# 10. Capacity Metric & Deterministic Tie-Breaking
# ===========================================================================

class TestCapacityMetricAndTieBreaking:
    """Verifies strict floor(b * N) capacity ceiling and deterministic tie-breaking."""

    @pytest.mark.parametrize("budget_fraction,total_n,expected_cap", [
        (0.01, 1000, 10),
        (0.02, 1000, 20),
        (0.05, 1000, 50),
        (0.10, 1000, 100),
        (0.05, 999, 49),   # floor(0.05 * 999 = 49.95) == 49
        (0.01, 99, 0),     # floor(0.01 * 99 = 0.99) == 0
        (0.02, 53, 1),     # floor(0.02 * 53 = 1.06) == 1
    ])
    def test_floor_integer_rounding_capacity_calculation(self, budget_fraction, total_n, expected_cap):
        """Capacity cap K = floor(b * N) is strictly enforced with integer truncation."""
        cap = math.floor(budget_fraction * total_n)
        assert cap == expected_cap

    def test_deterministic_tie_breaking_at_capacity_boundary(self):
        """Tied predictions at the budget boundary are resolved deterministically by secondary key."""
        # 10 orders, where 5 orders share identical score 0.85
        df = pd.DataFrame({
            "order_id": [f"ORD_{i:03d}" for i in range(10)],
            "predicted_score": [0.95, 0.90, 0.85, 0.85, 0.85, 0.85, 0.85, 0.30, 0.20, 0.10],
            "fraud_label": [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
        })

        budget_fraction = 0.40  # 40% of 10 = 4 orders max
        capacity_cap = math.floor(budget_fraction * len(df))  # 4

        # Deterministic sorting: highest score first, tie-break by order_id ascending
        sorted_df = df.sort_values(
            by=["predicted_score", "order_id"],
            ascending=[False, True],
        ).reset_index(drop=True)

        selected = sorted_df.iloc[:capacity_cap]

        assert len(selected) == 4
        # ORD_000 (0.95), ORD_001 (0.90), then lowest alphabetical tied: ORD_002, ORD_003
        assert list(selected["order_id"]) == ["ORD_000", "ORD_001", "ORD_002", "ORD_003"]
        assert len(selected) <= capacity_cap

        # Verify reproducibility regardless of initial dataframe shuffling
        shuffled_df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        shuffled_sorted = shuffled_df.sort_values(
            by=["predicted_score", "order_id"],
            ascending=[False, True],
        ).reset_index(drop=True)
        shuffled_selected = shuffled_sorted.iloc[:capacity_cap]

        assert list(shuffled_selected["order_id"]) == list(selected["order_id"])

    def test_zero_capacity_budget(self):
        """Zero budget fraction results in zero reviews."""
        assert math.floor(0.0 * 12129) == 0


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
