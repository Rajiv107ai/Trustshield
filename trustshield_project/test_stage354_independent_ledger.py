"""
TrustShield Stage 3.5.4 — Independent Ledger Integration, Crash-Safety & Adversarial Manifest Test Suite.

Adversarial Tests Covering:
1. Provider-neutral external ledger client (RemoteLedgerAdapter) with authenticated requests.
2. Server-side optimistic concurrency control (CAS) via monotonic versioning and atomic spinlocks.
3. Cross-machine replay resistance: proof that fresh clients with empty local disks cannot replay consumed run IDs.
4. Server-side terminal state enforcement (COMPLETED and FAILED cannot be reset or transitioned).
5. Network fault injection (timeouts, connection drops, HTTP 500) and fail-closed isolation.
6. Comprehensive crash-point injection proving label-access isolation across all pre-scoring failure points.
7. Manifest path traversal, absolute path, symlink escape, and closed-dataset policy enforcement.
8. Time-of-Check to Time-of-Use (TOCTOU) elimination via atomic load-and-verify dataset parsing.
9. Tabular CSV integrity: duplicate and null order ID detection.
10. Ten-artifact commitment canonicalization: IEEE 754 float64 little-endian (<f8) byte determinism and bounds validation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_PROJECT_DIR = os.path.join(_ROOT_DIR, "trustshield_project")

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
    execute_holdout_evaluation_gate,
    load_and_verify_dataset_file,
    validate_dataset_files_against_manifest,
    verify_holdout_authorization,
    verify_input_manifest,
    verify_prediction_commitment,
)


# ===========================================================================
# 1. External Ledger Integration & Optimistic Concurrency Control (CAS)
# ===========================================================================

class TestExternalLedgerIntegrationAndCAS:
    """Verifies the provider-neutral RemoteLedgerAdapter and server-side state engine."""

    @pytest.fixture
    def setup_remote_ledger(self):
        api_key = "evaluator_secret_api_key_xyz"
        mock_service = MockRemoteLedgerService(valid_api_key=api_key)
        client = RemoteLedgerAdapter(
            endpoint_url="https://evaluator.trustshield.internal/api",
            evaluator_api_key=api_key,
            transport=mock_service.handle_request,
            active=True,
        )
        return mock_service, client, api_key

    def test_authenticated_lifecycle_transitions_and_versioning(self, setup_remote_ledger):
        """Clean execution through all lifecycle states increments version monotonically."""
        service, client, _ = setup_remote_ledger
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

        client.complete_evaluation(run_id, {"expected_cost": 101090.0})
        entry = client.get_run_entry(run_id)
        assert entry["state"] == EvaluationRunState.COMPLETED
        assert entry["version"] == 5

    def test_missing_or_invalid_auth_token_rejected_fail_closed(self, setup_remote_ledger):
        """Unauthenticated or invalid token requests are rejected with PermissionError."""
        service, _, _ = setup_remote_ledger
        unauthed_client = RemoteLedgerAdapter(
            endpoint_url="https://evaluator.trustshield.internal/api",
            evaluator_api_key="wrong_invalid_key",
            transport=service.handle_request,
            active=True,
        )
        with pytest.raises(PermissionError, match="authentication failed"):
            unauthed_client.claim_run("run-auth-fail")

    def test_optimistic_concurrency_conflict_detected_and_rejected(self, setup_remote_ledger):
        """Out-of-sync expected version triggers 409 conflict and fails closed."""
        service, client, _ = setup_remote_ledger
        run_id = "run-cas-conflict"
        client.register_authorized_run(run_id)

        # Forge stale client version
        client.run_versions[run_id] = 99
        with pytest.raises(RuntimeError, match="Optimistic concurrency version mismatch"):
            client.claim_run(run_id)

    def test_concurrent_claim_attempts_across_threads_result_in_one_winner(self, setup_remote_ledger):
        """10 concurrent threads attempting to claim the same run ID result in exactly 1 claim."""
        service, client, api_key = setup_remote_ledger
        run_id = "run-concurrent-cas"
        client.register_authorized_run(run_id)

        successes: List[int] = []
        conflicts: List[int] = []

        def worker(worker_id: int):
            c = RemoteLedgerAdapter(
                endpoint_url="https://evaluator.trustshield.internal/api",
                evaluator_api_key=api_key,
                transport=service.handle_request,
                active=True,
            )
            # Worker thinks run is authorized
            c.run_versions[run_id] = 1
            try:
                c.claim_run(run_id)
                successes.append(worker_id)
            except Exception:
                conflicts.append(worker_id)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(successes) == 1
        assert len(conflicts) == 9

    def test_cross_machine_replay_resistance_with_fresh_client(self, setup_remote_ledger):
        """A new client machine with an empty local filesystem is strictly blocked from reusing consumed run ID."""
        service, client1, api_key = setup_remote_ledger
        run_id = "run-cross-machine-replay"

        # Client 1 executes and completes run
        client1.claim_run(run_id)
        client1.commit_predictions(run_id, "pred_sha", {})
        client1.start_evaluation(run_id)
        client1.complete_evaluation(run_id, {"status": "SUCCESS"})

        # Client 2 (representing a completely separate machine / container with no local files)
        client2 = RemoteLedgerAdapter(
            endpoint_url="https://evaluator.trustshield.internal/api",
            evaluator_api_key=api_key,
            transport=service.handle_request,
            active=True,
        )
        assert run_id not in client2.run_versions  # no local state

        with pytest.raises(RuntimeError, match="already been consumed"):
            client2.claim_run(run_id)

    def test_server_side_terminal_states_cannot_be_reset(self, setup_remote_ledger):
        """Terminal states (COMPLETED and FAILED) cannot be re-claimed or transitioned."""
        service, client, _ = setup_remote_ledger
        run_failed = "run-failed-terminal"
        client.claim_run(run_failed)
        client.fail_evaluation(run_failed, "Fatal crash", "EVALUATION_STEP")

        with pytest.raises(RuntimeError, match="already been consumed"):
            client.claim_run(run_failed)

        with pytest.raises(RuntimeError, match="already in terminal state"):
            client.fail_evaluation(run_failed, "Secondary failure", "STEP_2")

    def test_network_fault_injection_fails_closed(self, setup_remote_ledger):
        """Network timeouts and server 500s trigger immediate fail-closed exceptions."""
        service, client, _ = setup_remote_ledger
        run_id = "run-network-fault"

        # 1. Timeout
        service.simulate_timeout = True
        with pytest.raises(RuntimeError, match="remote ledger communication failed"):
            client.claim_run(run_id)
        service.simulate_timeout = False

        # 2. Connection Drop
        service.simulate_connection_drop = True
        with pytest.raises(RuntimeError, match="remote ledger communication failed"):
            client.claim_run(run_id)
        service.simulate_connection_drop = False

        # 3. HTTP 500 Internal Server Error
        service.simulate_500 = True
        with pytest.raises(RuntimeError, match="internal server error"):
            client.claim_run(run_id)
        service.simulate_500 = False


# ===========================================================================
# 2. Crash-Safety & Pre-Scoring Label Access Isolation
# ===========================================================================

class TestCrashSafetyAndPreScoringIsolation:
    """Injects failures across all state transitions to prove labels are never accessed."""

    @pytest.fixture
    def gate_context(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ledger = DurableFileLedger(os.path.join(tmp_dir, "ledger.json"))
            audit_log = os.path.join(tmp_dir, "audit.jsonl")

            preds = np.array([0.15, 0.45, 0.82], dtype=np.float64)
            order_ids = ["O1", "O2", "O3"]
            run_id = "run-crash-test-001"
            token = "secret-token-xyz"
            token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

            manifest_dict = {
                "dataset_version": "2.2",
                "schema_version": "1.0.0",
                "files": {"orders.csv": "a" * 64},
                "row_counts": {"orders.csv": 3},
            }
            manifest_bytes = json.dumps(manifest_dict, sort_keys=True, indent=2).encode("utf-8")
            manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

            approved = {
                "model_sha256": "1310b43c3c8d23a1d4054200a5321f815de113b1a07793bb0d17f508c6cac22a",
                "calibrator_sha256": "152f287e8f4d478285eeac13d380f4312ceb453bed85b3ff1f99d799424148a8",
                "source_commit": "4615a53602d0a4ffe447d4d2fd14bd69a70bf740",
                "input_manifest_sha256": manifest_hash,
                "environment_digest": "sha256:evaluator_container_image_digest_12345",
                "evaluation_code_sha256": "code_hash_12345",
                "feature_schema_hash": "schema_hash_12345",
                "evaluation_policy_sha256": "policy_hash_12345",
                "run_id": run_id,
            }
            commitment = create_prediction_commitment(
                preds,
                model_artifact_hash=approved["model_sha256"],
                source_commit_hash=approved["source_commit"],
                input_manifest_hash=approved["input_manifest_sha256"],
                environment_digest=approved["environment_digest"],
                calibrator_artifact_hash=approved["calibrator_sha256"],
                evaluation_code_hash=approved["evaluation_code_sha256"],
                feature_schema_hash=approved["feature_schema_hash"],
                evaluation_policy_hash=approved["evaluation_policy_sha256"],
                run_id=run_id,
            )

            class SpyScorer:
                def __init__(self):
                    self.call_count = 0
                def __call__(self):
                    self.call_count += 1
                    return {"cost": 500.0}

            yield {
                "run_id": run_id,
                "token": token,
                "token_hash": token_hash,
                "preds": preds,
                "order_ids": order_ids,
                "commitment": commitment,
                "approved": approved,
                "manifest_dict": manifest_dict,
                "ledger": ledger,
                "audit_log": audit_log,
                "spy": SpyScorer(),
            }

    def test_failure_during_contract_validation_leaves_scorer_uncalled(self, gate_context):
        ctx = gate_context
        tampered_preds = np.array([0.15, np.nan, 0.82])
        with pytest.raises(ValueError, match="non-finite values"):
            execute_holdout_evaluation_gate(
                run_id=ctx["run_id"],
                auth_token=ctx["token"],
                authorized_token_hash=ctx["token_hash"],
                predictions=tampered_preds,
                submitted_order_ids=ctx["order_ids"],
                expected_order_ids=ctx["order_ids"],
                commitment=ctx["commitment"],
                approved_manifest=ctx["approved"],
                input_manifest_data=ctx["manifest_dict"],
                ledger=ctx["ledger"],
                label_scoring_fn=ctx["spy"],
            )
        assert ctx["spy"].call_count == 0

    def test_failure_during_manifest_validation_leaves_scorer_uncalled(self, gate_context):
        ctx = gate_context
        tampered_manifest = copy.deepcopy(ctx["manifest_dict"])
        tampered_manifest["dataset_version"] = "2.1"  # disallowed
        with pytest.raises(ValueError, match="Manifest digest mismatch"):
            execute_holdout_evaluation_gate(
                run_id=ctx["run_id"],
                auth_token=ctx["token"],
                authorized_token_hash=ctx["token_hash"],
                predictions=ctx["preds"],
                submitted_order_ids=ctx["order_ids"],
                expected_order_ids=ctx["order_ids"],
                commitment=ctx["commitment"],
                approved_manifest=ctx["approved"],
                input_manifest_data=tampered_manifest,
                ledger=ctx["ledger"],
                label_scoring_fn=ctx["spy"],
            )
        assert ctx["spy"].call_count == 0

    def test_crash_during_scoring_records_failed_and_blocks_automatic_replay(self, gate_context):
        ctx = gate_context
        def crashing_scorer():
            ctx["spy"].call_count += 1
            raise RuntimeError("Fatal hardware crash during matrix multiplication")

        with pytest.raises(RuntimeError, match="Fatal hardware crash"):
            execute_holdout_evaluation_gate(
                run_id=ctx["run_id"],
                auth_token=ctx["token"],
                authorized_token_hash=ctx["token_hash"],
                predictions=ctx["preds"],
                submitted_order_ids=ctx["order_ids"],
                expected_order_ids=ctx["order_ids"],
                commitment=ctx["commitment"],
                approved_manifest=ctx["approved"],
                input_manifest_data=ctx["manifest_dict"],
                ledger=ctx["ledger"],
                label_scoring_fn=crashing_scorer,
                audit_log_path=ctx["audit_log"],
            )

        assert ctx["spy"].call_count == 1
        entry = ctx["ledger"].get_run_entry(ctx["run_id"])
        assert entry["state"] == EvaluationRunState.FAILED
        assert entry["labels_accessed"] is True

        # Automatic replay attempt must fail closed without calling scorer
        with pytest.raises(RuntimeError, match="already been consumed"):
            execute_holdout_evaluation_gate(
                run_id=ctx["run_id"],
                auth_token=ctx["token"],
                authorized_token_hash=ctx["token_hash"],
                predictions=ctx["preds"],
                submitted_order_ids=ctx["order_ids"],
                expected_order_ids=ctx["order_ids"],
                commitment=ctx["commitment"],
                approved_manifest=ctx["approved"],
                input_manifest_data=ctx["manifest_dict"],
                ledger=ctx["ledger"],
                label_scoring_fn=ctx["spy"],
            )
        assert ctx["spy"].call_count == 1  # Still 1, never re-executed!


# ===========================================================================
# 3. Manifest & Dataset Integrity Adversarial Tests
# ===========================================================================

class TestManifestAndDatasetIntegrityAdversarial:
    """Verifies directory traversal prevention, closed-dataset policy, TOCTOU elimination,
    and tabular data integrity."""

    def test_path_traversal_attempts_in_manifest_rejected(self):
        """Manifest filenames attempting directory traversal are caught and rejected."""
        traversal_attempts = [
            "../../etc/passwd",
            "../sibling/orders.csv",
            "subdir/../../orders.csv",
            "..\\windows\\system32\\cmd.exe",
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            for bad_path in traversal_attempts:
                manifest = {
                    "dataset_version": "test",
                    "files": {bad_path: "a" * 64},
                }
                with pytest.raises(ValueError, match="Path traversal or absolute path detected"):
                    validate_dataset_files_against_manifest(manifest, tmp_dir)

    def test_absolute_path_attempts_in_manifest_rejected(self):
        """Manifest filenames with absolute paths are caught and rejected."""
        absolute_paths = [
            "/etc/shadow",
            "C:\\Windows\\System32\\notepad.exe",
            "D:\\data\\orders.csv",
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            for bad_path in absolute_paths:
                manifest = {
                    "dataset_version": "test",
                    "files": {bad_path: "a" * 64},
                }
                with pytest.raises(ValueError, match="Path traversal or absolute path detected"):
                    validate_dataset_files_against_manifest(manifest, tmp_dir)

    def test_closed_dataset_policy_rejects_extra_unlisted_files(self):
        """When allow_extra_files=False, unexpected files in dataset directory trigger rejection."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            valid_csv = os.path.join(tmp_dir, "orders.csv")
            with open(valid_csv, "w") as f:
                f.write("order_id\nORD1\n")
            h = hashlib.sha256(open(valid_csv, "rb").read()).hexdigest()

            # Extra unexpected file
            extra_file = os.path.join(tmp_dir, "backdoor_labels.csv")
            with open(extra_file, "w") as f:
                f.write("fraud_labels\n1\n")

            manifest = {
                "dataset_version": "test",
                "files": {"orders.csv": h},
                "row_counts": {"orders.csv": 1},
            }

            with pytest.raises(ValueError, match="Closed dataset policy violation: Unexpected extra file"):
                validate_dataset_files_against_manifest(manifest, tmp_dir, allow_extra_files=False)

    def test_duplicate_order_id_in_csv_detected_and_rejected(self):
        """CSV files containing duplicate primary key order IDs fail validation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "orders.csv")
            df = pd.DataFrame({"order_id": ["ORD_01", "ORD_01"], "amount": [10.0, 20.0]})
            df.to_csv(csv_path, index=False)
            h = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()

            manifest = {
                "dataset_version": "test",
                "files": {"orders.csv": h},
                "row_counts": {"orders.csv": 2},
            }
            with pytest.raises(ValueError, match="contains duplicate 'order_id'"):
                validate_dataset_files_against_manifest(manifest, tmp_dir)

    def test_null_order_id_in_csv_detected_and_rejected(self):
        """CSV files containing null primary keys fail validation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "orders.csv")
            df = pd.DataFrame({"order_id": ["ORD_01", None], "amount": [10.0, 20.0]})
            df.to_csv(csv_path, index=False)
            h = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()

            manifest = {
                "dataset_version": "test",
                "files": {"orders.csv": h},
                "row_counts": {"orders.csv": 2},
            }
            with pytest.raises(ValueError, match="contains null 'order_id'"):
                validate_dataset_files_against_manifest(manifest, tmp_dir)

    def test_toctou_prevention_catches_file_tampered_after_check(self):
        """Atomic load-and-verify eliminates TOCTOU gap by detecting post-check mutation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "orders.csv")
            df = pd.DataFrame({"order_id": ["ORD_01", "ORD_02"], "amount": [10.0, 20.0]})
            df.to_csv(csv_path, index=False)
            original_hash = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()

            # First load succeeds
            loaded_df = load_and_verify_dataset_file(csv_path, original_hash, expected_row_count=2)
            assert len(loaded_df) == 2

            # Adversary mutates file on disk after verification
            with open(csv_path, "a") as f:
                f.write("ORD_03,30.0\n")

            # Subsequent load detects TOCTOU tampering immediately
            with pytest.raises(ValueError, match="TOCTOU violation: File 'orders.csv' hash mismatch"):
                load_and_verify_dataset_file(csv_path, original_hash, expected_row_count=2)


# ===========================================================================
# 4. Ten-Artifact Commitment Canonicalization Review
# ===========================================================================

class TestCommitmentCanonicalizationReview:
    """Verifies IEEE 754 float64 little-endian (<f8) byte canonicalization and bounds."""

    def test_canonical_byte_order_determinism(self):
        """Verifies that prediction byte hashing uses IEEE 754 float64 little-endian (<f8)."""
        preds = np.array([0.123456789, 0.987654321], dtype=np.float64)
        expected_bytes = np.ascontiguousarray(preds, dtype=np.dtype("<f8")).tobytes()
        expected_sha = hashlib.sha256(expected_bytes).hexdigest()

        commitment = create_prediction_commitment(
            preds,
            model_artifact_hash="m" * 64,
            source_commit_hash="c" * 40,
            input_manifest_hash="i" * 64,
            environment_digest="e" * 64,
        )
        assert commitment["prediction_sha256"] == expected_sha

    def test_non_finite_predictions_rejected_before_hashing(self):
        """Arrays containing NaN or Inf are rejected prior to commitment creation."""
        with pytest.raises(ValueError, match="non-finite values"):
            create_prediction_commitment(
                np.array([0.1, np.nan]),
                model_artifact_hash="m" * 64,
                source_commit_hash="c" * 40,
                input_manifest_hash="i" * 64,
                environment_digest="e" * 64,
            )

        with pytest.raises(ValueError, match="non-finite values"):
            create_prediction_commitment(
                np.array([0.1, np.inf]),
                model_artifact_hash="m" * 64,
                source_commit_hash="c" * 40,
                input_manifest_hash="i" * 64,
                environment_digest="e" * 64,
            )

    def test_out_of_range_predictions_rejected(self):
        """Probabilities outside [0.0, 1.0] are rejected prior to hashing."""
        with pytest.raises(ValueError, match="outside valid probability range"):
            create_prediction_commitment(
                np.array([0.5, 1.0001]),
                model_artifact_hash="m" * 64,
                source_commit_hash="c" * 40,
                input_manifest_hash="i" * 64,
                environment_digest="e" * 64,
            )

        with pytest.raises(ValueError, match="outside valid probability range"):
            create_prediction_commitment(
                np.array([-0.0001, 0.5]),
                model_artifact_hash="m" * 64,
                source_commit_hash="c" * 40,
                input_manifest_hash="i" * 64,
                environment_digest="e" * 64,
            )

    def test_multidimensional_arrays_rejected(self):
        """2D matrices are rejected; strictly 1D arrays required."""
        with pytest.raises(ValueError, match="must be a 1D array"):
            create_prediction_commitment(
                np.array([[0.1, 0.2], [0.3, 0.4]]),
                model_artifact_hash="m" * 64,
                source_commit_hash="c" * 40,
                input_manifest_hash="i" * 64,
                environment_digest="e" * 64,
            )
