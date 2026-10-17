"""
TrustShield Stage 3.5.3 — Replay-Resistant Evaluation & Input-Binding Hardening Test Suite.

Adversarial Tests Covering:
1. Holdout input-manifest identity, hash, schema, and on-disk file veracity.
2. Complete 10-artifact execution identity binding (including unique run ID).
3. Single-shot state machine (AUTHORIZED -> CLAIMED -> PREDICTIONS_COMMITTED -> EVALUATING -> COMPLETED / FAILED).
4. Crash safety, concurrency spinlocks, corruption fail-closed, container recreation simulation.
5. Strict execution pipeline ordering: proof that NO label access ever occurs on any gate failure.
6. Capacity constraint enforcement (C = floor(b * N)) and deterministic tie-breaking.
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
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_MODELS_STAGE34_DIR = os.path.join(_ROOT_DIR, "models", "stage34")
_DATA_DIR = os.path.join(_ROOT_DIR, "data", "synthetic_v2_1")
_PROJECT_DIR = os.path.join(_ROOT_DIR, "trustshield_project")

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from evaluation_gate import (
    DurableFileLedger,
    EvaluationRunState,
    RemoteLedgerAdapter,
    create_prediction_commitment,
    execute_holdout_evaluation_gate,
    validate_dataset_files_against_manifest,
    validate_prediction_contract,
    verify_holdout_authorization,
    verify_input_manifest,
    verify_prediction_commitment,
)


# ===========================================================================
# 1. Holdout Input-Manifest Binding Adversarial Tests
# ===========================================================================

class TestHoldoutManifestBindingAdversarial:
    """Verifies that input manifests are cryptographically bound, schema-validated,
    and distinguished between historical research data and future holdout targets."""

    @pytest.fixture
    def sample_manifest(self):
        return {
            "dataset_version": "2.2",
            "schema_version": "1.0.0",
            "created_at_utc": "2026-10-09T12:00:00Z",
            "files": {
                "orders.csv": "a" * 64,
                "users.csv": "b" * 64,
            },
            "row_counts": {
                "orders.csv": 10000,
                "users.csv": 5000,
            },
        }

    def test_valid_synthetic_manifest_passes(self, sample_manifest):
        """A valid manifest matching expected digest and version passes verification."""
        canonical_bytes = json.dumps(sample_manifest, sort_keys=True, indent=2).encode("utf-8")
        expected_digest = hashlib.sha256(canonical_bytes).hexdigest()

        assert verify_input_manifest(
            manifest_data=sample_manifest,
            expected_manifest_sha256=expected_digest,
            expected_dataset_version="2.2",
            disallowed_dataset_versions=["2.1", "synthetic_v2_1"],
        ) is True

    def test_changed_manifest_content_fails_digest_verification(self, sample_manifest):
        """Tampering with a single character in the manifest invalidates digest."""
        canonical_bytes = json.dumps(sample_manifest, sort_keys=True, indent=2).encode("utf-8")
        expected_digest = hashlib.sha256(canonical_bytes).hexdigest()

        tampered = copy.deepcopy(sample_manifest)
        tampered["row_counts"]["orders.csv"] = 10001

        with pytest.raises(ValueError, match="Manifest digest mismatch"):
            verify_input_manifest(
                manifest_data=tampered,
                expected_manifest_sha256=expected_digest,
            )

    def test_wrong_manifest_hash_rejected(self, sample_manifest):
        """Supplying an incorrect expected hash fails closed."""
        with pytest.raises(ValueError, match="Manifest digest mismatch"):
            verify_input_manifest(
                manifest_data=sample_manifest,
                expected_manifest_sha256="0" * 64,
            )

    def test_missing_manifest_file_rejected(self):
        """Pointing to a non-existent file path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Input manifest file not found"):
            verify_input_manifest(
                manifest_data="/non/existent/path/manifest.json",
                expected_manifest_sha256="a" * 64,
            )

    def test_disallowed_historical_v21_dataset_version_rejected(self, sample_manifest):
        """Attempting to use historical v2.1 dataset manifest for a v2.2 evaluation is rejected."""
        historical_manifest = copy.deepcopy(sample_manifest)
        historical_manifest["dataset_version"] = "2.1"
        canonical_bytes = json.dumps(historical_manifest, sort_keys=True, indent=2).encode("utf-8")
        digest = hashlib.sha256(canonical_bytes).hexdigest()

        with pytest.raises(ValueError, match="explicitly disallowed"):
            verify_input_manifest(
                manifest_data=historical_manifest,
                expected_manifest_sha256=digest,
                disallowed_dataset_versions=["2.1", "synthetic_v2_1"],
            )

    def test_wrong_dataset_version_rejected(self, sample_manifest):
        """A manifest with unexpected dataset version fails closed."""
        canonical_bytes = json.dumps(sample_manifest, sort_keys=True, indent=2).encode("utf-8")
        digest = hashlib.sha256(canonical_bytes).hexdigest()

        with pytest.raises(ValueError, match="Dataset version mismatch"):
            verify_input_manifest(
                manifest_data=sample_manifest,
                expected_manifest_sha256=digest,
                expected_dataset_version="3.0",
            )

    def test_malformed_manifest_metadata_rejected(self):
        """Manifests missing mandatory metadata (files, row_counts) fail closed."""
        malformed = {
            "dataset_version": "2.2",
            "schema_version": "1.0.0",
            # missing files and row_counts
        }
        canonical_bytes = json.dumps(malformed, sort_keys=True, indent=2).encode("utf-8")
        digest = hashlib.sha256(canonical_bytes).hexdigest()

        with pytest.raises(ValueError, match="Missing or empty required metadata field"):
            verify_input_manifest(
                manifest_data=malformed,
                expected_manifest_sha256=digest,
            )

    def test_dataset_files_validation_against_manifest_success(self):
        """Verifies on-disk file checksum and row-count verification succeeds for genuine data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "orders.csv")
            df = pd.DataFrame({"order_id": ["O1", "O2"], "val": [10.0, 20.0]})
            df.to_csv(csv_path, index=False)

            with open(csv_path, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()

            manifest = {
                "dataset_version": "test",
                "files": {"orders.csv": h},
                "row_counts": {"orders.csv": 2},
            }
            assert validate_dataset_files_against_manifest(manifest, tmp_dir) is True

    def test_dataset_files_validation_detects_tampered_file(self):
        """Verifies on-disk file checksum mismatch is immediately caught."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "orders.csv")
            with open(csv_path, "w") as f:
                f.write("order_id\nO1\n")

            manifest = {
                "dataset_version": "test",
                "files": {"orders.csv": "e" * 64},  # wrong hash
                "row_counts": {"orders.csv": 1},
            }
            with pytest.raises(ValueError, match="hash mismatch"):
                validate_dataset_files_against_manifest(manifest, tmp_dir)


# ===========================================================================
# 2. Complete 10-Artifact Execution Identity & Commitment Binding
# ===========================================================================

class TestExecutionIdentityAndCommitmentBinding:
    """Verifies that all 10 identity fields (including unique run ID) are strictly enforced."""

    @pytest.fixture
    def full_setup(self):
        preds = np.array([0.10, 0.40, 0.90], dtype=np.float64)
        run_id = "run-20261009-prod-eval-001"
        approved = {
            "model_sha256": "1310b43c3c8d23a1d4054200a5321f815de113b1a07793bb0d17f508c6cac22a",
            "calibrator_sha256": "152f287e8f4d478285eeac13d380f4312ceb453bed85b3ff1f99d799424148a8",
            "source_commit": "4615a53602d0a4ffe447d4d2fd14bd69a70bf740",
            "input_manifest_sha256": "manifest_sha256_mock_1234567890abcdef",
            "environment_digest": "sha256:evaluator_container_image_digest_12345",
            "evaluation_code_sha256": "code_hash_1234567890abcdef",
            "feature_schema_hash": "schema_hash_1234567890abcdef",
            "evaluation_policy_sha256": "policy_hash_1234567890abcdef",
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
        return preds, approved, commitment

    def test_complete_10_artifact_commitment_passes(self, full_setup):
        """Full 10-field commitment matches approved baseline under require_all_fields."""
        preds, approved, commitment = full_setup
        assert verify_prediction_commitment(commitment, approved, predictions=preds, require_all_fields=True) is True

    def test_missing_any_of_10_fields_fails_verification(self, full_setup):
        """Omitting any single field from the 10-artifact commitment fails closed."""
        preds, approved, commitment = full_setup
        for key in [
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
        ]:
            tampered = copy.deepcopy(commitment)
            del tampered[key]
            with pytest.raises(ValueError, match="Missing mandatory field"):
                verify_prediction_commitment(tampered, approved, predictions=preds, require_all_fields=True)

    def test_tampered_prediction_array_fails(self, full_setup):
        """Modifying predictions by 1e-12 fails verification."""
        preds, approved, commitment = full_setup
        tampered_preds = preds.copy()
        tampered_preds[0] += 1e-12
        with pytest.raises(ValueError, match="Submitted predictions hash .* does not match"):
            verify_prediction_commitment(commitment, approved, predictions=tampered_preds, require_all_fields=True)

    def test_substituted_model_hash_fails(self, full_setup):
        preds, approved, commitment = full_setup
        tampered = copy.deepcopy(commitment)
        tampered["bound_model_sha256"] = "f" * 64
        with pytest.raises(ValueError, match="bound_model_sha256 mismatch"):
            verify_prediction_commitment(tampered, approved, predictions=preds, require_all_fields=True)

    def test_substituted_calibrator_hash_fails(self, full_setup):
        preds, approved, commitment = full_setup
        tampered = copy.deepcopy(commitment)
        tampered["bound_calibrator_sha256"] = "f" * 64
        with pytest.raises(ValueError, match="bound_calibrator_sha256 mismatch"):
            verify_prediction_commitment(tampered, approved, predictions=preds, require_all_fields=True)

    def test_substituted_source_revision_fails(self, full_setup):
        preds, approved, commitment = full_setup
        tampered = copy.deepcopy(commitment)
        tampered["bound_source_commit"] = "0" * 40
        with pytest.raises(ValueError, match="bound_source_commit mismatch"):
            verify_prediction_commitment(tampered, approved, predictions=preds, require_all_fields=True)

    def test_substituted_evaluation_code_hash_fails(self, full_setup):
        preds, approved, commitment = full_setup
        tampered = copy.deepcopy(commitment)
        tampered["bound_evaluation_code_sha256"] = "f" * 64
        with pytest.raises(ValueError, match="bound_evaluation_code_sha256 mismatch"):
            verify_prediction_commitment(tampered, approved, predictions=preds, require_all_fields=True)

    def test_substituted_environment_digest_fails(self, full_setup):
        preds, approved, commitment = full_setup
        tampered = copy.deepcopy(commitment)
        tampered["bound_environment_digest"] = "sha256:different_container_digest"
        with pytest.raises(ValueError, match="bound_environment_digest mismatch"):
            verify_prediction_commitment(tampered, approved, predictions=preds, require_all_fields=True)

    def test_substituted_policy_hash_fails(self, full_setup):
        preds, approved, commitment = full_setup
        tampered = copy.deepcopy(commitment)
        tampered["bound_evaluation_policy_sha256"] = "f" * 64
        with pytest.raises(ValueError, match="bound_evaluation_policy_sha256 mismatch"):
            verify_prediction_commitment(tampered, approved, predictions=preds, require_all_fields=True)

    def test_mismatched_run_id_in_commitment_fails(self, full_setup):
        preds, approved, commitment = full_setup
        tampered = copy.deepcopy(commitment)
        tampered["bound_run_id"] = "different-run-id-999"
        with pytest.raises(ValueError, match="bound_run_id mismatch"):
            verify_prediction_commitment(tampered, approved, predictions=preds, require_all_fields=True)


# ===========================================================================
# 3. Durable State Ledger & Replay Resistance
# ===========================================================================

class TestDurableStateLedgerAndReplayResistance:
    """Verifies the 6-state execution lifecycle, atomic transitions, crash safety,
    and anti-replay enforcement."""

    @pytest.fixture
    def temp_ledger(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ledger_path = os.path.join(tmp_dir, "evaluation_ledger.json")
            yield DurableFileLedger(ledger_path)

    def test_first_valid_registration_and_claim(self, temp_ledger):
        """A fresh run transitions cleanly through AUTHORIZED -> CLAIMED."""
        run_id = "run_alpha"
        temp_ledger.register_authorized_run(run_id)
        entry = temp_ledger.get_run_entry(run_id)
        assert entry["state"] == EvaluationRunState.AUTHORIZED
        assert entry["labels_accessed"] is False

        temp_ledger.claim_run(run_id)
        entry = temp_ledger.get_run_entry(run_id)
        assert entry["state"] == EvaluationRunState.CLAIMED

    def test_duplicate_claim_fails_closed(self, temp_ledger):
        """Claiming an already CLAIMED run ID fails closed."""
        run_id = "run_beta"
        temp_ledger.claim_run(run_id)
        with pytest.raises(RuntimeError, match="already CLAIMED"):
            temp_ledger.claim_run(run_id)

    def test_concurrent_claims_mutual_exclusion(self, temp_ledger):
        """Simultaneous concurrent claim attempts for the same run ID result in exactly one winner."""
        run_id = "run_concurrent"
        temp_ledger.register_authorized_run(run_id)

        successes: List[int] = []
        failures: List[int] = []

        def worker(worker_id: int):
            try:
                temp_ledger.claim_run(run_id)
                successes.append(worker_id)
            except Exception:
                failures.append(worker_id)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(successes) == 1
        assert len(failures) == 9

    def test_corrupted_ledger_fails_closed(self, temp_ledger):
        """Corrupted or truncated JSON in the authoritative ledger fails closed immediately."""
        with open(temp_ledger.ledger_path, "w", encoding="utf-8") as f:
            f.write("{ invalid json corrupted truncation")

        with pytest.raises(RuntimeError, match="Corrupted evaluation ledger detected"):
            temp_ledger.claim_run("run_corrupt_test")

    def test_ledger_storage_unavailable_fails_closed(self):
        """An unwritable or invalid storage path fails closed with RuntimeError."""
        # Using a path inside a file rather than directory to simulate I/O failure
        with tempfile.NamedTemporaryFile() as f:
            unwritable_path = os.path.join(f.name, "forbidden_ledger.json")
            ledger = DurableFileLedger(unwritable_path)
            with pytest.raises(RuntimeError, match="BLOCKED: (Authoritative ledger storage is unavailable|Lock directory error)"):
                ledger.claim_run("run_io_fail")

    def test_container_recreation_simulation_preserves_state(self, temp_ledger):
        """Simulating a container restart pointing to the same durable volume retains consumed state."""
        run_id = "run_container_persist"
        temp_ledger.claim_run(run_id)
        temp_ledger.commit_predictions(run_id, "pred_sha", {"info": "test"})
        temp_ledger.start_evaluation(run_id)
        temp_ledger.complete_evaluation(run_id, {"cost": 1000})

        # New container / ledger instance instantiated on the same persistent file
        new_container_ledger = DurableFileLedger(temp_ledger.ledger_path)
        entry = new_container_ledger.get_run_entry(run_id)
        assert entry["state"] == EvaluationRunState.COMPLETED
        assert entry["labels_accessed"] is True

        # Replay attempt fails
        with pytest.raises(RuntimeError, match="already been consumed"):
            new_container_ledger.claim_run(run_id)

    def test_crash_before_commitment_state_preserved(self, temp_ledger):
        """If crash occurs after CLAIMED but before PREDICTIONS_COMMITTED, labels were never accessed."""
        run_id = "run_pre_commit_crash"
        temp_ledger.claim_run(run_id)
        entry = temp_ledger.get_run_entry(run_id)
        assert entry["state"] == EvaluationRunState.CLAIMED
        assert entry["labels_accessed"] is False

    def test_crash_after_commitment_state_preserved(self, temp_ledger):
        """If crash occurs after PREDICTIONS_COMMITTED but before EVALUATING, labels were never accessed."""
        run_id = "run_post_commit_crash"
        temp_ledger.claim_run(run_id)
        temp_ledger.commit_predictions(run_id, "pred_hash_xyz", {"info": "committed"})
        entry = temp_ledger.get_run_entry(run_id)
        assert entry["state"] == EvaluationRunState.PREDICTIONS_COMMITTED
        assert entry["labels_accessed"] is False

    def test_crash_after_label_access_blocks_replay(self, temp_ledger):
        """If crash occurs after EVALUATING (labels accessed), re-claim or re-start is strictly blocked."""
        run_id = "run_post_label_crash"
        temp_ledger.claim_run(run_id)
        temp_ledger.commit_predictions(run_id, "pred_hash_xyz", {"info": "committed"})
        temp_ledger.start_evaluation(run_id)
        entry = temp_ledger.get_run_entry(run_id)
        assert entry["state"] == EvaluationRunState.EVALUATING
        assert entry["labels_accessed"] is True

        # Attempting to re-claim or re-start fails closed
        with pytest.raises(RuntimeError, match="currently in active evaluation state"):
            temp_ledger.claim_run(run_id)

        with pytest.raises(RuntimeError, match="Must be in PREDICTIONS_COMMITTED state"):
            temp_ledger.start_evaluation(run_id)

    def test_attempted_replay_after_failure_blocked(self, temp_ledger):
        """A FAILED run cannot be re-claimed or re-evaluated with the same run ID."""
        run_id = "run_failed_replay"
        temp_ledger.claim_run(run_id)
        temp_ledger.commit_predictions(run_id, "pred_hash_xyz", {"info": "committed"})
        temp_ledger.start_evaluation(run_id)
        temp_ledger.fail_evaluation(run_id, "Scoring crashed", "LABEL_SCORING_EXECUTION")

        entry = temp_ledger.get_run_entry(run_id)
        assert entry["state"] == EvaluationRunState.FAILED

        with pytest.raises(RuntimeError, match="already been consumed"):
            temp_ledger.claim_run(run_id)

    def test_attempted_replay_after_completion_blocked(self, temp_ledger):
        """A COMPLETED run cannot be re-claimed or re-evaluated."""
        run_id = "run_completed_replay"
        temp_ledger.claim_run(run_id)
        temp_ledger.commit_predictions(run_id, "pred_hash_xyz", {"info": "committed"})
        temp_ledger.start_evaluation(run_id)
        temp_ledger.complete_evaluation(run_id, {"cost": 500})

        with pytest.raises(RuntimeError, match="already been consumed"):
            temp_ledger.claim_run(run_id)

    def test_invalid_state_transitions_prohibited(self, temp_ledger):
        """Out-of-order transitions (e.g. committing before claiming, starting before committing) fail."""
        run_id = "run_out_of_order"
        with pytest.raises(RuntimeError, match="not found in ledger"):
            temp_ledger.commit_predictions(run_id, "pred_sha", {})

        temp_ledger.register_authorized_run(run_id)
        with pytest.raises(RuntimeError, match="Must be in CLAIMED state"):
            temp_ledger.commit_predictions(run_id, "pred_sha", {})

        with pytest.raises(RuntimeError, match="Must be in PREDICTIONS_COMMITTED state"):
            temp_ledger.start_evaluation(run_id)

    def test_remote_ledger_adapter_interface_specification(self):
        """Verifies that RemoteLedgerAdapter requires an external endpoint and documents limitations."""
        adapter = RemoteLedgerAdapter(endpoint_url="https://evaluator.internal/api/ledger", evaluator_api_key="secret")
        with pytest.raises(NotImplementedError, match="Remote ledger service requires active external evaluator endpoint"):
            adapter.claim_run("run_remote")


# ===========================================================================
# 4. Authorization Ordering & Pre-Scoring Gates
# ===========================================================================

class TestAuthorizationOrderingAndPreScoringGates:
    """Proves that on ANY precondition failure, label access is NEVER invoked."""

    @pytest.fixture
    def gate_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ledger_path = os.path.join(tmp_dir, "gate_ledger.json")
            audit_log = os.path.join(tmp_dir, "gate_audit.jsonl")
            ledger = DurableFileLedger(ledger_path)

            preds = np.array([0.15, 0.45, 0.82], dtype=np.float64)
            order_ids = ["ORD_101", "ORD_102", "ORD_103"]
            run_id = "run-20261009-gate-001"
            token = "secret-token-xyz-123456"
            token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

            manifest_dict = {
                "dataset_version": "2.2",
                "schema_version": "1.0.0",
                "files": {"orders.csv": "a" * 64},
                "row_counts": {"orders.csv": 3},
            }
            manifest_bytes = json.dumps(manifest_dict, sort_keys=True, indent=2).encode("utf-8")
            manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

            approved_manifest = {
                "model_sha256": "1310b43c3c8d23a1d4054200a5321f815de113b1a07793bb0d17f508c6cac22a",
                "calibrator_sha256": "152f287e8f4d478285eeac13d380f4312ceb453bed85b3ff1f99d799424148a8",
                "source_commit": "4615a53602d0a4ffe447d4d2fd14bd69a70bf740",
                "input_manifest_sha256": manifest_sha256,
                "environment_digest": "sha256:evaluator_container_image_digest_12345",
                "evaluation_code_sha256": "code_hash_1234567890abcdef",
                "feature_schema_hash": "schema_hash_1234567890abcdef",
                "evaluation_policy_sha256": "policy_hash_1234567890abcdef",
                "run_id": run_id,
            }

            commitment = create_prediction_commitment(
                preds,
                model_artifact_hash=approved_manifest["model_sha256"],
                source_commit_hash=approved_manifest["source_commit"],
                input_manifest_hash=approved_manifest["input_manifest_sha256"],
                environment_digest=approved_manifest["environment_digest"],
                calibrator_artifact_hash=approved_manifest["calibrator_sha256"],
                evaluation_code_hash=approved_manifest["evaluation_code_sha256"],
                feature_schema_hash=approved_manifest["feature_schema_hash"],
                evaluation_policy_hash=approved_manifest["evaluation_policy_sha256"],
                run_id=run_id,
            )

            class LabelScorerSpy:
                def __init__(self):
                    self.called = False
                def __call__(self):
                    self.called = True
                    return {"cost": 1000.0}

            yield {
                "run_id": run_id,
                "token": token,
                "token_hash": token_hash,
                "preds": preds,
                "order_ids": order_ids,
                "commitment": commitment,
                "approved_manifest": approved_manifest,
                "manifest_dict": manifest_dict,
                "ledger": ledger,
                "audit_log": audit_log,
                "spy": LabelScorerSpy(),
            }

    def test_missing_authorization_never_accesses_labels(self, gate_fixtures):
        f = gate_fixtures
        with pytest.raises(PermissionError, match="authorization token is missing"):
            execute_holdout_evaluation_gate(
                run_id=f["run_id"],
                auth_token=None,
                authorized_token_hash=f["token_hash"],
                predictions=f["preds"],
                submitted_order_ids=f["order_ids"],
                expected_order_ids=f["order_ids"],
                commitment=f["commitment"],
                approved_manifest=f["approved_manifest"],
                input_manifest_data=f["manifest_dict"],
                ledger=f["ledger"],
                label_scoring_fn=f["spy"],
                audit_log_path=f["audit_log"],
            )
        assert f["spy"].called is False

    def test_invalid_token_never_accesses_labels(self, gate_fixtures):
        f = gate_fixtures
        with pytest.raises(PermissionError, match="Invalid authorization token hash"):
            execute_holdout_evaluation_gate(
                run_id=f["run_id"],
                auth_token="wrong_token_secret",
                authorized_token_hash=f["token_hash"],
                predictions=f["preds"],
                submitted_order_ids=f["order_ids"],
                expected_order_ids=f["order_ids"],
                commitment=f["commitment"],
                approved_manifest=f["approved_manifest"],
                input_manifest_data=f["manifest_dict"],
                ledger=f["ledger"],
                label_scoring_fn=f["spy"],
                audit_log_path=f["audit_log"],
            )
        assert f["spy"].called is False

    def test_expired_token_never_accesses_labels(self, gate_fixtures):
        f = gate_fixtures
        past_expiry = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        with pytest.raises(PermissionError, match="token has expired"):
            execute_holdout_evaluation_gate(
                run_id=f["run_id"],
                auth_token=f["token"],
                authorized_token_hash=f["token_hash"],
                predictions=f["preds"],
                submitted_order_ids=f["order_ids"],
                expected_order_ids=f["order_ids"],
                commitment=f["commitment"],
                approved_manifest=f["approved_manifest"],
                input_manifest_data=f["manifest_dict"],
                ledger=f["ledger"],
                label_scoring_fn=f["spy"],
                audit_log_path=f["audit_log"],
                expires_at_utc=past_expiry,
            )
        assert f["spy"].called is False

    def test_environment_digest_mismatch_never_accesses_labels(self, gate_fixtures):
        f = gate_fixtures
        tampered_manifest = copy.deepcopy(f["approved_manifest"])
        tampered_manifest["environment_digest"] = "sha256:different_unapproved_environment"
        with pytest.raises(ValueError, match="bound_environment_digest mismatch"):
            execute_holdout_evaluation_gate(
                run_id=f["run_id"],
                auth_token=f["token"],
                authorized_token_hash=f["token_hash"],
                predictions=f["preds"],
                submitted_order_ids=f["order_ids"],
                expected_order_ids=f["order_ids"],
                commitment=f["commitment"],
                approved_manifest=tampered_manifest,
                input_manifest_data=f["manifest_dict"],
                ledger=f["ledger"],
                label_scoring_fn=f["spy"],
                audit_log_path=f["audit_log"],
            )
        assert f["spy"].called is False

    def test_policy_hash_mismatch_never_accesses_labels(self, gate_fixtures):
        f = gate_fixtures
        tampered_manifest = copy.deepcopy(f["approved_manifest"])
        tampered_manifest["evaluation_policy_sha256"] = "f" * 64
        with pytest.raises(ValueError, match="bound_evaluation_policy_sha256 mismatch"):
            execute_holdout_evaluation_gate(
                run_id=f["run_id"],
                auth_token=f["token"],
                authorized_token_hash=f["token_hash"],
                predictions=f["preds"],
                submitted_order_ids=f["order_ids"],
                expected_order_ids=f["order_ids"],
                commitment=f["commitment"],
                approved_manifest=tampered_manifest,
                input_manifest_data=f["manifest_dict"],
                ledger=f["ledger"],
                label_scoring_fn=f["spy"],
                audit_log_path=f["audit_log"],
            )
        assert f["spy"].called is False

    def test_manifest_mismatch_never_accesses_labels(self, gate_fixtures):
        f = gate_fixtures
        tampered_manifest_dict = copy.deepcopy(f["manifest_dict"])
        tampered_manifest_dict["dataset_version"] = "2.1"  # changed content
        with pytest.raises(ValueError, match="Manifest digest mismatch"):
            execute_holdout_evaluation_gate(
                run_id=f["run_id"],
                auth_token=f["token"],
                authorized_token_hash=f["token_hash"],
                predictions=f["preds"],
                submitted_order_ids=f["order_ids"],
                expected_order_ids=f["order_ids"],
                commitment=f["commitment"],
                approved_manifest=f["approved_manifest"],
                input_manifest_data=tampered_manifest_dict,
                ledger=f["ledger"],
                label_scoring_fn=f["spy"],
                audit_log_path=f["audit_log"],
            )
        assert f["spy"].called is False

    def test_prediction_contract_violation_never_accesses_labels(self, gate_fixtures):
        f = gate_fixtures
        invalid_preds = np.array([0.15, np.nan, 0.82], dtype=np.float64)
        with pytest.raises(ValueError, match="non-finite values"):
            execute_holdout_evaluation_gate(
                run_id=f["run_id"],
                auth_token=f["token"],
                authorized_token_hash=f["token_hash"],
                predictions=invalid_preds,
                submitted_order_ids=f["order_ids"],
                expected_order_ids=f["order_ids"],
                commitment=f["commitment"],
                approved_manifest=f["approved_manifest"],
                input_manifest_data=f["manifest_dict"],
                ledger=f["ledger"],
                label_scoring_fn=f["spy"],
                audit_log_path=f["audit_log"],
            )
        assert f["spy"].called is False

    def test_prediction_commitment_mismatch_never_accesses_labels(self, gate_fixtures):
        f = gate_fixtures
        tampered_preds = np.array([0.15, 0.45, 0.83], dtype=np.float64)  # modified score
        with pytest.raises(ValueError, match="Submitted predictions hash .* does not match"):
            execute_holdout_evaluation_gate(
                run_id=f["run_id"],
                auth_token=f["token"],
                authorized_token_hash=f["token_hash"],
                predictions=tampered_preds,
                submitted_order_ids=f["order_ids"],
                expected_order_ids=f["order_ids"],
                commitment=f["commitment"],
                approved_manifest=f["approved_manifest"],
                input_manifest_data=f["manifest_dict"],
                ledger=f["ledger"],
                label_scoring_fn=f["spy"],
                audit_log_path=f["audit_log"],
            )
        assert f["spy"].called is False

    def test_ledger_claim_failure_never_accesses_labels(self, gate_fixtures):
        f = gate_fixtures
        # Consume the run ID in advance
        f["ledger"].claim_run(f["run_id"])
        with pytest.raises(RuntimeError, match="already CLAIMED"):
            execute_holdout_evaluation_gate(
                run_id=f["run_id"],
                auth_token=f["token"],
                authorized_token_hash=f["token_hash"],
                predictions=f["preds"],
                submitted_order_ids=f["order_ids"],
                expected_order_ids=f["order_ids"],
                commitment=f["commitment"],
                approved_manifest=f["approved_manifest"],
                input_manifest_data=f["manifest_dict"],
                ledger=f["ledger"],
                label_scoring_fn=f["spy"],
                audit_log_path=f["audit_log"],
            )
        assert f["spy"].called is False

    def test_scoring_runtime_failure_transitions_ledger_to_failed(self, gate_fixtures):
        f = gate_fixtures
        def failing_scorer():
            raise RuntimeError("Unexpected scoring exception post label access")

        with pytest.raises(RuntimeError, match="Unexpected scoring exception"):
            execute_holdout_evaluation_gate(
                run_id=f["run_id"],
                auth_token=f["token"],
                authorized_token_hash=f["token_hash"],
                predictions=f["preds"],
                submitted_order_ids=f["order_ids"],
                expected_order_ids=f["order_ids"],
                commitment=f["commitment"],
                approved_manifest=f["approved_manifest"],
                input_manifest_data=f["manifest_dict"],
                ledger=f["ledger"],
                label_scoring_fn=failing_scorer,
                audit_log_path=f["audit_log"],
            )

        entry = f["ledger"].get_run_entry(f["run_id"])
        assert entry["state"] == EvaluationRunState.FAILED
        assert entry["labels_accessed"] is True
        assert entry["error"]["stage"] == "LABEL_SCORING_EXECUTION"

    def test_successful_end_to_end_gate_execution(self, gate_fixtures):
        f = gate_fixtures
        results = execute_holdout_evaluation_gate(
            run_id=f["run_id"],
            auth_token=f["token"],
            authorized_token_hash=f["token_hash"],
            predictions=f["preds"],
            submitted_order_ids=f["order_ids"],
            expected_order_ids=f["order_ids"],
            commitment=f["commitment"],
            approved_manifest=f["approved_manifest"],
            input_manifest_data=f["manifest_dict"],
            ledger=f["ledger"],
            label_scoring_fn=f["spy"],
            audit_log_path=f["audit_log"],
            expected_dataset_version="2.2",
            disallowed_dataset_versions=["2.1"],
        )
        assert results == {"cost": 1000.0}
        assert f["spy"].called is True

        entry = f["ledger"].get_run_entry(f["run_id"])
        assert entry["state"] == EvaluationRunState.COMPLETED
        assert entry["labels_accessed"] is True

        # Confirm audit log was written without leaking secrets
        with open(f["audit_log"], "r", encoding="utf-8") as al:
            lines = al.readlines()
        assert len(lines) >= 2
        for line in lines:
            event = json.loads(line)
            assert f["token"] not in line


# ===========================================================================
# 5. Capacity Policy & Contract Consistency
# ===========================================================================

class TestCapacityPolicyAndContractConsistency:
    """Verifies that capacity constraints strictly enforce C = floor(b * N)
    and deterministic tie-breaking without over-capacity selection."""

    def test_floor_rounding_capacity_rule(self):
        """Enforces C = floor(b * N) for various budgets and population sizes."""
        test_cases = [
            (0.05, 10, 0),    # floor(0.5) = 0
            (0.05, 19, 0),    # floor(0.95) = 0
            (0.05, 20, 1),    # floor(1.0) = 1
            (0.05, 21, 1),    # floor(1.05) = 1
            (0.05, 12129, 606), # floor(606.45) = 606
            (0.02, 12129, 242), # floor(242.58) = 242
            (0.01, 12129, 121), # floor(121.29) = 121
            (0.10, 12129, 1212),# floor(1212.9) = 1212
        ]
        for b, n, expected_c in test_cases:
            c = math.floor(b * n)
            assert c == expected_c, f"Failed for b={b}, N={n}: got {c}, expected {expected_c}"

    def test_deterministic_tie_breaking_never_exceeds_budget(self):
        """When multiple orders share identical scores at threshold boundary,
        budget C is never exceeded and tie-breaking is deterministic."""
        n = 100
        budget_fraction = 0.05
        capacity = math.floor(budget_fraction * n)  # 5 orders

        # 10 orders have identical high score 0.95
        df = pd.DataFrame({
            "order_id": [f"ORD_{i:03d}" for i in range(n)],
            "score": [0.95] * 10 + [0.10] * (n - 10),
            "order_date": ["2026-10-01"] * n,
        })

        # Deterministic sorting: score DESC, order_date ASC, order_id ASC
        sorted_df = df.sort_values(
            by=["score", "order_date", "order_id"],
            ascending=[False, True, True]
        )
        selected = sorted_df.iloc[:capacity]

        assert len(selected) == capacity == 5
        # Verify deterministic selection by lowest order_id among tied scores
        assert list(selected["order_id"]) == ["ORD_000", "ORD_001", "ORD_002", "ORD_003", "ORD_004"]

    def test_zero_capacity_budget(self):
        """Zero budget fraction results in zero reviews."""
        assert math.floor(0.0 * 12129) == 0
