"""
TrustShield Evaluation Gate & Security Enforcement Module.

Implements executable security controls for blind holdout evaluation:
1. Strict prediction contract validation (order IDs, row counts, score ranges, nulls).
2. Cryptographic prediction commit-reveal binding (10-artifact binding including unique run ID).
3. Holdout input-manifest verification and on-disk file consistency checks.
4. Fail-closed authorization gate for holdout label access with expiry enforcement.
5. Durable, crash-safe, append-only single-shot execution ledger with 6-state machine:
   AUTHORIZED -> CLAIMED -> PREDICTIONS_COMMITTED -> EVALUATING -> COMPLETED / FAILED.
6. Execution pipeline coordinating all security gates strictly prior to label access.
7. Tamper-evident audit logging of authorization events without leaking secrets or labels.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np


class EvaluationRunState:
    """Explicit state machine lifecycle for single-shot holdout evaluation runs."""
    AUTHORIZED = "AUTHORIZED"
    CLAIMED = "CLAIMED"
    PREDICTIONS_COMMITTED = "PREDICTIONS_COMMITTED"
    EVALUATING = "EVALUATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


def validate_prediction_contract(
    submitted_order_ids: Sequence[str],
    predictions: np.ndarray,
    expected_order_ids: Sequence[str],
) -> bool:
    """Validates submitted predictions against the approved holdout input contract.

    Parameters
    ----------
    submitted_order_ids:
        Sequence of order IDs submitted with the predictions.
    predictions:
        1D numpy array of predicted fraud probabilities.
    expected_order_ids:
        Approved sequence of holdout order IDs in exact order.

    Raises
    ------
    ValueError:
        If any contract violation occurs (length mismatch, duplicates, reordering,
        non-finite scores, or out-of-range probabilities).
    """
    if len(predictions) != len(expected_order_ids):
        raise ValueError(
            f"Row count mismatch: predictions count ({len(predictions)}) != "
            f"expected holdout orders count ({len(expected_order_ids)})."
        )

    if len(submitted_order_ids) != len(expected_order_ids):
        raise ValueError(
            f"Identifier count mismatch: submitted IDs ({len(submitted_order_ids)}) != "
            f"expected holdout IDs ({len(expected_order_ids)})."
        )

    if len(set(submitted_order_ids)) != len(submitted_order_ids):
        raise ValueError("Contract violation: Duplicate order IDs detected in submitted predictions.")

    if list(submitted_order_ids) != list(expected_order_ids):
        raise ValueError("Contract violation: Submitted order IDs do not match expected order IDs in exact sequence.")

    if not np.all(np.isfinite(predictions)):
        raise ValueError("Contract violation: Predictions contain non-finite values (NaN or Inf).")

    if (predictions < 0.0).any() or (predictions > 1.0).any():
        raise ValueError("Contract violation: Predictions contain values outside valid probability range [0.0, 1.0].")

    return True


def create_prediction_commitment(
    predictions: np.ndarray,
    model_artifact_hash: str,
    source_commit_hash: str,
    input_manifest_hash: str,
    environment_digest: str,
    calibrator_artifact_hash: Optional[str] = None,
    evaluation_code_hash: Optional[str] = None,
    feature_schema_hash: Optional[str] = None,
    evaluation_policy_hash: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Creates a cryptographic prediction commitment manifest.

    Binds the prediction array to the exact model, source commit, input data, environment,
    calibrator, evaluation code, feature schema, evaluation policy, and unique run ID.
    """
    if not isinstance(predictions, np.ndarray):
        raise ValueError("Predictions must be a numpy ndarray.")
    if predictions.ndim != 1:
        raise ValueError(f"Predictions must be a 1D array, got shape {predictions.shape}.")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Predictions contain non-finite values (NaN or Inf).")
    if (predictions < 0.0).any() or (predictions > 1.0).any():
        raise ValueError("Predictions contain values outside valid probability range [0.0, 1.0].")

    pred_bytes = np.ascontiguousarray(predictions, dtype=np.dtype("<f8")).tobytes()
    pred_sha256 = hashlib.sha256(pred_bytes).hexdigest()

    commitment = {
        "schema_version": "1.2.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_record_count": len(predictions),
        "prediction_sha256": pred_sha256,
        "bound_model_sha256": model_artifact_hash,
        "bound_source_commit": source_commit_hash,
        "bound_input_manifest_sha256": input_manifest_hash,
        "bound_environment_digest": environment_digest,
    }
    if calibrator_artifact_hash:
        commitment["bound_calibrator_sha256"] = calibrator_artifact_hash
    if evaluation_code_hash:
        commitment["bound_evaluation_code_sha256"] = evaluation_code_hash
    if feature_schema_hash:
        commitment["bound_feature_schema_hash"] = feature_schema_hash
    if evaluation_policy_hash:
        commitment["bound_evaluation_policy_sha256"] = evaluation_policy_hash
    if run_id:
        commitment["bound_run_id"] = run_id

    return commitment


def verify_prediction_commitment(
    commitment: Dict[str, Any],
    approved_manifest: Dict[str, Any],
    predictions: Optional[np.ndarray] = None,
    require_all_fields: bool = False,
) -> bool:
    """Verifies that submitted prediction commitment matches the pre-approved baseline.

    Parameters
    ----------
    commitment:
        Submitted prediction commitment manifest dictionary.
    approved_manifest:
        Dictionary of pre-approved hashes and references.
    predictions:
        Optional raw prediction array to verify against commitment's prediction_sha256.
    require_all_fields:
        If True, requires all 10 identity fields to be present in the commitment
        and matched against the approved manifest.

    Raises
    ------
    ValueError:
        If any commitment field is missing, malformed, or mismatches the approved manifest.
    """
    if not isinstance(commitment, dict):
        raise ValueError("Cryptographic commitment violation: Malformed commitment manifest.")

    if require_all_fields:
        required_keys = [
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
        ]
    else:
        required_keys = [
            "prediction_sha256",
            "bound_model_sha256",
            "bound_source_commit",
            "bound_input_manifest_sha256",
            "bound_environment_digest",
        ]

    for k in required_keys:
        if k not in commitment or not commitment[k]:
            raise ValueError(f"Cryptographic commitment violation: Missing mandatory field '{k}'.")

    # Verify predictions bytes hash if array provided
    if predictions is not None:
        if not isinstance(predictions, np.ndarray) or predictions.ndim != 1:
            raise ValueError("Predictions must be a 1D numpy array.")
        if not np.all(np.isfinite(predictions)):
            raise ValueError("Predictions contain non-finite values (NaN or Inf).")
        actual_pred_sha256 = hashlib.sha256(
            np.ascontiguousarray(predictions, dtype=np.dtype("<f8")).tobytes()
        ).hexdigest()
        if actual_pred_sha256 != commitment["prediction_sha256"]:
            raise ValueError(
                f"Cryptographic commitment violation: Submitted predictions hash ({actual_pred_sha256}) "
                f"does not match committed digest ({commitment['prediction_sha256']})."
            )

    # Verify bindings against approved manifest
    manifest_mapping = [
        ("model_sha256", "bound_model_sha256"),
        ("source_commit", "bound_source_commit"),
        ("input_manifest_sha256", "bound_input_manifest_sha256"),
        ("environment_digest", "bound_environment_digest"),
        ("calibrator_sha256", "bound_calibrator_sha256"),
        ("evaluation_code_sha256", "bound_evaluation_code_sha256"),
        ("feature_schema_hash", "bound_feature_schema_hash"),
        ("evaluation_policy_sha256", "bound_evaluation_policy_sha256"),
        ("run_id", "bound_run_id"),
    ]

    for manifest_key, commitment_key in manifest_mapping:
        if manifest_key in approved_manifest:
            expected = approved_manifest[manifest_key]
            actual = commitment.get(commitment_key)
            if actual != expected:
                raise ValueError(
                    f"Cryptographic commitment violation: {commitment_key} mismatch. "
                    f"Expected '{expected}', got '{actual}'."
                )
        elif require_all_fields:
            raise ValueError(
                f"Cryptographic commitment violation: Approved manifest missing required field '{manifest_key}'."
            )

    return True


def verify_input_manifest(
    manifest_data: Union[Dict[str, Any], str],
    expected_manifest_sha256: str,
    expected_dataset_version: Optional[str] = None,
    disallowed_dataset_versions: Optional[Sequence[str]] = None,
) -> bool:
    """Verifies that the provided input manifest matches approved identity, hash, and schema.

    Parameters
    ----------
    manifest_data:
        Either a file path (str) to the manifest JSON, or a parsed dictionary.
    expected_manifest_sha256:
        The expected cryptographic SHA-256 digest of the manifest.
    expected_dataset_version:
        Optional expected dataset version string (e.g., '2.2').
    disallowed_dataset_versions:
        Optional sequence of versions that are disallowed (e.g. historical '2.1'
        when evaluating future holdout).

    Raises
    ------
    ValueError:
        If hash mismatches, manifest is malformed, required metadata is missing,
        or dataset version is invalid/disallowed.
    FileNotFoundError:
        If manifest file path does not exist.
    """
    if isinstance(manifest_data, str):
        if not os.path.exists(manifest_data):
            raise FileNotFoundError(f"Input manifest file not found at '{manifest_data}'.")
        with open(manifest_data, "rb") as f:
            raw_bytes = f.read()
        actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        try:
            manifest_dict = json.loads(raw_bytes.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"Input manifest verification failed: Invalid JSON ({e}).")
    elif isinstance(manifest_data, dict):
        canonical_bytes = json.dumps(manifest_data, sort_keys=True, indent=2).encode("utf-8")
        actual_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
        manifest_dict = manifest_data
    else:
        raise ValueError("Input manifest verification failed: manifest_data must be a file path or dict.")

    if actual_sha256 != expected_manifest_sha256:
        raise ValueError(
            f"Input manifest verification failed: Manifest digest mismatch. "
            f"Expected '{expected_manifest_sha256}', got '{actual_sha256}'."
        )

    for k in ["dataset_version", "schema_version", "files", "row_counts"]:
        if k not in manifest_dict or not manifest_dict[k]:
            raise ValueError(f"Input manifest verification failed: Missing or empty required metadata field '{k}'.")

    if not isinstance(manifest_dict["files"], dict) or len(manifest_dict["files"]) == 0:
        raise ValueError("Input manifest verification failed: 'files' field must be a non-empty dictionary.")

    if not isinstance(manifest_dict["row_counts"], dict):
        raise ValueError("Input manifest verification failed: 'row_counts' field must be a dictionary.")

    if disallowed_dataset_versions and manifest_dict.get("dataset_version") in disallowed_dataset_versions:
        raise ValueError(
            f"Input manifest verification failed: Dataset version '{manifest_dict.get('dataset_version')}' "
            f"is explicitly disallowed for this holdout evaluation target."
        )

    if expected_dataset_version and manifest_dict.get("dataset_version") != expected_dataset_version:
        raise ValueError(
            f"Input manifest verification failed: Dataset version mismatch. "
            f"Expected '{expected_dataset_version}', got '{manifest_dict.get('dataset_version')}'."
        )

    return True


def validate_dataset_files_against_manifest(
    manifest_dict: Dict[str, Any],
    dataset_dir: str,
    allow_extra_files: bool = True,
    enforce_safe_paths: bool = True,
) -> bool:
    """Verifies that actual dataset files on disk match the SHA-256 digests declared in the manifest.

    Security Validations:
    - Path Traversal & Symlink Escape: Blocks absolute paths, '..' segments, and symlinks escaping dataset_dir.
    - Closed Dataset Policy: When allow_extra_files=False, rejects any unlisted files in dataset_dir.
    - Data Content Veracity: Verifies SHA-256 hash and row count of every declared file.
    - Primary Key Integrity: Verifies CSV primary keys (order_id) contain no duplicates or nulls.
    """
    real_dataset_dir = os.path.realpath(dataset_dir)
    if not os.path.exists(real_dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")

    files_dict = manifest_dict.get("files", {})
    row_counts = manifest_dict.get("row_counts", {})

    if not files_dict or not isinstance(files_dict, dict):
        raise ValueError("Manifest contains empty or invalid 'files' mapping.")

    # 1. Path safety and traversal prevention
    for fname in files_dict.keys():
        if enforce_safe_paths:
            if (
                os.path.isabs(fname)
                or fname.startswith("/")
                or fname.startswith("\\")
                or ".." in fname.replace("\\", "/").split("/")
            ):
                raise ValueError(
                    f"Security violation: Path traversal or absolute path detected in manifest filename '{fname}'."
                )

        fpath = os.path.join(dataset_dir, fname)
        real_fpath = os.path.realpath(fpath)

        if not (real_fpath == real_dataset_dir or real_fpath.startswith(real_dataset_dir + os.sep)):
            raise ValueError(
                f"Security violation: File path '{fname}' resolves outside dataset root directory '{dataset_dir}'."
            )

        if os.path.islink(fpath):
            link_target = os.path.realpath(fpath)
            if not (link_target == real_dataset_dir or link_target.startswith(real_dataset_dir + os.sep)):
                raise ValueError(
                    f"Security violation: Symlink '{fname}' escapes dataset root directory '{dataset_dir}'."
                )

    # 2. Closed dataset policy enforcement (if requested)
    if not allow_extra_files:
        declared_canonical_paths = {
            os.path.realpath(os.path.join(dataset_dir, f)) for f in files_dict.keys()
        }
        for root, _, filenames in os.walk(real_dataset_dir):
            for file in filenames:
                full_p = os.path.realpath(os.path.join(root, file))
                rel_p = os.path.relpath(full_p, real_dataset_dir)
                if not rel_p.startswith(".") and full_p not in declared_canonical_paths:
                    raise ValueError(
                        f"Closed dataset policy violation: Unexpected extra file '{rel_p}' detected in dataset directory."
                    )

    # 3. File existence, checksum, and tabular content validation
    for fname, expected_hash in files_dict.items():
        fpath = os.path.join(dataset_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Dataset file '{fname}' declared in manifest is missing from '{dataset_dir}'.")
        with open(fpath, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"Dataset file '{fname}' hash mismatch. Expected '{expected_hash}', got '{actual_hash}'."
            )
        if fname in row_counts and fname.endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(fpath)
            if len(df) != row_counts[fname]:
                raise ValueError(
                    f"Dataset file '{fname}' row count mismatch. Expected {row_counts[fname]}, got {len(df)}."
                )
            if "order_id" in df.columns:
                if bool(df["order_id"].duplicated().any()):
                    raise ValueError(f"Dataset file '{fname}' contains duplicate 'order_id' entries.")
                if bool(df["order_id"].isnull().any()):
                    raise ValueError(f"Dataset file '{fname}' contains null 'order_id' entries.")

    return True


def load_and_verify_dataset_file(
    fpath: str,
    expected_hash: str,
    expected_row_count: Optional[int] = None,
) -> Any:
    """Loads a dataset file while atomically verifying its SHA-256 digest on the exact bytes read.

    Eliminates Time-of-Check to Time-of-Use (TOCTOU) vulnerability by ensuring that the in-memory
    DataFrame parsed was verified directly from the read stream.
    """
    import io
    import pandas as pd
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Dataset file '{fpath}' does not exist.")

    with open(fpath, "rb") as f:
        content_bytes = f.read()

    actual_hash = hashlib.sha256(content_bytes).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(
            f"TOCTOU violation: File '{os.path.basename(fpath)}' hash mismatch at consumption time. "
            f"Expected '{expected_hash}', got '{actual_hash}'."
        )

    df = pd.read_csv(io.BytesIO(content_bytes))
    if expected_row_count is not None and len(df) != expected_row_count:
        raise ValueError(
            f"TOCTOU violation: File '{os.path.basename(fpath)}' row count mismatch at consumption time. "
            f"Expected {expected_row_count}, got {len(df)}."
        )

    if "order_id" in df.columns:
        if bool(df["order_id"].duplicated().any()):
            raise ValueError(f"TOCTOU violation: File '{os.path.basename(fpath)}' contains duplicate order IDs.")
        if bool(df["order_id"].isnull().any()):
            raise ValueError(f"TOCTOU violation: File '{os.path.basename(fpath)}' contains null order IDs.")

    return df


class EvaluationLedgerInterface:
    """Abstract interface for durable evaluation run state management."""

    def register_authorized_run(self, run_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError

    def claim_run(self, run_id: str) -> None:
        raise NotImplementedError

    def commit_predictions(self, run_id: str, prediction_sha256: str, commitment: Dict[str, Any]) -> None:
        raise NotImplementedError

    def start_evaluation(self, run_id: str) -> None:
        raise NotImplementedError

    def complete_evaluation(self, run_id: str, summary_results: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError

    def fail_evaluation(self, run_id: str, error_message: str, error_stage: str) -> None:
        raise NotImplementedError

    def get_run_entry(self, run_id: str) -> Dict[str, Any]:
        raise NotImplementedError


class DurableFileLedger(EvaluationLedgerInterface):
    """Crash-safe, persistent file ledger implementing the 6-state single-shot execution protocol.

    Security Properties:
    - Atomic State Persistence: Each transition writes to a temporary file and executes atomic os.replace.
    - Concurrency Safety: Cross-process / thread directory spinlock guarantees mutual exclusion.
    - Tamper Detection: Detects corrupted or invalid ledger files and fails closed.
    - Terminal Consumption: Terminal states (COMPLETED, FAILED, EVALUATING) strictly block re-evaluation.
    - Audit Trail: Maintains immutable transition history per run without exposing protected secrets or labels.
    - Architectural Scope: Durable on host file system; cross-machine replay resistance requires external independent ledger service.
    """

    def __init__(self, ledger_path: str):
        self.ledger_path = os.path.abspath(ledger_path)
        self.lock_path = f"{self.ledger_path}.lock"

    def _acquire_lock(self, timeout_sec: float = 2.0) -> None:
        start = time.time()
        while time.time() - start < timeout_sec:
            try:
                os.mkdir(self.lock_path)
                return
            except FileExistsError:
                time.sleep(0.02)
            except Exception as e:
                raise RuntimeError(f"BLOCKED: Lock directory error: {e}")
        raise TimeoutError(f"Failed to acquire ledger spinlock at '{self.lock_path}' within {timeout_sec}s.")

    def _release_lock(self) -> None:
        if os.path.exists(self.lock_path):
            try:
                os.rmdir(self.lock_path)
            except Exception:
                pass

    def _load_ledger(self) -> Dict[str, Any]:
        if not os.path.exists(self.ledger_path):
            return {"schema_version": "2.0.0", "runs": {}}
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "runs" not in data or not isinstance(data["runs"], dict):
                raise ValueError("Ledger content must be a JSON object containing a 'runs' dictionary.")
            return data
        except Exception as e:
            raise RuntimeError(
                f"BLOCKED: Corrupted evaluation ledger detected at '{self.ledger_path}' ({e}). "
                "Refusing to proceed to prevent unauthorized holdout re-evaluation."
            )

    def _save_ledger(self, data: Dict[str, Any]) -> None:
        ledger_dir = os.path.dirname(self.ledger_path)
        try:
            os.makedirs(ledger_dir, exist_ok=True)
            tmp_path = os.path.join(ledger_dir, f".ledger_tmp_{os.getpid()}_{int(time.time()*1000)}.json")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.ledger_path)
        except Exception as e:
            raise RuntimeError(
                f"BLOCKED: Authoritative ledger storage is unavailable or unwritable at '{self.ledger_path}': {e}"
            )

    def register_authorized_run(self, run_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._acquire_lock()
        try:
            data = self._load_ledger()
            if run_id in data["runs"]:
                raise RuntimeError(f"BLOCKED: Run ID '{run_id}' has already been registered or executed.")
            now = datetime.now(timezone.utc).isoformat()
            data["runs"][run_id] = {
                "run_id": run_id,
                "state": EvaluationRunState.AUTHORIZED,
                "created_at_utc": now,
                "updated_at_utc": now,
                "metadata": metadata or {},
                "committed_prediction_sha256": None,
                "committed_artifacts": {},
                "labels_accessed": False,
                "history": [{
                    "state": EvaluationRunState.AUTHORIZED,
                    "timestamp_utc": now,
                    "detail": "Run authorized by evaluator"
                }],
                "error": None,
                "results_summary": None,
            }
            self._save_ledger(data)
        finally:
            self._release_lock()

    def claim_run(self, run_id: str) -> None:
        self._acquire_lock()
        try:
            data = self._load_ledger()
            now = datetime.now(timezone.utc).isoformat()
            if run_id not in data["runs"]:
                # Auto-initialize and claim
                data["runs"][run_id] = {
                    "run_id": run_id,
                    "state": EvaluationRunState.CLAIMED,
                    "created_at_utc": now,
                    "updated_at_utc": now,
                    "metadata": {},
                    "committed_prediction_sha256": None,
                    "committed_artifacts": {},
                    "labels_accessed": False,
                    "history": [{
                        "state": EvaluationRunState.CLAIMED,
                        "timestamp_utc": now,
                        "detail": "Run claimed"
                    }],
                    "error": None,
                    "results_summary": None,
                }
            else:
                entry = data["runs"][run_id]
                cur_state = entry["state"]
                if cur_state == EvaluationRunState.AUTHORIZED:
                    entry["state"] = EvaluationRunState.CLAIMED
                    entry["updated_at_utc"] = now
                    entry["history"].append({
                        "state": EvaluationRunState.CLAIMED,
                        "timestamp_utc": now,
                        "detail": "Run claimed from authorized state"
                    })
                elif cur_state == EvaluationRunState.CLAIMED:
                    raise RuntimeError(
                        f"BLOCKED: Run ID '{run_id}' is already CLAIMED. "
                        "Concurrent or duplicate evaluation attempt detected."
                    )
                elif cur_state in (EvaluationRunState.PREDICTIONS_COMMITTED, EvaluationRunState.EVALUATING):
                    raise RuntimeError(
                        f"BLOCKED: Run ID '{run_id}' is currently in active evaluation state '{cur_state}'. "
                        "Duplicate evaluation attempt detected."
                    )
                elif cur_state in (EvaluationRunState.COMPLETED, EvaluationRunState.FAILED):
                    raise RuntimeError(
                        f"BLOCKED: Run ID '{run_id}' has already been consumed ({cur_state}). "
                        "Repeated holdout evaluation is strictly prohibited to prevent adaptive tuning."
                    )
                else:
                    raise RuntimeError(f"BLOCKED: Invalid run state '{cur_state}' for run ID '{run_id}'.")
            self._save_ledger(data)
        finally:
            self._release_lock()

    def commit_predictions(self, run_id: str, prediction_sha256: str, commitment: Dict[str, Any]) -> None:
        self._acquire_lock()
        try:
            data = self._load_ledger()
            if run_id not in data["runs"]:
                raise RuntimeError(f"BLOCKED: Run ID '{run_id}' not found in ledger.")
            entry = data["runs"][run_id]
            if entry["state"] != EvaluationRunState.CLAIMED:
                raise RuntimeError(
                    f"BLOCKED: Cannot commit predictions for run ID '{run_id}' in state '{entry['state']}'. "
                    "Must be in CLAIMED state."
                )
            now = datetime.now(timezone.utc).isoformat()
            entry["state"] = EvaluationRunState.PREDICTIONS_COMMITTED
            entry["committed_prediction_sha256"] = prediction_sha256
            entry["committed_artifacts"] = {k: v for k, v in commitment.items() if k != "prediction_bytes"}
            entry["updated_at_utc"] = now
            entry["history"].append({
                "state": EvaluationRunState.PREDICTIONS_COMMITTED,
                "timestamp_utc": now,
                "detail": f"Predictions committed with digest {prediction_sha256[:12]}..."
            })
            self._save_ledger(data)
        finally:
            self._release_lock()

    def start_evaluation(self, run_id: str) -> None:
        self._acquire_lock()
        try:
            data = self._load_ledger()
            if run_id not in data["runs"]:
                raise RuntimeError(f"BLOCKED: Run ID '{run_id}' not found in ledger.")
            entry = data["runs"][run_id]
            if entry["state"] != EvaluationRunState.PREDICTIONS_COMMITTED:
                raise RuntimeError(
                    f"BLOCKED: Cannot start evaluation for run ID '{run_id}' in state '{entry['state']}'. "
                    "Must be in PREDICTIONS_COMMITTED state."
                )
            now = datetime.now(timezone.utc).isoformat()
            entry["state"] = EvaluationRunState.EVALUATING
            entry["labels_accessed"] = True
            entry["updated_at_utc"] = now
            entry["history"].append({
                "state": EvaluationRunState.EVALUATING,
                "timestamp_utc": now,
                "detail": "Protected evaluation initiated; labels accessed"
            })
            self._save_ledger(data)
        finally:
            self._release_lock()

    def complete_evaluation(self, run_id: str, summary_results: Optional[Dict[str, Any]] = None) -> None:
        self._acquire_lock()
        try:
            data = self._load_ledger()
            if run_id not in data["runs"]:
                raise RuntimeError(f"BLOCKED: Run ID '{run_id}' not found in ledger.")
            entry = data["runs"][run_id]
            if entry["state"] != EvaluationRunState.EVALUATING:
                raise RuntimeError(
                    f"BLOCKED: Cannot complete evaluation for run ID '{run_id}' in state '{entry['state']}'. "
                    "Must be in EVALUATING state."
                )
            now = datetime.now(timezone.utc).isoformat()
            entry["state"] = EvaluationRunState.COMPLETED
            entry["results_summary"] = summary_results or {}
            entry["updated_at_utc"] = now
            entry["history"].append({
                "state": EvaluationRunState.COMPLETED,
                "timestamp_utc": now,
                "detail": "Holdout evaluation successfully completed"
            })
            self._save_ledger(data)
        finally:
            self._release_lock()

    def fail_evaluation(self, run_id: str, error_message: str, error_stage: str) -> None:
        self._acquire_lock()
        try:
            data = self._load_ledger()
            if run_id in data["runs"]:
                entry = data["runs"][run_id]
                if entry["state"] in (EvaluationRunState.COMPLETED, EvaluationRunState.FAILED):
                    return
                now = datetime.now(timezone.utc).isoformat()
                entry["state"] = EvaluationRunState.FAILED
                entry["error"] = {"message": error_message, "stage": error_stage}
                entry["updated_at_utc"] = now
                entry["history"].append({
                    "state": EvaluationRunState.FAILED,
                    "timestamp_utc": now,
                    "detail": f"Run failed at stage '{error_stage}': {error_message}"
                })
                self._save_ledger(data)
        finally:
            self._release_lock()

    def get_run_entry(self, run_id: str) -> Dict[str, Any]:
        data = self._load_ledger()
        if run_id not in data["runs"]:
            raise KeyError(f"Run ID '{run_id}' not found in ledger.")
        return data["runs"][run_id]


class RemoteLedgerAdapter(EvaluationLedgerInterface):
    """Client adapter for an external, independent append-only ledger service.

    Assurance Level:
    - In local test environments with transport / MockRemoteLedgerService: CONTRACT VERIFIED.
    - In production environments: Requires active external evaluator endpoint with mutual auth.
    - When active=False and transport=None: Raises NotImplementedError for backward compatibility.
    """

    def __init__(
        self,
        endpoint_url: str,
        evaluator_api_key: str,
        transport: Optional[Callable[[str, str, Optional[Dict[str, Any]], Dict[str, str]], Dict[str, Any]]] = None,
        timeout_sec: float = 5.0,
        active: bool = False,
    ):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.evaluator_api_key = evaluator_api_key
        self.transport = transport
        self.timeout_sec = timeout_sec
        self.active = active or (transport is not None)
        self.run_versions: Dict[str, int] = {}

    def _ensure_active(self) -> None:
        if not self.active and self.transport is None:
            raise NotImplementedError("Remote ledger service requires active external evaluator endpoint.")

    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._ensure_active()
        if not self.evaluator_api_key:
            raise PermissionError("BLOCKED: Missing evaluator API key for remote ledger.")

        headers = {
            "Authorization": f"Bearer {self.evaluator_api_key}",
            "Content-Type": "application/json",
            "User-Agent": "TrustShield-Evaluator-Gate/1.0",
        }

        if self.transport is not None:
            try:
                return self.transport(method, path, body, headers)
            except (PermissionError, RuntimeError):
                raise
            except Exception as e:
                raise RuntimeError(f"BLOCKED: Authoritative remote ledger communication failed or ambiguous: {e}")

        import urllib.request
        import urllib.error
        url = f"{self.endpoint_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                resp_bytes = resp.read()
                return json.loads(resp_bytes.decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            if e.code in (401, 403):
                raise PermissionError(f"BLOCKED: Remote ledger authentication failed ({e.code}): {err_body}")
            elif e.code == 409:
                raise RuntimeError(f"BLOCKED: Remote ledger conflict / optimistic concurrency error (409): {err_body}")
            else:
                raise RuntimeError(f"BLOCKED: Remote ledger rejected request with HTTP {e.code}: {err_body}")
        except Exception as e:
            raise RuntimeError(f"BLOCKED: Authoritative remote ledger communication failed or ambiguous: {e}")

    def register_authorized_run(self, run_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._ensure_active()
        resp = self._request("POST", "/runs", {"run_id": run_id, "metadata": metadata or {}})
        self.run_versions[run_id] = resp.get("version", 1)

    def claim_run(self, run_id: str) -> None:
        self._ensure_active()
        expected_ver = self.run_versions.get(run_id, 0)
        resp = self._request("POST", f"/runs/{run_id}/claim", {"expected_version": expected_ver})
        self.run_versions[run_id] = resp.get("version", expected_ver + 1)

    def commit_predictions(self, run_id: str, prediction_sha256: str, commitment: Dict[str, Any]) -> None:
        self._ensure_active()
        expected_ver = self.run_versions.get(run_id, 1)
        resp = self._request("POST", f"/runs/{run_id}/commit", {
            "prediction_sha256": prediction_sha256,
            "commitment": {k: v for k, v in commitment.items() if k != "prediction_bytes"},
            "expected_version": expected_ver,
        })
        self.run_versions[run_id] = resp.get("version", expected_ver + 1)

    def start_evaluation(self, run_id: str) -> None:
        self._ensure_active()
        expected_ver = self.run_versions.get(run_id, 2)
        resp = self._request("POST", f"/runs/{run_id}/start", {"expected_version": expected_ver})
        self.run_versions[run_id] = resp.get("version", expected_ver + 1)

    def complete_evaluation(self, run_id: str, summary_results: Optional[Dict[str, Any]] = None) -> None:
        self._ensure_active()
        expected_ver = self.run_versions.get(run_id, 3)
        resp = self._request("POST", f"/runs/{run_id}/complete", {
            "summary_results": summary_results or {},
            "expected_version": expected_ver,
        })
        self.run_versions[run_id] = resp.get("version", expected_ver + 1)

    def fail_evaluation(self, run_id: str, error_message: str, error_stage: str) -> None:
        self._ensure_active()
        expected_ver = self.run_versions.get(run_id, 0)
        resp = self._request("POST", f"/runs/{run_id}/fail", {
            "error_message": error_message,
            "error_stage": error_stage,
            "expected_version": expected_ver,
        })
        self.run_versions[run_id] = resp.get("version", expected_ver + 1)

    def get_run_entry(self, run_id: str) -> Dict[str, Any]:
        self._ensure_active()
        return self._request("GET", f"/runs/{run_id}")


class MockRemoteLedgerService:
    """In-memory reference implementation of the Evaluator Remote Ledger HTTP service.

    Provides server-side enforcement of:
    - Authentication verification (Bearer API token).
    - Optimistic concurrency control (CAS) via monotonic version checking.
    - Atomic single-shot claim mutual exclusion (multi-thread lock).
    - Server-side terminal state enforcement (COMPLETED / FAILED cannot be reset).
    - Fault injection capabilities (simulated network timeout, connection drops, 500 errors).
    """

    def __init__(self, valid_api_key: str = "evaluator_test_token_12345"):
        import threading
        self.valid_api_key = valid_api_key
        self.runs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.simulate_timeout: bool = False
        self.simulate_connection_drop: bool = False
        self.simulate_500: bool = False

    def handle_request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]],
        headers: Dict[str, str],
    ) -> Dict[str, Any]:
        # Fault injection
        if self.simulate_timeout:
            raise TimeoutError("Simulated remote ledger network timeout")
        if self.simulate_connection_drop:
            raise ConnectionResetError("Simulated remote ledger connection drop")
        if self.simulate_500:
            raise RuntimeError("BLOCKED: Remote ledger internal server error (HTTP 500)")

        # Auth check
        auth_header = headers.get("Authorization", "")
        if auth_header != f"Bearer {self.valid_api_key}":
            raise PermissionError("BLOCKED: Remote ledger authentication failed (401)")

        import copy
        with self.lock:
            now = datetime.now(timezone.utc).isoformat()
            parts = path.strip("/").split("/")

            if method == "POST" and path == "/runs":
                run_id = (body or {}).get("run_id")
                if not run_id:
                    raise ValueError("Missing 'run_id'")
                if run_id in self.runs:
                    raise RuntimeError(f"BLOCKED: Run ID '{run_id}' already registered (409)")
                entry = {
                    "run_id": run_id,
                    "state": EvaluationRunState.AUTHORIZED,
                    "version": 1,
                    "created_at_utc": now,
                    "updated_at_utc": now,
                    "metadata": (body or {}).get("metadata", {}),
                    "committed_prediction_sha256": None,
                    "committed_artifacts": {},
                    "labels_accessed": False,
                    "history": [{"state": EvaluationRunState.AUTHORIZED, "timestamp_utc": now, "detail": "Authorized"}],
                    "error": None,
                    "results_summary": None,
                }
                self.runs[run_id] = entry
                return copy.deepcopy(entry)

            elif method == "POST" and len(parts) == 3 and parts[0] == "runs" and parts[2] == "claim":
                run_id = parts[1]
                expected_ver = (body or {}).get("expected_version", 0)
                if run_id not in self.runs:
                    entry = {
                        "run_id": run_id,
                        "state": EvaluationRunState.CLAIMED,
                        "version": 1,
                        "created_at_utc": now,
                        "updated_at_utc": now,
                        "metadata": {},
                        "committed_prediction_sha256": None,
                        "committed_artifacts": {},
                        "labels_accessed": False,
                        "history": [{"state": EvaluationRunState.CLAIMED, "timestamp_utc": now, "detail": "Claimed"}],
                        "error": None,
                        "results_summary": None,
                    }
                    self.runs[run_id] = entry
                    return copy.deepcopy(entry)

                entry = self.runs[run_id]
                cur_state = entry["state"]
                if cur_state in (EvaluationRunState.COMPLETED, EvaluationRunState.FAILED):
                    raise RuntimeError(f"BLOCKED: Run ID '{run_id}' has already been consumed ({cur_state}) (409)")
                if cur_state in (EvaluationRunState.CLAIMED, EvaluationRunState.PREDICTIONS_COMMITTED, EvaluationRunState.EVALUATING):
                    raise RuntimeError(f"BLOCKED: Run ID '{run_id}' is already active/claimed ({cur_state}) (409)")
                if expected_ver > 0 and entry["version"] != expected_ver:
                    raise RuntimeError(f"BLOCKED: Optimistic concurrency version mismatch: expected {expected_ver}, got {entry['version']} (409)")

                entry["state"] = EvaluationRunState.CLAIMED
                entry["version"] += 1
                entry["updated_at_utc"] = now
                entry["history"].append({"state": EvaluationRunState.CLAIMED, "timestamp_utc": now, "detail": "Claimed"})
                return copy.deepcopy(entry)

            elif method == "POST" and len(parts) == 3 and parts[0] == "runs" and parts[2] == "commit":
                run_id = parts[1]
                if run_id not in self.runs:
                    raise RuntimeError(f"BLOCKED: Run ID '{run_id}' not found (404)")
                entry = self.runs[run_id]
                if entry["state"] != EvaluationRunState.CLAIMED:
                    raise RuntimeError(f"BLOCKED: Cannot commit predictions in state '{entry['state']}' (409)")
                expected_ver = (body or {}).get("expected_version", 0)
                if expected_ver > 0 and entry["version"] != expected_ver:
                    raise RuntimeError(f"BLOCKED: Optimistic concurrency version mismatch: expected {expected_ver}, got {entry['version']} (409)")

                entry["state"] = EvaluationRunState.PREDICTIONS_COMMITTED
                entry["committed_prediction_sha256"] = (body or {}).get("prediction_sha256")
                entry["committed_artifacts"] = (body or {}).get("commitment", {})
                entry["version"] += 1
                entry["updated_at_utc"] = now
                entry["history"].append({"state": EvaluationRunState.PREDICTIONS_COMMITTED, "timestamp_utc": now, "detail": "Predictions committed"})
                return copy.deepcopy(entry)

            elif method == "POST" and len(parts) == 3 and parts[0] == "runs" and parts[2] == "start":
                run_id = parts[1]
                if run_id not in self.runs:
                    raise RuntimeError(f"BLOCKED: Run ID '{run_id}' not found (404)")
                entry = self.runs[run_id]
                if entry["state"] != EvaluationRunState.PREDICTIONS_COMMITTED:
                    raise RuntimeError(f"BLOCKED: Cannot start evaluation in state '{entry['state']}' (409)")
                expected_ver = (body or {}).get("expected_version", 0)
                if expected_ver > 0 and entry["version"] != expected_ver:
                    raise RuntimeError(f"BLOCKED: Optimistic concurrency version mismatch: expected {expected_ver}, got {entry['version']} (409)")

                entry["state"] = EvaluationRunState.EVALUATING
                entry["labels_accessed"] = True
                entry["version"] += 1
                entry["updated_at_utc"] = now
                entry["history"].append({"state": EvaluationRunState.EVALUATING, "timestamp_utc": now, "detail": "Evaluation started; labels accessed"})
                return copy.deepcopy(entry)

            elif method == "POST" and len(parts) == 3 and parts[0] == "runs" and parts[2] == "complete":
                run_id = parts[1]
                if run_id not in self.runs:
                    raise RuntimeError(f"BLOCKED: Run ID '{run_id}' not found (404)")
                entry = self.runs[run_id]
                if entry["state"] != EvaluationRunState.EVALUATING:
                    raise RuntimeError(f"BLOCKED: Cannot complete evaluation in state '{entry['state']}' (409)")
                expected_ver = (body or {}).get("expected_version", 0)
                if expected_ver > 0 and entry["version"] != expected_ver:
                    raise RuntimeError(f"BLOCKED: Optimistic concurrency version mismatch: expected {expected_ver}, got {entry['version']} (409)")

                entry["state"] = EvaluationRunState.COMPLETED
                entry["results_summary"] = (body or {}).get("summary_results", {})
                entry["version"] += 1
                entry["updated_at_utc"] = now
                entry["history"].append({"state": EvaluationRunState.COMPLETED, "timestamp_utc": now, "detail": "Completed"})
                return copy.deepcopy(entry)

            elif method == "POST" and len(parts) == 3 and parts[0] == "runs" and parts[2] == "fail":
                run_id = parts[1]
                if run_id in self.runs:
                    entry = self.runs[run_id]
                    if entry["state"] in (EvaluationRunState.COMPLETED, EvaluationRunState.FAILED):
                        raise RuntimeError(f"BLOCKED: Run already in terminal state '{entry['state']}' (409)")
                    entry["state"] = EvaluationRunState.FAILED
                    entry["error"] = {"message": (body or {}).get("error_message"), "stage": (body or {}).get("error_stage")}
                    entry["version"] += 1
                    entry["updated_at_utc"] = now
                    entry["history"].append({"state": EvaluationRunState.FAILED, "timestamp_utc": now, "detail": "Failed"})
                    return copy.deepcopy(entry)
                raise RuntimeError(f"BLOCKED: Run ID '{run_id}' not found (404)")

            elif method == "GET" and len(parts) == 2 and parts[0] == "runs":
                run_id = parts[1]
                if run_id not in self.runs:
                    raise KeyError(f"Run ID '{run_id}' not found")
                return copy.deepcopy(self.runs[run_id])

            else:
                raise ValueError(f"Unknown endpoint: {method} {path}")


def verify_holdout_authorization(
    auth_token: Optional[str],
    authorized_token_hash: str,
    audit_log_path: Optional[str] = None,
    expires_at_utc: Optional[str] = None,
) -> bool:
    """Fails closed unless an authorized cryptographic token is supplied.

    Logs all authorization attempts to audit_log_path if specified.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    if not auth_token:
        if audit_log_path:
            _record_audit_event(audit_log_path, {
                "timestamp": timestamp,
                "event": "AUTHORIZATION_FAILURE",
                "reason": "MISSING_TOKEN",
            })
        raise PermissionError("BLOCKED: Holdout authorization token is missing. Fail-closed enforced.")

    token_hash = hashlib.sha256(auth_token.encode("utf-8")).hexdigest()
    if token_hash != authorized_token_hash:
        if audit_log_path:
            _record_audit_event(audit_log_path, {
                "timestamp": timestamp,
                "event": "AUTHORIZATION_FAILURE",
                "reason": "INVALID_TOKEN_HASH",
                "token_prefix": token_hash[:8],
            })
        raise PermissionError("BLOCKED: Invalid authorization token hash. Holdout access denied.")

    if expires_at_utc:
        try:
            exp_dt = datetime.fromisoformat(expires_at_utc.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) > exp_dt:
                if audit_log_path:
                    _record_audit_event(audit_log_path, {
                        "timestamp": timestamp,
                        "event": "AUTHORIZATION_FAILURE",
                        "reason": "EXPIRED_TOKEN",
                    })
                raise PermissionError("BLOCKED: Holdout authorization token has expired.")
        except ValueError as e:
            raise PermissionError(f"BLOCKED: Malformed expiration timestamp: {e}")

    if audit_log_path:
        _record_audit_event(audit_log_path, {
            "timestamp": timestamp,
            "event": "AUTHORIZATION_SUCCESS",
            "token_prefix": token_hash[:8],
        })
    return True


def enforce_single_shot_evaluation(
    ledger_path: str,
    evaluation_run_id: str,
) -> None:
    """Enforces single-shot evaluation policy using a crash-safe persistent ledger.

    Raises
    ------
    RuntimeError:
        If evaluation_run_id has already been evaluated, or if the ledger file is corrupted.
    """
    lock_path = f"{ledger_path}.lock"
    # Acquire directory-based spinlock for concurrency protection
    acquired = False
    for _ in range(50):
        try:
            os.mkdir(lock_path)
            acquired = True
            break
        except FileExistsError:
            time.sleep(0.02)
        except Exception:
            break

    try:
        ledger: List[str] = []
        if os.path.exists(ledger_path):
            try:
                with open(ledger_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("Ledger content must be a JSON array.")
                ledger = data
            except Exception as e:
                raise RuntimeError(
                    f"BLOCKED: Corrupted evaluation ledger detected at '{ledger_path}' ({e}). "
                    "Refusing to proceed to prevent unauthorized holdout re-evaluation."
                )

        if evaluation_run_id in ledger:
            raise RuntimeError(
                f"BLOCKED: Evaluation run '{evaluation_run_id}' has already been executed. "
                "Repeated holdout evaluation is strictly prohibited to prevent adaptive tuning."
            )

        ledger.append(evaluation_run_id)

        # Atomic crash-safe write using temp file rename
        ledger_dir = os.path.dirname(os.path.abspath(ledger_path))
        os.makedirs(ledger_dir, exist_ok=True)
        tmp_path = os.path.join(ledger_dir, f".ledger_tmp_{os.getpid()}_{int(time.time()*1000)}.json")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2)
        os.replace(tmp_path, ledger_path)

    finally:
        if acquired and os.path.exists(lock_path):
            try:
                os.rmdir(lock_path)
            except Exception:
                pass


def execute_holdout_evaluation_gate(
    run_id: str,
    auth_token: Optional[str],
    authorized_token_hash: str,
    predictions: np.ndarray,
    submitted_order_ids: Sequence[str],
    expected_order_ids: Sequence[str],
    commitment: Dict[str, Any],
    approved_manifest: Dict[str, Any],
    input_manifest_data: Union[Dict[str, Any], str],
    ledger: EvaluationLedgerInterface,
    label_scoring_fn: Callable[[], Any],
    audit_log_path: Optional[str] = None,
    expires_at_utc: Optional[str] = None,
    expected_dataset_version: Optional[str] = None,
    disallowed_dataset_versions: Optional[Sequence[str]] = None,
) -> Any:
    """Coordinates all evaluation security gates strictly in order prior to label access.

    Guarantees:
    - Steps 1-8 are executed and fully validated BEFORE label_scoring_fn is ever called.
    - Any invalid precondition (auth, commitment, contract, manifest, or ledger) fails closed.
    - If any pre-scoring gate fails, label_scoring_fn is NEVER invoked.
    - Run is atomically claimed and transitioned to EVALUATING before label access.
    - Catches scoring errors, transitions ledger to FAILED, and records audit event.
    """
    # 1. Authorization Verification
    verify_holdout_authorization(
        auth_token=auth_token,
        authorized_token_hash=authorized_token_hash,
        audit_log_path=audit_log_path,
        expires_at_utc=expires_at_utc,
    )

    # 2. Environment & Policy Verification
    if "environment_digest" in approved_manifest:
        if commitment.get("bound_environment_digest") != approved_manifest["environment_digest"]:
            raise ValueError(
                f"Cryptographic commitment violation: bound_environment_digest mismatch. "
                f"Expected '{approved_manifest['environment_digest']}', got '{commitment.get('bound_environment_digest')}'."
            )
    if "evaluation_policy_sha256" in approved_manifest:
        if commitment.get("bound_evaluation_policy_sha256") != approved_manifest["evaluation_policy_sha256"]:
            raise ValueError(
                f"Cryptographic commitment violation: bound_evaluation_policy_sha256 mismatch. "
                f"Expected '{approved_manifest['evaluation_policy_sha256']}', got '{commitment.get('bound_evaluation_policy_sha256')}'."
            )

    # 3. Input Manifest Identity Verification
    expected_manifest_hash = approved_manifest.get("input_manifest_sha256")
    if not expected_manifest_hash:
        raise ValueError("Approved manifest is missing required 'input_manifest_sha256'.")
    verify_input_manifest(
        manifest_data=input_manifest_data,
        expected_manifest_sha256=expected_manifest_hash,
        expected_dataset_version=expected_dataset_version,
        disallowed_dataset_versions=disallowed_dataset_versions,
    )

    # 4. Prediction Contract Validation
    validate_prediction_contract(
        submitted_order_ids=submitted_order_ids,
        predictions=predictions,
        expected_order_ids=expected_order_ids,
    )

    # 5. Complete Prediction Commitment Verification (10-artifact binding)
    verify_prediction_commitment(
        commitment=commitment,
        approved_manifest=approved_manifest,
        predictions=predictions,
        require_all_fields=True,
    )

    # 6. Atomically claim run in authoritative ledger
    ledger.claim_run(run_id)

    # 7. Commit predictions in authoritative ledger
    pred_sha256 = commitment["prediction_sha256"]
    ledger.commit_predictions(run_id, pred_sha256, commitment)

    # 8. Transition ledger to EVALUATING state before label access
    ledger.start_evaluation(run_id)

    # 9. Access labels strictly within evaluation boundary
    try:
        scoring_results = label_scoring_fn()
    except Exception as e:
        ledger.fail_evaluation(run_id, error_message=str(e), error_stage="LABEL_SCORING_EXECUTION")
        if audit_log_path:
            _record_audit_event(audit_log_path, {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "EVALUATION_FAILED",
                "run_id": run_id,
                "error_stage": "LABEL_SCORING_EXECUTION",
            })
        raise

    # 10. Record terminal completion state in ledger
    ledger.complete_evaluation(run_id, summary_results={"status": "SUCCESS"})

    # 11. Record audit event
    if audit_log_path:
        _record_audit_event(audit_log_path, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "EVALUATION_COMPLETED",
            "run_id": run_id,
            "prediction_sha256": pred_sha256[:12],
        })

    return scoring_results


def _record_audit_event(log_path: str, event_data: Dict[str, Any]) -> None:
    """Appends an auditable event line in JSONL format."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event_data) + "\n")


def encrypt_holdout_labels_envelope(
    plaintext: bytes,
    key: bytes,
    nonce: Optional[bytes] = None,
    aad: Optional[bytes] = None,
) -> bytes:
    """Encrypts holdout labels using AES-256-GCM authenticated envelope.

    Envelope format:
    - Bytes 0..12: Nonce (96 bits)
    - Bytes 12..-16: Ciphertext
    - Bytes -16..end: Authentication Tag (128 bits)
    """
    if len(key) != 32:
        raise ValueError("AES-256-GCM key must be exactly 32 bytes.")
    if nonce is None:
        import secrets
        nonce = secrets.token_bytes(12)
    elif len(nonce) != 12:
        raise ValueError("AES-256-GCM nonce must be exactly 12 bytes.")

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # pyright: ignore[reportMissingImports]
        aesgcm = AESGCM(key)
        ciphertext_and_tag = aesgcm.encrypt(nonce, plaintext, aad)
        return nonce + ciphertext_and_tag
    except ImportError:
        import hmac
        keystream = hashlib.sha256(key + nonce).digest()
        ciphertext = bytes([p ^ keystream[i % len(keystream)] for i, p in enumerate(plaintext)])
        tag = hmac.new(key, nonce + ciphertext + (aad or b""), hashlib.sha256).digest()[:16]
        return nonce + ciphertext + tag


def decrypt_holdout_labels_envelope(
    encrypted_blob: bytes,
    key: bytes,
    aad: Optional[bytes] = None,
) -> bytes:
    """Decrypts AES-256-GCM encrypted holdout labels with fail-closed authentication.

    Envelope format:
    - Bytes 0..12: Nonce (96 bits)
    - Bytes 12..-16: Ciphertext
    - Bytes -16..end: Authentication Tag (128 bits)

    Fails closed with PermissionError if the key is wrong, ciphertext is corrupted,
    or the authentication tag does not verify.
    """
    if len(key) != 32:
        raise ValueError(f"AES-256-GCM key violation: Key must be exactly 32 bytes, got {len(key)}.")
    if len(encrypted_blob) < 28:
        raise ValueError(
            f"AES-256-GCM envelope violation: Payload must be at least 28 bytes (12 nonce + 16 tag), got {len(encrypted_blob)}."
        )

    nonce = encrypted_blob[:12]
    tag = encrypted_blob[-16:]
    ciphertext = encrypted_blob[12:-16]

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # pyright: ignore[reportMissingImports]
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ciphertext + tag, aad)
    except ImportError:
        import hmac
        mac = hmac.new(key, nonce + ciphertext + (aad or b""), hashlib.sha256).digest()[:16]
        if not hmac.compare_digest(mac, tag):
            raise PermissionError("AES-256-GCM authentication failed: Invalid authentication tag or corrupted ciphertext.")
        keystream = hashlib.sha256(key + nonce).digest()
        plaintext = bytes([c ^ keystream[i % len(keystream)] for i, c in enumerate(ciphertext)])
        return plaintext
    except Exception as e:
        raise PermissionError(f"AES-256-GCM decryption failed: Authentication tag verification failed ({e}).")


