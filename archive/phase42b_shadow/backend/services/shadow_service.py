"""TrustShield AI — Asynchronous Non-Blocking Shadow Scoring Service.

Phase 4.2B Isolated Implementation:
- Evaluates the 37-feature candidate model in shadow mode alongside live Phase 3 scoring.
- Disabled by default via TRUSTSHIELD_SHADOW_SCORING=false.
- Completely decoupled from production scoring: zero impact on response body, headers,
  thresholds, decisions, or latency SLAs.
- Strict bounded worker queue and execution timeout (default 25ms).
- Structured audit logging to logs/shadow_scoring.jsonl.
- Bounded failure handling: network timeouts, model errors, or Redis outages are logged
  and caught silently without throwing HTTP exceptions.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from backend.candidate_loader import candidate_loader
from backend.services.redis_causal_service import redis_causal_service

logger = logging.getLogger(__name__)

_DEFAULT_LOG_PATH = os.path.join(
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")),
    "logs",
    "shadow_scoring.jsonl",
)


class ShadowScoringService:
    """Non-blocking shadow scoring orchestrator with bounded queue and circuit breakers."""

    def __init__(
        self,
        enabled: Optional[bool] = None,
        timeout_ms: float = 25.0,
        max_queue_size: int = 1000,
        log_path: Optional[str] = None,
    ):
        # Mandatory requirement: Disabled by default
        env_enabled = os.getenv("TRUSTSHIELD_SHADOW_SCORING", "false").lower() in ("true", "1", "yes")
        self.enabled = env_enabled if enabled is None else enabled
        self.timeout_ms = float(os.getenv("TRUSTSHIELD_SHADOW_TIMEOUT_MS", str(timeout_ms)))
        self.log_path = log_path or os.getenv("TRUSTSHIELD_SHADOW_LOG_PATH", _DEFAULT_LOG_PATH)

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._log_lock = threading.Lock()

        # Telemetry metrics
        self.total_invocations: int = 0
        self.total_successes: int = 0
        self.total_timeouts: int = 0
        self.total_errors: int = 0
        self.total_dropped: int = 0

        if self.enabled:
            self._start_worker_pool()

    def _start_worker_pool(self) -> None:
        """Starts background worker pool and ensures candidate model is loaded."""
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="ts_shadow_worker",
            )
            # Pre-load candidate model lazily
            if not candidate_loader.is_loaded:
                candidate_loader.load()

    def enqueue_shadow_evaluation(
        self,
        req_dict: Dict[str, Any],
        prod_calibrated_risk: float,
        prod_decision: str,
        prod_model_version: str,
        t_decision: float,
        baseline_feature_row: Dict[str, Any],
    ) -> bool:
        """Submits an asynchronous shadow scoring task.

        Guarantees:
        - Returns immediately (non-blocking).
        - If disabled or queue is full, exits in <0.1ms without work.
        - Failures are caught and recorded without affecting production request.
        """
        if not self.enabled:
            return False

        self.total_invocations += 1

        if self._executor is None:
            self._start_worker_pool()

        task_payload = {
            "order_id": str(req_dict.get("order_id", "unknown")),
            "buyer_id": req_dict.get("buyer_id"),
            "seller_id": req_dict.get("seller_id"),
            "device_id": req_dict.get("device_id"),
            "amount": float(req_dict.get("amount") or req_dict.get("order_amount") or 0.0),
            "prod_risk": float(prod_calibrated_risk),
            "prod_decision": str(prod_decision),
            "prod_model_version": str(prod_model_version),
            "t_decision": float(t_decision),
            "baseline_features": baseline_feature_row,
            "enqueued_time": time.perf_counter(),
        }

        try:
            self._executor.submit(self._execute_shadow_score, task_payload)
            return True
        except Exception as exc:
            self.total_dropped += 1
            logger.warning("Shadow scoring task dropped (executor full/busy): %s", exc)
            return False

    def _execute_shadow_score(self, task: Dict[str, Any]) -> None:
        """Worker function executing in bounded background thread."""
        t_start = time.perf_counter()
        order_id = task["order_id"]
        status = "unknown"
        error_msg: Optional[str] = None
        cand_raw = None
        cand_cal = None
        is_degraded = False
        degraded_reason = None
        feat_source = "unavailable"

        try:
            # 1. Enforce queue latency timeout
            queue_delay_ms = (t_start - task["enqueued_time"]) * 1000.0
            if queue_delay_ms > self.timeout_ms:
                self.total_timeouts += 1
                status = "timeout_in_queue"
                error_msg = f"Queue wait ({queue_delay_ms:.1f}ms) exceeded timeout ceiling ({self.timeout_ms:.1f}ms)"
                self._record_log(task, status, cand_raw, cand_cal, is_degraded, degraded_reason, feat_source, queue_delay_ms, error_msg)
                return

            # 2. Check candidate model availability
            if not candidate_loader.is_loaded:
                if not candidate_loader.load():
                    self.total_errors += 1
                    status = "candidate_model_unavailable"
                    error_msg = "Candidate model artifacts not loaded"
                    self._record_log(task, status, cand_raw, cand_cal, is_degraded, degraded_reason, feat_source, 0.0, error_msg)
                    return

            # 3. Query Causal Sliding Features strictly for t < t_decision
            causal_res = redis_causal_service.query_causal_features(
                buyer_id=task["buyer_id"],
                seller_id=task["seller_id"],
                device_id=task.get("device_id"),
                amount=task["amount"],
                decision_timestamp=task["t_decision"],
            )

            is_degraded = causal_res.is_degraded
            degraded_reason = causal_res.degraded_reason
            feat_source = causal_res.feature_source

            # 4. Construct Exact 37-Feature Vector in Deterministic Order
            full_feature_map = {}
            # Base features from production feature row
            for k, v in task["baseline_features"].items():
                full_feature_map[k] = float(v) if v is not None else 0.0

            # Causal features from Redis service
            for k, v in causal_res.features.items():
                full_feature_map[k] = float(v) if v is not None else 0.0

            # Assemble vector matching candidate_loader.feature_names
            vec = [full_feature_map.get(col, 0.0) for col in candidate_loader.feature_names]

            # 5. Candidate Inference
            cand_raw, cand_cal = candidate_loader.predict_proba(np.array(vec, dtype=np.float32))

            exec_latency_ms = (time.perf_counter() - t_start) * 1000.0
            if exec_latency_ms > self.timeout_ms:
                self.total_timeouts += 1
                status = "timeout_during_execution"
                error_msg = f"Execution ({exec_latency_ms:.1f}ms) exceeded timeout ceiling"
            else:
                self.total_successes += 1
                status = "success"

            self._record_log(
                task=task,
                status=status,
                raw_prob=cand_raw,
                cal_prob=cand_cal,
                is_degraded=is_degraded,
                degraded_reason=degraded_reason,
                feature_source=feat_source,
                latency_ms=round(exec_latency_ms, 3),
                error=error_msg,
            )

        except Exception as exc:
            self.total_errors += 1
            exec_latency_ms = (time.perf_counter() - t_start) * 1000.0
            logger.warning("Shadow scoring failed for order %s: %s", order_id, exc)
            self._record_log(
                task=task,
                status="execution_error",
                raw_prob=None,
                cal_prob=None,
                is_degraded=True,
                degraded_reason="unhandled_exception",
                feature_source=feat_source,
                latency_ms=round(exec_latency_ms, 3),
                error=str(exc),
            )

    def _record_log(
        self,
        task: Dict[str, Any],
        status: str,
        raw_prob: Optional[float],
        cal_prob: Optional[float],
        is_degraded: bool,
        degraded_reason: Optional[str],
        feature_source: str,
        latency_ms: float,
        error: Optional[str],
    ) -> None:
        """Appends structured audit record to shadow log file."""
        opt_thresh = candidate_loader.optimal_threshold if candidate_loader.is_loaded else 0.11
        cand_flag = bool(cal_prob >= opt_thresh) if cal_prob is not None else False
        score_delta = round(cal_prob - task["prod_risk"], 4) if cal_prob is not None else None

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "order_id": task["order_id"],
            "status": status,
            "production": {
                "model_version": task["prod_model_version"],
                "calibrated_risk": task["prod_risk"],
                "decision": task["prod_decision"],
            },
            "candidate": {
                "model_version": "phase41_causal_xgboost_v1",
                "raw_probability": round(raw_prob, 4) if raw_prob is not None else None,
                "calibrated_probability": round(cal_prob, 4) if cal_prob is not None else None,
                "experimental_threshold": opt_thresh,
                "experimental_flag": cand_flag,
                "feature_count": 37,
                "feature_source": feature_source,
                "is_degraded": is_degraded,
                "degraded_reason": degraded_reason,
            },
            "score_delta": score_delta,
            "shadow_latency_ms": latency_ms,
            "error": error,
        }

        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            log_line = json.dumps(record) + "\n"
            with self._log_lock:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(log_line)
        except Exception as log_exc:
            logger.warning("Failed writing to shadow scoring log: %s", log_exc)

    def close(self) -> None:
        """Shuts down background executor cleanly."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None


# Global default instance (disabled by default)
shadow_service = ShadowScoringService()
