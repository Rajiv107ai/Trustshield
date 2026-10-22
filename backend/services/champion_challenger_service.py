"""TrustShield AI — Champion-Challenger Shadow Scoring & Conformal A/B Deployment Service.

Orchestrates real-time shadow evaluation and traffic routing between:
- Champion: Phase 3 Tabular + Graph Random Forest (Isotonic Calibrated, Authoritative)
- Challenger: v2.1 High-Velocity XGBoost Candidate (Calibrated, Experimental)

Supports:
1. SHADOW mode: 100% Champion decisions; 100% Challenger background scoring & telemetry
2. AB_SPLIT mode: Deterministic hash-based split routing (e.g. 80% Champion, 20% Challenger)
3. CANARY mode: Small percentage (5-10%) routed to Challenger with automated circuit breaker
4. Conformal Prediction Set comparisons ({0}, {1}, or ambiguous {0, 1})
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import sys
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd

from backend.model_loader import store
from trustshield_project.policy_config import RISK_POLICY

logger = logging.getLogger(__name__)

# Register calibration alias for legacy pickles
try:
    import trustshield_project.calibration as _cal
    sys.modules.setdefault("calibration", _cal)
except ImportError:
    pass

try:
    from trustshield_project.advanced_trust_engine import ConformalPredictor
except ImportError:
    from advanced_trust_engine import ConformalPredictor  # type: ignore


class ChampionChallengerService:
    """Thread-safe singleton managing Champion-Challenger shadow scoring and A/B routing."""

    def __init__(self):
        self._lock = threading.Lock()
        self.deployment_mode: str = os.getenv("TRUSTSHIELD_DEPLOYMENT_MODE", "SHADOW").upper()
        self.traffic_split_pct: float = float(os.getenv("TRUSTSHIELD_TRAFFIC_SPLIT_PCT", "10.0"))
        self.circuit_breaker_tripped: bool = False
        self.circuit_breaker_reason: Optional[str] = None

        # Maximum rolling comparisons kept in memory
        self.history_limit = 500
        self._comparisons: deque = deque(maxlen=self.history_limit)

        # Challenger Artifacts (lazy-loaded)
        self.challenger_model: Any = None
        self.challenger_calibrator: Any = None
        self.challenger_conformal: Optional[ConformalPredictor] = None
        self.challenger_cols: List[str] = []
        self._challenger_loaded: bool = False
        self._init_error: Optional[str] = None

    def _ensure_challenger_loaded(self) -> bool:
        if self._challenger_loaded and self.challenger_model is not None:
            return True

        with self._lock:
            if self._challenger_loaded and self.challenger_model is not None:
                return True

            try:
                base_dir = os.path.normpath(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
                )
                v21_dir = os.path.join(base_dir, "models", "v2_1")

                # Load Challenger Model (XGBoost v2.1 combined graph or fallback)
                model_path = os.path.join(v21_dir, "combined_graph_model.joblib")
                if not os.path.isfile(model_path):
                    model_path = os.path.join(v21_dir, "tabular_baseline_model.joblib")

                if os.path.isfile(model_path):
                    self.challenger_model = joblib.load(model_path)
                    raw_cols = getattr(self.challenger_model, "feature_names_in_", None)
                    if raw_cols is not None:
                        self.challenger_cols = list(raw_cols)
                    elif store.feature_meta is not None:
                        self.challenger_cols = store.feature_meta.get("all_feature_cols", [])

                    # Load Challenger Calibrator
                    cal_path = os.path.join(v21_dir, "combined_graph_calibrator.joblib")
                    if os.path.isfile(cal_path):
                        self.challenger_calibrator = joblib.load(cal_path)

                    # Initialize Challenger Conformal Predictor with validation data
                    val_pred_path = os.path.join(base_dir, "models", "phase3_val_predictions.npy")
                    val_label_path = os.path.join(base_dir, "models", "phase3_val_labels.npy")
                    if os.path.isfile(val_pred_path) and os.path.isfile(val_label_path):
                        p_val = np.load(val_pred_path)
                        y_val = np.load(val_label_path)
                        # Slightly stricter alpha=0.04 to test comparative coverage
                        self.challenger_conformal = ConformalPredictor(alpha=0.04).calibrate(
                            p_val, y_val, source="v21_challenger_validation"
                        )
                    else:
                        self.challenger_conformal = ConformalPredictor(alpha=0.05)

                    self._challenger_loaded = True
                    logger.info("Challenger model loaded successfully from %s", model_path)
                    return True
                else:
                    self._init_error = f"Challenger artifact not found in {v21_dir}"
                    logger.warning("Challenger model unavailable: %s", self._init_error)
                    return False
            except Exception as exc:
                self._init_error = str(exc)
                logger.warning("Failed to initialize challenger model: %s", exc)
                return False

    def get_status(self) -> Dict[str, Any]:
        """Compute aggregate real-time metrics across rolling evaluations."""
        self._ensure_challenger_loaded()

        with self._lock:
            items = list(self._comparisons)
            total = len(items)

            if total == 0:
                return {
                    "deployment_mode": self.deployment_mode,
                    "traffic_split_pct": self.traffic_split_pct,
                    "champion_spec": {
                        "id": "phase3_champion_rf",
                        "name": "Phase 3 Tabular+Graph Random Forest",
                        "status": "AUTHORITATIVE_PRODUCTION",
                        "calibrator": "Isotonic Regression",
                        "conformal": "Mondrian (alpha=0.05)",
                    },
                    "challenger_spec": {
                        "id": "v21_challenger_xgb",
                        "name": "v2.1 High-Velocity XGBoost Candidate",
                        "status": "SHADOW_DEPLOYMENT" if self.deployment_mode == "SHADOW" else "A_B_ACTIVE",
                        "calibrator": "ProbabilityCalibrator",
                        "conformal": "Mondrian (alpha=0.04)",
                        "loaded": self._challenger_loaded,
                    },
                    "circuit_breaker_tripped": self.circuit_breaker_tripped,
                    "circuit_breaker_reason": self.circuit_breaker_reason,
                    "total_evaluations": 0,
                    "decision_concordance_rate": 1.0,
                    "conformal_concordance_rate": 1.0,
                    "champion_certainty_rate": 1.0,
                    "challenger_certainty_rate": 1.0,
                    "avg_risk_champion": 0.0,
                    "avg_risk_challenger": 0.0,
                    "mean_absolute_risk_diff": 0.0,
                    "p95_latency_champion_ms": 0.0,
                    "p95_latency_challenger_ms": 0.0,
                    "recent_disagreements_count": 0,
                }

            decision_matches = sum(1 for c in items if c["decision_matches"])
            conformal_matches = sum(1 for c in items if c["conformal_matches"])
            champ_singletons = sum(1 for c in items if len(c["champion"]["conformal_prediction_set"]) == 1)
            chall_singletons = sum(1 for c in items if len(c["challenger"]["conformal_prediction_set"]) == 1)

            avg_risk_champ = float(np.mean([c["champion"]["calibrated_score"] for c in items]))
            avg_risk_chall = float(np.mean([c["challenger"]["calibrated_score"] for c in items]))
            mean_risk_diff = float(np.mean([abs(c["risk_delta"]) for c in items]))

            champ_lats = sorted([c["champion"]["latency_ms"] for c in items])
            chall_lats = sorted([c["challenger"]["latency_ms"] for c in items])
            p95_idx = int(math.ceil(0.95 * total)) - 1
            p95_champ = champ_lats[min(p95_idx, total - 1)]
            p95_chall = chall_lats[min(p95_idx, total - 1)]

            disagreements = [c for c in items if not c["decision_matches"]]

            # Check circuit breaker condition
            if self.deployment_mode == "CANARY" and not self.circuit_breaker_tripped and total >= 20:
                disagreement_rate = 1.0 - (decision_matches / total)
                if disagreement_rate > 0.35:
                    self.circuit_breaker_tripped = True
                    self.circuit_breaker_reason = f"Decision disagreement rate ({disagreement_rate:.1%}) exceeded safety threshold 35%."
                elif p95_chall > 150.0:
                    self.circuit_breaker_tripped = True
                    self.circuit_breaker_reason = f"Challenger P95 latency ({p95_chall:.1f}ms) exceeded maximum threshold (150ms)."

            return {
                "deployment_mode": self.deployment_mode,
                "traffic_split_pct": self.traffic_split_pct,
                "champion_spec": {
                    "id": "phase3_champion_rf",
                    "name": "Phase 3 Tabular+Graph Random Forest",
                    "status": "AUTHORITATIVE_PRODUCTION",
                    "calibrator": "Isotonic Regression",
                    "conformal": "Mondrian (alpha=0.05)",
                },
                "challenger_spec": {
                    "id": "v21_challenger_xgb",
                    "name": "v2.1 High-Velocity XGBoost Candidate",
                    "status": "SHADOW_DEPLOYMENT" if self.deployment_mode == "SHADOW" else "A_B_ACTIVE",
                    "calibrator": "ProbabilityCalibrator",
                    "conformal": "Mondrian (alpha=0.04)",
                    "loaded": self._challenger_loaded,
                },
                "circuit_breaker_tripped": self.circuit_breaker_tripped,
                "circuit_breaker_reason": self.circuit_breaker_reason,
                "total_evaluations": total,
                "decision_concordance_rate": round(decision_matches / total, 4),
                "conformal_concordance_rate": round(conformal_matches / total, 4),
                "champion_certainty_rate": round(champ_singletons / total, 4),
                "challenger_certainty_rate": round(chall_singletons / total, 4),
                "avg_risk_champion": round(avg_risk_champ, 4),
                "avg_risk_challenger": round(avg_risk_chall, 4),
                "mean_absolute_risk_diff": round(mean_risk_diff, 4),
                "p95_latency_champion_ms": round(p95_champ, 2),
                "p95_latency_challenger_ms": round(p95_chall, 2),
                "recent_disagreements_count": len(disagreements),
            }

    def configure(
        self,
        deployment_mode: str,
        traffic_split_pct: Optional[float] = None,
        challenger_model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Configure deployment mode, traffic split, and reset circuit breaker if reconfigured."""
        valid_modes = {"SHADOW", "AB_SPLIT", "CANARY"}
        mode = deployment_mode.strip().upper()
        if mode not in valid_modes:
            raise ValueError(f"Invalid deployment mode '{mode}'. Valid modes: {valid_modes}")

        with self._lock:
            self.deployment_mode = mode
            if traffic_split_pct is not None:
                self.traffic_split_pct = max(0.0, min(100.0, float(traffic_split_pct)))
            self.circuit_breaker_tripped = False
            self.circuit_breaker_reason = None

        return self.get_status()

    def _determine_decision(self, prob: float) -> str:
        """Evaluate operational policy decision from calibrated probability."""
        if prob >= RISK_POLICY.block:
            return "BLOCK"
        if prob >= RISK_POLICY.hold:
            return "HOLD"
        if prob >= RISK_POLICY.review:
            return "REVIEW"
        return "ALLOW"

    def score_dual(
        self,
        feature_row: Dict[str, Any],
        order_id: str = "PROBE_TX",
    ) -> Dict[str, Any]:
        """Execute both Champion and Challenger on identical feature row and compare results."""
        self._ensure_challenger_loaded()

        # 1. Evaluate Champion
        t_c0 = time.perf_counter()
        champ_cols = store.feature_meta.get("all_feature_cols", []) if store.feature_meta else list(feature_row.keys())
        full_champ_row = {c: float(feature_row.get(c, 0.0) or 0.0) for c in champ_cols}
        df_champ = pd.DataFrame([full_champ_row])[champ_cols]

        raw_champ = 0.5
        if store.combined_graph_model is not None:
            raw_champ = float(np.asarray(store.combined_graph_model.predict_proba(df_champ))[0, 1])

        cal_champ = raw_champ
        if store.calibrator is not None:
            try:
                cal_champ = float(store.calibrator.predict_proba(np.array([raw_champ]))[0])
            except Exception:
                cal_champ = raw_champ

        lat_champ = (time.perf_counter() - t_c0) * 1000.0
        dec_champ = self._determine_decision(cal_champ)
        trust_champ = round(100.0 * (1.0 - cal_champ), 2)

        conf_set_champ = [0, 1]
        if store.conformal_predictor is not None and getattr(store.conformal_predictor, "is_calibrated", False):
            conf_set_champ = sorted(list(store.conformal_predictor.predict_set(cal_champ)))
        else:
            conf_set_champ = [0, 1] if 0.35 <= cal_champ <= 0.65 else ([1] if cal_champ > 0.65 else [0])

        # 2. Evaluate Challenger
        t_ch0 = time.perf_counter()
        raw_chall = raw_champ
        cal_chall = cal_champ

        if self.challenger_model is not None:
            cols = self.challenger_cols or champ_cols
            full_chall_row = {c: float(feature_row.get(c, 0.0) or 0.0) for c in cols}
            df_chall = pd.DataFrame([full_chall_row])[cols]
            raw_chall = float(np.asarray(self.challenger_model.predict_proba(df_chall))[0, 1])

            if self.challenger_calibrator is not None:
                try:
                    cal_chall = float(self.challenger_calibrator.predict_proba(np.array([raw_chall]))[0])
                except Exception:
                    cal_chall = raw_chall
            else:
                cal_chall = raw_chall
        else:
            # Fallback perturbation for testing when artifact is missing
            raw_chall = max(0.0, min(1.0, raw_champ * 0.95 + 0.02))
            cal_chall = raw_chall

        lat_chall = (time.perf_counter() - t_ch0) * 1000.0
        dec_chall = self._determine_decision(cal_chall)
        trust_chall = round(100.0 * (1.0 - cal_chall), 2)

        conf_set_chall = [0, 1]
        if self.challenger_conformal is not None and getattr(self.challenger_conformal, "is_calibrated", False):
            conf_set_chall = sorted(list(self.challenger_conformal.predict_set(cal_chall)))
        else:
            conf_set_chall = [0, 1] if 0.35 <= cal_chall <= 0.65 else ([1] if cal_chall > 0.65 else [0])

        # 3. Concordance & Routing
        dec_matches = (dec_champ == dec_chall)
        conf_matches = (conf_set_champ == conf_set_chall)
        risk_delta = round(cal_chall - cal_champ, 4)

        # Deterministic routing hash
        hash_val = int(hashlib.sha256(order_id.encode("utf-8")).hexdigest()[:8], 16) % 100
        routed_model = "CHAMPION"
        routed_decision = dec_champ

        if self.deployment_mode in ("AB_SPLIT", "CANARY") and not self.circuit_breaker_tripped:
            if hash_val < self.traffic_split_pct:
                routed_model = "CHALLENGER"
                routed_decision = dec_chall

        # Conformal divergence explanation
        if conf_matches:
            conf_div = f"Both models agree on uncertainty set {conf_set_champ}."
        elif len(conf_set_champ) == 1 and len(conf_set_chall) == 2:
            conf_div = f"Champion confident {conf_set_champ}; Challenger flagged conformal ambiguity {conf_set_chall}."
        elif len(conf_set_champ) == 2 and len(conf_set_chall) == 1:
            conf_div = f"Champion ambiguous {conf_set_champ}; Challenger resolved singleton certainty {conf_set_chall}."
        else:
            conf_div = f"Opposite singletons: Champion {conf_set_champ} vs Challenger {conf_set_chall}."

        rec = (
            "Models in full concordance." if dec_matches and conf_matches
            else f"Decision disagreement ({dec_champ} vs {dec_chall}). Review edge case."
        )

        comparison = {
            "order_id": order_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "champion": {
                "model_id": "phase3_champion_rf",
                "model_name": "Phase 3 Tabular+Graph Random Forest",
                "algorithm": "RandomForestClassifier",
                "raw_score": round(raw_champ, 4),
                "calibrated_score": round(cal_champ, 4),
                "decision": dec_champ,
                "trust_score": trust_champ,
                "conformal_prediction_set": conf_set_champ,
                "is_conformal_singleton": len(conf_set_champ) == 1,
                "latency_ms": round(lat_champ, 2),
            },
            "challenger": {
                "model_id": "v21_challenger_xgb",
                "model_name": "v2.1 High-Velocity XGBoost Candidate",
                "algorithm": "XGBClassifier",
                "raw_score": round(raw_chall, 4),
                "calibrated_score": round(cal_chall, 4),
                "decision": dec_chall,
                "trust_score": trust_chall,
                "conformal_prediction_set": conf_set_chall,
                "is_conformal_singleton": len(conf_set_chall) == 1,
                "latency_ms": round(lat_chall, 2),
            },
            "decision_matches": dec_matches,
            "conformal_matches": conf_matches,
            "risk_delta": risk_delta,
            "conformal_divergence": conf_div,
            "routed_decision": routed_decision,
            "routed_model": routed_model,
            "deployment_mode": self.deployment_mode,
            "traffic_split_pct": self.traffic_split_pct,
            "recommendation": rec,
        }

        with self._lock:
            self._comparisons.append(comparison)

        return comparison

    def record_shadow_evaluation(
        self,
        req: Any,
        feature_row: Dict[str, Any],
        champ_score: float,
        champ_decision: str,
        champ_conformal: List[int],
    ) -> None:
        """Lightweight background recording during production /transaction/score."""
        order_id = getattr(req, "order_id", None) or "LIVE_TX"
        try:
            self.score_dual(feature_row, order_id=order_id)
        except Exception as exc:
            logger.debug("Shadow evaluation recording skipped: %s", exc)

    def get_disagreements(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent comparisons where Champion and Challenger produced divergent decisions or sets."""
        with self._lock:
            disagreements = [
                c for c in reversed(self._comparisons)
                if not c["decision_matches"] or not c["conformal_matches"]
            ]
            return disagreements[:limit]

    def reset_metrics(self) -> None:
        """Reset comparison buffer for testing or deployment milestones."""
        with self._lock:
            self._comparisons.clear()
            self.circuit_breaker_tripped = False
            self.circuit_breaker_reason = None


# Global singleton
champion_challenger_service = ChampionChallengerService()
