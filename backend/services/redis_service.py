"""TrustShield AI — Redis Feature Store & Embedding Cache Service.

Production-grade abstraction for GNN embeddings caching, feature context,
sub-millisecond fraud ring membership lookup, and disk joblib fallback.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

EXPECTED_EMBEDDING_DIMS: Dict[str, int] = {
    "buyer": 16,
    "seller": 16,
}


@dataclass
class EmbeddingResult:
    entity_type: str
    entity_id: str
    vector: Optional[np.ndarray]
    source: str  # "redis" | "disk_artifact" | "not_found" | "unavailable"
    latency_ms: float
    error: Optional[str] = None

    @property
    def found(self) -> bool:
        return self.vector is not None


@dataclass
class RedisHealthStatus:
    available: bool
    status: str  # "connected" | "unavailable" | "disabled"
    latency_ms: Optional[float]
    info: Dict[str, Any]


class RedisService:
    """Production Redis client wrapper with graceful degradation and disk fallback."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        connect_timeout: float = 2.0,
        socket_timeout: float = 2.0,
        enabled: Optional[bool] = None,
    ):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.connect_timeout = connect_timeout
        self.socket_timeout = socket_timeout
        env_enabled = os.getenv("REDIS_ENABLED", "false").lower() in ("true", "1", "yes")
        self.enabled = env_enabled if enabled is None else enabled

        self._client: Any = None
        self._connected: bool = False
        self._last_health_check: float = 0.0
        self._last_connect_attempt: float = 0.0
        self.reconnect_cooldown: float = 30.0
        self._cached_health: Optional[RedisHealthStatus] = None

        if self.enabled:
            self._init_client()
        else:
            self._cached_health = RedisHealthStatus(
                available=False, status="disabled", latency_ms=None,
                info={"detail": "Redis integration explicitly disabled via REDIS_ENABLED=false"},
            )

    def _init_client(self) -> None:
        now = time.time()
        if now - self._last_connect_attempt < self.reconnect_cooldown:
            return
        self._last_connect_attempt = now
        try:
            import redis  # type: ignore[import-not-found]
            self._client = redis.Redis.from_url(
                self.redis_url, socket_connect_timeout=self.connect_timeout,
                socket_timeout=self.socket_timeout, decode_responses=True,
            )
            self._client.ping()
            self._connected = True
            self._cached_health = RedisHealthStatus(available=True, status="connected", latency_ms=1.0, info={"url": self.redis_url})
        except Exception as exc:
            self._connected = False
            self._client = None
            self._cached_health = RedisHealthStatus(
                available=False, status="unavailable", latency_ms=None,
                info={"url": self.redis_url, "error": str(exc)},
            )
            logger.warning("Redis unavailable at %s (%s). Using disk artifacts.", self.redis_url, exc)

    def ping(self) -> RedisHealthStatus:
        if not self.enabled:
            return RedisHealthStatus(available=False, status="disabled", latency_ms=None, info={"detail": "Redis integration explicitly disabled via REDIS_ENABLED=false"})
        if self._client is None:
            self._init_client()
        if self._client is None or not self._connected:
            return self._cached_health or RedisHealthStatus(available=False, status="unavailable", latency_ms=None, info={"url": self.redis_url, "error": "Connection failed or in cooldown"})
        t0 = time.perf_counter()
        try:
            self._client.ping()
            self._connected = True
            self._cached_health = RedisHealthStatus(available=True, status="connected", latency_ms=round((time.perf_counter() - t0) * 1000.0, 3), info={"url": self.redis_url})
        except Exception as exc:
            self._connected = False
            self._cached_health = RedisHealthStatus(available=False, status="unavailable", latency_ms=None, info={"url": self.redis_url, "error": str(exc)})
        return self._cached_health

    @property
    def is_available(self) -> bool:
        if not self.enabled:
            return False
        if self._connected and self._client is not None:
            return True
        now = time.time()
        if now - self._last_health_check > self.reconnect_cooldown:
            self._last_health_check = now
            self._cached_health = self.ping()
        return bool(self._cached_health and self._cached_health.available)

    def _format_key(self, entity_type: str, entity_id: str) -> str:
        return f"trustshield:embedding:{entity_type.strip().lower()}:{str(entity_id).strip()}"

    def get_embedding(
        self,
        entity_type: str,
        entity_id: str,
        disk_fallback_dict: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingResult:
        t0 = time.perf_counter()
        clean_type, clean_id = entity_type.strip().lower(), str(entity_id).strip()
        if not clean_id:
            return EmbeddingResult(clean_type, clean_id, None, "not_found", 0.0, error="Empty entity_id provided")

        expected_dim = EXPECTED_EMBEDDING_DIMS.get(clean_type, 16)

        if self.is_available and self._client is not None:
            for key in [self._format_key(clean_type, clean_id), f"{clean_type}:{clean_id}:emb"]:
                try:
                    raw_val = self._client.get(key)
                    if raw_val is not None:
                        vec = np.asarray(json.loads(raw_val), dtype=np.float32)
                        if vec.shape == (expected_dim,):
                            lat = (time.perf_counter() - t0) * 1000.0
                            return EmbeddingResult(clean_type, clean_id, vec, "redis", round(lat, 3))
                except Exception as exc:
                    logger.warning("Redis embedding read error for %s: %s", key, exc)

        if disk_fallback_dict is not None and clean_id in disk_fallback_dict:
            vec = np.asarray(disk_fallback_dict[clean_id], dtype=np.float32)
            lat = (time.perf_counter() - t0) * 1000.0
            return EmbeddingResult(clean_type, clean_id, vec, "disk_artifact", round(lat, 3))

        lat = (time.perf_counter() - t0) * 1000.0
        return EmbeddingResult(clean_type, clean_id, None, "not_found", round(lat, 3), error=f"No embedding found for {clean_type} {clean_id}")

    def cache_embedding(
        self,
        entity_type: str,
        entity_id: str,
        vector: Union[np.ndarray, List[float]],
        ttl_seconds: int = 86400,
    ) -> bool:
        clean_type, clean_id = entity_type.strip().lower(), str(entity_id).strip()
        expected_dim = EXPECTED_EMBEDDING_DIMS.get(clean_type, 16)
        vec_arr = np.asarray(vector, dtype=np.float32).ravel()
        if len(vec_arr) != expected_dim:
            raise ValueError(f"Embedding dimension mismatch for {clean_type}: expected {expected_dim}, got {len(vec_arr)}")

        if not self.is_available or self._client is None:
            return False

        key = self._format_key(clean_type, clean_id)
        try:
            self._client.set(key, json.dumps(vec_arr.tolist()), ex=ttl_seconds)
            return True
        except Exception as exc:
            logger.warning("Failed to cache embedding for %s: %s", key, exc)
            return False

    def get_feature_context(self, entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        if not self.is_available or self._client is None:
            return None
        key = f"trustshield:context:{entity_type.strip().lower()}:{str(entity_id).strip()}"
        try:
            data = self._client.get(key)
            return json.loads(data) if data else None
        except Exception:
            return None

    def set_feature_context(self, entity_type: str, entity_id: str, context: Dict[str, Any], ttl_seconds: int = 3600) -> bool:
        if not self.is_available or self._client is None:
            return False
        key = f"trustshield:context:{entity_type.strip().lower()}:{str(entity_id).strip()}"
        try:
            self._client.set(key, json.dumps(context), ex=ttl_seconds)
            return True
        except Exception:
            return False

    def benchmark_latency(self, iterations: int = 50) -> Dict[str, float]:
        if not self.is_available or self._client is None:
            return {"available": 0.0, "p50_ms": -1.0, "p95_ms": -1.0, "p99_ms": -1.0}
        latencies = []
        test_key, test_val = "trustshield:benchmark:probe", json.dumps([0.1] * 16)
        try:
            for _ in range(iterations):
                t0 = time.perf_counter()
                self._client.set(test_key, test_val, ex=60)
                _ = self._client.get(test_key)
                latencies.append((time.perf_counter() - t0) * 1000.0)
            self._client.delete(test_key)
        except Exception as exc:
            logger.warning("Redis benchmark failed: %s", exc)
            return {"available": 0.0, "error": -1.0}

        arr = np.array(latencies)
        return {
            "available": 1.0, "count": float(len(arr)),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "mean_ms": round(float(np.mean(arr)), 3),
        }

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            self._connected = False


redis_service = RedisService()
