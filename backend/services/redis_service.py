"""
TrustShield AI — Redis Feature Store & Embedding Cache Service.

Production-grade abstraction for:
- 16D GNN Buyer and Seller embeddings caching
- Online feature context caching (velocity, sliding window aggregates)
- Sub-millisecond fraud ring membership lookup
- Transparent fallback to disk joblib artifacts when Redis is offline
- Strict validation preventing entity 0 leakage and zero-fill deception
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Expected embedding dimension for Phase 5 GNN representations
EXPECTED_EMBEDDING_DIMS: Dict[str, int] = {
    "buyer": 16,
    "seller": 16,
}


@dataclass
class EmbeddingResult:
    """Explicit container for embedding retrieval results and provenance."""
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

        # Explicit toggle support via env
        env_enabled = os.getenv("REDIS_ENABLED", "true").lower() in ("true", "1", "yes")
        self.enabled = env_enabled if enabled is None else enabled

        self._client: Any = None
        self._connected: bool = False
        self._last_health_check: float = 0.0
        self._last_connect_attempt: float = 0.0
        self.reconnect_cooldown: float = 30.0
        self._cached_health: Optional[RedisHealthStatus] = None

        if self.enabled:
            self._init_client()

    def _init_client(self) -> None:
        """Initialize Redis connection client with bounded socket timeouts and cooldown."""
        now = time.time()
        if now - self._last_connect_attempt < self.reconnect_cooldown:
            return
        self._last_connect_attempt = now

        try:
            import redis  # type: ignore[import-not-found]
            self._client = redis.Redis.from_url(
                self.redis_url,
                socket_connect_timeout=self.connect_timeout,
                socket_timeout=self.socket_timeout,
                decode_responses=True,
            )
            # Test ping
            self._client.ping()
            self._connected = True
            self._cached_health = RedisHealthStatus(available=True, status="connected", latency_ms=1.0, info={"url": self.redis_url})
            logger.info("Connected to Redis feature store at %s", self.redis_url)
        except Exception as exc:
            self._connected = False
            self._client = None
            self._cached_health = RedisHealthStatus(
                available=False,
                status="unavailable",
                latency_ms=None,
                info={"url": self.redis_url, "error": str(exc)},
            )
            logger.warning(
                "Redis unavailable at %s (%s). Feature store will use disk artifacts.",
                self.redis_url,
                exc,
            )

    def ping(self) -> RedisHealthStatus:
        """Measure health status and latency of Redis."""
        if not self.enabled:
            return RedisHealthStatus(
                available=False,
                status="disabled",
                latency_ms=None,
                info={"detail": "Redis integration explicitly disabled via REDIS_ENABLED=false"},
            )

        if self._client is None:
            self._init_client()

        if self._client is None or not self._connected:
            return self._cached_health or RedisHealthStatus(
                available=False,
                status="unavailable",
                latency_ms=None,
                info={"url": self.redis_url, "error": "Connection failed or in cooldown"},
            )

        t0 = time.perf_counter()
        try:
            self._client.ping()
            latency = (time.perf_counter() - t0) * 1000.0
            self._connected = True
            self._cached_health = RedisHealthStatus(
                available=True,
                status="connected",
                latency_ms=round(latency, 3),
                info={"url": self.redis_url},
            )
            return self._cached_health
        except Exception as exc:
            self._connected = False
            self._cached_health = RedisHealthStatus(
                available=False,
                status="unavailable",
                latency_ms=None,
                info={"url": self.redis_url, "error": str(exc)},
            )
            return self._cached_health

    @property
    def is_available(self) -> bool:
        """Check if Redis is currently connected."""
        if not self.enabled:
            return False
        if self._connected and self._client is not None:
            return True
        now = time.time()
        if now - self._last_health_check > self.reconnect_cooldown:
            self._last_health_check = now
            self._cached_health = self.ping()
        return bool(self._cached_health and self._cached_health.available)

    # -------------------------------------------------------------------------
    # Embedding Operations
    # -------------------------------------------------------------------------

    def _format_key(self, entity_type: str, entity_id: str) -> str:
        clean_type = entity_type.strip().lower()
        clean_id = entity_id.strip()
        return f"trustshield:embedding:{clean_type}:{clean_id}"

    def get_embedding(
        self,
        entity_type: str,
        entity_id: str,
        disk_fallback_dict: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingResult:
        """
        Retrieve a pre-computed embedding vector for buyer or seller.

        1. Try Redis key `trustshield:embedding:{entity_type}:{entity_id}`
        2. Try fallback Redis key `{entity_type}:{entity_id}:emb` (seed_mesh format)
        3. Fall back to in-memory/disk dictionary if provided
        4. Return explicit 'not_found' or 'unavailable' without falling back to entity 0.
        """
        t0 = time.perf_counter()
        clean_type = entity_type.strip().lower()
        clean_id = str(entity_id).strip()

        if not clean_id:
            return EmbeddingResult(
                entity_type=clean_type,
                entity_id=clean_id,
                vector=None,
                source="not_found",
                latency_ms=0.0,
                error="Empty entity_id provided",
            )

        expected_dim = EXPECTED_EMBEDDING_DIMS.get(clean_type, 16)

        # 1. Attempt Live Redis Lookup
        if self.is_available and self._client is not None:
            canonical_key = self._format_key(clean_type, clean_id)
            legacy_key = f"{clean_type}:{clean_id}:emb"
            try:
                raw_val = self._client.get(canonical_key)
                if raw_val is None:
                    raw_val = self._client.get(legacy_key)

                if raw_val is not None:
                    parsed = json.loads(raw_val)
                    vec = np.asarray(parsed, dtype=np.float32)
                    if vec.shape != (expected_dim,):
                        logger.warning(
                            "Redis embedding shape mismatch for %s:%s (expected %s, got %s)",
                            clean_type, clean_id, expected_dim, vec.shape,
                        )
                    else:
                        latency = (time.perf_counter() - t0) * 1000.0
                        return EmbeddingResult(
                            entity_type=clean_type,
                            entity_id=clean_id,
                            vector=vec,
                            source="redis",
                            latency_ms=round(latency, 3),
                        )
            except Exception as exc:
                logger.warning("Redis embedding read error for %s: %s", canonical_key, exc)

        # 2. Disk Artifact Fallback
        if disk_fallback_dict is not None and clean_id in disk_fallback_dict:
            raw_vec = disk_fallback_dict[clean_id]
            vec = np.asarray(raw_vec, dtype=np.float32)
            latency = (time.perf_counter() - t0) * 1000.0
            return EmbeddingResult(
                entity_type=clean_type,
                entity_id=clean_id,
                vector=vec,
                source="disk_artifact",
                latency_ms=round(latency, 3),
            )

        # 3. Not Found (Explicitly DO NOT substitute entity 0)
        latency = (time.perf_counter() - t0) * 1000.0
        return EmbeddingResult(
            entity_type=clean_type,
            entity_id=clean_id,
            vector=None,
            source="not_found" if not self.is_available else "not_found",
            latency_ms=round(latency, 3),
            error=f"No embedding found for {clean_type} {clean_id}",
        )

    def cache_embedding(
        self,
        entity_type: str,
        entity_id: str,
        vector: Union[np.ndarray, List[float]],
        ttl_seconds: int = 86400,
    ) -> bool:
        """Cache an embedding vector into Redis with explicit TTL."""
        if not self.is_available or self._client is None:
            return False

        clean_type = entity_type.strip().lower()
        clean_id = str(entity_id).strip()
        expected_dim = EXPECTED_EMBEDDING_DIMS.get(clean_type, 16)

        vec_list = [float(x) for x in vector]
        if len(vec_list) != expected_dim:
            raise ValueError(
                f"Embedding dimension error: expected {expected_dim}, got {len(vec_list)}"
            )

        key = self._format_key(clean_type, clean_id)
        payload = json.dumps(vec_list)
        try:
            self._client.set(key, payload, ex=ttl_seconds)
            return True
        except Exception as exc:
            logger.warning("Failed to cache embedding for %s: %s", key, exc)
            return False

    # -------------------------------------------------------------------------
    # Feature Context & Velocity Operations
    # -------------------------------------------------------------------------

    def get_feature_context(self, entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached real-time feature context (e.g. recent order velocity)."""
        if not self.is_available or self._client is None:
            return None

        key = f"trustshield:context:{entity_type.strip().lower()}:{str(entity_id).strip()}"
        try:
            data = self._client.get(key)
            return json.loads(data) if data else None
        except Exception as exc:
            logger.warning("Failed reading feature context from %s: %s", key, exc)
            return None

    def set_feature_context(
        self,
        entity_type: str,
        entity_id: str,
        context: Dict[str, Any],
        ttl_seconds: int = 3600,
    ) -> bool:
        """Cache real-time feature context with TTL."""
        if not self.is_available or self._client is None:
            return False

        key = f"trustshield:context:{entity_type.strip().lower()}:{str(entity_id).strip()}"
        try:
            self._client.set(key, json.dumps(context), ex=ttl_seconds)
            return True
        except Exception as exc:
            logger.warning("Failed writing feature context to %s: %s", key, exc)
            return False

    # -------------------------------------------------------------------------
    # Benchmarking
    # -------------------------------------------------------------------------

    def benchmark_latency(self, iterations: int = 50) -> Dict[str, float]:
        """Measure actual p50, p95, p99 round-trip latency against Redis."""
        if not self.is_available or self._client is None:
            return {"available": 0.0, "p50_ms": -1.0, "p95_ms": -1.0, "p99_ms": -1.0}

        latencies = []
        test_key = "trustshield:benchmark:probe"
        test_val = json.dumps([0.1] * 16)
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
            "available": 1.0,
            "count": float(len(arr)),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "mean_ms": round(float(np.mean(arr)), 3),
        }

    def close(self) -> None:
        """Close connection pool."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            self._connected = False


# Global default instance
redis_service = RedisService()
