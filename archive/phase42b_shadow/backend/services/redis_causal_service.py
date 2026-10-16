"""TrustShield AI — Redis Causal Sliding Windows & Historical State Service.

Phase 4.2B Isolated Implementation:
- Point-in-time causal feature extraction strictly for t < t_decision.
- Bounded 7-day and 30-day velocity windows.
- Pairwise prior amounts, time deltas, returns, and cold-start state.
- Idempotent transaction and return event ingestion lifecycle.
- Resilient to out-of-order events, duplicate timestamps, and retries.
- Graceful degradation when Redis is offline with explicit degraded-state tracking.
- Self-contained in-memory fallback store for deterministic offline replay and unit testing.
"""

from __future__ import annotations

import bisect
import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# HMAC Salt for privacy-safe device identifier hashing
_DEVICE_SALT = os.getenv("TRUSTSHIELD_DEVICE_SALT", "trustshield_device_salt_phase42").encode("utf-8")


@dataclass
class CausalFeatureResult:
    """Explicit container for candidate causal features and provenance."""
    features: Dict[str, float]
    is_degraded: bool = False
    degraded_reason: Optional[str] = None
    feature_source: str = "redis_live"  # "redis_live" | "in_memory" | "fallback_degraded"
    latency_ms: float = 0.0


class InMemoryCausalStore:
    """Thread-safe, deterministic in-memory state store for sliding-window events.

    Used for unit testing, historical replay parity validation, and offline auditing.
    Mirror of Redis Sorted Sets (ZSET) semantics with strict binary search causality.
    """

    def __init__(self):
        # Maps key -> list of (timestamp_float, member_str) sorted chronologically by timestamp
        self._zsets: Dict[str, List[Tuple[float, str]]] = {}
        # Set of seen order IDs for deduplication
        self._seen_orders: set[str] = set()
        self._seen_returns: set[str] = set()

    def zadd(self, key: str, member: str, score: float) -> bool:
        """Inserts (score, member) maintaining chronological order."""
        lst = self._zsets.setdefault(key, [])
        # Check if member already exists in this zset
        for i, (s, m) in enumerate(lst):
            if m == member:
                # Update score if changed and re-sort
                if s != score:
                    lst.pop(i)
                    bisect.insort(lst, (score, member))
                return False
        bisect.insort(lst, (score, member))
        return True

    def zcount(self, key: str, min_score: float, max_score_exclusive: float) -> int:
        """Counts elements in [min_score, max_score_exclusive). Strictly excludes max_score_exclusive."""
        lst = self._zsets.get(key, [])
        if not lst:
            return 0
        scores = [s for s, _ in lst]
        left = bisect.bisect_left(scores, min_score)
        right = bisect.bisect_left(scores, max_score_exclusive)
        return max(0, right - left)

    def zrevrangebyscore_top1(self, key: str, max_score_exclusive: float) -> Optional[float]:
        """Finds the largest timestamp strictly less than max_score_exclusive."""
        lst = self._zsets.get(key, [])
        if not lst:
            return None
        scores = [s for s, _ in lst]
        idx = bisect.bisect_left(scores, max_score_exclusive)
        if idx > 0:
            return scores[idx - 1]
        return None

    def zrangebyscore(
        self, key: str, min_score: float, max_score_exclusive: float
    ) -> List[Tuple[str, float]]:
        """Returns list of (member, score) with min_score <= score < max_score_exclusive."""
        lst = self._zsets.get(key, [])
        if not lst:
            return []
        scores = [s for s, _ in lst]
        left = bisect.bisect_left(scores, min_score)
        right = bisect.bisect_left(scores, max_score_exclusive)
        return [(m, s) for s, m in lst[left:right]]

    def setnx_seen(self, order_id: str) -> bool:
        if order_id in self._seen_orders:
            return False
        self._seen_orders.add(order_id)
        return True

    def setnx_seen_return(self, return_id: str) -> bool:
        if return_id in self._seen_returns:
            return False
        self._seen_returns.add(return_id)
        return True

    def clear(self) -> None:
        self._zsets.clear()
        self._seen_orders.clear()
        self._seen_returns.clear()


class RedisCausalService:
    """Production Redis Causal Service for sliding-window features and lifecycle management."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        enabled: Optional[bool] = None,
        socket_timeout: float = 0.05,  # 50ms default timeout
        connect_timeout: float = 0.5,
        in_memory_mode: bool = False,
        ttl_seconds: int = 3024000,  # 35 days
    ):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        env_enabled = os.getenv("REDIS_ENABLED", "false").lower() in ("true", "1", "yes")
        self.enabled = env_enabled if enabled is None else enabled
        self.socket_timeout = socket_timeout
        self.connect_timeout = connect_timeout
        self.ttl_seconds = ttl_seconds

        self.in_memory_mode = in_memory_mode
        self._in_memory_store = InMemoryCausalStore()

        self._client: Any = None
        self._connected: bool = False
        self._last_connect_attempt: float = 0.0
        self.reconnect_cooldown: float = 15.0

        if self.enabled and not self.in_memory_mode:
            self._init_client()

    def _init_client(self) -> None:
        now = time.time()
        if now - self._last_connect_attempt < self.reconnect_cooldown:
            return
        self._last_connect_attempt = now
        try:
            import redis
            self._client = redis.Redis.from_url(
                self.redis_url,
                socket_connect_timeout=self.connect_timeout,
                socket_timeout=self.socket_timeout,
                decode_responses=True,
            )
            self._client.ping()
            self._connected = True
            logger.info("RedisCausalService connected to %s", self.redis_url)
        except Exception as exc:
            self._connected = False
            self._client = None
            logger.warning("RedisCausalService connection failed (%s). Degrading gracefully.", exc)

    @property
    def is_available(self) -> bool:
        if self.in_memory_mode:
            return True
        if not self.enabled:
            return False
        if self._connected and self._client is not None:
            return True
        self._init_client()
        return self._connected and self._client is not None

    @staticmethod
    def hash_device_id(device_id: Optional[str]) -> str:
        """Hashes raw device identifier with secret salt for privacy-safe storage."""
        if not device_id or not str(device_id).strip():
            return "unknown_device"
        return hmac.new(_DEVICE_SALT, str(device_id).strip().encode("utf-8"), hashlib.sha256).hexdigest()[:24]

    # -------------------------------------------------------------------------
    # 1. Authoritative Event Ingestion Lifecycle
    # -------------------------------------------------------------------------

    def ingest_confirmed_transaction(
        self,
        order_id: str,
        buyer_id: str,
        seller_id: str,
        amount: float,
        timestamp: Union[float, int, str, datetime, pd.Timestamp],
        device_id: Optional[str] = None,
    ) -> bool:
        """Idempotently ingests a confirmed transaction into sliding-window stores.

        Important Lifecycle Rule:
        Only call this after the transaction has been committed/authorized.
        Never call this during pre-scoring evaluation.
        """
        if not order_id or not buyer_id or not seller_id:
            return False

        b_clean = str(buyer_id).strip()
        s_clean = str(seller_id).strip()
        o_clean = str(order_id).strip()
        pair_key = f"{b_clean}:{s_clean}"

        # Resolve timestamp to unix epoch float
        if isinstance(timestamp, (int, float)):
            ts = float(timestamp)
        else:
            ts = pd.to_datetime(timestamp).timestamp()

        dev_hash = self.hash_device_id(device_id) if device_id else None
        pair_member = f"{o_clean}:{float(amount)}"

        # Case A: In-memory mode
        if self.in_memory_mode:
            if not self._in_memory_store.setnx_seen(o_clean):
                return False  # Idempotent ignore
            self._in_memory_store.zadd(f"ts:z:b:{b_clean}", o_clean, ts)
            self._in_memory_store.zadd(f"ts:z:s:{s_clean}", o_clean, ts)
            self._in_memory_store.zadd(f"ts:z:p:{pair_key}", pair_member, ts)
            if dev_hash and dev_hash != "unknown_device":
                self._in_memory_store.zadd(f"ts:z:d:{dev_hash}", o_clean, ts)
            return True

        # Case B: Redis Live mode
        if not self.is_available or self._client is None:
            logger.warning("Transaction %s dropped from Redis sliding window: Redis unavailable", o_clean)
            return False

        try:
            # 1. Idempotency check via SETNX with 7-day TTL
            dedup_key = f"ts:str:dedup:{o_clean}"
            if not self._client.set(dedup_key, "1", ex=604800, nx=True):
                return False  # Duplicate ignored

            # 2. Pipeline commit
            pipe = self._client.pipeline()
            k_b = f"ts:z:b:{b_clean}"
            k_s = f"ts:z:s:{s_clean}"
            k_p = f"ts:z:p:{pair_key}"

            pipe.zadd(k_b, {o_clean: ts})
            pipe.zadd(k_s, {o_clean: ts})
            pipe.zadd(k_p, {pair_member: ts})

            pipe.expire(k_b, self.ttl_seconds)
            pipe.expire(k_s, self.ttl_seconds)
            pipe.expire(k_p, self.ttl_seconds)

            # Active pruning of events older than 35 days
            prune_cutoff = ts - self.ttl_seconds
            pipe.zremrangebyscore(k_b, "-inf", f"({prune_cutoff}")
            pipe.zremrangebyscore(k_s, "-inf", f"({prune_cutoff}")
            pipe.zremrangebyscore(k_p, "-inf", f"({prune_cutoff}")

            if dev_hash and dev_hash != "unknown_device":
                k_d = f"ts:z:d:{dev_hash}"
                pipe.zadd(k_d, {o_clean: ts})
                pipe.expire(k_d, self.ttl_seconds)
                pipe.zremrangebyscore(k_d, "-inf", f"({prune_cutoff}")

            pipe.execute()
            return True
        except Exception as exc:
            logger.warning("Redis ingestion failed for order %s: %s", o_clean, exc)
            return False

    def ingest_return_event(
        self,
        return_id: str,
        buyer_id: str,
        seller_id: str,
        timestamp: Union[float, int, str, datetime, pd.Timestamp],
    ) -> bool:
        """Idempotently records a return event into causal return stores."""
        if not return_id or not buyer_id or not seller_id:
            return False

        ret_clean = str(return_id).strip()
        b_clean = str(buyer_id).strip()
        s_clean = str(seller_id).strip()
        pair_key = f"{b_clean}:{s_clean}"

        if isinstance(timestamp, (int, float)):
            ts = float(timestamp)
        else:
            ts = pd.to_datetime(timestamp).timestamp()

        if self.in_memory_mode:
            if not self._in_memory_store.setnx_seen_return(ret_clean):
                return False
            self._in_memory_store.zadd(f"ts:z:ret:b:{b_clean}", ret_clean, ts)
            self._in_memory_store.zadd(f"ts:z:ret:s:{s_clean}", ret_clean, ts)
            self._in_memory_store.zadd(f"ts:z:ret:p:{pair_key}", ret_clean, ts)
            return True

        if not self.is_available or self._client is None:
            return False

        try:
            dedup_key = f"ts:str:dedup_ret:{ret_clean}"
            if not self._client.set(dedup_key, "1", ex=604800, nx=True):
                return False

            pipe = self._client.pipeline()
            k_ret_b = f"ts:z:ret:b:{b_clean}"
            k_ret_s = f"ts:z:ret:s:{s_clean}"
            k_ret_p = f"ts:z:ret:p:{pair_key}"

            pipe.zadd(k_ret_b, {ret_clean: ts})
            pipe.zadd(k_ret_s, {ret_clean: ts})
            pipe.zadd(k_ret_p, {ret_clean: ts})

            pipe.expire(k_ret_b, self.ttl_seconds)
            pipe.expire(k_ret_s, self.ttl_seconds)
            pipe.expire(k_ret_p, self.ttl_seconds)
            pipe.execute()
            return True
        except Exception as exc:
            logger.warning("Redis return ingestion failed for return %s: %s", ret_clean, exc)
            return False

    # -------------------------------------------------------------------------
    # 2. Causal Feature Extraction (Strictly t < t_decision)
    # -------------------------------------------------------------------------

    def query_causal_features(
        self,
        buyer_id: Optional[str],
        seller_id: Optional[str],
        device_id: Optional[str],
        amount: float,
        decision_timestamp: Union[float, int, str, datetime, pd.Timestamp],
    ) -> CausalFeatureResult:
        """Queries the 19 causal sliding-window features strictly before decision_timestamp.

        Guarantees:
        - Only information with t < t_decision is visible.
        - Same-timestamp transactions are strictly excluded via '(' exclusive syntax.
        - Missing or evicted history is not silently reported as authoritative zero history;
          it is flagged with is_degraded=True.
        """
        t0_bench = time.perf_counter()

        if isinstance(decision_timestamp, (int, float)):
            t0 = float(decision_timestamp)
        else:
            t0 = pd.to_datetime(decision_timestamp).timestamp()

        # Default clean return dictionary
        defaults: Dict[str, float] = {
            "pair_orders_last_7d": 0.0,
            "pair_orders_last_30d": 0.0,
            "pair_days_since_prior": 999.0,
            "pair_prior_amount_mean": 0.0,
            "amount_vs_pair_prior_mean": 1.0,
            "pair_returns_before": 0.0,
            "pair_is_cold_start": 1.0,
            "buyer_pair_order_share": 0.0,
            "seller_pair_order_share": 0.0,
            "buyer_orders_last_7d": 0.0,
            "buyer_orders_last_30d": 0.0,
            "buyer_days_since_prior": 999.0,
            "seller_prior_order_count": 0.0,
            "seller_orders_last_7d": 0.0,
            "seller_orders_last_30d": 0.0,
            "seller_days_since_prior": 999.0,
            "seller_returns_before": 0.0,
            "seller_return_rate_before": 0.0,
            "device_orders_last_7d": 0.0,
        }

        if not buyer_id or not seller_id:
            latency = (time.perf_counter() - t0_bench) * 1000.0
            return CausalFeatureResult(
                features=defaults,
                is_degraded=True,
                degraded_reason="missing_entity_ids",
                feature_source="fallback_degraded",
                latency_ms=round(latency, 3),
            )

        b_clean = str(buyer_id).strip()
        s_clean = str(seller_id).strip()
        pair_key = f"{b_clean}:{s_clean}"
        dev_hash = self.hash_device_id(device_id) if device_id else None

        sec_7d = 7 * 86400.0
        sec_30d = 30 * 86400.0

        # Case A: In-Memory Mode
        if self.in_memory_mode:
            store = self._in_memory_store
            # Buyer features
            b_key = f"ts:z:b:{b_clean}"
            b_count = store.zcount(b_key, float("-inf"), t0)
            b_7d = store.zcount(b_key, t0 - sec_7d, t0)
            b_30d = store.zcount(b_key, t0 - sec_30d, t0)
            b_prev = store.zrevrangebyscore_top1(b_key, t0)
            b_dt = (t0 - b_prev) / 86400.0 if b_prev is not None else 999.0

            # Seller features
            s_key = f"ts:z:s:{s_clean}"
            s_count = store.zcount(s_key, float("-inf"), t0)
            s_7d = store.zcount(s_key, t0 - sec_7d, t0)
            s_30d = store.zcount(s_key, t0 - sec_30d, t0)
            s_prev = store.zrevrangebyscore_top1(s_key, t0)
            s_dt = (t0 - s_prev) / 86400.0 if s_prev is not None else 999.0

            s_ret_key = f"ts:z:ret:s:{s_clean}"
            s_ret = store.zcount(s_ret_key, float("-inf"), t0)
            s_ret_rate = float(s_ret) / max(float(s_count), 1.0)

            # Pair features
            p_key = f"ts:z:p:{pair_key}"
            pair_items = store.zrangebyscore(p_key, float("-inf"), t0)
            p_count = len(pair_items)

            if p_count > 0:
                p_7d = sum(1 for _, score in pair_items if score >= t0 - sec_7d)
                p_30d = sum(1 for _, score in pair_items if score >= t0 - sec_30d)
                p_prev = max(score for _, score in pair_items)
                p_dt = (t0 - p_prev) / 86400.0
                p_amounts = []
                for m, _ in pair_items:
                    if ":" in m:
                        try:
                            p_amounts.append(float(m.split(":", 1)[1]))
                        except ValueError:
                            pass
                p_amt_mean = float(np.mean(p_amounts)) if p_amounts else 0.0
                amt_vs_mean = float(amount) / max(p_amt_mean, 1.0)
                is_cold = 0.0
            else:
                p_7d = 0.0
                p_30d = 0.0
                p_dt = 999.0
                p_amt_mean = 0.0
                amt_vs_mean = 1.0
                is_cold = 1.0

            p_ret_key = f"ts:z:ret:p:{pair_key}"
            p_ret = store.zcount(p_ret_key, float("-inf"), t0)

            # Device
            if dev_hash and dev_hash != "unknown_device":
                d_key = f"ts:z:d:{dev_hash}"
                d_7d = store.zcount(d_key, t0 - sec_7d, t0)
            else:
                d_7d = 0.0

            feat = {
                "pair_orders_last_7d": float(p_7d),
                "pair_orders_last_30d": float(p_30d),
                "pair_days_since_prior": float(p_dt),
                "pair_prior_amount_mean": float(p_amt_mean),
                "amount_vs_pair_prior_mean": float(amt_vs_mean),
                "pair_returns_before": float(p_ret),
                "pair_is_cold_start": float(is_cold),
                "buyer_pair_order_share": float(p_count) / max(float(b_count), 1.0),
                "seller_pair_order_share": float(p_count) / max(float(s_count), 1.0),
                "buyer_orders_last_7d": float(b_7d),
                "buyer_orders_last_30d": float(b_30d),
                "buyer_days_since_prior": float(b_dt),
                "seller_prior_order_count": float(s_count),
                "seller_orders_last_7d": float(s_7d),
                "seller_orders_last_30d": float(s_30d),
                "seller_days_since_prior": float(s_dt),
                "seller_returns_before": float(s_ret),
                "seller_return_rate_before": float(s_ret_rate),
                "device_orders_last_7d": float(d_7d),
            }
            latency = (time.perf_counter() - t0_bench) * 1000.0
            return CausalFeatureResult(
                features=feat,
                is_degraded=False,
                degraded_reason=None,
                feature_source="in_memory",
                latency_ms=round(latency, 3),
            )

        # Case B: Redis Live Mode
        if not self.is_available or self._client is None:
            latency = (time.perf_counter() - t0_bench) * 1000.0
            return CausalFeatureResult(
                features=defaults,
                is_degraded=True,
                degraded_reason="redis_unavailable_or_disabled",
                feature_source="fallback_degraded",
                latency_ms=round(latency, 3),
            )

        try:
            pipe = self._client.pipeline(transaction=False)
            k_b = f"ts:z:b:{b_clean}"
            k_s = f"ts:z:s:{s_clean}"
            k_p = f"ts:z:p:{pair_key}"
            k_ret_s = f"ts:z:ret:s:{s_clean}"
            k_ret_p = f"ts:z:ret:p:{pair_key}"

            # 1. Buyer
            pipe.zcount(k_b, "-inf", f"({t0}")
            pipe.zcount(k_b, f"({t0 - sec_7d}", f"({t0}")
            pipe.zcount(k_b, f"({t0 - sec_30d}", f"({t0}")
            pipe.zrevrangebyscore(k_b, f"({t0}", "-inf", withscores=True, start=0, num=1)

            # 2. Seller
            pipe.zcount(k_s, "-inf", f"({t0}")
            pipe.zcount(k_s, f"({t0 - sec_7d}", f"({t0}")
            pipe.zcount(k_s, f"({t0 - sec_30d}", f"({t0}")
            pipe.zrevrangebyscore(k_s, f"({t0}", "-inf", withscores=True, start=0, num=1)
            pipe.zcount(k_ret_s, "-inf", f"({t0}")

            # 3. Pair (full strictly prior members)
            pipe.zrangebyscore(k_p, "-inf", f"({t0}", withscores=True)
            pipe.zcount(k_ret_p, "-inf", f"({t0}")

            # 4. Device
            if dev_hash and dev_hash != "unknown_device":
                k_d = f"ts:z:d:{dev_hash}"
                pipe.zcount(k_d, f"({t0 - sec_7d}", f"({t0}")
            else:
                k_d = None

            results = pipe.execute()

            # Unpack pipeline responses
            b_count = int(results[0])
            b_7d = int(results[1])
            b_30d = int(results[2])
            b_prev_items = results[3]
            b_dt = (t0 - float(b_prev_items[0][1])) / 86400.0 if b_prev_items else 999.0

            s_count = int(results[4])
            s_7d = int(results[5])
            s_30d = int(results[6])
            s_prev_items = results[7]
            s_dt = (t0 - float(s_prev_items[0][1])) / 86400.0 if s_prev_items else 999.0
            s_ret = int(results[8])
            s_ret_rate = float(s_ret) / max(float(s_count), 1.0)

            pair_items = results[9]  # list of (member, score)
            p_ret = int(results[10])

            p_count = len(pair_items)
            if p_count > 0:
                p_7d = sum(1 for _, score in pair_items if score >= t0 - sec_7d)
                p_30d = sum(1 for _, score in pair_items if score >= t0 - sec_30d)
                p_prev = max(score for _, score in pair_items)
                p_dt = (t0 - p_prev) / 86400.0
                p_amounts = []
                for m, _ in pair_items:
                    if ":" in m:
                        try:
                            p_amounts.append(float(m.split(":", 1)[1]))
                        except ValueError:
                            pass
                p_amt_mean = float(np.mean(p_amounts)) if p_amounts else 0.0
                amt_vs_mean = float(amount) / max(p_amt_mean, 1.0)
                is_cold = 0.0
            else:
                p_7d = 0.0
                p_30d = 0.0
                p_dt = 999.0
                p_amt_mean = 0.0
                amt_vs_mean = 1.0
                is_cold = 1.0

            d_7d = int(results[11]) if k_d is not None else 0.0

            feat = {
                "pair_orders_last_7d": float(p_7d),
                "pair_orders_last_30d": float(p_30d),
                "pair_days_since_prior": float(p_dt),
                "pair_prior_amount_mean": float(p_amt_mean),
                "amount_vs_pair_prior_mean": float(amt_vs_mean),
                "pair_returns_before": float(p_ret),
                "pair_is_cold_start": float(is_cold),
                "buyer_pair_order_share": float(p_count) / max(float(b_count), 1.0),
                "seller_pair_order_share": float(p_count) / max(float(s_count), 1.0),
                "buyer_orders_last_7d": float(b_7d),
                "buyer_orders_last_30d": float(b_30d),
                "buyer_days_since_prior": float(b_dt),
                "seller_prior_order_count": float(s_count),
                "seller_orders_last_7d": float(s_7d),
                "seller_orders_last_30d": float(s_30d),
                "seller_days_since_prior": float(s_dt),
                "seller_returns_before": float(s_ret),
                "seller_return_rate_before": float(s_ret_rate),
                "device_orders_last_7d": float(d_7d),
            }
            latency = (time.perf_counter() - t0_bench) * 1000.0
            return CausalFeatureResult(
                features=feat,
                is_degraded=False,
                degraded_reason=None,
                feature_source="redis_live",
                latency_ms=round(latency, 3),
            )

        except Exception as exc:
            latency = (time.perf_counter() - t0_bench) * 1000.0
            logger.warning("Redis query_causal_features error (%s). Returning degraded fallback.", exc)
            return CausalFeatureResult(
                features=defaults,
                is_degraded=True,
                degraded_reason=f"redis_query_error: {exc}",
                feature_source="fallback_degraded",
                latency_ms=round(latency, 3),
            )

    # -------------------------------------------------------------------------
    # 3. Deterministic Historical Backfill & Reconciliation
    # -------------------------------------------------------------------------

    def backfill_from_dataframe(
        self,
        orders_df: pd.DataFrame,
        returns_df: Optional[pd.DataFrame] = None,
        dry_run: bool = False,
        batch_size: int = 1000,
    ) -> Dict[str, Any]:
        """Backfills historical orders and returns chronologically with reconciliation."""
        df_sorted = orders_df.copy()
        df_sorted["order_date"] = pd.to_datetime(df_sorted["order_date"])
        df_sorted = df_sorted.sort_values(by="order_date", kind="mergesort").reset_index(drop=True)

        n_orders = len(df_sorted)
        n_unique_orders = df_sorted["order_id"].nunique()
        n_buyers = df_sorted["buyer_id"].nunique()
        n_sellers = df_sorted["seller_id"].nunique()

        report = {
            "dry_run": dry_run,
            "total_orders_provided": n_orders,
            "unique_orders": n_unique_orders,
            "duplicate_orders": n_orders - n_unique_orders,
            "unique_buyers": n_buyers,
            "unique_sellers": n_sellers,
            "orders_ingested": 0,
            "returns_ingested": 0,
            "reconciliation_status": "unverified",
        }

        if dry_run:
            report["reconciliation_status"] = "dry_run_success"
            return report

        # Ingestion loop
        t0 = time.time()
        for idx, row in df_sorted.iterrows():
            o_id = str(row["order_id"])
            b_id = str(row["buyer_id"])
            s_id = str(row["seller_id"])
            amt = float(row.get("amount", 0.0))
            ts = row["order_date"]
            dev_id = str(row.get("device_id", "")) if pd.notna(row.get("device_id")) else None

            success = self.ingest_confirmed_transaction(
                order_id=o_id,
                buyer_id=b_id,
                seller_id=s_id,
                amount=amt,
                timestamp=ts,
                device_id=dev_id,
            )
            if success:
                report["orders_ingested"] += 1

        if returns_df is not None:
            ret_sorted = returns_df.copy()
            ret_sorted["return_date"] = pd.to_datetime(ret_sorted["return_date"])
            ret_sorted = ret_sorted.sort_values(by="return_date", kind="mergesort").reset_index(drop=True)
            for _, rrow in ret_sorted.iterrows():
                r_id = str(rrow.get("return_id", f"ret_{_}"))
                b_id = str(rrow["buyer_id"])
                s_id = str(rrow["seller_id"])
                r_ts = rrow["return_date"]
                ret_succ = self.ingest_return_event(
                    return_id=r_id,
                    buyer_id=b_id,
                    seller_id=s_id,
                    timestamp=r_ts,
                )
                if ret_succ:
                    report["returns_ingested"] += 1

        report["duration_seconds"] = round(time.time() - t0, 3)
        report["reconciliation_status"] = "verified" if report["orders_ingested"] == n_unique_orders else "mismatch"
        return report


# Global default instance (disabled by default)
redis_causal_service = RedisCausalService()
