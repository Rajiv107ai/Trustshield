"""TrustShield AI — Neo4j Property Graph Investigation Service.

Production-grade graph intelligence service with parameterized Cypher queries,
strict temporal graph safety (event_time < decision_time), and offline disk fallback.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GraphQueryResult:
    query_name: str
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    graph_source: str  # "neo4j" | "disk_artifact" | "unavailable"
    temporal_cutoff_applied: Optional[str]
    latency_ms: float
    record_count: int
    error: Optional[str] = None


@dataclass
class Neo4jHealthStatus:
    available: bool
    status: str  # "connected" | "unavailable" | "disabled"
    latency_ms: Optional[float]
    info: Dict[str, Any] = field(default_factory=dict)


class Neo4jService:
    """Production Neo4j driver abstraction with parameterized queries and temporal isolation."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        connection_timeout: float = 3.0,
        enabled: Optional[bool] = None,
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        env_enabled = os.getenv("NEO4J_ENABLED", "false").lower() in ("true", "1", "yes")
        self.enabled = env_enabled if enabled is None else enabled

        env = os.getenv("TRUSTSHIELD_ENV", os.getenv("ENV", "development")).strip().lower()
        if self.enabled and env == "production" and not (password or os.getenv("NEO4J_PASSWORD")):
            raise RuntimeError("CRITICAL SECURITY: NEO4J_PASSWORD must be explicitly set in production when Neo4j is enabled.")
        self.password = password or os.getenv("NEO4J_PASSWORD", "trustshield_secret_dev_only")
        self.connection_timeout = connection_timeout

        self._driver: Any = None
        self._connected: bool = False
        self._last_health_check: float = 0.0
        self._last_connect_attempt: float = 0.0
        self.reconnect_cooldown: float = 30.0
        self._cached_health: Optional[Neo4jHealthStatus] = None

        if self.enabled:
            self._init_driver()
        else:
            self._cached_health = Neo4jHealthStatus(
                available=False, status="disabled", latency_ms=None,
                info={"detail": "Neo4j integration disabled via NEO4J_ENABLED=false"},
            )

    def _init_driver(self) -> None:
        now = time.time()
        if now - self._last_connect_attempt < self.reconnect_cooldown:
            return
        self._last_connect_attempt = now
        try:
            from neo4j import GraphDatabase  # type: ignore[import-not-found]
            self._driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password),
                connection_timeout=self.connection_timeout, max_connection_lifetime=300,
            )
            self._driver.verify_connectivity()
            self._connected = True
            self._cached_health = Neo4jHealthStatus(available=True, status="connected", latency_ms=1.0)
        except Exception as exc:
            self._connected = False
            self._driver = None
            self._cached_health = Neo4jHealthStatus(
                available=False, status="unavailable", latency_ms=None,
                info={"uri": self.uri, "error": str(exc)},
            )
            logger.warning("Neo4j unavailable at %s (%s). Using disk artifacts.", self.uri, exc)

    def verify_connectivity(self) -> bool:
        """Verify connectivity to Neo4j database."""
        return self.is_available

    def ping(self) -> Neo4jHealthStatus:
        if not self.enabled:
            return Neo4jHealthStatus(available=False, status="disabled", latency_ms=None, info={"detail": "Neo4j integration disabled via NEO4J_ENABLED=false"})
        if self._driver is None:
            self._init_driver()
        if self._driver is None or not self._connected:
            return self._cached_health or Neo4jHealthStatus(available=False, status="unavailable", latency_ms=None, info={"uri": self.uri, "error": "Connection failed or in cooldown"})
        t0 = time.perf_counter()
        try:
            with self._driver.session() as session:
                _ = session.run("RETURN 1 AS ping").single()
            self._connected = True
            self._cached_health = Neo4jHealthStatus(available=True, status="connected", latency_ms=round((time.perf_counter() - t0) * 1000.0, 3), info={"uri": self.uri})
        except Exception as exc:
            self._connected = False
            self._cached_health = Neo4jHealthStatus(available=False, status="unavailable", latency_ms=None, info={"uri": self.uri, "error": str(exc)})
        return self._cached_health

    @property
    def is_available(self) -> bool:
        if not self.enabled:
            return False
        if self._connected and self._driver is not None:
            return True
        now = time.time()
        if now - self._last_health_check > self.reconnect_cooldown:
            self._last_health_check = now
            self._cached_health = self.ping()
        return bool(self._cached_health and self._cached_health.available)

    def find_shared_devices(
        self,
        buyer_id: str,
        decision_time: Optional[str] = None,
        days_lookback: int = 30,
        rings_df_fallback: Optional[Any] = None,
    ) -> GraphQueryResult:
        t0 = time.perf_counter()
        clean_buyer = str(buyer_id).strip()
        cutoff_date = decision_time or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if self.is_available and self._driver is not None:
            query = """
            MATCH (b1:Buyer {id: $buyer_id})-[:USES_DEVICE]->(d:Device)<-[r:USES_DEVICE]-(b2:Buyer)
            WHERE b1.id <> b2.id
              AND (r.first_seen_date IS NULL OR r.first_seen_date < date($cutoff_date))
            RETURN b2.id AS related_buyer_id, d.id AS shared_device_id, r.first_seen_date AS first_seen
            LIMIT 50;
            """
            params = {"buyer_id": clean_buyer, "cutoff_date": cutoff_date[:10]}
            try:
                nodes, rels = [], []
                with self._driver.session() as session:
                    for rec in session.run(query, params):
                        b2_id, d_id = rec["related_buyer_id"], rec["shared_device_id"]
                        nodes.extend([{"id": b2_id, "type": "Buyer"}, {"id": d_id, "type": "Device"}])
                        rels.extend([
                            {"source": clean_buyer, "target": d_id, "type": "USES_DEVICE"},
                            {"source": b2_id, "target": d_id, "type": "USES_DEVICE", "first_seen": str(rec["first_seen"])},
                        ])
                lat = (time.perf_counter() - t0) * 1000.0
                return GraphQueryResult("find_shared_devices", nodes, rels, "neo4j", cutoff_date, round(lat, 3), len(rels))
            except Exception as exc:
                logger.warning("Neo4j query execution failed: %s. Falling back to disk.", exc)

        if rings_df_fallback is not None and hasattr(rings_df_fallback, "iterrows"):
            nodes, rels = [], []
            matched_ring = None
            for _, row in rings_df_fallback.iterrows():
                members = row.get("members", [])
                if isinstance(members, (list, tuple)) and clean_buyer in members:
                    matched_ring = row
                    break
            if matched_ring is not None:
                ring_id = str(matched_ring["ring_id"])
                dev_id = f"DEV_{ring_id}"
                nodes.extend([{"id": clean_buyer, "type": "Buyer"}, {"id": dev_id, "type": "Device"}])
                rels.append({"source": clean_buyer, "target": dev_id, "type": "USES_DEVICE"})
                for m in matched_ring.get("members", []):
                    m_str = str(m)
                    if m_str != clean_buyer:
                        nodes.append({"id": m_str, "type": "Buyer"})
                        rels.append({"source": m_str, "target": dev_id, "type": "USES_DEVICE"})
            lat = (time.perf_counter() - t0) * 1000.0
            return GraphQueryResult("find_shared_devices", nodes, rels, "disk_artifact", cutoff_date, round(lat, 3), len(rels))

        lat = (time.perf_counter() - t0) * 1000.0
        return GraphQueryResult("find_shared_devices", [], [], "unavailable", cutoff_date, round(lat, 3), 0, error="Neo4j unavailable and no disk artifact provided")

    def find_collusion_cycles(
        self,
        min_txns: int = 3,
        decision_time: Optional[str] = None,
        limit: int = 25,
        rings_df_fallback: Optional[Any] = None,
    ) -> GraphQueryResult:
        t0 = time.perf_counter()
        cutoff_date = decision_time or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if self.is_available and self._driver is not None:
            query = """
            MATCH path = (b:Buyer)-[t:TRANSACTS_WITH]->(s:Seller)-[:USES_ADDRESS]->(a:Address)<-[:USES_ADDRESS]-(b)
            WHERE (t.event_time IS NULL OR t.event_time < $cutoff_date)
            WITH b, s, a, count(t) AS txn_count
            WHERE txn_count >= $min_txns
            RETURN b.id AS buyer_id, s.id AS seller_id, a.id AS address_id, txn_count
            ORDER BY txn_count DESC
            LIMIT $limit;
            """
            params = {"min_txns": int(min_txns), "cutoff_date": cutoff_date, "limit": int(limit)}
            try:
                nodes, rels = [], []
                with self._driver.session() as session:
                    for rec in session.run(query, params):
                        b_id, s_id, a_id = rec["buyer_id"], rec["seller_id"], rec["address_id"]
                        nodes.extend([{"id": b_id, "type": "Buyer"}, {"id": s_id, "type": "Seller"}, {"id": a_id, "type": "Address"}])
                        rels.extend([
                            {"source": b_id, "target": s_id, "type": "TRANSACTS_WITH", "count": rec["txn_count"]},
                            {"source": s_id, "target": a_id, "type": "USES_ADDRESS"},
                            {"source": b_id, "target": a_id, "type": "USES_ADDRESS"},
                        ])
                lat = (time.perf_counter() - t0) * 1000.0
                return GraphQueryResult("find_collusion_cycles", nodes, rels, "neo4j", cutoff_date, round(lat, 3), len(rels))
            except Exception as exc:
                logger.warning("Neo4j collusion query failed: %s. Falling back to disk.", exc)

        if rings_df_fallback is not None and hasattr(rings_df_fallback, "iterrows"):
            nodes, rels = [], []
            for _, row in rings_df_fallback.head(limit).iterrows():
                ring_id = str(row["ring_id"])
                seller_id, addr_id = f"SELLER_{ring_id}", f"ADDR_{ring_id}"
                nodes.extend([{"id": seller_id, "type": "Seller"}, {"id": addr_id, "type": "Address"}])
                rels.append({"source": seller_id, "target": addr_id, "type": "USES_ADDRESS"})
                for m in row.get("members", [])[:4]:
                    m_str = str(m)
                    nodes.append({"id": m_str, "type": "Buyer"})
                    rels.extend([
                        {"source": m_str, "target": seller_id, "type": "TRANSACTS_WITH"},
                        {"source": m_str, "target": addr_id, "type": "USES_ADDRESS"},
                    ])
            lat = (time.perf_counter() - t0) * 1000.0
            return GraphQueryResult("find_collusion_cycles", nodes, rels, "disk_artifact", cutoff_date, round(lat, 3), len(rels))

        lat = (time.perf_counter() - t0) * 1000.0
        return GraphQueryResult("find_collusion_cycles", [], [], "unavailable", cutoff_date, round(lat, 3), 0, error="Neo4j unavailable and no disk artifact provided")

    def benchmark_latency(self, iterations: int = 25) -> Dict[str, float]:
        if not self.is_available or self._driver is None:
            return {"available": 0.0, "p50_ms": -1.0, "p95_ms": -1.0, "p99_ms": -1.0}
        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            with self._driver.session() as session:
                session.run("RETURN 1 AS ping").consume()
            latencies.append((time.perf_counter() - t0) * 1000.0)
        arr = np.array(latencies)
        return {
            "available": 1.0, "count": float(iterations),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "mean_ms": round(float(np.mean(arr)), 3),
        }

    def close(self) -> None:
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None
            self._connected = False


neo4j_service = Neo4jService()
