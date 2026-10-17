#!/usr/bin/env python3
"""
TrustShield AI — Mesh Data Seeder (Neo4j & Redis)
Populates the service mesh with pre-computed models and graph intelligence artifacts.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TRUSTSHIELD_PROJECT_DIR = os.path.join(PROJECT_ROOT, "trustshield_project")

if TRUSTSHIELD_PROJECT_DIR not in sys.path:
    sys.path.insert(0, TRUSTSHIELD_PROJECT_DIR)


def load_platform_artifacts(models_dir: str = MODELS_DIR) -> Dict[str, Any]:
    """Load serialized model and intelligence artifacts from disk."""
    artifacts: Dict[str, Any] = {"rings_df": None, "buyer_embeddings": {}, "seller_embeddings": {}}
    rings_path = os.path.join(models_dir, "fraud_rings.joblib")
    if os.path.exists(rings_path):
        artifacts["rings_df"] = joblib.load(rings_path)

    for key, fname in [("buyer_embeddings", "buyer_embeddings.joblib"), ("seller_embeddings", "seller_embeddings.joblib")]:
        fpath = os.path.join(models_dir, fname)
        if os.path.exists(fpath):
            artifacts[key] = joblib.load(fpath)
    return artifacts


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def generate_cypher_statements(rings_df: Optional[pd.DataFrame], max_rings: int = 50) -> List[str]:
    """Generate executable Cypher statements representing schema constraints and rings."""
    statements: List[str] = [
        "// --- TrustShield Schema Constraints ---",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (b:Buyer) REQUIRE b.id IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Seller) REQUIRE s.id IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Device) REQUIRE d.id IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Address) REQUIRE a.id IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (r:FraudRing) REQUIRE r.id IS UNIQUE;",
    ]
    if rings_df is None or rings_df.empty:
        return statements

    selected = rings_df.head(max_rings if max_rings > 0 else len(rings_df))
    for _, row in selected.iterrows():
        r_id = str(row["ring_id"])
        avg_r = _safe_float(row.get("avg_risk_score"), 0.0)
        max_r = _safe_float(row.get("max_risk_score"), 0.0)
        size = _safe_int(row.get("size"), 2)
        n_high = _safe_int(row.get("n_high_risk_orders"), 0)
        dev, addr, sel = f"DEV_{r_id}", f"ADDR_{r_id}", f"SELLER_{r_id}"

        statements.extend([
            f"MERGE (r:FraudRing {{id: '{r_id}'}}) SET r.avg_risk_score = {avg_r:.4f}, r.max_risk_score = {max_r:.4f}, r.size = {size}, r.n_high_risk_orders = {n_high};",
            f"MERGE (d:Device {{id: '{dev}'}});",
            f"MERGE (a:Address {{id: '{addr}'}});",
            f"MERGE (s:Seller {{id: '{sel}', seller_id: '{sel}'}});",
            f"MATCH (s:Seller {{id: '{sel}'}}), (d:Device {{id: '{dev}'}}) MERGE (s)-[:USES_DEVICE {{first_seen_date: date()}}]->(d);",
            f"MATCH (s:Seller {{id: '{sel}'}}), (a:Address {{id: '{addr}'}}) MERGE (s)-[:USES_ADDRESS]->(a);",
        ])
        members = row.get("members", [])
        if isinstance(members, (list, tuple)):
            for m in members:
                b_id = str(m)
                statements.extend([
                    f"MERGE (b:Buyer {{id: '{b_id}', buyer_id: '{b_id}'}});",
                    f"MATCH (b:Buyer {{id: '{b_id}'}}), (r:FraudRing {{id: '{r_id}'}}) MERGE (b)-[:MEMBER_OF]->(r);",
                    f"MATCH (b:Buyer {{id: '{b_id}'}}), (d:Device {{id: '{dev}'}}) MERGE (b)-[:USES_DEVICE {{first_seen_date: date()}}]->(d);",
                    f"MATCH (b:Buyer {{id: '{b_id}'}}), (a:Address {{id: '{addr}'}}) MERGE (b)-[:USES_ADDRESS]->(a);",
                    f"MATCH (b:Buyer {{id: '{b_id}'}}), (s:Seller {{id: '{sel}'}}) MERGE (b)-[:TRANSACTS_WITH]->(s);",
                ])
    return statements


def seed_neo4j_http(statements: List[str], http_uri: str, user: str, password: str, batch_size: int = 50) -> bool:
    """Execute Cypher statements via Neo4j Transactional HTTP API."""
    clean = [s for s in statements if not s.startswith("//")]
    if not clean:
        return True
    endpoint = f"{http_uri.rstrip('/')}/db/neo4j/tx/commit"
    auth = "Basic " + base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")

    for i in range(0, len(clean), batch_size):
        payload = json.dumps({"statements": [{"statement": stmt} for stmt in clean[i:i + batch_size]]}).encode("utf-8")
        req = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json", "Authorization": auth, "Accept": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("errors"):
                    return False
        except Exception:
            return False
    return True


def seed_neo4j(statements: List[str], bolt_uri: str, http_uri: str, user: str, password: str, dry_run: bool = False) -> bool:
    """Execute Cypher seeding using Bolt driver if installed, falling back to HTTP API."""
    if dry_run:
        return True
    clean = [s for s in statements if not s.startswith("//")]
    try:
        from neo4j import GraphDatabase  # type: ignore[import-not-found]
        driver = GraphDatabase.driver(bolt_uri, auth=(user, password))
        with driver.session() as session:
            for stmt in clean:
                session.run(stmt)
        driver.close()
        return True
    except Exception:
        pass
    return seed_neo4j_http(clean, http_uri, user, password)


def encode_resp_command(cmd: str, *args: Any) -> bytes:
    """Encode command into raw Redis Serialization Protocol (RESP) format."""
    parts = [cmd] + [str(a) for a in args]
    buf = [f"*{len(parts)}\r\n".encode("utf-8")]
    for p in parts:
        b = str(p).encode("utf-8")
        buf.extend([f"${len(b)}\r\n".encode("utf-8"), b, b"\r\n"])
    return b"".join(buf)


def generate_redis_commands(
    buyer_embs: Dict[str, Any],
    seller_embs: Dict[str, Any],
    rings_df: Optional[pd.DataFrame],
    max_embs: int = 1000,
) -> List[Tuple[str, List[Any]]]:
    """Generate high-speed Redis commands for feature serving."""
    commands: List[Tuple[str, List[Any]]] = [
        ("HSET", [
            "trustshield:meta",
            "total_rings", str(len(rings_df) if rings_df is not None else 0),
            "seeded_at", datetime.now(timezone.utc).isoformat(),
            "feature_dim", "16",
        ])
    ]
    b_items = list(buyer_embs.items())[:max_embs] if max_embs > 0 else list(buyer_embs.items())
    for b_id, vec in b_items:
        commands.append(("SET", [f"buyer:{b_id}:emb", json.dumps([float(x) for x in vec])]))

    s_items = list(seller_embs.items())[:max_embs] if max_embs > 0 else list(seller_embs.items())
    for s_id, vec in s_items:
        commands.append(("SET", [f"seller:{s_id}:emb", json.dumps([float(x) for x in vec])]))

    if rings_df is not None and not rings_df.empty:
        for _, row in rings_df.iterrows():
            r_id = str(row["ring_id"])
            members = row.get("members", [])
            avg_r = _safe_float(row.get("avg_risk_score"), 0.0)
            max_r = _safe_float(row.get("max_risk_score"), 0.0)
            size = _safe_int(row.get("size"), 2)
            commands.append(("HSET", [
                f"fraud_ring:{r_id}:meta",
                "avg_risk_score", f"{avg_r:.4f}",
                "max_risk_score", f"{max_r:.4f}",
                "size", str(size),
            ]))
            if isinstance(members, (list, tuple)) and members:
                commands.append(("SADD", [f"fraud_ring:{r_id}:members"] + [str(m) for m in members]))
                for m in members:
                    commands.append(("SET", [f"buyer:{m}:ring", r_id]))
    return commands


def seed_redis_raw_socket(commands: List[Tuple[str, List[Any]]], host: str, port: int, batch_size: int = 500) -> bool:
    """Send Redis commands using raw TCP socket with RESP format."""
    try:
        sock = socket.create_connection((host, port), timeout=5)
        for i in range(0, len(commands), batch_size):
            chunk = commands[i:i + batch_size]
            sock.sendall(b"".join(encode_resp_command(cmd, *args) for cmd, args in chunk))
            for _ in chunk:
                if not sock.recv(1024):
                    break
        sock.close()
        return True
    except Exception:
        return False


def seed_redis(commands: List[Tuple[str, List[Any]]], redis_url: str, dry_run: bool = False) -> bool:
    """Execute Redis feature hydration with pipeline driver or raw socket fallback."""
    if dry_run:
        return True
    try:
        import redis  # type: ignore[import-not-found]
        client = redis.Redis.from_url(redis_url, decode_responses=True)
        pipe = client.pipeline(transaction=False)
        for cmd, args in commands:
            pipe.execute_command(cmd, *args)
        pipe.execute()
        return True
    except Exception:
        pass

    parsed = urllib.parse.urlparse(redis_url)
    return seed_redis_raw_socket(commands, parsed.hostname or "localhost", parsed.port or 6379)


def main():
    parser = argparse.ArgumentParser(description="TrustShield AI — Mesh Data Seeder (Neo4j & Redis)")
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-http", default=os.getenv("NEO4J_HTTP_URI", "http://localhost:7474"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "trustshield_secret"))
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--max-rings", type=int, default=50)
    parser.add_argument("--max-embeddings", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--export-cypher", default=os.path.join(SCRIPT_DIR, "seed_graph.cypher"))
    parser.add_argument("--export-redis", default=os.path.join(SCRIPT_DIR, "seed_redis.txt"))
    parser.add_argument("--skip-neo4j", action="store_true")
    parser.add_argument("--skip-redis", action="store_true")
    args = parser.parse_args()

    artifacts = load_platform_artifacts()
    rings_df, buyer_embs, seller_embs = artifacts["rings_df"], artifacts["buyer_embeddings"], artifacts["seller_embeddings"]

    if not args.skip_neo4j:
        cypher_stmts = generate_cypher_statements(rings_df, max_rings=args.max_rings)
        with open(args.export_cypher, "w", encoding="utf-8") as f:
            f.write("\n".join(cypher_stmts))
        seed_neo4j(cypher_stmts, args.neo4j_uri, args.neo4j_http, args.neo4j_user, args.neo4j_password, dry_run=args.dry_run)

    if not args.skip_redis:
        redis_cmds = generate_redis_commands(buyer_embs, seller_embs, rings_df, max_embs=args.max_embeddings)
        with open(args.export_redis, "w", encoding="utf-8") as f:
            for cmd, arg_list in redis_cmds:
                f.write(f"{cmd} {' '.join(str(a) for a in arg_list)}\n")
        seed_redis(redis_cmds, args.redis_url, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
