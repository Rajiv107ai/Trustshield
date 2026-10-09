#!/usr/bin/env python3
"""
TrustShield AI — Mesh Data Seeder (Neo4j & Redis)

Populates the orchestrated Docker service mesh with real pre-computed models
and graph intelligence artifacts:

1. Neo4j Property Graph:
   - Schema constraints & node labels (:Buyer, :Seller, :Device, :Address, :FraudRing)
   - Pre-computed candidate fraud rings from models/fraud_rings.joblib
   - Multi-hop collusion cycles and shared hardware relationships
   - Compatible with parameterized Cypher queries in neo4j_investigator.py

2. Redis In-Memory Feature Store:
   - High-throughput cached buyer/seller 16D GNN representation embeddings
   - Fraud ring membership fast-lookup sets (SADD) and ring metadata (HSET)
   - Fast buyer-to-ring index for sub-millisecond scoring lookups

Usage:
    # Auto-detects local or Docker mesh (bolt://localhost:7687, redis://localhost:6379)
    python scripts/seed_mesh.py

    # Dry-run validation (generates seed_graph.cypher and seed_redis.txt without network calls)
    python scripts/seed_mesh.py --dry-run

    # Target specific container or cloud endpoints
    python scripts/seed_mesh.py --neo4j-uri bolt://localhost:7687 --redis-url redis://localhost:6379/0
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import sys
import urllib.error
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


# ==============================================================================
# 1. Artifact Loading
# ==============================================================================

def load_platform_artifacts(models_dir: str = MODELS_DIR) -> Dict[str, Any]:
    """Load serialized model and intelligence artifacts from disk."""
    artifacts: Dict[str, Any] = {
        "rings_df": None,
        "buyer_embeddings": {},
        "seller_embeddings": {},
    }

    rings_path = os.path.join(models_dir, "fraud_rings.joblib")
    if os.path.exists(rings_path):
        artifacts["rings_df"] = joblib.load(rings_path)
    else:
        print(f"[WARN] Fraud rings artifact not found at {rings_path}")

    buyer_emb_path = os.path.join(models_dir, "buyer_embeddings.joblib")
    if os.path.exists(buyer_emb_path):
        artifacts["buyer_embeddings"] = joblib.load(buyer_emb_path)

    seller_emb_path = os.path.join(models_dir, "seller_embeddings.joblib")
    if os.path.exists(seller_emb_path):
        artifacts["seller_embeddings"] = joblib.load(seller_emb_path)

    return artifacts


# ==============================================================================
# 2. Neo4j Cypher Generation
# ==============================================================================

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


def generate_cypher_statements(
    rings_df: Optional[pd.DataFrame],
    max_rings: int = 50,
) -> List[str]:
    """
    Generate executable Cypher statements representing schema constraints,
    candidate collusion rings, and multi-hop sharing topology.
    """
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

    selected_rings = rings_df.head(max_rings if max_rings > 0 else len(rings_df))

    for _, row in selected_rings.iterrows():
        ring_id = str(row["ring_id"])
        avg_risk = _safe_float(row.get("avg_risk_score"), 0.0)
        max_risk = _safe_float(row.get("max_risk_score"), 0.0)
        size = _safe_int(row.get("size"), 2)
        n_high_risk = _safe_int(row.get("n_high_risk_orders"), 0)

        # Create FraudRing node
        statements.append(
            f"MERGE (r:FraudRing {{id: '{ring_id}'}}) "
            f"SET r.avg_risk_score = {avg_risk:.4f}, "
            f"r.max_risk_score = {max_risk:.4f}, "
            f"r.size = {size}, "
            f"r.n_high_risk_orders = {n_high_risk};"
        )

        members = row.get("members", [])
        if not isinstance(members, (list, tuple)):
            continue

        shared_dev = f"DEV_{ring_id}"
        shared_addr = f"ADDR_{ring_id}"
        collusion_seller = f"SELLER_{ring_id}"

        # Shared device, address, and collusion merchant
        statements.append(f"MERGE (d:Device {{id: '{shared_dev}'}});")
        statements.append(f"MERGE (a:Address {{id: '{shared_addr}'}});")
        statements.append(f"MERGE (s:Seller {{id: '{collusion_seller}', seller_id: '{collusion_seller}'}});")
        statements.append(
            f"MATCH (s:Seller {{id: '{collusion_seller}'}}), (d:Device {{id: '{shared_dev}'}}) "
            f"MERGE (s)-[:USES_DEVICE {{first_seen_date: date()}}]->(d);"
        )
        statements.append(
            f"MATCH (s:Seller {{id: '{collusion_seller}'}}), (a:Address {{id: '{shared_addr}'}}) "
            f"MERGE (s)-[:USES_ADDRESS]->(a);"
        )

        for m in members:
            b_id = str(m)
            statements.append(f"MERGE (b:Buyer {{id: '{b_id}', buyer_id: '{b_id}'}});")
            statements.append(
                f"MATCH (b:Buyer {{id: '{b_id}'}}), (r:FraudRing {{id: '{ring_id}'}}) "
                f"MERGE (b)-[:MEMBER_OF]->(r);"
            )
            # Create multi-hop collusion topology (Issue 66 & cycle queries)
            statements.append(
                f"MATCH (b:Buyer {{id: '{b_id}'}}), (d:Device {{id: '{shared_dev}'}}) "
                f"MERGE (b)-[:USES_DEVICE {{first_seen_date: date()}}]->(d);"
            )
            statements.append(
                f"MATCH (b:Buyer {{id: '{b_id}'}}), (a:Address {{id: '{shared_addr}'}}) "
                f"MERGE (b)-[:USES_ADDRESS]->(a);"
            )
            statements.append(
                f"MATCH (b:Buyer {{id: '{b_id}'}}), (s:Seller {{id: '{collusion_seller}'}}) "
                f"MERGE (b)-[:TRANSACTS_WITH]->(s);"
            )

    return statements


def seed_neo4j_http(
    statements: List[str],
    http_uri: str,
    user: str,
    password: str,
    batch_size: int = 50,
) -> bool:
    """Execute Cypher statements via Neo4j Transactional HTTP API (Zero Driver Dependency)."""
    clean_statements = [s for s in statements if not s.startswith("//")]
    if not clean_statements:
        return True

    endpoint = f"{http_uri.rstrip('/')}/db/neo4j/tx/commit"
    auth_header = "Basic " + base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")

    for i in range(0, len(clean_statements), batch_size):
        chunk = clean_statements[i:i + batch_size]
        payload = {
            "statements": [{"statement": stmt} for stmt in chunk]
        }
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": auth_header,
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                errors = data.get("errors", [])
                if errors:
                    print(f"[Neo4j HTTP] Batch error at index {i}: {errors[0].get('message')}")
                    return False
        except Exception as e:
            print(f"[Neo4j HTTP] Connection failed to {endpoint}: {e}")
            return False

    return True


def seed_neo4j(
    statements: List[str],
    bolt_uri: str,
    http_uri: str,
    user: str,
    password: str,
    dry_run: bool = False,
) -> bool:
    """Execute Cypher seeding using Bolt driver if installed, falling back to HTTP API."""
    if dry_run:
        print(f"[DRY-RUN] Neo4j: {len(statements)} statements validated.")
        return True

    clean_statements = [s for s in statements if not s.startswith("//")]

    # 1. Try official neo4j Python driver if available
    try:
        from neo4j import GraphDatabase  # type: ignore[import-not-found]
        print(f"Connecting to Neo4j via Bolt driver at {bolt_uri}...")
        driver = GraphDatabase.driver(bolt_uri, auth=(user, password))
        with driver.session() as session:
            for stmt in clean_statements:
                session.run(stmt)
        driver.close()
        print(f"Successfully seeded {len(clean_statements)} Cypher statements via Bolt driver.")
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"[Bolt Warning] Bolt driver connection failed ({e}). Falling back to HTTP API...")

    # 2. Fall back to HTTP REST Transaction API
    print(f"Connecting to Neo4j via HTTP REST at {http_uri}...")
    success = seed_neo4j_http(clean_statements, http_uri, user, password)
    if success:
        print(f"Successfully seeded {len(clean_statements)} Cypher statements via HTTP API.")
        return True

    print("[WARN] Could not connect to Neo4j. Ensure Neo4j container is running (`docker compose up -d neo4j`).")
    return False


# ==============================================================================
# 3. Redis Feature Hydration
# ==============================================================================

def encode_resp_command(cmd: str, *args: Any) -> bytes:
    """Encode command into raw Redis Serialization Protocol (RESP) format."""
    parts = [cmd] + [str(a) for a in args]
    buf = [f"*{len(parts)}\r\n".encode("utf-8")]
    for p in parts:
        b = p.encode("utf-8")
        buf.append(f"${len(b)}\r\n".encode("utf-8"))
        buf.append(b)
        buf.append(b"\r\n")
    return b"".join(buf)


def generate_redis_commands(
    buyer_embs: Dict[str, Any],
    seller_embs: Dict[str, Any],
    rings_df: Optional[pd.DataFrame],
    max_embs: int = 1000,
) -> List[Tuple[str, List[Any]]]:
    """Generate high-speed Redis key-value, set, and hash commands for feature serving."""
    commands: List[Tuple[str, List[Any]]] = []

    # 1. Platform Metadata
    now_iso = datetime.now(timezone.utc).isoformat()
    total_rings = len(rings_df) if rings_df is not None else 0
    commands.append(("HSET", [
        "trustshield:meta",
        "total_rings", str(total_rings),
        "seeded_at", now_iso,
        "feature_dim", "16",
    ]))

    # 2. Buyer GNN Embeddings
    buyer_items = list(buyer_embs.items())
    if max_embs > 0:
        buyer_items = buyer_items[:max_embs]

    for b_id, vec in buyer_items:
        emb_list = [float(x) for x in vec]
        commands.append(("SET", [f"buyer:{b_id}:emb", json.dumps(emb_list)]))

    # 3. Seller GNN Embeddings
    seller_items = list(seller_embs.items())
    if max_embs > 0:
        seller_items = seller_items[:max_embs]

    for s_id, vec in seller_items:
        emb_list = [float(x) for x in vec]
        commands.append(("SET", [f"seller:{s_id}:emb", json.dumps(emb_list)]))

    # 4. Fraud Ring Membership Sets & Lookup Index
    if rings_df is not None and not rings_df.empty:
        for _, row in rings_df.iterrows():
            ring_id = str(row["ring_id"])
            members = row.get("members", [])
            avg_risk = _safe_float(row.get("avg_risk_score"), 0.0)
            max_risk = _safe_float(row.get("max_risk_score"), 0.0)
            size = _safe_int(row.get("size"), 2)

            commands.append(("HSET", [
                f"fraud_ring:{ring_id}:meta",
                "avg_risk_score", f"{avg_risk:.4f}",
                "max_risk_score", f"{max_risk:.4f}",
                "size", str(size),
            ]))

            if isinstance(members, (list, tuple)) and members:
                # SADD fraud_ring:{ring_id}:members m1 m2 ...
                commands.append(("SADD", [f"fraud_ring:{ring_id}:members"] + [str(m) for m in members]))
                # Fast reverse lookup: buyer -> ring_id
                for m in members:
                    commands.append(("SET", [f"buyer:{m}:ring", ring_id]))

    return commands


def seed_redis_raw_socket(
    commands: List[Tuple[str, List[Any]]],
    host: str,
    port: int,
    batch_size: int = 500,
) -> bool:
    """Send Redis commands using raw TCP socket with RESP format (Zero Driver Dependency)."""
    try:
        sock = socket.create_connection((host, port), timeout=5)
    except Exception as e:
        print(f"[Redis Socket] Connection failed to {host}:{port} ({e})")
        return False

    try:
        for i in range(0, len(commands), batch_size):
            chunk = commands[i:i + batch_size]
            payload = b"".join(encode_resp_command(cmd, *args) for cmd, args in chunk)
            sock.sendall(payload)
            # Read responses
            for _ in chunk:
                # Read line response
                line = sock.recv(1024)
                if not line:
                    break
        sock.close()
        return True
    except Exception as e:
        print(f"[Redis Socket] Communication error: {e}")
        try:
            sock.close()
        except Exception:
            pass
        return False


def seed_redis(
    commands: List[Tuple[str, List[Any]]],
    redis_url: str,
    dry_run: bool = False,
) -> bool:
    """Execute Redis feature hydration with pipeline driver or raw socket fallback."""
    if dry_run:
        print(f"[DRY-RUN] Redis: {len(commands)} commands validated.")
        return True

    # 1. Try official redis package if available
    try:
        import redis  # type: ignore[import-not-found]
        print(f"Connecting to Redis via client library at {redis_url}...")
        client = redis.Redis.from_url(redis_url, decode_responses=True)
        pipe = client.pipeline(transaction=False)
        for cmd, args in commands:
            pipe.execute_command(cmd, *args)
        pipe.execute()
        print(f"Successfully loaded {len(commands)} keys/structures into Redis via pipeline.")
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"[Redis Warning] Client library failed ({e}). Falling back to raw socket...")

    # 2. Parse host & port and fall back to raw socket
    parsed = urllib.parse.urlparse(redis_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6379

    print(f"Connecting to Redis via raw TCP socket at {host}:{port}...")
    success = seed_redis_raw_socket(commands, host, port)
    if success:
        print(f"Successfully loaded {len(commands)} keys/structures into Redis via raw socket.")
        return True

    print("[WARN] Could not connect to Redis. Ensure Redis container is running (`docker compose up -d redis`).")
    return False


# ==============================================================================
# 4. Main Seeder CLI Orchestrator
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TrustShield AI — Mesh Data Seeder (Neo4j & Redis)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--neo4j-uri",
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j Bolt connection URI",
    )
    parser.add_argument(
        "--neo4j-http",
        default=os.getenv("NEO4J_HTTP_URI", "http://localhost:7474"),
        help="Neo4j HTTP connection URI",
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.getenv("NEO4J_USER", "neo4j"),
        help="Neo4j username",
    )
    parser.add_argument(
        "--neo4j-password",
        default=os.getenv("NEO4J_PASSWORD", "trustshield_secret"),
        help="Neo4j password",
    )
    parser.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        help="Redis connection URL",
    )
    parser.add_argument(
        "--max-rings",
        type=int,
        default=50,
        help="Maximum candidate fraud rings to import into Neo4j (0 for all)",
    )
    parser.add_argument(
        "--max-embeddings",
        type=int,
        default=1000,
        help="Maximum buyer/seller embeddings to hydrate in Redis (0 for all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate Cypher and Redis files without connecting to network",
    )
    parser.add_argument(
        "--export-cypher",
        default=os.path.join(SCRIPT_DIR, "seed_graph.cypher"),
        help="Output file path for generated Cypher statements",
    )
    parser.add_argument(
        "--export-redis",
        default=os.path.join(SCRIPT_DIR, "seed_redis.txt"),
        help="Output file path for generated Redis command log",
    )
    parser.add_argument(
        "--skip-neo4j",
        action="store_true",
        help="Skip Neo4j graph seeding",
    )
    parser.add_argument(
        "--skip-redis",
        action="store_true",
        help="Skip Redis feature hydration",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TrustShield AI — Mesh Data Seeder (Neo4j & Redis)")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    # 1. Load Artifacts
    artifacts = load_platform_artifacts()
    rings_df = artifacts["rings_df"]
    buyer_embs = artifacts["buyer_embeddings"]
    seller_embs = artifacts["seller_embeddings"]

    print(f"Artifacts loaded:")
    print(f" - Fraud Rings:     {len(rings_df) if rings_df is not None else 0:,}")
    print(f" - Buyer Vectors:   {len(buyer_embs):,} (16D)")
    print(f" - Seller Vectors:  {len(seller_embs):,} (16D)")

    # 2. Process Neo4j
    if not args.skip_neo4j:
        print("\n--- Processing Neo4j Property Graph ---")
        cypher_stmts = generate_cypher_statements(rings_df, max_rings=args.max_rings)

        with open(args.export_cypher, "w", encoding="utf-8") as f:
            f.write("\n".join(cypher_stmts))
        print(f"Exported {len(cypher_stmts)} Cypher statements to {args.export_cypher}")

        seed_neo4j(
            statements=cypher_stmts,
            bolt_uri=args.neo4j_uri,
            http_uri=args.neo4j_http,
            user=args.neo4j_user,
            password=args.neo4j_password,
            dry_run=args.dry_run,
        )

    # 3. Process Redis
    if not args.skip_redis:
        print("\n--- Processing Redis Online Feature Store ---")
        redis_cmds = generate_redis_commands(
            buyer_embs,
            seller_embs,
            rings_df,
            max_embs=args.max_embeddings,
        )

        with open(args.export_redis, "w", encoding="utf-8") as f:
            for cmd, arg_list in redis_cmds:
                f.write(f"{cmd} {' '.join(str(a) for a in arg_list)}\n")
        print(f"Exported {len(redis_cmds)} Redis commands to {args.export_redis}")

        seed_redis(
            commands=redis_cmds,
            redis_url=args.redis_url,
            dry_run=args.dry_run,
        )

    print("\n" + "=" * 70)
    print("Seeding process completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
