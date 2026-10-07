"""
Unit tests for scripts/seed_mesh.py (Mesh Data Seeder for Neo4j & Redis).
"""

from __future__ import annotations

import os
import sys
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from seed_mesh import (
    load_platform_artifacts,
    generate_cypher_statements,
    generate_redis_commands,
    encode_resp_command,
    seed_neo4j,
    seed_redis,
)


class TestMeshSeeder:
    """Test suite for Neo4j & Redis mesh seeder logic."""

    def test_load_platform_artifacts(self):
        artifacts = load_platform_artifacts()
        assert "rings_df" in artifacts
        assert "buyer_embeddings" in artifacts
        assert "seller_embeddings" in artifacts
        if artifacts["rings_df"] is not None:
            assert isinstance(artifacts["rings_df"], pd.DataFrame)
            assert not artifacts["rings_df"].empty

    def test_generate_cypher_statements_schema_and_rings(self):
        df_sample = pd.DataFrame([
            {
                "ring_id": "RING_TEST_01",
                "members": ["BUYER_01", "BUYER_02"],
                "avg_risk_score": 0.85,
                "max_risk_score": 0.95,
                "size": 2,
                "n_high_risk_orders": 10,
            }
        ])

        stmts = generate_cypher_statements(df_sample, max_rings=1)
        assert len(stmts) > 5

        # Check constraint statements
        assert any("CREATE CONSTRAINT" in s and "Buyer" in s for s in stmts)
        assert any("CREATE CONSTRAINT" in s and "FraudRing" in s for s in stmts)

        # Check ring merge statement
        assert any("RING_TEST_01" in s and "MERGE (r:FraudRing" in s for s in stmts)

        # Check collusion topology statements
        assert any("DEV_RING_TEST_01" in s for s in stmts)
        assert any("ADDR_RING_TEST_01" in s for s in stmts)
        assert any("SELLER_RING_TEST_01" in s for s in stmts)
        assert any("USES_DEVICE" in s for s in stmts)
        assert any("TRANSACTS_WITH" in s for s in stmts)

    def test_generate_redis_commands(self):
        buyer_embs = {"BUYER_01": [0.1] * 16, "BUYER_02": [0.2] * 16}
        seller_embs = {"SELLER_01": [0.5] * 16}
        df_sample = pd.DataFrame([
            {
                "ring_id": "RING_TEST_01",
                "members": ["BUYER_01", "BUYER_02"],
                "avg_risk_score": 0.85,
                "max_risk_score": 0.95,
                "size": 2,
            }
        ])

        cmds = generate_redis_commands(buyer_embs, seller_embs, df_sample, max_embs=2)

        # Validate meta key
        assert any(cmd[0] == "HSET" and cmd[1][0] == "trustshield:meta" for cmd in cmds)

        # Validate embeddings
        assert any(cmd[0] == "SET" and cmd[1][0] == "buyer:BUYER_01:emb" for cmd in cmds)
        assert any(cmd[0] == "SET" and cmd[1][0] == "seller:SELLER_01:emb" for cmd in cmds)

        # Validate ring sets and reverse index
        assert any(cmd[0] == "SADD" and cmd[1][0] == "fraud_ring:RING_TEST_01:members" for cmd in cmds)
        assert any(cmd[0] == "SET" and cmd[1][0] == "buyer:BUYER_01:ring" and cmd[1][1] == "RING_TEST_01" for cmd in cmds)

    def test_encode_resp_command(self):
        encoded = encode_resp_command("SET", "key1", "val1")
        expected = b"*3\r\n$3\r\nSET\r\n$4\r\nkey1\r\n$4\r\nval1\r\n"
        assert encoded == expected

    def test_dry_run_execution(self):
        stmts = ["CREATE CONSTRAINT IF NOT EXISTS FOR (b:Buyer) REQUIRE b.id IS UNIQUE;"]
        cmds = [("SET", ["test_key", "test_val"])]

        assert seed_neo4j(stmts, "bolt://dummy:7687", "http://dummy:7474", "user", "pass", dry_run=True) is True
        assert seed_redis(cmds, "redis://dummy:6379/0", dry_run=True) is True
