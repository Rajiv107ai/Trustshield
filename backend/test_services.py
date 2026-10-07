"""Unit tests for Redis and Neo4j Service Abstractions and Fallbacks."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from backend.services.redis_service import RedisService, EmbeddingResult
from backend.services.neo4j_service import Neo4jService, GraphQueryResult


# ==============================================================================
# 1. Redis Service Tests
# ==============================================================================

def test_redis_service_offline_fallback():
    """Verify RedisService gracefully falls back to disk artifact without crashing."""
    service = RedisService(redis_url="redis://nonexistent-host:9999/0", connect_timeout=0.1)
    assert service.is_available is False

    disk_embeddings = {
        "BUYER_100": np.ones(16, dtype=np.float32) * 0.5,
    }

    # Successful disk fallback
    res = service.get_embedding("buyer", "BUYER_100", disk_fallback_dict=disk_embeddings)
    assert isinstance(res, EmbeddingResult)
    assert res.found is True
    assert res.source == "disk_artifact"
    assert res.vector is not None
    assert len(res.vector) == 16
    assert pytest.approx(res.vector[0]) == 0.5

    # Missing entity returns not_found, NEVER arbitrary entity 0
    missing = service.get_embedding("buyer", "BUYER_999", disk_fallback_dict=disk_embeddings)
    assert missing.found is False
    assert missing.vector is None
    assert missing.source == "not_found"


def test_redis_service_with_mock_client():
    """Verify live Redis client interaction and caching."""
    service = RedisService(enabled=False)
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.get.return_value = '[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]'

    service._client = mock_client
    service._connected = True
    service.enabled = True

    # Get from Redis
    res = service.get_embedding("buyer", "BUYER_001")
    assert res.found is True
    assert res.source == "redis"
    assert res.vector is not None
    assert len(res.vector) == 16
    assert pytest.approx(res.vector[0]) == 0.1

    # Cache into Redis
    test_vec = np.ones(16, dtype=np.float32) * 0.75
    ok = service.cache_embedding("buyer", "BUYER_002", test_vec, ttl_seconds=300)
    assert ok is True
    assert mock_client.set.called


def test_redis_dimension_validation():
    """Verify vector dimension mismatch raises ValueError."""
    service = RedisService(enabled=False)
    service._client = MagicMock()
    service._connected = True
    service.enabled = True

    # Buyer requires 16D
    invalid_vec = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="expected 16"):
        service.cache_embedding("buyer", "BUYER_003", invalid_vec)


# ==============================================================================
# 2. Neo4j Service Tests
# ==============================================================================

def test_neo4j_service_offline_fallback():
    """Verify Neo4jService gracefully falls back to fraud_rings dataframe."""
    service = Neo4jService(uri="bolt://nonexistent-host:9999", connection_timeout=0.1)
    assert service.is_available is False

    mock_rings_df = pd.DataFrame([
        {
            "ring_id": "RING_TEST_42",
            "size": 3,
            "avg_risk_score": 0.88,
            "members": ["BUYER_MEMBER_A", "BUYER_MEMBER_B"],
        }
    ])

    res = service.find_shared_devices("BUYER_MEMBER_A", rings_df_fallback=mock_rings_df)
    assert isinstance(res, GraphQueryResult)
    assert res.graph_source == "disk_artifact"
    assert res.record_count > 0
    assert any(n["id"] == "DEV_RING_TEST_42" for n in res.nodes)
    assert any(n["id"] == "BUYER_MEMBER_B" for n in res.nodes)


def test_neo4j_temporal_safety_parameterization():
    """Verify all queries enforce parameterized Cypher with temporal cutoff."""
    service = Neo4jService(enabled=False)
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    service._driver = mock_driver
    service._connected = True
    service.enabled = True

    # Call with explicit decision time
    decision_time = "2025-06-01T12:00:00"
    service.find_shared_devices("BUYER_X", decision_time=decision_time)

    assert mock_session.run.called
    cypher_call = mock_session.run.call_args
    query_str, params = cypher_call[0][0], cypher_call[0][1]

    # Verify parameterized binding (NO string interpolation of ID or date)
    assert "$buyer_id" in query_str
    assert "$cutoff_date" in query_str
    assert params["buyer_id"] == "BUYER_X"
    assert params["cutoff_date"] == "2025-06-01"
