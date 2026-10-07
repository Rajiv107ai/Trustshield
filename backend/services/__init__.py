"""TrustShield AI — Backend Service Layer Abstractions."""
from backend.services.redis_service import RedisService, redis_service
from backend.services.neo4j_service import Neo4jService, neo4j_service
from backend.services.investigation_service import InvestigationService, investigation_service
from backend.services.metrics_service import MetricsService, metrics_service
from backend.services.stream_service import transaction_event_generator, get_active_sse_clients

__all__ = [
    "RedisService",
    "redis_service",
    "Neo4jService",
    "neo4j_service",
    "InvestigationService",
    "investigation_service",
    "MetricsService",
    "metrics_service",
    "transaction_event_generator",
    "get_active_sse_clients",
]
