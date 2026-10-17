"""TrustShield AI — Backend Service Layer Abstractions."""
from backend.services.investigation_service import InvestigationService, investigation_service
from backend.services.metrics_service import MetricsService, metrics_service, timing_metric

__all__ = [
    "InvestigationService",
    "investigation_service",
    "MetricsService",
    "metrics_service",
    "timing_metric",
]
