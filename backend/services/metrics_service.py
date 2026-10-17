"""TrustShield AI — Lightweight Structured Metrics & Observability.

Provides structured logging and in-memory execution timing metrics:
- Replaces heavyweight prometheus_client daemon collectors with clean structured logs
- Zero external dependencies
- Provides timing decorators and metric collectors for scoring latency and decision distributions
"""

from __future__ import annotations
import functools
import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("trustshield.metrics")


class _MetricCounter:
    def __init__(self, name: str):
        self.name = name
        self._counts: Dict[tuple, float] = defaultdict(float)

    def labels(self, **kwargs) -> "_MetricCounter":
        return self

    def inc(self, amount: float = 1.0) -> None:
        self._counts[()] += amount
        logger.debug("metric_counter_inc", extra={"metric": self.name, "amount": amount})


class _MetricHistogram:
    def __init__(self, name: str):
        self.name = name
        self._observations: list[float] = []

    def labels(self, **kwargs) -> "_MetricHistogram":
        return self

    def observe(self, amount: float) -> None:
        self._observations.append(amount)
        if len(self._observations) > 10000:
            self._observations = self._observations[-5000:]
        logger.debug("metric_histogram_observe", extra={"metric": self.name, "value": amount})


class MetricsService:
    """Lightweight structured metrics service with zero external dependencies."""

    def __init__(self):
        self.requests_total = _MetricCounter("trustshield_requests_total")
        self.request_duration = _MetricHistogram("trustshield_request_duration_seconds")
        self.decisions_total = _MetricCounter("trustshield_decisions_total")
        self.model_inference_seconds = _MetricHistogram("trustshield_model_inference_seconds")
        self.trust_engine_seconds = _MetricHistogram("trustshield_trust_engine_latency_seconds")
        self.dossier_duration = _MetricHistogram("trustshield_dossier_duration_seconds")
        self.dossier_total = _MetricCounter("trustshield_dossier_total")
        self._decisions: Dict[str, int] = defaultdict(int)

    def record_decision(self, decision: str, risk_label: str, model_version: str) -> None:
        self._decisions[decision] += 1
        logger.info(
            "operational_decision_recorded",
            extra={
                "decision": decision,
                "risk_label": risk_label,
                "model_version": model_version,
            },
        )

    def export_metrics(self) -> str:
        """Text format export of in-memory metrics for basic health/monitoring scrapes."""
        lines = [
            "# HELP trustshield_requests_total Total HTTP requests handled",
            "# TYPE trustshield_requests_total counter",
            f"trustshield_requests_total {sum(self.requests_total._counts.values())}",
            "# HELP trustshield_decisions_total Operational decisions routed",
            "# TYPE trustshield_decisions_total counter",
        ]
        for dec, count in self._decisions.items():
            lines.append(f'trustshield_decisions_total{{decision="{dec}"}} {count}')
        return "\n".join(lines) + "\n"


def timing_metric(metric_name: Optional[str] = None) -> Callable:
    """Decorator to measure and log execution duration of a function."""
    def decorator(func: Callable) -> Callable:
        name = metric_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - t0
                logger.info("execution_timed", extra={"metric": name, "elapsed_seconds": round(elapsed, 6)})
        return wrapper
    return decorator


metrics_service = MetricsService()
