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


DEFAULT_HISTOGRAM_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


class _LabeledCounterProxy:
    """Lightweight proxy that increments a specific label slot in a MetricCounter."""

    def __init__(self, parent: "_MetricCounter", label_key: tuple):
        self._parent = parent
        self._key = label_key

    def inc(self, amount: float = 1.0) -> None:
        self._parent._counts[self._key] += amount
        logger.debug("metric_counter_inc", extra={"metric": self._parent.name, "amount": amount})


class _MetricCounter:
    def __init__(self, name: str):
        self.name = name
        self._counts: Dict[tuple, float] = defaultdict(float)

    def labels(self, **kwargs) -> "_LabeledCounterProxy":
        label_key = tuple(sorted(kwargs.items()))
        return _LabeledCounterProxy(self, label_key)

    def inc(self, amount: float = 1.0) -> None:
        self._counts[()] += amount
        logger.debug("metric_counter_inc", extra={"metric": self.name, "amount": amount})


class _MetricHistogram:
    def __init__(self, name: str, buckets: tuple = DEFAULT_HISTOGRAM_BUCKETS):
        self.name = name
        self.buckets = buckets
        self._observations: list[float] = []

    def labels(self, **kwargs) -> "_MetricHistogram":
        return self

    def observe(self, amount: float) -> None:
        self._observations.append(amount)
        if len(self._observations) > 10000:
            self._observations = self._observations[-5000:]
        logger.debug("metric_histogram_observe", extra={"metric": self.name, "value": amount})

    def export(self) -> list[str]:
        lines = [
            f"# HELP {self.name} Execution duration in seconds",
            f"# TYPE {self.name} histogram",
        ]
        total_count = len(self._observations)
        total_sum = sum(self._observations)
        for b in self.buckets:
            count = sum(1 for v in self._observations if v <= b)
            lines.append(f'{self.name}_bucket{{le="{b}"}} {count}')
        lines.append(f'{self.name}_bucket{{le="+Inf"}} {total_count}')
        lines.append(f'{self.name}_sum {total_sum:.6f}')
        lines.append(f'{self.name}_count {total_count}')
        return lines


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
        self.service_ready: int = 1
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
        """Text format export of in-memory metrics for Prometheus scraping."""
        lines = [
            "# HELP trustshield_requests_total Total HTTP requests handled",
            "# TYPE trustshield_requests_total counter",
            f"trustshield_requests_total {sum(self.requests_total._counts.values())}",
        ]
        for label_key, count in sorted(self.requests_total._counts.items()):
            if label_key:
                label_str = ",".join(f'{k}="{v}"' for k, v in label_key)
                lines.append(f"trustshield_requests_total{{{label_str}}} {count}")

        lines.extend([
            "# HELP trustshield_service_ready Service readiness state (1=ready, 0=not ready)",
            "# TYPE trustshield_service_ready gauge",
            f"trustshield_service_ready {self.service_ready}",
            "# HELP trustshield_decisions_total Operational decisions routed",
            "# TYPE trustshield_decisions_total counter",
        ])
        for dec, count in self._decisions.items():
            lines.append(f'trustshield_decisions_total{{decision="{dec}"}} {count}')

        lines.extend(self.request_duration.export())
        lines.extend(self.model_inference_seconds.export())
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
