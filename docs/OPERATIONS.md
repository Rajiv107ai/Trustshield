# TrustShield AI — Operations & Maintenance Runbook

<!-- BEGIN GENERATED -->

## 1. Monitoring & Health Probes

TrustShield exposes three standardized diagnostic and telemetry endpoints:

| Endpoint | Protocol | Purpose | Expected Response |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | Container liveness check | `{"status": "ok", "models_loaded": true, "n_rings": 708}` |
| `/ready` | `GET` | Service readiness probe | `{"status": "ready", "models_ready": true, ...}` |
| `/metrics` | `GET` | Prometheus telemetry stream | OpenMetrics format counters & histograms |

### Prometheus Key Alert Rules

```yaml
groups:
  - name: trustshield_alerts
    rules:
      - alert: TrustShieldHighP99Latency
        expr: histogram_quantile(0.99, rate(trustshield_request_duration_seconds_bucket[5m])) > 0.150
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "TrustShield inference latency p99 exceeds 150ms"

      - alert: TrustShieldBackendDegraded
        expr: trustshield_service_ready == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "TrustShield /ready probe failing"

      - alert: HighReviewQueueRatio
        expr: rate(trustshield_decisions_total{decision="REVIEW"}[10m]) / rate(trustshield_decisions_total[10m]) > 0.15
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Over 15% of transactions routed to manual review queue"
```

---

## 2. Operational Decision Policies

The `CanonicalTrustEngine` applies a 4-tier decision routing policy:

| Decision | Risk Probability | Action Triggered | SLA / Handling |
| :--- | :--- | :--- | :--- |
| **`ALLOW`** | $P(\text{fraud}) < 0.25$ | Immediate checkout success | Automated (<5ms) |
| **`REVIEW`** | $0.25 \le P(\text{fraud}) < 0.55$ | Queued for forensic investigator | Investigator review within 15 mins |
| **`HOLD`** | $0.55 \le P(\text{fraud}) < 0.85$ | Temporary payment auth hold | Step-up biometric / OTP challenge |
| **`BLOCK`** | $P(\text{fraud}) \ge 0.85$ or Fraud Ring member | Immediate transaction rejection | Payment declined |

### Cold-Start Detection
When buyers or sellers have fewer than 3 historic orders or account age $< 7$ days:
- `is_cold_start = true` is flagged.
- Point-in-time graph features default safely to `0.0`.
- Decision engine relies on calibrated tabular priors, identity age, and price discrepancies.

---

## 3. Retraining & Artifact Updating Procedure

When updating models with new transaction history:

```bash
# 1. Generate updated point-in-time graph snapshot and rings
python scripts/evaluate_naive_connected_components_20seeds.py

# 2. Re-fit models and probability calibrators on historical cutoff
python scripts/train_and_save_models.py

# 3. Re-freeze golden baseline fixtures
python scripts/generate_phase3_golden_baseline.py

# 4. Verify test suite passes 100%
pytest -q

# 5. Run end-to-end smoke test
python scripts/e2e_smoke_validation.py
```

After updating `models/*.joblib`, restart the FastAPI service or container:
```bash
docker compose restart trustshield-api
```
On startup, `lifespan` automatically loads the new model artifacts into memory with zero downtime if using rolling deployments.

---

## 4. Disaster Recovery & Troubleshooting

### Incident: High P99 Latency
1. Check CPU and memory utilization with `top` or `docker stats`.
2. Ensure `X-API-Key` is supplied to prevent unauthenticated retry storms.
3. If SHAP explanation calculation is heavy during peak volume, verify requests are using `/transaction/score` (fast path, <10ms) rather than `/transaction/explain` (heavy path with full TreeSHAP waterfall).

### Incident: Sudden Surge in REVIEW Queue
1. Check `/health` to confirm `calibrator_active: true`.
2. Inspect if a sudden influx of cold-start accounts is ordering high-ticket items.
3. If legitimate merchant flash sale is underway, temporarily adjust `C_REVIEW` policy threshold via environment variables without restarting code.

<!-- END GENERATED -->
