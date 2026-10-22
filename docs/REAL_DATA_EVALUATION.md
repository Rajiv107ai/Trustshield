# Independent Real-Data Evaluation

TrustShield keeps the existing synthetic research pipeline intact. This
evaluation lane accepts only a normalized prediction/label table from private
storage and writes aggregate metrics plus a SHA-256 input manifest.

No raw event rows, customer identifiers, names, addresses, payment data, or
credentials belong in the repository or in the generated report.

## Required input columns

```text
event_id
event_time
score
label
label_time
outcome_window_days
amount
```

Optional approved subgroup columns include `merchant_age_band`, `region`,
`account_age_band`, `device_class`, `listing_category`, `payment_channel`, and
`fraud_type`. Subgroup collection requires an approved operational/legal basis.

## Evaluation command

```bash
python scripts/evaluate_independent_data.py \
  --input /private/evaluation/real_eval_v1.csv \
  --output reports/real_eval_v1_summary.json \
  --as-of 2026-10-20T00:00:00Z \
  --evaluation-type real_historical_temporal \
  --subgroup-field merchant_age_band
```

Labels are rejected unless their declared outcome window has matured by the
specified `--as-of` cutoff. The report includes PR-AUC, ROC-AUC, ECE, Brier
score, conformal coverage when supplied, review-budget recall, false-positive
rate, and a fraud-dollar prevention proxy.

The report is evidence for evaluation only. It does not automatically retrain
or promote a model.

Aggregate monitoring reports can be persisted through
`backend/services/monitoring_store.py`. Its local SQLite backend is for
development/reproducibility; production requires an externally managed
PostgreSQL monitoring store and never accepts row-level event data.
