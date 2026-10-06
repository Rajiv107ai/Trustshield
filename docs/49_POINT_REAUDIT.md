# TrustShield - 49-Point Fix Pack Re-Audit (Phase 1)

Audit date: 2026-10-05
Baseline: 71/71 tests passing (18 CLIP deselected), Python 3.14.7
Fix pack: trustshield_complete_fix_pack.zip v1.0.0-review-pack
Principle: Do NOT implement a fix merely because it exists in the ZIP.

CLASSIFICATION LEGEND
REQUIRED: Real gap. Must fix.
ALREADY_IMPLEMENTED: Repo already does this correctly.
PARTIALLY_IMPLEMENTED: Gap exists but repo has the foundation.
OPTIONAL: Would add value but not critical.
NOT_APPLICABLE: Fix targets something that does not apply.
WRONG_ASSUMPTION: Fix based on incorrect assumption about the codebase.

==============================================================================
MODULE: trust_engine/engine.py
==============================================================================

FIX-01: Unified Trust Engine with ALLOW/REVIEW/HOLD/BLOCK
Classification: REQUIRED | Severity: HIGH
Evidence: backend/main.py _risk_label() returns low/medium/high only. No decision layer.
Fix pack quality: GOOD. TrustEngine.score() takes ComponentRisk list, returns TrustResult.
WARNING: Default thresholds (allow=0.20, review=0.50, hold=0.75) are ARBITRARY - must be
validated on validation set before deploying.
Action: Integrate TrustEngine into backend/main.py.
Files: backend/main.py, backend/schemas.py, new trustshield_project/trust_engine.py
Retraining: NO

FIX-02: Weighted aggregation of component scores
Classification: REQUIRED | Severity: MEDIUM
Evidence: API uses Phase5 if loaded else Phase3 - no validated weighting.
Fix pack quality: ACCEPTABLE. Default equal weights are also arbitrary.
Action: Use TrustEngine with equal weights initially. Document weights need validation.
Files: backend/main.py
Retraining: NO

FIX-03: model_disagreement output field
Classification: REQUIRED | Severity: LOW-MEDIUM
Evidence: No model_disagreement in TransactionScoreResponse.
Fix pack quality: GOOD. TrustResult.model_disagreement = max(p) - min(p).
Action: Add field after Fix-01. Implement with Trust Engine integration.
Files: backend/schemas.py, backend/main.py
Retraining: NO

FIX-04: Feedback loop guard (model outputs as future features)
Classification: PARTIALLY_IMPLEMENTED | Severity: MEDIUM
Evidence: leakage_audit() BANNED_COLS covers fraud-label cols only, not model-output cols.
Fix pack quality: GOOD - FORBIDDEN_MODEL_DERIVED_FEATURES set is correct.
Action: Add model output column names to BANNED_COLS in baseline_model.leakage_audit().
Files: trustshield_project/baseline_model.py
Retraining: NO

FIX-05: Reason codes in /transaction/score response
Classification: REQUIRED | Severity: MEDIUM
Evidence: Only /listing/analyze has investigator_narrative. No reason_codes on /transaction/score.
Fix pack quality: MINIMAL (wraps TrustResult.reason_codes).
Action: Add reason_codes: list[str] to TransactionScoreResponse. Populate from TrustEngine.
Files: backend/schemas.py, backend/main.py
Retraining: NO

==============================================================================
MODULE: calibration/calibrator.py
==============================================================================

FIX-06: Probability calibration
Classification: REQUIRED | Severity: MEDIUM-HIGH
Evidence: Raw uncalibrated probabilities everywhere. Return fraud cost-optimal threshold
of 0.01 is suspiciously extreme - possibly artifact of uncalibrated outputs.
Fix pack quality: WRONG IMPLEMENTATION. Fix pack provides custom PlattCalibrator gradient
descent. sklearn.calibration.CalibratedClassifierCV is better and already available.
Action: Use sklearn CalibratedClassifierCV(method=isotonic, cv=prefit) on validation data.
Do NOT use fix pack custom PlattCalibrator.
Files: baseline_model.py, phase2_specialized_models.py, phase5_hybrid_model.py
Retraining: NO (calibration is fit post-training on validation predictions)

FIX-07: Cost threshold search with calibrated probabilities
Classification: PARTIALLY_IMPLEMENTED | Severity: MEDIUM
Evidence: find_cost_optimal_threshold() operates on uncalibrated scores.
Fix pack quality: EQUIVALENT to existing utils.py. No improvement needed.
Action: After Fix-06, re-run threshold search on calibrated probabilities.
Files: trustshield_project/utils.py (minor extension)
Retraining: NO

==============================================================================
MODULE: graph/temporal_history.py
==============================================================================

FIX-08: Strict temporal exclusion of equal timestamps
Classification: ALREADY_IMPLEMENTED | Severity: -
Evidence: _cumulative_count_asof() uses allow_exact_matches=False.
TestSameTimestampOrderLeakage passes in test_leakage.py.
Action: NO ACTION. Existing implementation is correct and tested.
Retraining: NO

FIX-09: assert_strict_temporal() validation function
Classification: OPTIONAL | Severity: LOW
Evidence: leakage_audit() checks counts/columns but not timestamp strict-before.
Fix pack quality: GOOD - useful as development-time defense-in-depth.
Action: Add assert_strict_temporal() to test_leakage.py. Low cost, good documentation.
Files: trustshield_project/test_leakage.py
Retraining: NO

==============================================================================
MODULE: graph/typed_graph.py
==============================================================================

FIX-10: TypedMultiGraph for relationship semantics
Classification: PARTIALLY_IMPLEMENTED | Severity: MEDIUM
Evidence: Graph is homogeneous. Edge kind as attribute only. GNN has no typed nodes.
Fix pack quality: GOOD infrastructure for CPU-side analysis. Does NOT solve HeteroGNN.
CRITICAL NOTE: TypedMultiGraph has no torch_geometric imports - it is NOT a GNN fix.
A real heterogeneous GNN requires PyG HeteroData (separate experiment).
Action: Adopt TypedMultiGraph for ring detection analysis only. Do NOT claim it improves
the GNN without running a HeteroData experiment.
Files: trustshield_project/graph_features.py (ring detection path)
Retraining: NO (for ring detection use). YES if used for GNN.

==============================================================================
MODULE: graph/cold_start.py
==============================================================================

FIX-11: Cold-start detection and confidence handling
Classification: PARTIALLY_IMPLEMENTED | Severity: MEDIUM
Evidence: API cold-start entities get zero embeddings silently. No confidence reduction exposed.
Fix pack quality: GOOD minimal implementation. min_history=3 is arbitrary but reasonable.
Action: Integrate is_cold_start() into API. Add cold_start: bool and confidence: float
to TransactionScoreResponse using buyer_orders_before feature already in vector.
Files: backend/schemas.py, backend/main.py
Retraining: NO

==============================================================================
MODULE: graph/online_features.py
==============================================================================

FIX-12: Online graph context for real-time feature lookup
Classification: PARTIALLY_IMPLEMENTED | Severity: LOW now / HIGH for production
Evidence: API accepts pre-computed features, defaults missing to 0. No live graph lookup.
Fix pack quality: ACCEPTABLE but depends on TypedMultiGraph (Fix-10, not yet done).
Action: DEFER. Requires persistent graph store (Redis or Neo4j). Do not add without
a real graph store architecture decision.
Files: backend/main.py (future)
Retraining: NO

==============================================================================
MODULE: validation/schema.py
==============================================================================

FIX-13: Input validation with field range checks
Classification: REQUIRED | Severity: MEDIUM
Evidence: TransactionScoreRequest: amount: Optional[float] = None - no ge/le validators.
Negative amounts produce nonsense scores silently.
Fix pack quality: REDUNDANT. Pydantic v2 Field(ge=0) is the correct approach.
Action: Use Pydantic v2 Field validators - NOT the fix pack custom validator.
Add ge=0 to amount, ge=0/le=1 to buyer_return_rate_before, seller_return_rate_before.
Files: backend/schemas.py
Retraining: NO

FIX-14: Missing value indicator flags
Classification: OPTIONAL | Severity: LOW
Evidence: Missing features default silently to 0. For ratio features this is ambiguous.
Fix pack quality: CORRECT but adds N binary columns requiring model retraining.
Action: DEFER until retraining is planned. Document known missingness ambiguity.
Files: Would require retraining
Retraining: YES

==============================================================================
MODULE: evaluation/metrics.py
==============================================================================

FIX-15: review_rate and FP/1000 metrics
Classification: REQUIRED | Severity: MEDIUM
Evidence: utils.py evaluate() has no review_rate or FP/1000.
Fix pack quality: GOOD. binary_metrics() includes both.
Action: Add review_rate and false_positives_per_1000 to utils.py evaluate().
Also add per-fraud-type metric breakdown (fake_listing, return_abuse, coordinated, collusion).
Files: trustshield_project/utils.py
Retraining: NO

FIX-16: group_metrics() for segment evaluation
Classification: OPTIONAL | Severity: LOW-MEDIUM
Evidence: No groupby evaluation in codebase.
Fix pack quality: GOOD.
Action: Add to utils.py. Stratify by seller_new/established, buyer_new/established.
Files: trustshield_project/utils.py
Retraining: NO

==============================================================================
MODULE: evaluation/fairness.py
==============================================================================

FIX-17: TPR/FPR/selection_rate by segment
Classification: OPTIONAL | Severity: LOW
Evidence: No fairness/segment analysis. No real protected attributes in synthetic data.
Fix pack quality: CORRECT.
Action: Implement for age-based segments only. Label as segment robustness, not fairness.
Files: trustshield_project/utils.py or new experiments/ script
Retraining: NO

==============================================================================
MODULE: evaluation/adversarial.py
==============================================================================

FIX-18: Bounded feature evasion testing
Classification: OPTIONAL | Severity: MEDIUM
Evidence: No adversarial testing.
Fix pack quality: CORRECT but generic (plus-minus-5-percent). Domain-specific is better.
Action: Use as starting point. Augment with: (1) increase days_to_return for return fraud
evasion, (2) raise price for fake listing evasion, (3) spread ring burst over longer window.
Do NOT use generic perturbations as primary evasion test.
Files: New experiments/evasion_tests.py
Retraining: NO

==============================================================================
MODULE: evaluation/bootstrap.py
==============================================================================

FIX-19: Bootstrap confidence intervals
Classification: REQUIRED | Severity: MEDIUM
Evidence: All metrics are point estimates. No CI.
Fix pack quality: CORRECT.
Action: Add bootstrap_ci() to evaluation pipeline. Report 95% CI for ROC-AUC and PR-AUC.
Files: trustshield_project/utils.py, graph_features.py
Retraining: NO

==============================================================================
MODULE: evaluation/experiments.py
==============================================================================

FIX-20: Multi-seed evaluation
Classification: REQUIRED | Severity: HIGH
Evidence: All results from seed=42 only. Single-seed is not defensible scientifically.
Fix pack quality: CORRECT. multi_seed() and summarize_seed_runs() are clean.
Action: Run Phase 3 XGBoost with seeds [42,43,44,45,46]. Report mean+-std for ROC-AUC/PR-AUC.
Requires parameterizing train_and_eval() to accept seed.
Files: trustshield_project/graph_features.py, new experiments/multi_seed_eval.py
Retraining: NO (evaluation change)

FIX-21: Prevalence sensitivity experiment
Classification: REQUIRED | Severity: MEDIUM-HIGH
Evidence: Only 7-percent fraud tested. PR-AUC is highly sensitive to prevalence.
Fix pack quality: CORRECT but reports at fixed thresholds not swept prevalence.
Action: Resample test set to 1%, 3%, 7%, 15% fraud. Measure PR-AUC degradation.
Files: New experiments/prevalence_eval.py
Retraining: NO

FIX-22: Distribution/regime shift experiment
Classification: REQUIRED | Severity: HIGH
Evidence: No experiment measuring degradation when fraud behavior changes.
Fix pack quality: MINIMAL. distribution_shift_report() segments by group, not changed regime.
Action: Train on standard regime, test on modified (e.g., return fraud with days_to_return
5-15 instead of 1-5). More valuable than fix pack group-based approach.
Files: trustshield_project/fraud_injection.py (parameterize), new experiments/regime_shift_eval.py
Retraining: NO (evaluation change)

FIX-23: Temporal drift per period
Classification: OPTIONAL | Severity: MEDIUM
Evidence: Test set is holdout by date but no per-month breakdown.
Fix pack quality: CORRECT.
Action: Split test period into monthly buckets, report ROC-AUC per bucket.
Files: New experiments/temporal_drift_eval.py
Retraining: NO

FIX-24: Cold-start evaluation
Classification: OPTIONAL | Severity: MEDIUM
Evidence: No cold-start specific evaluation.
Fix pack quality: CORRECT.
Action: Evaluate separately for buyers with buyer_orders_before < 3 vs >= 3.
Files: trustshield_project/test_graph_and_models.py or experiments/
Retraining: NO

==============================================================================
MODULE: evaluation/operational.py
==============================================================================

FIX-25: Operational metrics (fraud_dollars_caught, review_rate)
Classification: OPTIONAL | Severity: MEDIUM
Evidence: Only classification metrics reported. No business-case numbers.
Fix pack quality: CORRECT.
Action: Add fraud_dollars_caught and review_or_hold_rate to evaluation. Amount column exists.
Files: trustshield_project/utils.py
Retraining: NO

==============================================================================
MODULE: versioning/metadata.py
==============================================================================

FIX-26: Model artifact hashing and metadata
Classification: OPTIONAL | Severity: LOW
Evidence: No metadata file written with joblib artifacts. No version in /health beyond label.
Fix pack quality: GOOD.
Action: Add metadata writing to scripts/train_and_save_models.py. Expose in /health.
Files: scripts/train_and_save_models.py, backend/main.py
Retraining: NO

==============================================================================
MODULE: security/api_validation.py
==============================================================================

FIX-27: Transaction payload validation (overlapping with FIX-13)
Classification: ALREADY_COVERED | Severity: -
Evidence: Duplicate of FIX-13. Fix pack imports validate_record from validation.schema.
Action: Implement FIX-13 with Pydantic validators instead. Do not add separate layer.
Retraining: NO

==============================================================================
GAPS NOT IN THE FIX PACK - FOUND DURING AUDIT
==============================================================================

FIX-28: CRITICAL - Temporal leakage in run_phase_3() ring detection path
Classification: REQUIRED | Severity: HIGH
Evidence: graph_features.py L268: build_relationship_graph() called WITHOUT cutoff_date.
Uses ALL future-dated sharing relationships for ring detection during model evaluation.
Compare: phase5_hybrid_model.py L236-237 correctly passes cutoff_date=TRAIN_END and VAL_END.
Action: Fix graph_features.py L268-270. Pass cutoff_date parameter.
Files: trustshield_project/graph_features.py
Test required: YES - add to test_leakage.py
Retraining: POSSIBLY - must compare before/after metrics

FIX-29: No python -m compileall in CI
Classification: OPTIONAL | Severity: LOW
Evidence: No GitHub Actions, no Makefile, no pre-commit hook.
Action: Add compileall step to pytest.ini or Makefile.
Files: pytest.ini or new Makefile
Retraining: NO

FIX-30: Category median (listing features) - verified correct
Classification: ALREADY_IMPLEMENTED | Severity: -
Evidence: phase2_specialized_models.py L100-102 correctly uses train_mask.
Action: NO ACTION.
Retraining: NO

==============================================================================
WRONG ASSUMPTIONS IN FIX PACK
==============================================================================

FIX-31: test_api.py imports backend.trust_api_example - DOES NOT EXIST
Classification: WRONG_ASSUMPTION
Evidence: tests/test_api.py line 1: from backend.trust_api_example import health, score_transaction
No such module in repo. This test would fail immediately.
Action: DO NOT merge test_api.py from fix pack.

FIX-32: TypedMultiGraph is NOT a HeteroGNN solution
Classification: WRONG_ASSUMPTION
Evidence: typed_graph.py has no torch_geometric imports. It is CPU-side only.
Real HeteroGNN requires PyG HeteroData - a separate experiment.
Action: Scope TypedMultiGraph to ring detection only. Do not conflate with GNN improvement.

FIX-33: Custom PlattCalibrator when sklearn exists
Classification: WRONG_ASSUMPTION
Evidence: calibrator.py custom gradient descent when CalibratedClassifierCV is available.
Action: Use sklearn instead. Fix pack calibrator is inferior and adds untested code.

==============================================================================
IMPLEMENTATION PRIORITY ORDER
==============================================================================

PRIORITY 1 - Immediate, no retraining, high confidence:
  FIX-28: Fix temporal leakage in run_phase_3() - ONE LINE FIX
  FIX-04: Add model-output columns to leakage_audit() BANNED_COLS
  FIX-13: Add Pydantic range validators to backend/schemas.py
  FIX-15: Add review_rate + FP/1000 to utils.py evaluate()

PRIORITY 2 - Architecture changes, next sprint:
  FIX-01 + FIX-02 + FIX-03 + FIX-05: Trust Engine integration
  FIX-06: Calibration (sklearn CalibratedClassifierCV)
  FIX-11: Cold-start fields in API response

PRIORITY 3 - Research evaluation experiments:
  FIX-20: Multi-seed evaluation (seeds 42-46)
  FIX-19: Bootstrap confidence intervals
  FIX-21: Prevalence sensitivity (1%, 3%, 7%, 15%)
  FIX-22: Distribution/regime shift experiment

PRIORITY 4 - Optional, high ROI:
  FIX-09: assert_strict_temporal() test
  FIX-16 + FIX-17: Group metrics for segment robustness
  FIX-18: Domain-specific evasion tests
  FIX-23: Temporal drift per period
  FIX-25: Operational metrics (fraud dollars caught)

NOT IMPLEMENTING (wrong assumption or duplicate):
  FIX-27: Duplicate of FIX-13
  FIX-31: Wrong import assumption in fix pack test
  FIX-32: TypedMultiGraph is not a HeteroGNN fix
  FIX-33: Custom PlattCalibrator - use sklearn instead

==============================================================================
SUMMARY TABLE (33 distinct issues analyzed)
==============================================================================
FIX-01 Trust Engine + Decision          REQUIRED           HIGH      Implement
FIX-02 Weighted aggregation             REQUIRED           MEDIUM    With Fix-01
FIX-03 model_disagreement output        REQUIRED           LOW-MED   With Fix-01
FIX-04 Feedback loop guard              PARTIALLY_IMPL     MEDIUM    Extend leakage_audit
FIX-05 Reason codes in response         REQUIRED           MEDIUM    With Fix-01
FIX-06 Probability calibration          REQUIRED           MED-HIGH  Use sklearn not fix pack
FIX-07 Cost threshold + calibration     PARTIALLY_IMPL     MEDIUM    Extend after Fix-06
FIX-08 Same-timestamp exclusion         ALREADY_IMPL       -         No action
FIX-09 assert_strict_temporal           OPTIONAL           LOW       Add to test_leakage.py
FIX-10 TypedMultiGraph                  PARTIALLY_IMPL     MEDIUM    Ring analysis only
FIX-11 Cold-start detection             PARTIALLY_IMPL     MEDIUM    Add to API response
FIX-12 Online graph context             PARTIALLY_IMPL     LOW       DEFER
FIX-13 Input validation ranges          REQUIRED           MEDIUM    Pydantic validators
FIX-14 Missing value flags              OPTIONAL           LOW       DEFER (needs retraining)
FIX-15 review_rate + FP/1000            REQUIRED           MEDIUM    Extend evaluate()
FIX-16 group_metrics                    OPTIONAL           LOW-MED   Add to utils.py
FIX-17 group_error_rates                OPTIONAL           LOW       Add to experiments
FIX-18 Bounded evasion testing          OPTIONAL           MEDIUM    Domain-specific only
FIX-19 Bootstrap CIs                    REQUIRED           MEDIUM    Add to evaluation
FIX-20 Multi-seed evaluation            REQUIRED           HIGH      Parameterize train fn
FIX-21 Prevalence sensitivity           REQUIRED           MED-HIGH  Resample test set
FIX-22 Distribution shift               REQUIRED           HIGH      Changed regime tests
FIX-23 Temporal drift                   OPTIONAL           MEDIUM    Per-period buckets
FIX-24 Cold-start evaluation            OPTIONAL           MEDIUM    Segment evaluation
FIX-25 Operational metrics              OPTIONAL           MEDIUM    Extend evaluate()
FIX-26 Model versioning metadata        OPTIONAL           LOW       Train scripts
FIX-27 API validation (duplicate)       ALREADY_COVERED    -         Use Fix-13
FIX-28 TEMPORAL LEAKAGE run_phase_3     REQUIRED           HIGH      Fix NOW
FIX-29 compileall in CI                 OPTIONAL           LOW       Add to pytest.ini
FIX-30 Category median (listing)        ALREADY_IMPL       -         No action
FIX-31 trust_api_example wrong import   WRONG_ASSUMPTION   -         Do not merge
FIX-32 TypedMultiGraph != HeteroGNN     WRONG_ASSUMPTION   -         Scope correctly
FIX-33 Custom PlattCalibrator           WRONG_ASSUMPTION   -         Use sklearn
