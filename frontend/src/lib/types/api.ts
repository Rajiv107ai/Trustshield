// ============================================================================
// TRUSTSHIELD AI — TypeScript API Interfaces (Authoritative frontend API type contracts — synchronized with backend/schemas.py)
// ============================================================================

export type OperationalDecision = "ALLOW" | "REVIEW" | "HOLD" | "BLOCK";
export type RiskLabel = "low" | "medium" | "high";
export type ServiceState = "ready" | "degraded" | "not_ready";

// 0. Operator Actions
// These contracts mirror backend/contracts/actions.py.  An action is accepted
// by the API before a worker attempts any provider-side effect, so a pending
// receipt must never be presented as provider success by the UI.
export type ActionStatus = "pending" | "succeeded" | "failed";

export type ActionType =
  | "payment_hold"
  | "payment_release"
  | "payment_refund"
  | "marketplace_hold_order"
  | "marketplace_suspend_listing"
  | "marketplace_restore_listing"
  | "marketplace_suspend"
  | "marketplace_reinstate"
  | "identity_lock_account"
  | "identity_require_verification"
  | "identity_unlock_account"
  | "identity_lock"
  | "graph_flag_entity"
  | "graph_recalculate_risk"
  | "graph_add_case_edge"
  | "graph_flag"
  | "case_create"
  | "case_update"
  | "case_close"
  | "case_add_note";

export interface ActionTarget {
  target_type: string;
  target_id: string;
}

export interface ActionRequest {
  schema_version?: "actions.v1";
  action: ActionType;
  case_id: string;
  reason: string;
  idempotency_key: string;
  target: ActionTarget;
  target_confirmation: true;
  policy_version: string;
  requested_by?: string;
}

export interface ProviderReceipt {
  receipt_id: string;
  provider: string;
  action: ActionType;
  idempotency_key: string;
  provider_reference: string;
  target: ActionTarget;
  accepted: boolean;
  details?: Record<string, string | number | boolean | null>;
}

export interface ActionReceipt {
  schema_version: "actions.v1";
  command_id: string;
  action: ActionType;
  case_id: string;
  reason: string;
  idempotency_key: string;
  target: ActionTarget;
  target_confirmation: true;
  policy_version: string;
  status: ActionStatus;
  created_at: string;
  updated_at: string;
  attempts: number;
  provider_receipt?: ProviderReceipt | null;
  error?: string | null;
  requested_by?: string | null;
}

// 1. Health & Readiness Probes
export interface HealthResponse {
  status: string;
  models_loaded: boolean;
  phase5_loaded?: boolean;
  clip_loaded: boolean;
  shap_loaded?: boolean;
  rings_loaded: boolean;
  n_rings: number;
}

export interface ReadyResponse {
  status: ServiceState;
  models_ready: boolean;
  phase3_ready: boolean;
  phase5_ready?: boolean;
  clip_ready: boolean;
  shap_ready?: boolean;
  rings_ready: boolean;
  redis_ready?: boolean;
  neo4j_ready?: boolean;
  pit_history_ready?: boolean;
  graph_snapshot_as_of?: string;
  components?: Record<string, string>;
  details: Record<string, unknown>;
}

export interface VersionResponse {
  service: string;
  version: string;
  app_version: string;
  release_id?: string;
  manifest_present: boolean;
  manifest_valid: boolean;
  manifest_required: boolean;
  manifest_schema_version?: number;
  git_commit?: string;
  working_tree_clean?: boolean;
  validation_errors: string[];
  manifest?: Record<string, unknown>;
}

// 2. Transaction Scoring
export interface TransactionScoreRequest {
  order_id?: string;
  buyer_id?: string;
  seller_id?: string;
  listing_id?: string;
  order_date?: string;

  amount?: number;
  order_amount?: number;
  base_price?: number;
  category_median_price?: number;

  price_vs_base_price_ratio?: number;
  price_vs_category_median_ratio?: number;
  seller_age_days?: number;
  seller_total_listings_before?: number;
  buyer_age_days?: number;
  buyer_orders_before?: number;
  buyer_returns_before?: number;
  buyer_return_rate_before?: number;
  device_shared_buyer_count?: number;

  share_degree?: number;
  share_component_size?: number;
  buyer_seller_degree?: number;
  buyer_pagerank?: number;
  seller_buyer_degree?: number;
  seller_pagerank?: number;
  seller_buyer_concentration_hhi?: number;
  buyer_seller_edge_weight_before?: number;

  multimodal_similarity_score?: number;
  model_version?: string;
}

export interface TransactionScoreResponse {
  order_id?: string;
  overall_fraud_probability: number;
  raw_fraud_probability?: number;
  calibrated_fraud_probability?: number;
  policy_risk_score?: number;
  canonical_trust_engine_risk?: number;
  conformal_prediction_set?: number[];
  conformal_is_calibrated?: boolean;
  conformal_coverage_guarantee?: number;
  conformal_q_hat?: number;
  risk_label: RiskLabel;
  model_used: string;
  model_version: string;
  graph_features_source?: string;
  client_mismatch?: boolean;
  scoring_mode?: string;
  is_simulation?: boolean;
  evidence_provenance_warning?: string;
  note: string;

  decision: OperationalDecision;
  trust_score: number;             // [0, 100]
  confidence: number;              // [0, 1] based on 1 - H2(p)
  predictive_uncertainty?: number; // Shannon entropy [0, 1]
  cold_start: boolean;
  model_disagreement: number;      // Component spread
  reason_codes: string[];
  evidence_availability?: Record<string, boolean>;
  infrastructure_sources?: Record<string, string>;
  shap_attributions?: Record<string, number>;
  top_risk_drivers?: Array<Record<string, unknown>>;
}

// 3. Fraud Rings
export interface FraudRingItem {
  ring_id: string;
  size: number;
  n_orders: number;
  avg_risk_score: number;
  max_risk_score: number;
  n_high_risk_orders: number;
  risk_label: RiskLabel;
  members: string[];
}

export interface FraudRingsResponse {
  total_rings: number;
  high_risk_rings: number;
  rings: FraudRingItem[];
}

// 4. Multimodal Listing Scoring
export interface ListingScoreRequest {
  listing_id?: string;
  seller_id?: string;
  price?: number;
  base_price?: number;
  category_median_price?: number;
  seller_age_days_at_listing?: number;
  seller_listings_before?: number;

  product_id?: string;
  displayed_product_id?: string;
  image_ref?: string;
  multimodal_similarity_score?: number;
  synthetic_mismatch_score?: number;
}

export interface ListingScoreResponse {
  listing_id?: string;
  fake_listing_probability: number;
  risk_label: RiskLabel;
  model_used: string;
  clip_scored: boolean;
  multimodal_similarity_score: number;
  investigator_narrative: Record<string, unknown>;
}

// 5. Return Abuse Scoring
export interface ReturnScoreRequest {
  return_id?: string;
  order_id?: string;
  buyer_id?: string;
  seller_id?: string;
  days_to_return?: number;
  buyer_age_days_at_return?: number;
  seller_age_days_at_return?: number;
  order_amount?: number;
  buyer_prior_returns?: number;
  buyer_orders_before_return?: number;
  buyer_return_rate_before?: number;
  seller_prior_returns?: number;
  seller_orders_before_return?: number;
  seller_return_rate_before?: number;
  reason?: "changed_mind" | "defective" | "size_issue" | "wrong_item_received" | string;
}

export interface ReturnScoreResponse {
  return_id?: string;
  return_fraud_probability: number;
  risk_label: RiskLabel;
  decision: OperationalDecision;
  model_used: string;
  reason_codes: string[];
}

// 6. Forensic Investigation Dossier API
export interface DossierRequest {
  entity_type: "transaction" | "order" | "buyer" | "seller" | "ring" | string;
  entity_id: string;
  transaction_data?: Record<string, unknown>;
  include_graph_evidence?: boolean;
}

export interface DossierObservedEvidence {
  transaction_amount?: number;
  buyer_historical_return_rate?: number;
  account_age_days?: number;
  prior_order_count?: number;
  hardware_collision_detected: boolean;
  shared_device_count: number;
  verified_facts: string[];
}

export interface DossierModelInference {
  calibrated_risk_score: number;
  operational_decision: OperationalDecision;
  confidence_level: number;
  detector_disagreement: number;
  conformal_prediction_set: string;
  triggered_reason_codes: string[];
  shap_attributions?: Record<string, number>;
  top_risk_drivers?: Array<Record<string, unknown>>;
  shap_narrative?: string;
}

export interface DossierGraphFindings {
  graph_source: "neo4j" | "disk_artifact" | "unavailable" | string;
  cluster_id?: string;
  cluster_size: number;
  topology_summary: string;
  suspicious_relationships: Array<Record<string, unknown>>;
}

export interface DossierRecommendation {
  action: string;
  protocol_level: string;
  required_evidence_to_clear: string;
}

export interface DossierResponse {
  case_id: string;
  entity_type: string;
  entity_id: string;
  generated_at: string;
  risk_score: number;
  risk_level: string;
  decision: OperationalDecision;
  executive_summary: string;
  observed_evidence: DossierObservedEvidence;
  model_inference: DossierModelInference;
  graph_findings: DossierGraphFindings;
  timeline: Array<{ time: string; event: string; type: string }>;
  involved_entities: Record<string, string>;
  retrieved_policy_guidelines: string[];
  recommendation: DossierRecommendation;
  limitations: string[];
  provenance: Record<string, string>;
  grounding_verification_passed: boolean;
  evidence_provenance_warning?: string;
}

// 7. Real-time Stream Event
export interface StreamTransactionEvent {
  event_id: string;
  timestamp: string;
  order_id: string;
  buyer_id: string;
  seller_id: string;
  amount: number;
  risk_score: number;
  calibrated_risk: number;
  policy_risk_score?: number;
  risk_level: RiskLabel;
  decision: OperationalDecision;
  trust_score: number;
  reason_codes: string[];
  source: "simulation_stream" | "live_stream" | string;
  is_simulation?: boolean;
}

// 8. Empirical Benchmark Telemetry
export interface BenchmarkItem {
  unit: string;
  samples: number;
  p50: number;
  p95: number;
  p99: number;
  note?: string;
  cache_hits?: number;
}

export interface SystemBenchmarkResponse {
  status: string;
  benchmarks?: {
    hybrid_inference?: BenchmarkItem;
    trust_engine_scoring?: BenchmarkItem;
    redis_get_embedding?: BenchmarkItem;
    neo4j_neighborhood?: BenchmarkItem;
  };
  measured_at: string;
}

// 9. TreeSHAP Explainability API
export interface FeatureShapDriver {
  feature_name: string;
  friendly_name: string;
  feature_value: number;
  shap_value: number;
  abs_impact: number;
}

export interface TransactionExplainRequest {
  order_id?: string;
  transaction_data?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface TransactionExplainResponse {
  order_id: string;
  base_value: number;
  overall_fraud_probability: number;
  decision: OperationalDecision;
  model_used: string;
  top_positive_drivers: FeatureShapDriver[];
  top_negative_dampeners: FeatureShapDriver[];
  all_attributions: Record<string, number>;
  investigator_narrative: string;
}

// 10. Operational Model Monitoring & Governance
export interface MonitoringAlert {
  schema_version: "monitoring.v1";
  metric_name: string;
  metric_type: string;
  status: "OK" | "WARNING" | "CRITICAL";
  observed_value?: number;
  warning_threshold?: number;
  critical_threshold?: number;
  direction: "higher" | "lower";
  recommended_action: "none" | "investigation" | "shadow_evaluation" | "promotion_review";
  sample_count: number;
  reason?: string;
}

export interface MonitoringTopFeature {
  feature: string;
  psi: number;
  ks_statistic: number;
  status: "OK" | "WARNING" | "CRITICAL";
}

export interface MonitoringOverviewResponse {
  status: "OK" | "WARNING" | "CRITICAL";
  schema_version: "monitoring.v1";
  evaluated_at: string;
  drift: {
    status: "OK" | "WARNING" | "CRITICAL";
    recommended_action: string;
    top_drifted_features: MonitoringTopFeature[];
    decision_distribution: Record<string, number>;
    score_drift?: Record<string, unknown>;
  };
  calibration: {
    status: "OK" | "WARNING" | "CRITICAL";
    ece: number;
    brier_score: number;
    precision: number;
    recall: number;
    review_budget_capture?: Record<string, number>;
    matured_label_count: number;
  };
  active_alerts_count: number;
  active_alerts: MonitoringAlert[];
}

export interface FeatureServiceStatusResponse {
  status: string;
  ready: boolean;
  engine: string;
  redis_connected: boolean;
  orders_indexed: number;
  buyers_indexed: number;
  sellers_indexed: number;
  catalog_products_indexed: number;
  catalog_categories: number;
}

export interface BuyerFeatureProfileResponse {
  buyer_id: string;
  as_of: string;
  total_historical_orders: number;
  total_historical_returns: number;
  historical_return_rate: number;
  buyer_age_days: number;
  velocities: {
    window_1h_orders: number;
    window_1h_spend: number;
    window_24h_orders: number;
    window_24h_spend: number;
    window_7d_orders: number;
    window_7d_spend: number;
  };
  engine: string;
}

export interface DeviceCollisionProfileResponse {
  device_id: string;
  as_of: string;
  device_shared_buyer_count_7d: number;
  is_shared_device: boolean;
  engine: string;
}

export interface FeaturePreviewResponse {
  order_id?: string;
  feature_row: Record<string, number>;
  metadata: {
    as_of: string;
    is_historical_replay: boolean;
    engine: string;
    velocities: {
      buyer_orders_1h: number;
      buyer_orders_24h: number;
      buyer_spend_24h: number;
      device_shared_buyers_7d: number;
    };
    provenance: {
      source: string;
      zero_future_leakage_verified: boolean;
    };
  };
}

export interface CopilotToolExecution {
  agent: string;
  tool: string;
  args: Record<string, unknown>;
  summary: string;
  latency_ms: number;
}

export interface CryptographicDisputeBrief {
  brief_id: string;
  case_id: string;
  target_entity: string;
  generated_at: string;
  fraud_classification: string;
  compelling_evidence_points: string[];
  chargeback_defense_viability: "STRONG" | "MODERATE" | "WEAK";
  recommended_merchant_action: string;
  sha256_evidence_digest: string;
  hmac_signature: string;
  markdown_content: string;
}

export interface CopilotInvestigationResponse {
  case_id: string;
  entity_type: string;
  entity_id: string;
  generated_at: string;
  status: "COMPLETED" | "FLAGGED_INCONCLUSIVE" | string;
  primary_fraud_hypothesis: string;
  confidence_score: number;
  copilot_narrative: string;
  counterfactual_defense: string;
  graph_topology: Record<string, unknown>;
  velocity_profile: Record<string, unknown>;
  tool_executions: CopilotToolExecution[];
  recommended_action: {
    action_type: string;
    target_type: string;
    target_id: string;
    reason: string;
    recommended_policy?: string;
  };
  dispute_brief?: CryptographicDisputeBrief;
}

export interface VerifyBriefResponse {
  brief_id: string;
  is_valid: boolean;
  digest_matches: boolean;
  signature_matches: boolean;
  verification_timestamp: string;
  notes: string;
}

// 7. Champion-Challenger Shadow Scoring & Conformal A/B Deployment
export interface ModelEvaluationItem {
  model_id: string;
  model_name: string;
  algorithm: string;
  raw_score: number;
  calibrated_score: number;
  decision: OperationalDecision | string;
  trust_score: number;
  conformal_prediction_set: number[];
  is_conformal_singleton: boolean;
  latency_ms: number;
}

export interface ChampionChallengerComparisonResponse {
  order_id: string;
  timestamp: string;
  champion: ModelEvaluationItem;
  challenger: ModelEvaluationItem;
  decision_matches: boolean;
  conformal_matches: boolean;
  risk_delta: number;
  conformal_divergence: string;
  routed_decision: OperationalDecision | string;
  routed_model: "CHAMPION" | "CHALLENGER" | string;
  deployment_mode: "SHADOW" | "AB_SPLIT" | "CANARY" | string;
  traffic_split_pct: number;
  recommendation: string;
}

export interface ChampionChallengerStatusResponse {
  deployment_mode: "SHADOW" | "AB_SPLIT" | "CANARY" | string;
  traffic_split_pct: number;
  champion_spec: {
    id: string;
    name: string;
    status: string;
    calibrator: string;
    conformal: string;
  };
  challenger_spec: {
    id: string;
    name: string;
    status: string;
    calibrator: string;
    conformal: string;
    loaded: boolean;
  };
  circuit_breaker_tripped: boolean;
  circuit_breaker_reason?: string | null;
  total_evaluations: number;
  decision_concordance_rate: number;
  conformal_concordance_rate: number;
  champion_certainty_rate: number;
  challenger_certainty_rate: number;
  avg_risk_champion: number;
  avg_risk_challenger: number;
  mean_absolute_risk_diff: number;
  p95_latency_champion_ms: number;
  p95_latency_challenger_ms: number;
  recent_disagreements_count: number;
}


