// ============================================================================
// TRUSTSHIELD AI — TypeScript API Interfaces (100% mapped to backend/schemas.py)
// ============================================================================

export type OperationalDecision = "ALLOW" | "REVIEW" | "HOLD" | "BLOCK";
export type RiskLabel = "low" | "medium" | "high";
export type ServiceState = "ready" | "degraded" | "not_ready";

// 1. Health & Readiness Probes
export interface HealthResponse {
  status: string;
  models_loaded: boolean;
  phase5_loaded: boolean;
  clip_loaded: boolean;
  shap_loaded?: boolean;
  rings_loaded: boolean;
  n_rings: number;
}

export interface ReadyResponse {
  status: ServiceState;
  models_ready: boolean;
  phase3_ready: boolean;
  phase5_ready: boolean;
  clip_ready: boolean;
  shap_ready?: boolean;
  rings_ready: boolean;
  redis_ready?: boolean;
  neo4j_ready?: boolean;
  components?: Record<string, string>;
  details: Record<string, any>;
}

// 2. Transaction Scoring
export interface TransactionScoreRequest {
  order_id?: string;
  buyer_id?: string;
  seller_id?: string;
  listing_id?: string;

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
}

export interface TransactionScoreResponse {
  order_id?: string;
  overall_fraud_probability: number;
  raw_fraud_probability?: number;
  calibrated_fraud_probability?: number;
  canonical_trust_engine_risk?: number;
  risk_label: RiskLabel;
  model_used: string;
  model_version: string;
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
  top_risk_drivers?: Array<Record<string, any>>;
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
}

export interface ListingScoreResponse {
  listing_id?: string;
  fake_listing_probability: number;
  risk_label: RiskLabel;
  model_used: string;
  clip_scored: boolean;
  multimodal_similarity_score: number;
  investigator_narrative: Record<string, any>;
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
  transaction_data?: Record<string, any>;
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
  top_risk_drivers?: Array<Record<string, any>>;
  shap_narrative?: string;
}

export interface DossierGraphFindings {
  graph_source: "neo4j" | "disk_artifact" | "unavailable" | string;
  cluster_id?: string;
  cluster_size: number;
  topology_summary: string;
  suspicious_relationships: Array<Record<string, any>>;
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
  risk_level: RiskLabel;
  decision: OperationalDecision;
  trust_score: number;
  reason_codes: string[];
  source: "simulation_stream" | "live_stream" | string;
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
  benchmarks: {
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
  transaction_data?: Record<string, any>;
  [key: string]: any;
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
