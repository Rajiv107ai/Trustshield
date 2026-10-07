# 02 — API SCHEMAS & TYPESCRIPT DEFINITIONS

These TypeScript interfaces match 100% to the actual Pydantic v2 schemas in `backend/schemas.py`:

```typescript
// ============================================================================
// 1. Health & Readiness Probes (GET /health, GET /ready)
// ============================================================================

export interface HealthResponse {
  status: string;
  models_loaded: boolean;
  phase5_loaded: boolean;
  clip_loaded: boolean;
  rings_loaded: boolean;
  n_rings: number;
}

export type ServiceState = "ready" | "degraded" | "not_ready";

export interface ReadyResponse {
  status: ServiceState;
  models_ready: boolean;
  phase3_ready: boolean;
  phase5_ready: boolean;
  clip_ready: boolean;
  rings_ready: boolean;
  details: Record<string, any>;
}

// ============================================================================
// 2. Transaction Scoring (POST /transaction/score)
// ============================================================================

export type OperationalDecision = "ALLOW" | "REVIEW" | "HOLD" | "BLOCK";
export type RiskLabel = "low" | "medium" | "high";

export interface TransactionScoreRequest {
  // Identifiers (traceability)
  order_id?: string;
  buyer_id?: string;
  seller_id?: string;
  listing_id?: string;

  // Convenience raw inputs (backend derives ratios if supplied)
  amount?: number;
  order_amount?: number;
  base_price?: number;
  category_median_price?: number;

  // Pre-computed tabular features (exact model feature names)
  price_vs_base_price_ratio?: number;
  price_vs_category_median_ratio?: number;
  seller_age_days?: number;
  seller_total_listings_before?: number;
  buyer_age_days?: number;
  buyer_orders_before?: number;
  buyer_returns_before?: number;
  buyer_return_rate_before?: number;
  device_shared_buyer_count?: number;

  // Graph topology features
  share_degree?: number;
  share_component_size?: number;
  buyer_seller_degree?: number;
  buyer_pagerank?: number;
  seller_buyer_degree?: number;
  seller_pagerank?: number;
  seller_buyer_concentration_hhi?: number;
  buyer_seller_edge_weight_before?: number;

  // Multimodal score hint
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

  // Canonical Trust Engine Outputs (Real values, no synthetic placeholders)
  decision: OperationalDecision;
  trust_score: number;             // [0, 100] (100 = full trust)
  confidence: number;              // [0, 1] based on Shannon binary entropy 1 - H2(p)
  predictive_uncertainty?: number; // Shannon entropy [0, 1]
  cold_start: boolean;
  model_disagreement: number;      // Spread between component model probabilities
  reason_codes: string[];
  evidence_availability?: Record<string, boolean>;
}

// ============================================================================
// 3. Candidate Fraud Rings (GET /fraud-rings)
// ============================================================================

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

// ============================================================================
// 4. Multimodal Listing Intelligence (POST /listing/analyze)
// ============================================================================

export interface ListingScoreRequest {
  listing_id?: string;
  seller_id?: string;
  price?: number;
  base_price?: number;
  category_median_price?: number;
  seller_age_days_at_listing?: number;
  seller_listings_before?: number;
  
  // Real CLIP routing fields
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

// ============================================================================
// 5. Specialized Return Abuse Scoring (POST /return/analyze)
// ============================================================================

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
```
