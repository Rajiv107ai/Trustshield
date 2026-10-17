"""
TrustShield AI — Pydantic schemas for the FastAPI layer.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# /transaction/score  (POST)
# ---------------------------------------------------------------------------

class TransactionScoreRequest(BaseModel):
    order_id: Optional[str] = None
    buyer_id: Optional[str] = None
    seller_id: Optional[str] = None
    listing_id: Optional[str] = None
    order_date: Optional[str] = None

    # Tabular features
    price_vs_base_price_ratio: float = Field(default=0.0, ge=0.0)
    price_vs_category_median_ratio: float = Field(default=0.0, ge=0.0)
    seller_age_days: float = Field(default=0.0, ge=0.0)
    seller_total_listings_before: float = Field(default=0.0, ge=0.0)
    buyer_age_days: float = Field(default=0.0, ge=0.0)
    buyer_orders_before: int = Field(default=0, ge=0)
    buyer_returns_before: int = Field(default=0, ge=0)
    buyer_return_rate_before: float = Field(default=0.0, ge=0.0, le=1.0)
    device_shared_buyer_count: float = Field(default=1.0, ge=0.0)
    order_amount: float = Field(default=0.0, ge=0.0)

    # Graph features
    share_degree: float = 0.0
    share_component_size: float = 1.0
    buyer_seller_degree: float = 0.0
    buyer_pagerank: float = 0.0
    seller_buyer_degree: float = 0.0
    seller_pagerank: float = 0.0
    seller_buyer_concentration_hhi: float = 0.0
    buyer_seller_edge_weight_before: float = 0.0

    # Raw inputs for server-side derivation
    amount: Optional[float] = None
    base_price: Optional[float] = None
    category_median_price: Optional[float] = None
    multimodal_similarity_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    model_version: Optional[str] = None


class TransactionScoreResponse(BaseModel):
    order_id: Optional[str] = None
    overall_fraud_probability: float
    raw_fraud_probability: Optional[float] = None
    calibrated_fraud_probability: Optional[float] = None
    policy_risk_score: Optional[float] = None
    canonical_trust_engine_risk: Optional[float] = None
    conformal_prediction_set: Optional[List[int]] = None
    conformal_is_calibrated: bool = False
    conformal_coverage_guarantee: Optional[float] = 0.95
    conformal_q_hat: Optional[float] = None
    risk_label: str
    model_used: str
    model_version: str = "phase3"
    graph_features_source: str = "server"
    client_mismatch: bool = False
    note: str = ""
    decision: str
    trust_score: float = Field(ge=0.0, le=100.0)
    confidence: float = Field(ge=0.0, le=1.0)
    predictive_uncertainty: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cold_start: bool
    model_disagreement: float = Field(default=0.0, ge=0.0, le=1.0)
    reason_codes: List[str] = Field(default_factory=list)
    evidence_availability: Optional[Dict[str, bool]] = None
    infrastructure_sources: Optional[Dict[str, str]] = None
    shap_attributions: Optional[Dict[str, float]] = None
    top_risk_drivers: Optional[List[Dict[str, Any]]] = None
    scoring_mode: str = "production_model"
    is_simulation: bool = False


# ---------------------------------------------------------------------------
# /transaction/explain  (POST)
# ---------------------------------------------------------------------------

class FeatureShapDriver(BaseModel):
    feature_name: str
    friendly_name: str
    feature_value: float
    shap_value: float
    abs_impact: float


class TransactionExplainRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    order_id: Optional[str] = "ORDER_EXPLAIN_PROBE"
    transaction_data: Optional[Dict[str, Any]] = None


class TransactionExplainResponse(BaseModel):
    order_id: str
    base_value: float
    overall_fraud_probability: float
    decision: str
    model_used: str
    top_positive_drivers: List[FeatureShapDriver] = Field(default_factory=list)
    top_negative_dampeners: List[FeatureShapDriver] = Field(default_factory=list)
    all_attributions: Dict[str, float] = Field(default_factory=dict)
    investigator_narrative: str


# ---------------------------------------------------------------------------
# /fraud-rings  (GET)
# ---------------------------------------------------------------------------

class FraudRingItem(BaseModel):
    ring_id: str
    size: int
    n_orders: int
    avg_risk_score: float
    max_risk_score: float
    n_high_risk_orders: int
    risk_label: str
    members: List[str] = []


class FraudRingsResponse(BaseModel):
    total_rings: int
    high_risk_rings: int
    rings: List[FraudRingItem]


# ---------------------------------------------------------------------------
# /health  (GET)
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    phase5_loaded: bool = False
    clip_loaded: bool = False
    shap_loaded: bool = False
    rings_loaded: bool
    n_rings: int


# ---------------------------------------------------------------------------
# /listing/analyze  (POST)
# ---------------------------------------------------------------------------

class ListingScoreRequest(BaseModel):
    listing_id: Optional[str] = None
    seller_id: Optional[str] = None
    price: float = 100.0
    base_price: float = 100.0
    category_median_price: float = 100.0
    seller_age_days_at_listing: float = 30.0
    seller_listings_before: int = 5
    product_id: Optional[str] = None
    displayed_product_id: Optional[str] = None
    image_ref: Optional[str] = None
    multimodal_similarity_score: float = 0.85
    synthetic_mismatch_score: Optional[float] = None


class ListingScoreResponse(BaseModel):
    listing_id: Optional[str]
    fake_listing_probability: float
    risk_label: str
    model_used: str = "Fake Listing Detector (Phase 4 Multimodal XGBoost)"
    clip_scored: bool = False
    multimodal_similarity_score: float = 0.85
    investigator_narrative: Dict[str, Any]


# ---------------------------------------------------------------------------
# /return/analyze  (POST)
# ---------------------------------------------------------------------------

class ReturnScoreRequest(BaseModel):
    return_id: Optional[str] = None
    order_id: Optional[str] = None
    buyer_id: Optional[str] = None
    seller_id: Optional[str] = None
    days_to_return: float = Field(default=7.0, ge=0.0)
    buyer_age_days_at_return: float = Field(default=90.0, ge=0.0)
    seller_age_days_at_return: float = Field(default=180.0, ge=0.0)
    order_amount: float = Field(default=50.0, ge=0.0)
    buyer_prior_returns: int = Field(default=0, ge=0)
    buyer_orders_before_return: int = Field(default=5, ge=0)
    buyer_return_rate_before: float = Field(default=0.0, ge=0.0, le=1.0)
    seller_prior_returns: int = Field(default=2, ge=0)
    seller_orders_before_return: int = Field(default=20, ge=0)
    seller_return_rate_before: float = Field(default=0.1, ge=0.0, le=1.0)
    reason: Optional[str] = "defective"


class ReturnScoreResponse(BaseModel):
    return_id: Optional[str] = None
    return_fraud_probability: float
    risk_label: str
    decision: str
    model_used: str = "Return Fraud Detector (Phase 2 Specialized Model)"
    reason_codes: List[str] = []


# ---------------------------------------------------------------------------
# /ready (GET)
# ---------------------------------------------------------------------------

class ServiceState(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"
    NOT_READY = "not_ready"


class ReadyResponse(BaseModel):
    status: ServiceState
    models_ready: bool
    phase3_ready: bool
    phase5_ready: bool
    clip_ready: bool
    shap_ready: bool = False
    rings_ready: bool
    redis_ready: bool = False
    neo4j_ready: bool = False
    components: Dict[str, str] = Field(default_factory=dict)
    details: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# /investigation/generate-dossier (POST)
# ---------------------------------------------------------------------------

class DossierRequest(BaseModel):
    entity_type: str = "transaction"
    entity_id: str
    transaction_data: Optional[Dict[str, Any]] = None
    include_graph_evidence: bool = True


class DossierObservedEvidence(BaseModel):
    transaction_amount: Optional[float] = None
    buyer_historical_return_rate: Optional[float] = None
    account_age_days: Optional[float] = None
    prior_order_count: Optional[int] = None
    hardware_collision_detected: bool = False
    shared_device_count: int = 1
    verified_facts: List[str] = Field(default_factory=list)


class DossierModelInference(BaseModel):
    calibrated_risk_score: float
    operational_decision: str
    confidence_level: float
    detector_disagreement: float
    conformal_prediction_set: str
    triggered_reason_codes: List[str] = Field(default_factory=list)
    shap_attributions: Optional[Dict[str, float]] = None
    top_risk_drivers: Optional[List[Dict[str, Any]]] = None
    shap_narrative: Optional[str] = None


class DossierGraphFindings(BaseModel):
    graph_source: str
    cluster_id: Optional[str] = None
    cluster_size: int = 1
    topology_summary: str
    suspicious_relationships: List[Dict[str, Any]] = Field(default_factory=list)


class DossierRecommendation(BaseModel):
    action: str
    protocol_level: str
    required_evidence_to_clear: str


class DossierResponse(BaseModel):
    case_id: str
    entity_type: str
    entity_id: str
    generated_at: str
    risk_score: float
    risk_level: str
    decision: str
    executive_summary: str
    observed_evidence: DossierObservedEvidence
    model_inference: DossierModelInference
    graph_findings: DossierGraphFindings
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    involved_entities: Dict[str, str] = Field(default_factory=dict)
    retrieved_policy_guidelines: List[str] = Field(default_factory=list)
    recommendation: DossierRecommendation
    limitations: List[str] = Field(default_factory=list)
    provenance: Dict[str, str] = Field(default_factory=dict)
    grounding_verification_passed: bool = True


# ---------------------------------------------------------------------------
# /stream/transactions (GET SSE)
# ---------------------------------------------------------------------------

class StreamTransactionEvent(BaseModel):
    event_id: str
    timestamp: str
    order_id: str
    buyer_id: str
    seller_id: str
    amount: float
    risk_score: float
    calibrated_risk: float
    policy_risk_score: Optional[float] = None
    risk_level: str
    decision: str
    trust_score: float
    reason_codes: List[str] = Field(default_factory=list)
    source: str = "simulation_stream"
    is_simulation: bool = True
