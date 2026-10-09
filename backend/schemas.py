"""
TrustShield AI — Pydantic schemas for the FastAPI layer.

Kept minimal: only the fields the three MVP endpoints actually need.

TransactionScoreRequest uses the exact feature column names that the
trained combined graph model expects (tabular + graph features from
baseline_model.py and graph_features.py). This avoids the silent
all-zeros prediction bug that arises when schema field names don't
match model feature names.

Convenience raw-input fields (amount, base_price, category_median_price)
are provided so callers can supply primitive inputs and let main.py
derive the ratio features — useful when the caller doesn't have access
to pre-computed statistics.  If the derived fields ARE supplied directly,
the raw-input fields are ignored.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# /transaction/score  (POST)
# ---------------------------------------------------------------------------

class TransactionScoreRequest(BaseModel):
    """
    Features needed to score an incoming order for fraud.

    Primary fields use the EXACT column names the trained model expects.
    Callers that cannot compute ratios themselves may supply the raw
    amount / base_price / category_median_price fields instead and let
    the server derive price_vs_base_price_ratio and
    price_vs_category_median_ratio automatically.

    All optional with default=0 so the endpoint degrades gracefully on
    partial data rather than rejecting the request.
    """
    # --- Identifiers (not used by the model, passed through for tracing) ---
    order_id: Optional[str] = None
    buyer_id: Optional[str] = None
    seller_id: Optional[str] = None
    listing_id: Optional[str] = None

    # --- Tabular features (exact model feature column names) ---
    price_vs_base_price_ratio: float = Field(
        default=0.0,
        ge=0.0,
        description="order amount / catalog base price (computed server-side "
                    "from amount + base_price if not supplied directly)",
    )
    price_vs_category_median_ratio: float = Field(
        default=0.0,
        ge=0.0,
        description="order amount / category median price (computed server-side "
                    "from amount + category_median_price if not supplied directly)",
    )
    seller_age_days: float = Field(default=0.0, ge=0.0, description="Days since seller signup at order time")
    seller_total_listings_before: float = Field(default=0.0, ge=0.0, description="Seller's listing count before this order")
    buyer_age_days: float = Field(default=0.0, ge=0.0, description="Days since buyer signup at order time")
    buyer_orders_before: int = Field(default=0, ge=0, description="Buyer's prior order count")
    buyer_returns_before: int = Field(default=0, ge=0, description="Buyer's prior return count")
    buyer_return_rate_before: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="buyer_returns_before / max(buyer_orders_before, 1) "
                    "(computed server-side if not supplied directly)",
    )
    device_shared_buyer_count: float = Field(
        default=1.0,
        ge=0.0,
        description="Number of distinct buyers sharing this device (from training-set graph)",
    )
    order_amount: float = Field(default=0.0, ge=0.0, description="Order value")

    # --- Graph features (optional — filled with 0 if not provided) ---
    share_degree: float = Field(default=0.0, description="Buyer's degree in device+address sharing graph")
    share_component_size: float = Field(default=1.0, description="Size of buyer's connected component")
    buyer_seller_degree: float = Field(default=0.0)
    buyer_pagerank: float = Field(default=0.0)
    seller_buyer_degree: float = Field(default=0.0)
    seller_pagerank: float = Field(default=0.0)
    seller_buyer_concentration_hhi: float = Field(default=0.0)
    buyer_seller_edge_weight_before: float = Field(default=0.0)

    # --- Raw inputs for convenience (used to derive ratio features server-side) ---
    amount: Optional[float] = Field(
        default=None,
        description="Raw order amount — used with base_price / category_median_price "
                    "to compute ratio features if those are not supplied directly.",
    )
    base_price: Optional[float] = Field(
        default=None,
        description="Catalog base price for the listed product.",
    )
    category_median_price: Optional[float] = Field(
        default=None,
        description="Median listing price for the product's category (training-set statistic).",
    )
    multimodal_similarity_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Cosine similarity between listing image and text [0, 1]. When supplied, drives multimodal_risk in the trust engine.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "order_id": "ORDER_000123",
                "buyer_id": "BUYER_00042",
                "seller_id": "SELLER_00007",
                "order_amount": 85.99,
                "amount": 85.99,
                "base_price": 120.0,
                "category_median_price": 100.0,
                "buyer_orders_before": 3,
                "buyer_returns_before": 1,
                "buyer_age_days": 120.0,
                "seller_age_days": 300.0,
                "seller_total_listings_before": 42,
            }
        }
    )


class TransactionScoreResponse(BaseModel):
    order_id: Optional[str] = None
    overall_fraud_probability: float
    raw_fraud_probability: Optional[float] = Field(
        default=None,
        description="Raw uncalibrated classifier probability output from active ML model.",
    )
    calibrated_fraud_probability: Optional[float] = Field(
        default=None,
        description="Isotonically calibrated probability from primary predictive model.",
    )
    canonical_trust_engine_risk: Optional[float] = Field(
        default=None,
        description="Canonical Trust Engine composite risk score governing decision routing.",
    )
    risk_label: str                  # "low" | "medium" | "high"
    model_used: str
    model_version: str = "phase3"    # "phase3" | "phase5-hybrid"
    note: str = ""
    # Canonical Trust Engine real output fields (no fake defaults)
    decision: str = Field(
        description="Canonical Trust Engine decision: ALLOW, REVIEW, HOLD, or BLOCK.",
    )
    trust_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Calibrated trust score on [0, 100] scale (100 = full trust).",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score based on Shannon binary entropy [0, 1].",
    )
    predictive_uncertainty: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Predictive uncertainty (Shannon entropy in [0, 1]).",
    )
    cold_start: bool = Field(
        description="True when buyer or seller has minimal interaction history.",
    )
    model_disagreement: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Spread between highest and lowest component model probabilities.",
    )
    reason_codes: List[str] = Field(
        default_factory=list,
        description="Structured explainable reason codes for this decision.",
    )
    evidence_availability: Optional[Dict[str, bool]] = Field(
        default=None,
        description="Explicit breakdown of evidence channels available for this scoring evaluation.",
    )
    infrastructure_sources: Optional[Dict[str, str]] = Field(
        default=None,
        description="Explicit provenance of infrastructure lookups: 'redis' | 'neo4j' | 'disk_artifact' | 'unavailable'.",
    )
    shap_attributions: Optional[Dict[str, float]] = Field(
        default=None,
        description="Local TreeSHAP feature attributions indicating how each feature shifted predicted risk.",
    )
    top_risk_drivers: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Top positive risk driving features sorted by SHAP magnitude.",
    )
    scoring_mode: str = Field(
        default="production_model",
        description="Explicit provenance of scoring engine: 'production_model' | 'offline_fallback' | 'simulated'.",
    )
    is_simulation: bool = Field(
        default=False,
        description="Flag indicating whether this transaction scoring was artificially simulated.",
    )


# ---------------------------------------------------------------------------
# /transaction/explain  (POST) — TreeSHAP Feature Attribution API
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
    risk_label: str  # "low" | "medium" | "high"
    members: List[str] = []


class FraudRingsResponse(BaseModel):
    total_rings: int
    high_risk_rings: int           # avg_risk_score >= 0.5
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
# /listing/analyze  (POST) — Phase 4 Multimodal Detector
# ---------------------------------------------------------------------------

class ListingScoreRequest(BaseModel):
    """Request schema for the Phase 4 listing fraud detector.

    CLIP auto-scoring (preferred):
        Supply product_id + displayed_product_id (and optionally image_ref).
        The server will compute multimodal_similarity_score via real CLIP
        embeddings if the scorer is loaded, or fall back to TF-IDF.

    Legacy pre-computed mode:
        Supply multimodal_similarity_score directly (backward-compatible).
        If neither product_id nor similarity is supplied the field defaults to 0.85.
    """
    listing_id: Optional[str] = None
    seller_id: Optional[str] = None

    # --- Price features ---
    price: float = Field(default=100.0, description="Listing price")
    base_price: float = Field(default=100.0, description="Catalog base price")
    category_median_price: float = Field(default=100.0, description="Category median price")

    # --- Seller features ---
    seller_age_days_at_listing: float = Field(default=30.0, description="Seller age at listing time (days)")
    seller_listings_before: int = Field(default=5, description="Prior listings count")

    # --- CLIP routing fields (Phase 4 real signal) ---
    product_id: Optional[str] = Field(
        default=None,
        description=(
            "ABO product_id of the catalog item. When provided together with "
            "displayed_product_id, the server derives multimodal_similarity_score "
            "via CLIP image-text embeddings automatically."
        ),
    )
    displayed_product_id: Optional[str] = Field(
        default=None,
        description=(
            "product_id of the image actually shown in this listing. "
            "For genuine listings this equals product_id. "
            "For fake listings it differs — this drives the CLIP mismatch signal."
        ),
    )
    image_ref: Optional[str] = Field(
        default=None,
        description=(
            "Relative path to the displayed image within the ABO dataset "
            "(e.g. 'images/small/8c/8ccb5859.jpg'). Optional hint — the server "
            "resolves the image from displayed_product_id when omitted."
        ),
    )

    # --- Pre-computed fallback (backward-compatible) ---
    multimodal_similarity_score: float = Field(
        default=0.85,
        description=(
            "Cosine similarity between listing text and displayed image [0, 1]. "
            "Used directly when product_id is not supplied. "
            "0.80+ = normal, <0.65 = moderate concern, <0.45 = severe mismatch."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "listing_id": "LISTING_000123",
                "seller_id": "SELLER_00007",
                "price": 85.99,
                "base_price": 120.0,
                "category_median_price": 100.0,
                "seller_age_days_at_listing": 14.0,
                "seller_listings_before": 2,
                "product_id": "B06X9STHNG",
                "displayed_product_id": "B07P8ML82R",
            }
        }
    )


class ListingScoreResponse(BaseModel):
    listing_id: Optional[str]
    fake_listing_probability: float
    risk_label: str
    model_used: str = "Fake Listing Detector (Phase 4 Multimodal XGBoost)"
    clip_scored: bool = Field(
        default=False,
        description="True when multimodal_similarity_score was computed via real CLIP embeddings.",
    )
    multimodal_similarity_score: float = Field(
        default=0.85,
        description="The image-text similarity score used for this prediction.",
    )
    investigator_narrative: Dict[str, Any]


# ---------------------------------------------------------------------------
# /return/analyze  (POST) — Phase 2 Specialized Return Fraud Detector
# ---------------------------------------------------------------------------

class ReturnScoreRequest(BaseModel):
    return_id: Optional[str] = None
    order_id: Optional[str] = None
    buyer_id: Optional[str] = None
    seller_id: Optional[str] = None

    days_to_return: float = Field(default=7.0, ge=0.0, description="Days between order date and return request")
    buyer_age_days_at_return: float = Field(default=90.0, ge=0.0, description="Buyer account age in days at return time")
    seller_age_days_at_return: float = Field(default=180.0, ge=0.0, description="Seller account age in days at return time")
    order_amount: float = Field(default=50.0, ge=0.0, description="Monetary value of the returned item")
    buyer_prior_returns: int = Field(default=0, ge=0, description="Buyer's historical return count")
    buyer_orders_before_return: int = Field(default=5, ge=0, description="Buyer's historical order count")
    buyer_return_rate_before: float = Field(default=0.0, ge=0.0, le=1.0, description="Buyer's prior return rate")
    seller_prior_returns: int = Field(default=2, ge=0, description="Seller's historical return count")
    seller_orders_before_return: int = Field(default=20, ge=0, description="Seller's historical order count")
    seller_return_rate_before: float = Field(default=0.1, ge=0.0, le=1.0, description="Seller's prior return rate")
    reason: Optional[str] = Field(
        default="defective",
        description="Return reason: 'changed_mind', 'defective', 'size_issue', 'wrong_item_received'",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "return_id": "RET_000456",
                "order_id": "ORD_000123",
                "buyer_id": "BUYER_000042",
                "seller_id": "SELLER_000007",
                "days_to_return": 3.0,
                "buyer_age_days_at_return": 14.0,
                "seller_age_days_at_return": 300.0,
                "order_amount": 120.0,
                "buyer_prior_returns": 4,
                "buyer_orders_before_return": 5,
                "buyer_return_rate_before": 0.8,
                "seller_prior_returns": 2,
                "seller_orders_before_return": 50,
                "seller_return_rate_before": 0.04,
                "reason": "defective",
            }
        }
    )


class ReturnScoreResponse(BaseModel):
    return_id: Optional[str] = None
    return_fraud_probability: float
    risk_label: str  # "low" | "medium" | "high"
    decision: str    # "ALLOW" | "REVIEW" | "HOLD" | "BLOCK"
    model_used: str = "Return Fraud Detector (Phase 2 Specialized Model)"
    reason_codes: List[str] = []


# ---------------------------------------------------------------------------
# /ready (GET) - Readiness probe for service orchestrator / load balancer
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
# /investigation/generate-dossier (POST) — Forensic Dossier API
# ---------------------------------------------------------------------------

class DossierRequest(BaseModel):
    entity_type: str = Field(
        default="transaction",
        description="Type of entity under investigation: 'transaction' | 'order' | 'buyer' | 'seller' | 'ring'",
    )
    entity_id: str = Field(
        description="Unique identifier of entity to investigate (e.g. ORD_78901, BUYER_000042, RING_001)",
    )
    transaction_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional explicit transaction payload to override or supply missing offline context",
    )
    include_graph_evidence: bool = Field(
        default=True,
        description="Whether to query live Neo4j/graph artifacts for multi-hop neighbor analysis",
    )


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
    graph_source: str  # "neo4j" | "disk_artifact" | "unavailable"
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
# /stream/transactions (GET SSE) — Real-time Transaction Stream
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
    risk_level: str
    decision: str
    trust_score: float
    reason_codes: List[str] = Field(default_factory=list)
    source: str = "simulation_stream"  # "simulation_stream" | "live_stream"
    is_simulation: bool = True

