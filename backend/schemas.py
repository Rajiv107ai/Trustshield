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

from typing import List, Optional
from pydantic import BaseModel, Field


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
        description="order amount / catalog base price (computed server-side "
                    "from amount + base_price if not supplied directly)",
    )
    price_vs_category_median_ratio: float = Field(
        default=0.0,
        description="order amount / category median price (computed server-side "
                    "from amount + category_median_price if not supplied directly)",
    )
    seller_age_days: float = Field(default=0.0, description="Days since seller signup at order time")
    seller_total_listings_before: float = Field(default=0.0, description="Seller's listing count before this order")
    buyer_age_days: float = Field(default=0.0, description="Days since buyer signup at order time")
    buyer_orders_before: int = Field(default=0, description="Buyer's prior order count")
    buyer_returns_before: int = Field(default=0, description="Buyer's prior return count")
    buyer_return_rate_before: float = Field(
        default=0.0,
        description="buyer_returns_before / max(buyer_orders_before, 1) "
                    "(computed server-side if not supplied directly)",
    )
    device_shared_buyer_count: float = Field(
        default=1.0,
        description="Number of distinct buyers sharing this device (from training-set graph)",
    )
    order_amount: float = Field(default=0.0, description="Order value")

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

    class Config:
        json_schema_extra = {
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


class TransactionScoreResponse(BaseModel):
    order_id: Optional[str]
    overall_fraud_probability: float
    risk_label: str                # "low" | "medium" | "high"
    model_used: str
    model_version: str = "phase3"  # "phase3" | "phase5-hybrid"
    note: str = ""


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
    rings_loaded: bool
    n_rings: int
