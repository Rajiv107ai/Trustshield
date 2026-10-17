// ============================================================================
// TRUSTSHIELD AI — Scenario Presets & Human Translation Dictionary
// ============================================================================

import { TransactionScoreRequest } from "../types/api";

export interface ScenarioPreset {
  id: string;
  label: string;
  badge: string;
  decision: "ALLOW" | "REVIEW" | "HOLD" | "BLOCK";
  color: "emerald" | "amber" | "orange" | "rose";
  description: string;
  payload: TransactionScoreRequest;
}

const sp = (
  id: string,
  label: string,
  badge: string,
  decision: ScenarioPreset["decision"],
  color: ScenarioPreset["color"],
  description: string,
  payload: TransactionScoreRequest
): ScenarioPreset => ({ id, label, badge, decision, color, description, payload });

/* BEGIN GENERATED */
export const SCENARIO_PRESETS: ScenarioPreset[] = [
  sp("preset_normal_buyer", "1. Verified Repeat Buyer", "LOW RISK (ALLOW)", "ALLOW", "emerald",
    "Account active for 180 days, 15 prior successful orders, zero returns, single device.", {
      order_id: "ORD_SIM_SAFE_01", buyer_id: "BUYER_VERIFIED_77", seller_id: "SELLER_REPUTABLE_12",
      amount: 65.50, base_price: 65.00, category_median_price: 70.00, buyer_age_days: 180,
      buyer_orders_before: 15, buyer_returns_before: 0, device_shared_buyer_count: 1.0,
      share_degree: 0.0, buyer_pagerank: 0.00045, share_component_size: 1,
    }),
  sp("preset_price_arbitrage", "2. New Account Price Anomaly", "SUSPICIOUS (REVIEW)", "REVIEW", "amber",
    "New account (1 day old) ordering at 3.5x normal catalog price.", {
      order_id: "ORD_SIM_ANOMALY_02", buyer_id: "BUYER_NEWBIE_99", seller_id: "SELLER_UNKNOWN_44",
      amount: 450.00, base_price: 120.00, category_median_price: 110.00, buyer_age_days: 1,
      buyer_orders_before: 0, buyer_returns_before: 0, device_shared_buyer_count: 1.0,
      share_degree: 1.0, buyer_pagerank: 0.00012, share_component_size: 2,
    }),
  sp("preset_device_farm", "3. Device Farm Collusion Ring", "HIGH RISK (BLOCK)", "BLOCK", "rose",
    "Shared hardware device used across 9 distinct buyer accounts in 24 hours.", {
      order_id: "ORD_SIM_FARM_03", buyer_id: "BUYER_RING_MEMBER_04", seller_id: "SELLER_RING_LEADER_01",
      amount: 890.00, base_price: 850.00, category_median_price: 800.00, buyer_age_days: 4,
      buyer_orders_before: 1, buyer_returns_before: 1, device_shared_buyer_count: 9.0,
      share_degree: 8.5, seller_buyer_concentration_hhi: 0.88, buyer_pagerank: 0.0034, share_component_size: 14,
    }),
  sp("preset_serial_returner", "4. Serial Return Abuse", "HOLD (INVESTIGATE)", "HOLD", "orange",
    "Buyer with 80% historical return rate submitting an expensive claim.", {
      order_id: "ORD_SIM_RETURN_04", buyer_id: "BUYER_REFUND_ABUSER", seller_id: "SELLER_ELECTRONICS_09",
      amount: 320.00, base_price: 320.00, category_median_price: 300.00, buyer_age_days: 60,
      buyer_orders_before: 5, buyer_returns_before: 4, buyer_return_rate_before: 0.80,
      device_shared_buyer_count: 2.0, share_degree: 1.5, buyer_pagerank: 0.0008, share_component_size: 3,
    }),
];

export interface ReasonCodeTranslation {
  title: string;
  explanation: string;
  severity: "critical" | "warning" | "info";
}

const rc = (title: string, explanation: string, severity: ReasonCodeTranslation["severity"]): ReasonCodeTranslation =>
  ({ title, explanation, severity });

export const REASON_CODE_TRANSLATIONS: Record<string, ReasonCodeTranslation> = {
  HIGH_DEVICE_COLLISION: rc("Shared Device Alert", "This computer or mobile hardware fingerprint has been used by multiple separate buyer accounts recently.", "critical"),
  PRICE_OUTLIER_99TH_PCT: rc("Unusual Price Alert", "The purchase amount deviates sharply (>99th percentile) from category pricing benchmarks.", "warning"),
  COLD_START_BUYER_HIGH_VALUE: rc("New Account High Spend", "This customer profile is under 72 hours old and is executing an unusually high-value first purchase.", "warning"),
  RING_COLLUSION_SUSPECT: rc("Coordinated Ring Signal", "Graph topology detected dense bipartite ties and shared device/address collusion with this seller.", "critical"),
  CROSS_SELLER_IMAGE_REUSE: rc("Potential Counterfeit / Photo Theft", "The listing imagery has a 99%+ FAISS cosine collision with an established catalog product from another merchant.", "critical"),
  HIGH_RETURN_VELOCITY: rc("Serial Return Risk", "Historical behavioral telemetry shows a >75% return velocity and recurring empty-box chargebacks.", "warning"),
  SUSPICIOUS_BUYER_VELOCITY: rc("Rapid Transaction Burst", "Order frequency exceeds normal human checkout intervals within a 15-minute window.", "warning"),
  ANOMALOUS_MERCHANT_CONCENTRATION: rc("High Merchant Concentration", "Herfindahl-Hirschman Index indicates extreme captive purchasing exclusively with a single seller.", "warning"),
};
/* END GENERATED */
