// ============================================================================
// TRUSTSHIELD AI — Offline Demo Adapter (Deterministic Fixtures)
// Strictly separated from live HTTP transport. Contains NO duplicate scoring
// algorithms; provides static preset fixtures for offline UI presentation.
// ============================================================================

import {
  TransactionScoreRequest,
  TransactionScoreResponse,
  TransactionExplainRequest,
  TransactionExplainResponse,
  FraudRingsResponse,
  FraudRingItem,
  ListingScoreRequest,
  ListingScoreResponse,
  ReturnScoreRequest,
  ReturnScoreResponse,
  DossierRequest,
  DossierResponse,
  HealthResponse,
  ReadyResponse,
  SystemBenchmarkResponse,
  StreamTransactionEvent,
  RiskLabel,
  OperationalDecision,
} from "../types/api";
import { RISK_THRESHOLDS } from "../constants/thresholds";

export function demoGetHealth(): HealthResponse {
  return {
    status: "healthy (offline demo mode)",
    models_loaded: false,
    clip_loaded: false,
    rings_loaded: false,
    n_rings: 0,
  };
}

export function demoGetReady(): ReadyResponse {
  return {
    status: "not_ready",
    models_ready: false,
    phase3_ready: false,
    clip_ready: false,
    rings_ready: false,
    details: {
      mode: "Offline Demo Simulation",
    },
  };
}

function makeScoreResp(
  orderId: string, prob: number, set: number[], riskLabel: RiskLabel,
  decision: OperationalDecision, trustScore: number,
  confidence: number, uncertainty: number, coldStart: boolean,
  disagreement: number, reasonCodes: string[], name: string, note: string
): TransactionScoreResponse {
  return {
    order_id: orderId,
    overall_fraud_probability: prob,
    raw_fraud_probability: prob,
    calibrated_fraud_probability: prob,
    policy_risk_score: prob,
    canonical_trust_engine_risk: prob,
    conformal_prediction_set: set,
    conformal_is_calibrated: false,
    conformal_coverage_guarantee: undefined,
    risk_label: riskLabel,
    decision,
    trust_score: trustScore,
    confidence,
    predictive_uncertainty: uncertainty,
    cold_start: coldStart,
    model_disagreement: disagreement,
    reason_codes: reasonCodes,
    model_used: `Offline Simulation (${name})`,
    model_version: "offline-demo-simulation",
    note,
    evidence_availability: {
      tabular: true,
      graph_topology: false,
      gnn_embeddings: false,
      conformal_coverage: true,
      clip_visual: false,
    },
  };
}

export function demoScoreTransaction(req: TransactionScoreRequest): TransactionScoreResponse {
  const orderId = req.order_id || `ORD_DEMO_${Math.floor(Math.random() * 90000 + 10000)}`;
  const amt = req.amount ?? req.order_amount ?? 100.0;

  if (orderId === "ORD_SIM_SAFE_01" || orderId.includes("SAFE") || (amt < 100 && (req.buyer_orders_before ?? 0) >= 10)) {
    return makeScoreResp(orderId, 0.0245, [0], "low", "ALLOW", 97.55, 0.95, 0.16, false, 0.012,
      ["VERIFIED_BUYER_HISTORY"], "Verified Preset", "Evaluated in offline demonstration mode (verified preset scenario)");
  }
  if (orderId === "ORD_SIM_ANOMALY_02" || orderId.includes("ANOMALY") || (amt > 300 && amt <= 600)) {
    return makeScoreResp(orderId, 0.385, [0, 1], "medium", "REVIEW", 61.5, 0.72, 0.65, (req.buyer_age_days ?? 30) <= 3, 0.045,
      ["PRICE_OUTLIER_99TH_PCT", "COLD_START_BUYER_HIGH_VALUE"], "Price Anomaly Preset", "Evaluated in offline demonstration mode (price anomaly preset)");
  }
  if (orderId === "ORD_SIM_FARM_03" || orderId.includes("FARM") || (req.device_shared_buyer_count ?? 1) >= 5 || amt > 600) {
    return makeScoreResp(orderId, 0.895, [1], "high", "BLOCK", 10.5, 0.88, 0.48, false, 0.021,
      ["HIGH_DEVICE_COLLISION", "RING_COLLUSION_SUSPECT"], "Collusion Preset", "Evaluated in offline demonstration mode (collusion ring preset)");
  }
  if (orderId === "ORD_SIM_RETURN_04" || orderId.includes("RETURN") || (req.buyer_return_rate_before ?? 0) >= RISK_THRESHOLDS.REVIEW * 2) {
    return makeScoreResp(orderId, 0.582, [1], "high", "HOLD", 41.8, 0.78, 0.68, false, 0.038,
      ["HIGH_RETURN_VELOCITY", "SUSPICIOUS_REFUND_CLAIM"], "Return Abuse Preset", "Evaluated in offline demonstration mode (return abuse preset)");
  }
  return makeScoreResp(orderId, 0.05, [0], "low", "ALLOW", 95.0, 0.9, 0.28, (req.buyer_age_days ?? 30) <= 2, 0.02,
    ["CLEAN_BASELINE"], "Deterministic Preset", "Evaluated in offline demonstration mode. Connect FastAPI backend for live inference.");
}

const r = (ring_id: string, size: number, n_orders: number, avg_risk_score: number, max_risk_score: number, n_high_risk_orders: number, risk_label: RiskLabel, members: string[]): FraudRingItem =>
  ({ ring_id, size, n_orders, avg_risk_score, max_risk_score, n_high_risk_orders, risk_label, members });

export function demoGetFraudRings(): FraudRingsResponse {
  return {
    total_rings: 6,
    high_risk_rings: 4,
    rings: [
      r("RING_COLLUSION_01", 8, 42, 0.912, 0.985, 39, "high", ["BUYER_401", "BUYER_402", "BUYER_403", "BUYER_404", "BUYER_405", "SELLER_91", "SELLER_92", "DEV_FARM_99"]),
      r("RING_PRICE_ARBITRAGE_02", 5, 28, 0.845, 0.92, 24, "high", ["BUYER_112", "BUYER_114", "BUYER_118", "SELLER_44", "ADDR_WH_01"]),
      r("RING_RETURN_ABUSE_03", 4, 19, 0.78, 0.89, 15, "high", ["BUYER_REFUND_01", "BUYER_REFUND_02", "SELLER_88", "DEV_MOBILE_04"]),
      r("RING_DEVICE_FARM_04", 11, 63, 0.73, 0.865, 41, "high", ["BUYER_501", "BUYER_502", "BUYER_503", "BUYER_504", "BUYER_505", "BUYER_506", "SELLER_10", "SELLER_12"]),
      r("RING_COMMUNITY_05", 6, 31, 0.42, 0.58, 6, "medium", ["BUYER_201", "BUYER_202", "BUYER_203", "SELLER_31", "SELLER_32"]),
      r("RING_BENIGN_CLUSTER_06", 3, 12, 0.11, 0.22, 0, "low", ["BUYER_10", "BUYER_11", "SELLER_05"]),
    ],
  };
}

export function demoAnalyzeListing(req: ListingScoreRequest): ListingScoreResponse {
  const sim = req.multimodal_similarity_score ?? (req.displayed_product_id && req.displayed_product_id !== req.product_id ? 0.94 : 0.42);
  const fakeProb = sim > RISK_THRESHOLDS.BLOCK ? 0.78 : 0.08;
  return {
    listing_id: req.listing_id || "LISTING_DEMO_01",
    fake_listing_probability: fakeProb,
    risk_label: fakeProb >= RISK_THRESHOLDS.HOLD ? "high" : "low",
    model_used: "Offline Simulation (Heuristic Fallback)",
    clip_scored: false,
    multimodal_similarity_score: sim,
    investigator_narrative: {
      visual_collision_detected: sim > RISK_THRESHOLDS.BLOCK,
      catalog_match_id: req.product_id || "PROD_B07XYZ99",
      flagged_reason: sim > RISK_THRESHOLDS.BLOCK ? "Simulation: Cross-seller photo reuse detected" : "Simulation: Original imagery verified",
      faiss_cosine_similarity: sim,
    },
  };
}

export function demoAnalyzeReturn(req: ReturnScoreRequest): ReturnScoreResponse {
  const retRate = req.buyer_return_rate_before ?? (req.buyer_prior_returns && req.buyer_orders_before_return ? req.buyer_prior_returns / req.buyer_orders_before_return : 0.2);
  const prob = retRate > 0.6 ? 0.82 : 0.12;
  return {
    return_id: req.return_id || "RET_DEMO_01",
    return_fraud_probability: prob,
    risk_label: prob >= RISK_THRESHOLDS.HOLD ? "high" : "low",
    decision: prob >= RISK_THRESHOLDS.BLOCK ? "BLOCK" : prob >= RISK_THRESHOLDS.HOLD ? "HOLD" : prob >= RISK_THRESHOLDS.REVIEW ? "REVIEW" : "ALLOW",
    model_used: "Offline Simulation (Heuristic Fallback)",
    reason_codes: prob >= RISK_THRESHOLDS.HOLD ? ["HIGH_RETURN_VELOCITY", "SUSPICIOUS_REFUND_CLAIM"] : [],
  };
}

export function demoGenerateDossier(req: DossierRequest): DossierResponse {
  const isBlock = req.entity_id.includes("RING") || req.entity_id.includes("01");
  const risk = isBlock ? 0.942 : 0.085;
  const decision = isBlock ? "BLOCK" : "ALLOW";
  return {
    case_id: `CASE_${req.entity_id}`,
    entity_type: req.entity_type,
    entity_id: req.entity_id,
    generated_at: new Date().toISOString(),
    risk_score: risk,
    risk_level: isBlock ? "CRITICAL" : "LOW",
    decision,
    executive_summary: `Offline demonstration dossier synthesized for ${req.entity_type} ${req.entity_id}. Risk score ${risk * 100}% with automated decision ${decision}.`,
    observed_evidence: {
      transaction_amount: 890.0,
      buyer_historical_return_rate: 0.1,
      account_age_days: 14,
      prior_order_count: 2,
      hardware_collision_detected: isBlock,
      shared_device_count: isBlock ? 6 : 1,
      verified_facts: [
        "Order volume verified across demo ledger",
        isBlock ? "Hardware collision detected across demo buyer accounts" : "Clean device signature",
      ],
    },
    model_inference: {
      calibrated_risk_score: risk,
      operational_decision: decision,
      confidence_level: 0.91,
      detector_disagreement: 0.04,
      conformal_prediction_set: isBlock ? "{1}" : "{0}",
      triggered_reason_codes: isBlock ? ["HIGH_DEVICE_COLLISION", "RING_COLLUSION_SUSPECT"] : ["CLEAN_BASELINE"],
    },
    graph_findings: {
      graph_source: "offline_simulation",
      cluster_id: isBlock ? "RING_01" : undefined,
      cluster_size: isBlock ? 6 : 1,
      topology_summary: isBlock ? "6-node clique with shared device DEV_FARM_99" : "Isolated single-buyer topology",
      suspicious_relationships: isBlock ? [{ source: req.entity_id, target: "DEV_FARM_99", type: "USES_DEVICE" }] : [],
    },
    timeline: [
      { time: "T-24h", event: "Account created", type: "ACCOUNT" },
      { time: "T-2h", event: "Order placed", type: "ORDER" },
      { time: "T-0h", event: `Decision rendered: ${decision}`, type: "DECISION" },
    ],
    involved_entities: {
      entity_id: req.entity_id,
    },
    retrieved_policy_guidelines: [
      "[POL-01] Hardware Identifier & Device Collision Policy: Multi-account device collision requires step-up review.",
    ],
    recommendation: {
      action: isBlock ? "Decline order immediately and flag hardware fingerprint." : "Clear order for automated fulfillment.",
      protocol_level: isBlock ? "CRITICAL" : "LOW",
      required_evidence_to_clear: "Government photo ID and proof of delivery address.",
    },
    limitations: ["Offline deterministic simulation active while backend is offline or in explicit demo mode."],
    provenance: {
      model_version: "offline-simulation",
      graph_source: "synthetic_demo",
      engine: "TrustShield Offline Demo Generator",
    },
    grounding_verification_passed: false,
  };
}

export function demoBenchmark(): SystemBenchmarkResponse {
  return {
    status: "demo_simulation",
    benchmarks: {
      hybrid_inference: { unit: "ms", samples: 0, p50: 0, p95: 0, p99: 0, note: "Offline Demo (No live model benchmarked)" },
      trust_engine_scoring: { unit: "ms", samples: 0, p50: 0, p95: 0, p99: 0, note: "Offline Demo (No live model benchmarked)" },
      redis_get_embedding: { unit: "ms", samples: 0, p50: 0, p95: 0, p99: 0, note: "Redis service offline (simulation mode)" },
      neo4j_neighborhood: { unit: "ms", samples: 0, p50: 0, p95: 0, p99: 0, note: "Neo4j service offline (simulation mode)" },
    },
    measured_at: new Date().toISOString(),
  };
}

export function demoExplainTransaction(req: TransactionExplainRequest): TransactionExplainResponse {
  const orderId = req.order_id || "ORD_EXPLAIN_FALLBACK";
  return {
    order_id: orderId,
    base_value: -1.25,
    overall_fraud_probability: 0.15,
    decision: "ALLOW",
    model_used: "Offline Simulation (Heuristic Explainer)",
    top_positive_drivers: [
      {
        feature_name: "price_vs_base_price_ratio",
        friendly_name: "Price vs Catalog Base Price Ratio",
        feature_value: 1.05,
        shap_value: 0.24,
        abs_impact: 0.24,
      },
    ],
    top_negative_dampeners: [
      {
        feature_name: "seller_age_days",
        friendly_name: "Seller Account Age (Days)",
        feature_value: 180.0,
        shap_value: -0.42,
        abs_impact: 0.42,
      },
    ],
    all_attributions: {
      price_vs_base_price_ratio: 0.24,
      seller_age_days: -0.42,
    },
    investigator_narrative:
      "Offline demonstration explainer: mild risk mitigated by established seller tenure.",
  };
}

const s = (event_id: string, timestamp: string, order_id: string, buyer_id: string, seller_id: string, amount: number, risk_score: number, calibrated_risk: number, policy_risk_score: number, risk_level: RiskLabel, decision: OperationalDecision, trust_score: number, reason_codes: string[]): StreamTransactionEvent =>
  ({ event_id, timestamp, order_id, buyer_id, seller_id, amount, risk_score, calibrated_risk, policy_risk_score, risk_level, decision, trust_score, reason_codes, source: "offline_simulation" });

export const DEMO_STREAM_TEMPLATES: StreamTransactionEvent[] = [
  s("EVT_SIM_01", "12:00:01", "ORD_SIM_78930", "BUYER_RING_MEMBER_04", "SELLER_RING_LEADER_01", 890.0, 0.942, 0.938, 0.942, "high", "BLOCK", 6.2, ["HIGH_DEVICE_COLLISION", "RING_COLLUSION_SUSPECT"]),
  s("EVT_SIM_02", "12:00:03", "ORD_SIM_78931", "BUYER_ORGANIC_404", "SELLER_MERCHANT_10", 45.0, 0.035, 0.031, 0.035, "low", "ALLOW", 96.9, ["CLEAN_BASELINE"]),
  s("EVT_SIM_03", "12:00:05", "ORD_SIM_78932", "BUYER_REFUND_ABUSER", "SELLER_ELECTRONICS_09", 320.0, 0.785, 0.76, 0.785, "high", "HOLD", 24.0, ["HIGH_RETURN_VELOCITY", "SUSPICIOUS_REFUND_RATIO"]),
  s("EVT_SIM_04", "12:00:07", "ORD_SIM_78933", "BUYER_NEWBIE_99", "SELLER_UNKNOWN_44", 450.0, 0.38, 0.375, 0.38, "medium", "REVIEW", 62.5, ["PRICE_OUTLIER_99TH_PCT"]),
];
