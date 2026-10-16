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
} from "../types/api";

export function demoGetHealth(): HealthResponse {
  return {
    status: "healthy (offline demo mode)",
    models_loaded: false,
    phase5_loaded: false,
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
    phase5_ready: false,
    clip_ready: false,
    rings_ready: false,
    details: {
      mode: "Offline Demo Simulation",
      neo4j_mesh: "Offline (Demo mode)",
      redis_cache: "Offline (Demo mode)",
    },
  };
}

export function demoScoreTransaction(req: TransactionScoreRequest): TransactionScoreResponse {
  const orderId = req.order_id || `ORD_DEMO_${Math.floor(Math.random() * 90000 + 10000)}`;
  const amt = req.amount ?? req.order_amount ?? 100.0;

  // Preset 1: Verified repeat buyer (Allow)
  if (orderId === "ORD_SIM_SAFE_01" || orderId.includes("SAFE") || (amt < 100 && (req.buyer_orders_before ?? 0) >= 10)) {
    return {
      order_id: orderId,
      overall_fraud_probability: 0.0245,
      raw_fraud_probability: 0.0245,
      calibrated_fraud_probability: 0.0245,
      policy_risk_score: 0.0245,
      canonical_trust_engine_risk: 0.0245,
      conformal_prediction_set: [0],
      conformal_is_calibrated: false,
      conformal_coverage_guarantee: 0.95,
      risk_label: "low",
      decision: "ALLOW",
      trust_score: 97.55,
      confidence: 0.95,
      predictive_uncertainty: 0.16,
      cold_start: false,
      model_disagreement: 0.012,
      reason_codes: ["VERIFIED_BUYER_HISTORY"],
      model_used: "Offline Simulation (Verified Preset)",
      model_version: "offline-demo-simulation",
      note: "Evaluated in offline demonstration mode (verified preset scenario)",
      evidence_availability: {
        tabular: true,
        graph_topology: false,
        gnn_embeddings: false,
        conformal_coverage: true,
        clip_visual: false,
      },
    };
  }

  // Preset 2: Price anomaly / Arbitrage (Review)
  if (orderId === "ORD_SIM_ANOMALY_02" || orderId.includes("ANOMALY") || (amt > 300 && amt <= 600)) {
    return {
      order_id: orderId,
      overall_fraud_probability: 0.385,
      raw_fraud_probability: 0.385,
      calibrated_fraud_probability: 0.385,
      policy_risk_score: 0.385,
      canonical_trust_engine_risk: 0.385,
      conformal_prediction_set: [0, 1],
      conformal_is_calibrated: false,
      conformal_coverage_guarantee: 0.95,
      risk_label: "medium",
      decision: "REVIEW",
      trust_score: 61.5,
      confidence: 0.72,
      predictive_uncertainty: 0.65,
      cold_start: (req.buyer_age_days ?? 30) <= 3,
      model_disagreement: 0.045,
      reason_codes: ["PRICE_OUTLIER_99TH_PCT", "COLD_START_BUYER_HIGH_VALUE"],
      model_used: "Offline Simulation (Price Anomaly Preset)",
      model_version: "offline-demo-simulation",
      note: "Evaluated in offline demonstration mode (price anomaly preset)",
      evidence_availability: {
        tabular: true,
        graph_topology: false,
        gnn_embeddings: false,
        conformal_coverage: true,
        clip_visual: false,
      },
    };
  }

  // Preset 3: Device farm / Collusion (Block)
  if (orderId === "ORD_SIM_FARM_03" || orderId.includes("FARM") || (req.device_shared_buyer_count ?? 1) >= 5 || amt > 600) {
    return {
      order_id: orderId,
      overall_fraud_probability: 0.895,
      raw_fraud_probability: 0.895,
      calibrated_fraud_probability: 0.895,
      policy_risk_score: 0.895,
      canonical_trust_engine_risk: 0.895,
      conformal_prediction_set: [1],
      conformal_is_calibrated: false,
      conformal_coverage_guarantee: 0.95,
      risk_label: "high",
      decision: "BLOCK",
      trust_score: 10.5,
      confidence: 0.88,
      predictive_uncertainty: 0.48,
      cold_start: false,
      model_disagreement: 0.021,
      reason_codes: ["HIGH_DEVICE_COLLISION", "RING_COLLUSION_SUSPECT"],
      model_used: "Offline Simulation (Collusion Preset)",
      model_version: "offline-demo-simulation",
      note: "Evaluated in offline demonstration mode (collusion ring preset)",
      evidence_availability: {
        tabular: true,
        graph_topology: false,
        gnn_embeddings: false,
        conformal_coverage: true,
        clip_visual: false,
      },
    };
  }

  // Preset 4: Serial returner (Hold)
  if (orderId === "ORD_SIM_RETURN_04" || orderId.includes("RETURN") || (req.buyer_return_rate_before ?? 0) >= 0.5) {
    return {
      order_id: orderId,
      overall_fraud_probability: 0.582,
      raw_fraud_probability: 0.582,
      calibrated_fraud_probability: 0.582,
      policy_risk_score: 0.582,
      canonical_trust_engine_risk: 0.582,
      conformal_prediction_set: [1],
      conformal_is_calibrated: false,
      conformal_coverage_guarantee: 0.95,
      risk_label: "high",
      decision: "HOLD",
      trust_score: 41.8,
      confidence: 0.78,
      predictive_uncertainty: 0.68,
      cold_start: false,
      model_disagreement: 0.038,
      reason_codes: ["HIGH_RETURN_VELOCITY", "SUSPICIOUS_REFUND_CLAIM"],
      model_used: "Offline Simulation (Return Abuse Preset)",
      model_version: "offline-demo-simulation",
      note: "Evaluated in offline demonstration mode (return abuse preset)",
      evidence_availability: {
        tabular: true,
        graph_topology: false,
        gnn_embeddings: false,
        conformal_coverage: true,
        clip_visual: false,
      },
    };
  }

  // General Baseline Fallback
  return {
    order_id: orderId,
    overall_fraud_probability: 0.05,
    raw_fraud_probability: 0.05,
    calibrated_fraud_probability: 0.05,
    policy_risk_score: 0.05,
    canonical_trust_engine_risk: 0.05,
    conformal_prediction_set: [0],
    conformal_is_calibrated: false,
    conformal_coverage_guarantee: 0.95,
    risk_label: "low",
    decision: "ALLOW",
    trust_score: 95.0,
    confidence: 0.9,
    predictive_uncertainty: 0.28,
    cold_start: (req.buyer_age_days ?? 30) <= 2,
    model_disagreement: 0.02,
    reason_codes: ["CLEAN_BASELINE"],
    model_used: "Offline Simulation (Deterministic Preset)",
    model_version: "offline-demo-simulation",
    note: "Evaluated in offline demonstration mode. Connect FastAPI backend for live inference.",
    evidence_availability: {
      tabular: true,
      graph_topology: false,
      gnn_embeddings: false,
      conformal_coverage: true,
      clip_visual: false,
    },
  };
}

export function demoGetFraudRings(): FraudRingsResponse {
  return {
    total_rings: 6,
    high_risk_rings: 4,
    rings: [
      {
        ring_id: "RING_COLLUSION_01",
        size: 8,
        n_orders: 42,
        avg_risk_score: 0.912,
        max_risk_score: 0.985,
        n_high_risk_orders: 39,
        risk_label: "high",
        members: ["BUYER_401", "BUYER_402", "BUYER_403", "BUYER_404", "BUYER_405", "SELLER_91", "SELLER_92", "DEV_FARM_99"],
      },
      {
        ring_id: "RING_PRICE_ARBITRAGE_02",
        size: 5,
        n_orders: 28,
        avg_risk_score: 0.845,
        max_risk_score: 0.92,
        n_high_risk_orders: 24,
        risk_label: "high",
        members: ["BUYER_112", "BUYER_114", "BUYER_118", "SELLER_44", "ADDR_WH_01"],
      },
      {
        ring_id: "RING_RETURN_ABUSE_03",
        size: 4,
        n_orders: 19,
        avg_risk_score: 0.78,
        max_risk_score: 0.89,
        n_high_risk_orders: 15,
        risk_label: "high",
        members: ["BUYER_REFUND_01", "BUYER_REFUND_02", "SELLER_88", "DEV_MOBILE_04"],
      },
      {
        ring_id: "RING_DEVICE_FARM_04",
        size: 11,
        n_orders: 63,
        avg_risk_score: 0.73,
        max_risk_score: 0.865,
        n_high_risk_orders: 41,
        risk_label: "high",
        members: ["BUYER_501", "BUYER_502", "BUYER_503", "BUYER_504", "BUYER_505", "BUYER_506", "SELLER_10", "SELLER_12"],
      },
      {
        ring_id: "RING_COMMUNITY_05",
        size: 6,
        n_orders: 31,
        avg_risk_score: 0.42,
        max_risk_score: 0.58,
        n_high_risk_orders: 6,
        risk_label: "medium",
        members: ["BUYER_201", "BUYER_202", "BUYER_203", "SELLER_31", "SELLER_32"],
      },
      {
        ring_id: "RING_BENIGN_CLUSTER_06",
        size: 3,
        n_orders: 12,
        avg_risk_score: 0.11,
        max_risk_score: 0.22,
        n_high_risk_orders: 0,
        risk_label: "low",
        members: ["BUYER_10", "BUYER_11", "SELLER_05"],
      },
    ],
  };
}

export function demoAnalyzeListing(req: ListingScoreRequest): ListingScoreResponse {
  const sim = req.multimodal_similarity_score ?? (req.displayed_product_id && req.displayed_product_id !== req.product_id ? 0.94 : 0.42);
  const fakeProb = sim > 0.85 ? 0.78 : 0.08;
  return {
    listing_id: req.listing_id || "LISTING_DEMO_01",
    fake_listing_probability: fakeProb,
    risk_label: fakeProb > 0.5 ? "high" : "low",
    model_used: "Offline Simulation (Heuristic Fallback)",
    clip_scored: false,
    multimodal_similarity_score: sim,
    investigator_narrative: {
      visual_collision_detected: sim > 0.85,
      catalog_match_id: req.product_id || "PROD_B07XYZ99",
      flagged_reason: sim > 0.85 ? "Simulation: Cross-seller photo reuse detected" : "Simulation: Original imagery verified",
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
    risk_label: prob > 0.5 ? "high" : "low",
    decision: prob > 0.7 ? "BLOCK" : prob > 0.4 ? "HOLD" : "ALLOW",
    model_used: "Offline Simulation (Heuristic Fallback)",
    reason_codes: prob > 0.5 ? ["HIGH_RETURN_VELOCITY", "SUSPICIOUS_REFUND_CLAIM"] : [],
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

export const DEMO_STREAM_TEMPLATES: StreamTransactionEvent[] = [
  {
    event_id: "EVT_SIM_01",
    timestamp: "12:00:01",
    order_id: "ORD_SIM_78930",
    buyer_id: "BUYER_RING_MEMBER_04",
    seller_id: "SELLER_RING_LEADER_01",
    amount: 890.0,
    risk_score: 0.942,
    calibrated_risk: 0.938,
    policy_risk_score: 0.942,
    risk_level: "high",
    decision: "BLOCK",
    trust_score: 6.2,
    reason_codes: ["HIGH_DEVICE_COLLISION", "RING_COLLUSION_SUSPECT"],
    source: "offline_simulation",
  },
  {
    event_id: "EVT_SIM_02",
    timestamp: "12:00:03",
    order_id: "ORD_SIM_78931",
    buyer_id: "BUYER_ORGANIC_404",
    seller_id: "SELLER_MERCHANT_10",
    amount: 45.0,
    risk_score: 0.035,
    calibrated_risk: 0.031,
    policy_risk_score: 0.035,
    risk_level: "low",
    decision: "ALLOW",
    trust_score: 96.9,
    reason_codes: ["CLEAN_BASELINE"],
    source: "offline_simulation",
  },
  {
    event_id: "EVT_SIM_03",
    timestamp: "12:00:05",
    order_id: "ORD_SIM_78932",
    buyer_id: "BUYER_REFUND_ABUSER",
    seller_id: "SELLER_ELECTRONICS_09",
    amount: 320.0,
    risk_score: 0.785,
    calibrated_risk: 0.76,
    policy_risk_score: 0.785,
    risk_level: "high",
    decision: "HOLD",
    trust_score: 24.0,
    reason_codes: ["HIGH_RETURN_VELOCITY", "SUSPICIOUS_REFUND_RATIO"],
    source: "offline_simulation",
  },
  {
    event_id: "EVT_SIM_04",
    timestamp: "12:00:07",
    order_id: "ORD_SIM_78933",
    buyer_id: "BUYER_NEWBIE_99",
    seller_id: "SELLER_UNKNOWN_44",
    amount: 450.0,
    risk_score: 0.38,
    calibrated_risk: 0.375,
    policy_risk_score: 0.38,
    risk_level: "medium",
    decision: "REVIEW",
    trust_score: 62.5,
    reason_codes: ["PRICE_OUTLIER_99TH_PCT"],
    source: "offline_simulation",
  },
];
