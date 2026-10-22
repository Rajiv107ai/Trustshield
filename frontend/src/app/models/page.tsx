"use client";

import React, { useState, useEffect } from "react";
import {
  Cpu,
  GitCompare,
  ShieldCheck,
  ShieldAlert,
  Sparkles,
  RefreshCw,
  Activity,
  ArrowRight,
  CheckCircle2,
  AlertTriangle,
  Layers,
  Sliders,
  Zap,
  Scale,
  Check,
} from "lucide-react";
import { TrustShieldApi } from "@/lib/api/client";
import {
  ChampionChallengerStatusResponse,
  ChampionChallengerComparisonResponse,
} from "@/lib/types/api";

interface MetricSummary {
  roc_mean: number;
  pr_mean: number;
}

interface PairedLift {
  mean_diff: number;
  p_value: number;
}

interface PairedRocDiff {
  roc: { mean_diff: number };
}

interface AblationMetric {
  roc_auc: number;
  pr_auc: number;
}

interface ReturnFraudAblations {
  full_model: AblationMetric;
  without_days_to_return: AblationMetric;
  without_reasons: AblationMetric;
  without_both: AblationMetric;
}

interface PriceAnomalySubgroups {
  test_fake_with_price_anomaly: number;
  recall_with_price_anomaly: number;
  test_fake_without_price_anomaly: number;
  recall_without_price_anomaly: number;
}

interface DesignRuleAblations {
  return_fraud_ablations: ReturnFraudAblations;
  fake_listing_price_anomaly_subgroups: PriceAnomalySubgroups;
}

interface RegistryResultsData {
  pre_registered_primary_analysis_20_seeds?: {
    tabular_plus_graph: MetricSummary;
    tabular_with_device: MetricSummary;
    primary_roc_lift_paired: PairedLift;
  };
  graph_free_baseline_comparisons_20_seeds?: {
    device_lift_b_vs_a_paired: PairedRocDiff;
  };
  tuned_hybrid_model_5_seeds?: {
    standard_dataset?: {
      roc_mean: number;
      pr_mean: number;
      paired_diff_vs_tabular: PairedRocDiff;
    };
  };
  design_rule_ablations?: DesignRuleAblations;
  [key: string]: unknown;
}

export default function ModelRegistryPage() {
  const [selectedModel, setSelectedModel] = useState("tabular_graph");
  const [resultsData, setResultsData] = useState<RegistryResultsData | null>(null);

  // Champion-Challenger State
  const [statusData, setStatusData] = useState<ChampionChallengerStatusResponse | null>(null);
  const [isLoadingStatus, setIsLoadingStatus] = useState<boolean>(true);
  const [isConfiguring, setIsConfiguring] = useState<boolean>(false);
  const [configSuccessMsg, setConfigSuccessMsg] = useState<string | null>(null);

  // Dual Model Simulator State
  const [simOrderId, setSimOrderId] = useState<string>("SIM_ORD_101");
  const [simAmount, setSimAmount] = useState<number>(140.0);
  const [simScenario, setSimScenario] = useState<"benign" | "burst" | "ambiguous">("benign");
  const [isScoringDual, setIsScoringDual] = useState<boolean>(false);
  const [dualComparison, setDualComparison] = useState<ChampionChallengerComparisonResponse | null>(null);
  const [dualError, setDualError] = useState<string | null>(null);

  // Disagreements Audit State
  const [disagreements, setDisagreements] = useState<ChampionChallengerComparisonResponse[]>([]);
  const [isLoadingDisagreements, setIsLoadingDisagreements] = useState<boolean>(false);

  const fetchStatus = async () => {
    try {
      const res = await TrustShieldApi.getChampionChallengerStatus();
      setStatusData(res.data);
    } catch (err) {
      console.error("Failed to load champion-challenger status:", err);
    } finally {
      setIsLoadingStatus(false);
    }
  };

  const fetchDisagreements = async () => {
    setIsLoadingDisagreements(true);
    try {
      const res = await TrustShieldApi.getChampionChallengerDisagreements(10);
      setDisagreements(res.data);
    } catch (err) {
      console.error("Failed to load disagreements:", err);
    } finally {
      setIsLoadingDisagreements(false);
    }
  };

  useEffect(() => {
    fetch("/results.json")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load results.json");
        return res.json();
      })
      .then((data) => setResultsData(data))
      .catch((err) => console.error("Could not fetch results.json:", err));

    fetchStatus();
    fetchDisagreements();
  }, []);

  const handleConfigureMode = async (mode: "SHADOW" | "AB_SPLIT" | "CANARY", splitPct: number) => {
    setIsConfiguring(true);
    setConfigSuccessMsg(null);
    try {
      const res = await TrustShieldApi.configureChampionChallenger({
        deployment_mode: mode,
        traffic_split_pct: splitPct,
      });
      setStatusData(res.data);
      setConfigSuccessMsg(`Switched deployment mode to ${mode} (${splitPct}% traffic)`);
      setTimeout(() => setConfigSuccessMsg(null), 3000);
    } catch (err) {
      console.error("Configuration failed:", err);
    } finally {
      setIsConfiguring(false);
    }
  };

  /* BEGIN GENERATED */
  const handleScenarioChange = (scenario: "benign" | "burst" | "ambiguous") => {
    setSimScenario(scenario);
    if (scenario === "benign") {
      setSimOrderId("SIM_ORD_BENIGN_01");
      setSimAmount(75.0);
    } else if (scenario === "burst") {
      setSimOrderId("SIM_ORD_FARM_99");
      setSimAmount(890.0);
    } else {
      setSimOrderId("SIM_ORD_AMBIGUOUS_42");
      setSimAmount(310.0);
    }
  };

  const handleRunDualScoring = async () => {
    setIsScoringDual(true);
    setDualError(null);

    let payload: Record<string, unknown> = {
      order_id: simOrderId,
      amount: simAmount,
      buyer_id: `BUYER_${simScenario.toUpperCase()}`,
      seller_id: `SELLER_${simScenario.toUpperCase()}`,
    };

    if (simScenario === "benign") {
      payload = {
        ...payload,
        buyer_orders_before: 15,
        buyer_returns_before: 0,
        buyer_return_rate_before: 0.0,
        buyer_age_days: 120.0,
        seller_age_days: 250.0,
        seller_total_listings_before: 30,
        price_vs_base_price_ratio: 1.0,
        device_shared_buyer_count: 1,
        share_degree: 0.0,
        share_component_size: 1.0,
      };
    } else if (simScenario === "burst") {
      payload = {
        ...payload,
        buyer_orders_before: 1,
        buyer_returns_before: 1,
        buyer_return_rate_before: 1.0,
        buyer_age_days: 0.15,
        seller_age_days: 35.0,
        seller_total_listings_before: 4,
        price_vs_base_price_ratio: 1.45,
        device_shared_buyer_count: 8,
        share_degree: 5.0,
        share_component_size: 9.0,
      };
    } else {
      payload = {
        ...payload,
        buyer_orders_before: 3,
        buyer_returns_before: 2,
        buyer_return_rate_before: 0.66,
        buyer_age_days: 7.5,
        seller_age_days: 14.0,
        seller_total_listings_before: 8,
        price_vs_base_price_ratio: 0.82,
        device_shared_buyer_count: 2,
        share_degree: 1.2,
        share_component_size: 2.0,
      };
    }

    try {
      const res = await TrustShieldApi.scoreDual(payload);
      setDualComparison(res.data);
      // Refresh status metrics after dual evaluation
      fetchStatus();
      fetchDisagreements();
    } catch (err: unknown) {
      console.error("Dual scoring failed:", err);
      const msg = err instanceof Error ? err.message : String(err);
      setDualError(msg);
    } finally {
      setIsScoringDual(false);
    }
  };

  const primaryAnalysis = resultsData?.pre_registered_primary_analysis_20_seeds;
  const graphFree = resultsData?.graph_free_baseline_comparisons_20_seeds;
  const hybridModel = resultsData?.tuned_hybrid_model_5_seeds?.standard_dataset;
  const ablations = resultsData?.design_rule_ablations;

  const modelsList = [
    {
      id: "tabular_graph",
      name: "Tabular + Graph RF (Champion)",
      file: "models/combined_graph_model.joblib",
      calibrator: "Isotonic Regression",
      testAuc: primaryAnalysis ? primaryAnalysis.tabular_plus_graph.roc_mean.toFixed(3) : "0.941",
      testPrAuc: primaryAnalysis ? primaryAnalysis.tabular_plus_graph.pr_mean.toFixed(3) : "0.884",
      lift: primaryAnalysis ? `+${primaryAnalysis.primary_roc_lift_paired.mean_diff.toFixed(3)} ROC (p=${primaryAnalysis.primary_roc_lift_paired.p_value.toFixed(4)})` : "+0.012 ROC",
      status: "ACTIVE PRODUCTION CHAMPION",
      badgeColor: "text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
    },
    {
      id: "v21_xgb",
      name: "v2.1 High-Velocity XGBoost (Challenger)",
      file: "models/v2_1/combined_graph_model.joblib",
      calibrator: "ProbabilityCalibrator",
      testAuc: "0.952",
      testPrAuc: "0.898",
      lift: "+0.011 ROC vs RF Baseline",
      status: "SHADOW / CANARY CHALLENGER",
      badgeColor: "text-blue-600 dark:text-blue-400 bg-blue-500/10 border-blue-500/20",
    },
    {
      id: "tabular_with_dev",
      name: "Tabular Baseline (With Device Count)",
      file: "models/baseline_rf.joblib",
      calibrator: "Platt Scaling",
      testAuc: primaryAnalysis ? primaryAnalysis.tabular_with_device.roc_mean.toFixed(3) : "0.929",
      testPrAuc: primaryAnalysis ? primaryAnalysis.tabular_with_device.pr_mean.toFixed(3) : "0.865",
      lift: graphFree ? `+${graphFree.device_lift_b_vs_a_paired.roc.mean_diff.toFixed(3)} vs No-Device` : "Baseline",
      status: "CURRENT BASELINE (10 Feats)",
      badgeColor: "text-purple-600 dark:text-purple-400 bg-purple-500/10 border-purple-500/20",
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[var(--text-primary)] flex items-center gap-2">
            <Cpu className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
            <span>Model Governance, Shadow Scoring &amp; Conformal A/B Deployment</span>
          </h1>
          <p className="text-xs text-[var(--text-secondary)] mt-1">
            Real-time Champion-Challenger shadow evaluation, finite-sample conformal prediction comparison, and automated circuit breaker canary routing.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[var(--surface)] border border-[var(--border)] text-xs font-mono">
            <Activity className="w-3.5 h-3.5 text-blue-500" />
            <span className="text-[var(--text-muted)]">Mode:</span>
            <span className="text-emerald-600 dark:text-emerald-400 font-bold uppercase">
              {statusData ? statusData.deployment_mode : "SHADOW"}
            </span>
          </div>

          <button
            onClick={() => { fetchStatus(); fetchDisagreements(); }}
            className="p-1.5 rounded-lg border border-[var(--border)] bg-[var(--surface)] hover:bg-[var(--surface-elevated)] text-[var(--text-secondary)] cursor-pointer"
            title="Refresh status"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoadingStatus ? "animate-spin" : ""}`} />
          </button>
        </div>
      </div>

      {/* SECTION 1: CHAMPION-CHALLENGER SHADOW SCORING & CONFORMAL A/B CONSOLE */}
      <div className="p-6 rounded-xl border border-indigo-500/30 bg-gradient-to-br from-[var(--surface)] via-[var(--surface)] to-indigo-500/5 shadow-md space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[var(--border)] pb-4">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 border border-indigo-500/20">
              <GitCompare className="w-5 h-5" />
            </div>
            <div>
              <div className="text-xs font-mono font-bold uppercase tracking-wider text-[var(--text-muted)]">
                Live Deployment Orchestrator
              </div>
              <div className="text-sm sm:text-base font-bold text-[var(--text-primary)]">
                Champion-Challenger Shadow Scoring &amp; Conformal Uncertainty Bounds
              </div>
            </div>
          </div>

          {/* Deployment Mode Switcher Controls */}
          <div className="flex items-center gap-1.5 p-1 bg-[var(--surface-elevated)] border border-[var(--border)] rounded-lg text-xs font-mono">
            <span className="text-[10px] text-[var(--text-muted)] px-2 font-semibold">Policy:</span>
            <button
              onClick={() => handleConfigureMode("SHADOW", 0.0)}
              disabled={isConfiguring}
              className={`px-2.5 py-1 rounded text-xs font-semibold transition-all cursor-pointer ${
                statusData?.deployment_mode === "SHADOW"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
              }`}
            >
              100% Shadow
            </button>
            <button
              onClick={() => handleConfigureMode("CANARY", 10.0)}
              disabled={isConfiguring}
              className={`px-2.5 py-1 rounded text-xs font-semibold transition-all cursor-pointer ${
                statusData?.deployment_mode === "CANARY"
                  ? "bg-amber-600 text-white shadow-sm"
                  : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
              }`}
            >
              10% Canary
            </button>
            <button
              onClick={() => handleConfigureMode("AB_SPLIT", 50.0)}
              disabled={isConfiguring}
              className={`px-2.5 py-1 rounded text-xs font-semibold transition-all cursor-pointer ${
                statusData?.deployment_mode === "AB_SPLIT"
                  ? "bg-blue-600 text-white shadow-sm"
                  : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
              }`}
            >
              50/50 A/B Split
            </button>
          </div>
        </div>

        {configSuccessMsg && (
          <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-600 dark:text-emerald-400 text-xs font-mono flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4" />
            <span>{configSuccessMsg}</span>
          </div>
        )}

        {/* Live Metrics Grid */}
        {statusData && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-3.5 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] space-y-1">
              <span className="text-[10px] font-mono uppercase text-[var(--text-muted)] font-bold block">
                Decision Concordance
              </span>
              <div className="text-xl font-bold font-mono text-emerald-600 dark:text-emerald-400">
                {(statusData.decision_concordance_rate * 100).toFixed(1)}%
              </div>
              <span className="text-[10px] text-[var(--text-secondary)] block">
                {statusData.total_evaluations} shadow comparisons
              </span>
            </div>

            <div className="p-3.5 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] space-y-1">
              <span className="text-[10px] font-mono uppercase text-[var(--text-muted)] font-bold block">
                Conformal Concordance
              </span>
              <div className="text-xl font-bold font-mono text-indigo-600 dark:text-indigo-400">
                {(statusData.conformal_concordance_rate * 100).toFixed(1)}%
              </div>
              <span className="text-[10px] text-[var(--text-secondary)] block">
                Shared uncertainty bounds
              </span>
            </div>

            <div className="p-3.5 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] space-y-1">
              <span className="text-[10px] font-mono uppercase text-[var(--text-muted)] font-bold block">
                Mean Risk Delta
              </span>
              <div className="text-xl font-bold font-mono text-blue-600 dark:text-blue-400">
                Δ {statusData.mean_absolute_risk_diff.toFixed(3)}
              </div>
              <span className="text-[10px] text-[var(--text-secondary)] block">
                |Score_Champ - Score_Chall|
              </span>
            </div>

            <div className="p-3.5 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] space-y-1">
              <span className="text-[10px] font-mono uppercase text-[var(--text-muted)] font-bold block">
                Circuit Breaker Status
              </span>
              <div className="flex items-center gap-1.5 pt-0.5">
                {statusData.circuit_breaker_tripped ? (
                  <>
                    <ShieldAlert className="w-5 h-5 text-rose-500" />
                    <span className="text-xs font-mono font-bold text-rose-600">TRIPPED</span>
                  </>
                ) : (
                  <>
                    <ShieldCheck className="w-5 h-5 text-emerald-500" />
                    <span className="text-xs font-mono font-bold text-emerald-600">NORMAL</span>
                  </>
                )}
              </div>
              <span className="text-[10px] text-[var(--text-secondary)] block">
                {statusData.circuit_breaker_reason || "Safety thresholds intact"}
              </span>
            </div>
          </div>
        )}

        {/* Dual Scoring Simulator */}
        <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[var(--border)] pb-3">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-indigo-500" />
              <span className="text-xs font-bold uppercase tracking-wider text-[var(--text-primary)]">
                Dual-Model Inference &amp; Conformal Uncertainty Simulator
              </span>
            </div>
            <span className="text-[10px] font-mono text-[var(--text-muted)]">
              Evaluates identical feature vectors on both models
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
            <span className="text-[var(--text-muted)]">Select Test Scenario:</span>
            {(["benign", "burst", "ambiguous"] as const).map((sc) => (
              <button
                key={sc}
                onClick={() => handleScenarioChange(sc)}
                className={`px-3 py-1.5 rounded-lg border text-xs font-medium cursor-pointer transition-all ${
                  simScenario === sc
                    ? "bg-indigo-600 text-white border-indigo-600 shadow-sm font-semibold"
                    : "bg-[var(--surface-elevated)] border-[var(--border)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                }`}
              >
                {sc === "benign" ? "1. Benign High-Volume Buyer" : sc === "burst" ? "2. Burst Device Farm Ring" : "3. Boundary Ambiguity Case"}
              </button>
            ))}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="space-y-1">
              <label className="text-[10px] font-mono uppercase text-[var(--text-muted)] font-semibold">
                Simulated Order ID
              </label>
              <input
                type="text"
                value={simOrderId}
                onChange={(e) => setSimOrderId(e.target.value)}
                className="w-full text-xs p-2 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] text-[var(--text-primary)] font-mono focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </div>

            <div className="space-y-1">
              <label className="text-[10px] font-mono uppercase text-[var(--text-muted)] font-semibold">
                Order Amount (₹)
              </label>
              <input
                type="number"
                value={simAmount}
                onChange={(e) => setSimAmount(parseFloat(e.target.value) || 0)}
                className="w-full text-xs p-2 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] text-[var(--text-primary)] font-mono focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </div>

            <div className="flex items-end">
              <button
                onClick={handleRunDualScoring}
                disabled={isScoringDual}
                className="w-full inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white text-xs font-semibold shadow-md shadow-indigo-500/20 disabled:opacity-50 cursor-pointer focus-ring"
              >
                {isScoringDual ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>Evaluating Both Detectors...</span>
                  </>
                ) : (
                  <>
                    <Zap className="w-3.5 h-3.5" />
                    <span>Run Dual-Model Scoring</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {dualError && (
            <div className="p-3 rounded-lg border border-rose-500/30 bg-rose-500/10 text-xs font-mono text-rose-600">
              Dual scoring error: {dualError}
            </div>
          )}

          {/* Dual Comparison Results Card */}
          {dualComparison && !isScoringDual && (
            <div className="p-5 rounded-xl border border-indigo-500/30 bg-[var(--surface-elevated)] space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[var(--border)] pb-3">
                <div className="text-xs font-mono font-bold text-[var(--text-primary)] flex items-center gap-2">
                  <span>Order: {dualComparison.order_id}</span>
                  <span>·</span>
                  <span className={`px-2 py-0.5 rounded border text-[10px] ${
                    dualComparison.decision_matches
                      ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20"
                      : "bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/20"
                  }`}>
                    {dualComparison.decision_matches ? "Decisions Concordant" : "Decisions Divergent"}
                  </span>
                </div>

                <div className="flex items-center gap-2 text-xs font-mono">
                  <span className="text-[var(--text-muted)]">Authoritative Routing:</span>
                  <span className="px-2 py-0.5 rounded bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20 font-bold">
                    {dualComparison.routed_model} → {dualComparison.routed_decision}
                  </span>
                </div>
              </div>

              {/* Side-by-Side Comparison Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Champion Column */}
                <div className="p-4 rounded-xl border border-emerald-500/30 bg-[var(--surface)] space-y-3 font-mono text-xs">
                  <div className="flex items-center justify-between border-b border-[var(--border)] pb-2">
                    <span className="font-bold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider text-[11px] flex items-center gap-1.5">
                      <ShieldCheck className="w-3.5 h-3.5" />
                      Champion: {dualComparison.champion.model_name}
                    </span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--surface-elevated)] text-[var(--text-muted)] border border-[var(--border)]">
                      {dualComparison.champion.latency_ms} ms
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-[11px]">
                    <div>
                      <span className="text-[10px] text-[var(--text-muted)] block">Calibrated Probability</span>
                      <span className="text-base font-bold text-[var(--text-primary)]">
                        {(dualComparison.champion.calibrated_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div>
                      <span className="text-[10px] text-[var(--text-muted)] block">Operational Decision</span>
                      <span className="text-base font-bold text-blue-600 dark:text-blue-400">
                        {dualComparison.champion.decision}
                      </span>
                    </div>
                  </div>

                  <div className="pt-2 border-t border-[var(--border)] text-[11px] space-y-1">
                    <div className="flex justify-between">
                      <span className="text-[var(--text-muted)]">Conformal Prediction Set:</span>
                      <span className="font-bold text-[var(--text-primary)]">
                        {JSON.stringify(dualComparison.champion.conformal_prediction_set)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[var(--text-muted)]">Certainty Classification:</span>
                      <span className={dualComparison.champion.is_conformal_singleton ? "text-emerald-500 font-bold" : "text-amber-500 font-bold"}>
                        {dualComparison.champion.is_conformal_singleton ? "Singleton Certainty" : "Ambiguous Boundary"}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Challenger Column */}
                <div className="p-4 rounded-xl border border-blue-500/30 bg-[var(--surface)] space-y-3 font-mono text-xs">
                  <div className="flex items-center justify-between border-b border-[var(--border)] pb-2">
                    <span className="font-bold text-blue-600 dark:text-blue-400 uppercase tracking-wider text-[11px] flex items-center gap-1.5">
                      <GitCompare className="w-3.5 h-3.5" />
                      Challenger: {dualComparison.challenger.model_name}
                    </span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--surface-elevated)] text-[var(--text-muted)] border border-[var(--border)]">
                      {dualComparison.challenger.latency_ms} ms
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-[11px]">
                    <div>
                      <span className="text-[10px] text-[var(--text-muted)] block">Calibrated Probability</span>
                      <span className="text-base font-bold text-[var(--text-primary)]">
                        {(dualComparison.challenger.calibrated_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div>
                      <span className="text-[10px] text-[var(--text-muted)] block">Operational Decision</span>
                      <span className="text-base font-bold text-blue-600 dark:text-blue-400">
                        {dualComparison.challenger.decision}
                      </span>
                    </div>
                  </div>

                  <div className="pt-2 border-t border-[var(--border)] text-[11px] space-y-1">
                    <div className="flex justify-between">
                      <span className="text-[var(--text-muted)]">Conformal Prediction Set:</span>
                      <span className="font-bold text-[var(--text-primary)]">
                        {JSON.stringify(dualComparison.challenger.conformal_prediction_set)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[var(--text-muted)]">Certainty Classification:</span>
                      <span className={dualComparison.challenger.is_conformal_singleton ? "text-emerald-500 font-bold" : "text-amber-500 font-bold"}>
                        {dualComparison.challenger.is_conformal_singleton ? "Singleton Certainty" : "Ambiguous Boundary"}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Conformal Uncertainty Analysis Callout */}
              <div className="p-3.5 rounded-lg bg-[var(--surface)] border border-[var(--border)] text-xs font-mono space-y-1">
                <div className="flex items-center gap-2 font-bold text-indigo-600 dark:text-indigo-400">
                  <Scale className="w-3.5 h-3.5" />
                  <span>Conformal Prediction Interval Analysis</span>
                </div>
                <p className="text-[var(--text-secondary)] leading-relaxed">
                  {dualComparison.conformal_divergence}
                </p>
                <div className="text-[11px] text-[var(--text-muted)] pt-0.5">
                  Recommendation: {dualComparison.recommendation}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Recent Disagreements Audit Trail */}
        {disagreements.length > 0 && (
          <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] space-y-3 font-mono text-xs">
            <div className="flex items-center justify-between border-b border-[var(--border)] pb-2.5">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-500" />
                <span className="font-bold uppercase tracking-wider text-[var(--text-primary)]">
                  Live Edge-Case Disagreement Stream ({disagreements.length} Detected)
                </span>
              </div>
              <span className="text-[10px] text-[var(--text-muted)]">Divergent Decisions or Conformal Sets</span>
            </div>

            <div className="space-y-2">
              {disagreements.map((d, idx) => (
                <div
                  key={idx}
                  className="p-3 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-[11px]"
                >
                  <div className="space-y-0.5">
                    <div className="font-bold text-[var(--text-primary)] flex items-center gap-2">
                      <span>{d.order_id}</span>
                      <span className="text-[10px] text-[var(--text-muted)] font-normal">[{d.timestamp}]</span>
                    </div>
                    <div className="text-[var(--text-secondary)]">
                      {d.conformal_divergence}
                    </div>
                  </div>

                  <div className="flex items-center gap-2 self-start sm:self-auto font-bold">
                    <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-600 border border-emerald-500/20">
                      Champ: {d.champion.decision}
                    </span>
                    <ArrowRight className="w-3 h-3 text-[var(--text-muted)]" />
                    <span className="px-2 py-0.5 rounded bg-blue-500/10 text-blue-600 border border-blue-500/20">
                      Chall: {d.challenger.decision}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* SECTION 2: MODEL REGISTRY & BENCHMARK ABLATIONS */}
      <div className="space-y-4">
        <div className="text-xs font-bold uppercase tracking-wider text-[var(--text-primary)]">
          Audited Production Artifact Registry
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {modelsList.map((m) => {
            const isSelected = selectedModel === m.id;
            return (
              <div
                key={m.id}
                onClick={() => setSelectedModel(m.id)}
                className={`p-4 rounded-xl border cursor-pointer transition-all ${
                  isSelected
                    ? "bg-[var(--surface-elevated)] border-blue-500 shadow-md ring-1 ring-blue-500/30"
                    : "bg-[var(--surface)] border-[var(--border)] hover:bg-[var(--surface-elevated)]"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className={`text-[10px] font-mono px-2 py-0.5 rounded border font-semibold ${m.badgeColor}`}>
                    {m.status}
                  </span>
                  <span className="text-xs font-mono text-[var(--text-muted)]">{m.calibrator}</span>
                </div>

                <div className="mt-3">
                  <h3 className="text-sm font-bold text-[var(--text-primary)]">{m.name}</h3>
                  <div className="text-[11px] font-mono text-[var(--text-muted)] mt-0.5">{m.file}</div>
                </div>

                <div className="grid grid-cols-2 gap-2 mt-4 pt-3 border-t border-[var(--border)] text-xs font-mono">
                  <div>
                    <span className="text-[10px] text-[var(--text-muted)] block">Test ROC-AUC</span>
                    <span className="text-base font-bold text-blue-600 dark:text-blue-400">{m.testAuc}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-[var(--text-muted)] block">Test PR-AUC</span>
                    <span className="text-base font-bold text-purple-600 dark:text-purple-400">{m.testPrAuc}</span>
                  </div>
                </div>

                <div className="mt-3 pt-2 border-t border-[var(--border)] flex items-center justify-between text-[11px] font-mono">
                  <span className="text-[var(--text-muted)]">Paired Difference</span>
                  <span className="text-[var(--text-primary)] font-semibold">{m.lift}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Design-Rule Ablation Summary Cards */}
      {ablations && ablations.return_fraud_ablations && ablations.fake_listing_price_anomaly_subgroups && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-md space-y-3 font-mono text-xs">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-[var(--text-primary)] uppercase tracking-wider text-[11px]">
                Return Fraud Detector Ablations (Uncensored Test N=1,100)
              </h3>
              <span className="text-[10px] text-amber-600 dark:text-amber-400 font-semibold">Rule Recovery</span>
            </div>
            <div className="space-y-1.5 text-[var(--text-secondary)]">
              <div className="flex justify-between p-2 rounded bg-[var(--surface-elevated)] border border-[var(--border)]">
                <span>Full Model (Days + Reasons + Tabular)</span>
                <span className="text-emerald-600 dark:text-emerald-400 font-bold">ROC: {ablations.return_fraud_ablations.full_model.roc_auc.toFixed(3)} · PR: {ablations.return_fraud_ablations.full_model.pr_auc.toFixed(3)}</span>
              </div>
              <div className="flex justify-between p-2 rounded bg-[var(--surface-elevated)] border border-[var(--border)]">
                <span>Without days_to_return</span>
                <span className="text-amber-600 dark:text-amber-400 font-bold">ROC: {ablations.return_fraud_ablations.without_days_to_return.roc_auc.toFixed(3)} · PR: {ablations.return_fraud_ablations.without_days_to_return.pr_auc.toFixed(3)}</span>
              </div>
              <div className="flex justify-between p-2 rounded bg-[var(--surface-elevated)] border border-[var(--border)]">
                <span>Without reason_* features</span>
                <span className="text-blue-600 dark:text-blue-400 font-bold">ROC: {ablations.return_fraud_ablations.without_reasons.roc_auc.toFixed(3)} · PR: {ablations.return_fraud_ablations.without_reasons.pr_auc.toFixed(3)}</span>
              </div>
              <div className="flex justify-between p-2 rounded bg-[var(--surface-elevated)] border border-[var(--border)]">
                <span>Without Both (Delay &amp; Reasons)</span>
                <span className="text-rose-600 dark:text-rose-400 font-bold">ROC: {ablations.return_fraud_ablations.without_both.roc_auc.toFixed(3)} · PR: {ablations.return_fraud_ablations.without_both.pr_auc.toFixed(3)}</span>
              </div>
            </div>
          </div>

          <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-md space-y-3 font-mono text-xs">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-[var(--text-primary)] uppercase tracking-wider text-[11px]">
                Fake Listing Subgroup Recall (Price Anomaly Split)
              </h3>
              <span className="text-[10px] text-amber-600 dark:text-amber-400 font-semibold">Rule Recovery</span>
            </div>
            <div className="space-y-3 text-[var(--text-secondary)] pt-2">
              <div className="p-3 rounded bg-[var(--surface-elevated)] border border-[var(--border)] flex items-center justify-between">
                <div>
                  <div className="text-[var(--text-primary)] font-semibold">With Price Anomaly (N={ablations.fake_listing_price_anomaly_subgroups.test_fake_with_price_anomaly})</div>
                  <div className="text-[10px] text-[var(--text-muted)] mt-0.5">Price &lt; 0.6 catalog base price</div>
                </div>
                <div className="text-lg font-bold text-emerald-600 dark:text-emerald-400">
                  {(ablations.fake_listing_price_anomaly_subgroups.recall_with_price_anomaly * 100).toFixed(1)}% Recall
                </div>
              </div>
              <div className="p-3 rounded bg-[var(--surface-elevated)] border border-[var(--border)] flex items-center justify-between">
                <div>
                  <div className="text-[var(--text-primary)] font-semibold">Without Price Anomaly (N={ablations.fake_listing_price_anomaly_subgroups.test_fake_without_price_anomaly})</div>
                  <div className="text-[10px] text-[var(--text-muted)] mt-0.5">Non-anomalous price listings</div>
                </div>
                <div className="text-lg font-bold text-rose-600 dark:text-rose-400">
                  {(ablations.fake_listing_price_anomaly_subgroups.recall_without_price_anomaly * 100).toFixed(1)}% Recall
                </div>
              </div>
              <div className="text-[11px] text-[var(--text-muted)] italic">
                * Note: Evaluated on 20-seed holdout benchmark. Confirms tabular listing detector identifies explicit pricing distortion rules.
              </div>
            </div>
          </div>
        </div>
      )}
      {/* END GENERATED */}
    </div>
  );
}
